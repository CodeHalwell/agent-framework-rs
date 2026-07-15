//! Conversation-history context providers.
//!
//! Upstream moved conversation history out of the thread/session entirely:
//! it is now just another [`ContextProvider`] — a [`HistoryProvider`] —
//! that prepends its stored messages ahead of a run (`before_run`) and
//! records the run's request + response messages after a successful run
//! (`after_run`). [`InMemoryHistoryProvider`] is the in-process default;
//! [`FileHistoryProvider`] persists to a JSON file on disk.
//!
//! [`Agent`](crate::agent::Agent) and
//! [`WorkflowAgent`](crate::workflow::WorkflowAgent) auto-attach a fresh
//! [`InMemoryHistoryProvider`] (via [`ensure_history_provider`]) to any
//! non-service-managed [`AgentSession`] that doesn't already carry a history
//! provider, so local multi-turn conversations keep accumulating history the
//! way the old `AgentThread` message store used to.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::types::Message;

/// A [`ContextProvider`] that also manages conversation history.
///
/// This is a marker trait (over and above [`ContextProvider::is_history_provider`],
/// which drives runtime detection via trait objects): implementing it
/// documents that a type's `before_run`/`after_run` are the ones responsible
/// for a session's conversation history, distinguishing it from a generic
/// memory/RAG provider.
pub trait HistoryProvider: ContextProvider {}

/// Attach a fresh [`InMemoryHistoryProvider`] as the **first** context
/// provider on `session` when it is not service-managed and does not already
/// carry a history provider. A no-op for service-managed sessions (the
/// service owns history server-side) and for sessions that already have one
/// attached (detected via [`ContextProvider::is_history_provider`]).
pub fn ensure_history_provider(session: &mut AgentSession) {
    if session.service_session_id().is_none()
        && !session
            .context_providers
            .iter()
            .any(|p| p.is_history_provider())
    {
        session
            .context_providers
            .insert(0, Arc::new(InMemoryHistoryProvider::new()));
    }
}

/// In-memory [`HistoryProvider`]: keeps history in an `Arc<Mutex<Vec<Message>>>`,
/// shared across clones.
#[derive(Default, Clone)]
pub struct InMemoryHistoryProvider {
    messages: Arc<Mutex<Vec<Message>>>,
}

impl InMemoryHistoryProvider {
    /// An empty history provider.
    pub fn new() -> Self {
        Self::default()
    }

    /// A history provider seeded with `messages`.
    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages: Arc::new(Mutex::new(messages)),
        }
    }

    /// The stored messages, in chronological order.
    pub fn list_messages(&self) -> Vec<Message> {
        self.messages.lock().unwrap().clone()
    }

    /// Serialize the stored history to `{"messages": [...]}`.
    pub fn to_dict(&self) -> Value {
        serde_json::json!({ "messages": self.list_messages() })
    }

    /// Reconstruct a provider from state produced by [`InMemoryHistoryProvider::to_dict`].
    pub fn from_dict(state: &Value) -> Result<Self> {
        let messages = match state.get("messages") {
            Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| {
                Error::Serialization(format!("failed to restore history provider: {e}"))
            })?,
            _ => Vec::new(),
        };
        Ok(Self::with_messages(messages))
    }
}

#[async_trait]
impl ContextProvider for InMemoryHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.messages.lock().unwrap().clone();
        let existing = std::mem::take(&mut ctx.messages);
        ctx.messages = stored.into_iter().chain(existing).collect();
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_none() {
            let mut guard = self.messages.lock().unwrap();
            guard.extend(request_messages.iter().cloned());
            guard.extend(response_messages.iter().cloned());
        }
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for InMemoryHistoryProvider {}

/// A [`HistoryProvider`] that persists to a JSON file on disk, loading any
/// existing history from `path` on construction and rewriting the whole file
/// after every successful run.
///
/// Persistence is **atomic and concurrency-safe**: `after_run` serializes the
/// whole append→snapshot→write sequence behind an async `write_lock` (shared
/// across clones), writes to a temporary sibling file, and atomically renames
/// it into place. The in-memory history is only updated *after* the on-disk
/// write succeeds, so a failed write never diverges memory from disk, and two
/// concurrent runs sharing cloned providers can't lose each other's messages
/// via a snapshot/overwrite race.
#[derive(Clone)]
pub struct FileHistoryProvider {
    path: PathBuf,
    messages: Arc<Mutex<Vec<Message>>>,
    /// Serializes the append+snapshot+persist critical section across all
    /// clones so concurrent `after_run` calls can't interleave into a lost
    /// update. Held only in `after_run`; reads (`before_run`/`list_messages`)
    /// take the fast in-memory `messages` lock and never block on this.
    write_lock: Arc<tokio::sync::Mutex<()>>,
}

impl FileHistoryProvider {
    /// Open (or create) a file-backed history provider at `path`. A missing
    /// or empty file starts with no history; an existing file is parsed
    /// eagerly, so a malformed file fails the constructor rather than
    /// silently discarding history.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        let messages = if path.exists() {
            let data = std::fs::read_to_string(&path)
                .map_err(|e| Error::other(format!("failed to read history file {path:?}: {e}")))?;
            if data.trim().is_empty() {
                Vec::new()
            } else {
                let value: Value = serde_json::from_str(&data).map_err(|e| {
                    Error::Serialization(format!("failed to parse history file {path:?}: {e}"))
                })?;
                match value.get("messages") {
                    Some(v) if !v.is_null() => serde_json::from_value(v.clone()).map_err(|e| {
                        Error::Serialization(format!("failed to parse history file {path:?}: {e}"))
                    })?,
                    _ => Vec::new(),
                }
            }
        } else {
            Vec::new()
        };
        Ok(Self {
            path,
            messages: Arc::new(Mutex::new(messages)),
            write_lock: Arc::new(tokio::sync::Mutex::new(())),
        })
    }

    /// The path this provider persists to.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    /// The stored messages, in chronological order.
    pub fn list_messages(&self) -> Vec<Message> {
        self.messages.lock().unwrap().clone()
    }

    /// Serialize the stored history to `{"messages": [...]}`.
    pub fn to_dict(&self) -> Value {
        serde_json::json!({ "messages": self.list_messages() })
    }

    /// Atomically write `messages` as `{"messages": [...]}`: serialize, write to
    /// a uniquely named temporary sibling file, flush, then rename it over the
    /// destination. The rename is atomic on a POSIX filesystem, so a crash mid-
    /// write leaves either the old file or the new one, never a truncated file.
    async fn persist(&self, messages: &[Message]) -> Result<()> {
        let dict = serde_json::json!({ "messages": messages });
        let json = serde_json::to_string_pretty(&dict)
            .map_err(|e| Error::Serialization(format!("failed to serialize history: {e}")))?;
        // Temp file in the same directory so `rename` stays on one filesystem.
        // A uuid suffix keeps two providers on the same path from clobbering
        // each other's temp file.
        let file_name = self
            .path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("history.json");
        let tmp = self
            .path
            .with_file_name(format!("{file_name}.tmp.{}", uuid::Uuid::new_v4()));
        tokio::fs::write(&tmp, json)
            .await
            .map_err(|e| Error::other(format!("failed to write history temp file {tmp:?}: {e}")))?;
        tokio::fs::rename(&tmp, &self.path).await.map_err(|e| {
            // Best-effort cleanup of the temp file on a failed rename.
            let tmp = tmp.clone();
            tokio::spawn(async move {
                let _ = tokio::fs::remove_file(&tmp).await;
            });
            Error::other(format!(
                "failed to finalize history file {:?}: {e}",
                self.path
            ))
        })
    }
}

#[async_trait]
impl ContextProvider for FileHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.messages.lock().unwrap().clone();
        let existing = std::mem::take(&mut ctx.messages);
        ctx.messages = stored.into_iter().chain(existing).collect();
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        // Serialize the whole append→snapshot→persist sequence so two
        // concurrent runs (sharing cloned providers) can't interleave a
        // snapshot and an overwrite into a lost update.
        let _write = self.write_lock.lock().await;

        // Compute the next full history WITHOUT committing it to shared memory
        // yet: disk is the source of truth. We persist first and only update
        // the in-memory copy on success, so a failed write leaves memory and
        // disk consistent (the run's `after_run` returns the error and the
        // caller can retry) rather than diverging.
        let snapshot = {
            let guard = self.messages.lock().unwrap();
            let mut next = guard.clone();
            next.extend(request_messages.iter().cloned());
            next.extend(response_messages.iter().cloned());
            next
        };
        self.persist(&snapshot).await?;
        *self.messages.lock().unwrap() = snapshot;
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for FileHistoryProvider {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[tokio::test]
    async fn before_run_prepends_stored_messages_ahead_of_existing_context_messages() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q1"),
            Message::assistant("a1"),
        ]);
        let mut ctx = SessionContext::new(vec![Message::user("q2")]);
        ctx.messages
            .push(Message::system("injected by another provider"));
        provider.before_run(&mut ctx).await.unwrap();
        let texts: Vec<String> = ctx.messages.iter().map(|m| m.text()).collect();
        assert_eq!(
            texts,
            vec![
                "q1".to_string(),
                "a1".to_string(),
                "injected by another provider".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn after_run_appends_only_on_success() {
        let provider = InMemoryHistoryProvider::new();
        provider
            .after_run(&[Message::user("hi")], &[Message::assistant("hello")], None)
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);

        // A failed run must not record anything.
        provider
            .after_run(
                &[Message::user("again")],
                &[],
                Some(&Error::service("boom")),
            )
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);
    }

    #[test]
    fn to_dict_from_dict_round_trips_messages() {
        let provider = InMemoryHistoryProvider::with_messages(vec![
            Message::user("q1"),
            Message::assistant("a1"),
        ]);
        let state = provider.to_dict();
        let restored = InMemoryHistoryProvider::from_dict(&state).unwrap();
        let msgs = restored.list_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text(), "q1");
        assert_eq!(msgs[1].text(), "a1");
    }

    #[test]
    fn from_dict_tolerates_a_missing_messages_key() {
        let restored = InMemoryHistoryProvider::from_dict(&serde_json::json!({})).unwrap();
        assert!(restored.list_messages().is_empty());
    }

    #[test]
    fn ensure_history_provider_attaches_once_and_skips_service_managed() {
        let mut local = AgentSession::new();
        ensure_history_provider(&mut local);
        assert_eq!(local.context_providers.len(), 1);
        assert!(local.context_providers[0].is_history_provider());
        // A second call must not attach a duplicate.
        ensure_history_provider(&mut local);
        assert_eq!(local.context_providers.len(), 1);

        let mut service = AgentSession::service("svc-1");
        ensure_history_provider(&mut service);
        assert!(service.context_providers.is_empty());
    }

    #[tokio::test]
    async fn file_history_provider_persists_and_reloads() {
        let dir = std::env::temp_dir().join(format!("afr-history-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        let provider = FileHistoryProvider::new(&path).unwrap();
        assert!(provider.list_messages().is_empty());
        provider
            .after_run(&[Message::user("hi")], &[Message::assistant("hello")], None)
            .await
            .unwrap();
        assert_eq!(provider.list_messages().len(), 2);

        // A fresh provider opened on the same path picks up the persisted
        // history.
        let reloaded = FileHistoryProvider::new(&path).unwrap();
        let msgs = reloaded.list_messages();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text(), "hi");
        assert_eq!(msgs[1].text(), "hello");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn file_history_provider_concurrent_runs_do_not_lose_messages() {
        // Regression for the snapshot/overwrite race: many concurrent
        // `after_run` calls on cloned providers must all be durably recorded,
        // and the on-disk file must always be valid JSON (atomic rename).
        let dir = std::env::temp_dir().join(format!("afr-history-conc-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("history.json");

        let provider = FileHistoryProvider::new(&path).unwrap();
        const N: usize = 50;
        let mut handles = Vec::new();
        for i in 0..N {
            let p = provider.clone();
            handles.push(tokio::spawn(async move {
                p.after_run(
                    &[Message::user(format!("q{i}"))],
                    &[Message::assistant(format!("a{i}"))],
                    None,
                )
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }

        // Every run contributed a request + response message; none lost.
        assert_eq!(provider.list_messages().len(), N * 2);

        // The on-disk file is valid and holds the full history (atomic rename
        // means it is never a torn/partial write).
        let reloaded = FileHistoryProvider::new(&path).unwrap();
        assert_eq!(reloaded.list_messages().len(), N * 2);

        // No temp files left behind.
        let leftover: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
            .collect();
        assert!(leftover.is_empty(), "temp files leaked: {leftover:?}");

        std::fs::remove_dir_all(&dir).ok();
    }
}
