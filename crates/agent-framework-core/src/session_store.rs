//! Storing agent sessions between runs.
//!
//! Port of .NET's `AgentSessionStore` (`Microsoft.Agents.AI.Abstractions`,
//! with `DeleteSessionAsync` from upstream #8801) and Python's
//! `SessionStore` / `FileSessionStore` (`_sessions.py`). Both upstreams mark
//! the surface experimental; here it is behind `experimental-session-store`.
//!
//! An [`AgentSessionStore`] maps an [`AgentSessionStoreKey`] (a session id
//! plus optional named partitions, such as a tenant or user) to a snapshot of
//! an [`AgentSession`], scoped to the agent that owns it. A hosted endpoint
//! loads the session at the start of a request, runs the agent and saves the
//! session at the end, so a conversation continues across requests,
//! restarts or instances. Two stores are included:
//! [`InMemoryAgentSessionStore`] and [`FileAgentSessionStore`].
//!
//! # What a snapshot holds
//!
//! [`serialize_session`] records the session's id, its service conversation
//! id, its state bag, and the messages of any history provider that keeps
//! them only in process memory ([`InMemoryHistoryProvider`], the one an
//! [`Agent`](crate::agent::Agent) attaches by default). Upstream keeps local
//! chat history inside the session's state, so its snapshots carry it for
//! free; this port keeps it in the provider, so the snapshot asks for it
//! through [`ContextProvider::snapshot_history`].
//!
//! [`deserialize_session`] starts from the agent's
//! [`create_session`](SupportsAgentRun::create_session) and restores the
//! snapshot into it, so whatever the agent attaches to a new session is
//! attached to a restored one too. Other context providers on the original
//! session are not restored, as with [`AgentSession::from_dict`]: one that
//! persists itself (a file or database history) is reattached by the
//! application.
//!
//! # Divergences
//!
//! - **Agent identity.** .NET scopes stored sessions by the agent's id, or by
//!   a host-supplied storage identity. Here they are scoped by
//!   [`SupportsAgentRun::id`]; give an agent a stable id to share sessions
//!   across instances (the default id is generated per instance).
//! - **No service lookup.** .NET's `GetService` on the store has no
//!   counterpart; a store is an ordinary trait object.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::history::InMemoryHistoryProvider;
use crate::memory::ContextProvider;
use crate::session::AgentSession;
use crate::types::Message;

/// Identifies a stored session: a session id plus optional named
/// partitions.
///
/// Partitions scope a session id, so the same id under two tenants names two
/// sessions. Keys compare by value; partition order does not matter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AgentSessionStoreKey {
    session_id: String,
    partitions: BTreeMap<String, String>,
}

impl AgentSessionStoreKey {
    /// A key for `session_id` with no partitions. An empty or
    /// whitespace-only id is refused.
    pub fn new(session_id: impl Into<String>) -> Result<Self> {
        let session_id = session_id.into();
        if session_id.trim().is_empty() {
            return Err(Error::Configuration(
                "session store key session_id must be a non-empty string".into(),
            ));
        }
        Ok(Self {
            session_id,
            partitions: BTreeMap::new(),
        })
    }

    /// This key with partition `name` set to `value` (replacing any value it
    /// had). An empty or whitespace-only name or value is refused.
    pub fn with_partition(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self> {
        let (name, value) = (name.into(), value.into());
        if name.trim().is_empty() || value.trim().is_empty() {
            return Err(Error::Configuration(
                "session store key partitions must have non-empty names and values".into(),
            ));
        }
        self.partitions.insert(name, value);
        Ok(self)
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// The partitions, ordered by name.
    pub fn partitions(&self) -> &BTreeMap<String, String> {
        &self.partitions
    }
}

/// Stores and retrieves agent sessions. See the [module docs](self).
///
/// Every successful [`get_session`](Self::get_session) must return an
/// independent session: a caller may change it, or run concurrent branches
/// from the same key, without affecting the stored snapshot or each other.
#[async_trait]
pub trait AgentSessionStore: Send + Sync {
    /// Save `session` under `key` for `agent`, replacing any snapshot stored
    /// there.
    async fn save_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
        session: &AgentSession,
    ) -> Result<()>;

    /// The session stored under `key` for `agent`, or `None` when nothing is
    /// stored there. Never creates a session.
    async fn get_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<Option<AgentSession>>;

    /// The session stored under `key`, or a new one from the agent when
    /// nothing is stored there.
    async fn get_or_create_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<AgentSession> {
        Ok(match self.get_session(agent, key).await? {
            Some(session) => session,
            None => agent.create_session(),
        })
    }

    /// Delete the session stored under `key` for `agent`. Deleting a session
    /// that is not stored is not an error (upstream #8801).
    async fn delete_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<()>;
}

/// Snapshot `session` as JSON for an [`AgentSessionStore`]: its
/// [`to_dict`](AgentSession::to_dict) fields plus, under `"history"`, the
/// messages of the first context provider that keeps them only in memory.
pub fn serialize_session(session: &AgentSession) -> Value {
    let mut snapshot = session.to_dict();
    if let Some(messages) = session
        .context_providers
        .iter()
        .find_map(|p| p.snapshot_history())
    {
        snapshot["history"] = serde_json::to_value(messages).unwrap_or(Value::Null);
    }
    snapshot
}

/// Restore a [`serialize_session`] snapshot as a session of `agent`.
///
/// Starts from `agent.create_session()`, then restores the session id, the
/// service conversation id and the state bag. Recorded history replaces the
/// fresh session's in-memory history provider (or is attached when the agent
/// attached none). A snapshot with no recorded history (a service-managed
/// session, whose history the service keeps) drops the fresh in-memory
/// history provider instead, so the restored session does not start a
/// second, local copy of a conversation the service already holds.
pub fn deserialize_session(agent: &dyn SupportsAgentRun, snapshot: &Value) -> Result<AgentSession> {
    let restored = AgentSession::from_dict(snapshot)?;
    let mut session = agent.create_session();
    session.restore_from(&restored);
    let Some(history) = snapshot.get("history").filter(|v| !v.is_null()) else {
        session
            .context_providers
            .retain(|p| p.snapshot_history().is_none());
        return Ok(session);
    };
    {
        let messages: Vec<Message> = serde_json::from_value(history.clone())
            .map_err(|e| Error::Serialization(format!("failed to restore session history: {e}")))?;
        let provider: Arc<dyn ContextProvider> =
            Arc::new(InMemoryHistoryProvider::with_messages(messages));
        match session
            .context_providers
            .iter()
            .position(|p| p.snapshot_history().is_some())
        {
            Some(index) => session.context_providers[index] = provider,
            None => session.context_providers.insert(0, provider),
        }
    }
    Ok(session)
}

/// A process-local [`AgentSessionStore`] holding JSON snapshots in memory.
/// Cheap to clone; clones share storage. Nothing is evicted.
#[derive(Debug, Clone, Default)]
pub struct InMemoryAgentSessionStore {
    sessions: Arc<Mutex<HashMap<(String, AgentSessionStoreKey), Value>>>,
}

impl InMemoryAgentSessionStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many sessions are stored, across all agents.
    pub fn len(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl AgentSessionStore for InMemoryAgentSessionStore {
    async fn save_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
        session: &AgentSession,
    ) -> Result<()> {
        let snapshot = serialize_session(session);
        self.sessions
            .lock()
            .unwrap()
            .insert((agent.id().to_string(), key.clone()), snapshot);
        Ok(())
    }

    async fn get_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<Option<AgentSession>> {
        let snapshot = self
            .sessions
            .lock()
            .unwrap()
            .get(&(agent.id().to_string(), key.clone()))
            .cloned();
        snapshot.map(|s| deserialize_session(agent, &s)).transpose()
    }

    async fn delete_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<()> {
        self.sessions
            .lock()
            .unwrap()
            .remove(&(agent.id().to_string(), key.clone()));
        Ok(())
    }
}

/// An [`AgentSessionStore`] keeping one JSON file per session under a root
/// directory: `<root>/<agent id>/<partitions>/<session id>.json`.
///
/// Each path component is encoded so that any id is a safe, portable file
/// name and distinct keys never share a file: lowercase ASCII letters,
/// digits, `-` and `_` are kept and every other byte becomes `~XX`.
/// Uppercase letters are encoded too, so `Foo` and `foo` stay distinct on
/// a case-insensitive file system, and the first letter of a Windows
/// device name (`con`, `nul`, `lpt1`, …) is encoded so such ids stay
/// writable on Windows. A component longer than
/// 255 bytes once encoded is refused rather than shortened, since a
/// shortened name could collide.
///
/// Writes go to a uniquely named temporary file that is then renamed over
/// the destination, so a reader never sees a partly written snapshot.
/// Concurrent writers are last-writer-wins. Snapshots are plaintext: treat
/// the root as trusted application storage, not a secret store.
#[derive(Debug, Clone)]
pub struct FileAgentSessionStore {
    root: PathBuf,
}

impl FileAgentSessionStore {
    /// A store rooted at `root`. Directories are created on first save.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// The root directory.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }

    fn session_path(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<PathBuf> {
        let partitions: Vec<String> = key
            .partitions()
            .iter()
            .map(|(name, value)| format!("{}={}", encode_component(name), encode_component(value)))
            .collect();
        let components = [
            encode_component(agent.id()),
            format!("p-{}", partitions.join("&")),
            format!("{}.json", encode_component(key.session_id())),
        ];
        let mut path = self.root.clone();
        for component in components {
            if component.len() > 255 {
                return Err(Error::Configuration(format!(
                    "session store path component is too long ({} bytes encoded, at most 255)",
                    component.len()
                )));
            }
            path.push(component);
        }
        Ok(path)
    }
}

/// Encode `raw` as a file-name-safe, injective component (see
/// [`FileAgentSessionStore`]).
fn encode_component(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for byte in raw.bytes() {
        if byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-' || byte == b'_' {
            out.push(byte as char);
        } else {
            out.push_str(&format!("~{byte:02X}"));
        }
    }
    // Windows reserves device names such as `con` and `lpt1`, even with an
    // extension, so escape the first letter of one to keep it writable.
    if is_windows_device_name(&out) {
        out = format!("~{:02X}{}", out.as_bytes()[0], &out[1..]);
    }
    out
}

/// Whether `name` is a Win32 reserved device name (`con`, `prn`, `aux`,
/// `nul`, `com0`–`com9`, `lpt0`–`lpt9`). Encoded components are lowercase.
fn is_windows_device_name(name: &str) -> bool {
    match name.as_bytes() {
        b"con" | b"prn" | b"aux" | b"nul" => true,
        [b'c', b'o', b'm', d] | [b'l', b'p', b't', d] => d.is_ascii_digit(),
        _ => false,
    }
}

#[async_trait]
impl AgentSessionStore for FileAgentSessionStore {
    async fn save_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
        session: &AgentSession,
    ) -> Result<()> {
        let path = self.session_path(agent, key)?;
        let json = serde_json::to_vec_pretty(&serialize_session(session))
            .map_err(|e| Error::Serialization(format!("failed to serialize session: {e}")))?;
        let dir = path.parent().expect("session paths have a parent");
        tokio::fs::create_dir_all(dir).await.map_err(|e| {
            Error::other(format!("failed to create session directory {dir:?}: {e}"))
        })?;
        // Named independently of the destination, so a destination name
        // near the 255-byte limit still leaves room for it.
        let tmp = dir.join(format!(".{}.tmp", uuid::Uuid::new_v4()));
        if let Err(e) = tokio::fs::write(&tmp, &json).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::other(format!(
                "failed to write session file {tmp:?}: {e}"
            )));
        }
        if let Err(e) = tokio::fs::rename(&tmp, &path).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::other(format!(
                "failed to finalize session file {path:?}: {e}"
            )));
        }
        Ok(())
    }

    async fn get_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<Option<AgentSession>> {
        let path = self.session_path(agent, key)?;
        let data = match tokio::fs::read(&path).await {
            Ok(data) => data,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => {
                return Err(Error::other(format!(
                    "failed to read session file {path:?}: {e}"
                )))
            }
        };
        let snapshot: Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Serialization(format!("failed to parse session file {path:?}: {e}"))
        })?;
        deserialize_session(agent, &snapshot).map(Some)
    }

    async fn delete_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<()> {
        let path = self.session_path(agent, key)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(Error::other(format!(
                "failed to delete session file {path:?}: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_refuse_blank_parts_and_ignore_partition_order() {
        assert!(AgentSessionStoreKey::new(" ").is_err());
        let key = AgentSessionStoreKey::new("s1").unwrap();
        assert!(key.clone().with_partition("", "x").is_err());
        assert!(key.clone().with_partition("tenant", " ").is_err());
        let a = key
            .clone()
            .with_partition("tenant", "t1")
            .unwrap()
            .with_partition("user", "u1")
            .unwrap();
        let b = key
            .with_partition("user", "u1")
            .unwrap()
            .with_partition("tenant", "t1")
            .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn encoding_is_safe_and_injective() {
        assert_eq!(encode_component("abc-_9"), "abc-_9");
        assert_eq!(encode_component("../x"), "~2E~2E~2Fx");
        assert_eq!(encode_component("~"), "~7E");
        // `~2F` as text and `/` must not collide.
        assert_ne!(encode_component("~2F"), encode_component("/"));
        assert_eq!(encode_component("con"), "~63on");
        assert_eq!(encode_component("lpt1"), "~6Cpt1");
        assert_eq!(encode_component("console"), "console");
        assert_eq!(encode_component("com10"), "com10");
        assert_ne!(encode_component("con"), encode_component("~63on"));
    }
}
