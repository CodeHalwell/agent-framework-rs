//! A [`ContextProvider`] that stores and retrieves conversation memories in
//! Redis, scoped by application/agent/user/thread id.
//!
//! Mirrors the *scoping and prompt-injection* behavior of the Python
//! `agent_framework_redis.RedisProvider`: `invoked()` persists the
//! request/response exchange as JSON entries tagged with the provider's
//! scope, and `invoking()` retrieves matching entries and injects them into
//! the conversation as a single `user`-role [`ChatMessage`] prefixed by
//! [`DEFAULT_CONTEXT_PROMPT`] (same default text as Python's
//! `ContextProvider.DEFAULT_CONTEXT_PROMPT`).
//!
//! # Divergence from Python: no RediSearch / vector search
//!
//! **This is the single biggest behavioral difference from the Python
//! package and is called out prominently here on purpose.** Python's
//! `RedisProvider` is built on [redisvl](https://github.com/redis/redis-vl-python)
//! and RediSearch: it maintains a `FT.CREATE`d index with `TAG`/`TEXT`/`VECTOR`
//! fields, and `invoking()` runs a server-side `TextQuery` (BM25 full-text)
//! or `HybridQuery` (BM25 + KNN vector similarity, when a vectorizer is
//! configured) against that index.
//!
//! RediSearch is a separate Redis module (not bundled with open-source
//! Redis/`redis-server`, and not exposed by the plain `redis` crate used
//! here) and embedding/vector similarity is out of scope for this port.
//! Instead, [`RedisContextProvider`]:
//!
//! 1. Stores each memory as its own key `{key_prefix}:entry:{uuid}` holding
//!    a JSON blob (content, role, scope fields, a client-assigned recency
//!    rank) — no index, no schema, no `FT.CREATE`/`overwrite_index`.
//! 2. On `invoking()`, enumerates candidates with `SCAN MATCH
//!    {key_prefix}:entry:*` (cursor-based, non-blocking — never `KEYS`),
//!    `MGET`s their values, and **filters entirely client-side**:
//!    - scope filter: exact-match AND over whichever of
//!      application/agent/user/thread id are configured (same semantics as
//!      Python's `Tag(k) == v` conjunction), then
//!    - *optional* naive full-text filter: if the query text is non-empty,
//!      split it into lowercased alphanumeric tokens, drop a small built-in
//!      stopword list (`the`, `is`, `of`, ...), and keep only entries whose
//!      content *contains* (substring, not word-boundary) at least one
//!      remaining token — a keyword check, not BM25 ranking or embeddings,
//!      then
//!    - sort by recency (most recent first) and take the configured
//!      `limit` (default 10, matching Python's `num_results` default).
//!
//! There is no relevance ranking beyond "did a token match" plus recency,
//! no stemming, and — because every `invoking()` call does a full
//! `SCAN` over the prefix — retrieval is `O(entries under key_prefix)`
//! rather than `O(log n)`/index-accelerated. This is adequate for modest
//! memory volumes (demos, tests, small deployments) but is **not** a
//! drop-in performance or relevance replacement for the RediSearch-backed
//! Python provider. Reach for a real vector store (or wait for a RediSearch
//! binding to land in the Rust `redis` crate ecosystem) before relying on
//! this for large-scale semantic recall.
//!
//! `application_id` participates in the scope filter here — unlike in the
//! sibling `agent-framework-mem0` crate's `Mem0Provider`, where it is
//! write-only metadata — matching the Python `RedisProvider`, which
//! includes `application_id` in its combined `Tag` filter for both reads
//! and writes.

use async_trait::async_trait;
use redis::AsyncCommands;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{Context, ContextProvider};
use agent_framework_core::types::{ChatMessage, Role};

use crate::internal::{map_redis_err, LazyConnection};

/// Default Redis key prefix, matching the Python provider's default `prefix`.
pub const DEFAULT_KEY_PREFIX: &str = "context";

/// Default context-injection header, byte-for-byte identical to Python's
/// `agent_framework.ContextProvider.DEFAULT_CONTEXT_PROMPT`.
pub const DEFAULT_CONTEXT_PROMPT: &str =
    "## Memories\nConsider the following memories when answering user questions:";

/// Default number of memories returned by `invoking()`, matching Python's
/// `RedisProvider._redis_search(..., num_results=10)` default.
pub const DEFAULT_LIMIT: usize = 10;

/// The wire format for one stored memory: a JSON blob written to (and read
/// back from) a single Redis string key. Private to this module — the
/// SCAN-based simplification documented above is entirely our own, so there
/// is no Python shape to mirror here (contrast with
/// [`crate::RedisChatMessageStore`], whose per-message JSON *is* meant to be
/// wire-compatible-in-spirit with the Python store).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MemoryEntry {
    content: String,
    role: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    application_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    agent_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    user_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    thread_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    author_name: Option<String>,
    /// Client-assigned recency rank: `unix_millis * 1000 + index_in_batch`.
    /// Higher is more recent. Avoids a round trip through a Redis-side
    /// counter while still giving messages added in the same `invoked()`
    /// call (which may share a millisecond) a stable relative order.
    rank: i64,
}

/// The scope a provider instance is configured with. `None` fields are
/// wildcards (unconstrained); `Some` fields must match exactly. Mirrors the
/// ANDed `Tag(k) == v for k, v in filters.items() if v` behavior of
/// Python's `RedisProvider._build_filter_from_dict`.
#[derive(Debug, Clone, Default, PartialEq)]
struct Scope {
    application_id: Option<String>,
    agent_id: Option<String>,
    user_id: Option<String>,
    thread_id: Option<String>,
}

impl Scope {
    fn is_empty(&self) -> bool {
        self.application_id.is_none()
            && self.agent_id.is_none()
            && self.user_id.is_none()
            && self.thread_id.is_none()
    }

    fn matches(&self, entry: &MemoryEntry) -> bool {
        fn field_matches(configured: &Option<String>, actual: &Option<String>) -> bool {
            match configured {
                None => true,
                Some(want) => actual.as_deref() == Some(want.as_str()),
            }
        }
        field_matches(&self.application_id, &entry.application_id)
            && field_matches(&self.agent_id, &entry.agent_id)
            && field_matches(&self.user_id, &entry.user_id)
            && field_matches(&self.thread_id, &entry.thread_id)
    }
}

/// Extremely common English function words, excluded from the query token
/// set before matching. Without this, a query like "What is the capital of
/// France?" would token-match almost any stored memory purely because both
/// share the word "the" — full-blown stopword lists / stemming are exactly
/// the machinery a real search engine (or RediSearch) provides and this
/// crate does not; this is the minimum needed to keep "simple" from also
/// meaning "matches everything".
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "and", "or", "but", "not", "with", "by", "from", "as", "it", "its", "this",
    "that", "these", "those", "i", "you", "he", "she", "we", "they", "what", "which", "who",
    "whom", "do", "does", "did", "have", "has", "had", "my", "your", "me", "about",
];

/// Pure selection logic — no I/O, fully unit-testable: scope-filter, then
/// (if `query` is non-empty) keep only entries with at least one matching
/// lowercase, non-stopword token, then sort by recency and take the top
/// `limit`.
fn select_recent(
    mut entries: Vec<MemoryEntry>,
    scope: &Scope,
    query: Option<&str>,
    limit: usize,
) -> Vec<MemoryEntry> {
    entries.retain(|e| scope.matches(e));

    if let Some(q) = query {
        let tokens: Vec<String> = q
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty() && !STOPWORDS.contains(t))
            .map(str::to_string)
            .collect();
        if !tokens.is_empty() {
            entries.retain(|e| {
                let content_lower = e.content.to_lowercase();
                tokens.iter().any(|t| content_lower.contains(t.as_str()))
            });
        }
    }

    entries.sort_by(|a, b| b.rank.cmp(&a.rank));
    entries.truncate(limit);
    entries
}

fn is_storable_role(role: &Role) -> bool {
    let r = role.as_str();
    r == Role::USER || r == Role::ASSISTANT || r == Role::SYSTEM
}

fn format_context(context_prompt: &str, joined_memories: &str) -> Context {
    if joined_memories.is_empty() {
        Context::default()
    } else {
        Context {
            messages: vec![ChatMessage::user(format!(
                "{context_prompt}\n{joined_memories}"
            ))],
            ..Default::default()
        }
    }
}

fn now_millis() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

/// Redis-backed [`ContextProvider`]: recency-based memory storage/retrieval
/// scoped by application/agent/user/thread id. See the module docs for the
/// RediSearch-vs-SCAN divergence from Python's `RedisProvider`.
///
/// ```no_run
/// use agent_framework_redis::RedisContextProvider;
/// use agent_framework_core::memory::ContextProvider;
/// use agent_framework_core::types::ChatMessage;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let provider = RedisContextProvider::new("redis://127.0.0.1:6379")?
///     .with_user_id("user-42")
///     .with_limit(5);
///
/// let request = vec![ChatMessage::user("I love hiking in the Cascades")];
/// provider.invoked(&request, &[]).await?;
///
/// let ctx = provider.invoking(&[ChatMessage::user("Any outdoor hobbies?")]).await?;
/// # Ok(())
/// # }
/// ```
pub struct RedisContextProvider {
    conn: LazyConnection,
    key_prefix: String,
    application_id: Option<String>,
    agent_id: Option<String>,
    user_id: Option<String>,
    thread_id: Option<String>,
    scope_to_per_operation_thread_id: bool,
    context_prompt: String,
    limit: usize,
    per_operation_thread_id: Mutex<Option<String>>,
}

impl RedisContextProvider {
    /// Create a provider for `redis_url` with no scope configured yet (at
    /// least one of application/agent/user/thread id must be set via the
    /// builder methods before `invoking`/`invoked` are called).
    pub fn new(redis_url: impl Into<String>) -> Result<Self> {
        let redis_url = redis_url.into();
        let conn = LazyConnection::open(&redis_url)?;
        Ok(Self {
            conn,
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            application_id: None,
            agent_id: None,
            user_id: None,
            thread_id: None,
            scope_to_per_operation_thread_id: false,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            limit: DEFAULT_LIMIT,
            per_operation_thread_id: Mutex::new(None),
        })
    }

    /// Namespace Redis keys under `key_prefix` (builder style). Defaults to
    /// `"context"`.
    pub fn with_key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = key_prefix.into();
        self
    }

    /// Scope memories to an application id (builder style).
    pub fn with_application_id(mut self, application_id: impl Into<String>) -> Self {
        self.application_id = Some(application_id.into());
        self
    }

    /// Scope memories to an agent id (builder style).
    pub fn with_agent_id(mut self, agent_id: impl Into<String>) -> Self {
        self.agent_id = Some(agent_id.into());
        self
    }

    /// Scope memories to a user id (builder style).
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Scope memories to a thread id (builder style).
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// When `true`, the thread id used for scoping is captured from the
    /// first [`ContextProvider::thread_created`] call instead of the
    /// static `thread_id` above, and a conflicting thread id on a later
    /// call is an error (builder style).
    pub fn with_scope_to_per_operation_thread_id(mut self, value: bool) -> Self {
        self.scope_to_per_operation_thread_id = value;
        self
    }

    /// Override the header prepended to injected memories (builder style).
    /// Defaults to [`DEFAULT_CONTEXT_PROMPT`].
    pub fn with_context_prompt(mut self, context_prompt: impl Into<String>) -> Self {
        self.context_prompt = context_prompt.into();
        self
    }

    /// Maximum memories returned per `invoking()` call (builder style).
    /// Defaults to [`DEFAULT_LIMIT`].
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    fn validate_filters(&self) -> Result<()> {
        if self.application_id.is_none()
            && self.agent_id.is_none()
            && self.user_id.is_none()
            && self.thread_id.is_none()
        {
            return Err(Error::Configuration(
                "At least one of the filters: agent_id, user_id, application_id, or thread_id is required."
                    .into(),
            ));
        }
        Ok(())
    }

    async fn effective_thread_id(&self) -> Option<String> {
        if self.scope_to_per_operation_thread_id {
            self.per_operation_thread_id.lock().await.clone()
        } else {
            self.thread_id.clone()
        }
    }

    async fn scope(&self) -> Scope {
        Scope {
            application_id: self.application_id.clone(),
            agent_id: self.agent_id.clone(),
            user_id: self.user_id.clone(),
            thread_id: self.effective_thread_id().await,
        }
    }

    fn entry_key(&self, id: &str) -> String {
        format!("{}:entry:{}", self.key_prefix, id)
    }

    fn scan_pattern(&self) -> String {
        format!("{}:entry:*", self.key_prefix)
    }

    /// `SCAN MATCH {key_prefix}:entry:*`, then `MGET` the matched keys and
    /// parse each JSON value. Keys that vanish between `SCAN` and `MGET`
    /// (e.g. concurrently cleared) or whose value fails to parse are
    /// silently skipped rather than failing the whole call.
    async fn scan_entries(&self) -> Result<Vec<MemoryEntry>> {
        let mut conn = self.conn.get().await?;
        let pattern = self.scan_pattern();

        let mut keys: Vec<String> = Vec::new();
        {
            let mut iter: redis::AsyncIter<'_, String> =
                conn.scan_match(&pattern).await.map_err(map_redis_err)?;
            while let Some(item) = iter.next_item().await {
                keys.push(item.map_err(map_redis_err)?);
            }
        }
        if keys.is_empty() {
            return Ok(Vec::new());
        }

        let raw: Vec<Option<String>> = conn.mget(&keys).await.map_err(map_redis_err)?;
        Ok(raw
            .into_iter()
            .flatten()
            .filter_map(|v| serde_json::from_str::<MemoryEntry>(&v).ok())
            .collect())
    }
}

#[async_trait]
impl ContextProvider for RedisContextProvider {
    async fn thread_created(&self, thread_id: Option<&str>) -> Result<()> {
        let mut guard = self.per_operation_thread_id.lock().await;
        if self.scope_to_per_operation_thread_id {
            if let (Some(new_id), Some(existing)) = (thread_id, guard.as_deref()) {
                if new_id != existing {
                    return Err(Error::other(
                        "RedisContextProvider can only be used with one thread, when scope_to_per_operation_thread_id is True.",
                    ));
                }
            }
        }
        if guard.is_none() {
            *guard = thread_id.map(String::from);
        }
        Ok(())
    }

    async fn invoked(
        &self,
        request_messages: &[ChatMessage],
        response_messages: &[ChatMessage],
    ) -> Result<()> {
        self.validate_filters()?;
        let scope = self.scope().await;
        let now = now_millis();

        let entries: Vec<MemoryEntry> = request_messages
            .iter()
            .chain(response_messages.iter())
            .enumerate()
            .filter(|(_, m)| is_storable_role(&m.role))
            .filter_map(|(i, m)| {
                let text = m.text();
                if text.trim().is_empty() {
                    return None;
                }
                Some(MemoryEntry {
                    content: text,
                    role: m.role.as_str().to_string(),
                    application_id: scope.application_id.clone(),
                    agent_id: scope.agent_id.clone(),
                    user_id: scope.user_id.clone(),
                    thread_id: scope.thread_id.clone(),
                    message_id: m.message_id.clone(),
                    author_name: m.author_name.clone(),
                    rank: now * 1000 + i as i64,
                })
            })
            .collect();

        if entries.is_empty() {
            return Ok(());
        }

        let mut conn = self.conn.get().await?;
        let mut pipe = redis::pipe();
        pipe.atomic();
        for entry in &entries {
            let key = self.entry_key(&Uuid::new_v4().to_string());
            let payload = serde_json::to_string(entry)?;
            pipe.set(key, payload);
        }
        let _: () = pipe.query_async(&mut conn).await.map_err(map_redis_err)?;
        Ok(())
    }

    async fn invoking(&self, messages: &[ChatMessage]) -> Result<Context> {
        self.validate_filters()?;

        let input_text = messages
            .iter()
            .map(ChatMessage::text)
            .filter(|t| !t.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        if input_text.trim().is_empty() {
            // Python's RedisProvider unconditionally calls `_redis_search`,
            // which raises on empty text; we instead treat "nothing to
            // search for" as "no memories" — see module docs.
            return Ok(Context::default());
        }

        let scope = self.scope().await;
        if scope.is_empty() {
            // Unreachable given `validate_filters` above (it guarantees at
            // least one scope field), but keep `Scope` honest in isolation.
            return Ok(Context::default());
        }

        let entries = self.scan_entries().await?;
        let hits = select_recent(entries, &scope, Some(&input_text), self.limit);
        let joined = hits
            .iter()
            .map(|e| e.content.as_str())
            .filter(|c| !c.is_empty())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(format_context(&self.context_prompt, &joined))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider() -> RedisContextProvider {
        RedisContextProvider::new("redis://127.0.0.1:6379/0")
            .expect("valid redis url")
            .with_user_id("u1")
    }

    fn entry(content: &str, rank: i64) -> MemoryEntry {
        MemoryEntry {
            content: content.to_string(),
            role: "user".to_string(),
            application_id: None,
            agent_id: None,
            user_id: Some("u1".to_string()),
            thread_id: None,
            message_id: None,
            author_name: None,
            rank,
        }
    }

    // region: key construction

    #[test]
    fn entry_key_and_scan_pattern_use_key_prefix() {
        let p = provider();
        assert_eq!(p.entry_key("abc"), "context:entry:abc");
        assert_eq!(p.scan_pattern(), "context:entry:*");
    }

    #[test]
    fn custom_key_prefix_propagates() {
        let p = provider().with_key_prefix("myapp");
        assert_eq!(p.entry_key("abc"), "myapp:entry:abc");
        assert_eq!(p.scan_pattern(), "myapp:entry:*");
    }

    #[test]
    fn invalid_redis_url_is_rejected() {
        assert!(RedisContextProvider::new("not-a-redis-url").is_err());
    }

    // endregion

    // region: validate_filters

    #[test]
    fn validate_filters_rejects_no_scope() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        assert!(p.validate_filters().is_err());
    }

    #[test]
    fn validate_filters_accepts_any_single_scope_field() {
        let base = || RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        assert!(base().with_user_id("u").validate_filters().is_ok());
        assert!(base().with_agent_id("a").validate_filters().is_ok());
        assert!(base().with_application_id("ap").validate_filters().is_ok());
        assert!(base().with_thread_id("t").validate_filters().is_ok());
    }

    // endregion

    // region: Scope matching (pure, no server)

    #[test]
    fn scope_with_no_fields_matches_everything() {
        let scope = Scope::default();
        assert!(scope.matches(&entry("hello", 1)));
    }

    #[test]
    fn scope_field_is_wildcard_when_unset() {
        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        // agent_id/application_id/thread_id are None on both scope and
        // entry, so only user_id needs to match.
        assert!(scope.matches(&entry("hi", 1)));
    }

    #[test]
    fn scope_rejects_mismatched_field() {
        let scope = Scope {
            user_id: Some("someone-else".to_string()),
            ..Default::default()
        };
        assert!(!scope.matches(&entry("hi", 1)));
    }

    #[test]
    fn scope_requires_all_configured_fields_to_match() {
        let mut e = entry("hi", 1);
        e.agent_id = Some("agentA".to_string());
        let scope = Scope {
            user_id: Some("u1".to_string()),
            agent_id: Some("agentB".to_string()),
            ..Default::default()
        };
        assert!(!scope.matches(&e));
    }

    // endregion

    // region: select_recent (pure, no server)

    #[test]
    fn select_recent_orders_by_rank_descending() {
        let entries = vec![entry("old", 1), entry("newest", 3), entry("mid", 2)];
        let scope = Scope::default();
        let hits = select_recent(entries, &scope, None, 10);
        assert_eq!(
            hits.iter().map(|e| e.content.as_str()).collect::<Vec<_>>(),
            vec!["newest", "mid", "old"]
        );
    }

    #[test]
    fn select_recent_respects_limit() {
        let entries = vec![entry("a", 1), entry("b", 2), entry("c", 3)];
        let hits = select_recent(entries, &Scope::default(), None, 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].content, "c");
        assert_eq!(hits[1].content, "b");
    }

    #[test]
    fn select_recent_filters_by_scope() {
        let mut other_user = entry("secret", 5);
        other_user.user_id = Some("someone-else".to_string());
        let entries = vec![entry("mine", 1), other_user];

        let scope = Scope {
            user_id: Some("u1".to_string()),
            ..Default::default()
        };
        let hits = select_recent(entries, &scope, None, 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "mine");
    }

    #[test]
    fn select_recent_text_filter_matches_token_case_insensitively() {
        let entries = vec![
            entry("User likes outdoor activities", 1),
            entry("User lives in Seattle", 2),
            entry("Completely unrelated fact", 3),
        ];
        let hits = select_recent(entries, &Scope::default(), Some("SEATTLE weather"), 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].content, "User lives in Seattle");
    }

    #[test]
    fn select_recent_text_filter_excludes_non_matching() {
        let entries = vec![entry("apples and oranges", 1)];
        let hits = select_recent(entries, &Scope::default(), Some("bananas"), 10);
        assert!(hits.is_empty());
    }

    #[test]
    fn select_recent_none_query_skips_text_filter() {
        let entries = vec![entry("anything at all", 1)];
        let hits = select_recent(entries, &Scope::default(), None, 10);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn select_recent_blank_query_skips_text_filter() {
        let entries = vec![entry("anything at all", 1)];
        let hits = select_recent(entries, &Scope::default(), Some("   "), 10);
        assert_eq!(hits.len(), 1);
    }

    // endregion

    // region: format_context

    #[test]
    fn format_context_empty_when_no_hits() {
        let ctx = format_context(DEFAULT_CONTEXT_PROMPT, "");
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn format_context_builds_user_message_with_prompt_header() {
        let ctx = format_context(DEFAULT_CONTEXT_PROMPT, "A\nB");
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].role, Role::user());
        assert_eq!(
            ctx.messages[0].text(),
            "## Memories\nConsider the following memories when answering user questions:\nA\nB"
        );
    }

    // endregion

    // region: is_storable_role

    #[test]
    fn is_storable_role_allows_user_assistant_system() {
        assert!(is_storable_role(&Role::user()));
        assert!(is_storable_role(&Role::assistant()));
        assert!(is_storable_role(&Role::system()));
    }

    #[test]
    fn is_storable_role_rejects_tool() {
        assert!(!is_storable_role(&Role::tool()));
    }

    // endregion

    // region: thread_created / per-operation scoping (async, no server: pure Mutex state)

    #[tokio::test]
    async fn thread_created_sets_per_operation_thread_id() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        p.thread_created(Some("t1")).await.unwrap();
        assert_eq!(
            p.per_operation_thread_id.lock().await.as_deref(),
            Some("t1")
        );
    }

    #[tokio::test]
    async fn thread_created_does_not_overwrite_existing() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        p.thread_created(Some("t1")).await.unwrap();
        p.thread_created(Some("t1")).await.unwrap();
        assert_eq!(
            p.per_operation_thread_id.lock().await.as_deref(),
            Some("t1")
        );
    }

    #[tokio::test]
    async fn thread_created_conflict_when_scoped() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        p.thread_created(Some("t1")).await.unwrap();
        let err = p.thread_created(Some("t2")).await.unwrap_err();
        assert!(err.to_string().contains("only be used with one thread"));
    }

    #[tokio::test]
    async fn thread_created_allows_none_repeatedly() {
        let p = provider().with_scope_to_per_operation_thread_id(true);
        p.thread_created(None).await.unwrap();
        p.thread_created(None).await.unwrap();
        p.thread_created(Some("t1")).await.unwrap();
        assert_eq!(
            p.per_operation_thread_id.lock().await.as_deref(),
            Some("t1")
        );
    }

    #[tokio::test]
    async fn thread_created_without_scoping_never_conflicts() {
        let p = provider(); // scope_to_per_operation_thread_id defaults to false
        p.thread_created(Some("t1")).await.unwrap();
        // No conflict error even though the id changes, since scoping is off.
        p.thread_created(Some("t2")).await.unwrap();
    }

    // endregion

    // region: invoking()/invoked() input validation (async, no server: fails before any I/O)

    #[tokio::test]
    async fn invoking_fails_without_scope_configured() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        let err = p.invoking(&[ChatMessage::user("hi")]).await.unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    #[tokio::test]
    async fn invoked_fails_without_scope_configured() {
        let p = RedisContextProvider::new("redis://127.0.0.1:6379/0").unwrap();
        let err = p
            .invoked(&[ChatMessage::user("hi")], &[])
            .await
            .unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    // endregion
}
