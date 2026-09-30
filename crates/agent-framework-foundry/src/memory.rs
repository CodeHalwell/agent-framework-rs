//! Microsoft Foundry managed memory as a [`ContextProvider`].
//!
//! Rust equivalent of upstream's `FoundryMemoryProvider`
//! (`agent_framework_foundry/_memory_provider.py`): before a run it searches a
//! Foundry **memory store** for memories relevant to the input and injects
//! them ahead of the conversation; after a run it writes the turn back so
//! later runs can retrieve it.
//!
//! # The wire surface
//!
//! Upstream reaches this through `azure-ai-projects`'
//! `project_client.beta.memory_stores`. There is no such SDK here, so the two
//! operations are spoken directly against the Foundry **project** endpoint —
//! the same endpoint [`crate::FoundryChatClient`] takes:
//!
//! * `POST {endpoint}/memory_stores/{name}:search_memories?api-version=v1`
//!   with `{scope, items?, previous_search_id?}`, answering
//!   `{search_id, memories: [{memory_item: {content, ...}}], usage}`.
//! * `POST {endpoint}/memory_stores/{name}:update_memories?api-version=v1`
//!   with `{scope, items?, previous_update_id?, update_delay?}`, answering
//!   `{update_id, status, ...}`.
//!
//! Both drop null fields rather than sending them, as the service contract
//! does. The bearer token is requested for [`crate::FOUNDRY_SCOPE`]
//! (`https://ai.azure.com/.default`) — the project audience, which is what
//! this control plane expects and what `AIProjectClient` itself defaults to.
//!
//! # Divergences from upstream
//!
//! - **State lives on the provider, keyed by scope.** Python's hooks receive a
//!   per-run `state` dict; the Rust [`ContextProvider`] has no such parameter,
//!   and [`ContextProvider::after_run`] receives no [`SessionContext`] at all.
//!   The initialization latch, the static memories and the incremental
//!   search/update ids therefore live in one [`Mutex`]-guarded map on the
//!   provider, **keyed by scope** — a provider is normally shared by `Arc`
//!   across an agent's runs, so a single flat state would let one session
//!   read another's profile and resume its cursor.
//!
//!   `after_run` is the harder half, because nothing in the call says which
//!   session it belongs to. When a scope is configured it is unambiguous.
//!   When it is derived from the session, the provider will write under that
//!   session only while it has seen exactly one; once a second appears the
//!   provider is shared and any write would be a guess, so it warns and
//!   declines rather than filing one user's conversation against another's.
//!   **Set [`FoundryMemoryProvider::with_scope`] on a provider shared across
//!   sessions** — the session fallback is a single-session convenience.
//! - **The update is awaited, not polled.** Upstream calls
//!   `begin_update_memories`, which returns a long-running-operation poller,
//!   and then reads only `update_id` off it without ever polling — a
//!   fire-and-forget write. The single POST here is exactly that much of the
//!   operation, so the observable behaviour matches; what is skipped is a
//!   poller upstream also skips.
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use agent_framework_foundry::{memory::FoundryMemoryProvider, FOUNDRY_SCOPE};
//! # use agent_framework_azure::AzureCliCredential;
//! # fn demo() -> agent_framework_core::error::Result<()> {
//! let provider = FoundryMemoryProvider::new(
//!     "https://my-res.services.ai.azure.com/api/projects/my-proj",
//!     "my-memory-store",
//!     Arc::new(AzureCliCredential::new(FOUNDRY_SCOPE)),
//! );
//! # Ok(())
//! # }
//! ```

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::{Message, Role};
use async_trait::async_trait;
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

/// The `api-version` the Foundry projects data plane takes, matching
/// `azure-ai-projects`' own default.
pub const DEFAULT_API_VERSION: &str = "v1";

/// How many scopes keep a cached profile and cursors at once, before the
/// least recently used is dropped. Override with
/// [`FoundryMemoryProvider::with_max_cached_scopes`].
pub const DEFAULT_MAX_CACHED_SCOPES: usize = 512;

/// The instruction prefixed to retrieved memories, as upstream words it.
pub const DEFAULT_CONTEXT_PROMPT: &str =
    "## Memories\nConsider the following memories when answering user questions:";

/// What upstream keeps in the per-run `state` dict its hooks are handed. The
/// Rust trait has no equivalent, so it lives on the provider — **keyed by
/// scope**, because a provider is normally shared by `Arc` across an agent's
/// runs and two scopes must not see each other's memories or resume each
/// other's cursor.
#[derive(Debug, Default)]
struct ScopeState {
    /// Whether the one-off static-memory (user profile) fetch has been
    /// attempted for this scope. Set even when that fetch *fails*, as
    /// upstream does, so a failing store is not re-queried on every run.
    initialized: bool,
    static_memories: Vec<String>,
    previous_search_id: Option<String>,
    previous_update_id: Option<String>,
}

/// Which session ids have come through `before_run`.
///
/// `after_run` is handed no [`SessionContext`], so when the scope is derived
/// from the session there is nothing in the call itself to correlate a write
/// to. While one session has been seen that is unambiguous. Once a second
/// appears, the provider is shared across sessions and any write it made
/// would be a guess — so it stops writing instead of writing to the wrong
/// scope. Configuring [`FoundryMemoryProvider::with_scope`] avoids the
/// question entirely and is the right shape for a shared provider.
#[derive(Debug, Default, PartialEq)]
enum SeenSessions {
    #[default]
    None,
    One(String),
    Many,
}

impl SeenSessions {
    fn observe(&mut self, session_id: &str) {
        match self {
            Self::None => *self = Self::One(session_id.to_string()),
            Self::One(existing) if existing == session_id => {}
            _ => *self = Self::Many,
        }
    }

    /// The session to scope by, when exactly one has ever been seen.
    fn unambiguous(&self) -> Option<&str> {
        match self {
            Self::One(id) => Some(id),
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    scopes: std::collections::HashMap<String, ScopeSlot>,
    sessions: SeenSessions,
    /// Monotonic counter standing in for a clock, so the least recently used
    /// scope can be identified without one.
    tick: u64,
}

/// A scope's cached state behind **its own** lock.
///
/// One lock per provider was wrong in both directions. Held across the
/// requests it serialized unrelated scopes; released across them it let two
/// runs for the *same* scope race — duplicating the profile fetch, and
/// forking the update-cursor chain when both snapshotted one
/// `previous_update_id` and both wrote back. A lock per scope is what the
/// operations actually need: the snapshot → request → commit sequence is
/// atomic within a scope, and scopes never wait on each other.
#[derive(Debug, Default)]
struct ScopeSlot {
    last_used: u64,
    state: Arc<Mutex<ScopeState>>,
}

impl State {
    /// The lock for `scope`, marking it most recently used and evicting the
    /// least recently used slot if the cache is full.
    ///
    /// Returns an `Arc`, so the caller holds the scope's lock without
    /// holding this one.
    ///
    /// Only a slot **nobody is using** may be evicted, which an earlier
    /// version of this comment got wrong by calling an in-flight eviction
    /// harmless. It is not: drop a slot a run still holds and the next run
    /// for that scope builds a *second* mutex, races the first, and both
    /// resume the same cursor — defeating precisely the per-scope
    /// serialization this cache sits behind. When every candidate is busy
    /// the map simply runs over capacity until one frees up, which is the
    /// right trade: the bound exists to stop unbounded growth, not to be
    /// honoured at the cost of correctness.
    fn touch(&mut self, scope: &str, capacity: usize) -> Arc<Mutex<ScopeState>> {
        self.tick += 1;
        let tick = self.tick;
        let slot = self.scopes.entry(scope.to_string()).or_default();
        slot.last_used = tick;
        // Cloned before trimming, so this scope is never its own victim.
        let handle = slot.state.clone();
        self.trim(capacity);
        handle
    }

    /// Evict idle slots, least recently used first, until the cache is back
    /// within `capacity`.
    ///
    /// Run on **every** touch, and looping rather than dropping one slot.
    /// Gating eviction on inserting a new scope, and stopping after a single
    /// victim, left no way back down: a burst that outgrew the bound because
    /// every slot was busy stayed oversized for good, since later touches of
    /// existing scopes skipped eviction entirely and each new-scope touch
    /// removed one and added one. Trimming here means the overrun that
    /// keeping in-flight scopes alive requires is temporary, which is the
    /// only thing that makes it acceptable.
    fn trim(&mut self, capacity: usize) {
        while self.scopes.len() > capacity {
            let victim = self
                .scopes
                .iter()
                // A count of one means this map holds the only reference, so
                // no run is inside that scope's lock.
                .filter(|(_, slot)| Arc::strong_count(&slot.state) == 1)
                .min_by_key(|(_, slot)| slot.last_used)
                .map(|(name, _)| name.clone());
            // Every remaining slot is in use: stay over capacity for now and
            // come back to it on a later touch.
            let Some(victim) = victim else { break };
            self.scopes.remove(&victim);
        }
    }
}

/// A [`ContextProvider`] backed by a Foundry memory store.
pub struct FoundryMemoryProvider {
    http: reqwest::Client,
    endpoint: String,
    memory_store_name: String,
    scope: Option<String>,
    context_prompt: String,
    update_delay: Option<u64>,
    api_version: String,
    credential: Arc<dyn TokenCredential>,
    token_scope: String,
    state: Mutex<State>,
    max_cached_scopes: usize,
    /// Latch so the "no scope" warning is logged once rather than per run.
    warned_no_scope: AtomicBool,
}

impl std::fmt::Debug for FoundryMemoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryMemoryProvider")
            .field("endpoint", &self.endpoint)
            .field("memory_store_name", &self.memory_store_name)
            .field("scope", &self.scope)
            .field("api_version", &self.api_version)
            .finish_non_exhaustive()
    }
}

impl FoundryMemoryProvider {
    /// Create a provider against a Foundry project endpoint and a memory
    /// store name.
    ///
    /// The credential should be scoped to [`crate::FOUNDRY_SCOPE`].
    pub fn new(
        endpoint: impl Into<String>,
        memory_store_name: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            endpoint: endpoint.into().trim_end_matches('/').to_string(),
            memory_store_name: memory_store_name.into(),
            scope: None,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            update_delay: None,
            api_version: DEFAULT_API_VERSION.to_string(),
            credential,
            token_scope: crate::FOUNDRY_SCOPE.to_string(),
            state: Mutex::new(State::default()),
            max_cached_scopes: DEFAULT_MAX_CACHED_SCOPES,
            warned_no_scope: AtomicBool::new(false),
        }
    }

    /// Pin the memory scope — the namespace that isolates one user's or one
    /// tenant's memories. Without it the session id is used, as upstream
    /// does (`self.scope or context.session_id`).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into());
        self
    }

    /// Override the instruction prefixed to injected memories.
    pub fn with_context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context_prompt = prompt.into();
        self
    }

    /// Seconds the service waits before processing an update, cancelling and
    /// restarting the timer if another update arrives meanwhile. `0` writes
    /// immediately. The service defaults to 300 when this is unset.
    pub fn with_update_delay(mut self, seconds: u64) -> Self {
        self.update_delay = Some(seconds);
        self
    }

    /// Cap how many scopes keep cached state (default
    /// [`DEFAULT_MAX_CACHED_SCOPES`]). Eviction costs a scope only a
    /// re-fetched profile and a restarted search cursor.
    pub fn with_max_cached_scopes(mut self, max: usize) -> Self {
        self.max_cached_scopes = max.max(1);
        self
    }

    /// Override the `api-version` query parameter.
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// Override the Entra ID scope requested for the bearer token.
    pub fn with_token_scope(mut self, scope: impl Into<String>) -> Self {
        self.token_scope = scope.into();
        self
    }

    /// `{endpoint}/memory_stores/{name}:{action}?api-version=…`
    ///
    /// The `:action` suffix is the service's own spelling for these two
    /// operations; it is not a path segment and must not be escaped.
    fn url(&self, action: &str) -> String {
        format!(
            "{}/memory_stores/{}:{}?api-version={}",
            self.endpoint, self.memory_store_name, action, self.api_version
        )
    }

    /// The scope to read and write under: the configured one, else the
    /// session id captured from the run.
    fn resolve_scope(&self, session_id: Option<&str>) -> Option<String> {
        self.scope
            .clone()
            .or_else(|| session_id.map(str::to_string))
    }

    /// Warn once that there is nothing to scope memories by, and skip.
    fn warn_no_scope(&self) {
        if !self.warned_no_scope.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                store = %self.memory_store_name,
                "Foundry memory: no scope is resolvable — none configured, and the session \
                 is absent or this provider has served more than one. Skipping memory access; \
                 set a fixed scope with `with_scope` when sharing one provider."
            );
        }
    }

    async fn post(&self, action: &str, body: &Value) -> Result<Value> {
        let token = self
            .credential
            .get_token_for_scope(&self.token_scope)
            .await?;
        let resp = self
            .http
            .post(self.url(action))
            .bearer_auth(token)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("Foundry memory request failed: {e}")))?;
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::service(format!(
                "Foundry memory API error {status}: {text}"
            )));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text)
            .map_err(|e| Error::service(format!("Foundry memory returned invalid JSON: {e}")))
    }

    /// One `:search_memories` call. `items` omitted fetches the scope's
    /// static memories (the user profile), which is what upstream's
    /// first-run call does.
    async fn search(
        &self,
        scope: &str,
        items: Option<Vec<Value>>,
        previous_search_id: Option<&str>,
    ) -> Result<SearchResult> {
        let mut body = Map::new();
        body.insert("scope".into(), json!(scope));
        if let Some(items) = items {
            body.insert("items".into(), json!(items));
        }
        if let Some(id) = previous_search_id {
            body.insert("previous_search_id".into(), json!(id));
        }
        let value = self.post("search_memories", &Value::Object(body)).await?;
        Ok(SearchResult {
            search_id: value
                .get("search_id")
                .and_then(Value::as_str)
                .map(str::to_string),
            contents: memory_contents(&value),
        })
    }
}

/// The part of a `:search_memories` response this provider reads.
#[derive(Debug, Default)]
struct SearchResult {
    search_id: Option<String>,
    contents: Vec<String>,
}

/// Pull `memories[].memory_item.content` out of a search response, skipping
/// entries that carry no text rather than injecting empty lines.
fn memory_contents(value: &Value) -> Vec<String> {
    value
        .get("memories")
        .and_then(Value::as_array)
        .map(|memories| {
            memories
                .iter()
                .filter_map(|m| {
                    m.get("memory_item")
                        .and_then(|i| i.get("content"))
                        .and_then(Value::as_str)
                })
                .filter(|c| !c.trim().is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Render messages as the service's `items` array.
///
/// Only `user` and `assistant` turns with text are sent. Upstream's role
/// filter also admits `system`, but the branch that follows it emits nothing
/// for that role, so a system turn is dropped there too — this reproduces the
/// effective behaviour rather than the dead half of the condition.
fn message_items(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .filter_map(|m| {
            let role = match m.role.as_str() {
                Role::USER => Role::USER,
                Role::ASSISTANT => Role::ASSISTANT,
                _ => return None,
            };
            let text = m.text();
            if text.trim().is_empty() {
                return None;
            }
            Some(json!({ "type": "message", "role": role, "content": text }))
        })
        .collect()
}

#[async_trait]
impl ContextProvider for FoundryMemoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        // The provider-wide lock is held only long enough to resolve the
        // scope and hand back *its* lock; the scope's own lock then covers
        // the whole snapshot → request → commit sequence. Unrelated scopes
        // never wait on each other, and two runs for the same scope cannot
        // duplicate the profile fetch or fork a cursor.
        let (scope, slot) = {
            let mut state = self.state.lock().await;
            if let Some(id) = ctx.session_id.as_deref() {
                state.sessions.observe(id);
            }
            // An explicit scope is authoritative; otherwise this run's own
            // session id, which is unambiguous here because the context
            // names it.
            let Some(scope) = self.resolve_scope(ctx.session_id.as_deref()) else {
                self.warn_no_scope();
                return Ok(());
            };
            let slot = state.touch(&scope, self.max_cached_scopes);
            (scope, slot)
        };
        let mut entry = slot.lock().await;

        // First run *for this scope*: fetch its static memories (user
        // profile).
        if !entry.initialized {
            match self.search(&scope, None, None).await {
                Ok(result) => entry.static_memories = result.contents,
                Err(e) => {
                    // Leave whatever is known in place. Nothing else can be
                    // mid-fetch for this scope now, but a *later* run should
                    // still not find a profile this failure wiped.
                    tracing::warn!(error = %e, "Foundry memory: static memory retrieval failed");
                }
            }
            // Set either way, as upstream does, so an unreachable store costs
            // one request rather than one per run.
            entry.initialized = true;
        }
        let mut memories = entry.static_memories.clone();

        // The contextual search needs something to search *with*. An
        // instruction-only run, or one carrying nothing but system or tool
        // turns, has nothing — but the profile fetched above still belongs in
        // the context, so this skips the request rather than the injection.
        // (Upstream returns outright here, losing the profile; that made the
        // empty-input path disagree with the search-failed path below, which
        // does inject it.)
        let items = message_items(&ctx.input_messages);
        if !items.is_empty() {
            match self
                .search(&scope, Some(items), entry.previous_search_id.as_deref())
                .await
            {
                Ok(result) => {
                    // Upstream advances the incremental cursor only when the
                    // search actually returned something, so an empty answer
                    // does not reset the next search's starting point.
                    if !result.contents.is_empty() {
                        if let Some(id) = result.search_id {
                            entry.previous_search_id = Some(id);
                        }
                    }
                    memories.extend(result.contents);
                }
                Err(e) => {
                    // Retrieval is an enhancement: a memory store that is
                    // down must not take the agent down with it, and must not
                    // cost the run a profile already in hand.
                    tracing::warn!(error = %e, "Foundry memory: contextual search failed");
                }
            }
        }

        if memories.is_empty() {
            return Ok(());
        }
        ctx.messages.push(Message::user(format!(
            "{}\n{}",
            self.context_prompt,
            memories.join("\n")
        )));
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        let (scope, slot) = {
            let mut state = self.state.lock().await;
            // This call carries no session, so the scope must come from
            // configuration or from the one session this provider has seen.
            // Writing under a guessed scope would file one user's
            // conversation against another's, so ambiguity means not writing.
            let Some(scope) = self
                .scope
                .clone()
                .or_else(|| state.sessions.unambiguous().map(str::to_string))
            else {
                self.warn_no_scope();
                return Ok(());
            };
            let slot = state.touch(&scope, self.max_cached_scopes);
            (scope, slot)
        };

        let mut all: Vec<Message> = request_messages.to_vec();
        all.extend_from_slice(response_messages);
        let items = message_items(&all);
        if items.is_empty() {
            return Ok(());
        }

        // Held across the request, so two turns for this scope chain their
        // updates instead of both resuming the same `previous_update_id` and
        // forking the chain.
        let mut entry = slot.lock().await;
        let mut body = Map::new();
        body.insert("scope".into(), json!(scope));
        body.insert("items".into(), json!(items));
        if let Some(id) = &entry.previous_update_id {
            body.insert("previous_update_id".into(), json!(id));
        }
        if let Some(delay) = self.update_delay {
            body.insert("update_delay".into(), json!(delay));
        }

        match self.post("update_memories", &Value::Object(body)).await {
            Ok(value) => {
                if let Some(id) = value.get("update_id").and_then(Value::as_str) {
                    entry.previous_update_id = Some(id.to_string());
                }
            }
            // Storing is an enhancement too, and `after_run` also runs on the
            // failure path — raising here would replace the agent's own error
            // with this one.
            Err(e) => tracing::warn!(error = %e, "Foundry memory: update failed"),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_azure::StaticTokenCredential;

    fn provider() -> FoundryMemoryProvider {
        FoundryMemoryProvider::new(
            "https://res.services.ai.azure.com/api/projects/proj/",
            "store",
            Arc::new(StaticTokenCredential::new("t")),
        )
    }

    /// The `:action` suffix is part of the service's route spelling, and the
    /// endpoint's trailing slash must not double up.
    #[test]
    fn the_route_carries_the_action_suffix_and_api_version() {
        assert_eq!(
            provider().url("search_memories"),
            "https://res.services.ai.azure.com/api/projects/proj/memory_stores/store:search_memories?api-version=v1"
        );
        assert_eq!(
            provider()
                .url("update_memories")
                .split(':')
                .next_back()
                .unwrap(),
            "update_memories?api-version=v1"
        );
    }

    /// Upstream reads `self.scope or context.session_id`, in that order.
    #[test]
    fn an_explicit_scope_wins_over_the_session_id() {
        assert_eq!(
            provider().resolve_scope(Some("sess")).as_deref(),
            Some("sess")
        );
        let pinned = provider().with_scope("user-42");
        assert_eq!(
            pinned.resolve_scope(Some("sess")).as_deref(),
            Some("user-42")
        );
        assert_eq!(pinned.resolve_scope(None).as_deref(), Some("user-42"));
        // Neither available is the one case there is nothing to key by.
        assert!(provider().resolve_scope(None).is_none());
    }

    /// Only user and assistant turns with text become items. Upstream's role
    /// filter also lets `system` through, but its following branch emits
    /// nothing for it — so a system turn is dropped there too.
    #[test]
    fn only_user_and_assistant_turns_with_text_become_items() {
        let items = message_items(&[
            Message::user("hello"),
            Message::assistant("hi there"),
            Message::system("you are helpful"),
            Message::user("   "),
            Message::user(""),
        ]);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["role"], json!("user"));
        assert_eq!(items[0]["type"], json!("message"));
        assert_eq!(items[0]["content"], json!("hello"));
        assert_eq!(items[1]["role"], json!("assistant"));
    }

    #[test]
    fn memory_contents_skips_entries_with_no_text() {
        let body = json!({
            "search_id": "s1",
            "memories": [
                {"memory_item": {"content": "likes tea"}},
                {"memory_item": {"content": "   "}},
                {"memory_item": {}},
                {"other": 1},
                {"memory_item": {"content": "lives in Leeds"}}
            ]
        });
        assert_eq!(memory_contents(&body), vec!["likes tea", "lives in Leeds"]);
        // A response with no memories key at all is empty, not an error.
        assert!(memory_contents(&json!({"search_id": "s"})).is_empty());
    }

    /// Keeping in-flight scopes alive means the cache can exceed its bound.
    /// That is only acceptable if it comes back down — which it did not: the
    /// old rule evicted at most one slot, and only when inserting a new
    /// scope, so a burst of concurrent sessions stayed resident for good.
    #[tokio::test]
    async fn the_scope_cache_returns_to_capacity_after_a_burst() {
        let p = provider().with_max_cached_scopes(2);
        let mut state = p.state.lock().await;

        // Four scopes, all "in flight" — the handles stand in for runs
        // holding their locks, so none may be evicted.
        let held: Vec<_> = ["a", "b", "c", "d"]
            .iter()
            .map(|s| state.touch(s, p.max_cached_scopes))
            .collect();
        assert_eq!(
            state.scopes.len(),
            4,
            "an in-flight scope is never evicted, so the bound gives way"
        );

        // The runs finish.
        drop(held);

        // The next touch must bring it back within the bound: `e` plus one
        // survivor, the most recently used of the idle four.
        let _e = state.touch("e", p.max_cached_scopes);
        assert_eq!(state.scopes.len(), 2, "the overrun must be temporary");
        assert!(state.scopes.contains_key("e"));
        assert!(
            state.scopes.contains_key("d"),
            "eviction is least-recently-used, so the newest survivor stays"
        );
    }

    /// Nothing to scope by must not become a request with a missing required
    /// field; it skips, and warns only once however many runs go through.
    #[tokio::test]
    async fn without_a_scope_the_provider_skips_rather_than_calling() {
        let p = provider();
        let mut ctx = SessionContext::new(vec![Message::user("hi")]);
        p.before_run(&mut ctx).await.unwrap();
        p.after_run(&[Message::user("hi")], &[], None)
            .await
            .unwrap();
        assert!(ctx.messages.is_empty(), "nothing should be injected");
        assert!(p.warned_no_scope.load(Ordering::Relaxed));
    }
}
