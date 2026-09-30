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
    scopes: std::collections::HashMap<String, ScopeState>,
    sessions: SeenSessions,
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
        let mut state = self.state.lock().await;
        if let Some(id) = ctx.session_id.as_deref() {
            state.sessions.observe(id);
        }
        // An explicit scope is authoritative; otherwise this run's own
        // session id, which is unambiguous here because the context names it.
        let Some(scope) = self.resolve_scope(ctx.session_id.as_deref()) else {
            self.warn_no_scope();
            return Ok(());
        };
        let entry = state.scopes.entry(scope.clone()).or_default();

        // First run *for this scope*: fetch its static memories (user
        // profile). The latch is set even on failure, as upstream does, so
        // one unreachable store does not mean one failed request per run
        // forever.
        if !entry.initialized {
            match self.search(&scope, None, None).await {
                Ok(result) => {
                    state
                        .scopes
                        .entry(scope.clone())
                        .or_default()
                        .static_memories = result.contents
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Foundry memory: static memory retrieval failed");
                    state
                        .scopes
                        .entry(scope.clone())
                        .or_default()
                        .static_memories
                        .clear();
                }
            }
            state.scopes.entry(scope.clone()).or_default().initialized = true;
        }

        let items = message_items(&ctx.input_messages);
        if items.is_empty() {
            return Ok(());
        }

        let previous = state
            .scopes
            .get(&scope)
            .and_then(|e| e.previous_search_id.clone());
        let contextual = match self.search(&scope, Some(items), previous.as_deref()).await {
            Ok(result) => {
                // Upstream advances the incremental cursor only when the
                // search actually returned something, so an empty answer does
                // not reset the next search's starting point.
                if !result.contents.is_empty() {
                    if let Some(id) = result.search_id {
                        state
                            .scopes
                            .entry(scope.clone())
                            .or_default()
                            .previous_search_id = Some(id);
                    }
                }
                result.contents
            }
            Err(e) => {
                // Retrieval is an enhancement: a memory store that is down
                // must not take the agent down with it.
                tracing::warn!(error = %e, "Foundry memory: contextual search failed");
                return Ok(());
            }
        };

        let mut all = state
            .scopes
            .get(&scope)
            .map(|e| e.static_memories.clone())
            .unwrap_or_default();
        all.extend(contextual);
        if all.is_empty() {
            return Ok(());
        }
        ctx.messages.push(Message::user(format!(
            "{}\n{}",
            self.context_prompt,
            all.join("\n")
        )));
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        let mut state = self.state.lock().await;
        // This call carries no session, so the scope must come from
        // configuration or from the one session this provider has seen.
        // Writing under a guessed scope would file one user's conversation
        // against another's, so ambiguity means not writing.
        let Some(scope) = self
            .scope
            .clone()
            .or_else(|| state.sessions.unambiguous().map(str::to_string))
        else {
            self.warn_no_scope();
            return Ok(());
        };

        let mut all: Vec<Message> = request_messages.to_vec();
        all.extend_from_slice(response_messages);
        let items = message_items(&all);
        if items.is_empty() {
            return Ok(());
        }

        let mut body = Map::new();
        body.insert("scope".into(), json!(scope));
        body.insert("items".into(), json!(items));
        if let Some(id) = state
            .scopes
            .get(&scope)
            .and_then(|e| e.previous_update_id.clone())
        {
            body.insert("previous_update_id".into(), json!(id));
        }
        if let Some(delay) = self.update_delay {
            body.insert("update_delay".into(), json!(delay));
        }

        match self.post("update_memories", &Value::Object(body)).await {
            Ok(value) => {
                if let Some(id) = value.get("update_id").and_then(Value::as_str) {
                    state.scopes.entry(scope).or_default().previous_update_id =
                        Some(id.to_string());
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
