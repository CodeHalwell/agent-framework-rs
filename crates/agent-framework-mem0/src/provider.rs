//! [`Mem0Provider`]: a [`ContextProvider`] backed by the hosted
//! [Mem0](https://mem0.ai) memory API.
//!
//! Mirrors `agent_framework_mem0.Mem0Provider` from the Python Agent
//! Framework: `invoked()` adds the request/response exchange to Mem0,
//! scoped by `user_id`/`agent_id`/`run_id`/`application_id`; `invoking()`
//! searches Mem0 with the latest input text (plus the same scope) and
//! injects the hits into the conversation as a single `user`-role
//! [`ChatMessage`] prefixed by [`DEFAULT_CONTEXT_PROMPT`].
//!
//! # Divergence from Python: hand-rolled REST calls, not the `mem0` SDK
//!
//! The Python package calls `mem0.AsyncMemoryClient.add(...)` /
//! `.search(...)`; this crate has no Python SDK to lean on, so it talks to
//! the REST API directly with `reqwest`, matching this work package's
//! explicit endpoint choice:
//!
//! - `invoked()` &rarr; `POST {api_base}/v1/memories/` with a body shaped
//!   exactly like the Python provider's kwargs to `.add()`: top-level
//!   `messages`, `user_id`, `agent_id`, `run_id`, and
//!   `metadata: { application_id }` — all four scope-ish fields are sent
//!   even when `None` (serializing to JSON `null`), because that is
//!   observably what the Python provider passes to the SDK (see
//!   `test_messages_adding_with_agent_id`, which asserts
//!   `call_args.kwargs["user_id"] is None`).
//! - `invoking()` &rarr; `POST {api_base}/v2/memories/search/`. Here the
//!   Python provider's call (`mem0_client.search(query=..., user_id=...,
//!   agent_id=..., run_id=...)`) does *not* map onto the documented `/v2/`
//!   wire contract directly — v2 search expects scope constraints nested
//!   under a `filters` object, so the SDK must be doing that translation
//!   internally. This crate performs the same translation explicitly: a
//!   `filters` object containing only the scope fields that are actually
//!   set (unset fields are *omitted*, not sent as `null` — a `null` equality
//!   filter is not the same thing as "no constraint" and real Mem0 deployments
//!   reject an empty/all-null filter with a 4xx).
//! - Mem0's public API has since moved on to a versioned `/v3/` surface; this
//!   crate intentionally targets `/v1/` and `/v2/` to match the Python
//!   package's behavior at the time of this port rather than the latest
//!   hosted API. If you're integrating against a Mem0 deployment that has
//!   dropped `/v1/`/`/v2/` support, override the base URL and adjust, or
//!   treat this crate as a template.
//! - `application_id` here is **write-only**: it rides along as
//!   `metadata.application_id` on `invoked()` but is never sent as a search
//!   constraint on `invoking()`, matching the Python provider (whose
//!   `invoking()` only ever forwards `user_id`/`agent_id`/`run_id` to
//!   `.search()`). This is unlike the sibling `agent-framework-redis`
//!   crate's `RedisContextProvider`, where `application_id` participates in
//!   the scope filter on both reads and writes.
//! - Response parsing defensively accepts either a bare JSON array (the
//!   `v2`/list-style response) or `{"results": [...]}` (the historical
//!   `v1.1`-style response), exactly like the Python provider's
//!   `isinstance` check — because, again, without the SDK's source this
//!   crate can't be certain which shape a given deployment returns.
//! - Authentication uses the documented `Authorization: Token <api_key>`
//!   header scheme.
//!
//! No knobs for search result limit/threshold are exposed because the
//! Python `Mem0Provider` doesn't expose any either (unlike, say, a
//! `top_k`/`threshold` pair some Mem0 endpoints accept) — this crate only
//! adds configuration surface area the Python provider actually has.

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{Context, ContextProvider};
use agent_framework_core::types::{ChatMessage, Role};

/// Default Mem0 API base URL, matching the hosted service.
pub const DEFAULT_API_BASE: &str = "https://api.mem0.ai";

/// `POST` path for adding memories (v1), relative to the API base.
pub const ADD_PATH: &str = "/v1/memories/";

/// `POST` path for searching memories (v2), relative to the API base.
pub const SEARCH_PATH: &str = "/v2/memories/search/";

/// Default context-injection header, byte-for-byte identical to Python's
/// `agent_framework.ContextProvider.DEFAULT_CONTEXT_PROMPT`.
pub const DEFAULT_CONTEXT_PROMPT: &str =
    "## Memories\nConsider the following memories when answering user questions:";

fn is_storable_role(role: &Role) -> bool {
    let r = role.as_str();
    r == Role::USER || r == Role::ASSISTANT || r == Role::SYSTEM
}

/// Build the `/v1/memories/` request body from the request+response
/// exchange, or `None` if there are no messages worth persisting (mirrors
/// the Python provider's `if messages: await self.mem0_client.add(...)`
/// guard — the HTTP call is skipped entirely rather than sent empty).
fn build_add_body(
    messages: &[&ChatMessage],
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
    application_id: Option<&str>,
) -> Option<Value> {
    let shaped: Vec<Value> = messages
        .iter()
        .filter(|m| is_storable_role(&m.role) && !m.text().trim().is_empty())
        .map(|m| serde_json::json!({ "role": m.role.as_str(), "content": m.text() }))
        .collect();
    if shaped.is_empty() {
        return None;
    }
    Some(serde_json::json!({
        "messages": shaped,
        "user_id": user_id,
        "agent_id": agent_id,
        "run_id": run_id,
        "metadata": { "application_id": application_id },
    }))
}

/// Build the `/v2/memories/search/` request body. Unlike [`build_add_body`],
/// scope fields that are `None` are *omitted* from `filters` rather than
/// sent as `null` — see the module docs for why.
fn build_search_body(
    query: &str,
    user_id: Option<&str>,
    agent_id: Option<&str>,
    run_id: Option<&str>,
) -> Value {
    let mut filters = serde_json::Map::new();
    if let Some(v) = user_id {
        filters.insert("user_id".to_string(), Value::String(v.to_string()));
    }
    if let Some(v) = agent_id {
        filters.insert("agent_id".to_string(), Value::String(v.to_string()));
    }
    if let Some(v) = run_id {
        filters.insert("run_id".to_string(), Value::String(v.to_string()));
    }
    serde_json::json!({
        "query": query,
        "filters": Value::Object(filters),
    })
}

/// Parse a `/v2/memories/search/` response into the flat list of memory
/// texts, matching the Python provider's defensive `isinstance` handling:
/// a bare array, a `{"results": [...]}` wrapper, or (as a last resort) a
/// single unrecognized object treated as one result. Entries without a
/// `"memory"` field contribute an empty string rather than being dropped,
/// matching Python's unconditional `memory.get("memory", "")`.
fn parse_search_response(value: &Value) -> Vec<String> {
    let memories: Vec<&Value> = match value {
        Value::Array(items) => items.iter().collect(),
        Value::Object(map) if map.contains_key("results") => map
            .get("results")
            .and_then(Value::as_array)
            .map(|a| a.iter().collect())
            .unwrap_or_default(),
        other => vec![other],
    };
    memories
        .into_iter()
        .map(|m| {
            m.get("memory")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        })
        .collect()
}

/// Join memory texts and wrap them in a `user`-role [`ChatMessage`] under
/// `context_prompt`, or an empty [`Context`] if there's nothing to inject
/// (mirrors Python's `Context(messages=[...] if line_separated_memories
/// else None)`).
fn format_context(context_prompt: &str, memory_texts: &[String]) -> Context {
    let joined = memory_texts.join("\n");
    if joined.is_empty() {
        Context::default()
    } else {
        Context {
            messages: vec![ChatMessage::user(format!("{context_prompt}\n{joined}"))],
            ..Default::default()
        }
    }
}

/// Map a non-2xx HTTP response into a service [`Error`]. Factored out as a
/// pure function so error-surfacing can be unit-tested without a network
/// call.
fn map_http_error(status: reqwest::StatusCode, body: &str) -> Error {
    Error::service(format!("Mem0 API error {status}: {body}"))
}

/// A [`ContextProvider`] backed by the hosted Mem0 memory API. See the
/// module docs for the REST contract this crate targets and how it differs
/// from the Python package's SDK-mediated behavior.
///
/// ```no_run
/// use agent_framework_mem0::Mem0Provider;
/// use agent_framework_core::memory::ContextProvider;
/// use agent_framework_core::types::ChatMessage;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let provider = Mem0Provider::from_env()?.with_user_id("user-42");
///
/// let request = vec![ChatMessage::user("I moved to Austin last month")];
/// provider.invoked(&request, &[], None).await?;
///
/// let ctx = provider
///     .invoking(&[ChatMessage::user("Where do I live?")])
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct Mem0Provider {
    http: reqwest::Client,
    api_key: String,
    api_base: String,
    application_id: Option<String>,
    agent_id: Option<String>,
    user_id: Option<String>,
    thread_id: Option<String>,
    scope_to_per_operation_thread_id: bool,
    context_prompt: String,
    per_operation_thread_id: Mutex<Option<String>>,
}

impl std::fmt::Debug for Mem0Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mem0Provider")
            .field("api_base", &self.api_base)
            .field("application_id", &self.application_id)
            .field("agent_id", &self.agent_id)
            .field("user_id", &self.user_id)
            .field("thread_id", &self.thread_id)
            .field(
                "scope_to_per_operation_thread_id",
                &self.scope_to_per_operation_thread_id,
            )
            .finish_non_exhaustive()
    }
}

impl Mem0Provider {
    /// Create a provider authenticating with `api_key`, using
    /// [`DEFAULT_API_BASE`] and no scope configured yet (at least one of
    /// application/agent/user/thread id must be set via the builder methods
    /// before `invoking`/`invoked` are called).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            api_base: DEFAULT_API_BASE.to_string(),
            application_id: None,
            agent_id: None,
            user_id: None,
            thread_id: None,
            scope_to_per_operation_thread_id: false,
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            per_operation_thread_id: Mutex::new(None),
        }
    }

    /// Build a provider from the `MEM0_API_KEY` (required) and
    /// `MEM0_API_BASE` (optional override of [`DEFAULT_API_BASE`])
    /// environment variables.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MEM0_API_KEY")
            .map_err(|_| Error::Configuration("MEM0_API_KEY is not set".into()))?;
        let mut provider = Self::new(api_key);
        if let Ok(base) = std::env::var("MEM0_API_BASE") {
            provider = provider.with_api_base(base);
        }
        Ok(provider)
    }

    /// Override the API base URL (builder style). Defaults to
    /// [`DEFAULT_API_BASE`].
    pub fn with_api_base(mut self, api_base: impl Into<String>) -> Self {
        self.api_base = api_base.into();
        self
    }

    /// Scope memories to an application id (builder style). Sent as
    /// `metadata.application_id` on writes; never used to filter reads
    /// (matches Python — see module docs).
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

    /// Scope memories to a thread id, sent to Mem0 as `run_id` (builder
    /// style).
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// When `true`, the thread id used for scoping (`run_id`) is captured
    /// from the first [`ContextProvider::thread_created`] call instead of
    /// the static `thread_id` above, and a conflicting thread id on a later
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

    async fn effective_run_id(&self) -> Option<String> {
        if self.scope_to_per_operation_thread_id {
            self.per_operation_thread_id.lock().await.clone()
        } else {
            self.thread_id.clone()
        }
    }

    async fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{path}", self.api_base.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .header("Authorization", format!("Token {}", self.api_key))
            .header("Content-Type", "application/json")
            .header("Accept", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request to Mem0 API failed: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::service(format!("failed reading Mem0 API response body: {e}")))?;

        if !status.is_success() {
            return Err(map_http_error(status, &text));
        }
        serde_json::from_str(&text).map_err(|e| {
            Error::service(format!(
                "invalid Mem0 API response JSON: {e} (body: {text})"
            ))
        })
    }
}

#[async_trait]
impl ContextProvider for Mem0Provider {
    async fn thread_created(&self, thread_id: Option<&str>) -> Result<()> {
        let mut guard = self.per_operation_thread_id.lock().await;
        if self.scope_to_per_operation_thread_id {
            if let (Some(new_id), Some(existing)) = (thread_id, guard.as_deref()) {
                if new_id != existing {
                    return Err(Error::other(
                        "Mem0Provider can only be used with one thread at a time when scope_to_per_operation_thread_id is True.",
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
        _error: Option<&Error>,
    ) -> Result<()> {
        self.validate_filters()?;
        let all: Vec<&ChatMessage> = request_messages
            .iter()
            .chain(response_messages.iter())
            .collect();
        let run_id = self.effective_run_id().await;
        let Some(body) = build_add_body(
            &all,
            self.user_id.as_deref(),
            self.agent_id.as_deref(),
            run_id.as_deref(),
            self.application_id.as_deref(),
        ) else {
            return Ok(());
        };
        self.post(ADD_PATH, &body).await?;
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
        // Mirrors Python: "Validate input text is not empty before
        // searching (possible for function approval responses)".
        if input_text.trim().is_empty() {
            return Ok(Context::default());
        }

        let run_id = self.effective_run_id().await;
        let body = build_search_body(
            &input_text,
            self.user_id.as_deref(),
            self.agent_id.as_deref(),
            run_id.as_deref(),
        );
        let value = self.post(SEARCH_PATH, &body).await?;
        let memory_texts = parse_search_response(&value);
        Ok(format_context(&self.context_prompt, &memory_texts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{FunctionApprovalRequestContent, FunctionCallContent};

    fn msg(role: Role, text: &str) -> ChatMessage {
        ChatMessage::new(role, text)
    }

    // region: build_add_body

    #[test]
    fn build_add_body_single_message() {
        let m = msg(Role::user(), "Hello!");
        let body = build_add_body(&[&m], Some("user123"), None, None, None).unwrap();
        assert_eq!(
            body["messages"],
            serde_json::json!([{"role": "user", "content": "Hello!"}])
        );
        assert_eq!(body["user_id"], serde_json::json!("user123"));
        assert_eq!(body["agent_id"], Value::Null);
        assert_eq!(body["run_id"], Value::Null);
    }

    #[test]
    fn build_add_body_multiple_messages_preserve_order_and_roles() {
        let a = msg(Role::user(), "Hello, how are you?");
        let b = msg(Role::assistant(), "I'm doing well, thank you!");
        let c = msg(Role::system(), "You are a helpful assistant");
        let body = build_add_body(&[&a, &b, &c], Some("user123"), None, None, None).unwrap();
        assert_eq!(
            body["messages"],
            serde_json::json!([
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
                {"role": "system", "content": "You are a helpful assistant"},
            ])
        );
    }

    #[test]
    fn build_add_body_with_agent_id_leaves_user_id_null() {
        let m = msg(Role::user(), "hi");
        let body = build_add_body(&[&m], None, Some("agent123"), None, None).unwrap();
        assert_eq!(body["agent_id"], serde_json::json!("agent123"));
        assert_eq!(body["user_id"], Value::Null);
    }

    #[test]
    fn build_add_body_always_includes_application_id_metadata() {
        let m = msg(Role::user(), "hi");
        let body = build_add_body(&[&m], Some("u"), None, None, Some("app123")).unwrap();
        assert_eq!(
            body["metadata"],
            serde_json::json!({"application_id": "app123"})
        );

        // Even when application_id is unset, the metadata key is present
        // with a null value, matching Python's unconditional
        // `metadata={"application_id": self.application_id}`.
        let body2 = build_add_body(&[&m], Some("u"), None, None, None).unwrap();
        assert_eq!(
            body2["metadata"],
            serde_json::json!({"application_id": null})
        );
    }

    #[test]
    fn build_add_body_with_scoped_run_id() {
        let m = msg(Role::user(), "hi");
        let body = build_add_body(&[&m], Some("u"), None, Some("operation_thread"), None).unwrap();
        assert_eq!(body["run_id"], serde_json::json!("operation_thread"));
    }

    #[test]
    fn build_add_body_filters_blank_and_whitespace_messages() {
        let a = msg(Role::user(), "");
        let b = msg(Role::user(), "   ");
        let c = msg(Role::user(), "Valid message");
        let body = build_add_body(&[&a, &b, &c], Some("u"), None, None, None).unwrap();
        assert_eq!(
            body["messages"],
            serde_json::json!([{"role": "user", "content": "Valid message"}])
        );
    }

    #[test]
    fn build_add_body_filters_disallowed_roles() {
        let tool_msg = msg(Role::tool(), "tool output");
        let blank = msg(Role::user(), "   ");
        assert!(build_add_body(&[&tool_msg, &blank], Some("u"), None, None, None).is_none());
    }

    #[test]
    fn build_add_body_none_when_no_valid_messages() {
        let a = msg(Role::user(), "");
        let b = msg(Role::user(), "   ");
        assert!(build_add_body(&[&a, &b], Some("u"), None, None, None).is_none());
    }

    #[test]
    fn build_add_body_ignores_non_text_content_messages() {
        let m = ChatMessage::with_contents(
            Role::user(),
            vec![
                agent_framework_core::types::Content::FunctionApprovalRequest(
                    FunctionApprovalRequestContent {
                        id: "approval_1".to_string(),
                        function_call: FunctionCallContent::new("1", "test_func", None),
                    },
                ),
            ],
        );
        assert!(build_add_body(&[&m], Some("u"), None, None, None).is_none());
    }

    // endregion

    // region: build_search_body

    #[test]
    fn build_search_body_includes_query() {
        let body = build_search_body("What's the weather?", Some("user123"), None, None);
        assert_eq!(body["query"], serde_json::json!("What's the weather?"));
    }

    #[test]
    fn build_search_body_filters_include_only_set_scope_fields() {
        let body = build_search_body("q", Some("user123"), None, None);
        assert_eq!(body["filters"], serde_json::json!({"user_id": "user123"}));

        let body = build_search_body("q", None, Some("agent123"), None);
        assert_eq!(body["filters"], serde_json::json!({"agent_id": "agent123"}));

        let body = build_search_body("q", None, None, Some("operation_thread"));
        assert_eq!(
            body["filters"],
            serde_json::json!({"run_id": "operation_thread"})
        );
    }

    #[test]
    fn build_search_body_filters_combine_all_scope_fields() {
        let body = build_search_body("q", Some("u"), Some("a"), Some("r"));
        assert_eq!(
            body["filters"],
            serde_json::json!({"user_id": "u", "agent_id": "a", "run_id": "r"})
        );
    }

    #[test]
    fn build_search_body_filters_empty_when_no_scope() {
        let body = build_search_body("q", None, None, None);
        assert_eq!(body["filters"], serde_json::json!({}));
    }

    #[test]
    fn build_search_body_never_includes_application_id() {
        // application_id is not a parameter of build_search_body at all —
        // this is a structural assertion that the function signature keeps
        // it out, matching Python's invoking() never forwarding it.
        let body = build_search_body("q", Some("u"), Some("a"), Some("r"));
        assert!(body["filters"].get("application_id").is_none());
    }

    // endregion

    // region: parse_search_response

    #[test]
    fn parse_search_response_bare_array() {
        let value = serde_json::json!([
            {"memory": "User likes outdoor activities"},
            {"memory": "User lives in Seattle"},
        ]);
        assert_eq!(
            parse_search_response(&value),
            vec!["User likes outdoor activities", "User lives in Seattle"]
        );
    }

    #[test]
    fn parse_search_response_results_wrapper() {
        let value = serde_json::json!({"results": [{"memory": "Previous conversation context"}]});
        assert_eq!(
            parse_search_response(&value),
            vec!["Previous conversation context"]
        );
    }

    #[test]
    fn parse_search_response_empty_array() {
        let value = serde_json::json!([]);
        assert!(parse_search_response(&value).is_empty());
    }

    #[test]
    fn parse_search_response_missing_memory_field_yields_empty_string() {
        let value = serde_json::json!([{"score": 0.9}]);
        assert_eq!(parse_search_response(&value), vec![""]);
    }

    #[test]
    fn parse_search_response_fallback_wraps_unrecognized_object() {
        let value = serde_json::json!({"memory": "single object, no results wrapper"});
        assert_eq!(
            parse_search_response(&value),
            vec!["single object, no results wrapper"]
        );
    }

    // endregion

    // region: format_context

    #[test]
    fn format_context_single_memory() {
        let ctx = format_context(
            DEFAULT_CONTEXT_PROMPT,
            &[
                "User likes outdoor activities".to_string(),
                "User lives in Seattle".to_string(),
            ],
        );
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(ctx.messages[0].role, Role::user());
        assert_eq!(
            ctx.messages[0].text(),
            "## Memories\nConsider the following memories when answering user questions:\nUser likes outdoor activities\nUser lives in Seattle"
        );
    }

    #[test]
    fn format_context_empty_when_no_memories() {
        let ctx = format_context(DEFAULT_CONTEXT_PROMPT, &[]);
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn format_context_custom_prompt() {
        let custom = "## Custom Context\nRemember these details:";
        let ctx = format_context(custom, &["Test memory".to_string()]);
        assert_eq!(
            ctx.messages[0].text(),
            "## Custom Context\nRemember these details:\nTest memory"
        );
    }

    // endregion

    // region: map_http_error

    #[test]
    fn map_http_error_includes_status_and_body() {
        let err = map_http_error(reqwest::StatusCode::INTERNAL_SERVER_ERROR, "boom");
        let msg = err.to_string();
        assert!(msg.contains("500"));
        assert!(msg.contains("boom"));
    }

    #[test]
    fn map_http_error_includes_unauthorized() {
        let err = map_http_error(reqwest::StatusCode::UNAUTHORIZED, "invalid api key");
        let msg = err.to_string();
        assert!(msg.contains("401"));
        assert!(msg.contains("invalid api key"));
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

    // region: validate_filters / builders

    #[test]
    fn init_without_filters_succeeds_validation_happens_at_call_time() {
        let provider = Mem0Provider::new("key");
        assert!(provider.user_id.is_none());
        assert!(provider.validate_filters().is_err());
    }

    #[test]
    fn validate_filters_accepts_any_single_scope_field() {
        assert!(Mem0Provider::new("k")
            .with_user_id("u")
            .validate_filters()
            .is_ok());
        assert!(Mem0Provider::new("k")
            .with_agent_id("a")
            .validate_filters()
            .is_ok());
        assert!(Mem0Provider::new("k")
            .with_application_id("ap")
            .validate_filters()
            .is_ok());
        assert!(Mem0Provider::new("k")
            .with_thread_id("t")
            .validate_filters()
            .is_ok());
    }

    #[test]
    fn builders_set_expected_fields() {
        let p = Mem0Provider::new("key")
            .with_user_id("user123")
            .with_agent_id("agent123")
            .with_application_id("app123")
            .with_thread_id("thread123")
            .with_context_prompt("custom prompt");
        assert_eq!(p.user_id.as_deref(), Some("user123"));
        assert_eq!(p.agent_id.as_deref(), Some("agent123"));
        assert_eq!(p.application_id.as_deref(), Some("app123"));
        assert_eq!(p.thread_id.as_deref(), Some("thread123"));
        assert_eq!(p.context_prompt, "custom prompt");
        assert_eq!(p.api_base, DEFAULT_API_BASE);
    }

    #[test]
    fn debug_impl_does_not_leak_api_key() {
        let p = Mem0Provider::new("super-secret-key").with_user_id("u1");
        let debug_str = format!("{p:?}");
        assert!(!debug_str.contains("super-secret-key"));
    }

    // endregion

    // region: from_env

    /// Guards `MEM0_API_KEY` / `MEM0_API_BASE` mutation: tests within a
    /// crate run on multiple threads, and env vars are process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_errors_when_api_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::remove_var("MEM0_API_KEY");
            std::env::remove_var("MEM0_API_BASE");
        }
        let result = Mem0Provider::from_env();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("MEM0_API_KEY"));
    }

    #[test]
    fn from_env_reads_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var("MEM0_API_KEY", "mem0-test-key");
            std::env::set_var("MEM0_API_BASE", "https://example.test");
        }
        let provider = Mem0Provider::from_env().unwrap();
        assert_eq!(provider.api_key, "mem0-test-key");
        assert_eq!(provider.api_base, "https://example.test");
        unsafe {
            std::env::remove_var("MEM0_API_KEY");
            std::env::remove_var("MEM0_API_BASE");
        }
    }

    #[test]
    fn from_env_defaults_api_base_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        unsafe {
            std::env::set_var("MEM0_API_KEY", "mem0-test-key");
            std::env::remove_var("MEM0_API_BASE");
        }
        let provider = Mem0Provider::from_env().unwrap();
        assert_eq!(provider.api_base, DEFAULT_API_BASE);
        unsafe {
            std::env::remove_var("MEM0_API_KEY");
        }
    }

    // endregion

    // region: thread_created / per-operation scoping (async, no network: pure Mutex state)

    #[tokio::test]
    async fn thread_created_sets_per_operation_thread_id() {
        let p = Mem0Provider::new("k").with_user_id("u1");
        p.thread_created(Some("thread123")).await.unwrap();
        assert_eq!(
            p.per_operation_thread_id.lock().await.as_deref(),
            Some("thread123")
        );
    }

    #[tokio::test]
    async fn thread_created_does_not_overwrite_existing() {
        let p = Mem0Provider::new("k").with_user_id("u1");
        p.thread_created(Some("thread123")).await.unwrap();
        p.thread_created(Some("other")).await.unwrap();
        assert_eq!(
            p.per_operation_thread_id.lock().await.as_deref(),
            Some("thread123")
        );
    }

    #[tokio::test]
    async fn thread_created_conflict_when_scoped() {
        let p = Mem0Provider::new("k")
            .with_user_id("u1")
            .with_scope_to_per_operation_thread_id(true);
        p.thread_created(Some("thread123")).await.unwrap();
        let err = p
            .thread_created(Some("different_thread"))
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("can only be used with one thread at a time"));
    }

    #[tokio::test]
    async fn thread_created_allows_same_id_and_none_when_scoped() {
        let p = Mem0Provider::new("k")
            .with_user_id("u1")
            .with_scope_to_per_operation_thread_id(true);
        p.thread_created(Some("thread123")).await.unwrap();
        p.thread_created(Some("thread123")).await.unwrap();
        p.thread_created(None).await.unwrap();
    }

    // endregion

    // region: invoking()/invoked() input validation (async, no network: fails before any I/O)

    #[tokio::test]
    async fn invoking_fails_without_filters() {
        let p = Mem0Provider::new("k");
        let err = p.invoking(&[ChatMessage::user("Hi")]).await.unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    #[tokio::test]
    async fn invoked_fails_without_filters() {
        let p = Mem0Provider::new("k");
        let err = p
            .invoked(&[ChatMessage::user("Hi")], &[], None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("At least one of the filters"));
    }

    #[tokio::test]
    async fn invoking_returns_empty_context_for_blank_input_without_network_call() {
        // user_id is set (passes validate_filters), but the only message has
        // no text content, so invoking() must short-circuit before ever
        // building a request — if it didn't, this test would hang/error
        // trying to reach api.mem0.ai.
        let p = Mem0Provider::new("k").with_user_id("u1");
        let m = ChatMessage::with_contents(
            Role::user(),
            vec![
                agent_framework_core::types::Content::FunctionApprovalRequest(
                    FunctionApprovalRequestContent {
                        id: "approval_1".to_string(),
                        function_call: FunctionCallContent::new("1", "test_func", None),
                    },
                ),
            ],
        );
        let ctx = p.invoking(&[m]).await.unwrap();
        assert!(ctx.messages.is_empty());
    }

    // endregion
}
