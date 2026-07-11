//! [`OpenAIAssistantsClient`]: a [`ChatClient`] for the OpenAI Assistants API
//! (the `beta` `/assistants`, `/threads`, and `/threads/{id}/runs` surface).
//!
//! This is a faithful port of the Python
//! `agent_framework.openai._assistants_client.OpenAIAssistantsClient`. Unlike
//! Chat Completions / Responses, the Assistants API is *thread*-based: messages
//! live on a server-side thread, and a *run* against an *assistant* streams the
//! reply back over SSE. This client maps the framework's stateless
//! [`ChatClient`] surface onto that model:
//!
//! * **Threads ↔ `conversation_id`.** [`ChatOptions::conversation_id`] is the
//!   Assistants `thread_id`. With no id, a fresh thread is created per call
//!   (seeded with the request's new messages as `additional_messages`), and the
//!   returned [`ChatResponse::conversation_id`] carries the new thread id so a
//!   caller (or [`ChatAgent`]) can continue the conversation.
//! * **Assistants lifecycle.** A client either targets an existing assistant
//!   ([`OpenAIAssistantsClient::with_assistant_id`], never deleted) or lazily
//!   creates a *transient* assistant on first use (from the client `model` plus
//!   optional name/description). Because Rust has no async `Drop`, the transient
//!   assistant is **not** cleaned up automatically: call
//!   [`OpenAIAssistantsClient::close`] (or its alias
//!   [`OpenAIAssistantsClient::delete_assistant`]) to delete it. This mirrors
//!   Python's `should_delete_assistant` bookkeeping, but where Python can clean
//!   up in `__aexit__`, here the caller must invoke `close()` explicitly. A
//!   configured (`with_assistant_id`) assistant is never deleted by `close()`.
//! * **Function tools round-trip a composite `call_id`.** A function tool call
//!   surfaced by a `requires_action` run event is given
//!   `call_id = json!([run_id, tool_call_id])` (exactly as Python,
//!   `_assistants_client.py:374-385`); the matching [`FunctionResultContent`]
//!   is expected to carry that same string, from which the run id and tool-call
//!   id are recovered to `submit_tool_outputs` against the still-active run
//!   (`_assistants_client.py:490-524`). Because the framework's tool loop does
//!   not resend server-side thread state, submitting tool results requires a
//!   known `thread_id`: with none, this client raises the same error Python
//!   does (`_assistants_client.py:201-202`).
//!
//! Every request carries the `OpenAI-Beta: assistants=v2` header the Assistants
//! v2 API requires. The base URL is overridable for Azure/OpenAI-compatible
//! servers, and error/`Retry-After` handling matches [`crate::OpenAIClient`].
//!
//! ```no_run
//! use agent_framework_openai::assistants::OpenAIAssistantsClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIAssistantsClient::new("sk-...", "gpt-4o-mini");
//! let agent = ChatAgent::builder(client.clone())
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! // Delete the transient assistant this client created on first use.
//! client.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! [`ChatOptions::conversation_id`]: agent_framework_core::types::ChatOptions::conversation_id
//! [`ChatResponse::conversation_id`]: agent_framework_core::types::ChatResponse::conversation_id
//! [`ChatAgent`]: agent_framework_core::agent::ChatAgent

use std::collections::VecDeque;
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::tools::ToolKind;
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content, FunctionArguments,
    FunctionCallContent, FunctionResultContent, Role, TextContent, ToolMode, UsageContent,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use tokio::sync::Mutex;

use crate::convert::{parse_usage, result_to_string, top_level_media_type};
use crate::{ByteStream, DEFAULT_BASE_URL};

/// The beta header every Assistants v2 endpoint requires.
const ASSISTANTS_BETA_HEADER: &str = "OpenAI-Beta";
const ASSISTANTS_BETA_VALUE: &str = "assistants=v2";

/// An OpenAI Assistants API chat client.
///
/// See the [module docs](crate::assistants) for the thread/assistant model and
/// the transient-assistant cleanup contract ([`OpenAIAssistantsClient::close`]).
#[derive(Clone)]
pub struct OpenAIAssistantsClient {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
    /// Name applied when lazily creating a transient assistant.
    assistant_name: Option<String>,
    /// Description applied when lazily creating a transient assistant.
    assistant_description: Option<String>,
    /// Default thread id, used when a request supplies no `conversation_id`.
    thread_id: Option<String>,
    /// A caller-supplied assistant id: reused for every run and **never**
    /// deleted by [`OpenAIAssistantsClient::close`] (mirrors Python's
    /// `assistant_id` constructor argument).
    configured_assistant_id: Option<String>,
    /// Bookkeeping for a *transient* assistant created on first use. Shared
    /// across clones so the id is created once and any clone can `close()` it.
    created: Arc<Mutex<CreatedAssistant>>,
}

/// Tracks the lazily-created transient assistant and whether `close()` should
/// delete it — the Rust equivalent of Python's `assistant_id` +
/// `_should_delete_assistant` pair (`_assistants_client.py:150-154, 164-170`).
#[derive(Default)]
struct CreatedAssistant {
    id: Option<String>,
    should_delete: bool,
}

impl std::fmt::Debug for OpenAIAssistantsClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIAssistantsClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("organization", &self.organization)
            .field("configured_assistant_id", &self.configured_assistant_id)
            .field("thread_id", &self.thread_id)
            .finish_non_exhaustive()
    }
}

impl OpenAIAssistantsClient {
    /// Create a client for the given API key and default model.
    ///
    /// With no [`with_assistant_id`](Self::with_assistant_id), the `model` is
    /// used to lazily create a transient assistant on first use; remember to
    /// [`close`](Self::close) it afterwards.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            organization: None,
            assistant_name: None,
            assistant_description: None,
            thread_id: None,
            configured_assistant_id: None,
            created: Arc::new(Mutex::new(CreatedAssistant::default())),
        }
    }

    /// Build a client from the `OPENAI_API_KEY` (and optional
    /// `OPENAI_BASE_URL`) environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("OPENAI_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for Azure OpenAI or compatible servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Target an existing assistant. This id is reused for every run and is
    /// **never** deleted by [`close`](Self::close).
    pub fn with_assistant_id(mut self, assistant_id: impl Into<String>) -> Self {
        self.configured_assistant_id = Some(assistant_id.into());
        self
    }

    /// The name to apply when lazily creating a transient assistant.
    pub fn with_assistant_name(mut self, name: impl Into<String>) -> Self {
        self.assistant_name = Some(name.into());
        self
    }

    /// The description to apply when lazily creating a transient assistant.
    pub fn with_assistant_description(mut self, description: impl Into<String>) -> Self {
        self.assistant_description = Some(description.into());
        self
    }

    /// A default thread id, used when a request supplies no `conversation_id`.
    pub fn with_thread_id(mut self, thread_id: impl Into<String>) -> Self {
        self.thread_id = Some(thread_id.into());
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// The caller-configured assistant id, if any. A *transient* assistant
    /// created on first use is not reported here (it is internal cleanup
    /// state); use [`close`](Self::close) to delete it.
    pub fn assistant_id(&self) -> Option<&str> {
        self.configured_assistant_id.as_deref()
    }

    // region: HTTP plumbing

    /// A request builder pre-loaded with auth, the assistants-v2 beta header,
    /// and the optional organization header.
    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        );
        let mut req = self
            .http
            .request(method, url)
            .bearer_auth(&self.api_key)
            .header(ASSISTANTS_BETA_HEADER, ASSISTANTS_BETA_VALUE);
        if let Some(org) = &self.organization {
            req = req.header("OpenAI-Organization", org);
        }
        req
    }

    /// Send a request, classifying a non-success status via
    /// [`crate::classify_service_error`] (carrying any `Retry-After` on the
    /// [`Error::ServiceStatus`] fallback), matching [`crate::OpenAIClient`].
    async fn send(&self, req: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = crate::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(crate::classify_service_error(
                status.as_u16(),
                &text,
                format!("OpenAI API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }

    async fn post_json(&self, path: &str, body: &Value) -> Result<Value> {
        let resp = self
            .send(self.request(reqwest::Method::POST, path).json(body))
            .await?;
        resp.json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))
    }

    // endregion

    // region: assistant lifecycle

    /// Create a transient assistant from the client model + name/description and
    /// return its id (`_assistants_client.py:221-231`).
    async fn create_assistant(&self) -> Result<String> {
        let mut body = Map::new();
        body.insert("model".into(), json!(self.model));
        if let Some(name) = &self.assistant_name {
            body.insert("name".into(), json!(name));
        }
        if let Some(desc) = &self.assistant_description {
            body.insert("description".into(), json!(desc));
        }
        let value = self.post_json("assistants", &Value::Object(body)).await?;
        value
            .get("id")
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| Error::service("assistant creation response missing 'id'"))
    }

    /// Resolve the assistant to run against, creating a transient one on first
    /// use (`_assistants_client.py:214-234`). The transient creation is
    /// serialized so concurrent first-calls create exactly one assistant.
    async fn ensure_assistant(&self) -> Result<String> {
        if let Some(id) = &self.configured_assistant_id {
            return Ok(id.clone());
        }
        let mut guard = self.created.lock().await;
        if let Some(id) = &guard.id {
            return Ok(id.clone());
        }
        let id = self.create_assistant().await?;
        guard.id = Some(id.clone());
        guard.should_delete = true;
        Ok(id)
    }

    /// Delete the transient assistant this client created, if any.
    ///
    /// The Rust stand-in for Python's `close()` / `__aexit__`
    /// (`_assistants_client.py:160-170`): since there is no async `Drop`, the
    /// caller must invoke this explicitly. A configured
    /// ([`with_assistant_id`](Self::with_assistant_id)) assistant is never
    /// deleted. Idempotent — a second call is a no-op. On a delete failure the
    /// bookkeeping is left intact so the call can be retried.
    pub async fn close(&self) -> Result<()> {
        let mut guard = self.created.lock().await;
        if guard.should_delete {
            if let Some(id) = guard.id.clone() {
                self.send(self.request(reqwest::Method::DELETE, &format!("assistants/{id}")))
                    .await?;
            }
            guard.id = None;
            guard.should_delete = false;
        }
        Ok(())
    }

    /// Alias for [`close`](Self::close), matching the Python method name used in
    /// the task surface.
    pub async fn delete_assistant(&self) -> Result<()> {
        self.close().await
    }

    // endregion

    // region: threads & runs

    /// The most recent still-active run on a thread, if any
    /// (`_assistants_client.py:271-280`).
    async fn get_active_thread_run(&self, thread_id: &str) -> Result<Option<ActiveRun>> {
        let resp = self
            .send(self.request(
                reqwest::Method::GET,
                &format!("threads/{thread_id}/runs?limit=1&order=desc"),
            ))
            .await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        let Some(run) = value
            .get("data")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
        else {
            return Ok(None);
        };
        let status = run.get("status").and_then(Value::as_str).unwrap_or("");
        if matches!(status, "completed" | "cancelled" | "failed" | "expired") {
            return Ok(None);
        }
        Ok(Some(ActiveRun {
            run_id: run
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            thread_id: run
                .get("thread_id")
                .and_then(Value::as_str)
                .unwrap_or(thread_id)
                .to_string(),
        }))
    }

    /// Create a thread seeded with the run's initial messages / tool resources /
    /// metadata, returning its id (`_assistants_client.py:285-294`).
    async fn create_thread(
        &self,
        messages: Option<Value>,
        tool_resources: Option<Value>,
        metadata: Option<Value>,
    ) -> Result<String> {
        let body = build_thread_body(messages, tool_resources, metadata);
        let value = self.post_json("threads", &body).await?;
        value
            .get("id")
            .and_then(Value::as_str)
            .map(String::from)
            .ok_or_else(|| Error::service("thread creation response missing 'id'"))
    }

    /// Prepare the thread for a new run: create one when absent (pulling the
    /// thread-scoped fields out of `run_options`), or cancel an active run on an
    /// existing thread before starting a new one
    /// (`_assistants_client.py:282-300`).
    async fn prepare_thread(
        &self,
        thread_id: Option<String>,
        active_run: Option<&ActiveRun>,
        run_options: &mut Map<String, Value>,
    ) -> Result<String> {
        match thread_id {
            None => {
                let (messages, tool_resources, metadata) = take_thread_fields(run_options);
                self.create_thread(messages, tool_resources, metadata).await
            }
            Some(tid) => {
                if let Some(active) = active_run {
                    self.send(self.request(
                        reqwest::Method::POST,
                        &format!("threads/{tid}/runs/{}/cancel", active.run_id),
                    ))
                    .await?;
                }
                Ok(tid)
            }
        }
    }

    /// Open the SSE stream for a run — either submitting tool outputs to the
    /// active run, or creating a new run on the (possibly newly created) thread
    /// (`_assistants_client.py:236-269`). Returns the raw SSE response plus the
    /// final thread id it ran against.
    async fn create_assistant_stream(
        &self,
        thread_id: Option<String>,
        assistant_id: &str,
        mut run_options: Map<String, Value>,
        tool_results: Option<Vec<FunctionResultContent>>,
    ) -> Result<(reqwest::Response, String)> {
        let active_run = match &thread_id {
            Some(tid) => self.get_active_thread_run(tid).await?,
            None => None,
        };
        let (tool_run_id, tool_outputs) =
            convert_function_results_to_tool_output(tool_results.as_deref());

        // If there's an active run and we hold tool outputs for exactly it,
        // submit them and continue that run's stream.
        if let (Some(active), Some(run_id), Some(outputs)) =
            (&active_run, &tool_run_id, &tool_outputs)
        {
            if run_id == &active.run_id && !outputs.is_empty() {
                let body = json!({ "tool_outputs": outputs, "stream": true });
                let resp = self
                    .send(
                        self.request(
                            reqwest::Method::POST,
                            &format!(
                                "threads/{}/runs/{}/submit_tool_outputs",
                                active.thread_id, run_id
                            ),
                        )
                        .json(&body),
                    )
                    .await?;
                return Ok((resp, active.thread_id.clone()));
            }
        }

        // Otherwise create (or cancel + reuse) the thread, then start a new run.
        let final_thread_id = self
            .prepare_thread(thread_id, active_run.as_ref(), &mut run_options)
            .await?;
        let body = build_run_body(run_options, assistant_id);
        let resp = self
            .send(
                self.request(
                    reqwest::Method::POST,
                    &format!("threads/{final_thread_id}/runs"),
                )
                .json(&body),
            )
            .await?;
        Ok((resp, final_thread_id))
    }

    // endregion
}

/// A thread's active run: its id plus the thread it belongs to.
struct ActiveRun {
    run_id: String,
    thread_id: String,
}

#[async_trait::async_trait]
impl ChatClient for OpenAIAssistantsClient {
    /// Non-streaming: aggregate the streamed run, mirroring Python's
    /// `_inner_get_response`, which builds the response from the streaming
    /// generator (`_assistants_client.py:172-182`).
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let response_format = options.response_format.clone();
        let mut stream = self.get_streaming_response(messages, options).await?;
        let mut updates = Vec::new();
        while let Some(item) = stream.next().await {
            updates.push(item?);
        }
        Ok(ChatResponse::from_updates_with_format(
            updates,
            response_format.as_ref(),
        ))
    }

    /// Streaming: resolve the thread + assistant, open the run's SSE stream, and
    /// map its events to updates (`_assistants_client.py:184-212`).
    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let (run_options, tool_results) = prepare_options(&messages, &options);

        let thread_id = options
            .conversation_id
            .clone()
            .or_else(|| self.thread_id.clone());

        // Tool results can only be submitted against a known thread's active
        // run (`_assistants_client.py:201-202`).
        if thread_id.is_none() && tool_results.is_some() {
            return Err(Error::service(
                "No thread ID was provided, but chat messages includes tool results.",
            ));
        }

        let assistant_id = self.ensure_assistant().await?;
        let (resp, final_thread_id) = self
            .create_assistant_stream(thread_id, &assistant_id, run_options, tool_results)
            .await?;
        let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
        Ok(assistants_sse_stream(byte_stream, final_thread_id).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.model)
    }
}

// region: request conversion

/// Build the `run_options` map and any tool results from a request, mirroring
/// Python's `_prepare_options` (`_assistants_client.py:387-488`).
///
/// System/developer message text becomes the run `instructions` (the Assistants
/// API has no such message roles); every other message with renderable content
/// becomes an `additional_messages` entry (role coerced to `user`/`assistant`);
/// [`FunctionResultContent`] items are split out as tool results to submit.
///
/// Divergences from the Python assistants client, all additive and using the
/// established hosted-tool parameter keys: a hosted file-search tool's
/// `vector_store_ids` and a code-interpreter tool's `file_ids` are folded into
/// `tool_resources` (the API's required location for them), and the typed
/// [`ChatOptions::metadata`] field is forwarded (Python only forwards metadata
/// passed via kwargs). Scalar options are omitted when `None` rather than sent
/// as JSON `null`, matching this crate's other clients.
fn prepare_options(
    messages: &[ChatMessage],
    options: &ChatOptions,
) -> (Map<String, Value>, Option<Vec<FunctionResultContent>>) {
    let mut run_options = Map::new();

    if let Some(model) = &options.model_id {
        run_options.insert("model".into(), json!(model));
    }
    if let Some(t) = options.temperature {
        run_options.insert("temperature".into(), json!(t));
    }
    if let Some(p) = options.top_p {
        run_options.insert("top_p".into(), json!(p));
    }
    if let Some(mt) = options.max_tokens {
        run_options.insert("max_completion_tokens".into(), json!(mt));
    }
    if let Some(allow) = options.allow_multiple_tool_calls {
        run_options.insert("parallel_tool_calls".into(), json!(allow));
    }

    // Tools + tool_choice are only emitted when a tool_choice is set, and tool
    // definitions are dropped entirely for `tool_choice = "none"` — matching the
    // Python gate (`_assistants_client.py:404-435`).
    if let Some(tool_choice) = &options.tool_choice {
        let mut tool_defs: Vec<Value> = Vec::new();
        let mut tool_resources = Map::new();
        if *tool_choice != ToolMode::None {
            for tool in &options.tools {
                match &tool.kind {
                    ToolKind::Function => tool_defs.push(tool.to_openai_spec()),
                    ToolKind::HostedCodeInterpreter => {
                        tool_defs.push(json!({ "type": "code_interpreter" }));
                        if let Some(file_ids) = tool.parameters.get("file_ids") {
                            fold_tool_resource(
                                &mut tool_resources,
                                "code_interpreter",
                                "file_ids",
                                file_ids.clone(),
                            );
                        }
                    }
                    ToolKind::HostedFileSearch { max_results } => {
                        let mut spec = Map::new();
                        spec.insert("type".into(), json!("file_search"));
                        if let Some(n) = max_results {
                            spec.insert("max_num_results".into(), json!(n));
                        }
                        tool_defs.push(Value::Object(spec));
                        if let Some(ids) = tool.parameters.get("vector_store_ids") {
                            fold_tool_resource(
                                &mut tool_resources,
                                "file_search",
                                "vector_store_ids",
                                ids.clone(),
                            );
                        }
                    }
                    // Other hosted kinds have no Assistants mapping (Python skips
                    // them too — only AIFunction / code_interpreter / file_search
                    // are handled).
                    _ => {}
                }
            }
        }
        if !tool_defs.is_empty() {
            run_options.insert("tools".into(), json!(tool_defs));
        }
        if !tool_resources.is_empty() {
            run_options.insert("tool_resources".into(), Value::Object(tool_resources));
        }
        match tool_choice {
            ToolMode::None => {
                run_options.insert("tool_choice".into(), json!("none"));
            }
            ToolMode::Auto => {
                run_options.insert("tool_choice".into(), json!("auto"));
            }
            ToolMode::Required(Some(name)) => {
                run_options.insert(
                    "tool_choice".into(),
                    json!({ "type": "function", "function": { "name": name } }),
                );
            }
            // A bare "required" (no function name) is intentionally not sent,
            // mirroring Python's `elif ... required_function_name is not None`.
            ToolMode::Required(None) => {}
        }
    }

    if let Some(fmt) = &options.response_format {
        // `ResponseFormat` serializes to the same `{type, json_schema?}` object
        // the Assistants `response_format` field accepts.
        run_options.insert("response_format".into(), json!(fmt));
    }

    if let Some(metadata) = &options.metadata {
        run_options.insert("metadata".into(), json!(metadata));
    }

    let mut instructions = String::new();
    let mut additional_messages: Vec<Value> = Vec::new();
    let mut tool_results: Option<Vec<FunctionResultContent>> = None;

    for msg in messages {
        let role = msg.role.as_str();
        if role == Role::SYSTEM || role == "developer" {
            for content in &msg.contents {
                if let Content::Text(t) = content {
                    instructions.push_str(&t.text);
                }
            }
            continue;
        }

        let mut message_contents: Vec<Value> = Vec::new();
        for content in &msg.contents {
            match content {
                Content::Text(t) => {
                    message_contents.push(json!({ "type": "text", "text": t.text }));
                }
                Content::Uri(u) if top_level_media_type(&u.media_type) == "image" => {
                    message_contents
                        .push(json!({ "type": "image_url", "image_url": { "url": u.uri } }));
                }
                Content::FunctionResult(fr) => {
                    tool_results.get_or_insert_with(Vec::new).push(fr.clone());
                }
                _ => {}
            }
        }

        if !message_contents.is_empty() {
            let msg_role = if msg.role == Role::assistant() {
                "assistant"
            } else {
                "user"
            };
            additional_messages.push(json!({ "role": msg_role, "content": message_contents }));
        }
    }

    if !additional_messages.is_empty() {
        run_options.insert("additional_messages".into(), json!(additional_messages));
    }
    if !instructions.is_empty() {
        run_options.insert("instructions".into(), json!(instructions));
    }

    // Passthrough of caller-supplied extras (the Rust analogue of Python's
    // `run_options = {**kwargs}`); typed/computed fields above win on conflict.
    for (k, v) in &options.additional_properties {
        run_options.entry(k.clone()).or_insert_with(|| v.clone());
    }

    (run_options, tool_results)
}

/// Insert `value` at `tool_resources[resource][key]`, creating the nested object
/// if needed.
fn fold_tool_resource(
    tool_resources: &mut Map<String, Value>,
    resource: &str,
    key: &str,
    value: Value,
) {
    let entry = tool_resources
        .entry(resource.to_string())
        .or_insert_with(|| json!({}));
    if let Some(obj) = entry.as_object_mut() {
        obj.insert(key.to_string(), value);
    }
}

/// Split the thread-scoped fields out of `run_options` for a new thread,
/// removing `additional_messages` and `tool_resources` from the run body while
/// leaving `metadata` in place (which then applies to both the thread and the
/// run) — mirroring Python's `_prepare_thread` mutation
/// (`_assistants_client.py:287-294`).
fn take_thread_fields(
    run_options: &mut Map<String, Value>,
) -> (Option<Value>, Option<Value>, Option<Value>) {
    let messages = run_options.remove("additional_messages");
    let tool_resources = run_options.remove("tool_resources");
    let metadata = run_options.get("metadata").cloned();
    (messages, tool_resources, metadata)
}

/// Assemble a thread-creation body, omitting an empty `messages` list.
fn build_thread_body(
    messages: Option<Value>,
    tool_resources: Option<Value>,
    metadata: Option<Value>,
) -> Value {
    let mut body = Map::new();
    if let Some(messages) = messages {
        if messages.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
            body.insert("messages".into(), messages);
        }
    }
    if let Some(tr) = tool_resources {
        body.insert("tool_resources".into(), tr);
    }
    if let Some(md) = metadata {
        body.insert("metadata".into(), md);
    }
    Value::Object(body)
}

/// Assemble a run-creation body: the prepared `run_options` plus the target
/// assistant and the streaming flag.
fn build_run_body(mut run_options: Map<String, Value>, assistant_id: &str) -> Value {
    run_options.insert("assistant_id".into(), json!(assistant_id));
    run_options.insert("stream".into(), json!(true));
    Value::Object(run_options)
}

/// Recover `(run_id, tool_outputs)` from function results, decoding each
/// composite `call_id = [run_id, tool_call_id]` and dropping any that are
/// malformed or belong to a different run — mirroring Python's
/// `_convert_function_results_to_tool_output` (`_assistants_client.py:490-524`).
fn convert_function_results_to_tool_output(
    tool_results: Option<&[FunctionResultContent]>,
) -> (Option<String>, Option<Vec<Value>>) {
    let Some(results) = tool_results else {
        return (None, None);
    };
    let mut run_id: Option<String> = None;
    let mut tool_outputs: Option<Vec<Value>> = None;

    for fr in results {
        // The call_id was encoded as a JSON `[run_id, tool_call_id]` pair.
        let Ok(ids) = serde_json::from_str::<Vec<String>>(&fr.call_id) else {
            continue;
        };
        if ids.len() != 2 || ids[0].is_empty() || ids[1].is_empty() {
            continue;
        }
        if let Some(existing) = &run_id {
            if existing != &ids[0] {
                continue;
            }
        }
        run_id = Some(ids[0].clone());
        let call_id = &ids[1];

        let rendered = result_to_string(fr);
        let output = if rendered.is_empty() {
            "No output received.".to_string()
        } else {
            rendered
        };
        tool_outputs
            .get_or_insert_with(Vec::new)
            .push(json!({ "tool_call_id": call_id, "output": output }));
    }

    (run_id, tool_outputs)
}

// endregion

// region: streaming

/// Turn an Assistants SSE response's byte stream into a stream of updates.
///
/// Assistants SSE frames carry the kind on an `event:` line (distinct from the
/// `data:` JSON), so the event name is tracked across lines and paired with the
/// following `data:` payload.
fn assistants_sse_stream(
    byte_stream: ByteStream,
    thread_id: String,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    futures::stream::unfold(
        AssistantsSseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            current_event: None,
            response_id: None,
            thread_id,
            done: false,
        },
        |mut state| async move {
            loop {
                if let Some(update) = state.queued.pop_front() {
                    return Some((Ok(update), state));
                }
                if state.done {
                    return None;
                }
                match state.byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        let decoded = state.utf8.push(&bytes);
                        state.buffer.push_str(&decoded);
                        while let Some(pos) = state.buffer.find('\n') {
                            let line = state.buffer[..pos].trim().to_string();
                            state.buffer.drain(..=pos);
                            if line.is_empty() {
                                // Event boundary: the SSE field set resets.
                                state.current_event = None;
                                continue;
                            }
                            if let Some(event) = line.strip_prefix("event:") {
                                state.current_event = Some(event.trim().to_string());
                                continue;
                            }
                            let Some(data) = line.strip_prefix("data:") else {
                                continue;
                            };
                            let data = data.trim();
                            if data.is_empty() {
                                continue;
                            }
                            let event = state.current_event.take().unwrap_or_default();
                            if event == "done" || data == "[DONE]" {
                                state.done = true;
                                continue;
                            }
                            let Ok(value) = serde_json::from_str::<Value>(data) else {
                                continue;
                            };
                            match process_assistants_event(
                                &event,
                                &value,
                                &mut state.response_id,
                                &state.thread_id,
                            ) {
                                EventOutcome::Updates(updates) => {
                                    state.queued.extend(updates);
                                }
                                EventOutcome::Error(e) => {
                                    state.done = true;
                                    return Some((Err(e), state));
                                }
                                EventOutcome::None => {}
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((Err(Error::service(format!("stream error: {e}"))), state));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// State carried across `unfold` iterations while parsing the Assistants SSE
/// stream.
struct AssistantsSseState {
    byte_stream: ByteStream,
    buffer: String,
    utf8: Utf8StreamDecoder,
    queued: VecDeque<ChatResponseUpdate>,
    /// The last-seen `event:` line, paired with the next `data:` payload.
    current_event: Option<String>,
    /// The active run id, learned from `thread.run.step.created` and used as
    /// each update's `message_id` and as the first half of an emitted call_id.
    response_id: Option<String>,
    thread_id: String,
    done: bool,
}

/// A per-event control-flow value. The `Updates` variant can be large, but it is
/// destructured immediately in the stream loop and never stored in bulk, so the
/// size skew is accepted rather than boxed (which would allocate per event).
#[allow(clippy::large_enum_variant)]
enum EventOutcome {
    Updates(Vec<ChatResponseUpdate>),
    Error(Error),
    None,
}

/// Map one Assistants SSE event to updates, an error, or nothing, mirroring the
/// event set Python's `_process_stream_events` handles
/// (`_assistants_client.py:302-372`).
///
/// Divergences from Python: unhandled/lifecycle events (`thread.run.created`,
/// `queued`, `in_progress`, `message.created`, step lifecycle, …) yield nothing
/// here rather than an empty content-free update — matching this crate's
/// [`crate::responses`] parser and avoiding a spurious empty leading message in
/// the aggregated response. And `thread.run.failed` / an `error` event surface
/// as an [`Error`] (as the underlying OpenAI SDK does) instead of being ignored.
fn process_assistants_event(
    event: &str,
    data: &Value,
    response_id: &mut Option<String>,
    thread_id: &str,
) -> EventOutcome {
    match event {
        "thread.run.step.created" => {
            if let Some(rid) = data.get("run_id").and_then(Value::as_str) {
                *response_id = Some(rid.to_string());
            }
            EventOutcome::None
        }
        "thread.message.delta" => {
            let delta = data.get("delta");
            let role = match delta.and_then(|d| d.get("role")).and_then(Value::as_str) {
                Some("user") => Role::user(),
                _ => Role::assistant(),
            };
            let mut updates = Vec::new();
            if let Some(blocks) = delta
                .and_then(|d| d.get("content"))
                .and_then(Value::as_array)
            {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) != Some("text") {
                        continue;
                    }
                    let text = block
                        .get("text")
                        .and_then(|t| t.get("value"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if text.is_empty() {
                        continue;
                    }
                    updates.push(ChatResponseUpdate {
                        role: Some(role.clone()),
                        contents: vec![Content::Text(TextContent::new(text))],
                        conversation_id: Some(thread_id.to_string()),
                        message_id: response_id.clone(),
                        response_id: response_id.clone(),
                        ..Default::default()
                    });
                }
            }
            if updates.is_empty() {
                EventOutcome::None
            } else {
                EventOutcome::Updates(updates)
            }
        }
        "thread.run.requires_action" => {
            let contents = create_function_call_contents(data, response_id);
            if contents.is_empty() {
                EventOutcome::None
            } else {
                EventOutcome::Updates(vec![ChatResponseUpdate {
                    role: Some(Role::assistant()),
                    contents,
                    conversation_id: Some(thread_id.to_string()),
                    message_id: response_id.clone(),
                    response_id: response_id.clone(),
                    ..Default::default()
                }])
            }
        }
        "thread.run.completed" => match data.get("usage").filter(|u| u.is_object()) {
            Some(usage) => EventOutcome::Updates(vec![ChatResponseUpdate {
                role: Some(Role::assistant()),
                contents: vec![Content::Usage(UsageContent {
                    details: parse_usage(usage),
                })],
                conversation_id: Some(thread_id.to_string()),
                message_id: response_id.clone(),
                response_id: response_id.clone(),
                ..Default::default()
            }]),
            None => EventOutcome::None,
        },
        "thread.run.failed" => {
            let msg = data
                .get("last_error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("run failed")
                .to_string();
            EventOutcome::Error(Error::service(msg))
        }
        "error" => {
            let msg = data
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .or_else(|| data.get("message").and_then(Value::as_str))
                .unwrap_or("assistants stream error")
                .to_string();
            EventOutcome::Error(Error::service(msg))
        }
        _ => EventOutcome::None,
    }
}

/// Build function-call contents from a `requires_action` run, encoding each
/// call id as the JSON `[run_id, tool_call_id]` pair the tool-result round-trip
/// later decodes (`_assistants_client.py:374-385`).
fn create_function_call_contents(data: &Value, response_id: &Option<String>) -> Vec<Content> {
    let mut contents = Vec::new();
    let Some(tool_calls) = data
        .get("required_action")
        .and_then(|ra| ra.get("submit_tool_outputs"))
        .and_then(|sto| sto.get("tool_calls"))
        .and_then(Value::as_array)
    else {
        return contents;
    };
    let run = response_id
        .clone()
        .map(Value::String)
        .unwrap_or(Value::Null);
    for tc in tool_calls {
        let tool_call_id = tc.get("id").and_then(Value::as_str).unwrap_or_default();
        let function = tc.get("function");
        let name = function
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        let arguments = function
            .and_then(|f| f.get("arguments"))
            .and_then(Value::as_str)
            .unwrap_or("{}");
        let call_id = serde_json::to_string(&Value::Array(vec![
            run.clone(),
            Value::String(tool_call_id.to_string()),
        ]))
        .unwrap_or_default();
        contents.push(Content::FunctionCall(FunctionCallContent::new(
            call_id,
            name,
            Some(FunctionArguments::Raw(arguments.to_string())),
        )));
    }
    contents
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{
        hosted_code_interpreter, hosted_file_search, ApprovalMode, ToolDefinition,
    };
    use agent_framework_core::types::UriContent;

    fn client() -> OpenAIAssistantsClient {
        OpenAIAssistantsClient::new("sk-test", "gpt-4o-mini")
    }

    fn function_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: "desc".into(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        }
    }

    // region: request builder (headers / auth / url)

    #[test]
    fn requests_carry_beta_header_auth_and_base_url() {
        let req = client()
            .request(reqwest::Method::POST, "threads")
            .build()
            .unwrap();
        assert_eq!(req.headers().get("OpenAI-Beta").unwrap(), "assistants=v2");
        assert_eq!(
            req.headers()
                .get(reqwest::header::AUTHORIZATION)
                .unwrap()
                .to_str()
                .unwrap(),
            "Bearer sk-test"
        );
        assert_eq!(req.url().as_str(), "https://api.openai.com/v1/threads");
        // No organization header unless configured.
        assert!(req.headers().get("OpenAI-Organization").is_none());
    }

    #[test]
    fn requests_include_organization_and_respect_base_url_override() {
        let c = client()
            .with_organization("org-42")
            .with_base_url("https://example.test/v1/");
        let req = c
            .request(
                reqwest::Method::GET,
                "threads/thread_1/runs?limit=1&order=desc",
            )
            .build()
            .unwrap();
        assert_eq!(req.headers().get("OpenAI-Organization").unwrap(), "org-42");
        assert_eq!(
            req.url().as_str(),
            "https://example.test/v1/threads/thread_1/runs?limit=1&order=desc"
        );
    }

    // endregion

    // region: prepare_options -> run body

    #[test]
    fn prepare_options_maps_scalars() {
        let mut options = ChatOptions::new()
            .with_model("gpt-4o")
            .with_temperature(0.5)
            .with_max_tokens(256);
        options.top_p = Some(0.25);
        options.allow_multiple_tool_calls = Some(true);
        let (run, tool_results) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert_eq!(run["model"], json!("gpt-4o"));
        assert_eq!(run["temperature"], json!(0.5));
        assert_eq!(run["top_p"], json!(0.25));
        assert_eq!(run["max_completion_tokens"], json!(256));
        assert_eq!(run["parallel_tool_calls"], json!(true));
        assert!(tool_results.is_none());
        // Scalars are omitted, not sent as null, when unset.
        let (bare, _) = prepare_options(&[ChatMessage::user("hi")], &ChatOptions::new());
        assert!(bare.get("temperature").is_none());
        assert!(bare.get("model").is_none());
        assert!(bare.get("tools").is_none());
        assert!(bare.get("tool_choice").is_none());
    }

    #[test]
    fn prepare_options_maps_function_and_hosted_tools() {
        let mut options = ChatOptions::new().with_tool_choice(ToolMode::Auto);
        options.tools = vec![
            function_tool("get_weather"),
            hosted_code_interpreter(),
            hosted_file_search(Some(7)),
        ];
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        let tools = run["tools"].as_array().unwrap();
        assert_eq!(tools[0]["type"], json!("function"));
        assert_eq!(tools[0]["function"]["name"], json!("get_weather"));
        assert_eq!(tools[1], json!({ "type": "code_interpreter" }));
        assert_eq!(
            tools[2],
            json!({ "type": "file_search", "max_num_results": 7 })
        );
        assert_eq!(run["tool_choice"], json!("auto"));
    }

    #[test]
    fn prepare_options_tools_dropped_when_tool_choice_none() {
        let mut options = ChatOptions::new().with_tool_choice(ToolMode::None);
        options.tools = vec![function_tool("f")];
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert!(run.get("tools").is_none());
        assert_eq!(run["tool_choice"], json!("none"));
    }

    #[test]
    fn prepare_options_tools_omitted_without_tool_choice() {
        // Python gates the whole tools block on `tool_choice is not None`.
        let mut options = ChatOptions::new();
        options.tools = vec![function_tool("f")];
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert!(run.get("tools").is_none());
        assert!(run.get("tool_choice").is_none());
    }

    #[test]
    fn prepare_options_tool_choice_required_named_and_any() {
        let mut named =
            ChatOptions::new().with_tool_choice(ToolMode::Required(Some("get_weather".into())));
        named.tools = vec![function_tool("get_weather")];
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &named);
        assert_eq!(
            run["tool_choice"],
            json!({ "type": "function", "function": { "name": "get_weather" } })
        );

        // A bare "required" (no function name) sends no tool_choice, mirroring
        // Python (which only maps the named-required case).
        let mut any = ChatOptions::new().with_tool_choice(ToolMode::Required(None));
        any.tools = vec![function_tool("get_weather")];
        let (run_any, _) = prepare_options(&[ChatMessage::user("hi")], &any);
        assert!(run_any.get("tool_choice").is_none());
        // Tools are still emitted (tool_choice is Some and != none).
        assert!(run_any.get("tools").is_some());
    }

    #[test]
    fn prepare_options_hosted_tools_fold_vector_stores_and_file_ids_into_tool_resources() {
        let mut options = ChatOptions::new().with_tool_choice(ToolMode::Auto);
        options.tools = vec![
            hosted_file_search(None).vector_store_ids(vec!["vs_1".into(), "vs_2".into()]),
            hosted_code_interpreter().file_ids(vec!["file-1".into()]),
        ];
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert_eq!(
            run["tool_resources"],
            json!({
                "file_search": { "vector_store_ids": ["vs_1", "vs_2"] },
                "code_interpreter": { "file_ids": ["file-1"] },
            })
        );
    }

    #[test]
    fn prepare_options_response_format_json_schema() {
        use agent_framework_core::types::ResponseFormat;
        let options = ChatOptions::new().with_response_format(ResponseFormat::JsonSchema {
            name: "answer".into(),
            description: None,
            schema: json!({ "type": "object" }),
            strict: Some(true),
        });
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert_eq!(
            run["response_format"],
            json!({
                "type": "json_schema",
                "json_schema": { "name": "answer", "schema": { "type": "object" }, "strict": true },
            })
        );
    }

    #[test]
    fn prepare_options_metadata_and_additional_properties_passthrough() {
        let mut options = ChatOptions::new();
        options.metadata = Some(std::collections::HashMap::from([(
            "session".to_string(),
            "abc".to_string(),
        )]));
        options
            .additional_properties
            .insert("max_prompt_tokens".into(), json!(1024));
        let (run, _) = prepare_options(&[ChatMessage::user("hi")], &options);
        assert_eq!(run["metadata"], json!({ "session": "abc" }));
        assert_eq!(run["max_prompt_tokens"], json!(1024));
    }

    #[test]
    fn prepare_options_system_and_developer_messages_become_instructions() {
        let messages = vec![
            ChatMessage::system("Be terse."),
            ChatMessage::new(Role::new("developer"), "Prefer bullet points."),
            ChatMessage::user("hi"),
        ];
        let (run, _) = prepare_options(&messages, &ChatOptions::new());
        // Concatenated with no separator, matching Python's `"".join`.
        assert_eq!(run["instructions"], json!("Be terse.Prefer bullet points."));
        // Only the user message becomes an additional message.
        let add = run["additional_messages"].as_array().unwrap();
        assert_eq!(add.len(), 1);
        assert_eq!(add[0]["role"], json!("user"));
    }

    #[test]
    fn prepare_options_additional_messages_text_image_and_roles() {
        let assistant_msg = ChatMessage::with_contents(
            Role::assistant(),
            vec![
                Content::Text(TextContent::new("prior")),
                Content::Uri(UriContent {
                    uri: "https://ex.com/cat.png".into(),
                    media_type: "image/png".into(),
                }),
            ],
        );
        let (run, _) = prepare_options(
            &[assistant_msg, ChatMessage::user("now")],
            &ChatOptions::new(),
        );
        let add = run["additional_messages"].as_array().unwrap();
        assert_eq!(
            add[0],
            json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "prior" },
                    { "type": "image_url", "image_url": { "url": "https://ex.com/cat.png" } },
                ],
            })
        );
        assert_eq!(add[1]["role"], json!("user"));
    }

    #[test]
    fn prepare_options_function_results_split_out_as_tool_results() {
        let call_id = serde_json::to_string(&json!(["run_1", "call_1"])).unwrap();
        let tool_msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id.clone(),
                Some(json!("18C")),
            ))],
        );
        let (run, tool_results) = prepare_options(&[tool_msg], &ChatOptions::new());
        // A pure tool message produces no additional_messages.
        assert!(run.get("additional_messages").is_none());
        let results = tool_results.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].call_id, call_id);
    }

    // endregion

    // region: thread / run body assembly

    #[test]
    fn take_thread_fields_removes_messages_and_tool_resources_keeps_metadata() {
        let mut run = Map::new();
        run.insert("additional_messages".into(), json!([{ "role": "user" }]));
        run.insert("tool_resources".into(), json!({ "file_search": {} }));
        run.insert("metadata".into(), json!({ "k": "v" }));
        run.insert("temperature".into(), json!(0.5));
        let (messages, tool_resources, metadata) = take_thread_fields(&mut run);
        assert_eq!(messages, Some(json!([{ "role": "user" }])));
        assert_eq!(tool_resources, Some(json!({ "file_search": {} })));
        assert_eq!(metadata, Some(json!({ "k": "v" })));
        // additional_messages + tool_resources are removed from the run body;
        // metadata + other fields remain.
        assert!(run.get("additional_messages").is_none());
        assert!(run.get("tool_resources").is_none());
        assert_eq!(run["metadata"], json!({ "k": "v" }));
        assert_eq!(run["temperature"], json!(0.5));
    }

    #[test]
    fn build_thread_body_omits_empty_messages() {
        let body = build_thread_body(Some(json!([])), None, Some(json!({ "k": "v" })));
        assert!(body.get("messages").is_none());
        assert_eq!(body["metadata"], json!({ "k": "v" }));

        let body2 = build_thread_body(
            Some(json!([{ "role": "user", "content": [] }])),
            Some(json!({ "file_search": { "vector_store_ids": ["vs_1"] } })),
            None,
        );
        assert_eq!(body2["messages"][0]["role"], json!("user"));
        assert_eq!(
            body2["tool_resources"]["file_search"]["vector_store_ids"][0],
            json!("vs_1")
        );
    }

    #[test]
    fn build_run_body_adds_assistant_id_and_stream() {
        let mut run = Map::new();
        run.insert("temperature".into(), json!(0.5));
        let body = build_run_body(run, "asst_123");
        assert_eq!(body["assistant_id"], json!("asst_123"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["temperature"], json!(0.5));
    }

    // endregion

    // region: call_id encode / decode round-trip

    #[test]
    fn call_id_round_trips_through_requires_action_and_tool_output() {
        // Encode: a requires_action event yields a composite call_id.
        let data = json!({
            "id": "run_1",
            "required_action": {
                "type": "submit_tool_outputs",
                "submit_tool_outputs": {
                    "tool_calls": [
                        { "id": "call_a", "type": "function",
                          "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } },
                    ],
                },
            },
        });
        let contents = create_function_call_contents(&data, &Some("run_1".to_string()));
        let Content::FunctionCall(fc) = &contents[0] else {
            panic!("expected function call");
        };
        assert_eq!(fc.call_id, r#"["run_1","call_a"]"#);
        assert_eq!(fc.name, "get_weather");

        // Decode: the matching function result recovers run id + tool_call id.
        let fr = FunctionResultContent::new(fc.call_id.clone(), Some(json!("18C and sunny")));
        let (run_id, outputs) = convert_function_results_to_tool_output(Some(&[fr]));
        assert_eq!(run_id.as_deref(), Some("run_1"));
        let outputs = outputs.unwrap();
        assert_eq!(
            outputs[0],
            json!({ "tool_call_id": "call_a", "output": "18C and sunny" })
        );
    }

    #[test]
    fn convert_tool_output_uses_placeholder_for_empty_result() {
        let call_id = serde_json::to_string(&json!(["run_1", "call_a"])).unwrap();
        let fr = FunctionResultContent::new(call_id, None);
        let (_run, outputs) = convert_function_results_to_tool_output(Some(&[fr]));
        assert_eq!(outputs.unwrap()[0]["output"], json!("No output received."));
    }

    #[test]
    fn convert_tool_output_skips_malformed_and_mismatched_run_ids() {
        // Not a 2-element array -> skipped; wrong run id vs first -> skipped.
        let bad = FunctionResultContent::new("not-json".to_string(), Some(json!("x")));
        let one = FunctionResultContent::new(
            serde_json::to_string(&json!(["run_1", "call_a"])).unwrap(),
            Some(json!("a")),
        );
        let other = FunctionResultContent::new(
            serde_json::to_string(&json!(["run_2", "call_b"])).unwrap(),
            Some(json!("b")),
        );
        let (run_id, outputs) = convert_function_results_to_tool_output(Some(&[bad, one, other]));
        assert_eq!(run_id.as_deref(), Some("run_1"));
        let outputs = outputs.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0]["tool_call_id"], json!("call_a"));
    }

    // endregion

    // region: SSE event-stream parsing

    /// Build an Assistants SSE body (event + data lines) ending with `[DONE]`.
    fn sse(events: &[(&str, Value)]) -> String {
        let mut out = String::new();
        for (event, data) in events {
            out.push_str(&format!("event: {event}\ndata: {data}\n\n"));
        }
        out.push_str("event: done\ndata: [DONE]\n\n");
        out
    }

    async fn collect(text: String, thread_id: &str) -> Vec<Result<ChatResponseUpdate>> {
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        assistants_sse_stream(byte_stream, thread_id.to_string())
            .collect::<Vec<_>>()
            .await
    }

    fn oks(items: Vec<Result<ChatResponseUpdate>>) -> Vec<ChatResponseUpdate> {
        items.into_iter().map(|r| r.unwrap()).collect()
    }

    #[tokio::test]
    async fn stream_text_and_usage_aggregate() {
        let text = sse(&[
            (
                "thread.run.created",
                json!({ "id": "run_1", "object": "thread.run" }),
            ),
            (
                "thread.run.step.created",
                json!({ "id": "step_1", "run_id": "run_1" }),
            ),
            (
                "thread.message.delta",
                json!({ "delta": { "role": "assistant", "content": [
                    { "index": 0, "type": "text", "text": { "value": "Hel" } }
                ] } }),
            ),
            (
                "thread.message.delta",
                json!({ "delta": { "role": "assistant", "content": [
                    { "index": 0, "type": "text", "text": { "value": "lo!" } }
                ] } }),
            ),
            (
                "thread.run.completed",
                json!({ "id": "run_1", "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 } }),
            ),
        ]);
        let updates = oks(collect(text, "thread_abc").await);
        // Lifecycle events (run.created / step.created) emit nothing.
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.conversation_id.as_deref(), Some("thread_abc"));
        // message_id / response_id are the run id (from step.created).
        assert_eq!(resp.response_id.as_deref(), Some("run_1"));
        assert_eq!(resp.messages.len(), 1);
        assert_eq!(resp.messages[0].message_id.as_deref(), Some("run_1"));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(10));
        assert_eq!(usage.output_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(15));
    }

    #[tokio::test]
    async fn stream_requires_action_extracts_function_calls() {
        let text = sse(&[
            (
                "thread.run.step.created",
                json!({ "id": "step_1", "run_id": "run_9" }),
            ),
            (
                "thread.run.requires_action",
                json!({
                    "id": "run_9",
                    "required_action": {
                        "type": "submit_tool_outputs",
                        "submit_tool_outputs": { "tool_calls": [
                            { "id": "call_x", "type": "function",
                              "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } }
                        ] },
                    },
                }),
            ),
        ]);
        let updates = oks(collect(text, "thread_1").await);
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        // The call id encodes [run_id, tool_call_id].
        assert_eq!(calls[0].call_id, r#"["run_9","call_x"]"#);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &json!("Paris")
        );
        assert_eq!(resp.conversation_id.as_deref(), Some("thread_1"));
    }

    #[tokio::test]
    async fn stream_run_failed_surfaces_error() {
        let text = sse(&[(
            "thread.run.failed",
            json!({ "id": "run_1", "status": "failed", "last_error": { "code": "server_error", "message": "boom" } }),
        )]);
        let items = collect(text, "thread_1").await;
        let err = items
            .into_iter()
            .find_map(Result::err)
            .expect("expected an error from thread.run.failed");
        assert!(err.to_string().contains("boom"));
    }

    #[tokio::test]
    async fn stream_error_event_surfaces_error() {
        let text = sse(&[(
            "error",
            json!({ "error": { "message": "rate limited", "type": "rate_limit" } }),
        )]);
        let items = collect(text, "thread_1").await;
        let err = items
            .into_iter()
            .find_map(Result::err)
            .expect("expected an error from an error event");
        assert!(err.to_string().contains("rate limited"));
    }

    #[tokio::test]
    async fn stream_user_role_delta_maps_to_user() {
        let text = sse(&[
            ("thread.run.step.created", json!({ "run_id": "run_1" })),
            (
                "thread.message.delta",
                json!({ "delta": { "role": "user", "content": [
                    { "type": "text", "text": { "value": "echo" } }
                ] } }),
            ),
        ]);
        let updates = oks(collect(text, "thread_1").await);
        let with_content: Vec<_> = updates.iter().filter(|u| !u.contents.is_empty()).collect();
        assert_eq!(with_content.len(), 1);
        assert_eq!(with_content[0].role, Some(Role::user()));
    }

    // endregion

    // region: constructor surface

    #[test]
    fn builders_set_assistant_and_thread_config() {
        let c = client()
            .with_assistant_id("asst_existing")
            .with_assistant_name("Helper")
            .with_assistant_description("A helper")
            .with_thread_id("thread_default");
        assert_eq!(c.assistant_id(), Some("asst_existing"));
        assert_eq!(c.model(), "gpt-4o-mini");
        assert_eq!(c.assistant_name.as_deref(), Some("Helper"));
        assert_eq!(c.assistant_description.as_deref(), Some("A helper"));
        assert_eq!(c.thread_id.as_deref(), Some("thread_default"));
    }

    /// Guards `OPENAI_*` env mutation across the two env-var tests (process-wide
    /// and read on multiple test threads).
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-env-123");
            std::env::set_var("OPENAI_BASE_URL", "https://example.test/v1");
        }
        let c = OpenAIAssistantsClient::from_env("gpt-4o-mini").unwrap();
        assert_eq!(c.api_key, "sk-env-123");
        assert_eq!(c.base_url, "https://example.test/v1");
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_BASE_URL");
        }
    }

    #[test]
    fn from_env_errors_when_api_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_BASE_URL");
        }
        assert!(OpenAIAssistantsClient::from_env("gpt-4o-mini").is_err());
    }

    // endregion
}
