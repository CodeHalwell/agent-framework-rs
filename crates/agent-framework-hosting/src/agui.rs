//! AG-UI protocol hosting (CopilotKit's SupportsAgentRun-User Interaction protocol).
//!
//! Serves one or more agents over the AG-UI protocol, mirroring the Python
//! `agent_framework_ag_ui` package. AG-UI is Server-Sent Events with camelCase
//! JSON events discriminated by a SCREAMING_SNAKE `type` field.
//!
//! # Route
//! `POST {path}` (default `/`) accepts a [`RunAgentInput`]-shaped body
//! (`threadId`, `runId`, `messages[]`, `tools[]`, `state?`, `context?`,
//! `forwardedProps?`) and responds `text/event-stream`.
//!
//! # Event sequence
//! Emitted with the exact `type` strings the Python bridge
//! (`AgentFrameworkEventBridge`) and the upstream `ag-ui-protocol` SDK use:
//!
//! - `RUN_STARTED` `{threadId, runId}`
//! - per assistant text: `TEXT_MESSAGE_START` `{messageId, role}` →
//!   `TEXT_MESSAGE_CONTENT` `{messageId, delta}`* → `TEXT_MESSAGE_END`
//!   `{messageId}` (one text message per run, closed at the end — mirrors the
//!   bridge, which never resets `current_message_id`)
//! - per tool call: `TOOL_CALL_START` `{toolCallId, toolCallName,
//!   parentMessageId?}` → `TOOL_CALL_ARGS` `{toolCallId, delta}` →
//!   `TOOL_CALL_END` `{toolCallId}` (emitted on the tool's result, or at
//!   finalize for an unexecuted / frontend tool call)
//! - per tool result: `TOOL_CALL_END` `{toolCallId}` then `TOOL_CALL_RESULT`
//!   `{messageId, toolCallId, content, role}`
//! - per approval request: `TOOL_CALL_END` `{toolCallId}` then `CUSTOM`
//!   `{name: "function_approval_request", value}`
//! - `RUN_FINISHED` `{threadId, runId}` — or, on agent failure, `RUN_ERROR`
//!   `{message, code?}` in its place.
//!
//! Wire fidelity: `type` values are the SDK's SCREAMING_SNAKE strings; fields
//! are camelCase; `None`/absent fields are omitted (the SDK encodes events with
//! `by_alias=True, exclude_none=True`); frames are `data: {json}\n\n` with no
//! `[DONE]` sentinel (AG-UI, unlike OpenAI streaming, has none).
//!
//! # The frontend-tool pattern
//! In Python, client-declared `tools` are registered on the agent as
//! *declaration-only* `AIFunction`s: when the model calls one it is **not**
//! executed server-side, so no result is produced, and the bridge surfaces it
//! as `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` **without** a
//! `TOOL_CALL_RESULT` — the browser executes it. We mirror the *event* half of
//! this faithfully: any tool call the agent emits without a matching result is
//! closed with a `TOOL_CALL_END` at finalize and never gets a `TOOL_CALL_RESULT`.
//!
//! # Divergences from the Python reference
//! - **Live streaming.** The object-safe [`SupportsAgentRun`] trait exposes `run_stream`,
//!   so this router drives it and frames each [`AgentResponseUpdate`] into
//!   AG-UI events as it arrives — one `TEXT_MESSAGE_CONTENT` per text delta,
//!   `TOOL_CALL_*` per call fragment. `type` ordering and payloads match the
//!   bridge. (Agents whose `run_stream` falls back to the buffered default —
//!   e.g. a plain `Arc<dyn SupportsAgentRun>` that only implements `run` — still emit one
//!   `TEXT_MESSAGE_CONTENT` per message, as before.)
//!
//!   [`AgentResponseUpdate`]: agent_framework_core::types::AgentResponseUpdate
//! - **Client `tools` are injected as declaration-only tools.** Client-declared
//!   `tools` in the `RunAgentInput` are mapped to declaration-only
//!   [`ToolDefinition`](agent_framework_core::tools::ToolDefinition)s (no
//!   executor) and passed to the agent via per-run
//!   [`AgentRunOptions`](agent_framework_core::agent::AgentRunOptions). When the
//!   model calls one, the function-invocation loop returns the call unexecuted
//!   (see [`crate::agui`]'s frontend-tool note), and it is framed as
//!   `TOOL_CALL_*` **without** a `TOOL_CALL_RESULT` — the browser runs it. An
//!   agent that does not support per-run options (anything other than a
//!   `Agent`) logs a warning and ignores the injected tools.
//! - **State events are not emitted.** `STATE_SNAPSHOT` / `STATE_DELTA` /
//!   `MESSAGES_SNAPSHOT` are driven by Python's `predict_state_config` /
//!   `state_schema` (agentic generative-UI / predictive-state features) which
//!   have no equivalent on the core `SupportsAgentRun` trait. Inbound `state` is accepted
//!   and ignored. `CUSTOM` is emitted only for approval requests.

use std::collections::HashSet;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
use agent_framework_core::types::{
    Content, FunctionApprovalRequestContent, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, Role, TextContent,
};

use crate::registry::IntoAgentRegistration;
use crate::sse::sse_events_stream;
use crate::util;

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// One agent bound to a path.
struct AgUiRoute {
    path: String,
    name: String,
    agent: Arc<dyn SupportsAgentRun>,
}

/// Per-route handler state.
struct AgUiState {
    name: String,
    agent: Arc<dyn SupportsAgentRun>,
}

/// Serves agents over the AG-UI protocol.
///
/// A single agent is bound with [`AgUiRouter::for_agent`] (default path `/`);
/// [`AgUiRouter::path`] overrides that path and [`AgUiRouter::agent`] adds more
/// agents at further paths.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::agent::Agent;
/// # use agent_framework_hosting::agui::AgUiRouter;
/// # fn demo(chat: Agent, planner: Agent) -> axum::Router {
/// AgUiRouter::for_agent("chat", chat)
///     .path("/agent")
///     .agent("planner", planner, "/planner")
///     .into_router()
/// # }
/// ```
pub struct AgUiRouter {
    routes: Vec<AgUiRoute>,
}

impl AgUiRouter {
    /// Serve `agent` at the default path `/`, identified by `name`.
    ///
    /// Accepts a [`Agent`](agent_framework_core::agent::Agent), a
    /// [`WorkflowAgent`](agent_framework_core::workflow::WorkflowAgent), or an
    /// `Arc<dyn SupportsAgentRun>` (see [`IntoAgentRegistration`]).
    pub fn for_agent(name: impl Into<String>, agent: impl IntoAgentRegistration) -> Self {
        Self {
            routes: vec![AgUiRoute {
                path: "/".to_string(),
                name: name.into(),
                agent: agent.into_agent_registration().agent,
            }],
        }
    }

    /// Override the path of the primary agent (the one from
    /// [`AgUiRouter::for_agent`]).
    pub fn path(mut self, path: impl Into<String>) -> Self {
        if let Some(first) = self.routes.first_mut() {
            first.path = path.into();
        }
        self
    }

    /// Add another agent at `path`. Paths must be unique across the router.
    pub fn agent(
        mut self,
        name: impl Into<String>,
        agent: impl IntoAgentRegistration,
        path: impl Into<String>,
    ) -> Self {
        self.routes.push(AgUiRoute {
            path: path.into(),
            name: name.into(),
            agent: agent.into_agent_registration().agent,
        });
        self
    }

    /// Build the axum router: one `POST {path}` per registered agent. Composable
    /// and nestable into a larger app.
    pub fn into_router(self) -> Router {
        let mut router = Router::new();
        for route in self.routes {
            let state = Arc::new(AgUiState {
                name: route.name,
                agent: route.agent,
            });
            let sub = Router::new()
                .route(&route.path, post(run_agent))
                .with_state(state);
            router = router.merge(sub);
        }
        router
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn run_agent(State(state): State<Arc<AgUiState>>, body: String) -> Response {
    // Parse the body ourselves so malformed input becomes a clean 400 with an
    // AG-UI-shaped error body rather than an axum extractor rejection.
    let input: RunAgentInput = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("Invalid RunAgentInput: {e}")),
    };

    // Thread/run ids are echoed back; generate them when the client omits them
    // (mirrors the Python orchestrator's `input_data.get(...) or uuid4()`).
    let thread_id = input
        .thread_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    tracing::debug!(agent = %state.name, thread_id = %thread_id, run_id = %run_id, "AG-UI run");

    let messages = input_to_messages(&input.messages);

    // Client-declared tools become declaration-only per-run tools; when the
    // model calls one it is returned unexecuted so the browser runs it.
    let additional_tools = parse_client_tools(&input.tools);
    let run_options = if additional_tools.is_empty() {
        None
    } else {
        Some(AgentRunOptions {
            additional_tools,
            ..Default::default()
        })
    };

    // Live streaming: drive `run_stream` and frame each update into AG-UI
    // events as it arrives (one `TEXT_MESSAGE_CONTENT` per text delta, etc.).
    let agent = state.agent.clone();
    let (tx, rx) = futures::channel::mpsc::unbounded::<Value>();
    tokio::spawn(async move {
        let _ = tx.unbounded_send(run_started(&thread_id, &run_id));
        match agent.run_stream(messages, None, run_options).await {
            Ok(mut stream) => {
                let mut framing = AgUiFraming::new();
                let mut errored = false;
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(update) => {
                            let mut events = Vec::new();
                            framing.push_contents(&update.contents, &mut events);
                            for ev in events {
                                let _ = tx.unbounded_send(ev);
                            }
                        }
                        // SupportsAgentRun failure mid-stream → RUN_ERROR in place of
                        // RUN_FINISHED (still a 200 SSE stream; error in-band).
                        Err(e) => {
                            let _ = tx.unbounded_send(
                                json!({ "type": "RUN_ERROR", "message": e.to_string() }),
                            );
                            errored = true;
                            break;
                        }
                    }
                }
                if !errored {
                    let mut events = Vec::new();
                    framing.finalize(&mut events);
                    events.push(run_finished(&thread_id, &run_id));
                    for ev in events {
                        let _ = tx.unbounded_send(ev);
                    }
                }
            }
            // Failure before the stream even opens.
            Err(e) => {
                let _ = tx.unbounded_send(json!({ "type": "RUN_ERROR", "message": e.to_string() }));
            }
        }
    });

    sse_events_stream(rx)
}

/// Map AG-UI client-declared tools (`{name, description, parameters}`) into
/// **declaration-only** [`ToolDefinition`]s (no executor). Passed to the agent
/// as per-run `additional_tools`: when the model calls one, the
/// function-invocation loop returns the call to the caller unexecuted, and the
/// AG-UI framing surfaces it as `TOOL_CALL_*` without a `TOOL_CALL_RESULT` —
/// the browser executes it (the frontend-tool pattern).
fn parse_client_tools(tools: &[Value]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .filter_map(|t| {
            let obj = t.as_object()?;
            let name = obj.get("name").and_then(Value::as_str)?.to_string();
            if name.is_empty() {
                return None;
            }
            let description = obj
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let parameters = obj
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            Some(ToolDefinition {
                name,
                description,
                parameters,
                kind: ToolKind::Function,
                approval_mode: ApprovalMode::NeverRequire,
                executor: None,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Event framing
// ---------------------------------------------------------------------------

fn run_started(thread_id: &str, run_id: &str) -> Value {
    json!({ "type": "RUN_STARTED", "threadId": thread_id, "runId": run_id })
}

fn run_finished(thread_id: &str, run_id: &str) -> Value {
    json!({ "type": "RUN_FINISHED", "threadId": thread_id, "runId": run_id })
}

/// Incremental AG-UI event framing, driven one streamed update's contents at a
/// time (via [`AgUiFraming::push_contents`]) and closed with
/// [`AgUiFraming::finalize`]. This reproduces the bridge's stateful framing
/// across the whole run: a single text message spans the run (lazily opened on
/// the first text delta, never reset), and tool calls are tracked so unresulted
/// ones (frontend tools) are closed at the end.
struct AgUiFraming {
    /// The run's single text message id, opened on the first text delta.
    message_id: Option<String>,
    /// Tool calls opened (in order) and those already closed via a result.
    started_tools: Vec<String>,
    ended_tools: HashSet<String>,
}

impl AgUiFraming {
    fn new() -> Self {
        Self {
            message_id: None,
            started_tools: Vec::new(),
            ended_tools: HashSet::new(),
        }
    }

    /// Frame one streamed update's `contents`, pushing AG-UI events onto `out`.
    fn push_contents(&mut self, contents: &[Content], out: &mut Vec<Value>) {
        for content in contents {
            match content {
                Content::Text(TextContent { text, .. }) if !text.is_empty() => {
                    let mid = match &self.message_id {
                        Some(mid) => mid.clone(),
                        None => {
                            let mid = util::msg_id();
                            out.push(json!({
                                "type": "TEXT_MESSAGE_START",
                                "messageId": mid,
                                "role": "assistant",
                            }));
                            self.message_id = Some(mid.clone());
                            mid
                        }
                    };
                    out.push(json!({
                        "type": "TEXT_MESSAGE_CONTENT",
                        "messageId": mid,
                        "delta": text,
                    }));
                }
                Content::FunctionCall(fc) => {
                    let tool_call_id = coalesce_call_id(fc, self.started_tools.last());
                    if !fc.name.is_empty() {
                        let mut start = json!({
                            "type": "TOOL_CALL_START",
                            "toolCallId": tool_call_id,
                            "toolCallName": fc.name,
                        });
                        // parentMessageId is included only when a text message
                        // is open (omitted-when-None, per the SDK).
                        if let Some(mid) = &self.message_id {
                            start["parentMessageId"] = json!(mid);
                        }
                        out.push(start);
                        self.started_tools.push(tool_call_id.clone());
                    }
                    if let Some(delta) = arguments_delta(fc) {
                        if !delta.is_empty() {
                            out.push(json!({
                                "type": "TOOL_CALL_ARGS",
                                "toolCallId": tool_call_id,
                                "delta": delta,
                            }));
                        }
                    }
                }
                Content::FunctionResult(fr) => {
                    if !fr.call_id.is_empty() {
                        out.push(json!({ "type": "TOOL_CALL_END", "toolCallId": fr.call_id }));
                        self.ended_tools.insert(fr.call_id.clone());
                    }
                    out.push(json!({
                        "type": "TOOL_CALL_RESULT",
                        "messageId": util::msg_id(),
                        "toolCallId": fr.call_id,
                        "content": result_content(fr),
                        "role": "tool",
                    }));
                }
                Content::FunctionApprovalRequest(ar) => {
                    if !ar.function_call.call_id.is_empty() {
                        out.push(json!({
                            "type": "TOOL_CALL_END",
                            "toolCallId": ar.function_call.call_id,
                        }));
                        self.ended_tools.insert(ar.function_call.call_id.clone());
                    }
                    out.push(approval_custom_event(ar));
                }
                // Reasoning, data, uri, error, usage, hosted-file/-vector-store,
                // approval *responses* → no AG-UI event (documented subset).
                _ => {}
            }
        }
    }

    /// Emit the run's closing events: close any tool call that never received a
    /// result (frontend-tool pattern), in open order, then close the run's text
    /// message if one was opened. Mirrors the orchestrator's
    /// `pending_without_end` sweep.
    fn finalize(&mut self, out: &mut Vec<Value>) {
        for tool_call_id in &self.started_tools {
            if !self.ended_tools.contains(tool_call_id) {
                out.push(json!({ "type": "TOOL_CALL_END", "toolCallId": tool_call_id }));
            }
        }
        if let Some(mid) = &self.message_id {
            out.push(json!({ "type": "TEXT_MESSAGE_END", "messageId": mid }));
        }
    }
}

/// The AG-UI `CUSTOM` event carrying a function-approval request (mirrors the
/// bridge's `function_approval_request` custom event).
fn approval_custom_event(ar: &FunctionApprovalRequestContent) -> Value {
    let arguments = ar
        .function_call
        .parse_arguments()
        .map(|m| Value::Object(m.into_iter().collect()))
        .unwrap_or(Value::Null);
    json!({
        "type": "CUSTOM",
        "name": "function_approval_request",
        "value": {
            "id": ar.id,
            "function_call": {
                "call_id": ar.function_call.call_id,
                "name": ar.function_call.name,
                "arguments": arguments,
            },
        },
    })
}

/// Resolve the tool-call id for a call fragment: its own `call_id`, else the
/// currently open tool call, else a fresh id (mirrors `_coalesce_tool_call_id`).
fn coalesce_call_id(fc: &FunctionCallContent, current: Option<&String>) -> String {
    if !fc.call_id.is_empty() {
        fc.call_id.clone()
    } else if let Some(cur) = current {
        cur.clone()
    } else {
        util::short_hex()
    }
}

/// The `TOOL_CALL_ARGS` delta string for a call: a raw argument string as-is, a
/// parsed object JSON-serialized (mirrors the bridge's `delta_str`).
fn arguments_delta(fc: &FunctionCallContent) -> Option<String> {
    match &fc.arguments {
        None => None,
        Some(FunctionArguments::Raw(s)) => Some(s.clone()),
        Some(FunctionArguments::Object(map)) => {
            Some(serde_json::to_string(map).unwrap_or_default())
        }
    }
}

/// The `TOOL_CALL_RESULT` content string: a dict/array JSON-serialized, any
/// other value stringified, `None`/exception → the exception text or empty
/// (mirrors the bridge's `result_content`).
fn result_content(fr: &FunctionResultContent) -> String {
    match &fr.result {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => fr.exception.clone().unwrap_or_default(),
        Some(other) => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Input model & message mapping
// ---------------------------------------------------------------------------

/// The AG-UI `RunAgentInput` (tolerant subset).
///
/// Fields are camelCase per the protocol; snake_case aliases are also accepted
/// for the routing ids (the Python server reads `input_data["thread_id"]` /
/// `["run_id"]`). Unknown fields are ignored.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct RunAgentInput {
    /// Conversation thread id (echoed on `RUN_STARTED`/`RUN_FINISHED`).
    #[serde(alias = "thread_id")]
    pub thread_id: Option<String>,
    /// Current run id (echoed on `RUN_STARTED`/`RUN_FINISHED`).
    #[serde(alias = "run_id")]
    pub run_id: Option<String>,
    /// Conversation history (AG-UI `Message` objects).
    pub messages: Vec<Value>,
    /// Client-declared tools, injected as declaration-only per-run tools so
    /// their calls round-trip back to the browser (see module docs).
    pub tools: Vec<Value>,
    /// Shared state (accepted and ignored — see module docs).
    pub state: Option<Value>,
    /// Contextual objects (accepted and ignored).
    pub context: Vec<Value>,
    /// Extra forwarded properties (accepted and ignored).
    pub forwarded_props: Option<Value>,
}

/// Map AG-UI `messages[]` to core [`Message`]s.
///
/// Mirrors `agui_messages_to_agent_framework`: `role:"tool"` →
/// [`FunctionResultContent`]; an assistant with `toolCalls` →
/// [`FunctionCallContent`]s (plus any text); everything else → a text message
/// under the mapped role.
fn input_to_messages(messages: &[Value]) -> Vec<Message> {
    messages.iter().filter_map(map_message).collect()
}

fn map_message(msg: &Value) -> Option<Message> {
    let obj = msg.as_object()?;
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");

    // Tool result message.
    if role == "tool" {
        let call_id = obj
            .get("toolCallId")
            .or_else(|| obj.get("tool_call_id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let result = obj
            .get("content")
            .or_else(|| obj.get("result"))
            .cloned()
            .unwrap_or(Value::String(String::new()));
        return Some(Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(result),
            ))],
        ));
    }

    // Assistant message carrying tool calls.
    let tool_calls = obj.get("toolCalls").or_else(|| obj.get("tool_calls"));
    if let Some(Value::Array(calls)) = tool_calls {
        let mut contents: Vec<Content> = Vec::new();
        if let Some(text) = obj.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                contents.push(Content::Text(TextContent::new(text)));
            }
        }
        for tc in calls {
            if let Some(fc) = tool_call_to_content(tc) {
                contents.push(Content::FunctionCall(fc));
            }
        }
        return Some(Message::with_contents(Role::assistant(), contents));
    }

    // Plain text message under its role.
    let text = content_text(obj.get("content"));
    Some(Message::new(role_from(role), text))
}

/// Convert one AG-UI `ToolCall` (`{id, type:"function", function:{name,
/// arguments}}`) into a [`FunctionCallContent`].
fn tool_call_to_content(tc: &Value) -> Option<FunctionCallContent> {
    let obj = tc.as_object()?;
    if obj.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let func = obj.get("function").and_then(Value::as_object);
    let call_id = obj.get("id").and_then(Value::as_str).unwrap_or_default();
    let name = func
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = func.and_then(|f| f.get("arguments")).map(|a| match a {
        Value::String(s) => FunctionArguments::Raw(s.clone()),
        other => FunctionArguments::Raw(other.to_string()),
    });
    Some(FunctionCallContent::new(call_id, name, arguments))
}

/// Extract display text from an AG-UI `content` value (a string, or an array of
/// content parts each carrying `text`).
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Map an AG-UI role to a core [`Role`]. Unknown roles (e.g. `developer`) fall
/// back to `user`, matching the Python adapter's `.get(role, Role.USER)`.
fn role_from(role: &str) -> Role {
    match role {
        "assistant" => Role::assistant(),
        "system" => Role::system(),
        "tool" => Role::tool(),
        "user" => Role::user(),
        _ => Role::user(),
    }
}

fn bad_request(message: String) -> Response {
    (
        axum::http::StatusCode::BAD_REQUEST,
        Json(Value::Object({
            let mut m = Map::new();
            m.insert("error".to_string(), json!({ "message": message }));
            m
        })),
    )
        .into_response()
}
