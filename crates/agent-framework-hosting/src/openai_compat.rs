//! OpenAI Chat Completions-compatible hosting.
//!
//! Serves one agent at `POST /v1/chat/completions` in the OpenAI
//! `chat.completion` shape, including streaming (`data: {chunk}` /
//! `data: [DONE]`). This lets any OpenAI-Chat client talk to an agent.
//!
//! # Divergences
//! - Streaming drives the core `SupportsAgentRun::run_stream` and frames each update as a
//!   `chat.completion.chunk` live (one content chunk per non-empty update);
//!   non-streaming requests stay on `SupportsAgentRun::run`. Chunk framing matches the
//!   OpenAI streaming protocol.
//! - `usage` uses the agent's reported token counts when available, otherwise a
//!   ~4-chars-per-token estimate.
//!
//! # Tool calls
//! A turn whose agent declares client-side calls serializes them as
//! `message.tool_calls` (buffered) or `delta.tool_calls` keyed by a stable
//! per-call `index` (streaming), so the client receives the id, name and
//! arguments it needs to execute them; arguments go out as a JSON *string*,
//! as the wire format has it. A call the agent answered itself is not one
//! of these and is never serialized. The follow-up request is read back the
//! same way: `tool_calls` on the assistant message and `tool_call_id` on a
//! `role: "tool"` message become the call/result pair the agent needs.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Value};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::types::{
    AgentResponse, Content, FinishReason, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, Role, UsageDetails,
};

use crate::registry::IntoAgentRegistration;
use crate::sse::sse_response_stream;
use crate::util;

/// Serves one agent over the OpenAI Chat Completions API.
pub struct OpenAiRouter {
    model: String,
    agent: Arc<dyn SupportsAgentRun>,
}

impl OpenAiRouter {
    /// Build a chat-completions host for `agent`, advertised under model id
    /// `name`.
    ///
    /// Accepts a [`Agent`](agent_framework_core::agent::Agent), a
    /// [`WorkflowAgent`](agent_framework_core::workflow::WorkflowAgent), or an
    /// `Arc<dyn SupportsAgentRun>`.
    pub fn for_agent(name: impl Into<String>, agent: impl IntoAgentRegistration) -> Self {
        Self {
            model: name.into(),
            agent: agent.into_agent_registration().agent,
        }
    }

    /// Build the axum router (`POST /v1/chat/completions`). Composable into a
    /// larger app.
    pub fn into_router(self) -> Router {
        let state = Arc::new(OpenAiState {
            model: self.model,
            agent: self.agent,
        });
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(state)
    }
}

struct OpenAiState {
    model: String,
    agent: Arc<dyn SupportsAgentRun>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct ChatCompletionsRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    messages: Vec<IncomingMessage>,
    #[serde(default)]
    stream: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct IncomingMessage {
    role: String,
    #[serde(default)]
    content: Value,
    /// The assistant's own calls, replayed on the follow-up request. A
    /// client that executes a call sends the whole turn back — the
    /// assistant message carrying `tool_calls`, then one `role: "tool"`
    /// message per result — and dropping these would leave the results
    /// below answering calls the conversation no longer contains.
    #[serde(default)]
    tool_calls: Option<Vec<IncomingToolCall>>,
    /// Which call a `role: "tool"` message answers.
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct IncomingToolCall {
    #[serde(default)]
    id: String,
    #[serde(default)]
    function: IncomingToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct IncomingToolCallFunction {
    #[serde(default)]
    name: String,
    /// A JSON **string**, as the wire format has it — kept raw rather than
    /// parsed, so a fragment or a provider's exact formatting survives.
    #[serde(default)]
    arguments: Option<String>,
}

async fn chat_completions(
    State(state): State<Arc<OpenAiState>>,
    Json(request): Json<ChatCompletionsRequest>,
) -> Response {
    let model = request.model.clone().unwrap_or_else(|| state.model.clone());
    let messages = to_chat_messages(&request.messages);
    let input_len: usize = request
        .messages
        .iter()
        .map(|m| content_text(&m.content).len())
        .sum();

    let id = format!("chatcmpl-{}", util::short_hex());
    let created = util::now_ts() as u64;

    if request.stream {
        // Live streaming: drive `run_stream` and frame each update as a
        // `chat.completion.chunk` as it arrives. The channel is bounded, so a
        // slow client applies backpressure (the producer's `send().await`
        // suspends) and a disconnected client cancels the run: `disconnect`'s
        // `closed()` fires — or a `send` fails — and the producer future is
        // dropped, dropping the agent `stream` with it.
        let agent = state.agent.clone();
        let (tx, rx) = crate::sse::bounded_sse_channel();
        let disconnect = tx.clone();
        tokio::spawn(async move {
            let produce = async move {
                // First chunk carries the assistant role.
                if tx
                    .send(chunk(
                        &id,
                        created,
                        &model,
                        json!({ "role": "assistant" }),
                        Value::Null,
                    ))
                    .await
                    .is_err()
                {
                    return;
                }
                match agent.run_stream(messages, None, None).await {
                    Ok(mut stream) => {
                        // Whichever update carries a reason carries it; the
                        // last one wins, matching how `ChatResponse` aggregates
                        // a stream. Held across the loop so the terminal chunk
                        // can report it instead of asserting `"stop"`.
                        let mut finish_reason: Option<FinishReason> = None;
                        // `call_id`s in the order they were first seen. A
                        // streamed call arrives as fragments sharing one id,
                        // and OpenAI keys its deltas by a stable `index`, so
                        // the position here *is* that index.
                        let mut calls = util::StreamingCalls::default();
                        while let Some(item) = stream.next().await {
                            match item {
                                Ok(update) => {
                                    if update.finish_reason.is_some() {
                                        finish_reason = update.finish_reason.clone();
                                    }
                                    let text = update.text();
                                    if !text.is_empty()
                                        && tx
                                            .send(chunk(
                                                &id,
                                                created,
                                                &model,
                                                json!({ "content": text }),
                                                Value::Null,
                                            ))
                                            .await
                                            .is_err()
                                    {
                                        return;
                                    }
                                    // Held rather than sent: a call the
                                    // agent answers itself arrives with its
                                    // result in a *later* update, and an
                                    // already-sent delta cannot be recalled.
                                    calls.push(&update.contents);
                                }
                                Err(e) => {
                                    let _ = tx.send(stream_error(&e.to_string())).await;
                                    return;
                                }
                            }
                        }
                        // Now that the stream has ended, the calls left
                        // unanswered are the ones the client must run.
                        let deltas = tool_call_deltas(&calls);
                        if !deltas.is_empty()
                            && tx
                                .send(chunk(
                                    &id,
                                    created,
                                    &model,
                                    json!({ "tool_calls": deltas }),
                                    Value::Null,
                                ))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        // Terminal chunk. The same closed-enum rule as the
                        // buffered path: an unfamiliar provider reason is
                        // reported as `length` with the raw value alongside,
                        // never emitted into the enum itself.
                        let (reason, raw) =
                            finish_reason_of(finish_reason.as_ref(), !deltas.is_empty());
                        let raw = raw.map(str::to_string);
                        let _ = tx
                            .send(chunk_with_raw_reason(
                                &id,
                                created,
                                &model,
                                json!({}),
                                Value::String(reason.to_string()),
                                raw,
                            ))
                            .await;
                    }
                    Err(e) => {
                        let _ = tx.send(stream_error(&e.to_string())).await;
                    }
                }
            };
            // Whichever finishes first wins; if the client disconnects,
            // `closed()` completes and `produce` (holding the agent stream) is
            // dropped, cancelling the underlying run.
            tokio::select! {
                _ = disconnect.closed() => {}
                _ = produce => {}
            }
        });
        sse_response_stream(rx)
    } else {
        let response = match state.agent.run(messages, None).await {
            Ok(r) => r,
            Err(e) => return error_response(e.to_string()),
        };
        Json(completion_object(
            &response, &id, created, &model, input_len,
        ))
        .into_response()
    }
}

/// Build the non-streaming `chat.completion` response object.
fn completion_object(
    resp: &AgentResponse,
    id: &str,
    created: u64,
    model: &str,
    input_len: usize,
) -> Value {
    let text = resp.text();
    let calls = util::function_calls_of(&resp.messages);
    let (reason, raw) = finish_reason_of(resp.finish_reason.as_ref(), !calls.is_empty());
    let (prompt, completion) = token_counts(&resp.usage_details, input_len, text.len());
    // OpenAI sends `content: null` when a turn is only a tool call, and a
    // client that renders `content` verbatim would otherwise print "".
    let content = if text.is_empty() && !calls.is_empty() {
        Value::Null
    } else {
        Value::String(text.clone())
    };
    let tool_calls: Vec<Value> = calls
        .iter()
        .map(|call| {
            json!({
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": util::arguments_string(call),
                },
            })
        })
        .collect();
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                // Omitted entirely when the turn declared no call, so an
                // ordinary completion is byte-identical to before.
                "tool_calls": (!tool_calls.is_empty()).then_some(tool_calls),
            },
            "finish_reason": reason,
            // Non-standard, and omitted unless the provider's own reason had
            // to be approximated to keep the enum above legal.
            "x_finish_reason": raw,
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    })
}

/// The chat-completions `finish_reason` for a run, and the provider's own
/// reason when the two differ.
///
/// Hardcoding `"stop"` — which this did — told every client that a turn cut
/// off by the Azure OpenAI content filter, or by the token budget, had ended
/// normally. Those are precisely the two cases an OpenAI-compatible client
/// branches on, and neither is distinguishable from the text alone.
///
/// The wire field is a **closed** set (`stop` / `length` / `tool_calls` /
/// `content_filter` / `function_call`), and the core vocabulary now names
/// all five. Three of them pass straight through.
///
/// The two tool reasons are a *promise* that `message.tool_calls` carries a
/// call to execute, so they pass through only when `has_tool_calls` says the
/// response actually carries one. A turn that reports `tool_calls` while
/// declaring none — a provider quirk, or a message whose call content got
/// dropped upstream — degrades to `stop` with the real reason preserved,
/// rather than instructing a client to run something it cannot see.
///
/// A provider-specific reason cannot pass through — Anthropic's converter
/// deliberately preserves `model_context_window_exceeded` — because a
/// generated client whose enum is strict can reject the entire response
/// over a value outside that set, which costs the caller far more than the
/// reason is worth.
///
/// So an unfamiliar reason is reported as `length` and carried verbatim in
/// the non-standard `x_finish_reason` beside it. `length` is a deliberate
/// approximation: it is the only legal value that says "you did not get a
/// complete answer" without asserting a specific cause the way
/// `content_filter` would, and it is accurate for the truncation-shaped
/// reasons providers actually emit here. The earlier objection to a wrong
/// familiar value was that it *lost* the real one — which it no longer does.
fn finish_reason_of(
    finish_reason: Option<&FinishReason>,
    has_tool_calls: bool,
) -> (&'static str, Option<&str>) {
    let Some(reason) = finish_reason.map(FinishReason::as_str) else {
        return (FinishReason::STOP, None);
    };
    match reason {
        FinishReason::STOP => (FinishReason::STOP, None),
        FinishReason::LENGTH => (FinishReason::LENGTH, None),
        FinishReason::CONTENT_FILTER => (FinishReason::CONTENT_FILTER, None),
        // A tool reason is a *promise*: it tells the client to go and
        // execute the call in `message.tool_calls`. `completion_object`
        // now serializes those, so the promise passes through whenever the
        // response actually carries a call. Without one, advertising it
        // hands the client an instruction with no id, name or arguments to
        // act on — worse than under-reporting — so it degrades to `stop`
        // with the real reason alongside.
        FinishReason::TOOL_CALLS if has_tool_calls => (FinishReason::TOOL_CALLS, None),
        FinishReason::FUNCTION_CALL if has_tool_calls => (FinishReason::FUNCTION_CALL, None),
        FinishReason::TOOL_CALLS | FinishReason::FUNCTION_CALL => {
            (FinishReason::STOP, Some(reason))
        }
        other => (FinishReason::LENGTH, Some(other)),
    }
}

/// The streaming `tool_calls` deltas for a finished stream, one per
/// outstanding call, in the order OpenAI keys fragments by.
///
/// Each call gets a single delta carrying its complete arguments, since
/// [`util::StreamingCalls`] reassembled the fragments before the client
/// could see them. The `index` is still what correlates a delta with a
/// call, and is still what a client accumulating by index expects.
fn tool_call_deltas(calls: &util::StreamingCalls) -> Vec<Value> {
    calls
        .outstanding()
        .into_iter()
        .enumerate()
        .map(|(index, (call_id, name, arguments))| {
            json!({
                "index": index,
                "id": call_id,
                "type": "function",
                "function": { "name": name, "arguments": arguments },
            })
        })
        .collect()
}

/// Build one streaming `chat.completion.chunk` with the given `delta` and
/// `finish_reason`.
fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Value) -> Value {
    chunk_with_raw_reason(id, created, model, delta, finish, None)
}

/// [`chunk`], plus the provider's own finish reason when `finish_reason` had
/// to be approximated to stay inside the closed enum. Omitted otherwise, so
/// an ordinary chunk is byte-identical to before.
fn chunk_with_raw_reason(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish: Value,
    raw: Option<String>,
) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
            "x_finish_reason": raw,
        }],
    })
}

/// An in-band error frame for a failure that occurs after streaming began.
fn stream_error(message: &str) -> Value {
    json!({
        "error": {
            "message": format!("Agent execution failed: {message}"),
            "type": "server_error",
            "code": null,
        }
    })
}

fn error_response(message: String) -> Response {
    (
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "error": {
                "message": format!("Agent execution failed: {message}"),
                "type": "server_error",
                "code": null,
            }
        })),
    )
        .into_response()
}

/// Convert incoming OpenAI chat messages to core [`Message`]s.
///
/// Text alone is not enough once this surface advertises tool calls: the
/// client's follow-up replays the assistant turn with `tool_calls` and adds
/// a `role: "tool"` message per result, and every provider converter builds
/// its wire tool messages from [`FunctionResultContent`] rather than from
/// tool-role text. Reading only `content` turned that result into an empty
/// tool message and lost the correlation, so a call advertised here could
/// never be completed.
fn to_chat_messages(messages: &[IncomingMessage]) -> Vec<Message> {
    messages.iter().map(to_chat_message).collect()
}

fn to_chat_message(m: &IncomingMessage) -> Message {
    let role = role_from(&m.role);
    let text = content_text(&m.content);
    let mut contents = Vec::new();

    // A tool message's `content` *is* the result, so it becomes the result
    // rather than text beside it.
    if let Some(call_id) = &m.tool_call_id {
        contents.push(Content::FunctionResult(FunctionResultContent::new(
            call_id.clone(),
            Some(function_result_value(&m.content)),
        )));
    } else if !text.is_empty() {
        contents.push(Content::text(text));
    }

    contents.extend(m.tool_calls.iter().flatten().map(|call| {
        Content::FunctionCall(FunctionCallContent::new(
            call.id.clone(),
            call.function.name.clone(),
            call.function.arguments.clone().map(FunctionArguments::Raw),
        ))
    }));

    // An assistant turn that was only a tool call arrives as
    // `content: null`, and a message with no contents at all would be
    // dropped by some converters.
    if contents.is_empty() {
        contents.push(Content::text(String::new()));
    }
    Message {
        role,
        contents,
        author_name: None,
        message_id: None,
        additional_properties: Default::default(),
    }
}

/// A tool message's content as a result value.
///
/// OpenAI types the field as a string, and tools overwhelmingly return JSON
/// in it, so a string that parses is stored as the structured value it
/// represents and anything else stays a string. Storing the raw text
/// unconditionally would make every result an opaque blob to a tool
/// expecting its own shape back.
fn function_result_value(content: &Value) -> Value {
    match content {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Value::Null => Value::String(String::new()),
        other => other.clone(),
    }
}

fn role_from(role: &str) -> Role {
    match role {
        "user" => Role::user(),
        "assistant" => Role::assistant(),
        "system" => Role::system(),
        "tool" => Role::tool(),
        other => Role::new(other),
    }
}

/// Extract text from an OpenAI chat `content` (string or array of parts).
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>()
            .join(""),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// Resolve `(prompt_tokens, completion_tokens)` from usage or an estimate.
fn token_counts(usage: &Option<UsageDetails>, input_len: usize, output_len: usize) -> (u64, u64) {
    match usage {
        Some(u) => (
            u.input_token_count.unwrap_or((input_len / 4) as u64),
            u.output_token_count.unwrap_or((output_len / 4) as u64),
        ),
        None => ((input_len / 4) as u64, (output_len / 4) as u64),
    }
}
