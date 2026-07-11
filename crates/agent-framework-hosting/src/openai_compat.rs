//! OpenAI Chat Completions-compatible hosting.
//!
//! Serves one agent at `POST /v1/chat/completions` in the OpenAI
//! `chat.completion` shape, including streaming (`data: {chunk}` /
//! `data: [DONE]`). This lets any OpenAI-Chat client talk to an agent.
//!
//! # Divergences
//! - Streaming is realized by running the agent to completion and then framing
//!   the result as `chat.completion.chunk`s (the core `Agent` trait exposes only
//!   `run`). Chunk framing matches the OpenAI streaming protocol.
//! - `usage` uses the agent's reported token counts when available, otherwise a
//!   ~4-chars-per-token estimate.

use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};

use agent_framework_core::agent::Agent;
use agent_framework_core::types::{AgentRunResponse, ChatMessage, Role, UsageDetails};

use crate::registry::IntoAgentRegistration;
use crate::sse::sse_response;
use crate::util;

/// Serves one agent over the OpenAI Chat Completions API.
pub struct OpenAiRouter {
    model: String,
    agent: Arc<dyn Agent>,
}

impl OpenAiRouter {
    /// Build a chat-completions host for `agent`, advertised under model id
    /// `name`.
    ///
    /// Accepts a [`ChatAgent`](agent_framework_core::agent::ChatAgent), a
    /// [`WorkflowAgent`](agent_framework_core::workflow::WorkflowAgent), or an
    /// `Arc<dyn Agent>`.
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
    agent: Arc<dyn Agent>,
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

    let response = match state.agent.run(messages, None).await {
        Ok(r) => r,
        Err(e) => return error_response(e.to_string()),
    };

    let id = format!("chatcmpl-{}", util::short_hex());
    let created = util::now_ts() as u64;

    if request.stream {
        sse_response(stream_chunks(&response, &id, created, &model))
    } else {
        Json(completion_object(
            &response, &id, created, &model, input_len,
        ))
        .into_response()
    }
}

/// Build the non-streaming `chat.completion` response object.
fn completion_object(
    resp: &AgentRunResponse,
    id: &str,
    created: u64,
    model: &str,
    input_len: usize,
) -> Value {
    let text = resp.text();
    let (prompt, completion) = token_counts(&resp.usage_details, input_len, text.len());
    json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    })
}

/// Build the streaming `chat.completion.chunk` sequence.
fn stream_chunks(resp: &AgentRunResponse, id: &str, created: u64, model: &str) -> Vec<Value> {
    let mut chunks = Vec::new();
    let head = |delta: Value, finish: Value| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    };

    // First chunk carries the assistant role.
    chunks.push(head(json!({ "role": "assistant" }), Value::Null));

    // One content chunk per non-empty message.
    for message in &resp.messages {
        let text = message.text();
        if text.is_empty() {
            continue;
        }
        chunks.push(head(json!({ "content": text }), Value::Null));
    }

    // Terminal chunk.
    chunks.push(head(json!({}), Value::String("stop".to_string())));
    chunks
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

fn to_chat_messages(messages: &[IncomingMessage]) -> Vec<ChatMessage> {
    messages
        .iter()
        .map(|m| ChatMessage::new(role_from(&m.role), content_text(&m.content)))
        .collect()
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
