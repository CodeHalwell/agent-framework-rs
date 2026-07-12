//! Reusable OpenAI-Responses request/response conversion.
//!
//! Framework-agnostic OpenAI-Responses-shape types plus the two conversion
//! functions that translate between them and `agent-framework-core` types:
//! [`responses_to_run`] (request input → [`Message`]s) and
//! [`responses_from_run`] (a completed [`AgentResponse`] → [`ResponseObject`]).
//! Mirrors the Python `hosting-responses` package
//! (`responses_to_run`/`responses_from_run`; UPSTREAM_DRIFT.md §14): a small,
//! standalone conversion surface any host — not just [`crate::devui`] — can use
//! to speak the OpenAI Responses wire shape.
//!
//! [`crate::devui`] is the only current caller; it layers DevUI-specific
//! concerns (entity routing, the `~4-chars-per-token` usage estimate for runs
//! that report no usage, SSE framing) on top of this module.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use agent_framework_core::types::{AgentResponse, Message, Role, UsageDetails};

/// `POST /v1/responses` request — a subset of DevUI's `AgentFrameworkRequest`
/// (itself an OpenAI `ResponseCreateParams` superset). Unknown fields are
/// ignored.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct ResponsesRequest {
    /// OpenAI `model`; accepted as an entity-id fallback.
    #[serde(default)]
    pub model: Option<String>,
    /// The user input: a string, an OpenAI input-items array, or a structured
    /// object (for workflows).
    #[serde(default)]
    pub input: Value,
    /// Whether to stream SSE. DevUI defaults this to `false`.
    #[serde(default)]
    pub stream: bool,
    /// Routing metadata; DevUI reads the entity id from `metadata.entity_id`.
    #[serde(default)]
    pub metadata: Option<Map<String, Value>>,
    /// Advanced routing; `extra_body.entity_id` is accepted as a fallback.
    #[serde(default)]
    pub extra_body: Option<Map<String, Value>>,
}

impl ResponsesRequest {
    /// Resolve the target entity id.
    ///
    /// `metadata.entity_id`/`extra_body.entity_id`/`model`-as-fallback is
    /// [`crate::devui`]'s routing convention, not part of the OpenAI Responses
    /// wire shape; it lives here only because it is an inherent method on
    /// [`ResponsesRequest`], which this module owns.
    pub fn entity_id(&self) -> Option<String> {
        fn get<'a>(m: &'a Option<Map<String, Value>>, k: &str) -> Option<&'a str> {
            m.as_ref().and_then(|m| m.get(k)).and_then(Value::as_str)
        }
        get(&self.metadata, "entity_id")
            .or_else(|| get(&self.extra_body, "entity_id"))
            .map(str::to_string)
            .or_else(|| self.model.clone())
    }
}

/// Per-response token usage — mirrors OpenAI `ResponseUsage`.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens_details: OutputTokensDetails,
}

#[derive(Debug, Clone, Serialize)]
pub struct InputTokensDetails {
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: u64,
}

/// A single `output_text` content part — mirrors OpenAI `ResponseOutputText`.
#[derive(Debug, Clone, Serialize)]
pub struct OutputText {
    #[serde(rename = "type")]
    pub content_type: &'static str,
    pub text: String,
    pub annotations: Vec<Value>,
}

impl OutputText {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content_type: "output_text",
            text: text.into(),
            annotations: Vec::new(),
        }
    }
}

/// An assistant message output item — mirrors OpenAI `ResponseOutputMessage`.
#[derive(Debug, Clone, Serialize)]
pub struct OutputMessage {
    #[serde(rename = "type")]
    pub item_type: &'static str,
    pub id: String,
    pub role: &'static str,
    pub content: Vec<OutputText>,
    pub status: &'static str,
}

impl OutputMessage {
    pub fn assistant_text(id: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            item_type: "message",
            id: id.into(),
            role: "assistant",
            content: vec![OutputText::new(text)],
            status: "completed",
        }
    }
}

/// The aggregated final response — mirrors OpenAI `Response`.
///
/// Divergences from DevUI: adds a convenience top-level `output_text` (the
/// aggregated assistant text); for workflow entities adds `outputs` (the raw
/// workflow output values) and `pending_requests` (outstanding request-info
/// entries), since DevUI's text-only aggregation would otherwise drop them.
#[derive(Debug, Clone, Serialize)]
pub struct ResponseObject {
    pub id: String,
    pub object: &'static str,
    pub created_at: f64,
    pub model: String,
    pub status: &'static str,
    pub output: Vec<OutputMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<Value>>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pending_requests: Vec<Value>,
    pub parallel_tool_calls: bool,
    pub tool_choice: &'static str,
    pub tools: Vec<Value>,
}

impl ResponseObject {
    /// A skeleton `in_progress` response used by the `response.created` /
    /// `response.in_progress` streaming events.
    pub fn in_progress(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            object: "response",
            created_at: crate::util::now_ts(),
            model: model.into(),
            status: "in_progress",
            output: Vec::new(),
            output_text: None,
            usage: None,
            outputs: None,
            pending_requests: Vec::new(),
            parallel_tool_calls: false,
            tool_choice: "none",
            tools: Vec::new(),
        }
    }
}

/// Build an OpenAI-style error body: `{"error": {"message", "type", "code"}}`.
///
/// Mirrors DevUI's `OpenAIError.create`.
pub fn openai_error(message: impl Into<String>, error_type: &str, code: Option<&str>) -> Value {
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": error_type,
            "code": code,
        }
    })
}

// ---------------------------------------------------------------------------
// Conversion: ResponsesRequest -> Vec<Message>
// ---------------------------------------------------------------------------

/// Parse a [`ResponsesRequest`]'s OpenAI-style `input` into chat messages for
/// an agent run. Mirrors `hosting-responses`' `responses_to_run`.
///
/// Accepts a bare string, an array of input items (OpenAI `{type:"message",
/// content:[…]}` or `{role, content}`), or falls back to a stringified value.
pub fn responses_to_run(request: &ResponsesRequest) -> Vec<Message> {
    input_to_messages(&request.input)
}

/// Shared with [`crate::devui`]'s workflow input handling, which needs to
/// convert a bare `input` value (not a full [`ResponsesRequest`]) to messages.
pub(crate) fn input_to_messages(input: &Value) -> Vec<Message> {
    match input {
        Value::String(s) => vec![Message::user(s.clone())],
        Value::Null => vec![Message::user(String::new())],
        Value::Array(items) => {
            let msgs: Vec<Message> = items.iter().filter_map(item_to_message).collect();
            if msgs.is_empty() {
                vec![Message::user(String::new())]
            } else {
                msgs
            }
        }
        obj @ Value::Object(_) => item_to_message(obj)
            .map(|m| vec![m])
            .unwrap_or_else(|| vec![Message::user(obj.to_string())]),
        other => vec![Message::user(other.to_string())],
    }
}

/// Convert one input item into a chat message, if it carries text.
fn item_to_message(item: &Value) -> Option<Message> {
    match item {
        Value::String(s) => Some(Message::user(s.clone())),
        Value::Object(map) => {
            let role = map
                .get("role")
                .and_then(Value::as_str)
                .map(role_from)
                .unwrap_or_else(Role::user);
            let text = map.get("content").map(content_text).unwrap_or_default();
            Some(Message::new(role, text))
        }
        _ => None,
    }
}

/// Extract text from an OpenAI `content` value (string or array of parts).
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| p.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
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

// ---------------------------------------------------------------------------
// Conversion: AgentResponse -> ResponseObject
// ---------------------------------------------------------------------------

/// Build the aggregated OpenAI-Responses output object for a completed
/// [`AgentResponse`]. Mirrors `hosting-responses`' `responses_from_run`.
///
/// Maps `resp`'s text into a single assistant `output`/`output_text`, and its
/// [`UsageDetails`], when present, into [`Usage`]. `id`/`model` are threaded
/// through as-is (callers, e.g. [`crate::devui`], own id generation and
/// entity-to-model resolution). When `resp` carries no usage details, `usage`
/// is `None`; callers that want a token-count estimate (as DevUI does) fill
/// one in afterward.
pub fn responses_from_run(resp: &AgentResponse, id: &str, model: &str) -> ResponseObject {
    let text = resp.text();
    let mid = crate::util::msg_id();
    ResponseObject {
        id: id.to_string(),
        object: "response",
        created_at: crate::util::now_ts(),
        model: model.to_string(),
        status: "completed",
        output: vec![OutputMessage::assistant_text(mid, text.clone())],
        output_text: Some(text),
        usage: resp.usage_details.as_ref().map(usage_from_details),
        outputs: None,
        pending_requests: Vec::new(),
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    }
}

/// Map a core [`UsageDetails`] onto the OpenAI-Responses [`Usage`] shape.
fn usage_from_details(u: &UsageDetails) -> Usage {
    let input = u.input_token_count.unwrap_or(0);
    let output = u.output_token_count.unwrap_or(0);
    let total = u.total_token_count.unwrap_or(input + output);
    Usage {
        input_tokens: input,
        output_tokens: output,
        total_tokens: total,
        input_tokens_details: InputTokensDetails { cached_tokens: 0 },
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(input: Value) -> ResponsesRequest {
        ResponsesRequest {
            input,
            ..Default::default()
        }
    }

    #[test]
    fn entity_id_prefers_metadata_then_extra_body_then_model() {
        let mut r = req(Value::Null);
        r.model = Some("from-model".to_string());
        assert_eq!(r.entity_id(), Some("from-model".to_string()));

        let mut extra = Map::new();
        extra.insert(
            "entity_id".to_string(),
            Value::String("from-extra".to_string()),
        );
        r.extra_body = Some(extra);
        assert_eq!(r.entity_id(), Some("from-extra".to_string()));

        let mut meta = Map::new();
        meta.insert(
            "entity_id".to_string(),
            Value::String("from-meta".to_string()),
        );
        r.metadata = Some(meta);
        assert_eq!(r.entity_id(), Some("from-meta".to_string()));
    }

    #[test]
    fn responses_to_run_string_input() {
        let messages = responses_to_run(&req(Value::String("hello".to_string())));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::user());
        assert_eq!(messages[0].text(), "hello");
    }

    #[test]
    fn responses_to_run_array_input() {
        let input = serde_json::json!([
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": [ { "type": "input_text", "text": "hi" } ] },
        ]);
        let messages = responses_to_run(&req(input));
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::system());
        assert_eq!(messages[0].text(), "be terse");
        assert_eq!(messages[1].role, Role::user());
        assert_eq!(messages[1].text(), "hi");
    }

    #[test]
    fn responses_to_run_null_input_yields_empty_user_message() {
        let messages = responses_to_run(&req(Value::Null));
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, Role::user());
        assert_eq!(messages[0].text(), "");
    }

    #[test]
    fn responses_from_run_maps_text_and_usage() {
        let resp = AgentResponse {
            messages: vec![Message::assistant("hello there")],
            usage_details: Some(UsageDetails {
                input_token_count: Some(3),
                output_token_count: Some(5),
                total_token_count: Some(8),
                ..Default::default()
            }),
            ..Default::default()
        };
        let obj = responses_from_run(&resp, "resp_123", "my-agent");
        assert_eq!(obj.id, "resp_123");
        assert_eq!(obj.model, "my-agent");
        assert_eq!(obj.status, "completed");
        assert_eq!(obj.output_text.as_deref(), Some("hello there"));
        assert_eq!(obj.output.len(), 1);
        assert_eq!(obj.output[0].content[0].text, "hello there");
        let usage = obj.usage.expect("usage present");
        assert_eq!(usage.input_tokens, 3);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 8);
        assert_eq!(usage.input_tokens_details.cached_tokens, 0);
        assert_eq!(usage.output_tokens_details.reasoning_tokens, 0);
    }

    #[test]
    fn responses_from_run_without_usage_details_leaves_usage_none() {
        let resp = AgentResponse {
            messages: vec![Message::assistant("hi")],
            usage_details: None,
            ..Default::default()
        };
        let obj = responses_from_run(&resp, "resp_1", "m");
        assert!(obj.usage.is_none());
    }
}
