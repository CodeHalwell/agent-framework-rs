//! Serde models for the DevUI-style API.
//!
//! Field names mirror the Python DevUI server's models
//! (`agent_framework_devui/models`) so responses are interchangeable with the
//! reference implementation. Divergences are documented on the relevant fields.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// `GET /health` payload — mirrors DevUI's `health_check`.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub entities_count: usize,
    pub framework: &'static str,
}

/// Entity descriptor — mirrors DevUI's `EntityInfo`.
///
/// Divergences from DevUI: `null`-valued optional fields are omitted rather than
/// serialized as `null`; `tools`/`model_id` are populated only when cheaply
/// available from the concrete agent type (the core `Agent` trait exposes
/// neither, so both are usually absent — see crate docs).
#[derive(Debug, Clone, Serialize)]
pub struct EntityInfo {
    pub id: String,
    #[serde(rename = "type")]
    pub entity_type: &'static str,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub framework: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<String>>,
    pub metadata: Map<String, Value>,
    pub source: &'static str,
    // Agent-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    // Workflow-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub executors: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_executor_id: Option<String>,
}

/// `GET /v1/entities` payload — mirrors DevUI's `DiscoveryResponse`.
#[derive(Debug, Clone, Serialize)]
pub struct DiscoveryResponse {
    pub entities: Vec<EntityInfo>,
}

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
    /// DevUI's convention is `metadata.entity_id`; we also accept
    /// `extra_body.entity_id` and, finally, the OpenAI `model` field, so a plain
    /// OpenAI client that sets `model` to the entity id works too.
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
