//! JSON-RPC 2.0 message shapes and MCP-specific payload types.
//!
//! This is a "typed-enough" layer: request/notification envelopes and the
//! handful of MCP result shapes we care about (`initialize`, `tools/list`,
//! `tools/call`) get real structs; everything else stays as [`Value`] so we
//! don't have to model the entire MCP schema to talk to a server.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicI64, Ordering};

/// The MCP protocol version this client requests during `initialize`.
pub const PROTOCOL_VERSION: &str = "2025-06-18";

/// Protocol versions this client understands well enough to proceed if a
/// server negotiates down to one of them.
pub const COMPATIBLE_PROTOCOL_VERSIONS: &[&str] = &["2025-06-18", "2025-03-26", "2024-11-05"];

/// A monotonically increasing JSON-RPC request id generator, starting at 1.
#[derive(Debug, Default)]
pub struct IdGenerator(AtomicI64);

impl IdGenerator {
    pub fn new() -> Self {
        Self(AtomicI64::new(0))
    }

    /// Return the next id.
    pub fn next(&self) -> i64 {
        self.0.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// Build a JSON-RPC request envelope: `{"jsonrpc":"2.0","id":..,"method":..,"params":..}`.
pub fn build_request(id: i64, method: &str, params: Value) -> Value {
    let mut obj = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
    });
    if !params.is_null() {
        obj["params"] = params;
    }
    obj
}

/// Build a JSON-RPC notification envelope (no `id`): `{"jsonrpc":"2.0","method":..,"params":..}`.
pub fn build_notification(method: &str, params: Value) -> Value {
    let mut obj = json!({
        "jsonrpc": "2.0",
        "method": method,
    });
    if !params.is_null() {
        obj["params"] = params;
    }
    obj
}

/// A JSON-RPC error object (the `error` field of a response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MCP error {}: {}", self.code, self.message)
    }
}

/// A parsed incoming JSON-RPC message, classified by shape.
#[derive(Debug)]
pub enum IncomingMessage {
    /// A response (success or error) correlated to one of our outgoing requests.
    Response {
        id: i64,
        result: Result<Value, RpcError>,
    },
    /// A notification from the server: no `id`, no response expected.
    Notification { method: String, params: Value },
    /// A request FROM the server (e.g. sampling, roots). Not supported by this
    /// client; transports log and ignore these.
    ServerRequest {
        id: Value,
        method: String,
        params: Value,
    },
    /// Valid JSON that didn't match any recognized JSON-RPC shape.
    Malformed(Value),
}

/// Classify a raw JSON value as a JSON-RPC response, notification, or
/// server-initiated request.
pub fn parse_incoming(value: Value) -> IncomingMessage {
    let has_result_or_error = value.get("result").is_some() || value.get("error").is_some();
    if has_result_or_error {
        if let Some(id) = value.get("id").and_then(Value::as_i64) {
            if let Some(err) = value.get("error") {
                let rpc_err: RpcError = serde_json::from_value(err.clone()).unwrap_or(RpcError {
                    code: -32603,
                    message: "unrecognized error shape from MCP server".to_string(),
                    data: Some(err.clone()),
                });
                return IncomingMessage::Response {
                    id,
                    result: Err(rpc_err),
                };
            }
            let result = value.get("result").cloned().unwrap_or(Value::Null);
            return IncomingMessage::Response {
                id,
                result: Ok(result),
            };
        }
        return IncomingMessage::Malformed(value);
    }
    if let Some(method) = value.get("method").and_then(Value::as_str) {
        let params = value.get("params").cloned().unwrap_or(Value::Null);
        if let Some(id) = value.get("id").cloned() {
            return IncomingMessage::ServerRequest {
                id,
                method: method.to_string(),
                params,
            };
        }
        return IncomingMessage::Notification {
            method: method.to_string(),
            params,
        };
    }
    IncomingMessage::Malformed(value)
}

/// `clientInfo`/`serverInfo`: an MCP implementation's name and version.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Implementation {
    pub name: String,
    #[serde(default)]
    pub version: String,
}

/// The result of a successful `initialize` request.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    pub protocol_version: String,
    #[serde(default)]
    pub capabilities: Value,
    pub server_info: Implementation,
    #[serde(default)]
    pub instructions: Option<String>,
}

/// A tool descriptor as returned by `tools/list`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolDescriptor {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default = "empty_object_schema")]
    pub input_schema: Value,
    #[serde(default)]
    pub output_schema: Option<Value>,
}

fn empty_object_schema() -> Value {
    json!({ "type": "object", "properties": {} })
}

/// One page of `tools/list` results.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ListToolsResult {
    #[serde(default)]
    pub tools: Vec<ToolDescriptor>,
    #[serde(default)]
    pub next_cursor: Option<String>,
}

/// One content block inside a `tools/call` result.
///
/// MCP defines more block kinds than this (annotations, etc.); anything we
/// don't specifically model is preserved verbatim in [`ContentBlock::Unknown`]
/// so no information is silently dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text(String),
    Image {
        data: String,
        mime_type: String,
    },
    Audio {
        data: String,
        mime_type: String,
    },
    Resource(Value),
    ResourceLink {
        uri: String,
        mime_type: Option<String>,
        name: Option<String>,
    },
    Unknown(Value),
}

impl ContentBlock {
    /// Parse a single content block from its raw JSON representation.
    pub fn from_value(v: &Value) -> ContentBlock {
        match v.get("type").and_then(Value::as_str) {
            Some("text") => ContentBlock::Text(
                v.get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            ),
            Some("image") => ContentBlock::Image {
                data: v
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            Some("audio") => ContentBlock::Audio {
                data: v
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
            },
            Some("resource") => {
                ContentBlock::Resource(v.get("resource").cloned().unwrap_or(Value::Null))
            }
            Some("resource_link") => ContentBlock::ResourceLink {
                uri: v
                    .get("uri")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                mime_type: v
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                name: v.get("name").and_then(Value::as_str).map(str::to_string),
            },
            _ => ContentBlock::Unknown(v.clone()),
        }
    }

    /// Reconstruct the block's JSON representation (used when preserving
    /// multi-block results as a structured array).
    pub fn to_json(&self) -> Value {
        match self {
            ContentBlock::Text(text) => json!({ "type": "text", "text": text }),
            ContentBlock::Image { data, mime_type } => {
                json!({ "type": "image", "data": data, "mimeType": mime_type })
            }
            ContentBlock::Audio { data, mime_type } => {
                json!({ "type": "audio", "data": data, "mimeType": mime_type })
            }
            ContentBlock::Resource(resource) => json!({ "type": "resource", "resource": resource }),
            ContentBlock::ResourceLink {
                uri,
                mime_type,
                name,
            } => {
                let mut obj = json!({ "type": "resource_link", "uri": uri });
                if let Some(m) = mime_type {
                    obj["mimeType"] = json!(m);
                }
                if let Some(n) = name {
                    obj["name"] = json!(n);
                }
                obj
            }
            ContentBlock::Unknown(v) => v.clone(),
        }
    }
}

/// The result of a `tools/call` request.
#[derive(Debug, Clone, Default)]
pub struct CallToolResult {
    pub content: Vec<ContentBlock>,
    pub is_error: bool,
    pub structured_content: Option<Value>,
}

impl CallToolResult {
    /// Parse a `tools/call` result from its raw JSON representation.
    pub fn from_value(v: &Value) -> Self {
        let content = v
            .get("content")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().map(ContentBlock::from_value).collect())
            .unwrap_or_default();
        let is_error = v.get("isError").and_then(Value::as_bool).unwrap_or(false);
        let structured_content = v.get("structuredContent").cloned();
        Self {
            content,
            is_error,
            structured_content,
        }
    }

    /// Map the result's content into a single JSON value for handing back to
    /// a model, independent of whether it was an error:
    ///
    /// - A single text block becomes a JSON string, or the value it parses as
    ///   if the text itself is valid JSON.
    /// - No content but a `structuredContent` payload returns that payload.
    /// - Anything else (zero or multiple / non-text blocks) becomes a JSON
    ///   array preserving each block's shape.
    pub fn to_value(&self) -> Value {
        if self.content.len() == 1 {
            if let ContentBlock::Text(text) = &self.content[0] {
                return match serde_json::from_str::<Value>(text) {
                    Ok(parsed) => parsed,
                    Err(_) => Value::String(text.clone()),
                };
            }
        }
        if self.content.is_empty() {
            if let Some(structured) = &self.structured_content {
                return structured.clone();
            }
            return Value::Null;
        }
        Value::Array(self.content.iter().map(ContentBlock::to_json).collect())
    }

    /// A human-readable message extracted from the text blocks, used when
    /// `is_error` is true and we need a message for [`agent_framework_core::error::Error::Tool`].
    pub fn error_message(&self) -> String {
        let joined = self
            .content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if joined.is_empty() {
            "MCP tool call reported an error".to_string()
        } else {
            joined
        }
    }
}

/// Normalize an MCP tool/prompt name to the identifier pattern most model
/// providers require (`A-Za-z0-9_.-`), replacing any other character with `-`.
///
/// Mirrors `_normalize_mcp_name` in the Python reference implementation.
pub fn normalize_mcp_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_generator_starts_at_one_and_increments() {
        let gen = IdGenerator::new();
        assert_eq!(gen.next(), 1);
        assert_eq!(gen.next(), 2);
        assert_eq!(gen.next(), 3);
    }

    #[test]
    fn build_request_includes_params_when_present() {
        let req = build_request(7, "tools/call", json!({"name": "echo"}));
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], 7);
        assert_eq!(req["method"], "tools/call");
        assert_eq!(req["params"]["name"], "echo");
    }

    #[test]
    fn build_request_omits_null_params() {
        let req = build_request(1, "ping", Value::Null);
        assert!(req.get("params").is_none());
    }

    #[test]
    fn build_notification_has_no_id() {
        let note = build_notification("notifications/initialized", json!({}));
        assert!(note.get("id").is_none());
        assert_eq!(note["method"], "notifications/initialized");
    }

    #[test]
    fn parse_incoming_classifies_response() {
        let msg = json!({"jsonrpc":"2.0","id":3,"result":{"ok":true}});
        match parse_incoming(msg) {
            IncomingMessage::Response { id, result } => {
                assert_eq!(id, 3);
                assert_eq!(result.unwrap(), json!({"ok": true}));
            }
            other => panic!("expected Response, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_error_response() {
        let msg = json!({"jsonrpc":"2.0","id":3,"error":{"code":-32601,"message":"not found"}});
        match parse_incoming(msg) {
            IncomingMessage::Response { id, result } => {
                assert_eq!(id, 3);
                let err = result.unwrap_err();
                assert_eq!(err.code, -32601);
                assert_eq!(err.message, "not found");
            }
            other => panic!("expected Response, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_notification() {
        let msg =
            json!({"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info"}});
        match parse_incoming(msg) {
            IncomingMessage::Notification { method, params } => {
                assert_eq!(method, "notifications/message");
                assert_eq!(params["level"], "info");
            }
            other => panic!("expected Notification, got {other:?}"),
        }
    }

    #[test]
    fn parse_incoming_classifies_server_request() {
        let msg =
            json!({"jsonrpc":"2.0","id":"srv-1","method":"sampling/createMessage","params":{}});
        match parse_incoming(msg) {
            IncomingMessage::ServerRequest { method, .. } => {
                assert_eq!(method, "sampling/createMessage");
            }
            other => panic!("expected ServerRequest, got {other:?}"),
        }
    }

    #[test]
    fn call_tool_result_single_text_block_becomes_string() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "hello world"}],
            "isError": false,
        }));
        assert_eq!(result.to_value(), json!("hello world"));
    }

    #[test]
    fn call_tool_result_single_text_block_parses_json() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "42"}],
            "isError": false,
        }));
        assert_eq!(result.to_value(), json!(42));
    }

    #[test]
    fn call_tool_result_multi_block_preserves_structure() {
        let result = CallToolResult::from_value(&json!({
            "content": [
                {"type": "text", "text": "first"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
            ],
            "isError": false,
        }));
        assert_eq!(
            result.to_value(),
            json!([
                {"type": "text", "text": "first"},
                {"type": "image", "data": "aGk=", "mimeType": "image/png"},
            ])
        );
    }

    #[test]
    fn call_tool_result_is_error_flag_parsed() {
        let result = CallToolResult::from_value(&json!({
            "content": [{"type": "text", "text": "boom"}],
            "isError": true,
        }));
        assert!(result.is_error);
        assert_eq!(result.error_message(), "boom");
    }

    #[test]
    fn call_tool_result_empty_content_falls_back_to_structured_content() {
        let result = CallToolResult::from_value(&json!({
            "content": [],
            "structuredContent": {"count": 3},
        }));
        assert_eq!(result.to_value(), json!({"count": 3}));
    }

    #[test]
    fn normalize_mcp_name_replaces_disallowed_chars() {
        assert_eq!(
            normalize_mcp_name("weather/get current"),
            "weather-get-current"
        );
        assert_eq!(normalize_mcp_name("valid_Name-1.0"), "valid_Name-1.0");
    }
}
