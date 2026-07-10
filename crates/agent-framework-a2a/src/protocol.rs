//! JSON-RPC 2.0 envelope helpers, tolerant parsing for the `message/send` /
//! `message/stream` result unions, and Server-Sent-Events framing for
//! `message/stream`.

use serde_json::{json, Value};

use agent_framework_core::error::{Error, Result};

use crate::types::{Message, MessageStreamEvent, SendMessageResult, Task};

/// Build a JSON-RPC 2.0 request envelope:
/// `{"jsonrpc":"2.0","id":..,"method":..,"params":..}`.
pub fn build_request(id: &str, method: &str, params: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": method,
        "params": params,
    })
}

/// A JSON-RPC error object (the `error` field of a response).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    #[serde(default)]
    pub data: Option<Value>,
}

/// Extract the `result` payload from a JSON-RPC response body, mapping an
/// `{"error": ...}` envelope to [`Error::Service`] carrying the JSON-RPC
/// code and message.
pub fn extract_result(value: &Value) -> Result<Value> {
    if let Some(err) = value.get("error") {
        let rpc_err: RpcError = serde_json::from_value(err.clone()).unwrap_or(RpcError {
            code: -32603,
            message: "unrecognized A2A error shape".to_string(),
            data: Some(err.clone()),
        });
        return Err(Error::service(format!(
            "A2A error {}: {}",
            rpc_err.code, rpc_err.message
        )));
    }
    value
        .get("result")
        .cloned()
        .ok_or_else(|| Error::service("A2A response missing both 'result' and 'error'"))
}

/// Parse a `message/send` result value into a [`SendMessageResult`].
///
/// Prefers the spec's `kind` discriminator; if a server omits it (some
/// minimal/non-SDK implementations do, since `kind` was a later spec
/// addition), falls back to shape-based inference: a `status` field means a
/// [`Task`], otherwise a [`Message`].
pub fn parse_send_message_result(value: &Value) -> Result<SendMessageResult> {
    if let Ok(tagged) = serde_json::from_value::<SendMessageResult>(value.clone()) {
        return Ok(tagged);
    }
    if value.get("status").is_some() {
        return serde_json::from_value::<Task>(value.clone())
            .map(SendMessageResult::Task)
            .map_err(|e| Error::Serialization(format!("invalid A2A Task: {e}")));
    }
    serde_json::from_value::<Message>(value.clone())
        .map(SendMessageResult::Message)
        .map_err(|e| Error::Serialization(format!("invalid A2A Message: {e}")))
}

/// Parse a `message/stream` event value into a [`MessageStreamEvent`], with
/// the same `kind`-first / shape-fallback strategy as
/// [`parse_send_message_result`].
pub fn parse_stream_event(value: &Value) -> Result<MessageStreamEvent> {
    if let Ok(tagged) = serde_json::from_value::<MessageStreamEvent>(value.clone()) {
        return Ok(tagged);
    }
    let has = |key: &str| value.get(key).is_some();
    if has("artifact") {
        return serde_json::from_value(value.clone())
            .map(MessageStreamEvent::ArtifactUpdate)
            .map_err(|e| {
                Error::Serialization(format!("invalid A2A TaskArtifactUpdateEvent: {e}"))
            });
    }
    if has("status") {
        if has("taskId") {
            return serde_json::from_value(value.clone())
                .map(MessageStreamEvent::StatusUpdate)
                .map_err(|e| {
                    Error::Serialization(format!("invalid A2A TaskStatusUpdateEvent: {e}"))
                });
        }
        return serde_json::from_value::<Task>(value.clone())
            .map(MessageStreamEvent::Task)
            .map_err(|e| Error::Serialization(format!("invalid A2A Task: {e}")));
    }
    serde_json::from_value::<Message>(value.clone())
        .map(MessageStreamEvent::Message)
        .map_err(|e| Error::Serialization(format!("invalid A2A Message: {e}")))
}

/// Drain every complete `\n\n`-delimited SSE event currently in `buf`,
/// parsing each event's joined `data:` lines as a JSON-RPC response and
/// mapping its `result` (or `error`) into a [`MessageStreamEvent`].
///
/// Consumes complete events from the front of `buf`, leaving any trailing
/// partial event in place for the next call once more bytes have arrived.
/// A frame with no `data:` line, or whose data isn't valid JSON, is logged
/// and skipped rather than surfaced as a stream error — a well-behaved A2A
/// server should never send one, but an SSE comment/keep-alive frame
/// legitimately might.
pub fn drain_sse_events(buf: &mut String) -> Vec<Result<MessageStreamEvent>> {
    let mut out = Vec::new();
    loop {
        let Some(pos) = buf.find("\n\n") else {
            break;
        };
        let event_text = buf[..pos].to_string();
        buf.drain(..pos + 2);
        if let Some(parsed) = parse_sse_event_text(&event_text) {
            out.push(parsed);
        }
    }
    out
}

/// Parse one SSE event block's `data:` line(s) (joined, per the SSE spec) as
/// a JSON-RPC message; `None` means "nothing to yield" (nothing that looked
/// like a `data:` field, or it didn't parse as JSON).
fn parse_sse_event_text(event_text: &str) -> Option<Result<MessageStreamEvent>> {
    let mut data = String::new();
    for line in event_text.split('\n') {
        let line = line.trim_end_matches('\r');
        if let Some(rest) = line.strip_prefix("data:") {
            let piece = rest.strip_prefix(' ').unwrap_or(rest);
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(piece);
        }
        // Other SSE fields (event:, id:, retry:, `:`-prefixed comments) carry
        // no JSON-RPC framing information and are ignored.
    }
    if data.is_empty() {
        return None;
    }
    let value: Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, data = %data, "A2A: unparseable SSE event data");
            return None;
        }
    };
    let result = match extract_result(&value) {
        Ok(r) => r,
        Err(e) => return Some(Err(e)),
    };
    Some(parse_stream_event(&result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::MessageRole;

    #[test]
    fn build_request_has_jsonrpc_envelope_shape() {
        let req = build_request("req-1", "message/send", json!({"message": {}}));
        assert_eq!(req["jsonrpc"], "2.0");
        assert_eq!(req["id"], "req-1");
        assert_eq!(req["method"], "message/send");
        assert!(req["params"].is_object());
    }

    #[test]
    fn extract_result_returns_result_value() {
        let body = json!({"jsonrpc": "2.0", "id": "1", "result": {"ok": true}});
        assert_eq!(extract_result(&body).unwrap(), json!({"ok": true}));
    }

    #[test]
    fn extract_result_maps_error_to_service_error_with_code_and_message() {
        let body = json!({
            "jsonrpc": "2.0",
            "id": "1",
            "error": {"code": -32001, "message": "Task not found"}
        });
        let err = extract_result(&body).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("-32001"), "got: {text}");
        assert!(text.contains("Task not found"), "got: {text}");
        assert!(matches!(err, Error::Service(_)));
    }

    #[test]
    fn extract_result_errors_when_neither_result_nor_error_present() {
        let body = json!({"jsonrpc": "2.0", "id": "1"});
        assert!(extract_result(&body).is_err());
    }

    fn sample_message_json() -> Value {
        json!({
            "kind": "message",
            "role": "agent",
            "parts": [{"kind": "text", "text": "hello"}],
            "messageId": "msg-1"
        })
    }

    fn sample_task_json() -> Value {
        json!({
            "kind": "task",
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "completed"}
        })
    }

    #[test]
    fn parse_send_message_result_with_kind_tag_message() {
        let result = parse_send_message_result(&sample_message_json()).unwrap();
        match result {
            SendMessageResult::Message(m) => {
                assert_eq!(m.message_id, "msg-1");
                assert_eq!(m.role, MessageRole::Agent);
            }
            other => panic!("expected Message, got {other:?}"),
        }
    }

    #[test]
    fn parse_send_message_result_with_kind_tag_task() {
        let result = parse_send_message_result(&sample_task_json()).unwrap();
        match result {
            SendMessageResult::Task(t) => assert_eq!(t.id, "task-1"),
            other => panic!("expected Task, got {other:?}"),
        }
    }

    #[test]
    fn parse_send_message_result_without_kind_infers_message() {
        let mut value = sample_message_json();
        value.as_object_mut().unwrap().remove("kind");
        let result = parse_send_message_result(&value).unwrap();
        assert!(matches!(result, SendMessageResult::Message(_)));
    }

    #[test]
    fn parse_send_message_result_without_kind_infers_task_from_status_field() {
        let mut value = sample_task_json();
        value.as_object_mut().unwrap().remove("kind");
        let result = parse_send_message_result(&value).unwrap();
        assert!(matches!(result, SendMessageResult::Task(_)));
    }

    #[test]
    fn parse_stream_event_status_update_without_kind() {
        let value = json!({
            "taskId": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "working"},
            "final": false
        });
        let event = parse_stream_event(&value).unwrap();
        match event {
            MessageStreamEvent::StatusUpdate(u) => {
                assert_eq!(u.task_id, "task-1");
                assert!(!u.is_final);
            }
            other => panic!("expected StatusUpdate, got {other:?}"),
        }
    }

    #[test]
    fn parse_stream_event_artifact_update_without_kind() {
        let value = json!({
            "taskId": "task-1",
            "contextId": "ctx-1",
            "artifact": {"artifactId": "art-1", "parts": []}
        });
        let event = parse_stream_event(&value).unwrap();
        assert!(matches!(event, MessageStreamEvent::ArtifactUpdate(_)));
    }

    #[test]
    fn parse_stream_event_task_without_kind() {
        let mut value = sample_task_json();
        value.as_object_mut().unwrap().remove("kind");
        let event = parse_stream_event(&value).unwrap();
        assert!(matches!(event, MessageStreamEvent::Task(_)));
    }

    #[test]
    fn parse_stream_event_message_without_kind() {
        let mut value = sample_message_json();
        value.as_object_mut().unwrap().remove("kind");
        let event = parse_stream_event(&value).unwrap();
        assert!(matches!(event, MessageStreamEvent::Message(_)));
    }

    #[test]
    fn drain_sse_events_parses_multiple_events_and_retains_partial_trailing_event() {
        let mut buf = format!(
            "data: {}\n\ndata: {}\n\ndata: {{\"jsonrpc\":\"2.0\"",
            json!({"jsonrpc": "2.0", "id": "1", "result": sample_message_json()}),
            json!({"jsonrpc": "2.0", "id": "1", "result": sample_task_json()}),
        );
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 2);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            MessageStreamEvent::Message(_)
        ));
        assert!(matches!(
            events[1].as_ref().unwrap(),
            MessageStreamEvent::Task(_)
        ));
        // The partial trailing event (no closing "\n\n" yet) stays buffered.
        assert!(buf.contains("\"jsonrpc\":\"2.0\""));
    }

    #[test]
    fn drain_sse_events_surfaces_rpc_error_frame() {
        let mut buf = format!(
            "data: {}\n\n",
            json!({"jsonrpc": "2.0", "id": "1", "error": {"code": -32002, "message": "not cancelable"}})
        );
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 1);
        let err = events[0].as_ref().unwrap_err();
        assert!(err.to_string().contains("not cancelable"));
    }

    #[test]
    fn drain_sse_events_skips_comment_only_frame() {
        let mut buf = ": keep-alive\n\n".to_string();
        let events = drain_sse_events(&mut buf);
        assert!(events.is_empty());
        assert!(buf.is_empty());
    }

    #[test]
    fn drain_sse_events_handles_multiline_data_fields() {
        // Per the SSE spec, multiple `data:` lines in one event are joined
        // with `\n` before parsing.
        let mut buf = "data: {\"jsonrpc\":\"2.0\",\ndata: \"id\":\"1\",\"result\":{\"kind\":\"message\",\"role\":\"agent\",\"parts\":[],\"messageId\":\"m1\"}}\n\n".to_string();
        let events = drain_sse_events(&mut buf);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0].as_ref().unwrap(),
            MessageStreamEvent::Message(_)
        ));
    }
}
