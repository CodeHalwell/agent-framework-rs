//! Streamable HTTP transport: POSTs each JSON-RPC message to a single MCP
//! endpoint, accepting either an `application/json` response body (one
//! response) or a `text/event-stream` body (SSE frames scanned for the
//! response whose `id` matches the request).

use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::Value;
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{self, IdGenerator, IncomingMessage};
use crate::transport::McpTransport;

const SESSION_ID_HEADER: &str = "mcp-session-id";

/// An MCP transport that POSTs JSON-RPC messages to a streamable-HTTP endpoint.
///
/// Captures the `Mcp-Session-Id` response header (typically returned from
/// `initialize`) and replays it on subsequent requests. Standalone GET-based
/// SSE listening (for server-initiated messages outside of a request/response
/// cycle) is not implemented — see the crate docs.
pub struct McpStreamableHttpTransport {
    http: reqwest::Client,
    url: String,
    headers: HeaderMap,
    timeout: Option<Duration>,
    session_id: RwLock<Option<String>>,
    next_id: IdGenerator,
}

impl McpStreamableHttpTransport {
    /// Create a transport posting to `url`, with optional extra headers
    /// (e.g. `Authorization`) and a per-request timeout.
    pub fn new(url: impl Into<String>, headers: HeaderMap, timeout: Option<Duration>) -> Self {
        Self {
            http: reqwest::Client::new(),
            url: url.into(),
            headers,
            timeout,
            session_id: RwLock::new(None),
            next_id: IdGenerator::new(),
        }
    }

    /// Build a [`HeaderMap`] from `(name, value)` pairs, for use with [`Self::new`].
    pub fn header_map(pairs: &[(String, String)]) -> Result<HeaderMap> {
        let mut map = HeaderMap::new();
        for (k, v) in pairs {
            let name = HeaderName::try_from(k.as_str())
                .map_err(|e| Error::Configuration(format!("invalid MCP header name '{k}': {e}")))?;
            let value = HeaderValue::try_from(v.as_str()).map_err(|e| {
                Error::Configuration(format!("invalid MCP header value for '{k}': {e}"))
            })?;
            map.insert(name, value);
        }
        Ok(map)
    }

    /// The `Mcp-Session-Id` captured from a previous response, if any.
    pub async fn session_id(&self) -> Option<String> {
        self.session_id.read().await.clone()
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let mut req = self
            .http
            .post(&self.url)
            .header(ACCEPT, "application/json, text/event-stream")
            .header(CONTENT_TYPE, "application/json")
            .headers(self.headers.clone())
            .json(body);
        if let Some(timeout) = self.timeout {
            req = req.timeout(timeout);
        }
        if let Some(session_id) = self.session_id.read().await.clone() {
            req = req.header(SESSION_ID_HEADER, session_id);
        }
        req.send()
            .await
            .map_err(|e| Error::service(format!("MCP HTTP request failed: {e}")))
    }

    async fn capture_session_id(&self, headers: &HeaderMap) {
        if let Some(v) = headers.get(SESSION_ID_HEADER).and_then(|v| v.to_str().ok()) {
            *self.session_id.write().await = Some(v.to_string());
        }
    }

    fn next_request_id(&self) -> i64 {
        self.next_id.next()
    }
}

#[async_trait]
impl McpTransport for McpStreamableHttpTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_request_id();
        let body = protocol::build_request(id, method, params);
        let resp = self.post(&body).await?;
        self.capture_session_id(resp.headers()).await;

        let status = resp.status();
        let content_type = resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service(format!("MCP HTTP {status}: {text}")));
        }

        if content_type.contains("text/event-stream") {
            read_sse_for_id(resp, id).await
        } else {
            let value: Value = resp
                .json()
                .await
                .map_err(|e| Error::service(format!("invalid MCP JSON response: {e}")))?;
            extract_json_response(&value, id)
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let body = protocol::build_notification(method, params);
        let resp = self.post(&body).await?;
        self.capture_session_id(resp.headers()).await;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service(format!("MCP HTTP {status}: {text}")));
        }
        Ok(())
    }

    async fn close(&self) -> Result<()> {
        if let Some(session_id) = self.session_id().await {
            // Best effort: the server may not support/require an explicit
            // session teardown, so failures here are not propagated.
            let result = self
                .http
                .delete(&self.url)
                .header(SESSION_ID_HEADER, session_id)
                .headers(self.headers.clone())
                .send()
                .await;
            if let Err(e) = result {
                tracing::debug!(error = %e, "MCP: best-effort session DELETE failed");
            }
        }
        Ok(())
    }
}

/// Extract the JSON-RPC response matching `expected_id` from a single
/// `application/json` response body.
fn extract_json_response(value: &Value, expected_id: i64) -> Result<Value> {
    match protocol::parse_incoming(value.clone()) {
        IncomingMessage::Response { id, result } if id == expected_id => match result {
            Ok(v) => Ok(v),
            Err(e) => Err(Error::service(e.to_string())),
        },
        IncomingMessage::Response { id, .. } => Err(Error::service(format!(
            "MCP response id mismatch: expected {expected_id}, got {id}"
        ))),
        _ => Err(Error::service(
            "MCP HTTP JSON response body was not a JSON-RPC response",
        )),
    }
}

/// Incrementally read a `text/event-stream` response body, returning as soon
/// as the JSON-RPC response matching `expected_id` is seen (other frames,
/// e.g. server notifications, are logged and skipped). Re-scans the
/// accumulated buffer via [`parse_sse_buffer`] as each chunk arrives — simple
/// and plenty fast for the small, infrequent bodies MCP responses produce.
async fn read_sse_for_id(resp: reqwest::Response, expected_id: i64) -> Result<Value> {
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    loop {
        if let Some(result) = parse_sse_buffer(&buf, expected_id) {
            return result;
        }
        match stream.next().await {
            Some(Ok(bytes)) => buf.push_str(&String::from_utf8_lossy(&bytes)),
            Some(Err(e)) => return Err(Error::service(format!("MCP SSE stream error: {e}"))),
            None => break,
        }
    }
    Err(Error::service(format!(
        "MCP SSE stream ended without a response for request id {expected_id}"
    )))
}

/// Parse a buffer of `text/event-stream` bytes (one or more `\n\n`-separated
/// events) for the JSON-RPC response matching `expected_id`. Pure/sync, so it
/// doubles as the incremental scanner above and as something unit-testable
/// against fixture strings without any I/O.
pub(crate) fn parse_sse_buffer(buf: &str, expected_id: i64) -> Option<Result<Value>> {
    for event_text in buf.split("\n\n") {
        if event_text.trim().is_empty() {
            continue;
        }
        if let Some(result) = extract_sse_event(event_text, expected_id) {
            return Some(result);
        }
    }
    None
}

/// Parse one SSE event block (its `data:` line(s), joined) as a JSON-RPC
/// message; returns `Some` only if it is the response matching `expected_id`.
fn extract_sse_event(event_text: &str, expected_id: i64) -> Option<Result<Value>> {
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
        // Other SSE fields (event:, id:, retry:, comments starting with ':')
        // are not meaningful for JSON-RPC framing and are ignored.
    }
    if data.is_empty() {
        return None;
    }
    let value: Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(error = %e, data = %data, "MCP: unparseable SSE event data");
            return None;
        }
    };
    match protocol::parse_incoming(value) {
        IncomingMessage::Response { id, result } if id == expected_id => Some(match result {
            Ok(v) => Ok(v),
            Err(e) => Err(Error::service(e.to_string())),
        }),
        IncomingMessage::Response { .. } => None,
        IncomingMessage::Notification { method, params } => {
            tracing::debug!(method = %method, params = %params, "MCP SSE: server notification");
            None
        }
        IncomingMessage::ServerRequest { id, method, params } => {
            tracing::warn!(
                id = %id,
                method = %method,
                params = %params,
                "MCP SSE: server-initiated request ignored (unsupported)"
            );
            None
        }
        IncomingMessage::Malformed(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn json_branch_extracts_matching_response() {
        let body = json!({"jsonrpc":"2.0","id":1,"result":{"tools":[]}});
        let value = extract_json_response(&body, 1).unwrap();
        assert_eq!(value, json!({"tools": []}));
    }

    #[test]
    fn json_branch_rejects_mismatched_id() {
        let body = json!({"jsonrpc":"2.0","id":2,"result":{}});
        let err = extract_json_response(&body, 1).unwrap_err();
        assert!(err.to_string().contains("id mismatch"));
    }

    #[test]
    fn json_branch_surfaces_rpc_error() {
        let body = json!({"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"bad params"}});
        let err = extract_json_response(&body, 1).unwrap_err();
        assert!(err.to_string().contains("bad params"));
    }

    #[test]
    fn sse_buffer_finds_response_after_a_notification() {
        let buf = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\"}}\n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{\"ok\":true}}\n",
            "\n",
        );
        let result = parse_sse_buffer(buf, 5)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn sse_buffer_handles_multiline_data_fields() {
        // Per the SSE spec, multiple `data:` lines in one event are joined with `\n`.
        let buf = "data: {\"jsonrpc\":\"2.0\",\ndata: \"id\":9,\"result\":{}}\n\n";
        let result = parse_sse_buffer(buf, 9)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn sse_buffer_returns_none_when_id_never_appears() {
        let buf = "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n";
        assert!(parse_sse_buffer(buf, 999).is_none());
    }

    #[test]
    fn sse_buffer_surfaces_error_response() {
        let buf =
            "data: {\"jsonrpc\":\"2.0\",\"id\":4,\"error\":{\"code\":-1,\"message\":\"nope\"}}\n\n";
        let result = parse_sse_buffer(buf, 4).expect("should find response");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn header_map_builds_from_pairs() {
        let map = McpStreamableHttpTransport::header_map(&[(
            "Authorization".to_string(),
            "Bearer x".to_string(),
        )])
        .unwrap();
        assert_eq!(map.get("authorization").unwrap(), "Bearer x");
    }

    #[test]
    fn header_map_rejects_invalid_header_value() {
        let err = McpStreamableHttpTransport::header_map(&[(
            "X-Test".to_string(),
            "bad\nvalue".to_string(),
        )])
        .unwrap_err();
        match err {
            Error::Configuration(_) => {}
            other => panic!("expected Configuration error, got {other:?}"),
        }
    }
}
