//! Streamable HTTP transport: POSTs each JSON-RPC message to a single MCP
//! endpoint, accepting either an `application/json` response body (one
//! response) or a `text/event-stream` body (SSE frames scanned for the
//! response whose `id` matches the request).

use std::sync::Mutex as StdMutex;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, ACCEPT, CONTENT_TYPE};
use serde_json::{json, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;

use crate::protocol::{self, IdGenerator, IncomingMessage, RpcError};
use crate::sampling::{BoxedNotificationHandler, BoxedServerRequestHandler};
use crate::transport::McpTransport;

const SESSION_ID_HEADER: &str = "mcp-session-id";

/// An MCP transport that POSTs JSON-RPC messages to a streamable-HTTP endpoint.
///
/// Captures the `Mcp-Session-Id` response header (typically returned from
/// `initialize`) and replays it on subsequent requests. A server-initiated
/// request embedded in the SSE response to an active `call()` is routed to
/// whatever handler [`Self::set_server_request_handler`] installed, and
/// answered with a best-effort follow-up POST to the same endpoint (there is
/// no persistent duplex connection to write a response over directly, unlike
/// the stdio/websocket transports). A notification (e.g.
/// `notifications/tools/list_changed`) embedded the same way is routed to
/// whatever handler [`Self::set_notification_handler`] installed; unlike a
/// request, nothing is written back. Standalone GET-based SSE listening (for
/// server-initiated messages outside of any request/response cycle) is not
/// implemented — see the crate docs. This means an HTTP-transported MCP
/// server's `list_changed` notification is only ever noticed if it happens
/// to arrive embedded in the SSE response to a call already in flight.
pub struct McpStreamableHttpTransport {
    http: reqwest::Client,
    url: String,
    headers: HeaderMap,
    timeout: Option<Duration>,
    session_id: RwLock<Option<String>>,
    next_id: IdGenerator,
    server_request_handler: StdMutex<Option<BoxedServerRequestHandler>>,
    /// Handler for server notifications (e.g. `notifications/tools/list_changed`),
    /// installed via [`McpTransport::set_notification_handler`]. Only ever
    /// invoked for a notification embedded in the SSE response to an active
    /// [`McpTransport::call`] — see the crate docs on standalone GET-based
    /// SSE listening not being implemented.
    notification_handler: StdMutex<Option<BoxedNotificationHandler>>,
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
            server_request_handler: StdMutex::new(None),
            notification_handler: StdMutex::new(None),
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
            self.read_sse_for_id(resp, id).await
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

    fn set_server_request_handler(&self, handler: BoxedServerRequestHandler) {
        *self.server_request_handler.lock().unwrap() = Some(handler);
    }

    fn set_notification_handler(&self, handler: BoxedNotificationHandler) {
        *self.notification_handler.lock().unwrap() = Some(handler);
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

impl McpStreamableHttpTransport {
    /// Incrementally read a `text/event-stream` response body, returning as
    /// soon as the JSON-RPC response matching `expected_id` is seen.
    ///
    /// Uses an [`SseDecoder`] that frames events on any SSE line-delimiter
    /// (`\n\n`, `\r\n\r\n`, or `\r\r`), emits each complete event exactly once,
    /// and drains consumed bytes so the working buffer stays small (no
    /// full-buffer rescans, no unbounded growth, and hard caps on event/total
    /// size). Every event is dispatched exactly once as it completes, so no
    /// separate dedup bookkeeping is needed.
    ///
    /// All events produced by a single chunk are processed *before* returning,
    /// so a server-initiated request or notification landing in the same chunk
    /// as the expected response is still dispatched — the response is only
    /// returned after that chunk's events have all been handled.
    async fn read_sse_for_id(&self, resp: reqwest::Response, expected_id: i64) -> Result<Value> {
        let mut stream = resp.bytes_stream();
        let mut utf8 = Utf8StreamDecoder::new();
        let mut decoder = SseDecoder::new();
        loop {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    let decoded = utf8.push(&bytes);
                    // Process every event this chunk completed, dispatching
                    // server-requests/notifications, and remember the first
                    // matching response — but keep processing the rest of the
                    // chunk's events before returning it.
                    let mut answer: Option<Result<Value>> = None;
                    for event_text in decoder.push(&decoded)? {
                        if let Some(result) =
                            self.classify_sse_event(&event_text, expected_id).await
                        {
                            if answer.is_none() {
                                answer = Some(result);
                            }
                        }
                    }
                    if let Some(result) = answer {
                        return result;
                    }
                }
                Some(Err(e)) => return Err(Error::service(format!("MCP SSE stream error: {e}"))),
                None => {
                    // Flush any trailing event not terminated by a final blank
                    // line (some servers omit it), then stop.
                    for event_text in decoder.flush() {
                        if let Some(result) =
                            self.classify_sse_event(&event_text, expected_id).await
                        {
                            return result;
                        }
                    }
                    break;
                }
            }
        }
        Err(Error::service(format!(
            "MCP SSE stream ended without a response for request id {expected_id}"
        )))
    }

    /// Parse one complete SSE event block and route it: dispatch a
    /// server-initiated request (best-effort follow-up POST) or a notification
    /// to their handlers, and return `Some(result)` only for the JSON-RPC
    /// response matching `expected_id`.
    async fn classify_sse_event(
        &self,
        event_text: &str,
        expected_id: i64,
    ) -> Option<Result<Value>> {
        let value = sse_event_json(event_text)?;
        match protocol::parse_incoming(value) {
            IncomingMessage::Response { id, result } if id == expected_id => Some(match result {
                Ok(v) => Ok(v),
                Err(e) => Err(Error::service(e.to_string())),
            }),
            IncomingMessage::Response { .. } => None,
            IncomingMessage::ServerRequest { id, method, params } => {
                self.respond_to_server_request(id, method, params).await;
                None
            }
            IncomingMessage::Notification { method, params } => {
                let handler = self.notification_handler.lock().unwrap().clone();
                if let Some(handler) = handler {
                    handler(method, params).await;
                }
                None
            }
            IncomingMessage::Malformed(_) => None,
        }
    }

    /// Compute the response to one server-initiated request (via whatever
    /// handler is registered) and send it back with a best-effort follow-up
    /// POST to the same endpoint. Failures (transport-level, or a
    /// non-success HTTP status) are logged, not propagated: this happens
    /// deep inside a `call()` that is itself still waiting on its own
    /// expected response.
    async fn respond_to_server_request(&self, id: Value, method: String, params: Value) {
        let handler = self.server_request_handler.lock().unwrap().clone();
        let result = match handler {
            Some(h) => h(method.clone(), params).await,
            None => Err(RpcError {
                code: -32601,
                message: format!("Method not found: {method}"),
                data: None,
            }),
        };
        let envelope = match result {
            Ok(value) => json!({ "jsonrpc": "2.0", "id": id, "result": value }),
            Err(err) => json!({ "jsonrpc": "2.0", "id": id, "error": err }),
        };
        match self.post(&envelope).await {
            Ok(resp) if !resp.status().is_success() => {
                tracing::debug!(
                    status = %resp.status(),
                    "MCP: server rejected our response to its server-initiated request"
                );
            }
            Err(e) => {
                tracing::debug!(
                    error = %e,
                    "MCP: best-effort response POST for a server-initiated request failed"
                );
            }
            _ => {}
        }
    }
}

/// Default cap on the working (undelimited) buffer: a single event that never
/// completes must not grow without bound.
const DEFAULT_MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
/// Default cap on the total decoded bytes for one response body, so a
/// long-lived / adversarial stream cannot exhaust memory.
const DEFAULT_MAX_TOTAL_BYTES: usize = 64 * 1024 * 1024;

/// An incremental `text/event-stream` decoder.
///
/// Frames events on any SSE line-delimiter — `\n\n`, `\r\n\r\n`, or `\r\r` —
/// (fixing the previous `split("\n\n")`-only framing that dropped CRLF-framed
/// events), emits each completed event exactly once, and **drains** consumed
/// bytes from the working buffer so it never rescans or accumulates the whole
/// response (the old approach was near-quadratic on fragmented streams). Total
/// and per-event byte caps bound memory against slow or hostile servers.
pub(crate) struct SseDecoder {
    /// Bytes received but not yet framed into a complete event.
    buf: String,
    /// Total decoded bytes seen across the whole response.
    total_bytes: usize,
    max_event_bytes: usize,
    max_total_bytes: usize,
}

impl SseDecoder {
    fn new() -> Self {
        Self::with_limits(DEFAULT_MAX_EVENT_BYTES, DEFAULT_MAX_TOTAL_BYTES)
    }

    fn with_limits(max_event_bytes: usize, max_total_bytes: usize) -> Self {
        Self {
            buf: String::new(),
            total_bytes: 0,
            max_event_bytes,
            max_total_bytes,
        }
    }

    /// Push a decoded UTF-8 `chunk`, returning any event blocks it completed
    /// (the text between delimiters, delimiters removed), drained from the
    /// buffer. Errors if the total or a single un-terminated event exceeds the
    /// configured caps.
    fn push(&mut self, chunk: &str) -> Result<Vec<String>> {
        self.total_bytes = self.total_bytes.saturating_add(chunk.len());
        if self.total_bytes > self.max_total_bytes {
            return Err(Error::service(format!(
                "MCP SSE response exceeded the maximum size ({} bytes)",
                self.max_total_bytes
            )));
        }
        self.buf.push_str(chunk);
        let mut events = Vec::new();
        while let Some((idx, len)) = find_delimiter(&self.buf) {
            let event: String = self.buf[..idx].to_string();
            // Drop the event text and its delimiter from the front of the
            // buffer so we never rescan already-consumed bytes.
            self.buf.drain(..idx + len);
            events.push(event);
        }
        if self.buf.len() > self.max_event_bytes {
            return Err(Error::service(format!(
                "MCP SSE event exceeded the maximum size ({} bytes)",
                self.max_event_bytes
            )));
        }
        Ok(events)
    }

    /// Return any trailing buffered event not terminated by a final blank line,
    /// draining the buffer. Called once at end-of-stream.
    fn flush(&mut self) -> Vec<String> {
        if self.buf.trim().is_empty() {
            self.buf.clear();
            return Vec::new();
        }
        vec![std::mem::take(&mut self.buf)]
    }
}

/// Find the earliest SSE event delimiter in `buf`, returning its byte offset
/// and length. Recognizes `\r\n\r\n` (4), `\n\n` (2), and `\r\r` (2). The three
/// patterns never begin at the same offset (their first bytes differ where they
/// could collide), so the smallest start offset is unambiguous.
fn find_delimiter(buf: &str) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (pat, len) in [("\r\n\r\n", 4usize), ("\n\n", 2), ("\r\r", 2)] {
        if let Some(idx) = buf.find(pat) {
            best = match best {
                Some((b, _)) if b <= idx => best,
                _ => Some((idx, len)),
            };
        }
    }
    best
}

/// Join an SSE event block's `data:` line(s) (per the SSE spec) and parse
/// the result as JSON. `None` if there's no `data:` field, or it isn't
/// valid JSON (logged in the latter case).
fn sse_event_json(event_text: &str) -> Option<Value> {
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
    match serde_json::from_str(&data) {
        Ok(v) => Some(v),
        Err(e) => {
            tracing::warn!(error = %e, data = %data, "MCP: unparseable SSE event data");
            None
        }
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

    // -- Incremental SSE decoder ----------------------------------------

    /// Feed `chunks` through a fresh decoder and collect every event it emits
    /// (including any trailing event surfaced by `flush`).
    fn decode_events(chunks: &[&str]) -> Vec<String> {
        let mut decoder = SseDecoder::new();
        let mut out = Vec::new();
        for c in chunks {
            out.extend(decoder.push(c).expect("decode chunk"));
        }
        out.extend(decoder.flush());
        out
    }

    /// Find the JSON-RPC response matching `expected_id` among decoded events.
    fn find_response(events: &[String], expected_id: i64) -> Option<Result<Value>> {
        for e in events {
            let Some(value) = sse_event_json(e) else {
                continue;
            };
            if let IncomingMessage::Response { id, result } = protocol::parse_incoming(value) {
                if id == expected_id {
                    return Some(result.map_err(|e| Error::service(e.to_string())));
                }
            }
        }
        None
    }

    #[test]
    fn decoder_finds_response_after_a_notification() {
        let events = decode_events(&[concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\",\"params\":{\"level\":\"info\"}}\n",
            "\n",
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":5,\"result\":{\"ok\":true}}\n",
            "\n",
        )]);
        let result = find_response(&events, 5)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn decoder_handles_crlf_framing() {
        // A stream framed with `\r\n\r\n` must split into separate events (the
        // old `split("\n\n")` framing missed this entirely).
        let events = decode_events(&[concat!(
            "event: message\r\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/message\"}\r\n",
            "\r\n",
            "event: message\r\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\r\n",
            "\r\n",
        )]);
        assert_eq!(events.len(), 2, "CRLF-framed events must split: {events:?}");
        let result = find_response(&events, 7)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn decoder_handles_bare_cr_framing() {
        let events = decode_events(&["data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\r\r"]);
        assert_eq!(events.len(), 1);
        assert_eq!(find_response(&events, 1).unwrap().unwrap(), json!({}));
    }

    #[test]
    fn decoder_handles_delimiter_split_across_chunks() {
        // The `\r\n\r\n` delimiter arrives split across two network chunks.
        let mut decoder = SseDecoder::new();
        let mut events = decoder
            .push("data: {\"jsonrpc\":\"2.0\",\"id\":3,\"result\":{}}\r\n")
            .unwrap();
        assert!(events.is_empty(), "event not complete until delimiter seen");
        events.extend(decoder.push("\r\ntrailing").unwrap());
        assert_eq!(events.len(), 1);
        assert_eq!(find_response(&events, 3).unwrap().unwrap(), json!({}));
    }

    #[test]
    fn decoder_skips_comments_and_reads_multiline_data() {
        // A leading comment line (starts with ':') is ignored; multiple `data:`
        // lines in one event are joined with `\n`.
        let events = decode_events(&[
            ": this is a comment\ndata: {\"jsonrpc\":\"2.0\",\ndata: \"id\":9,\"result\":{}}\n\n",
        ]);
        assert_eq!(find_response(&events, 9).unwrap().unwrap(), json!({}));
    }

    #[test]
    fn decoder_finds_response_even_with_an_interleaved_server_request() {
        let events = decode_events(&[concat!(
            "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n",
        )]);
        let result = find_response(&events, 1)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[test]
    fn decoder_surfaces_error_response() {
        let events = decode_events(&[
            "data: {\"jsonrpc\":\"2.0\",\"id\":4,\"error\":{\"code\":-1,\"message\":\"nope\"}}\n\n",
        ]);
        let err = find_response(&events, 4)
            .expect("should find response")
            .unwrap_err();
        assert!(err.to_string().contains("nope"));
    }

    #[test]
    fn decoder_returns_none_when_id_never_appears() {
        let events = decode_events(&["data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n"]);
        assert!(find_response(&events, 999).is_none());
    }

    #[test]
    fn decoder_drains_consumed_bytes() {
        // After emitting an event, the working buffer must not retain it (so a
        // long stream never grows without bound / rescans).
        let mut decoder = SseDecoder::new();
        let _ = decoder
            .push("data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n\n")
            .unwrap();
        assert!(
            decoder.buf.is_empty(),
            "buffer must be drained after an event"
        );
    }

    #[test]
    fn decoder_enforces_total_size_cap() {
        let mut decoder = SseDecoder::with_limits(1024, 16);
        let err = decoder
            .push("data: aaaaaaaaaaaaaaaaaaaaaaaa\n\n")
            .unwrap_err();
        assert!(err.to_string().contains("maximum size"), "{err}");
    }

    #[test]
    fn decoder_enforces_per_event_size_cap() {
        // A single event that never terminates must not grow past the cap.
        let mut decoder = SseDecoder::with_limits(8, 1024);
        let err = decoder
            .push("data: never-ending event with no delimiter")
            .unwrap_err();
        assert!(err.to_string().contains("event exceeded"), "{err}");
    }

    #[test]
    fn find_delimiter_prefers_earliest_delimiter() {
        assert_eq!(find_delimiter("ab\r\n\r\ncd"), Some((2, 4)));
        assert_eq!(find_delimiter("ab\n\ncd"), Some((2, 2)));
        assert_eq!(find_delimiter("ab\r\rcd"), Some((2, 2)));
        assert_eq!(find_delimiter("no delimiter here"), None);
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

    // -- Server-initiated request parsing (see `sampling.rs` for full
    // dispatch-logic tests, and `websocket_e2e.rs`/`stdio_e2e.rs` for
    // full-duplex round trips) ------------------------------------------

    #[test]
    fn sse_event_json_parses_a_server_initiated_request_frame() {
        let event = "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"sampling/createMessage\",\"params\":{\"maxTokens\":10}}";
        let value = sse_event_json(event).expect("should parse as JSON");
        match protocol::parse_incoming(value) {
            IncomingMessage::ServerRequest { id, method, params } => {
                assert_eq!(id, json!("srv-1"));
                assert_eq!(method, "sampling/createMessage");
                assert_eq!(params["maxTokens"], 10);
            }
            other => panic!("expected ServerRequest, got {other:?}"),
        }
    }

    #[test]
    fn sse_event_json_returns_none_for_non_data_event() {
        assert!(sse_event_json("event: ping\nretry: 1000").is_none());
    }
}
