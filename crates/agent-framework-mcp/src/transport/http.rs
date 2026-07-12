//! Streamable HTTP transport: POSTs each JSON-RPC message to a single MCP
//! endpoint, accepting either an `application/json` response body (one
//! response) or a `text/event-stream` body (SSE frames scanned for the
//! response whose `id` matches the request).

use std::collections::HashSet;
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
    /// soon as the JSON-RPC response matching `expected_id` is seen. Re-scans
    /// the accumulated buffer via [`parse_sse_buffer`] as each chunk
    /// arrives — simple and plenty fast for the small, infrequent bodies
    /// MCP responses produce. Any server-initiated request seen along the
    /// way is dispatched via [`Self::dispatch_buffered_server_requests`], and
    /// any notification (e.g. `notifications/tools/list_changed`) via
    /// [`Self::dispatch_buffered_notifications`]; both are deduplicated so a
    /// message appearing in an earlier, still-buffered scan is only ever
    /// dispatched once.
    ///
    /// Dispatch always runs against the latest buffer content *before* the
    /// "did we find our answer yet" check on each iteration — including the
    /// answer's own chunk — so a server-initiated message landing in the
    /// same chunk as the expected response is not skipped just because that
    /// chunk also happens to satisfy the early return.
    async fn read_sse_for_id(&self, resp: reqwest::Response, expected_id: i64) -> Result<Value> {
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut utf8 = Utf8StreamDecoder::new();
        let mut handled_server_request_ids: HashSet<String> = HashSet::new();
        let mut handled_notifications: HashSet<usize> = HashSet::new();
        loop {
            match stream.next().await {
                Some(Ok(bytes)) => {
                    let decoded = utf8.push(&bytes);
                    buf.push_str(&decoded);
                }
                Some(Err(e)) => return Err(Error::service(format!("MCP SSE stream error: {e}"))),
                None => break,
            }
            self.dispatch_buffered_server_requests(&buf, &mut handled_server_request_ids)
                .await;
            self.dispatch_buffered_notifications(&buf, &mut handled_notifications)
                .await;
            if let Some(result) = parse_sse_buffer(&buf, expected_id) {
                return result;
            }
        }
        Err(Error::service(format!(
            "MCP SSE stream ended without a response for request id {expected_id}"
        )))
    }

    /// Scan `buf` for server-initiated request events and answer each one
    /// not already present in `handled` exactly once. `buf` only grows
    /// during one `read_sse_for_id` call, so `handled` (keyed by the
    /// request's JSON-RPC id) is what keeps a full-buffer rescan from
    /// re-dispatching the same request on every new chunk.
    async fn dispatch_buffered_server_requests(&self, buf: &str, handled: &mut HashSet<String>) {
        for event_text in buf.split("\n\n") {
            if event_text.trim().is_empty() {
                continue;
            }
            let Some(value) = sse_event_json(event_text) else {
                continue;
            };
            if let IncomingMessage::ServerRequest { id, method, params } =
                protocol::parse_incoming(value)
            {
                if handled.insert(id.to_string()) {
                    self.respond_to_server_request(id, method, params).await;
                }
            }
        }
    }

    /// Scan `buf` for notification events (no `id`, e.g.
    /// `notifications/tools/list_changed`) and dispatch each one not already
    /// present in `handled` exactly once, to whatever handler
    /// [`McpTransport::set_notification_handler`] installed. `handled` is
    /// keyed by the event's split index — stable because `buf` only grows
    /// during one `read_sse_for_id` call — since notifications carry no id;
    /// keying by index (not raw text) keeps two byte-identical notifications
    /// distinct and avoids per-rescan string allocations.
    async fn dispatch_buffered_notifications(&self, buf: &str, handled: &mut HashSet<usize>) {
        for (idx, event_text) in buf.split("\n\n").enumerate() {
            if event_text.trim().is_empty() {
                continue;
            }
            let Some(value) = sse_event_json(event_text) else {
                continue;
            };
            if let IncomingMessage::Notification { method, params } =
                protocol::parse_incoming(value)
            {
                if handled.insert(idx) {
                    let handler = self.notification_handler.lock().unwrap().clone();
                    if let Some(handler) = handler {
                        handler(method, params).await;
                    }
                }
            }
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

/// Parse one SSE event block as a JSON-RPC message; returns `Some` only if
/// it is the response matching `expected_id`. A server-initiated request is
/// logged here (this function has no way to answer it — see
/// [`McpStreamableHttpTransport::dispatch_buffered_server_requests`], which
/// scans the same buffer separately for those).
fn extract_sse_event(event_text: &str, expected_id: i64) -> Option<Result<Value>> {
    let value = sse_event_json(event_text)?;
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
        IncomingMessage::ServerRequest { id, method, .. } => {
            tracing::debug!(
                id = %id,
                method = %method,
                "MCP SSE: server-initiated request seen while scanning for a response; \
                 handled separately via dispatch_buffered_server_requests"
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

    #[tokio::test]
    async fn identical_notifications_each_dispatch_once() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;

        let transport =
            McpStreamableHttpTransport::new("http://localhost:0/mcp", HeaderMap::new(), None);
        let count = Arc::new(AtomicUsize::new(0));
        let seen = count.clone();
        transport.set_notification_handler(Arc::new(move |_method, _params| {
            let seen = seen.clone();
            Box::pin(async move {
                seen.fetch_add(1, Ordering::SeqCst);
            })
        }));

        // Two byte-identical notification events: index-keyed dedup must
        // dispatch BOTH (text-keyed dedup would collapse them into one),
        // while a rescan of the same grown buffer must not re-dispatch.
        let notif = concat!(
            "event: message\n",
            "data: {\"jsonrpc\":\"2.0\",\"method\":\"notifications/tools/list_changed\"}\n",
            "\n",
        );
        let buf = format!("{notif}{notif}");
        let mut handled = HashSet::new();
        transport
            .dispatch_buffered_notifications(&buf, &mut handled)
            .await;
        assert_eq!(count.load(Ordering::SeqCst), 2);
        transport
            .dispatch_buffered_notifications(&buf, &mut handled)
            .await;
        assert_eq!(
            count.load(Ordering::SeqCst),
            2,
            "a rescan must not re-dispatch already-handled events"
        );
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

    #[test]
    fn extract_sse_event_ignores_server_initiated_requests() {
        // `extract_sse_event` only ever resolves the expected *response*; a
        // server-initiated request frame is logged and skipped here (real
        // dispatch happens via `dispatch_buffered_server_requests`, scanning
        // the same buffer separately).
        let event = "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}";
        assert!(extract_sse_event(event, 1).is_none());
    }

    #[test]
    fn parse_sse_buffer_finds_response_even_with_an_interleaved_server_request() {
        let buf = concat!(
            "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}\n\n",
            "data: {\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"ok\":true}}\n\n",
        );
        let result = parse_sse_buffer(buf, 1)
            .expect("should find response")
            .unwrap();
        assert_eq!(result, json!({"ok": true}));
    }

    #[tokio::test]
    async fn dispatch_buffered_server_requests_answers_once_and_dedupes_on_rescan() {
        let transport = McpStreamableHttpTransport::new(
            "http://127.0.0.1:0/unused".to_string(),
            Default::default(),
            None,
        );
        let buf = "data: {\"jsonrpc\":\"2.0\",\"id\":\"srv-1\",\"method\":\"ping\"}\n\n";
        let mut handled = HashSet::new();
        // No live server behind this transport's URL, so `respond_to_server_request`'s
        // best-effort POST will fail — that failure is swallowed, which is
        // exactly the point: this only asserts dedup bookkeeping.
        transport
            .dispatch_buffered_server_requests(buf, &mut handled)
            .await;
        assert_eq!(handled.len(), 1);
        // Re-scanning the same (unchanged) buffer must not add a duplicate
        // dispatch — the HashSet-based bookkeeping the whole point of this
        // test.
        transport
            .dispatch_buffered_server_requests(buf, &mut handled)
            .await;
        assert_eq!(handled.len(), 1);
    }
}
