//! Loopback tests: a bare `std::net::TcpListener` thread plays a fake
//! Microsoft Graph `processContent` endpoint, exercising the real
//! `reqwest`-based request/response path (headers, JSON body,
//! block/allow-driven status) through the actual [`PurviewAgentMiddleware`]
//! pipeline end to end. Mirrors `agent-framework-a2a`'s and
//! `agent-framework-mem0`'s `tests/*loopback*.rs`, and
//! `agent-framework-copilotstudio`'s `tests/loopback.rs`.
//!
//! Per the work package's "NO live network" rule, nothing here ever leaves
//! localhost — [`PurviewSettings::graph_base_uri`] is pointed straight at
//! the loopback server.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::middleware::{AgentContext, MiddlewarePipeline, Terminal};
use agent_framework_core::tools::BoxFuture;
use agent_framework_core::types::{AgentRunResponse, ChatMessage};
use agent_framework_purview::{
    PurviewAgentMiddleware, PurviewAppLocation, PurviewLocationType, PurviewSettings,
    StaticTokenProvider,
};
use serde_json::Value;

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn accept_with_timeout(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).expect("set nonblocking");
    for _ in 0..500 {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).expect("set blocking");
                return stream;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("accept failed: {e}"),
        }
    }
    panic!("timed out waiting for a client connection");
}

struct CapturedRequest {
    method: String,
    path: String,
    header_block: String,
    body: Vec<u8>,
}

impl CapturedRequest {
    fn body_json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("request body is valid JSON")
    }

    fn header(&self, name: &str) -> Option<String> {
        let needle = format!("{}:", name.to_ascii_lowercase());
        self.header_block.lines().find_map(|l| {
            if l.to_ascii_lowercase().starts_with(&needle) {
                l.split_once(':').map(|(_, v)| v.trim().to_string())
            } else {
                None
            }
        })
    }
}

fn read_http_request(stream: &mut TcpStream) -> CapturedRequest {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        assert!(n > 0, "connection closed before headers completed");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut request_line_parts = header_str
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace();
    let method = request_line_parts.next().unwrap_or_default().to_string();
    let path = request_line_parts.next().unwrap_or_default().to_string();
    let content_length: usize = header_str
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).expect("read request body");
        assert!(n > 0, "connection closed before body completed");
        buf.extend_from_slice(&chunk[..n]);
    }
    CapturedRequest {
        method,
        path,
        header_block: header_str,
        body: buf[body_start..body_start + content_length].to_vec(),
    }
}

fn write_json_response(stream: &mut TcpStream, body: &Value) {
    let payload = body.to_string();
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

/// Spawn a loopback server accepting `respond_fns.len()` connections in
/// sequence, feeding each captured request to the matching response
/// closure. Returns the base URL and a join handle yielding every captured
/// request, in order.
type RespondFn = Box<dyn FnOnce(&mut TcpStream) + Send>;

fn serve_sequence(
    respond_fns: Vec<RespondFn>,
) -> (String, std::thread::JoinHandle<Vec<CapturedRequest>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::with_capacity(respond_fns.len());
        for respond in respond_fns {
            let mut stream = accept_with_timeout(&listener);
            let request = read_http_request(&mut stream);
            respond(&mut stream);
            requests.push(request);
        }
        requests
    });
    (format!("http://{addr}"), handle)
}

fn settings_pointed_at(base_url: &str) -> PurviewSettings {
    PurviewSettings::new("Test App")
        .with_tenant_id("12345678-1234-1234-1234-123456789012")
        .with_purview_app_location(PurviewAppLocation::new(
            PurviewLocationType::Application,
            "app-42",
        ))
        .with_graph_base_uri(base_url.to_string())
}

fn user_message() -> ChatMessage {
    let mut m = ChatMessage::user("What's our Q3 roadmap?");
    m.additional_properties.insert(
        "user_id".to_string(),
        serde_json::json!("87654321-4321-4321-4321-210987654321"),
    );
    m
}

fn terminal_returning(called: Arc<AtomicBool>, text: &'static str) -> Terminal<AgentContext> {
    Box::new(move |mut ctx: AgentContext| {
        called.store(true, Ordering::SeqCst);
        Box::pin(async move {
            ctx.result = Some(AgentRunResponse {
                messages: vec![ChatMessage::assistant(text)],
                ..Default::default()
            });
            Ok(ctx)
        }) as BoxFuture<agent_framework_core::error::Result<AgentContext>>
    })
}

#[tokio::test]
async fn allow_end_to_end_both_directions_pass() {
    let (base_url, handle) = serve_sequence(vec![
        // Prompt check: allow (no policyActions at all).
        Box::new(|stream| write_json_response(stream, &serde_json::json!({"id": "1"}))),
        // Response check: allow.
        Box::new(|stream| write_json_response(stream, &serde_json::json!({"id": "2"}))),
    ]);

    let middleware = PurviewAgentMiddleware::new(
        StaticTokenProvider::new("test-token"),
        settings_pointed_at(&base_url),
    );
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
    let called = Arc::new(AtomicBool::new(false));
    let ctx = AgentContext::new(vec![user_message()], false);

    let result_ctx = pipeline
        .execute(
            ctx,
            terminal_returning(called.clone(), "Here's the roadmap."),
        )
        .await
        .unwrap();

    assert!(
        called.load(Ordering::SeqCst),
        "next must run when the prompt is allowed"
    );
    assert!(!result_ctx.terminate);
    assert_eq!(result_ctx.result.unwrap().text(), "Here's the roadmap.");

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 2);
    for req in &requests {
        assert_eq!(req.method, "POST");
        assert_eq!(
            req.path,
            "/users/87654321-4321-4321-4321-210987654321/dataSecurityAndGovernance/processContent"
        );
        assert_eq!(
            req.header("authorization").as_deref(),
            Some("Bearer test-token")
        );
        assert_eq!(
            req.header("content-type").as_deref(),
            Some("application/json")
        );
        let body = req.body_json();
        assert_eq!(
            body["userId"],
            serde_json::json!("87654321-4321-4321-4321-210987654321")
        );
        assert_eq!(
            body["tenantId"],
            serde_json::json!("12345678-1234-1234-1234-123456789012")
        );
        assert_eq!(
            body["contentToProcess"]["activityMetadata"]["activity"],
            serde_json::json!("uploadText")
        );
    }
    // Prompt-phase request carries the user's outgoing text ...
    assert_eq!(
        requests[0].body_json()["contentToProcess"]["contentEntries"][0]["content"]["data"],
        serde_json::json!("What's our Q3 roadmap?")
    );
    // ... the response-phase request carries the agent's reply text.
    assert_eq!(
        requests[1].body_json()["contentToProcess"]["contentEntries"][0]["content"]["data"],
        serde_json::json!("Here's the roadmap.")
    );
}

#[tokio::test]
async fn block_on_prompt_short_circuits_before_next_and_before_any_response_check() {
    let (base_url, handle) = serve_sequence(vec![Box::new(|stream| {
        write_json_response(
            stream,
            &serde_json::json!({
                "id": "1",
                "policyActions": [{"action": "blockAccess", "restrictionAction": "block"}]
            }),
        );
    })]);

    let middleware = PurviewAgentMiddleware::new(
        StaticTokenProvider::new("test-token"),
        settings_pointed_at(&base_url),
    );
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
    let called = Arc::new(AtomicBool::new(false));
    let ctx = AgentContext::new(vec![user_message()], false);

    let result_ctx = pipeline
        .execute(
            ctx,
            terminal_returning(called.clone(), "should never be produced"),
        )
        .await
        .unwrap();

    assert!(
        !called.load(Ordering::SeqCst),
        "next must not run when the prompt is blocked"
    );
    assert!(result_ctx.terminate);
    assert_eq!(
        result_ctx.result.unwrap().text(),
        "Prompt blocked by policy"
    );

    // Exactly one request was made (the prompt check); if a response check
    // had incorrectly fired too, the server (programmed for one connection
    // only) would leave the join handle hanging or the connection refused.
    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 1);
}

#[tokio::test]
async fn block_on_response_replaces_result_without_terminating() {
    let (base_url, handle) = serve_sequence(vec![
        // Prompt check: allow.
        Box::new(|stream| write_json_response(stream, &serde_json::json!({"id": "1"}))),
        // Response check: block.
        Box::new(|stream| {
            write_json_response(
                stream,
                &serde_json::json!({
                    "id": "2",
                    "policyActions": [{"action": "other", "restrictionAction": "block"}]
                }),
            );
        }),
    ]);

    let middleware = PurviewAgentMiddleware::new(
        StaticTokenProvider::new("test-token"),
        settings_pointed_at(&base_url),
    );
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
    let called = Arc::new(AtomicBool::new(false));
    let ctx = AgentContext::new(vec![user_message()], false);

    let result_ctx = pipeline
        .execute(
            ctx,
            terminal_returning(called.clone(), "This leaks sensitive data."),
        )
        .await
        .unwrap();

    assert!(
        called.load(Ordering::SeqCst),
        "next must run -- only the response is blocked"
    );
    // Unlike the prompt-blocked case, `terminate` is not set: the run
    // already completed, there's nothing left to short-circuit.
    assert!(!result_ctx.terminate);
    assert_eq!(
        result_ctx.result.unwrap().text(),
        "Response blocked by policy"
    );

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 2);
}

#[tokio::test]
async fn non_success_status_surfaces_as_service_error_and_stops_the_pipeline() {
    let (base_url, _handle) = serve_sequence(vec![Box::new(|stream| {
        let payload = "insufficient privileges";
        let response = format!(
            "HTTP/1.1 403 Forbidden\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    })]);

    let middleware = PurviewAgentMiddleware::new(
        StaticTokenProvider::new("test-token"),
        settings_pointed_at(&base_url),
    );
    let pipeline = MiddlewarePipeline::new(vec![Arc::new(middleware)]);
    let called = Arc::new(AtomicBool::new(false));
    let ctx = AgentContext::new(vec![user_message()], false);

    let result = pipeline
        .execute(ctx, terminal_returning(called.clone(), "unreachable"))
        .await;
    let err = match result {
        Ok(_) => panic!("expected Err, got Ok"),
        Err(e) => e,
    };
    assert_eq!(err.status(), Some(403));
    assert!(!called.load(Ordering::SeqCst));
}
