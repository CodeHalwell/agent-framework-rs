//! Loopback tests: a bare `std::net::TcpListener` thread speaks just enough
//! HTTP/1.1 to serve one canned response, exercising the real `reqwest`
//! request/response path (headers, JSON body, status-code handling) end to
//! end without any external network access or additional dependencies.
//! Mirrors `agent-framework-mcp`'s `tests/http_loopback.rs`.
//!
//! This is how "HTTP error surfacing" and "request body building" are
//! verified against the *actual* `reqwest` call path (as opposed to the
//! pure `build_add_body`/`build_search_body` unit tests in `src/provider.rs`,
//! which check the same logic without any I/O at all). Per the work
//! package's "NO live-network tests" rule, nothing here ever leaves
//! localhost.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use agent_framework_core::memory::ContextProvider;
use agent_framework_core::types::ChatMessage;
use agent_framework_mem0::Mem0Provider;
use serde_json::{json, Value};

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Accept one connection on `listener`, bounded by a generous retry loop so a
/// misbehaving client can't hang the test suite forever.
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

/// One fully-read HTTP/1.1 request: method, path, raw header block, and body.
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

    /// Case-insensitive header lookup.
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

/// Read one HTTP/1.1 request's request-line, headers, and
/// `Content-Length` body from `stream`.
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

fn write_status_response(stream: &mut TcpStream, status: u16, reason: &str, body: &Value) {
    let payload = body.to_string();
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

fn write_json_response(stream: &mut TcpStream, body: &Value) {
    write_status_response(stream, 200, "OK", body);
}

/// Spawn a single-shot loopback HTTP server: accepts exactly one connection,
/// captures the request, hands it to `respond` to write a response, then the
/// thread exits, returning the captured request via the join handle.
fn serve_one<F>(respond: F) -> (String, std::thread::JoinHandle<CapturedRequest>)
where
    F: FnOnce(&mut TcpStream) + Send + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let handle = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let request = read_http_request(&mut stream);
        respond(&mut stream);
        request
    });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn invoked_posts_to_v1_memories_with_body_and_auth_header() {
    let (base_url, handle) = serve_one(|stream| {
        write_json_response(stream, &json!({"message": "queued"}));
    });

    let provider = Mem0Provider::new("test-api-key")
        .with_api_base(base_url)
        .with_user_id("user-42")
        .with_application_id("app-1");

    provider
        .invoked(
            &[ChatMessage::user("I moved to Austin last month")],
            &[],
            None,
        )
        .await
        .unwrap();

    let request = handle.join().expect("server thread panicked");
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v1/memories/");
    assert_eq!(
        request.header("authorization").as_deref(),
        Some("Token test-api-key")
    );
    assert_eq!(
        request.header("content-type").as_deref(),
        Some("application/json")
    );

    let body = request.body_json();
    assert_eq!(body["user_id"], json!("user-42"));
    assert_eq!(body["metadata"], json!({"application_id": "app-1"}));
    assert_eq!(
        body["messages"],
        json!([{"role": "user", "content": "I moved to Austin last month"}])
    );
}

#[tokio::test]
async fn invoked_combines_request_and_response_messages() {
    let (base_url, handle) = serve_one(|stream| {
        write_json_response(stream, &json!({"message": "queued"}));
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_user_id("u1");
    provider
        .invoked(
            &[ChatMessage::user("What's the weather?")],
            &[ChatMessage::assistant("It's sunny today.")],
            None,
        )
        .await
        .unwrap();

    let request = handle.join().unwrap();
    let body = request.body_json();
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "It's sunny today."},
        ])
    );
}

#[tokio::test]
async fn invoking_posts_to_v2_search_and_parses_hits_into_context() {
    let (base_url, handle) = serve_one(|stream| {
        write_json_response(
            stream,
            &json!([
                {"memory": "User likes outdoor activities"},
                {"memory": "User lives in Seattle"},
            ]),
        );
    });

    let provider = Mem0Provider::new("test-api-key")
        .with_api_base(base_url)
        .with_user_id("user-42");

    let ctx = provider
        .invoking(&[ChatMessage::user("What's the weather like where I live?")])
        .await
        .unwrap();

    let request = handle.join().expect("server thread panicked");
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/v2/memories/search/");

    let body = request.body_json();
    assert_eq!(
        body["query"],
        json!("What's the weather like where I live?")
    );
    assert_eq!(body["filters"], json!({"user_id": "user-42"}));

    assert_eq!(ctx.messages.len(), 1);
    assert_eq!(
        ctx.messages[0].text(),
        "## Memories\nConsider the following memories when answering user questions:\n\
         User likes outdoor activities\nUser lives in Seattle"
    );
}

#[tokio::test]
async fn invoking_handles_results_wrapper_response_shape() {
    let (base_url, _handle) = serve_one(|stream| {
        write_json_response(
            stream,
            &json!({"results": [{"memory": "Previous conversation context"}]}),
        );
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_agent_id("agent-1");
    let ctx = provider
        .invoking(&[ChatMessage::user("hello")])
        .await
        .unwrap();
    assert!(ctx.messages[0]
        .text()
        .contains("Previous conversation context"));
}

#[tokio::test]
async fn invoking_empty_results_returns_empty_context() {
    let (base_url, _handle) = serve_one(|stream| {
        write_json_response(stream, &json!([]));
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_user_id("u1");
    let ctx = provider
        .invoking(&[ChatMessage::user("hello")])
        .await
        .unwrap();
    assert!(ctx.messages.is_empty());
}

#[tokio::test]
async fn invoking_surfaces_non_2xx_status_as_service_error() {
    let (base_url, _handle) = serve_one(|stream| {
        write_status_response(
            stream,
            401,
            "Unauthorized",
            &json!({"detail": "Invalid API key."}),
        );
    });

    let provider = Mem0Provider::new("bad-key")
        .with_api_base(base_url)
        .with_user_id("u1");
    let err = provider
        .invoking(&[ChatMessage::user("hi")])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("401"), "message was: {msg}");
    assert!(msg.contains("Invalid API key"), "message was: {msg}");
}

#[tokio::test]
async fn invoked_surfaces_non_2xx_status_as_service_error() {
    let (base_url, _handle) = serve_one(|stream| {
        write_status_response(
            stream,
            500,
            "Internal Server Error",
            &json!({"detail": "boom"}),
        );
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_user_id("u1");
    let err = provider
        .invoked(&[ChatMessage::user("hi")], &[], None)
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("500"), "message was: {msg}");
    assert!(msg.contains("boom"), "message was: {msg}");
}

#[tokio::test]
async fn invoking_surfaces_malformed_json_response_as_error() {
    let (base_url, _handle) = serve_one(|stream| {
        let payload = "not valid json";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream
            .write_all(response.as_bytes())
            .expect("write response");
        stream.flush().expect("flush response");
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_user_id("u1");
    let err = provider
        .invoking(&[ChatMessage::user("hi")])
        .await
        .unwrap_err();
    assert!(err.to_string().contains("invalid Mem0 API response JSON"));
}

#[tokio::test]
async fn invoking_scopes_search_to_agent_and_run_id() {
    let (base_url, handle) = serve_one(|stream| {
        write_json_response(stream, &json!([]));
    });

    let provider = Mem0Provider::new("k")
        .with_api_base(base_url)
        .with_agent_id("agent-1")
        .with_thread_id("thread-1");

    provider
        .invoking(&[ChatMessage::user("hello")])
        .await
        .unwrap();

    let request = handle.join().unwrap();
    let body = request.body_json();
    assert_eq!(
        body["filters"],
        json!({"agent_id": "agent-1", "run_id": "thread-1"})
    );
}
