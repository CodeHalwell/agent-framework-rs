//! Loopback test: a bare `std::net::TcpListener` thread speaks just enough
//! HTTP/1.1 to play a fake Direct-to-Engine server across a full
//! `start_conversation` + two `ask_question`-equivalent turns, exercising the
//! real `reqwest`-based request/response path end to end (headers, SSE-framed
//! body, conversation-id continuity across calls) without any external
//! network access. Mirrors `agent-framework-a2a`'s and
//! `agent-framework-mem0`'s `tests/*loopback*.rs`.
//!
//! Per the work package's "NO live network" rule, nothing here ever leaves
//! localhost — [`CopilotStudioConnectionSettings::with_direct_connect_url`]
//! points the client straight at the loopback server, bypassing
//! environment-id/cloud host construction (covered separately, without any
//! I/O, by `settings::tests` in `src/settings.rs`).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use agent_framework_copilotstudio::{
    CopilotStudioAgent, CopilotStudioConnectionSettings, StaticTokenProvider,
};
use agent_framework_core::prelude::*;
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

/// Write an SSE `event: activity` / `data: {...}` framed response carrying
/// exactly one activity — the real Direct-to-Engine response shape.
fn write_sse_activity_response(stream: &mut TcpStream, activity: &Value) {
    let payload = format!("event: activity\ndata: {}\n\n", activity);
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

/// One connection's response-writing behavior, boxed so [`serve_sequence`]
/// can queue up a different one per accepted connection.
type RespondFn = Box<dyn FnOnce(&mut TcpStream) + Send>;

/// Spawn a loopback server that accepts `respond_fns.len()` connections in
/// sequence, one per call's outgoing request, feeding each captured request
/// to the corresponding response closure. Returns the base URL and a join
/// handle yielding every captured request, in order.
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

#[tokio::test]
async fn full_exchange_start_conversation_then_two_turns_with_continuity() {
    let (base_url, handle) = serve_sequence(vec![
        // 1. start_conversation (POST .../conversations)
        Box::new(|stream| {
            write_sse_activity_response(
                stream,
                &serde_json::json!({
                    "type": "event",
                    "conversation": { "id": "conv-42" }
                }),
            );
        }),
        // 2. first turn (POST .../conversations/conv-42): a typing update
        //    followed by the final message, mirroring a real turn -- only
        //    the message activity should surface as a response message.
        Box::new(|stream| {
            let payload = format!(
                "event: activity\ndata: {}\n\nevent: activity\ndata: {}\n\n",
                serde_json::json!({"type": "typing", "id": "t1"}),
                serde_json::json!({
                    "type": "message",
                    "id": "m1",
                    "text": "Bonjour! The capital of France is Paris.",
                    "from": {"name": "My Copilot"},
                    "conversation": {"id": "conv-42"}
                }),
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                payload.len(),
                payload
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.flush().unwrap();
        }),
        // 3. second turn (POST .../conversations/conv-42 again -- no second
        //    start_conversation call, proving continuity).
        Box::new(|stream| {
            write_sse_activity_response(
                stream,
                &serde_json::json!({
                    "type": "message",
                    "id": "m2",
                    "text": "Tomorrow looks sunny.",
                    "conversation": {"id": "conv-42"}
                }),
            );
        }),
    ]);

    let connection = CopilotStudioConnectionSettings::new("unused-env-id", "unused-schema")
        .with_direct_connect_url(base_url);
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new("test-token"))
        .with_name("My Copilot");

    let mut thread = agent.get_new_thread();
    let first = agent
        .run(
            vec![Message::user("What is the capital of France?")],
            Some(&mut thread),
        )
        .await
        .unwrap();
    assert_eq!(first.text(), "Bonjour! The capital of France is Paris.");
    assert_eq!(first.response_id.as_deref(), Some("m1"));
    assert_eq!(thread.service_thread_id(), Some("conv-42"));

    let second = agent
        .run(
            vec![Message::user("What about tomorrow?")],
            Some(&mut thread),
        )
        .await
        .unwrap();
    assert_eq!(second.text(), "Tomorrow looks sunny.");
    // Conversation id is unchanged: no second `start_conversation` call was
    // made (the server only programmed 3 responses total; if a fourth
    // connection had been attempted, `handle.join()` below would hang/panic).
    assert_eq!(thread.service_thread_id(), Some("conv-42"));

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 3);

    // Request 1: start_conversation.
    assert_eq!(requests[0].method, "POST");
    assert_eq!(
        requests[0].path,
        "/conversations?api-version=2022-03-01-preview"
    );
    assert_eq!(
        requests[0].body_json(),
        serde_json::json!({ "emitStartConversationEvent": true })
    );
    assert_eq!(
        requests[0].header("authorization").as_deref(),
        Some("Bearer test-token")
    );
    assert_eq!(
        requests[0].header("accept").as_deref(),
        Some("text/event-stream")
    );

    // Request 2: first turn, addressed at the conversation created above.
    assert_eq!(requests[1].method, "POST");
    assert_eq!(
        requests[1].path,
        "/conversations/conv-42?api-version=2022-03-01-preview"
    );
    assert_eq!(
        requests[1].body_json(),
        serde_json::json!({
            "activity": {
                "type": "message",
                "text": "What is the capital of France?",
                "conversation": { "id": "conv-42" }
            }
        })
    );

    // Request 3: second turn, same conversation id -- proves continuity was
    // carried on the wire, not just in the in-memory thread state.
    assert_eq!(
        requests[2].path,
        "/conversations/conv-42?api-version=2022-03-01-preview"
    );
    assert_eq!(
        requests[2].body_json()["activity"]["text"],
        serde_json::json!("What about tomorrow?")
    );
}

#[tokio::test]
async fn start_conversation_failure_surfaces_service_error() {
    let (base_url, _handle) = serve_sequence(vec![Box::new(|stream| {
        // No activity carrying a conversation id at all.
        write_sse_activity_response(stream, &serde_json::json!({"type": "event"}));
    })]);

    let connection =
        CopilotStudioConnectionSettings::new("unused", "unused").with_direct_connect_url(base_url);
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new("tok"));

    let err = agent.run_once("hi").await.unwrap_err();
    assert!(err
        .to_string()
        .contains("Failed to start a new conversation"));
}

#[tokio::test]
async fn non_success_status_surfaces_as_service_error() {
    let (base_url, _handle) = serve_sequence(vec![Box::new(|stream| {
        let payload = "unauthorized";
        let response = format!(
            "HTTP/1.1 401 Unauthorized\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            payload.len(),
            payload
        );
        stream.write_all(response.as_bytes()).unwrap();
        stream.flush().unwrap();
    })]);

    let connection =
        CopilotStudioConnectionSettings::new("unused", "unused").with_direct_connect_url(base_url);
    let agent = CopilotStudioAgent::new(connection, StaticTokenProvider::new("bad-token"));

    let err = agent.run_once("hi").await.unwrap_err();
    assert_eq!(err.status(), Some(401));
}
