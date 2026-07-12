//! Hermetic loopback tests for the new `A2AClient` methods (push
//! notification config, `resubscribe`, the authenticated extended card),
//! against a hand-rolled fake A2A server built directly on a bare
//! `std::net::TcpListener` — no external process, no real network. Mirrors
//! `agent-framework-mcp`'s `tests/http_loopback.rs`: exercises the real
//! `reqwest`-based HTTP path end to end rather than mocking it away.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use agent_framework_a2a::{A2AClient, PushNotificationAuthenticationInfo, PushNotificationConfig};
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

/// Read one HTTP/1.1 request's request line, headers, and
/// `Content-Length` body from `stream`.
fn read_http_request(stream: &mut TcpStream) -> (String, Vec<u8>) {
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
    let request_line = header_str.lines().next().unwrap_or_default().to_string();
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
    (
        request_line,
        buf[body_start..body_start + content_length].to_vec(),
    )
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

/// Write a complete `text/event-stream` response body in one shot (no
/// `Content-Length`; the connection close itself marks the end of the body,
/// which is what the client's incremental SSE reader relies on).
fn write_sse_response(stream: &mut TcpStream, events: &[Value]) {
    let mut body = String::new();
    for event in events {
        body.push_str("data: ");
        body.push_str(&event.to_string());
        body.push_str("\n\n");
    }
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n{body}"
    );
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

fn extract_jsonrpc_id(body: &[u8]) -> Value {
    let request: Value = serde_json::from_slice(body).expect("parse JSON-RPC request body");
    request.get("id").cloned().unwrap_or(Value::Null)
}

#[tokio::test]
async fn set_push_notification_config_round_trips_over_loopback() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let (_line, body) = read_http_request(&mut stream);
        let request: Value = serde_json::from_slice(&body).expect("parse JSON-RPC request body");
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some("tasks/pushNotificationConfig/set")
        );
        assert_eq!(request["params"]["taskId"], "task-1");
        assert_eq!(
            request["params"]["pushNotificationConfig"]["url"],
            "https://example.com/hook"
        );
        assert_eq!(
            request["params"]["pushNotificationConfig"]["authentication"]["schemes"][0],
            "Bearer"
        );
        let id = extract_jsonrpc_id(&body);
        write_json_response(
            &mut stream,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "taskId": "task-1",
                    "pushNotificationConfig": {
                        "id": "cfg-assigned",
                        "url": "https://example.com/hook",
                        "authentication": {"schemes": ["Bearer"], "credentials": "tok"},
                    },
                },
            }),
        );
    });

    let url = format!("http://{addr}/rpc");
    let client = A2AClient::from_url(url);
    let config = PushNotificationConfig::new("https://example.com/hook").with_authentication(
        PushNotificationAuthenticationInfo {
            schemes: vec!["Bearer".into()],
            credentials: Some("tok".into()),
        },
    );

    let result = tokio::time::timeout(
        Duration::from_secs(10),
        client.set_push_notification_config("task-1", config),
    )
    .await
    .expect("client call timed out")
    .expect("client call failed");

    assert_eq!(result.task_id, "task-1");
    assert_eq!(
        result.push_notification_config.id.as_deref(),
        Some("cfg-assigned")
    );
    assert_eq!(
        result.push_notification_config.url,
        "https://example.com/hook"
    );

    server.join().expect("loopback server thread panicked");
}

#[tokio::test]
async fn get_push_notification_config_sends_task_id_under_the_id_field() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let (_line, body) = read_http_request(&mut stream);
        let request: Value = serde_json::from_slice(&body).expect("parse JSON-RPC request body");
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some("tasks/pushNotificationConfig/get")
        );
        // The get params shape genuinely differs from set's: the task id is
        // sent under "id", not "taskId" -- a real A2A 0.3.0 spec/SDK
        // wire-level inconsistency, faithfully preserved by this port.
        assert_eq!(request["params"]["id"], "task-1");
        assert!(request["params"].get("taskId").is_none());
        let id = extract_jsonrpc_id(&body);
        write_json_response(
            &mut stream,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "taskId": "task-1",
                    "pushNotificationConfig": {"url": "https://example.com/hook"},
                },
            }),
        );
    });

    let url = format!("http://{addr}/rpc");
    let client = A2AClient::from_url(url);
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        client.get_push_notification_config("task-1"),
    )
    .await
    .expect("client call timed out")
    .expect("client call failed");

    assert_eq!(result.task_id, "task-1");
    assert_eq!(
        result.push_notification_config.url,
        "https://example.com/hook"
    );

    server.join().expect("loopback server thread panicked");
}

#[tokio::test]
async fn get_agent_card_auto_upgrades_to_extended_card_when_supported() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");
    let base_url = format!("http://{addr}");
    let base_url_for_server = base_url.clone();

    let server = std::thread::spawn(move || {
        // First connection: `.well-known/agent-card.json` discovery GET,
        // returning a base card that claims extended-card support.
        let mut stream = accept_with_timeout(&listener);
        let (line, _body) = read_http_request(&mut stream);
        assert!(
            line.contains("/.well-known/agent-card.json"),
            "expected a discovery GET, got: {line}"
        );
        write_json_response(
            &mut stream,
            &json!({
                "name": "Base Agent",
                "description": "the bare card",
                "url": base_url_for_server,
                "supportsAuthenticatedExtendedCard": true,
            }),
        );

        // Second connection: the client's automatic follow-up JSON-RPC call
        // for the extended card.
        let mut stream2 = accept_with_timeout(&listener);
        let (_line2, body2) = read_http_request(&mut stream2);
        let request: Value = serde_json::from_slice(&body2).expect("parse JSON-RPC request body");
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some("agent/getAuthenticatedExtendedCard")
        );
        let id = request.get("id").cloned().unwrap_or(Value::Null);
        write_json_response(
            &mut stream2,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "name": "Extended Agent",
                    "description": "the fuller, authenticated card",
                    "url": base_url_for_server,
                    "supportsAuthenticatedExtendedCard": true,
                },
            }),
        );
    });

    let client = A2AClient::from_url(base_url);
    let card = tokio::time::timeout(Duration::from_secs(10), client.get_agent_card())
        .await
        .expect("get_agent_card timed out")
        .expect("get_agent_card failed");

    assert_eq!(card.name, "Extended Agent");
    assert_eq!(card.description, "the fuller, authenticated card");
    // A second call must be fully cached -- no further network access, so
    // dropping the (already-exhausted) server thread here is safe.
    let cached = client.get_agent_card().await.unwrap();
    assert_eq!(cached.name, "Extended Agent");

    server.join().expect("loopback server thread panicked");
}

#[tokio::test]
async fn get_agent_card_falls_back_to_base_card_when_extended_fetch_fails() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");
    let base_url = format!("http://{addr}");
    let base_url_for_server = base_url.clone();

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let (_line, _body) = read_http_request(&mut stream);
        write_json_response(
            &mut stream,
            &json!({
                "name": "Base Agent",
                "url": base_url_for_server,
                "supportsAuthenticatedExtendedCard": true,
            }),
        );

        // Second connection: the extended-card call, this time answered
        // with a JSON-RPC error (e.g. missing/invalid auth) -- discovery as
        // a whole must still succeed, falling back to the base card.
        let mut stream2 = accept_with_timeout(&listener);
        let (_line2, body2) = read_http_request(&mut stream2);
        let id = extract_jsonrpc_id(&body2);
        write_json_response(
            &mut stream2,
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": {"code": -32001, "message": "authentication required"},
            }),
        );
    });

    let client = A2AClient::from_url(base_url);
    let card = tokio::time::timeout(Duration::from_secs(10), client.get_agent_card())
        .await
        .expect("get_agent_card timed out")
        .expect("get_agent_card should still succeed, falling back to the base card");

    assert_eq!(card.name, "Base Agent");

    server.join().expect("loopback server thread panicked");
}

#[tokio::test]
async fn resubscribe_streams_task_status_updates_over_a_real_loopback_socket() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let (_line, body) = read_http_request(&mut stream);
        let request: Value = serde_json::from_slice(&body).expect("parse JSON-RPC request body");
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some("tasks/resubscribe")
        );
        assert_eq!(request["params"]["id"], "task-1");
        let id = extract_jsonrpc_id(&body);
        write_sse_response(
            &mut stream,
            &[json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "taskId": "task-1",
                    "contextId": "ctx-1",
                    "status": {"state": "completed"},
                    "final": true,
                },
            })],
        );
    });

    let url = format!("http://{addr}/rpc");
    let client = A2AClient::from_url(url);

    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        use agent_framework_a2a::MessageStreamEvent;
        use futures::StreamExt;

        let mut stream = client.resubscribe("task-1").await.expect("resubscribe");
        let first = stream
            .next()
            .await
            .expect("stream ended without any event")
            .expect("event parse error");
        match first {
            MessageStreamEvent::StatusUpdate(update) => {
                assert_eq!(update.task_id, "task-1");
                assert!(update.is_final);
            }
            other => panic!("expected StatusUpdate, got {other:?}"),
        }
    })
    .await;

    outcome.expect("resubscribe_streams_task_status_updates_over_a_real_loopback_socket timed out");
    server.join().expect("loopback server thread panicked");
}

#[tokio::test]
async fn a2a_agent_run_stream_maps_sse_events_to_updates() {
    use agent_framework_a2a::{
        A2AAgent, AgentCard, Message, MessageRole, Part, SendMessageResult, TextPart,
    };
    use agent_framework_core::agent::SupportsAgentRun;
    use agent_framework_core::types::Message as CoreMessage;
    use futures::StreamExt;

    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");

    let server = std::thread::spawn(move || {
        let mut stream = accept_with_timeout(&listener);
        let (_line, body) = read_http_request(&mut stream);
        let request: Value = serde_json::from_slice(&body).expect("parse JSON-RPC request body");
        assert_eq!(
            request.get("method").and_then(Value::as_str),
            Some("message/stream"),
            "A2AAgent::run_stream must use the streaming endpoint"
        );
        let id = extract_jsonrpc_id(&body);
        let answer = Message {
            kind: "message".to_string(),
            role: MessageRole::Agent,
            parts: vec![Part::Text(TextPart {
                text: "The weather is sunny.".into(),
                metadata: None,
            })],
            message_id: "m1".into(),
            task_id: None,
            context_id: Some("ctx-1".into()),
            metadata: None,
        };
        write_sse_response(
            &mut stream,
            &[json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": serde_json::to_value(SendMessageResult::Message(answer)).unwrap(),
            })],
        );
    });

    // `from_card` caches the card, so run_stream performs no discovery GET —
    // the fake server only needs to serve the single `message/stream` POST.
    let card = AgentCard {
        name: "weather".into(),
        url: format!("http://{addr}/rpc"),
        ..Default::default()
    };
    let agent = A2AAgent::from_card("weather", card);

    let updates = tokio::time::timeout(Duration::from_secs(10), async {
        let mut stream =
            SupportsAgentRun::run_stream(&agent, vec![CoreMessage::user("weather?")], None, None)
                .await
                .expect("run_stream opens");
        let mut collected = Vec::new();
        while let Some(update) = stream.next().await {
            collected.push(update.expect("update ok"));
        }
        collected
    })
    .await
    .expect("run_stream timed out");

    server.join().expect("loopback server thread panicked");

    let text: String = updates.iter().map(|u| u.text()).collect();
    assert_eq!(text, "The weather is sunny.");
    assert_eq!(updates[0].author_name.as_deref(), Some("weather"));
}
