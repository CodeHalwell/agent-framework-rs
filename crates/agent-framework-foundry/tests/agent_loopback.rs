//! Hermetic loopback tests for [`FoundryChatClient`] driven against a
//! hand-rolled fake Foundry Responses server built on a bare
//! `std::net::TcpListener` — no external process, no real network. Reuses the
//! same fake-server pattern as `agent-framework-azure`'s
//! `tests/credentials_loopback.rs`, since [`FoundryChatClient`] delegates
//! straight through to `agent_framework_azure::responses::AzureOpenAIResponsesClient`:
//! these tests exercise that delegation end to end through the real
//! `reqwest` path — the outbound URL shape (`{endpoint}/openai/v1/responses`,
//! no `?api-version=` query), the `Authorization: Bearer` header from a
//! [`TokenCredential`], and both the non-streaming and streaming Responses
//! JSON/SSE shapes (text + a function tool call).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{ChatOptions, Content, Message};
use agent_framework_foundry::FoundryChatClient;
use futures::StreamExt;
use serde_json::Value;

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// One recorded request: `(request line, header block, body)`.
type Recorded = (String, String, Vec<u8>);

fn read_http_request(stream: &mut TcpStream) -> Recorded {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        if n == 0 {
            break buf.len();
        }
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
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf
        .get(body_start..body_start + content_length)
        .unwrap_or(&[])
        .to_vec();
    (request_line, header_str, body)
}

fn write_response(stream: &mut TcpStream, event_stream: bool, body: &str) {
    let head = if event_stream {
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            .to_string()
    } else {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
    };
    stream.write_all(head.as_bytes()).expect("write head");
    stream.write_all(body.as_bytes()).expect("write body");
    stream.flush().expect("flush");
}

/// A fake HTTP server that answers every request with the same canned body
/// and records the raw request strings.
struct FakeServer {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeServer {
    fn start(event_stream: bool, body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        listener.set_nonblocking(true).expect("set nonblocking");
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let requests_bg = requests.clone();
        let stop_bg = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).expect("blocking");
                        let req = read_http_request(&mut stream);
                        requests_bg.lock().unwrap().push(req);
                        write_response(&mut stream, event_stream, body);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept failed: {e}"),
                }
            }
        });

        Self {
            addr,
            requests,
            stop,
            handle: Some(handle),
        }
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn client(endpoint: &str) -> FoundryChatClient {
    FoundryChatClient::with_token_credential(
        endpoint,
        "gpt-4o",
        Arc::new(StaticTokenCredential::new("test-token")),
    )
}

/// Non-streaming round trip: a Responses JSON body with plain assistant text
/// parses through, and the outbound request hits the documented path-versioned
/// v1 route with a bearer token — no `?api-version=` query parameter.
#[tokio::test]
async fn non_streaming_round_trip_hits_v1_responses_route_with_bearer_auth() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_abc123","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello from Foundry!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#,
    );

    let c = client(&server.addr);
    let resp = c
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    assert_eq!(resp.text(), "Hello from Foundry!");
    assert_eq!(resp.response_id.as_deref(), Some("resp_abc123"));
    assert_eq!(resp.usage_details.unwrap().total_token_count, Some(15));

    let (line, headers, body) = server.requests().remove(0);
    assert_eq!(
        line, "POST /openai/v1/responses HTTP/1.1",
        "the Foundry v1 GA route is path-versioned with no api-version query"
    );
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("authorization: bearer test-token"),
        "headers: {headers}"
    );
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["model"], serde_json::json!("gpt-4o"));
    assert_eq!(
        body_json["input"],
        serde_json::json!([{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "hi" }
        ]}])
    );
}

/// A Responses body describing a function tool call round-trips into a
/// `FunctionCallContent`, proving the reused `agent_framework_openai`
/// conversion is wired all the way through `FoundryChatClient`.
#[tokio::test]
async fn tool_call_response_round_trips_into_function_call_content() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_call1","model":"gpt-4o","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}]}"#,
    );

    let c = client(&server.addr);
    let resp = c
        .get_response(vec![Message::user("weather?")], ChatOptions::new())
        .await
        .unwrap();

    let calls = resp.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].call_id, "call_1");
}

/// Streaming round trip: Responses SSE events parse into text deltas plus a
/// final `response.completed` usage/conversation-id update.
#[tokio::test]
async fn streaming_round_trip_yields_text_and_usage() {
    let sse_body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream_1\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n";
    let server = FakeServer::start(true, sse_body);

    let c = client(&server.addr);
    let mut stream = c
        .get_streaming_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let mut text = String::new();
    let mut total_tokens = None;
    while let Some(update) = stream.next().await {
        let update = update.expect("stream update should parse cleanly");
        text.push_str(&update.text_content());
        for content in &update.contents {
            if let Content::Usage(u) = content {
                total_tokens = u.details.total_token_count;
            }
        }
    }

    assert_eq!(text, "Hi");
    assert_eq!(total_tokens, Some(5));

    let (line, _headers, body) = server.requests().remove(0);
    assert_eq!(line, "POST /openai/v1/responses HTTP/1.1");
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["stream"], serde_json::json!(true));
}

/// `FoundryChatClient::new` (API-key auth) hits the same route with an
/// `api-key` header instead of `Authorization: Bearer`.
#[tokio::test]
async fn api_key_client_uses_api_key_header() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}"#,
    );

    let c = FoundryChatClient::new(&server.addr, "gpt-4o", "test-api-key");
    let _ = c
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let (_line, headers, _body) = server.requests().remove(0);
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("api-key: test-api-key"),
        "headers: {headers}"
    );
}
