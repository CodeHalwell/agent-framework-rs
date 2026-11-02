//! Real streaming for Anthropic on Vertex AI, Azure AI Foundry and Amazon
//! Bedrock, against a scripted loopback server. Each of these used to send a
//! non-streaming request and replay the finished response as one update.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_anthropic::{
    AnthropicBedrockClient, AnthropicFoundryClient, AnthropicVertexClient, StaticVertexToken,
};
use agent_framework_azure::StaticTokenCredential;
use agent_framework_bedrock::eventstream::encode_message;
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
use base64::Engine as _;
use futures::StreamExt;
use serde_json::{json, Value};

/// One recorded request: (request line, body JSON).
type Seen = (String, Value);

struct Server {
    addr: String,
    seen: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
}

fn read(stream: &mut TcpStream) -> Seen {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let end = loop {
        let n = stream.read(&mut chunk).unwrap();
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if n == 0 {
            break buf.len();
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).to_string();
    let len: usize = head
        .to_ascii_lowercase()
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    while buf.len() < end + 4 + len {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    (
        head.lines().next().unwrap_or_default().to_string(),
        serde_json::from_slice(&buf[end + 4..end + 4 + len]).unwrap_or(Value::Null),
    )
}

impl Server {
    /// Answer each request with the next `(content type, body)`, repeating
    /// the last once the script runs out.
    fn start(script: Vec<(&'static str, Vec<u8>)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (seen_bg, stop_bg) = (seen.clone(), stop.clone());
        std::thread::spawn(move || {
            let mut i = 0usize;
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).unwrap();
                        seen_bg.lock().unwrap().push(read(&mut s));
                        let (ctype, body) = &script[i.min(script.len() - 1)];
                        i += 1;
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        s.write_all(head.as_bytes()).unwrap();
                        s.write_all(body).unwrap();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("{e}"),
                }
            }
        });
        Self { addr, seen, stop }
    }
    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

/// Messages API events for a turn answering `parts` as separate text deltas
/// and stopping for `stop_reason`.
fn events(parts: &[&str], stop_reason: &str) -> Vec<Value> {
    let mut out = vec![
        json!({ "type": "message_start", "message": { "id": "msg_1", "model": "claude",
                "usage": { "input_tokens": 3, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
    ];
    for p in parts {
        out.push(json!({ "type": "content_block_delta", "index": 0,
                         "delta": { "type": "text_delta", "text": p } }));
    }
    out.push(json!({ "type": "content_block_stop", "index": 0 }));
    out.push(
        json!({ "type": "message_delta", "delta": { "stop_reason": stop_reason },
                     "usage": { "output_tokens": 5 } }),
    );
    out.push(json!({ "type": "message_stop" }));
    out
}

fn sse(events: &[Value]) -> (&'static str, Vec<u8>) {
    let body: String = events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect();
    ("text/event-stream", body.into_bytes())
}

fn bedrock_frames(events: &[Value]) -> (&'static str, Vec<u8>) {
    let mut body = Vec::new();
    for e in events {
        let b64 = base64::engine::general_purpose::STANDARD.encode(e.to_string());
        let payload = json!({ "bytes": b64 }).to_string();
        body.extend(encode_message(
            &[
                (":message-type", "event"),
                (":event-type", "chunk"),
                (":content-type", "application/json"),
            ],
            payload.as_bytes(),
        ));
    }
    ("application/vnd.amazon.eventstream", body)
}

async fn collect(stream: agent_framework_core::client::ChatStream) -> Vec<ChatResponseUpdate> {
    stream.map(|u| u.expect("update")).collect().await
}

fn text_updates(updates: &[ChatResponseUpdate]) -> usize {
    updates
        .iter()
        .filter(|u| {
            u.contents
                .iter()
                .any(|c| c.as_text().is_some_and(|t| !t.is_empty()))
        })
        .count()
}

#[tokio::test]
async fn vertex_streams_from_stream_raw_predict() {
    let server = Server::start(vec![sse(&events(&["Hel", "lo"], "end_turn"))]);
    let client = AnthropicVertexClient::new(
        "proj",
        "us-east5",
        "claude",
        Arc::new(StaticVertexToken::new("t")),
    )
    .with_endpoint(&server.addr);
    let updates = collect(
        client
            .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
            .await
            .unwrap(),
    )
    .await;
    assert!(
        text_updates(&updates) >= 2,
        "text arrives incrementally, not as one update"
    );
    assert_eq!(ChatResponse::from_updates(updates).text(), "Hello");
    let (line, body) = &server.seen()[0];
    assert!(line.contains(":streamRawPredict"), "{line}");
    assert_eq!(body["stream"], true);
}

#[tokio::test]
async fn foundry_streams_sse() {
    let server = Server::start(vec![sse(&events(&["Hel", "lo"], "end_turn"))]);
    let client = AnthropicFoundryClient::with_token_credential(
        &server.addr,
        "claude",
        Arc::new(StaticTokenCredential::new("t")),
    );
    let updates = collect(
        client
            .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
            .await
            .unwrap(),
    )
    .await;
    assert!(text_updates(&updates) >= 2);
    assert_eq!(ChatResponse::from_updates(updates).text(), "Hello");
    assert_eq!(server.seen()[0].1["stream"], true);
}

fn bedrock(addr: &str) -> AnthropicBedrockClient {
    AnthropicBedrockClient::new("AKID", "secret", "us-east-1", "anthropic.claude")
        .with_endpoint(addr)
        .unwrap()
}

#[tokio::test]
async fn bedrock_streams_the_aws_event_stream() {
    let server = Server::start(vec![bedrock_frames(&events(&["Hel", "lo"], "end_turn"))]);
    let updates = collect(
        bedrock(&server.addr)
            .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
            .await
            .unwrap(),
    )
    .await;
    assert!(text_updates(&updates) >= 2);
    assert_eq!(ChatResponse::from_updates(updates).text(), "Hello");
    let (line, body) = &server.seen()[0];
    assert!(line.contains("/invoke-with-response-stream"), "{line}");
    // On Bedrock the route selects streaming; the body must not.
    assert!(body.get("stream").is_none(), "{body}");
}

#[tokio::test]
async fn a_bedrock_stream_exception_ends_the_stream_with_its_error() {
    let mut body = bedrock_frames(&events(&["partial"], "end_turn")).1;
    body.truncate(0);
    body.extend(encode_message(
        &[
            (":message-type", "exception"),
            (":exception-type", "throttlingException"),
        ],
        br#"{"message":"slow down"}"#,
    ));
    let server = Server::start(vec![("application/vnd.amazon.eventstream", body)]);
    let items: Vec<_> = bedrock(&server.addr)
        .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .unwrap()
        .collect()
        .await;
    let err = items
        .into_iter()
        .find_map(Result::err)
        .expect("an error item");
    assert!(err.to_string().contains("throttlingException"), "{err}");
}

#[tokio::test]
async fn a_paused_bedrock_stream_is_continued() {
    let server = Server::start(vec![
        bedrock_frames(&events(&["first "], "pause_turn")),
        bedrock_frames(&events(&["second"], "end_turn")),
    ]);
    let updates = collect(
        bedrock(&server.addr)
            .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(server.seen().len(), 2, "the paused turn is continued");
    assert_eq!(ChatResponse::from_updates(updates).text(), "first second");
}

#[test]
fn a_bedrock_endpoint_must_be_http_with_a_host() {
    let c = || AnthropicBedrockClient::new("a", "b", "us-east-1", "m");
    assert!(c().with_endpoint("ftp://x").is_err());
    assert!(c().with_endpoint("https://").is_err());
    assert!(c()
        .with_endpoint("https://vpce-123.bedrock-runtime.us-east-1.vpce.amazonaws.com")
        .is_ok());
}
