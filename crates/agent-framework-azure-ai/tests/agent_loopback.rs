//! Hermetic loopback tests for [`AzureAIAgentClient`] driven against a
//! hand-rolled fake Azure AI Agents server built on a bare
//! `std::net::TcpListener` — no external process, no real network. Exercises
//! the real `reqwest` path end to end: agent/thread/message/run creation, the
//! non-streaming poll fallback, thread continuity, the tool round-trip state
//! machine (`requires_action` → `submit_tool_outputs`), and SSE streaming.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::StaticTokenCredential;
use agent_framework_azure_ai::AzureAIAgentClient;
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{
    ChatMessage, ChatOptions, Content, FinishReason, FunctionResultContent, Role,
};
use futures::StreamExt;
use serde_json::{json, Value};

#[derive(Clone)]
struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

impl Request {
    fn route(&self) -> &str {
        self.path.split('?').next().unwrap_or(&self.path)
    }
    fn body_str(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
    fn body_json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

struct Response {
    status: String,
    event_stream: bool,
    body: String,
}

impl Response {
    fn json(body: impl Into<String>) -> Self {
        Self {
            status: "200 OK".into(),
            event_stream: false,
            body: body.into(),
        }
    }
    fn sse(body: impl Into<String>) -> Self {
        Self {
            status: "200 OK".into(),
            event_stream: true,
            body: body.into(),
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_http_request(stream: &mut TcpStream) -> Request {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read headers");
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut parts = header_str.lines().next().unwrap_or_default().split(' ');
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
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
        let n = stream.read(&mut chunk).expect("read body");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf
        .get(body_start..body_start + content_length)
        .unwrap_or(&[])
        .to_vec();
    Request { method, path, body }
}

fn write_response(stream: &mut TcpStream, resp: &Response) {
    let head = if resp.event_stream {
        format!(
            "HTTP/1.1 {}\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n",
            resp.status
        )
    } else {
        format!(
            "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            resp.status,
            resp.body.len()
        )
    };
    stream.write_all(head.as_bytes()).expect("write head");
    stream.write_all(resp.body.as_bytes()).expect("write body");
    stream.flush().expect("flush");
}

struct MockServer {
    addr: String,
    requests: Arc<Mutex<Vec<Request>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl MockServer {
    fn start(handler: impl Fn(&Request) -> Response + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let handler = Arc::new(handler);

        let requests_bg = requests.clone();
        let stop_bg = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).expect("blocking");
                        let req = read_http_request(&mut stream);
                        let resp = handler(&req);
                        requests_bg.lock().unwrap().push(req);
                        write_response(&mut stream, &resp);
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

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn client(endpoint: &str) -> AzureAIAgentClient {
    AzureAIAgentClient::new(
        endpoint,
        "gpt-4o",
        Arc::new(StaticTokenCredential::new("test-token")),
    )
}

/// Default routing for the common agent/thread/message endpoints. The caller
/// supplies the `/runs` and `/submit_tool_outputs` responses per scenario.
fn route_default(req: &Request) -> Option<Response> {
    match (req.method.as_str(), req.route()) {
        ("POST", p) if p.ends_with("/assistants") => Some(Response::json(r#"{"id":"asst_1"}"#)),
        ("POST", p) if p.ends_with("/threads") => Some(Response::json(r#"{"id":"thread_1"}"#)),
        ("POST", p) if p.ends_with("/messages") => Some(Response::json(r#"{"id":"msg_1"}"#)),
        ("GET", p) if p.ends_with("/messages") => Some(Response::json(
            r#"{"data":[{"role":"assistant","content":[{"type":"text","text":{"value":"Hello from the agent"}}]}]}"#,
        )),
        _ => None,
    }
}

#[tokio::test]
async fn non_streaming_run_creates_thread_and_returns_text() {
    let server = MockServer::start(|req| {
        if let Some(r) = route_default(req) {
            return r;
        }
        match (req.method.as_str(), req.route()) {
            ("POST", p) if p.ends_with("/runs") => Response::json(
                r#"{"id":"run_1","status":"completed","usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}"#,
            ),
            _ => Response::json("{}"),
        }
    });

    let c = client(&server.addr);
    let resp = c
        .get_response(vec![ChatMessage::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    assert_eq!(resp.text(), "Hello from the agent");
    assert_eq!(resp.conversation_id.as_deref(), Some("thread_1"));
    assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
    assert_eq!(resp.usage_details.unwrap().total_token_count, Some(12));

    // The run body targeted the auto-created assistant.
    let runs = server
        .requests()
        .into_iter()
        .find(|r| r.method == "POST" && r.route().ends_with("/runs"))
        .unwrap();
    assert_eq!(runs.body_json()["assistant_id"], json!("asst_1"));
}

#[tokio::test]
async fn thread_continuity_reuses_conversation_id() {
    let server = MockServer::start(|req| {
        route_default(req).unwrap_or_else(|| {
            if req.method == "POST" && req.route().ends_with("/runs") {
                Response::json(r#"{"id":"run_1","status":"completed"}"#)
            } else {
                Response::json("{}")
            }
        })
    });

    let c = client(&server.addr);

    // First turn: no conversation id → a thread is created and surfaced.
    let first = c
        .get_response(vec![ChatMessage::user("hi")], ChatOptions::new())
        .await
        .unwrap();
    let thread_id = first.conversation_id.clone().unwrap();
    assert_eq!(thread_id, "thread_1");

    // Second turn: carry the conversation id back → no new thread, no new agent.
    let mut options = ChatOptions::new();
    options.conversation_id = Some(thread_id.clone());
    let _ = c
        .get_response(vec![ChatMessage::user("again")], options)
        .await
        .unwrap();

    let reqs = server.requests();
    let thread_creates = reqs
        .iter()
        .filter(|r| r.method == "POST" && r.route().ends_with("/threads"))
        .count();
    let agent_creates = reqs
        .iter()
        .filter(|r| r.method == "POST" && r.route().ends_with("/assistants"))
        .count();
    assert_eq!(thread_creates, 1, "thread should be created exactly once");
    assert_eq!(agent_creates, 1, "agent should be created exactly once");

    // The second run's messages were posted to the existing thread.
    let second_msg = reqs
        .iter()
        .filter(|r| r.method == "POST" && r.route().contains("/threads/thread_1/messages"))
        .count();
    assert!(
        second_msg >= 2,
        "each turn posts a user message to the thread"
    );
}

#[tokio::test]
async fn tool_round_trip_submits_outputs_to_active_run() {
    // The first run requires a tool call; after outputs are submitted the run
    // completes.
    let server = MockServer::start(|req| {
        route_default(req).unwrap_or_else(|| {
            match (req.method.as_str(), req.route()) {
                ("POST", p) if p.ends_with("/submit_tool_outputs") => {
                    Response::json(r#"{"id":"run_1","status":"completed"}"#)
                }
                ("POST", p) if p.ends_with("/runs") => Response::json(
                    r#"{"id":"run_1","status":"requires_action","required_action":{"type":"submit_tool_outputs","submit_tool_outputs":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}}]}}}"#,
                ),
                _ => Response::json("{}"),
            }
        })
    });

    let c = client(&server.addr);

    // Turn 1: model asks for a tool call.
    let first = c
        .get_response(vec![ChatMessage::user("weather?")], ChatOptions::new())
        .await
        .unwrap();
    assert_eq!(first.finish_reason, Some(FinishReason::tool_calls()));
    let calls = first.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    let call_id = calls[0].call_id.clone();
    let thread_id = first.conversation_id.clone().unwrap();

    // Turn 2: hand back the tool result on the same conversation.
    let tool_msg = ChatMessage::with_contents(
        Role::tool(),
        vec![Content::FunctionResult(FunctionResultContent::new(
            call_id,
            Some(json!("sunny, 25C")),
        ))],
    );
    let mut options = ChatOptions::new();
    options.conversation_id = Some(thread_id);
    let second = c.get_response(vec![tool_msg], options).await.unwrap();
    assert_eq!(second.text(), "Hello from the agent");

    // The tool output was submitted to the active run with the bare call id.
    let submit = server
        .requests()
        .into_iter()
        .find(|r| r.method == "POST" && r.route().ends_with("/submit_tool_outputs"))
        .expect("submit_tool_outputs was called");
    assert!(
        submit
            .route()
            .contains("/threads/thread_1/runs/run_1/submit_tool_outputs"),
        "route: {}",
        submit.route()
    );
    let body = submit.body_json();
    assert_eq!(body["tool_outputs"][0]["tool_call_id"], json!("call_1"));
    assert_eq!(body["tool_outputs"][0]["output"], json!("sunny, 25C"));
}

#[tokio::test]
async fn streaming_run_yields_text_and_usage() {
    let sse = concat!(
        "event: thread.run.created\ndata: {\"id\":\"run_1\",\"model\":\"gpt-4o\"}\n\n",
        "event: thread.message.delta\ndata: {\"id\":\"msg_1\",\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":{\"value\":\"Hi \"}}]}}\n\n",
        "event: thread.message.delta\ndata: {\"id\":\"msg_1\",\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":{\"value\":\"there\"}}]}}\n\n",
        "event: thread.run.step.completed\ndata: {\"id\":\"step_1\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "event: thread.run.completed\ndata: {\"id\":\"run_1\",\"status\":\"completed\"}\n\n",
        "event: done\ndata: [DONE]\n\n",
    );
    let server = MockServer::start(move |req| {
        route_default(req).unwrap_or_else(|| {
            if req.method == "POST" && req.route().ends_with("/runs") {
                assert!(
                    req.body_str().contains("\"stream\":true"),
                    "streaming run body must set stream:true"
                );
                Response::sse(sse)
            } else {
                Response::json("{}")
            }
        })
    });

    let c = client(&server.addr);
    let mut stream = c
        .get_streaming_response(vec![ChatMessage::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let mut text = String::new();
    let mut conversation_id = None;
    let mut total_tokens = None;
    while let Some(update) = stream.next().await {
        let update = update.unwrap();
        text.push_str(&update.text_content());
        conversation_id = conversation_id.or(update.conversation_id.clone());
        for content in &update.contents {
            if let Content::Usage(u) = content {
                total_tokens = u.details.total_token_count;
            }
        }
    }

    assert_eq!(text, "Hi there");
    assert_eq!(conversation_id.as_deref(), Some("thread_1"));
    assert_eq!(total_tokens, Some(7));
}
