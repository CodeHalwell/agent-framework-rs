//! Hermetic loopback tests for `pause_turn` continuation and server-tool
//! replay (upstream #9718), driven against a scripted fake Messages API: each
//! request is answered with the next canned body, and every request body is
//! recorded so the *replayed* conversation can be inspected.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_anthropic::AnthropicClient;
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, Message};
use futures::StreamExt;
use serde_json::{json, Value};

fn read_request_body(stream: &mut TcpStream) -> Value {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read headers");
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos;
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let len: usize = headers
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let start = header_end + 4;
    while buf.len() < start + len {
        let n = stream.read(&mut chunk).expect("read body");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    serde_json::from_slice(&buf[start..start + len]).unwrap_or(Value::Null)
}

/// A fake Messages API answering each request with the next scripted body
/// (repeating the last one once the script runs out).
struct ScriptedServer {
    addr: String,
    requests: Arc<Mutex<Vec<Value>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ScriptedServer {
    fn start(sse: bool, script: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let (requests_bg, stop_bg) = (requests.clone(), stop.clone());
        let handle = std::thread::spawn(move || {
            let mut next = 0usize;
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        requests_bg
                            .lock()
                            .unwrap()
                            .push(read_request_body(&mut stream));
                        let body = &script[next.min(script.len() - 1)];
                        next += 1;
                        let ctype = if sse {
                            "text/event-stream"
                        } else {
                            "application/json"
                        };
                        let head = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        stream.write_all(head.as_bytes()).unwrap();
                        stream.write_all(body.as_bytes()).unwrap();
                        stream.flush().unwrap();
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept: {e}"),
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

    fn requests(&self) -> Vec<Value> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for ScriptedServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn client(addr: &str) -> AnthropicClient {
    AnthropicClient::new("test-key", "claude-test").with_base_url(addr)
}

/// A turn the API paused mid-way through a server-side web search: the call,
/// its result (with the `encrypted_content` the API checks on replay), and no
/// answer yet.
fn paused_search_turn() -> String {
    json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [
            { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
              "input": { "query": "rust" } },
            { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
              "content": [{ "type": "web_search_result", "url": "https://example.com",
                            "title": "Example", "encrypted_content": "ENC" }] }
        ],
        "stop_reason": "pause_turn",
        "usage": { "input_tokens": 10, "output_tokens": 5 }
    })
    .to_string()
}

fn finished_turn() -> String {
    json!({
        "id": "msg_2", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{ "type": "text", "text": "Rust is a systems language." }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 20, "output_tokens": 7 }
    })
    .to_string()
}

/// The role and blocks of the last message of a recorded request body.
fn last_message_blocks(request: &Value) -> (String, Vec<Value>) {
    let last = request["messages"].as_array().unwrap().last().unwrap();
    (
        last["role"].as_str().unwrap().to_string(),
        last["content"].as_array().unwrap().clone(),
    )
}

#[tokio::test]
async fn a_paused_turn_is_continued_until_the_model_finishes() {
    // Before #9718 `pause_turn` was mapped to `stop`, so the caller got the
    // half-finished turn — a search with no answer after it — as a final
    // response, and only one request was ever made.
    let server = ScriptedServer::start(false, vec![paused_search_turn(), finished_turn()]);
    let response = client(&server.addr)
        .get_response(vec![Message::user("what is rust?")], ChatOptions::default())
        .await
        .unwrap();

    assert_eq!(
        server.requests().len(),
        2,
        "the paused turn must be continued"
    );
    assert!(
        response.text().contains("Rust is a systems language."),
        "the continuation's answer must reach the caller: {:?}",
        response.text()
    );
    // One turn to the caller: the search call and its result stay with the
    // answer, and usage is summed across both requests.
    let contents: Vec<&Content> = response.messages.iter().flat_map(|m| &m.contents).collect();
    assert!(contents
        .iter()
        .any(|c| matches!(c, Content::FunctionCall(fc) if fc.call_id == "srvtoolu_1")));
    assert!(contents
        .iter()
        .any(|c| matches!(c, Content::FunctionResult(fr) if fr.call_id == "srvtoolu_1")));
    let usage = response.usage_details.expect("usage");
    assert_eq!(usage.input_token_count, Some(30));
    assert_eq!(usage.output_token_count, Some(12));
}

#[tokio::test]
async fn the_continuation_replays_the_paused_turn_as_server_tool_blocks() {
    let server = ScriptedServer::start(false, vec![paused_search_turn(), finished_turn()]);
    client(&server.addr)
        .get_response(vec![Message::user("what is rust?")], ChatOptions::default())
        .await
        .unwrap();

    let second = &server.requests()[1];
    let (role, blocks) = last_message_blocks(second);
    assert_eq!(
        role, "assistant",
        "the paused turn is the trailing assistant turn"
    );
    assert_eq!(blocks[0]["type"], "server_tool_use", "{blocks:?}");
    assert_eq!(blocks[0]["id"], "srvtoolu_1");
    assert_eq!(blocks[0]["input"], json!({ "query": "rust" }));
    assert_eq!(blocks[1]["type"], "web_search_tool_result", "{blocks:?}");
    assert_eq!(blocks[1]["tool_use_id"], "srvtoolu_1");
    // The API validates `encrypted_content` on replay, so it must come back
    // byte for byte.
    assert_eq!(blocks[1]["content"][0]["encrypted_content"], "ENC");
    // And never as client tool blocks, which the API rejects in this turn.
    let types: Vec<&str> = second["messages"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|m| m["content"].as_array().unwrap())
        .filter_map(|b| b["type"].as_str())
        .collect();
    assert!(
        !types.contains(&"tool_use") && !types.contains(&"tool_result"),
        "{types:?}"
    );
}

#[tokio::test]
async fn continuation_stops_at_the_limit() {
    // A turn that never finishes is continued ten times and then returned as
    // it stands, rather than looping for ever.
    let server = ScriptedServer::start(false, vec![paused_search_turn()]);
    client(&server.addr)
        .get_response(vec![Message::user("q")], ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(
        server.requests().len(),
        11,
        "one request plus ten continuations"
    );
}

#[tokio::test]
async fn a_turn_that_cannot_be_replayed_faithfully_is_returned_paused() {
    // A server tool this client does not replay verbatim would come back
    // reshaped, and the model would resume from a turn that is not the one
    // it paused. The paused response is returned instead.
    let paused_unknown = json!({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-test",
        "content": [{ "type": "server_tool_use", "id": "srvtoolu_9",
                      "name": "tool_search_tool_regex", "input": {} }],
        "stop_reason": "pause_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    })
    .to_string();
    let server = ScriptedServer::start(false, vec![paused_unknown, finished_turn()]);
    client(&server.addr)
        .get_response(vec![Message::user("q")], ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(server.requests().len(), 1);
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("event: {}\ndata: {e}\n\n", e["type"].as_str().unwrap()))
        .collect()
}

#[tokio::test]
async fn a_paused_stream_is_continued_and_its_streamed_input_survives_replay() {
    // The streaming path rebuilds the paused turn from its updates before
    // replaying it, so the server tool's input — which streams in as
    // `input_json_delta` fragments after an empty `content_block_start` —
    // has to survive that rebuild (the other half of #9718).
    let paused = sse(&[
        json!({ "type": "message_start", "message": { "id": "msg_1", "model": "claude-test",
                "usage": { "input_tokens": 10, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block":
                { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {} } }),
        json!({ "type": "content_block_delta", "index": 0, "delta":
                { "type": "input_json_delta", "partial_json": "{\"query\": \"ru" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta":
                { "type": "input_json_delta", "partial_json": "st\"}" } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "content_block_start", "index": 1, "content_block":
                { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
                  "content": [{ "type": "web_search_result", "url": "https://example.com",
                                "title": "Example", "encrypted_content": "ENC" }] } }),
        json!({ "type": "content_block_stop", "index": 1 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "pause_turn" },
                "usage": { "output_tokens": 5 } }),
        json!({ "type": "message_stop" }),
    ]);
    let finished = sse(&[
        json!({ "type": "message_start", "message": { "id": "msg_2", "model": "claude-test",
                "usage": { "input_tokens": 20, "output_tokens": 1 } } }),
        json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        json!({ "type": "content_block_delta", "index": 0, "delta":
                { "type": "text_delta", "text": "Rust is a systems language." } }),
        json!({ "type": "content_block_stop", "index": 0 }),
        json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" },
                "usage": { "output_tokens": 7 } }),
        json!({ "type": "message_stop" }),
    ]);
    let server = ScriptedServer::start(true, vec![paused, finished]);

    let updates: Vec<_> = client(&server.addr)
        .get_streaming_response(vec![Message::user("what is rust?")], ChatOptions::default())
        .await
        .unwrap()
        .collect()
        .await;
    let updates: Vec<_> = updates.into_iter().map(Result::unwrap).collect();
    let response = ChatResponse::from_updates(updates);

    assert_eq!(
        server.requests().len(),
        2,
        "the paused stream must be continued"
    );
    assert!(
        response.text().contains("Rust is a systems language."),
        "{:?}",
        response.text()
    );

    let (role, blocks) = last_message_blocks(&server.requests()[1]);
    assert_eq!(role, "assistant");
    assert_eq!(blocks[0]["type"], "server_tool_use", "{blocks:?}");
    assert_eq!(
        blocks[0]["input"],
        json!({ "query": "rust" }),
        "the streamed input must be reassembled, not replayed empty"
    );
    assert_eq!(blocks[1]["type"], "web_search_tool_result");
    assert_eq!(blocks[1]["content"][0]["encrypted_content"], "ENC");
}
