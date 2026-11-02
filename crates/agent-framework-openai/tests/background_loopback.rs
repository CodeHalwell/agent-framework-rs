//! Hermetic loopback tests for background responses on
//! [`OpenAIChatClient`]: a hand-rolled fake HTTP server on a bare
//! `std::net::TcpListener` (the style of `embeddings_loopback.rs`) checks the
//! background submit, polling by continuation token
//! (`GET /responses/{id}`), and resuming a stream
//! (`GET /responses/{id}?stream=true&starting_after=N`).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, ContinuationToken, Message,
};
use agent_framework_openai::responses::{OpenAIChatClient, OpenAIContinuationToken};
use futures::StreamExt;
use serde_json::{json, Value};

/// One scripted reply: content type and body.
type Reply = (&'static str, String);

/// Serve each reply in turn, one connection per request, recording each raw
/// request (request line, headers and body).
fn scripted_server(replies: Vec<Reply>) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_writer = seen.clone();
    std::thread::spawn(move || {
        for (content_type, body) in replies {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            let (mut header_end, mut content_length) = (None, 0usize);
            loop {
                let n = stream.read(&mut chunk).expect("read request");
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                if header_end.is_none() {
                    if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = Some(pos);
                        let headers = String::from_utf8_lossy(&buf[..pos]).to_ascii_lowercase();
                        content_length = headers
                            .lines()
                            .find_map(|l| l.strip_prefix("content-length:"))
                            .and_then(|v| v.trim().parse().ok())
                            .unwrap_or(0);
                    }
                }
                if let Some(pos) = header_end {
                    if buf.len() >= pos + 4 + content_length {
                        break;
                    }
                }
            }
            seen_writer
                .lock()
                .unwrap()
                .push(String::from_utf8_lossy(&buf).to_string());
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).expect("write");
        }
    });
    (format!("http://{addr}"), seen)
}

fn json_reply(value: Value) -> Reply {
    ("application/json", value.to_string())
}

fn sse_reply(events: &[Value]) -> Reply {
    let mut out = String::new();
    for event in events {
        out.push_str(&format!(
            "event: {}\ndata: {event}\n\n",
            event["type"].as_str().unwrap()
        ));
    }
    ("text/event-stream", out)
}

fn request_line(raw: &str) -> &str {
    raw.lines().next().unwrap_or_default()
}

fn request_body(raw: &str) -> Value {
    let body = raw.split("\r\n\r\n").nth(1).unwrap_or_default();
    serde_json::from_str(body).unwrap_or(Value::Null)
}

fn client(base_url: String) -> OpenAIChatClient {
    OpenAIChatClient::new("sk-test", "gpt-test").with_base_url(base_url)
}

fn completed_response(id: &str, text: &str) -> Value {
    json!({
        "id": id,
        "model": "gpt-test",
        "status": "completed",
        "output": [{
            "type": "message", "id": "msg_1", "role": "assistant",
            "content": [{ "type": "output_text", "text": text, "annotations": [] }]
        }],
        "usage": { "input_tokens": 3, "output_tokens": 2, "total_tokens": 5 }
    })
}

#[tokio::test]
async fn a_background_request_is_submitted_then_polled_by_its_token() {
    let (base_url, seen) = scripted_server(vec![
        json_reply(json!({ "id": "resp_bg", "status": "queued", "output": [] })),
        json_reply(json!({ "id": "resp_bg", "status": "in_progress", "output": [] })),
        json_reply(completed_response("resp_bg", "all done")),
    ]);
    let client = client(base_url);

    let queued = client
        .get_response(
            vec![Message::user("think hard")],
            ChatOptions::new().with_background(true),
        )
        .await
        .unwrap();
    let token = queued.continuation_token.clone().expect("a pending token");
    assert_eq!(
        OpenAIContinuationToken::from_token(&token).unwrap(),
        OpenAIContinuationToken::new("resp_bg")
    );
    assert!(queued.finish_reason.is_none(), "not finished yet");

    let poll = |token: ContinuationToken| {
        client.get_response(
            Vec::new(),
            ChatOptions::new()
                .with_background(true)
                .with_continuation_token(token),
        )
    };
    let running = poll(token.clone()).await.unwrap();
    assert_eq!(running.continuation_token, Some(token.clone()));

    let done = poll(token).await.unwrap();
    assert!(done.continuation_token.is_none());
    assert_eq!(done.text(), "all done");
    assert_eq!(
        done.finish_reason.as_ref().map(|r| r.as_str()),
        Some("stop")
    );

    let seen = seen.lock().unwrap();
    assert_eq!(request_line(&seen[0]), "POST /responses HTTP/1.1");
    let body = request_body(&seen[0]);
    assert_eq!(body["background"], json!(true));
    assert_eq!(request_line(&seen[1]), "GET /responses/resp_bg HTTP/1.1");
    assert_eq!(request_line(&seen[2]), "GET /responses/resp_bg HTTP/1.1");
    assert!(seen[1]
        .to_ascii_lowercase()
        .contains("authorization: bearer sk-test"));
}

#[tokio::test]
async fn an_ordinary_request_sends_no_background_flag() {
    let (base_url, seen) = scripted_server(vec![json_reply(completed_response("r", "hi"))]);
    let response = client(base_url)
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();
    assert!(response.continuation_token.is_none());
    assert!(request_body(&seen.lock().unwrap()[0])
        .get("background")
        .is_none());
}

async fn collect(stream: agent_framework_core::client::ChatStream) -> Vec<ChatResponseUpdate> {
    stream.map(|u| u.expect("update")).collect().await
}

#[tokio::test]
async fn an_interrupted_background_stream_resumes_after_its_last_event() {
    let (base_url, seen) = scripted_server(vec![
        // The first stream drops after the first text delta.
        sse_reply(&[
            json!({ "type": "response.created", "sequence_number": 0,
                    "response": { "id": "resp_s", "model": "gpt-test", "status": "queued" } }),
            json!({ "type": "response.in_progress", "sequence_number": 1,
                    "response": { "id": "resp_s", "status": "in_progress" } }),
            json!({ "type": "response.output_text.delta", "sequence_number": 2,
                    "item_id": "msg_1", "delta": "Hel" }),
        ]),
        // The resumed stream picks up after event 2.
        sse_reply(&[
            json!({ "type": "response.output_text.delta", "sequence_number": 3,
                    "item_id": "msg_1", "delta": "lo" }),
            json!({ "type": "response.completed", "sequence_number": 4,
                    "response": { "id": "resp_s", "model": "gpt-test", "status": "completed",
                                  "output": [],
                                  "usage": { "input_tokens": 1, "output_tokens": 2,
                                             "total_tokens": 3 } } }),
        ]),
    ]);
    let client = client(base_url);

    let first = collect(
        client
            .get_streaming_response(
                vec![Message::user("say hello")],
                ChatOptions::new().with_background(true),
            )
            .await
            .unwrap(),
    )
    .await;
    let last = first.last().unwrap();
    let token = last.continuation_token.clone().expect("resumable");
    assert_eq!(
        OpenAIContinuationToken::from_token(&token).unwrap(),
        OpenAIContinuationToken {
            response_id: "resp_s".into(),
            sequence_number: Some(2),
        }
    );
    let partial = ChatResponse::from_updates(first);
    assert_eq!(partial.text(), "Hel");
    assert_eq!(partial.continuation_token, Some(token.clone()));
    assert_eq!(partial.conversation_id.as_deref(), Some("resp_s"));

    let rest = collect(
        client
            .get_streaming_response(
                Vec::new(),
                ChatOptions::new().with_continuation_token(token),
            )
            .await
            .unwrap(),
    )
    .await;
    assert!(rest[0].continuation_token.is_some());
    let finished = ChatResponse::from_updates(rest);
    assert_eq!(finished.text(), "lo");
    assert!(
        finished.continuation_token.is_none(),
        "the response completed"
    );
    assert_eq!(finished.conversation_id.as_deref(), Some("resp_s"));
    assert_eq!(
        finished.finish_reason.as_ref().map(|r| r.as_str()),
        Some("stop")
    );

    let seen = seen.lock().unwrap();
    assert_eq!(request_line(&seen[0]), "POST /responses HTTP/1.1");
    let body = request_body(&seen[0]);
    assert_eq!(body["background"], json!(true));
    assert_eq!(body["stream"], json!(true));
    assert_eq!(
        request_line(&seen[1]),
        "GET /responses/resp_s?stream=true&starting_after=2 HTTP/1.1"
    );
}

#[tokio::test]
async fn an_ordinary_stream_carries_no_continuation_token() {
    let (base_url, _seen) = scripted_server(vec![sse_reply(&[
        json!({ "type": "response.created", "sequence_number": 0,
                "response": { "id": "resp_n", "status": "in_progress" } }),
        json!({ "type": "response.output_text.delta", "sequence_number": 1,
                "item_id": "m", "delta": "hi" }),
    ])]);
    let updates = collect(
        client(base_url)
            .get_streaming_response(vec![Message::user("hi")], ChatOptions::new())
            .await
            .unwrap(),
    )
    .await;
    assert!(updates.iter().all(|u| u.continuation_token.is_none()));
}

#[test]
fn a_token_is_read_back_from_json_or_a_bare_response_id() {
    let token = OpenAIContinuationToken {
        response_id: "resp_1".into(),
        sequence_number: Some(7),
    };
    assert_eq!(
        OpenAIContinuationToken::from_token(&token.to_token()).unwrap(),
        token
    );
    assert_eq!(
        OpenAIContinuationToken::from_token(&ContinuationToken::new("resp_2")).unwrap(),
        OpenAIContinuationToken::new("resp_2")
    );
    assert!(OpenAIContinuationToken::from_token(&ContinuationToken::new("{\"x\":1}")).is_err());
    assert!(OpenAIContinuationToken::from_token(&ContinuationToken::new("  ")).is_err());
}
