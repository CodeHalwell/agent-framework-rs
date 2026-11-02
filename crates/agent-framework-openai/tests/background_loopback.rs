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
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, ContinuationToken, FunctionArguments,
    Message,
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
            sequence_number: Some(2),
            ..OpenAIContinuationToken::new("resp_s")
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
        sequence_number: Some(7),
        function_calls: [(1, "call_1".to_string())].into(),
        ..OpenAIContinuationToken::new("resp_1")
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

fn function_call_item(arguments: &str) -> Value {
    json!({ "type": "function_call", "id": "fc_1", "call_id": "call_1",
            "name": "lookup", "arguments": arguments })
}

/// The arguments of the one function call in `response`, joined.
fn call_arguments(response: &ChatResponse) -> (usize, String, String) {
    let calls: Vec<_> = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| match c {
            Content::FunctionCall(fc) => Some(fc),
            _ => None,
        })
        .collect();
    let raw = match calls.first().and_then(|c| c.arguments.as_ref()) {
        Some(FunctionArguments::Raw(raw)) => raw.clone(),
        Some(other) => serde_json::to_string(other).unwrap(),
        None => String::new(),
    };
    (
        calls.len(),
        calls.first().map(|c| c.name.clone()).unwrap_or_default(),
        raw,
    )
}

fn call_stream_tail() -> Vec<Value> {
    vec![
        json!({ "type": "response.function_call_arguments.delta", "sequence_number": 4,
                "item_id": "fc_1", "output_index": 0, "delta": "\"paris\"}" }),
        json!({ "type": "response.function_call_arguments.done", "sequence_number": 5,
                "item_id": "fc_1", "output_index": 0,
                "arguments": "{\"city\":\"paris\"}" }),
        json!({ "type": "response.output_item.done", "sequence_number": 6,
                "output_index": 0, "item": function_call_item("{\"city\":\"paris\"}") }),
        json!({ "type": "response.completed", "sequence_number": 7,
                "response": { "id": "resp_f", "status": "completed",
                              "output": [function_call_item("{\"city\":\"paris\"}")] } }),
    ]
}

#[tokio::test]
async fn a_function_call_interrupted_mid_arguments_resumes_with_all_of_them() {
    let (base_url, seen) = scripted_server(vec![
        // Cut after the call was announced and part of its arguments sent.
        sse_reply(&[
            json!({ "type": "response.created", "sequence_number": 0,
                    "response": { "id": "resp_f", "status": "queued" } }),
            json!({ "type": "response.output_item.added", "sequence_number": 1,
                    "output_index": 0, "item": function_call_item("") }),
            json!({ "type": "response.function_call_arguments.delta", "sequence_number": 2,
                    "item_id": "fc_1", "output_index": 0, "delta": "{\"city\":" }),
        ]),
        sse_reply(&call_stream_tail()),
    ]);
    let client = client(base_url);
    let first = collect(
        client
            .get_streaming_response(
                vec![Message::user("weather?")],
                ChatOptions::new().with_background(true),
            )
            .await
            .unwrap(),
    )
    .await;
    let token = first.last().unwrap().continuation_token.clone().unwrap();
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
    assert!(request_line(&seen.lock().unwrap()[1]).contains("starting_after=2"));

    // The caller holding both connections' updates has the whole call, once.
    let combined = ChatResponse::from_updates(first.into_iter().chain(rest).collect());
    assert_eq!(
        call_arguments(&combined),
        (1, "lookup".into(), "{\"city\":\"paris\"}".into())
    );
}

#[tokio::test]
async fn a_function_call_never_announced_to_a_resumed_stream_comes_from_its_done_item() {
    // Resumed after the announcement with no record of it (a token carrying
    // only the response id and position): the done item supplies the call.
    let (base_url, _seen) = scripted_server(vec![sse_reply(&call_stream_tail())]);
    let token = OpenAIContinuationToken {
        sequence_number: Some(3),
        ..OpenAIContinuationToken::new("resp_f")
    };
    let rest = collect(
        client(base_url)
            .get_streaming_response(
                Vec::new(),
                ChatOptions::new().with_continuation_token(token.to_token()),
            )
            .await
            .unwrap(),
    )
    .await;
    let response = ChatResponse::from_updates(rest);
    assert_eq!(
        call_arguments(&response),
        (1, "lookup".into(), "{\"city\":\"paris\"}".into())
    );
}

#[tokio::test]
async fn a_cancelled_background_stream_ends_the_operation() {
    let (base_url, _seen) = scripted_server(vec![sse_reply(&[
        json!({ "type": "response.created", "sequence_number": 0,
                "response": { "id": "resp_c", "model": "gpt-test", "status": "queued" } }),
        json!({ "type": "response.output_text.delta", "sequence_number": 1,
                "item_id": "msg_1", "delta": "partial" }),
        json!({ "type": "response.cancelled", "sequence_number": 2,
                "response": { "id": "resp_c", "model": "gpt-test", "status": "cancelled",
                              "output": [] } }),
    ])]);
    let updates = collect(
        client(base_url)
            .get_streaming_response(
                vec![Message::user("go")],
                ChatOptions::new().with_background(true),
            )
            .await
            .unwrap(),
    )
    .await;
    let last = updates.last().unwrap();
    assert!(
        last.continuation_token.is_none(),
        "cancellation is terminal"
    );
    let aggregated = ChatResponse::from_updates(updates);
    assert!(
        aggregated.continuation_token.is_none(),
        "a cancelled operation is not pending"
    );
    assert_eq!(aggregated.text(), "partial");
}
