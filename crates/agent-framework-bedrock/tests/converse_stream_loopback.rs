//! `ConverseStream` against a loopback server that speaks the AWS binary
//! event-stream framing. The client used to send the non-streaming
//! `converse` request and replay the finished response as one update.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use agent_framework_bedrock::eventstream::encode_message;
use agent_framework_bedrock::BedrockChatClient;
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{ChatOptions, ChatResponse, Content, FinishReason, Message};
use futures::StreamExt;
use serde_json::{json, Value};

fn event(kind: &str, payload: Value) -> Vec<u8> {
    encode_message(
        &[
            (":message-type", "event"),
            (":event-type", kind),
            (":content-type", "application/json"),
        ],
        payload.to_string().as_bytes(),
    )
}

/// Serve exactly one request with `body`, recording its request line.
fn serve_once(body: Vec<u8>) -> (String, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(String::new()));
    let seen_bg = seen.clone();
    std::thread::spawn(move || {
        let (mut s, _): (TcpStream, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let n = s.read(&mut chunk).unwrap();
            buf.extend_from_slice(&chunk[..n]);
            if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..p]).to_string();
                let len: usize = head
                    .to_ascii_lowercase()
                    .lines()
                    .find_map(|l| {
                        l.strip_prefix("content-length:")
                            .map(|v| v.trim().parse().unwrap())
                    })
                    .unwrap_or(0);
                while buf.len() < p + 4 + len {
                    let n = s.read(&mut chunk).unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                }
                *seen_bg.lock().unwrap() = head.lines().next().unwrap().to_string();
                break;
            }
        }
        let head = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.amazon.eventstream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        s.write_all(head.as_bytes()).unwrap();
        s.write_all(&body).unwrap();
    });
    (addr, seen)
}

fn client(addr: &str) -> BedrockChatClient {
    BedrockChatClient::new("AKID", "secret", "us-east-1", "anthropic.claude")
        .with_endpoint(addr)
        .unwrap()
}

#[tokio::test]
async fn converse_stream_delivers_text_tool_calls_reasoning_usage_and_stop_reason() {
    let mut body = Vec::new();
    for e in [
        event("messageStart", json!({ "role": "assistant" })),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 0, "delta": { "reasoningContent": { "text": "think" } } }),
        ),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 0, "delta": { "reasoningContent": { "signature": "SIG" } } }),
        ),
        event("contentBlockStop", json!({ "contentBlockIndex": 0 })),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 1, "delta": { "text": "Hel" } }),
        ),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 1, "delta": { "text": "lo" } }),
        ),
        event("contentBlockStop", json!({ "contentBlockIndex": 1 })),
        event(
            "contentBlockStart",
            json!({ "contentBlockIndex": 2, "start": { "toolUse": { "toolUseId": "tu_1", "name": "get_weather" } } }),
        ),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 2, "delta": { "toolUse": { "input": "{\"city\":" } } }),
        ),
        event(
            "contentBlockDelta",
            json!({ "contentBlockIndex": 2, "delta": { "toolUse": { "input": "\"Paris\"}" } } }),
        ),
        event("contentBlockStop", json!({ "contentBlockIndex": 2 })),
        event("messageStop", json!({ "stopReason": "tool_use" })),
        event(
            "metadata",
            json!({ "usage": { "inputTokens": 7, "outputTokens": 9, "totalTokens": 16 } }),
        ),
    ] {
        body.extend(e);
    }
    let (addr, seen) = serve_once(body);
    let updates: Vec<_> = client(&addr)
        .get_streaming_response(vec![Message::user("weather?")], ChatOptions::default())
        .await
        .unwrap()
        .map(|u| u.unwrap())
        .collect()
        .await;

    assert!(
        seen.lock().unwrap().contains("/converse-stream"),
        "{}",
        seen.lock().unwrap()
    );
    assert!(
        updates.len() > 3,
        "incremental updates, not one: {}",
        updates.len()
    );

    let response = ChatResponse::from_updates(updates);
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.finish_reason, Some(FinishReason::tool_calls()));
    let usage = response.usage_details.clone().unwrap();
    assert_eq!(
        (usage.input_token_count, usage.output_token_count),
        (Some(7), Some(9))
    );

    let calls = response.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].call_id, "tu_1");
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].parse_arguments().unwrap().get("city"),
        Some(&json!("Paris"))
    );

    // The streamed reasoning keeps its signature, so the turn replays: the
    // next request carries `reasoningText` with that signature.
    let reasoning = response
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .find_map(|c| match c {
            Content::TextReasoning(r) => Some(r.clone()),
            _ => None,
        })
        .expect("reasoning");
    assert_eq!(reasoning.text, "think");
    assert_eq!(reasoning.protected_data.as_deref(), Some("SIG"));
    let replay = agent_framework_bedrock::convert::build_request(
        &[Message::user("weather?"), response.messages[0].clone()],
        &ChatOptions::default(),
        "anthropic.claude",
    );
    let block = &replay["messages"][1]["content"][0]["reasoningContent"]["reasoningText"];
    assert_eq!(block, &json!({ "text": "think", "signature": "SIG" }));
}

#[tokio::test]
async fn a_converse_stream_exception_is_the_streams_error() {
    let mut body = event(
        "contentBlockDelta",
        json!({ "contentBlockIndex": 0, "delta": { "text": "par" } }),
    );
    body.extend(encode_message(
        &[
            (":message-type", "exception"),
            (":exception-type", "modelStreamErrorException"),
        ],
        br#"{"message":"model failed"}"#,
    ));
    let (addr, _) = serve_once(body);
    let items: Vec<_> = client(&addr)
        .get_streaming_response(vec![Message::user("q")], ChatOptions::default())
        .await
        .unwrap()
        .collect()
        .await;
    assert!(
        items[0].is_ok(),
        "the text before the failure still arrives"
    );
    let err = items.into_iter().find_map(Result::err).expect("an error");
    assert!(
        err.to_string().contains("modelStreamErrorException"),
        "{err}"
    );
}
