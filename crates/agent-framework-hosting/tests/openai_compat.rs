//! OpenAI Chat Completions hosting tests: non-streaming JSON and streaming
//! chunk framing.

mod common;

use agent_framework_hosting::openai_compat::OpenAiRouter;
use axum::http::StatusCode;
use serde_json::json;

use common::{parse_sse, post_json, post_raw, MockAgent, StreamingAgent};

fn router() -> axum::Router {
    OpenAiRouter::for_agent("assistant", MockAgent::new("a1").with_usage(5, 3).arc()).into_router()
}

#[tokio::test]
async fn chat_completions_non_stream() {
    let body = json!({
        "model": "assistant",
        "messages": [
            { "role": "system", "content": "be terse" },
            { "role": "user", "content": "ping" },
        ],
    });
    let (status, resp) = post_json(router(), "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(resp["model"], "assistant");
    assert_eq!(resp["choices"][0]["index"], 0);
    assert_eq!(resp["choices"][0]["message"]["role"], "assistant");
    assert_eq!(
        resp["choices"][0]["message"]["content"],
        "echo: be terse ping"
    );
    assert_eq!(resp["choices"][0]["finish_reason"], "stop");
    // Usage flows through from the agent.
    assert_eq!(resp["usage"]["prompt_tokens"], 5);
    assert_eq!(resp["usage"]["completion_tokens"], 3);
    assert_eq!(resp["usage"]["total_tokens"], 8);
}

#[tokio::test]
async fn chat_completions_stream_chunks() {
    let body = json!({
        "model": "assistant",
        "stream": true,
        "messages": [{ "role": "user", "content": "ping" }],
    });
    let (status, text) = post_raw(router(), "/v1/chat/completions", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let payloads = parse_sse(&text);
    assert_eq!(payloads.last().unwrap(), "[DONE]");

    let chunks: Vec<serde_json::Value> = payloads
        .iter()
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();

    // Every chunk is a chat.completion.chunk with a consistent id.
    let id = chunks[0]["id"].as_str().unwrap().to_string();
    assert!(id.starts_with("chatcmpl-"));
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk");
        assert_eq!(c["id"], id);
    }

    // First chunk sets the assistant role.
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");

    // A content chunk carries the reply text.
    let content = chunks
        .iter()
        .find_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .unwrap();
    assert_eq!(content, "echo: ping");

    // The final chunk closes with finish_reason "stop".
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn chat_completions_content_parts_array() {
    // OpenAI clients may send content as an array of parts.
    let body = json!({
        "messages": [{
            "role": "user",
            "content": [{ "type": "text", "text": "hi there" }],
        }],
    });
    let (status, resp) = post_json(router(), "/v1/chat/completions", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["choices"][0]["message"]["content"], "echo: hi there");
}

#[tokio::test]
async fn chat_completions_stream_emits_incremental_content_chunks() {
    // A multi-delta streaming agent yields one content chunk per update.
    let router = OpenAiRouter::for_agent(
        "assistant",
        StreamingAgent::new("s1", vec!["Hel", "lo ", "world"]).arc(),
    )
    .into_router();

    let body = json!({
        "model": "assistant",
        "messages": [{ "role": "user", "content": "hi" }],
        "stream": true,
    });
    let (status, text) = post_raw(router, "/v1/chat/completions", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let data = parse_sse(&text);
    assert_eq!(data.last().unwrap(), "[DONE]");
    let chunks: Vec<serde_json::Value> = data
        .iter()
        .filter(|d| *d != "[DONE]")
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();

    // First chunk: role. Then one content chunk per delta. Last: finish_reason.
    let contents: Vec<&str> = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(contents, vec!["Hel", "lo ", "world"]);
    assert_eq!(
        chunks.first().unwrap()["choices"][0]["delta"]["role"],
        "assistant"
    );
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}
