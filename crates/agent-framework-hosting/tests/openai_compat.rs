//! OpenAI Chat Completions hosting tests: non-streaming JSON and streaming
//! chunk framing.

mod common;

use agent_framework_hosting::openai_compat::OpenAiRouter;
use axum::http::StatusCode;
use serde_json::json;

use common::{parse_sse, post_json, post_raw, CancelTrackingAgent, MockAgent, StreamingAgent};

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

// ---------------------------------------------------------------------------
// Streaming backpressure & disconnect cancellation (bounded channel)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_disconnect_cancels_the_agent_run() {
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use std::sync::atomic::Ordering;
    use std::time::Duration;
    use tower::ServiceExt;

    let agent = CancelTrackingAgent::new("a1");
    let cancelled = agent.cancelled();
    let produced = agent.produced();
    let app = OpenAiRouter::for_agent("assistant", agent.arc()).into_router();

    let body = json!({
        "model": "assistant",
        "messages": [{ "role": "user", "content": "hi" }],
        "stream": true,
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("request builds");

    let response = app.oneshot(request).await.expect("router responds");
    assert_eq!(response.status(), StatusCode::OK);

    // Read a couple of SSE frames so streaming has genuinely started, then
    // "disconnect" by dropping the response body.
    let mut resp_body = response.into_body();
    let _ = resp_body.frame().await;
    let _ = resp_body.frame().await;
    drop(resp_body);

    // The producer must observe the disconnect and drop the agent stream,
    // flipping the cancel flag. Poll briefly for the async task to react.
    let mut cancelled_observed = false;
    for _ in 0..200 {
        if cancelled.load(Ordering::SeqCst) {
            cancelled_observed = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(
        cancelled_observed,
        "the agent run must be cancelled when the client disconnects"
    );

    // Backpressure + cancellation means the producer stopped early — it did not
    // run away producing the full million-delta stream for a client that left.
    let count = produced.load(Ordering::SeqCst);
    assert!(
        count < 10_000,
        "producer kept generating after disconnect (produced {count})"
    );
}
