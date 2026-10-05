//! Hermetic loopback test for [`GeminiEmbeddingClient`]: a fake
//! `batchEmbedContents` endpoint on a bare `std::net::TcpListener` exercises
//! the real `reqwest` path end to end.
//!
//! The unit tests in `embeddings.rs` pin the request body and the response
//! parse separately. What they cannot show is that the two meet: that a call
//! through the `EmbeddingClient` trait puts the *task-prefixed* text on the
//! wire, authenticates by header, and returns vectors paired with the inputs
//! that produced them. The harness is `agent-framework-openai`'s, so the two
//! embedding clients are exercised the same way.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::types::EmbeddingGenerationOptions;
use agent_framework_gemini::{GeminiEmbeddingClient, GeminiEmbeddingOptions, GeminiEmbeddingTask};

/// Serve exactly one request with `body`, recording the raw request bytes.
fn one_shot_server(status_and_body: (u16, String)) -> (String, Arc<Mutex<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(String::new()));
    let seen_writer = seen.clone();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        let mut buf = Vec::new();
        let mut chunk = [0u8; 4096];
        // Read until the full body arrived (headers + Content-Length bytes).
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
        *seen_writer.lock().unwrap() = String::from_utf8_lossy(&buf).to_string();
        let (status, body) = status_and_body;
        let reason = if status == 200 { "OK" } else { "ERR" };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        );
        stream.write_all(response.as_bytes()).expect("write");
    });
    (format!("http://{addr}"), seen)
}

#[tokio::test]
async fn a_round_trip_prefixes_the_text_authenticates_by_header_and_pairs_the_vectors() {
    let body = serde_json::json!({
        "embeddings": [
            { "values": [0.1, 0.2, 0.3] },
            { "values": [0.4, 0.5, 0.6] },
        ]
    })
    .to_string();
    let (base_url, seen) = one_shot_server((200, body));

    let client = GeminiEmbeddingClient::new("test-key").with_base_url(base_url);
    let batch = client
        .get_embeddings(
            vec!["chapter 4".into(), "chapter 5".into()],
            Some(
                EmbeddingGenerationOptions::new()
                    .with_task(GeminiEmbeddingTask::RetrievalDocument)
                    .with_title("Ownership"),
            ),
        )
        .await
        .expect("embeddings round trip");

    // Vectors come back in request order, attributed to the model that made
    // them.
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].vector, vec![0.1, 0.2, 0.3]);
    assert_eq!(batch[1].vector, vec![0.4, 0.5, 0.6]);
    assert_eq!(batch[0].model.as_deref(), Some("gemini-embedding-2"));

    let request = seen.lock().unwrap().clone();
    assert!(
        request.starts_with("POST /v1beta/models/gemini-embedding-2:batchEmbedContents HTTP/1.1"),
        "{request}"
    );
    // Header auth rather than `?key=`, so the key stays out of logs and
    // proxies — the same choice this crate's chat client documents.
    assert!(
        request
            .to_ascii_lowercase()
            .contains("x-goog-api-key: test-key"),
        "{request}"
    );
    assert!(
        !request.lines().next().unwrap_or_default().contains("key="),
        "the api key must not reach the URL: {request}"
    );
    // The task prefix is the capability itself: without it Embedding 2 is
    // conditioned on nothing and the vectors are quietly worse, which no
    // response field would reveal.
    assert!(
        request.contains(r#"title: Ownership | text: chapter 4"#),
        "{request}"
    );
    assert!(
        request.contains(r#"title: Ownership | text: chapter 5"#),
        "{request}"
    );
}

#[tokio::test]
async fn a_task_less_call_never_reaches_the_network() {
    // The refusal is client-side, so no request should be attempted: a round
    // trip that can only return a mis-conditioned vector is not worth
    // spending, and the server recording nothing is how that is visible.
    let (base_url, seen) = one_shot_server((200, r#"{"embeddings":[]}"#.to_string()));
    let client = GeminiEmbeddingClient::new("test-key").with_base_url(base_url);

    let err = client
        .get_embeddings(vec!["hi".into()], None)
        .await
        .expect_err("a task is required");
    assert!(err.to_string().contains("a task is required"), "{err}");
    assert!(
        seen.lock().unwrap().is_empty(),
        "nothing should have been sent"
    );
}

#[tokio::test]
async fn an_empty_input_list_short_circuits_without_a_request() {
    let (base_url, seen) = one_shot_server((200, r#"{"embeddings":[]}"#.to_string()));
    let client = GeminiEmbeddingClient::new("test-key").with_base_url(base_url);

    // Nothing to embed, and notably this does *not* trip the required-task
    // check: there is no text to condition, so there is nothing to get wrong.
    let batch = client.get_embeddings(vec![], None).await.unwrap();
    assert!(batch.is_empty());
    assert!(seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_service_error_is_classified_rather_than_returned_as_a_body() {
    let (base_url, _seen) = one_shot_server((
        429,
        serde_json::json!({ "error": { "code": 429, "message": "quota exceeded" } }).to_string(),
    ));
    let client = GeminiEmbeddingClient::new("test-key").with_base_url(base_url);

    // Shares the chat client's classifier, so a rate limit is a typed,
    // retryable error here too rather than an opaque string.
    let err = client
        .get_embeddings(
            vec!["hi".into()],
            Some(EmbeddingGenerationOptions::new().with_task(GeminiEmbeddingTask::RetrievalQuery)),
        )
        .await
        .expect_err("a 429 must be an error");
    assert!(err.to_string().contains("quota exceeded"), "{err}");
}
