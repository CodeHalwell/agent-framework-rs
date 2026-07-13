//! Hermetic loopback test for [`OpenAIEmbeddingClient`]: a hand-rolled fake
//! HTTP server on a bare `std::net::TcpListener` exercises the real `reqwest`
//! path end to end (request shape, auth header, response parsing). Mirrors
//! `agent-framework-azure`'s `tests/credentials_loopback.rs`.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::types::EmbeddingGenerationOptions;
use agent_framework_openai::OpenAIEmbeddingClient;

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
async fn embeddings_request_and_response_round_trip() {
    let body = serde_json::json!({
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            { "object": "embedding", "index": 1, "embedding": [0.3, 0.4] },
            { "object": "embedding", "index": 0, "embedding": [0.1, 0.2] },
        ],
        "usage": { "prompt_tokens": 4, "total_tokens": 4 }
    })
    .to_string();
    let (base_url, seen) = one_shot_server((200, body));

    let client =
        OpenAIEmbeddingClient::new("sk-test", "text-embedding-3-small").with_base_url(base_url);
    let batch = client
        .get_embeddings(
            vec!["alpha".into(), "beta".into()],
            Some(EmbeddingGenerationOptions::new().with_dimensions(2)),
        )
        .await
        .expect("embeddings round trip");

    // Response parsed with input order restored via `index`.
    assert_eq!(batch.len(), 2);
    assert_eq!(batch[0].vector, vec![0.1, 0.2]);
    assert_eq!(batch[1].vector, vec![0.3, 0.4]);
    assert_eq!(batch[0].model.as_deref(), Some("text-embedding-3-small"));
    assert_eq!(batch.usage.as_ref().unwrap().input_token_count, Some(4));

    // Request went to /embeddings with bearer auth and the expected body.
    let request = seen.lock().unwrap().clone();
    assert!(
        request.starts_with("POST /embeddings HTTP/1.1"),
        "{request}"
    );
    assert!(
        request
            .to_ascii_lowercase()
            .contains("authorization: bearer sk-test"),
        "{request}"
    );
    assert!(
        request.contains("\"input\":[\"alpha\",\"beta\"]"),
        "{request}"
    );
    assert!(request.contains("\"dimensions\":2"), "{request}");
}

#[tokio::test]
async fn embeddings_service_error_maps_to_service_status() {
    let (base_url, _seen) = one_shot_server((429, r#"{"error":{"message":"slow down"}}"#.into()));
    let client = OpenAIEmbeddingClient::new("sk-test", "m").with_base_url(base_url);
    let err = client
        .get_embeddings(vec!["x".into()], None)
        .await
        .expect_err("429 must error");
    assert!(
        matches!(
            err,
            agent_framework_core::error::Error::ServiceStatus { status: 429, .. }
        ),
        "got: {err:?}"
    );
}
