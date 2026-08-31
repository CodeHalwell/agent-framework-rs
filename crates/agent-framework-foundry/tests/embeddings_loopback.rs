//! Hermetic loopback tests for [`FoundryEmbeddingClient`]: a hand-rolled fake
//! Foundry Models server on a bare `std::net::TcpListener` exercises the real
//! `reqwest` path end to end — the outbound URL shape
//! (`{endpoint}/embeddings?api-version=`), both auth headers, the request
//! body, and the OpenAI-shaped response parsing this client shares with
//! `agent-framework-openai`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::types::EmbeddingGenerationOptions;
use agent_framework_foundry::FoundryEmbeddingClient;

/// One recorded request: its start-line, headers (lowercased names), body.
#[derive(Clone, Debug)]
struct Recorded {
    start_line: String,
    headers: HashMap<String, String>,
    body: String,
}

/// Serve exactly one request with `(status, body)`, recording the request.
fn one_shot_server(status: u16, body: &'static str) -> (String, Arc<Mutex<Option<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(None));
    let seen_writer = seen.clone();
    std::thread::spawn(move || {
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
        let raw = String::from_utf8_lossy(&buf).to_string();
        let (head, req_body) = raw.split_once("\r\n\r\n").unwrap_or((raw.as_str(), ""));
        let mut lines = head.lines();
        let start_line = lines.next().unwrap_or_default().to_string();
        let headers = lines
            .filter_map(|l| l.split_once(':'))
            .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
            .collect();
        *seen_writer.lock().unwrap() = Some(Recorded {
            start_line,
            headers,
            body: req_body.to_string(),
        });

        let reason = if status == 200 { "OK" } else { "ERR" };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        );
        stream.write_all(response.as_bytes()).expect("write");
    });
    (format!("http://{addr}"), seen)
}

/// Two vectors returned out of order, each tagged with its input `index` —
/// the response parser must sort them back.
const OUT_OF_ORDER_BODY: &str = r#"{
  "data": [
    {"index": 1, "embedding": [2.0, 2.5]},
    {"index": 0, "embedding": [1.0, 1.5]}
  ],
  "model": "text-embedding-3-small",
  "usage": {"prompt_tokens": 7, "total_tokens": 7}
}"#;

#[tokio::test]
async fn posts_to_the_models_embeddings_route_with_the_api_key_header() {
    let (endpoint, seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let client = FoundryEmbeddingClient::new(
        format!("{endpoint}/models"),
        "text-embedding-3-small",
        "secret-key",
    );

    let batch = client
        .get_embeddings(vec!["alpha".into(), "beta".into()], None)
        .await
        .expect("embeddings");

    let req = seen.lock().unwrap().clone().expect("one request");
    assert!(
        req.start_line
            .starts_with("POST /models/embeddings?api-version="),
        "unexpected request line: {}",
        req.start_line
    );
    // Foundry Models is key-authenticated with `api-key`, not `Authorization`.
    assert_eq!(
        req.headers.get("api-key").map(String::as_str),
        Some("secret-key")
    );
    assert!(!req.headers.contains_key("authorization"));

    let body: serde_json::Value = serde_json::from_str(&req.body).expect("json body");
    assert_eq!(
        body,
        serde_json::json!({ "input": ["alpha", "beta"], "model": "text-embedding-3-small" })
    );

    // The service answered out of order; `index` puts it back.
    assert_eq!(batch.embeddings.len(), 2);
    assert_eq!(batch.embeddings[0].vector, vec![1.0, 1.5]);
    assert_eq!(batch.embeddings[1].vector, vec![2.0, 2.5]);
    assert_eq!(
        batch.embeddings[0].model.as_deref(),
        Some("text-embedding-3-small")
    );

    let usage = batch.usage.expect("usage reported");
    assert_eq!(usage.input_token_count, Some(7));
    assert_eq!(usage.total_token_count, Some(7));
}

#[tokio::test]
async fn a_token_credential_authenticates_with_a_bearer_header() {
    let (endpoint, seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let credential = Arc::new(StaticTokenCredential::new("entra-token"));
    let client = FoundryEmbeddingClient::with_credential(
        format!("{endpoint}/models"),
        "text-embedding-3-small",
        credential,
    );

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let req = seen.lock().unwrap().clone().expect("one request");
    assert_eq!(
        req.headers.get("authorization").map(String::as_str),
        Some("Bearer entra-token")
    );
    assert!(!req.headers.contains_key("api-key"));
}

/// A credential that records which audience it was asked for.
#[derive(Clone, Default)]
struct RecordingCredential {
    scopes: Arc<Mutex<Vec<String>>>,
}

#[async_trait::async_trait]
impl agent_framework_azure::TokenCredential for RecordingCredential {
    async fn get_token(&self) -> agent_framework_core::error::Result<String> {
        self.scopes.lock().unwrap().push("<unscoped>".into());
        Ok("unscoped-token".into())
    }

    async fn get_token_for_scope(
        &self,
        scope: &str,
    ) -> agent_framework_core::error::Result<String> {
        self.scopes.lock().unwrap().push(scope.to_string());
        Ok("scoped-token".into())
    }
}

/// The Models inference data plane wants the cognitive-services audience, not
/// the `https://ai.azure.com/.default` project audience the Responses API
/// uses. Asking a real credential for the wrong one yields a token the service
/// rejects, and the failure looks like a permissions problem rather than a
/// scope bug — so this pins that the client requests the scope explicitly
/// rather than falling through to the credential's own default.
#[tokio::test]
async fn the_credential_is_asked_for_the_models_inference_scope() {
    let (endpoint, _seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let credential = RecordingCredential::default();
    let client = FoundryEmbeddingClient::with_credential(
        format!("{endpoint}/models"),
        "m",
        Arc::new(credential.clone()),
    );

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let scopes = credential.scopes.lock().unwrap().clone();
    assert_eq!(
        scopes,
        vec!["https://cognitiveservices.azure.com/.default".to_string()],
        "must request the Models data-plane audience, not the credential default"
    );
}

#[tokio::test]
async fn an_overridden_scope_reaches_the_credential() {
    let (endpoint, _seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let credential = RecordingCredential::default();
    let client = FoundryEmbeddingClient::with_credential(
        format!("{endpoint}/models"),
        "m",
        Arc::new(credential.clone()),
    )
    .with_scope("https://sovereign.example/.default");

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    assert_eq!(
        credential.scopes.lock().unwrap().clone(),
        vec!["https://sovereign.example/.default".to_string()]
    );
}

#[tokio::test]
async fn a_response_without_a_model_falls_back_to_the_requested_one() {
    const NO_MODEL_BODY: &str = r#"{"data": [{"index": 0, "embedding": [0.5]}]}"#;
    let (endpoint, _seen) = one_shot_server(200, NO_MODEL_BODY);
    let client = FoundryEmbeddingClient::new(format!("{endpoint}/models"), "client-default", "k");

    let mut options = EmbeddingGenerationOptions::new();
    options.model = Some("per-request-model".into());

    let batch = client
        .get_embeddings(vec!["alpha".into()], Some(options))
        .await
        .expect("embeddings");

    // Upstream stamps `response.model or text_model`: the vector stays
    // identifiable even when the service omits the model.
    assert_eq!(
        batch.embeddings[0].model.as_deref(),
        Some("per-request-model")
    );
}

#[tokio::test]
async fn an_overridden_api_version_reaches_the_query_string() {
    let (endpoint, seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let client = FoundryEmbeddingClient::new(format!("{endpoint}/models"), "m", "k")
        .with_api_version("2099-01-01");

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let req = seen.lock().unwrap().clone().expect("one request");
    assert_eq!(
        req.start_line,
        "POST /models/embeddings?api-version=2099-01-01 HTTP/1.1"
    );
}

/// Expanding `extra_parameters` into the body is only half the job: Azure AI
/// Inference rejects body fields outside its schema unless the
/// `extra-parameters: pass-through` header opts into forwarding them, which
/// is what the SDK sets alongside `model_extras`. Expanding without the
/// header would turn a call that works upstream into a 4xx.
#[tokio::test]
async fn expanded_extras_carry_the_pass_through_header() {
    let (endpoint, seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let client = FoundryEmbeddingClient::new(format!("{endpoint}/models"), "m", "k");

    let mut options = EmbeddingGenerationOptions::new();
    options
        .additional_properties
        .insert("extra_parameters".into(), serde_json::json!({ "knob": 1 }));

    client
        .get_embeddings(vec!["alpha".into()], Some(options))
        .await
        .expect("embeddings");

    let req = seen.lock().unwrap().clone().expect("one request");
    assert_eq!(
        req.headers.get("extra-parameters").map(String::as_str),
        Some("pass-through")
    );
    let body: serde_json::Value = serde_json::from_str(&req.body).expect("json body");
    assert_eq!(body["knob"], serde_json::json!(1));
    assert!(body.get("extra_parameters").is_none());
}

#[tokio::test]
async fn no_extras_means_no_pass_through_header() {
    let (endpoint, seen) = one_shot_server(200, OUT_OF_ORDER_BODY);
    let client = FoundryEmbeddingClient::new(format!("{endpoint}/models"), "m", "k");

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let req = seen.lock().unwrap().clone().expect("one request");
    assert!(
        !req.headers.contains_key("extra-parameters"),
        "the header must not ride along on ordinary requests"
    );
}

#[tokio::test]
async fn a_429_is_classified_as_a_retryable_service_status() {
    const THROTTLED: &str = r#"{"error": {"message": "rate limited"}}"#;
    let (endpoint, _seen) = one_shot_server(429, THROTTLED);
    let client = FoundryEmbeddingClient::new(format!("{endpoint}/models"), "m", "k");

    let err = client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect_err("429 must surface");

    match err {
        agent_framework_core::error::Error::ServiceStatus { status, .. } => {
            assert_eq!(status, 429);
        }
        other => panic!("expected ServiceStatus, got {other:?}"),
    }
}

#[tokio::test]
async fn empty_input_short_circuits_without_a_request() {
    // Nothing is listening on port 1, so any request at all would fail.
    let client = FoundryEmbeddingClient::new("http://127.0.0.1:1/models", "m", "k");
    let batch = client.get_embeddings(Vec::new(), None).await.expect("ok");
    assert!(batch.embeddings.is_empty());
    assert!(batch.usage.is_none());
}
