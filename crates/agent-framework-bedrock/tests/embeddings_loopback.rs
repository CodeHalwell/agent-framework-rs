//! Hermetic loopback test for [`BedrockEmbeddingClient`]: a hand-rolled fake
//! HTTP server on a bare `std::net::TcpListener` exercises the real `reqwest`
//! path end to end (request path, SigV4 headers, Titan body shape, response
//! parsing, per-value fan-out and usage summing). Mirrors
//! `agent-framework-openai`'s `tests/embeddings_loopback.rs`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_bedrock::BedrockEmbeddingClient;
use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::types::EmbeddingGenerationOptions;

/// One recorded request: its start-line, headers (lowercased names) and body.
#[derive(Clone, Debug)]
struct Recorded {
    start_line: String,
    headers: HashMap<String, String>,
    body: String,
}

/// Read one full HTTP request (headers + `Content-Length` bytes) off `stream`.
fn read_request(stream: &mut std::net::TcpStream) -> Recorded {
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
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((raw.as_str(), ""));
    let mut lines = head.lines();
    let start_line = lines.next().unwrap_or_default().to_string();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    Recorded {
        start_line,
        headers,
        body: body.to_string(),
    }
}

/// Serve `count` requests, answering each with a vector derived from the
/// request's own `inputText` so responses cannot be correlated by arrival
/// order alone — which is what makes the ordering assertion meaningful.
fn embedding_server(count: usize) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_writer = seen.clone();
    std::thread::spawn(move || {
        for _ in 0..count {
            let (mut stream, _) = listener.accept().expect("accept");
            let recorded = read_request(&mut stream);

            // Echo a vector keyed to the input: "alpha" -> [1.0], "beta" -> [2.0].
            let parsed: serde_json::Value =
                serde_json::from_str(&recorded.body).unwrap_or(serde_json::Value::Null);
            let input = parsed
                .get("inputText")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let (vector, tokens) = match input.as_str() {
                "alpha" => ("[1.0, 1.5]", 3),
                "beta" => ("[2.0, 2.5]", 4),
                _ => ("[0.0, 0.0]", 0),
            };
            let body = format!(r#"{{"embedding": {vector}, "inputTextTokenCount": {tokens}}}"#);

            seen_writer.lock().unwrap().push(recorded);
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            stream.write_all(response.as_bytes()).expect("write");
        }
    });
    (format!("http://{addr}"), seen)
}

#[tokio::test]
async fn embeds_each_value_in_its_own_signed_request_and_keeps_input_order() {
    let (endpoint, seen) = embedding_server(2);
    let client = BedrockEmbeddingClient::new("AKIDEXAMPLE", "secret", "us-east-1", "m")
        .with_endpoint(&endpoint);

    let batch = client
        .get_embeddings(vec!["alpha".into(), "beta".into()], None)
        .await
        .expect("embeddings");

    // Titan takes one input per call, so two values means two requests.
    let recorded = seen.lock().unwrap().clone();
    assert_eq!(recorded.len(), 2, "one request per value");

    // Order is the *input* order, not the order the server happened to
    // answer in — the server keys its vectors off inputText for exactly this.
    assert_eq!(batch.embeddings.len(), 2);
    assert_eq!(batch.embeddings[0].vector, vec![1.0, 1.5]);
    assert_eq!(batch.embeddings[1].vector, vec![2.0, 2.5]);
    assert_eq!(batch.embeddings[0].model.as_deref(), Some("m"));

    // inputTextTokenCount is summed across the batch.
    let usage = batch.usage.expect("usage reported");
    assert_eq!(usage.input_token_count, Some(7));

    for req in &recorded {
        assert_eq!(req.start_line, "POST /model/m/invoke HTTP/1.1");
        let auth = req.headers.get("authorization").expect("signed");
        assert!(
            auth.starts_with("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"),
            "unexpected authorization header: {auth}"
        );
        assert!(
            auth.contains("SignedHeaders=host;x-amz-content-sha256;x-amz-date"),
            "unexpected signed headers: {auth}"
        );
        assert!(req.headers.contains_key("x-amz-date"));
        assert!(req.headers.contains_key("x-amz-content-sha256"));
        assert_eq!(
            req.headers.get("accept").map(String::as_str),
            Some("application/json")
        );
    }

    let bodies: Vec<&str> = recorded.iter().map(|r| r.body.as_str()).collect();
    assert!(bodies.contains(&r#"{"inputText":"alpha"}"#));
    assert!(bodies.contains(&r#"{"inputText":"beta"}"#));
}

#[tokio::test]
async fn per_request_model_and_titan_options_reach_the_wire() {
    let (endpoint, seen) = embedding_server(1);
    let client = BedrockEmbeddingClient::new("AKIDEXAMPLE", "secret", "us-east-1", "default-model")
        .with_endpoint(&endpoint);

    let mut options = EmbeddingGenerationOptions::new();
    options.model = Some("amazon.titan-embed-text-v2:0".into());
    options.dimensions = Some(256);
    options
        .additional_properties
        .insert("normalize".into(), serde_json::json!(true));

    let batch = client
        .get_embeddings(vec!["alpha".into()], Some(options))
        .await
        .expect("embeddings");

    let recorded = seen.lock().unwrap().clone();
    let req = recorded.first().expect("one request");

    // The `:` in the model id is percent-encoded in the path, and the same
    // encoding feeds the canonical URI the signature is computed over.
    assert_eq!(
        req.start_line,
        "POST /model/amazon.titan-embed-text-v2%3A0/invoke HTTP/1.1"
    );
    let body: serde_json::Value = serde_json::from_str(&req.body).expect("json body");
    assert_eq!(
        body,
        serde_json::json!({ "inputText": "alpha", "dimensions": 256, "normalize": true })
    );
    // The per-request model, not the client default, is stamped on the result.
    assert_eq!(
        batch.embeddings[0].model.as_deref(),
        Some("amazon.titan-embed-text-v2:0")
    );
}

#[tokio::test]
async fn a_session_token_is_signed_and_sent() {
    let (endpoint, seen) = embedding_server(1);
    let client = BedrockEmbeddingClient::new("AKIDEXAMPLE", "secret", "us-east-1", "m")
        .with_endpoint(&endpoint)
        .with_session_token("session-token-value");

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let recorded = seen.lock().unwrap().clone();
    let req = recorded.first().expect("one request");
    assert_eq!(
        req.headers.get("x-amz-security-token").map(String::as_str),
        Some("session-token-value")
    );
    let auth = req.headers.get("authorization").expect("signed");
    assert!(
        auth.contains("x-amz-security-token"),
        "session token must be inside SignedHeaders, not just sent: {auth}"
    );
}

/// An endpoint carrying a path prefix (a PrivateLink/VPC endpoint or a proxy
/// mounted under a sub-path) must have that prefix in *both* the requested
/// path and the canonical URI the signature covers. Signing the bare
/// `/model/.../invoke` while sending `/prefix/model/.../invoke` produces a
/// signature AWS rejects, and nothing about the sent request looks wrong —
/// so this recomputes the expected `Authorization` from the request's own
/// `x-amz-date` and body and compares it, rather than only eyeballing the
/// request line.
#[tokio::test]
async fn an_endpoint_path_prefix_is_signed_as_well_as_sent() {
    let (endpoint, seen) = embedding_server(1);
    let client = BedrockEmbeddingClient::new("AKIDEXAMPLE", "secret", "us-east-1", "m")
        .with_endpoint(format!("{endpoint}/prefix"));

    client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect("embeddings");

    let recorded = seen.lock().unwrap().clone();
    let req = recorded.first().expect("one request");
    assert_eq!(req.start_line, "POST /prefix/model/m/invoke HTTP/1.1");

    let host = req.headers.get("host").expect("host header").clone();
    let amz_date = req.headers.get("x-amz-date").expect("x-amz-date").clone();
    let expected = agent_framework_bedrock::sigv4::authorization_header(
        &agent_framework_bedrock::sigv4::SigV4Params {
            access_key: "AKIDEXAMPLE",
            secret_key: "secret",
            session_token: None,
            region: "us-east-1",
            service: "bedrock",
            host: &host,
            method: "POST",
            // The whole point: the prefix belongs in the canonical URI.
            canonical_uri: "/prefix/model/m/invoke",
            canonical_query: "",
            payload: req.body.as_bytes(),
            amz_date: &amz_date,
            date_stamp: &amz_date[..8],
        },
    )
    .0;
    assert_eq!(
        req.headers.get("authorization").map(String::as_str),
        Some(expected.as_str()),
        "the Authorization header must sign the prefixed path that was actually requested"
    );
}

#[tokio::test]
async fn a_service_error_is_classified_rather_than_swallowed() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        read_request(&mut stream);
        let body = r#"{"message":"Too many requests"}"#;
        let response = format!(
            "HTTP/1.1 429 ERR\r\nContent-Type: application/json\r\nx-amzn-errortype: ThrottlingException\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        );
        stream.write_all(response.as_bytes()).expect("write");
    });

    let client = BedrockEmbeddingClient::new("AKIDEXAMPLE", "secret", "us-east-1", "m")
        .with_endpoint(format!("http://{addr}"));

    let err = client
        .get_embeddings(vec!["alpha".into()], None)
        .await
        .expect_err("429 must surface");

    // A throttle has to reach the retry layer as a retryable ServiceStatus,
    // carrying the status, or `RetryingChatClient`-style backoff cannot work.
    match err {
        agent_framework_core::error::Error::ServiceStatus { status, .. } => {
            assert_eq!(status, 429);
        }
        other => panic!("expected ServiceStatus, got {other:?}"),
    }
}
