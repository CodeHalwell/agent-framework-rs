//! Hermetic loopback tests for the Entra ID credential chain and the Azure
//! OpenAI `ServiceStatus` mapping, driven against hand-rolled fake HTTP servers
//! built on a bare `std::net::TcpListener` — no external process, no real
//! network. Mirrors `agent-framework-a2a`'s `tests/loopback.rs`: exercises the
//! real `reqwest` path end to end.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::{
    AzureOpenAIClient, AzureOpenAIResponsesClient, ClientSecretCredential,
    ManagedIdentityCredential, TokenCredential, WorkloadIdentityCredential,
};
use agent_framework_core::client::ChatClient;
use agent_framework_core::error::Error;
use agent_framework_core::types::{ChatOptions, Message};
use futures::StreamExt;

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Read one HTTP/1.1 request (request line, headers, `Content-Length` body).
fn read_http_request(stream: &mut TcpStream) -> (String, String, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let request_line = header_str.lines().next().unwrap_or_default().to_string();
    let content_length: usize = header_str
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).expect("read request body");
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = buf
        .get(body_start..body_start + content_length)
        .unwrap_or(&[])
        .to_vec();
    (request_line, header_str, body)
}

fn write_response(stream: &mut TcpStream, status_line: &str, extra: &[(&str, &str)], body: &str) {
    let mut response = format!(
        "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n",
        status_line,
        body.len()
    );
    for (k, v) in extra {
        response.push_str(&format!("{k}: {v}\r\n"));
    }
    response.push_str("\r\n");
    response.push_str(body);
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

/// One recorded request: `(request line, header block, body)`.
type Recorded = (String, String, Vec<u8>);

/// A fake HTTP server that answers every request with the same canned body and
/// records the raw request strings, so tests can both assert on the request and
/// count how many actually arrived (to prove token caching).
struct FakeServer {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeServer {
    fn start(status_line: &'static str, extra: Vec<(String, String)>, body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback listener");
        listener.set_nonblocking(true).expect("set nonblocking");
        let addr = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));

        let requests_bg = requests.clone();
        let stop_bg = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).expect("blocking");
                        let req = read_http_request(&mut stream);
                        requests_bg.lock().unwrap().push(req);
                        let extra_refs: Vec<(&str, &str)> = extra
                            .iter()
                            .map(|(k, v)| (k.as_str(), v.as_str()))
                            .collect();
                        write_response(&mut stream, status_line, &extra_refs, body);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                    Err(e) => panic!("accept failed: {e}"),
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

    fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[tokio::test]
async fn client_secret_credential_fetches_and_caches() {
    let server = FakeServer::start(
        "200 OK",
        vec![],
        r#"{"token_type":"Bearer","expires_in":3600,"access_token":"secret-token"}"#,
    );

    let cred = ClientSecretCredential::new(
        "my-tenant",
        "my-client",
        "my-secret",
        "https://ai.azure.com/.default",
    )
    .with_authority(&server.addr);

    // First call hits the network.
    assert_eq!(cred.get_token().await.unwrap(), "secret-token");
    // Second call is served from cache (expires_in=3600 keeps it fresh).
    assert_eq!(cred.get_token().await.unwrap(), "secret-token");

    // Give the server a beat, then assert exactly one request arrived.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        server.request_count(),
        1,
        "cached token should not re-request"
    );

    let (line, _headers, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /my-tenant/oauth2/v2.0/token"),
        "got: {line}"
    );
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("grant_type=client_credentials"),
        "body: {body}"
    );
    assert!(body.contains("client_id=my-client"), "body: {body}");
    // The scope is form-encoded (":" and "/" are percent-escaped).
    assert!(body.contains("scope="), "body: {body}");
}

#[tokio::test]
async fn managed_identity_credential_fetches_and_caches() {
    let server = FakeServer::start(
        "200 OK",
        vec![],
        r#"{"token_type":"Bearer","expires_in":"3600","resource":"https://ai.azure.com","access_token":"imds-token"}"#,
    );

    let cred = ManagedIdentityCredential::new("https://ai.azure.com/.default")
        .with_endpoint(format!("{}/metadata/identity/oauth2/token", server.addr));

    assert_eq!(cred.get_token().await.unwrap(), "imds-token");
    assert_eq!(cred.get_token().await.unwrap(), "imds-token");

    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        server.request_count(),
        1,
        "cached token should not re-request"
    );

    let (line, headers, _body) = server.requests().remove(0);
    assert!(
        line.starts_with("GET /metadata/identity/oauth2/token"),
        "got: {line}"
    );
    // IMDS requires the Metadata header, the api-version, and the resource
    // (the scope with `/.default` stripped).
    assert!(
        headers.to_ascii_lowercase().contains("metadata: true"),
        "headers: {headers}"
    );
    assert!(line.contains("api-version=2018-02-01"), "line: {line}");
    assert!(
        line.contains("resource=https%3A%2F%2Fai.azure.com")
            || line.contains("resource=https://ai.azure.com"),
        "line: {line}"
    );
    assert!(
        !line.contains(".default"),
        "resource must strip /.default: {line}"
    );
}

#[tokio::test]
async fn workload_identity_credential_sends_client_assertion_and_caches() {
    let server = FakeServer::start(
        "200 OK",
        vec![],
        r#"{"token_type":"Bearer","expires_in":3600,"access_token":"workload-token"}"#,
    );

    // A real scratch file: the credential re-reads it on every token request.
    let token_path = std::env::temp_dir().join(format!(
        "af-workload-identity-loopback-{}.jwt",
        std::process::id()
    ));
    std::fs::write(&token_path, "federated-jwt-contents").expect("write fake token file");

    // `with_overrides` supplies everything explicitly so this test never
    // touches real process env vars (consistent with the rest of this file).
    let cred = WorkloadIdentityCredential::with_overrides(
        "https://ai.azure.com/.default",
        Some("my-tenant".to_string()),
        Some("my-client".to_string()),
        Some(token_path.to_str().unwrap().to_string()),
    )
    .expect("all required fields supplied")
    .with_authority(&server.addr);

    // First call hits the network.
    assert_eq!(cred.get_token().await.unwrap(), "workload-token");
    // Second call is served from cache (expires_in=3600 keeps it fresh).
    assert_eq!(cred.get_token().await.unwrap(), "workload-token");

    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        server.request_count(),
        1,
        "cached token should not re-request"
    );

    let (line, _headers, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /my-tenant/oauth2/v2.0/token"),
        "got: {line}"
    );
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("grant_type=client_credentials"),
        "body: {body}"
    );
    assert!(body.contains("client_id=my-client"), "body: {body}");
    assert!(
        body.contains("client_assertion=federated-jwt-contents"),
        "body: {body}"
    );
    // `:` is percent-escaped in form encoding.
    assert!(
        body.contains("client_assertion_type=urn%3Aietf%3Aparams%3Aoauth%3Aclient-assertion-type%3Ajwt-bearer"),
        "body: {body}"
    );

    std::fs::remove_file(&token_path).ok();
}

#[tokio::test]
async fn azure_openai_maps_429_to_service_status_with_retry_after() {
    let server = FakeServer::start(
        "429 Too Many Requests",
        vec![("Retry-After".to_string(), "2".to_string())],
        r#"{"error":{"message":"rate limited"}}"#,
    );

    let client = AzureOpenAIClient::new(&server.addr, "gpt-4o", "test-key");
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap_err();

    assert_eq!(err.status(), Some(429), "error: {err}");
    assert_eq!(err.retry_after(), Some(2.0), "error: {err}");
}

/// Real end-to-end proof (actual `reqwest` round trip, not just a unit test of
/// the classification function) that `AzureOpenAIClient` reuses
/// `agent_framework_openai::classify_service_error` and gets the same
/// granular variants OpenAI's own client would.
#[tokio::test]
async fn azure_openai_maps_401_to_invalid_auth() {
    let server = FakeServer::start(
        "401 Unauthorized",
        vec![],
        r#"{"error":{"message":"Access denied","code":"invalid_api_key"}}"#,
    );

    let client = AzureOpenAIClient::new(&server.addr, "gpt-4o", "test-key");
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap_err();

    assert!(
        matches!(err, Error::ServiceInvalidAuth { .. }),
        "error: {err:?}"
    );
    // Not a ServiceStatus, so the retry layer never sees a status/retry_after
    // for it — it just treats it as non-retryable directly.
    assert_eq!(err.status(), None, "error: {err}");
}

#[tokio::test]
async fn azure_openai_maps_plain_400_to_invalid_request() {
    let server = FakeServer::start(
        "400 Bad Request",
        vec![],
        r#"{"error":{"message":"Unrecognized field","type":"invalid_request_error"}}"#,
    );

    let client = AzureOpenAIClient::new(&server.addr, "gpt-4o", "test-key");
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap_err();

    assert!(
        matches!(err, Error::ServiceInvalidRequest { .. }),
        "error: {err:?}"
    );
}

/// Azure OpenAI's content-filter shape: a `400` whose `error.code` is
/// `"content_filter"` (with a nested `innererror.code` giving the
/// `ResponsibleAIPolicyViolation` detail) must classify as
/// `ServiceContentFilter`, not the generic `ServiceInvalidRequest` a plain
/// `400` gets.
#[tokio::test]
async fn azure_openai_maps_content_filter_400_to_content_filter() {
    let server = FakeServer::start(
        "400 Bad Request",
        vec![],
        r#"{"error":{"message":"The response was filtered due to the prompt triggering Azure OpenAI's content management policy.","code":"content_filter","status":400,"innererror":{"code":"ResponsibleAIPolicyViolation","content_filter_result":{"violence":{"filtered":true,"severity":"medium"}}}}}"#,
    );

    let client = AzureOpenAIClient::new(&server.addr, "gpt-4o", "test-key");
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap_err();

    assert!(
        matches!(err, Error::ServiceContentFilter { .. }),
        "error: {err:?}"
    );
}

/// Same classification, proven through `AzureOpenAIResponsesClient` too (a
/// separate `post` in `responses.rs` that also delegates to
/// `agent_framework_openai::classify_service_error`).
#[tokio::test]
async fn azure_openai_responses_client_maps_403_to_invalid_auth() {
    let server = FakeServer::start(
        "403 Forbidden",
        vec![],
        r#"{"error":{"message":"Forbidden"}}"#,
    );

    let client = AzureOpenAIResponsesClient::new(&server.addr, "my-deployment", "test-key");
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap_err();

    assert!(
        matches!(err, Error::ServiceInvalidAuth { .. }),
        "error: {err:?}"
    );
}

/// A managed-identity credential that fails (non-success status) should surface
/// a clear error rather than caching anything, so the chain can fall through.
#[tokio::test]
async fn managed_identity_non_success_is_an_error() {
    let server = FakeServer::start("400 Bad Request", vec![], r#"{"error":"bad"}"#);
    let cred = ManagedIdentityCredential::new("https://ai.azure.com/.default")
        .with_endpoint(format!("{}/token", server.addr));
    assert!(cred.get_token().await.is_err());
}

/// End-to-end round trip for `AzureOpenAIResponsesClient::get_response`: a
/// real `reqwest` POST against a loopback server, asserting both the outbound
/// request (route, api-version, auth header, `model` = deployment name) and
/// that the JSON response is parsed via the reused
/// `agent_framework_openai::responses::parse_response`.
#[tokio::test]
async fn azure_openai_responses_client_loopback_round_trip() {
    let server = FakeServer::start(
        "200 OK",
        vec![],
        r#"{"id":"resp_abc123","model":"my-deployment","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello from Azure Responses!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#,
    );

    let client = AzureOpenAIResponsesClient::new(&server.addr, "my-deployment", "test-key");
    let response = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    assert_eq!(response.text(), "Hello from Azure Responses!");
    assert_eq!(response.response_id.as_deref(), Some("resp_abc123"));
    assert_eq!(response.usage_details.unwrap().total_token_count, Some(15));

    std::thread::sleep(Duration::from_millis(50));
    let (line, headers, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /openai/v1/responses?api-version=preview"),
        "got: {line}"
    );
    assert!(
        headers.to_ascii_lowercase().contains("api-key: test-key"),
        "headers: {headers}"
    );
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["model"], serde_json::json!("my-deployment"));
    assert_eq!(
        body_json["input"],
        serde_json::json!([{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "hi" }
        ]}])
    );
}

/// SSE parse smoke test: `AzureOpenAIResponsesClient::get_streaming_response`
/// against a loopback server emitting real Responses-API SSE events, proving
/// the reused `agent_framework_openai::responses::parse_responses_sse_stream`
/// is wired all the way through to a live `reqwest::Response` byte stream
/// (not just exercised against synthetic `serde_json::Value`s, as
/// `agent-framework-openai`'s own unit tests do).
#[tokio::test]
async fn azure_openai_responses_client_streams_sse_events() {
    let sse_body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream_1\",\"model\":\"my-deployment\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n";
    let server = FakeServer::start("200 OK", vec![], sse_body);

    let client = AzureOpenAIResponsesClient::new(&server.addr, "my-deployment", "test-key");
    let mut stream = client
        .get_streaming_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let mut text = String::new();
    let mut saw_completed = false;
    while let Some(update) = stream.next().await {
        let update = update.expect("stream update should parse cleanly");
        text.push_str(&update.text_content());
        if update.response_id.as_deref() == Some("resp_stream_1") {
            saw_completed = true;
            // `store != Some(false)` auto-populates `conversation_id`,
            // identical to `OpenAIChatClient`'s streaming behavior.
            assert_eq!(update.conversation_id.as_deref(), Some("resp_stream_1"));
        }
    }

    assert_eq!(text, "Hi");
    assert!(saw_completed, "expected a response.completed update");

    std::thread::sleep(Duration::from_millis(50));
    let (line, _headers, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /openai/v1/responses?api-version=preview"),
        "got: {line}"
    );
    let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["stream"], serde_json::json!(true));
}
