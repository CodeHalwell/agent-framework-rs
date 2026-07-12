//! Hermetic loopback tests for [`AzureAISearchProvider`] against a hand-rolled
//! fake Azure AI Search server on a bare `std::net::TcpListener` — no external
//! process, no real network. Verifies the request (route, auth header, body)
//! and the response → [`SessionContext`] formatting for both api-key and
//! token-credential auth.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::StaticTokenCredential;
use agent_framework_azure_ai_search::AzureAISearchProvider;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::Message;
use serde_json::Value;

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn read_http_request(stream: &mut TcpStream) -> (String, String, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read headers");
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
        let n = stream.read(&mut chunk).expect("read body");
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

/// One recorded request: `(request line, header block, body)`.
type Recorded = (String, String, Vec<u8>);

fn write_json(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).expect("write");
    stream.flush().expect("flush");
}

struct FakeSearch {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeSearch {
    fn start(body: &'static str) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.set_nonblocking(true).expect("nonblocking");
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
                        write_json(&mut stream, body);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("accept: {e}"),
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

    fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for FakeSearch {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

const RESULTS: &str = r#"{"value":[
    {"@search.score":1.5,"id":"doc1","content":"The sky is blue."},
    {"@search.score":1.2,"id":"doc2","content":"Water is wet."}
]}"#;

#[tokio::test]
async fn api_key_search_formats_context_with_citations() {
    let server = FakeSearch::start(RESULTS);
    let provider = AzureAISearchProvider::with_api_key(&server.addr, "my-index", "secret-key")
        .with_top(2)
        .with_semantic_configuration("sem");

    let mut ctx = SessionContext::new(vec![Message::user("what color is the sky?")]);
    provider.before_run(&mut ctx).await.unwrap();

    // Header + one cited block per result, folded into instructions.
    let instructions = ctx.instructions.unwrap();
    assert!(instructions.starts_with("Use the following context to answer the question:"));
    assert!(instructions.contains("[Source: doc1] The sky is blue."));
    assert!(instructions.contains("[Source: doc2] Water is wet."));

    // The request used the OData index path, the api-key header, and a semantic body.
    let (line, headers, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /indexes('my-index')/docs/search"),
        "line: {line}"
    );
    assert!(
        headers.to_ascii_lowercase().contains("api-key: secret-key"),
        "headers: {headers}"
    );
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["search"], serde_json::json!("what color is the sky?"));
    assert_eq!(body["top"], serde_json::json!(2));
    assert_eq!(body["queryType"], serde_json::json!("semantic"));
    assert_eq!(body["semanticConfiguration"], serde_json::json!("sem"));
}

#[tokio::test]
async fn token_credential_search_uses_bearer_auth() {
    let server = FakeSearch::start(RESULTS);
    let provider = AzureAISearchProvider::with_token_credential(
        &server.addr,
        "my-index",
        Arc::new(StaticTokenCredential::new("jwt-token")),
    );

    let mut ctx = SessionContext::new(vec![Message::user("query")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx.instructions.is_some());

    let (_line, headers, _body) = server.requests().remove(0);
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("authorization: bearer jwt-token"),
        "headers: {headers}"
    );
    // Bearer auth must not also send an api-key header.
    assert!(
        !headers.to_ascii_lowercase().contains("api-key:"),
        "headers: {headers}"
    );
}

#[tokio::test]
async fn empty_query_makes_no_request() {
    let server = FakeSearch::start(RESULTS);
    let provider = AzureAISearchProvider::with_api_key(&server.addr, "idx", "key");

    // No user message → empty context, no search issued.
    let mut ctx = SessionContext::new(vec![Message::system("system only")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx.instructions.is_none());
    assert!(ctx.messages.is_empty());

    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(server.requests().len(), 0, "no query should have been sent");
}

#[tokio::test]
async fn no_results_yields_empty_context() {
    let server = FakeSearch::start(r#"{"value":[]}"#);
    let provider = AzureAISearchProvider::with_api_key(&server.addr, "idx", "key");
    let mut ctx = SessionContext::new(vec![Message::user("nothing matches")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx.instructions.is_none());
}
