//! Hermetic loopback tests for [`FoundryChatClient`] driven against a
//! hand-rolled fake Foundry Responses server built on a bare
//! `std::net::TcpListener` — no external process, no real network. Reuses the
//! same fake-server pattern as `agent-framework-azure`'s
//! `tests/credentials_loopback.rs`, since [`FoundryChatClient`] delegates
//! straight through to `agent_framework_azure::responses::AzureOpenAIResponsesClient`:
//! these tests exercise that delegation end to end through the real
//! `reqwest` path — the outbound URL shape (`{endpoint}/openai/v1/responses`,
//! no `?api-version=` query), the `Authorization: Bearer` header from a
//! [`TokenCredential`], and both the non-streaming and streaming Responses
//! JSON/SSE shapes (text + a function tool call).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::agent::{Agent, AgentRunOptions, SupportsAgentRun};
use agent_framework_core::client::ChatClient;
use agent_framework_core::types::{ChatOptions, Content, Message};
use agent_framework_foundry::{FoundryChatClient, FoundryClientHeaders};
use futures::StreamExt;
use serde_json::Value;

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// One recorded request: `(request line, header block, body)`.
type Recorded = (String, String, Vec<u8>);

fn read_http_request(stream: &mut TcpStream) -> Recorded {
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

fn write_response(stream: &mut TcpStream, event_stream: bool, body: &str) {
    let head = if event_stream {
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            .to_string()
    } else {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
    };
    stream.write_all(head.as_bytes()).expect("write head");
    stream.write_all(body.as_bytes()).expect("write body");
    stream.flush().expect("flush");
}

/// A fake HTTP server that answers every request with the same canned body
/// and records the raw request strings.
struct FakeServer {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeServer {
    fn start(event_stream: bool, body: &'static str) -> Self {
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
                        write_response(&mut stream, event_stream, body);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
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

/// A minimal completed Responses body, for tests whose subject is the
/// outbound request rather than the parse.
const OK_BODY: &str = r#"{"id":"resp_h","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}"#;

fn client(endpoint: &str) -> FoundryChatClient {
    FoundryChatClient::with_token_credential(
        endpoint,
        "gpt-4o",
        Arc::new(StaticTokenCredential::new("test-token")),
    )
}

/// The point of the whole `x-client-*` capability: a header stamped on the
/// per-call `ChatOptions` has to arrive on the outbound HTTP request, and must
/// *not* arrive in the request body. Nothing else in this port proves the
/// first half — the carrier lives in `additional_properties`, every other
/// entry of which is merged into the JSON body, so "it ended up in the body"
/// is the specific way this would silently fail.
#[tokio::test]
async fn stamped_client_headers_reach_the_wire_and_stay_out_of_the_body() {
    let server = FakeServer::start(false, OK_BODY);

    let options = ChatOptions::new()
        .with_client_header("x-client-end-user-id", "user-42")
        .unwrap()
        .with_client_header("X-Client-Request-Id", "req-7")
        .unwrap();
    let c = client(&server.addr);
    c.get_response(vec![Message::user("hi")], options)
        .await
        .unwrap();

    let (_line, headers, body) = server.requests().remove(0);
    let headers = headers.to_ascii_lowercase();
    assert!(
        headers.contains("x-client-end-user-id: user-42"),
        "headers: {headers}"
    );
    assert!(
        headers.contains("x-client-request-id: req-7"),
        "a header name's case is the caller's; only the prefix check is \
         case-insensitive. headers: {headers}"
    );
    // The client's own credential still goes, beside them rather than
    // replaced by them.
    assert!(
        headers.contains("authorization: bearer test-token"),
        "headers: {headers}"
    );

    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body_json.get("agent_framework.client_headers").is_none(),
        "the carrier must be lifted out of the body, not merged into it: {body_json}"
    );
    assert!(
        !String::from_utf8_lossy(&body).contains("user-42"),
        "no part of a client header belongs in the request body"
    );
}

/// A header this client sets itself cannot be overridden per call. Because
/// `reqwest` *appends* a header rather than replacing one, a permitted
/// `authorization` entry would put two credentials on one request and leave
/// the choice to the service — so this is refused before the request is
/// built, and refused at the transport rather than only at the typed surface
/// that normally stamps these (which would not have accepted the name
/// anyway).
#[tokio::test]
async fn a_client_header_cannot_override_the_clients_own_authentication() {
    let server = FakeServer::start(false, OK_BODY);

    let mut options = ChatOptions::new();
    options.additional_properties.insert(
        "agent_framework.client_headers".into(),
        serde_json::json!({ "Authorization": "Bearer attacker-token" }),
    );
    let err = client(&server.addr)
        .get_response(vec![Message::user("hi")], options)
        .await
        .expect_err("an authorization override must be refused");
    let msg = err.to_string();
    assert!(
        msg.contains("set by the client itself"),
        "the error should name the reason, got: {msg}"
    );
    assert!(
        server.requests().is_empty(),
        "nothing should have been sent"
    );
}

/// The use case the capability exists for: a multi-tenant caller attesting
/// *this run's* end user. One agent serves every tenant, so the identity can
/// only ride on the per-run options — which is why it is worth proving the
/// whole `Agent::run_with_options` → `ChatOptions` merge → transport path,
/// not just the chat client in isolation.
#[tokio::test]
async fn a_client_header_set_per_run_reaches_the_wire_through_an_agent() {
    let server = FakeServer::start(false, OK_BODY);

    let agent = Agent::builder(client(&server.addr))
        .instructions("be brief")
        .build();
    let options = AgentRunOptions {
        chat_options: Some(
            ChatOptions::new()
                .with_client_header("x-client-end-user-id", "tenant-a-user-9")
                .unwrap(),
        ),
        ..Default::default()
    };
    agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();

    let (_line, headers, body) = server.requests().remove(0);
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("x-client-end-user-id: tenant-a-user-9"),
        "headers: {headers}"
    );
    // The agent's own options still took effect, so the carrier rode along
    // with them rather than displacing them in the merge.
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["instructions"], serde_json::json!("be brief"));
}

/// Non-streaming is not the only path: the streaming one builds its own
/// request, so it needs its own proof that the headers are applied there too.
#[tokio::test]
async fn stamped_client_headers_reach_the_wire_on_the_streaming_path() {
    let server = FakeServer::start(
        true,
        "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\n\
         event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"id\":\"r\",\"status\":\"completed\",\"output\":[]}}\n\n",
    );

    let options = ChatOptions::new()
        .with_client_header("x-client-end-user-id", "user-42")
        .unwrap();
    let mut stream = client(&server.addr)
        .get_streaming_response(vec![Message::user("hi")], options)
        .await
        .unwrap();
    while stream.next().await.is_some() {}

    let (_line, headers, _body) = server.requests().remove(0);
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("x-client-end-user-id: user-42"),
        "headers: {headers}"
    );
}

/// Non-streaming round trip: a Responses JSON body with plain assistant text
/// parses through, and the outbound request hits the documented path-versioned
/// v1 route with a bearer token — no `?api-version=` query parameter.
#[tokio::test]
async fn non_streaming_round_trip_hits_v1_responses_route_with_bearer_auth() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_abc123","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Hello from Foundry!"}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}"#,
    );

    let c = client(&server.addr);
    let resp = c
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    assert_eq!(resp.text(), "Hello from Foundry!");
    assert_eq!(resp.response_id.as_deref(), Some("resp_abc123"));
    assert_eq!(resp.usage_details.unwrap().total_token_count, Some(15));

    let (line, headers, body) = server.requests().remove(0);
    assert_eq!(
        line, "POST /openai/v1/responses HTTP/1.1",
        "the Foundry v1 GA route is path-versioned with no api-version query"
    );
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("authorization: bearer test-token"),
        "headers: {headers}"
    );
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["model"], serde_json::json!("gpt-4o"));
    assert_eq!(
        body_json["input"],
        serde_json::json!([{ "type": "message", "role": "user", "content": [
            { "type": "input_text", "text": "hi" }
        ]}])
    );
}

/// A Responses body describing a function tool call round-trips into a
/// `FunctionCallContent`, proving the reused `agent_framework_openai`
/// conversion is wired all the way through `FoundryChatClient`.
#[tokio::test]
async fn tool_call_response_round_trips_into_function_call_content() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_call1","model":"gpt-4o","status":"completed","output":[{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"loc\":\"NYC\"}"}]}"#,
    );

    let c = client(&server.addr);
    let resp = c
        .get_response(vec![Message::user("weather?")], ChatOptions::new())
        .await
        .unwrap();

    let calls = resp.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(calls[0].call_id, "call_1");
}

/// Streaming round trip: Responses SSE events parse into text deltas plus a
/// final `response.completed` usage/conversation-id update.
#[tokio::test]
async fn streaming_round_trip_yields_text_and_usage() {
    let sse_body = "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hi\"}\n\n\
data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_stream_1\",\"model\":\"gpt-4o\",\"status\":\"completed\",\"usage\":{\"input_tokens\":3,\"output_tokens\":2,\"total_tokens\":5}}}\n\n";
    let server = FakeServer::start(true, sse_body);

    let c = client(&server.addr);
    let mut stream = c
        .get_streaming_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let mut text = String::new();
    let mut total_tokens = None;
    while let Some(update) = stream.next().await {
        let update = update.expect("stream update should parse cleanly");
        text.push_str(&update.text_content());
        for content in &update.contents {
            if let Content::Usage(u) = content {
                total_tokens = u.details.total_token_count;
            }
        }
    }

    assert_eq!(text, "Hi");
    assert_eq!(total_tokens, Some(5));

    let (line, _headers, body) = server.requests().remove(0);
    assert_eq!(line, "POST /openai/v1/responses HTTP/1.1");
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body_json["stream"], serde_json::json!(true));
}

/// `FoundryChatClient::new` (API-key auth) hits the same route with an
/// `api-key` header instead of `Authorization: Bearer`.
#[tokio::test]
async fn api_key_client_uses_api_key_header() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}"#,
    );

    let c = FoundryChatClient::new(&server.addr, "gpt-4o", "test-api-key");
    let _ = c
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let (_line, headers, _body) = server.requests().remove(0);
    assert!(
        headers
            .to_ascii_lowercase()
            .contains("api-key: test-api-key"),
        "headers: {headers}"
    );
}

/// Foundry does not want encrypted reasoning unless it is asked for by name,
/// so — unlike OpenAI and Azure OpenAI, which add it to every stateless
/// request — `FoundryChatClient` must send no `include` at all. Asserted here
/// on the real outbound body rather than on the flag, so the test covers the
/// wiring through `AzureOpenAIResponsesClient` and not just its default.
/// Mirrors upstream #7536.
#[tokio::test]
async fn foundry_does_not_implicitly_request_encrypted_reasoning() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}"#,
    );

    let c = client(&server.addr);
    c.get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();

    let (_, _, body) = server.requests().remove(0);
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body_json.get("include").is_none(),
        "Foundry should send no include, got: {body_json}"
    );
}

/// The opt-out governs only what the client adds unprompted: a caller that
/// names `reasoning.encrypted_content` still gets it on the wire.
#[tokio::test]
async fn foundry_still_honors_an_explicitly_requested_include() {
    let server = FakeServer::start(
        false,
        r#"{"id":"resp_1","model":"gpt-4o","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}]}"#,
    );

    let mut options = ChatOptions::new();
    options.additional_properties.insert(
        "include".into(),
        serde_json::json!(["reasoning.encrypted_content"]),
    );

    let c = client(&server.addr);
    c.get_response(vec![Message::user("hi")], options)
        .await
        .unwrap();

    let (_, _, body) = server.requests().remove(0);
    let body_json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(
        body_json["include"],
        serde_json::json!(["reasoning.encrypted_content"])
    );
}
