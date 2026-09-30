//! Hermetic loopback tests for [`FoundryMemoryProvider`]: a fake Foundry
//! projects data plane on a bare `std::net::TcpListener` exercises the real
//! `reqwest` path — the two `:search_memories` / `:update_memories` routes,
//! the bearer token, the request bodies (including the null-dropping the
//! service contract specifies), and the incremental search/update cursors.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use agent_framework_azure::StaticTokenCredential;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::Message;
use agent_framework_foundry::FoundryMemoryProvider;
use serde_json::Value;

#[derive(Clone, Debug)]
struct Recorded {
    start_line: String,
    headers: HashMap<String, String>,
    body: Value,
}

/// Read one HTTP request off a stream, honouring `Content-Length`.
fn read_request(stream: &mut std::net::TcpStream) -> Option<Recorded> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let (mut header_end, mut content_length) = (None, 0usize);
    loop {
        let n = stream.read(&mut chunk).ok()?;
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
    Some(Recorded {
        start_line,
        headers,
        body: serde_json::from_str(req_body).unwrap_or(Value::Null),
    })
}

fn write_response(stream: &mut std::net::TcpStream, status: u16, body: &str) {
    let reason = if status == 200 { "OK" } else { "ERR" };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len(),
    );
    let _ = stream.write_all(response.as_bytes());
}

/// Serve `responses` in order, one per connection, recording each request.
fn server(responses: Vec<(u16, String)>) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let writer = seen.clone();
    std::thread::spawn(move || {
        for (status, body) in responses {
            let Ok((mut stream, _)) = listener.accept() else {
                return;
            };
            let Some(req) = read_request(&mut stream) else {
                return;
            };
            writer.lock().unwrap().push(req);
            write_response(&mut stream, status, &body);
        }
    });
    (format!("http://{addr}"), seen)
}

/// Like [`server`], but each connection is handled on its own thread and the
/// reply is delayed. Serving concurrently is the point: a sequential server
/// would serialize the clients by itself and the test could not tell that
/// apart from the provider serializing them.
fn slow_server(
    responses: Vec<(u16, String)>,
    delay: std::time::Duration,
) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let addr = listener.local_addr().unwrap();
    let seen = Arc::new(Mutex::new(Vec::new()));
    let writer = seen.clone();
    let queue = Arc::new(Mutex::new(std::collections::VecDeque::from(responses)));
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut stream) = conn else { return };
            let writer = writer.clone();
            let queue = queue.clone();
            std::thread::spawn(move || {
                let Some(req) = read_request(&mut stream) else {
                    return;
                };
                writer.lock().unwrap().push(req);
                let next = queue.lock().unwrap().pop_front();
                let (status, body) = next.unwrap_or((200, "{}".to_string()));
                std::thread::sleep(delay);
                write_response(&mut stream, status, &body);
            });
        }
    });
    (format!("http://{addr}"), seen)
}

fn search_body(search_id: &str, contents: &[&str]) -> String {
    let memories: Vec<Value> = contents
        .iter()
        .map(|c| serde_json::json!({"memory_item": {"content": c}}))
        .collect();
    serde_json::json!({"search_id": search_id, "memories": memories, "usage": {}}).to_string()
}

fn provider(endpoint: &str) -> FoundryMemoryProvider {
    FoundryMemoryProvider::new(
        endpoint,
        "store",
        Arc::new(StaticTokenCredential::new("tok")),
    )
}

/// `before_run` makes two calls on the first run: the static (profile) fetch
/// with no `items`, then the contextual search with them. Both carry the
/// bearer token and the `:search_memories` route, and the retrieved memories
/// arrive as one injected message behind the context prompt.
#[tokio::test]
async fn before_run_fetches_static_then_contextual_memories_and_injects_them() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s-static", &["lives in Leeds"])),
        (200, search_body("s-ctx", &["prefers metric units"])),
    ]);
    let p = provider(&endpoint);
    let mut ctx = SessionContext::new(vec![Message::user("how far is it?")]);
    ctx.session_id = Some("sess-1".into());

    p.before_run(&mut ctx).await.expect("before_run");

    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.len(), 2, "one static fetch, one contextual search");

    assert!(
        reqs[0]
            .start_line
            .contains("/memory_stores/store:search_memories?api-version=v1"),
        "{}",
        reqs[0].start_line
    );
    assert_eq!(reqs[0].headers.get("authorization").unwrap(), "Bearer tok");
    // The static fetch is scope-only: no `items`, and no cursor yet.
    assert_eq!(reqs[0].body["scope"], serde_json::json!("sess-1"));
    assert!(reqs[0].body.get("items").is_none());
    assert!(reqs[0].body.get("previous_search_id").is_none());

    // The contextual search carries the turn as one message item.
    assert_eq!(
        reqs[1].body["items"][0]["type"],
        serde_json::json!("message")
    );
    assert_eq!(reqs[1].body["items"][0]["role"], serde_json::json!("user"));
    assert_eq!(
        reqs[1].body["items"][0]["content"],
        serde_json::json!("how far is it?")
    );

    assert_eq!(ctx.messages.len(), 1);
    let injected = ctx.messages[0].text();
    assert!(injected.starts_with("## Memories"), "{injected}");
    assert!(injected.contains("lives in Leeds"));
    assert!(injected.contains("prefers metric units"));
}

/// The incremental cursors: a search id is sent on the next contextual
/// search, and the static fetch happens once per provider rather than once
/// per run.
#[tokio::test]
async fn the_search_cursor_advances_and_the_static_fetch_happens_once() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s-static", &["a"])),
        (200, search_body("s-1", &["b"])),
        (200, search_body("s-2", &["c"])),
    ]);
    let p = provider(&endpoint);

    for text in ["first", "second"] {
        let mut ctx = SessionContext::new(vec![Message::user(text)]);
        ctx.session_id = Some("sess-1".into());
        p.before_run(&mut ctx).await.expect("before_run");
    }

    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.len(), 3, "the static fetch must not repeat");
    assert!(reqs[1].body.get("previous_search_id").is_none());
    assert_eq!(
        reqs[2].body["previous_search_id"],
        serde_json::json!("s-1"),
        "the second search resumes from the first's id"
    );
}

/// `after_run` writes the turn back and remembers the update id for the next
/// write, so updates chain incrementally.
#[tokio::test]
async fn after_run_posts_the_turn_and_chains_the_update_cursor() {
    let (endpoint, seen) = server(vec![
        (200, r#"{"update_id":"u-1","status":"queued"}"#.to_string()),
        (200, r#"{"update_id":"u-2","status":"queued"}"#.to_string()),
    ]);
    let p = provider(&endpoint)
        .with_scope("user-42")
        .with_update_delay(0);

    let req = [Message::user("remember I like tea")];
    let resp = [Message::assistant("noted")];
    p.after_run(&req, &resp, None).await.expect("after_run");
    p.after_run(&req, &resp, None).await.expect("after_run");

    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.len(), 2);
    assert!(
        reqs[0]
            .start_line
            .contains("/memory_stores/store:update_memories?api-version=v1"),
        "{}",
        reqs[0].start_line
    );
    // The pinned scope wins over the (absent) session id.
    assert_eq!(reqs[0].body["scope"], serde_json::json!("user-42"));
    assert_eq!(reqs[0].body["update_delay"], serde_json::json!(0));
    assert_eq!(reqs[0].body["items"].as_array().unwrap().len(), 2);
    assert!(reqs[0].body.get("previous_update_id").is_none());
    assert_eq!(reqs[1].body["previous_update_id"], serde_json::json!("u-1"));
}

/// Memory is an enhancement: a store returning 500 must leave the run intact
/// rather than surfacing as the agent's error. `after_run` in particular also
/// runs on the failure path, where raising would mask the real error.
#[tokio::test]
async fn a_failing_store_does_not_fail_the_run() {
    let (endpoint, _seen) = server(vec![
        (500, r#"{"error":"boom"}"#.to_string()),
        (500, r#"{"error":"boom"}"#.to_string()),
        (500, r#"{"error":"boom"}"#.to_string()),
    ]);
    let p = provider(&endpoint).with_scope("user-42");

    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    p.before_run(&mut ctx)
        .await
        .expect("before_run must not fail");
    assert!(
        ctx.messages.is_empty(),
        "nothing to inject when search fails"
    );

    p.after_run(&[Message::user("hi")], &[], None)
        .await
        .expect("after_run must not fail");
}

/// A run with nothing worth storing makes no call at all — an empty `items`
/// array would be a wasted round trip the service has nothing to do with.
#[tokio::test]
async fn a_turn_with_no_storable_text_makes_no_update_call() {
    let (endpoint, seen) = server(vec![(200, r#"{"update_id":"u-1"}"#.to_string())]);
    let p = provider(&endpoint).with_scope("user-42");

    p.after_run(&[Message::system("you are helpful")], &[], None)
        .await
        .expect("after_run");

    assert!(
        seen.lock().unwrap().is_empty(),
        "a system-only turn has nothing to store"
    );
}

/// Two sessions through one shared provider must not share a profile or a
/// cursor: each scope gets its own static fetch and its own search id.
#[tokio::test]
async fn two_sessions_through_one_provider_do_not_share_state() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s-static-1", &["alice likes tea"])),
        (200, search_body("s-ctx-1", &["alice asked about tea"])),
        (200, search_body("s-static-2", &["bob likes coffee"])),
        (200, search_body("s-ctx-2", &["bob asked about coffee"])),
    ]);
    let p = provider(&endpoint);

    let mut a = SessionContext::new(vec![Message::user("tea?")]);
    a.session_id = Some("alice".into());
    p.before_run(&mut a).await.expect("before_run");

    let mut b = SessionContext::new(vec![Message::user("coffee?")]);
    b.session_id = Some("bob".into());
    p.before_run(&mut b).await.expect("before_run");

    let reqs = seen.lock().unwrap().clone();
    assert_eq!(reqs.len(), 4, "each scope needs its own static fetch");
    assert_eq!(reqs[2].body["scope"], serde_json::json!("bob"));
    assert!(
        reqs[2].body.get("items").is_none(),
        "bob's static fetch must not be skipped by alice's latch"
    );
    assert!(
        reqs[3].body.get("previous_search_id").is_none(),
        "bob must not resume alice's search cursor"
    );

    // And neither run sees the other's memories.
    assert!(a.messages[0].text().contains("alice"));
    assert!(!a.messages[0].text().contains("bob"));
    assert!(b.messages[0].text().contains("bob"));
    assert!(!b.messages[0].text().contains("alice"));
}

/// `after_run` carries no session. Once a provider has served two, a write
/// would be filing one user's conversation under another's scope — so it
/// declines instead of guessing.
#[tokio::test]
async fn after_run_declines_to_write_once_the_session_is_ambiguous() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s1", &[])),
        (200, search_body("s2", &[])),
        (200, r#"{"update_id":"u-1"}"#.to_string()),
    ]);
    let p = provider(&endpoint);

    for session in ["alice", "bob"] {
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some(session.into());
        p.before_run(&mut ctx).await.expect("before_run");
    }
    let before = seen.lock().unwrap().len();

    p.after_run(&[Message::user("remember this")], &[], None)
        .await
        .expect("after_run must not fail");

    assert_eq!(
        seen.lock().unwrap().len(),
        before,
        "an ambiguous scope must produce no write at all"
    );
}

/// The escape hatch the docs point at: a pinned scope is unambiguous however
/// many sessions share the provider, so writes keep working.
#[tokio::test]
async fn a_pinned_scope_keeps_writing_across_many_sessions() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s1", &[])),
        (200, search_body("s2", &[])),
        (200, r#"{"update_id":"u-1"}"#.to_string()),
    ]);
    let p = provider(&endpoint).with_scope("tenant-7");

    for session in ["alice", "bob"] {
        let mut ctx = SessionContext::new(vec![]);
        ctx.session_id = Some(session.into());
        p.before_run(&mut ctx).await.expect("before_run");
    }
    p.after_run(&[Message::user("remember this")], &[], None)
        .await
        .expect("after_run");

    let reqs = seen.lock().unwrap().clone();
    let update = reqs.last().expect("an update was sent");
    assert!(update.start_line.contains(":update_memories"));
    assert_eq!(update.body["scope"], serde_json::json!("tenant-7"));
}

/// A transient failure of the *contextual* search must not throw away the
/// static profile the *static* search already succeeded in fetching. The
/// provider swallows memory-service failures by design; returning early
/// dropped known-good context along with the failure.
#[tokio::test]
async fn a_failed_contextual_search_still_injects_the_cached_profile() {
    let (endpoint, seen) = server(vec![
        (200, search_body("s-static", &["lives in Leeds"])),
        (500, r#"{"error":"transient"}"#.to_string()),
    ]);
    let p = provider(&endpoint).with_scope("user-42");

    let mut ctx = SessionContext::new(vec![Message::user("how far is it?")]);
    p.before_run(&mut ctx)
        .await
        .expect("before_run must not fail");

    assert_eq!(
        seen.lock().unwrap().len(),
        2,
        "both searches were attempted"
    );
    assert_eq!(
        ctx.messages.len(),
        1,
        "the profile survives the contextual failure"
    );
    assert!(ctx.messages[0].text().contains("lives in Leeds"));
}

/// A run with no storable input — instruction-only, or carrying nothing but
/// system turns — has nothing to search *with*, but the profile already
/// fetched for the scope still belongs in its context. Skipping the request
/// is right; skipping the injection loses managed memory on a valid run.
#[tokio::test]
async fn a_run_with_no_searchable_input_still_gets_the_profile() {
    let (endpoint, seen) = server(vec![(200, search_body("s-static", &["lives in Leeds"]))]);
    let p = provider(&endpoint).with_scope("user-42");

    let mut ctx = SessionContext::new(vec![Message::system("you are helpful")]);
    p.before_run(&mut ctx).await.expect("before_run");

    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "only the static fetch — there is nothing to search with"
    );
    assert_eq!(ctx.messages.len(), 1, "the profile is still injected");
    assert!(ctx.messages[0].text().contains("lives in Leeds"));
}

/// The provider is built to be shared, so it must not hold its one lock
/// across a request: a slow scope would stall every unrelated one. Two
/// concurrent runs against a server that answers slowly should overlap
/// rather than serialize.
#[tokio::test]
async fn concurrent_runs_on_different_scopes_are_not_serialized() {
    // Four slow responses: a static fetch per scope, then a search per scope.
    let (endpoint, _seen) = slow_server(
        vec![
            (200, search_body("s1", &["a"])),
            (200, search_body("s2", &["b"])),
            (200, search_body("s3", &["c"])),
            (200, search_body("s4", &["d"])),
        ],
        std::time::Duration::from_millis(120),
    );
    let p = std::sync::Arc::new(provider(&endpoint));

    let run = |scope: &'static str| {
        let p = p.clone();
        async move {
            let mut ctx = SessionContext::new(vec![Message::user("hi")]);
            ctx.session_id = Some(scope.into());
            p.before_run(&mut ctx).await.expect("before_run");
        }
    };

    let started = std::time::Instant::now();
    tokio::join!(run("alice"), run("bob"));
    let elapsed = started.elapsed();

    // Serialized, the two runs' four requests cost ~480ms; overlapped, ~240ms.
    assert!(
        elapsed < std::time::Duration::from_millis(400),
        "runs on different scopes serialized: {elapsed:?}"
    );
}
