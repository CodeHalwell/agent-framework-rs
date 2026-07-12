//! Loopback tests: a bare `std::net::TcpListener` thread speaks just enough
//! HTTP/1.1 to serve canned responses, exercising the real `reqwest`
//! request/response path (headers, auth signature, JSON body,
//! `x-ms-continuation` pagination, status-code handling) end to end without
//! any external network access. Mirrors `agent-framework-mem0`'s
//! `tests/http_loopback.rs` and `agent-framework-mcp`'s
//! `tests/http_loopback.rs`, extended with [`serve_sequence`] since a
//! `CosmosChatMessageStore` round trip needs more than one request/response
//! pair on the same connection sequence (create, then query; or paginated
//! query pages). Per the work package's "NO live network" rule, nothing
//! here ever leaves localhost — there is no live-Cosmos test anywhere in
//! this crate.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use agent_framework_core::threads::ChatMessageStore;
use agent_framework_core::types::Message;
use agent_framework_core::workflow::{CheckpointStorage, WorkflowCheckpoint};
use agent_framework_cosmos::{CosmosChatMessageStore, CosmosCheckpointStorage};
use serde_json::{json, Value};

/// A minimal but non-trivial [`WorkflowCheckpoint`] for the checkpoint-storage
/// loopback tests below.
fn sample_checkpoint(checkpoint_id: &str, workflow_id: &str) -> WorkflowCheckpoint {
    let mut executor_states = std::collections::HashMap::new();
    executor_states.insert("executor-a".to_string(), json!({"count": 2}));
    WorkflowCheckpoint {
        checkpoint_id: checkpoint_id.to_string(),
        workflow_id: workflow_id.to_string(),
        workflow_name: Some("my-workflow".to_string()),
        timestamp_millis: 1_700_000_000_000,
        iteration_count: 3,
        messages: Vec::new(),
        executor_states,
        shared_state: std::collections::HashMap::new(),
        pending_requests: Vec::new(),
        fanin_state: std::collections::HashMap::new(),
        metadata: std::collections::HashMap::new(),
        graph_signature: "sig-123".to_string(),
        version: "1.0".to_string(),
    }
}

/// A synthetic, deterministic base64 test key (bytes 0..=63), built at
/// runtime so secret scanners never see a high-entropy literal. The
/// loopback assertions check header *shape*, not signature values, so any
/// well-formed key works.
fn test_key() -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes: Vec<u8> = (0u8..64).collect();
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Accept one connection on `listener`, bounded by a generous retry loop so
/// a misbehaving client can't hang the test suite forever.
fn accept_with_timeout(listener: &TcpListener) -> TcpStream {
    listener.set_nonblocking(true).expect("set nonblocking");
    for _ in 0..500 {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).expect("set blocking");
                return stream;
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => panic!("accept failed: {e}"),
        }
    }
    panic!("timed out waiting for a client connection");
}

/// One fully-read HTTP/1.1 request: method, path, raw header block, and body.
struct CapturedRequest {
    method: String,
    path: String,
    header_block: String,
    body: Vec<u8>,
}

impl CapturedRequest {
    fn body_json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("request body is valid JSON")
    }

    /// Case-insensitive header lookup.
    fn header(&self, name: &str) -> Option<String> {
        let needle = format!("{}:", name.to_ascii_lowercase());
        self.header_block.lines().find_map(|l| {
            if l.to_ascii_lowercase().starts_with(&needle) {
                l.split_once(':').map(|(_, v)| v.trim().to_string())
            } else {
                None
            }
        })
    }
}

/// Read one HTTP/1.1 request's request-line, headers, and `Content-Length`
/// body from `stream`.
fn read_http_request(stream: &mut TcpStream) -> CapturedRequest {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut chunk).expect("read request headers");
        assert!(n > 0, "connection closed before headers completed");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos;
        }
    };
    let header_str = String::from_utf8_lossy(&buf[..header_end]).to_string();

    let mut request_line_parts = header_str
        .lines()
        .next()
        .unwrap_or_default()
        .split_whitespace();
    let method = request_line_parts.next().unwrap_or_default().to_string();
    let path = request_line_parts.next().unwrap_or_default().to_string();

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
        assert!(n > 0, "connection closed before body completed");
        buf.extend_from_slice(&chunk[..n]);
    }

    CapturedRequest {
        method,
        path,
        header_block: header_str,
        body: buf[body_start..body_start + content_length].to_vec(),
    }
}

fn write_status_response(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    extra_headers: &[(&str, String)],
    body: &Value,
) {
    let payload = body.to_string();
    let mut response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        payload.len()
    );
    for (name, value) in extra_headers {
        response.push_str(&format!("{name}: {value}\r\n"));
    }
    response.push_str("Connection: close\r\n\r\n");
    response.push_str(&payload);
    stream
        .write_all(response.as_bytes())
        .expect("write response");
    stream.flush().expect("flush response");
}

/// Spawn a loopback HTTP server that serves exactly `count` requests in
/// sequence, one connection per request (every response carries
/// `Connection: close`, so `reqwest` can't accidentally pipeline two logical
/// requests onto one connection and desync this server's simple
/// one-request-per-accept loop). `respond(i, &request)` is called for each
/// of the `count` requests, in order, and must return `(status, reason,
/// extra_headers, body)` for that request. Returns the server's base URL
/// and a join handle yielding every [`CapturedRequest`] in order.
fn serve_sequence<F>(
    count: usize,
    mut respond: F,
) -> (String, std::thread::JoinHandle<Vec<CapturedRequest>>)
where
    F: FnMut(usize, &CapturedRequest) -> (u16, &'static str, Vec<(&'static str, String)>, Value)
        + Send
        + 'static,
{
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let handle = std::thread::spawn(move || {
        let mut requests = Vec::with_capacity(count);
        for i in 0..count {
            let mut stream = accept_with_timeout(&listener);
            let request = read_http_request(&mut stream);
            let (status, reason, extra_headers, body) = respond(i, &request);
            write_status_response(&mut stream, status, reason, &extra_headers, &body);
            requests.push(request);
        }
        requests
    });
    (format!("http://{addr}"), handle)
}

fn assert_valid_cosmos_headers(request: &CapturedRequest, expected_partition_key: &str) {
    let auth = request
        .header("authorization")
        .expect("Authorization header present");
    assert!(!auth.is_empty());
    // The whole `type=master&ver=1.0&sig=...` string is percent-encoded, so
    // the literal `=`/`&` from those key-value pairs must not appear raw.
    assert!(auth.starts_with("type%3Dmaster%26ver%3D1.0%26sig%3D"));

    let date = request.header("x-ms-date").expect("x-ms-date present");
    // RFC 1123 shape: "Tue, 29 Mar 2016 02:28:29 GMT".
    assert!(date.ends_with("GMT"));
    assert!(date.contains(", "));

    assert_eq!(
        request.header("x-ms-version").as_deref(),
        Some("2018-12-31")
    );

    let pk_header = request
        .header("x-ms-documentdb-partitionkey")
        .expect("partition key header present");
    assert_eq!(pk_header, format!("[\"{expected_partition_key}\"]"));
}

#[tokio::test]
async fn create_document_sends_signed_headers_and_expected_body() {
    let (base_url, handle) = serve_sequence(1, |_i, request| {
        let mut created = request.body_json();
        created["_rid"] = json!("abc==");
        created["_ts"] = json!(1_700_000_000);
        (201, "Created", vec![], created)
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-1".to_string()),
    )
    .unwrap();

    store
        .add_messages(vec![Message::user("Hello, Cosmos!")])
        .await
        .unwrap();

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];

    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/dbs/agent-framework/colls/chat-messages/docs"
    );
    assert_eq!(
        request.header("content-type").as_deref(),
        Some("application/json")
    );
    assert_valid_cosmos_headers(request, "thread-1");

    let body = request.body_json();
    assert_eq!(body["threadId"], json!("thread-1"));
    assert!(body["id"].as_str().is_some());
    assert!(body["seq"].is_number());
    let inner: Message = serde_json::from_str(body["message"].as_str().unwrap()).unwrap();
    assert_eq!(inner.text(), "Hello, Cosmos!");
}

#[tokio::test]
async fn query_documents_sends_isquery_headers_and_query_body() {
    let (base_url, handle) = serve_sequence(1, |_i, _request| {
        (
            200,
            "OK",
            vec![],
            json!({"_rid": "abc==", "Documents": [], "_count": 0}),
        )
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-2".to_string()),
    )
    .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert!(messages.is_empty());

    let requests = handle.join().unwrap();
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/dbs/agent-framework/colls/chat-messages/docs"
    );
    assert_eq!(
        request.header("content-type").as_deref(),
        Some("application/query+json")
    );
    assert_eq!(
        request.header("x-ms-documentdb-isquery").as_deref(),
        Some("True")
    );
    assert_valid_cosmos_headers(request, "thread-2");

    let body = request.body_json();
    assert!(body["query"]
        .as_str()
        .unwrap()
        .contains("WHERE c.threadId = @threadId"));
    assert!(body["query"].as_str().unwrap().contains("ORDER BY c.seq"));
    assert_eq!(
        body["parameters"],
        json!([{"name": "@threadId", "value": "thread-2"}])
    );
}

/// The core required scenario: add a message, then list it back, entirely
/// against canned responses — proves the create-document write shape and
/// the query-response read shape are mutually compatible end to end.
#[tokio::test]
async fn add_then_list_round_trip() {
    let (base_url, handle) = serve_sequence(2, {
        let mut created_doc: Option<Value> = None;
        move |i, request| match i {
            0 => {
                let mut doc = request.body_json();
                doc["_rid"] = json!("abc==");
                doc["_ts"] = json!(1_700_000_000);
                created_doc = Some(doc.clone());
                (201, "Created", vec![], doc)
            }
            1 => {
                let doc = created_doc.clone().expect("create request came first");
                (
                    200,
                    "OK",
                    vec![],
                    json!({"_rid": "abc==", "Documents": [doc], "_count": 1}),
                )
            }
            _ => unreachable!("only 2 requests expected"),
        }
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-roundtrip".to_string()),
    )
    .unwrap();

    store
        .add_messages(vec![Message::user("round trip me")])
        .await
        .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "round trip me");

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "POST");
    assert!(requests[0].header("x-ms-documentdb-isquery").is_none());
    assert_eq!(requests[1].method, "POST");
    assert_eq!(
        requests[1].header("x-ms-documentdb-isquery").as_deref(),
        Some("True")
    );
}

#[tokio::test]
async fn add_multiple_messages_then_list_preserves_order() {
    let (base_url, handle) = serve_sequence(3, {
        let mut created: Vec<Value> = Vec::new();
        move |i, request| {
            if i < 2 {
                let mut doc = request.body_json();
                doc["_rid"] = json!(format!("r{i}"));
                created.push(doc.clone());
                (201, "Created", vec![], doc)
            } else {
                (
                    200,
                    "OK",
                    vec![],
                    json!({"_rid": "abc==", "Documents": created.clone(), "_count": created.len()}),
                )
            }
        }
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-multi".to_string()),
    )
    .unwrap();

    store
        .add_messages(vec![Message::user("first"), Message::assistant("second")])
        .await
        .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text(), "first");
    assert_eq!(messages[1].text(), "second");

    let requests = handle.join().unwrap();
    // Two create_document POSTs with strictly increasing `seq`, then one query.
    let seq0 = requests[0].body_json()["seq"].as_i64().unwrap();
    let seq1 = requests[1].body_json()["seq"].as_i64().unwrap();
    assert!(
        seq1 > seq0,
        "second message's seq must sort after the first's"
    );
}

#[tokio::test]
async fn query_documents_follows_continuation_token_across_pages() {
    let page1_doc = json!({
        "id": "d1", "threadId": "thread-page", "seq": 1,
        "message": serde_json::to_string(&Message::user("page one")).unwrap(),
    });
    let page2_doc = json!({
        "id": "d2", "threadId": "thread-page", "seq": 2,
        "message": serde_json::to_string(&Message::user("page two")).unwrap(),
    });

    let (base_url, handle) = serve_sequence(2, move |i, _request| match i {
        0 => (
            200,
            "OK",
            vec![("x-ms-continuation", "token-abc".to_string())],
            json!({"_rid": "r", "Documents": [page1_doc.clone()], "_count": 1}),
        ),
        1 => (
            200,
            "OK",
            vec![],
            json!({"_rid": "r", "Documents": [page2_doc.clone()], "_count": 1}),
        ),
        _ => unreachable!(),
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-page".to_string()),
    )
    .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text(), "page one");
    assert_eq!(messages[1].text(), "page two");

    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 2);
    // The second request must carry back the continuation token the first
    // response returned.
    assert_eq!(
        requests[1].header("x-ms-continuation").as_deref(),
        Some("token-abc")
    );
    assert!(requests[0].header("x-ms-continuation").is_none());
}

#[tokio::test]
async fn ensure_created_posts_database_then_container() {
    let (base_url, handle) = serve_sequence(2, |i, _request| match i {
        0 => (201, "Created", vec![], json!({"id": "agent-framework"})),
        1 => (201, "Created", vec![], json!({"id": "chat-messages"})),
        _ => unreachable!(),
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-ensure".to_string()),
    )
    .unwrap();

    store.ensure_created().await.unwrap();

    let requests = handle.join().unwrap();
    assert_eq!(requests[0].method, "POST");
    assert_eq!(requests[0].path, "/dbs");
    assert_eq!(requests[0].body_json(), json!({"id": "agent-framework"}));

    assert_eq!(requests[1].method, "POST");
    assert_eq!(requests[1].path, "/dbs/agent-framework/colls");
    let body = requests[1].body_json();
    assert_eq!(body["id"], json!("chat-messages"));
    assert_eq!(
        body["partitionKey"],
        json!({"paths": ["/threadId"], "kind": "Hash"})
    );
}

#[tokio::test]
async fn ensure_created_tolerates_409_conflict_on_both_calls() {
    let (base_url, handle) = serve_sequence(2, |_i, _request| {
        (
            409,
            "Conflict",
            vec![],
            json!({"code": "Conflict", "message": "already exists"}),
        )
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-ensure-conflict".to_string()),
    )
    .unwrap();

    // Must succeed even though both calls report 409.
    store.ensure_created().await.unwrap();
    handle.join().unwrap();
}

#[tokio::test]
async fn clear_queries_ids_then_deletes_each() {
    let (base_url, handle) = serve_sequence(3, |i, _request| match i {
        0 => (
            200,
            "OK",
            vec![],
            json!({"_rid": "r", "Documents": ["id-1", "id-2"], "_count": 2}),
        ),
        1 | 2 => (204, "No Content", vec![], Value::Null),
        _ => unreachable!(),
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-clear".to_string()),
    )
    .unwrap();

    store.clear().await.unwrap();

    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].method, "POST"); // the id query
    assert_eq!(requests[1].method, "DELETE");
    assert_eq!(requests[2].method, "DELETE");
    assert!(requests[1].path.ends_with("/id-1") || requests[1].path.ends_with("/id-2"));
}

#[tokio::test]
async fn create_document_surfaces_non_2xx_status_as_service_error() {
    let (base_url, _handle) = serve_sequence(1, |_i, _request| {
        (
            403,
            "Forbidden",
            vec![],
            json!({"code": "Forbidden", "message": "storage limit reached"}),
        )
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-error".to_string()),
    )
    .unwrap();

    let err = store
        .add_messages(vec![Message::user("hi")])
        .await
        .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("403"), "message was: {msg}");
    assert!(msg.contains("storage limit reached"), "message was: {msg}");
}

#[tokio::test]
async fn query_documents_tolerates_response_missing_documents_field() {
    let (base_url, _handle) = serve_sequence(1, |_i, _request| {
        // A well-formed JSON body that nonetheless lacks the `Documents`
        // array `parse_query_response` expects — must be treated as "no
        // results" rather than an error (see its unit tests in `client.rs`
        // for the equivalent hermetic, non-network check).
        (200, "OK", vec![], json!({"unexpected": "shape"}))
    });

    let store = CosmosChatMessageStore::new(
        base_url,
        test_key(),
        "agent-framework",
        "chat-messages",
        Some("thread-malformed".to_string()),
    )
    .unwrap();

    // Missing `Documents` is tolerated as "no results" (defensive parsing),
    // not an error — list_messages should return an empty, not fail.
    let messages = store.list_messages().await.unwrap();
    assert!(messages.is_empty());
}

// region: CosmosCheckpointStorage

#[tokio::test]
async fn checkpoint_save_sends_upsert_header_and_document_shape() {
    let (base_url, handle) = serve_sequence(1, |_i, request| {
        let mut created = request.body_json();
        created["_rid"] = json!("abc==");
        created["_ts"] = json!(1_700_000_000);
        (200, "OK", vec![], created)
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let id = storage
        .save(sample_checkpoint("cp-1", "wf-1"))
        .await
        .unwrap();
    assert_eq!(id, "cp-1");

    let requests = handle.join().expect("server thread panicked");
    assert_eq!(requests.len(), 1);
    let request = &requests[0];

    assert_eq!(request.method, "POST");
    assert_eq!(
        request.path,
        "/dbs/agent-framework/colls/workflow-checkpoints/docs"
    );
    // save() upserts (create-or-replace) rather than plain-create, since a
    // resumed workflow may re-save the same checkpoint_id.
    assert_eq!(
        request.header("x-ms-documentdb-is-upsert").as_deref(),
        Some("True")
    );
    // Partitioned by the checkpoint's own id (see module docs), not
    // workflow_id.
    assert_valid_cosmos_headers(request, "cp-1");

    let body = request.body_json();
    assert_eq!(body["id"], json!("cp-1"));
    assert_eq!(body["workflowId"], json!("wf-1"));
    let inner: WorkflowCheckpoint =
        serde_json::from_str(body["checkpoint"].as_str().unwrap()).unwrap();
    assert_eq!(inner.checkpoint_id, "cp-1");
    assert_eq!(inner.iteration_count, 3);
}

#[tokio::test]
async fn checkpoint_load_returns_none_on_404() {
    let (base_url, handle) = serve_sequence(1, |_i, _request| {
        (
            404,
            "Not Found",
            vec![],
            json!({"code": "NotFound", "message": "Resource not found"}),
        )
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let loaded = storage.load("missing-checkpoint").await.unwrap();
    assert!(loaded.is_none());

    let requests = handle.join().unwrap();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].path,
        "/dbs/agent-framework/colls/workflow-checkpoints/docs/missing-checkpoint"
    );
    assert_valid_cosmos_headers(&requests[0], "missing-checkpoint");
}

#[tokio::test]
async fn checkpoint_load_found_document_round_trips() {
    let checkpoint = sample_checkpoint("cp-2", "wf-2");
    let checkpoint_json = serde_json::to_string(&checkpoint).unwrap();
    let doc = json!({
        "id": "cp-2",
        "workflowId": "wf-2",
        "checkpoint": checkpoint_json,
        "_rid": "r",
        "_ts": 1_700_000_000,
    });

    let (base_url, handle) =
        serve_sequence(1, move |_i, _request| (200, "OK", vec![], doc.clone()));

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let loaded = storage
        .load("cp-2")
        .await
        .unwrap()
        .expect("checkpoint found");
    assert_eq!(loaded.checkpoint_id, "cp-2");
    assert_eq!(loaded.workflow_id, "wf-2");
    assert_eq!(loaded.iteration_count, 3);
    assert_eq!(loaded.graph_signature, "sig-123");

    let requests = handle.join().unwrap();
    assert_eq!(requests[0].method, "GET");
    assert_eq!(
        requests[0].path,
        "/dbs/agent-framework/colls/workflow-checkpoints/docs/cp-2"
    );
}

#[tokio::test]
async fn checkpoint_list_without_workflow_filter_uses_cross_partition_header() {
    let (base_url, handle) = serve_sequence(1, |_i, _request| {
        (
            200,
            "OK",
            vec![],
            json!({"_rid": "r", "Documents": [], "_count": 0}),
        )
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let out = storage.list(None).await.unwrap();
    assert!(out.is_empty());

    let requests = handle.join().unwrap();
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(
        request
            .header("x-ms-documentdb-query-enablecrosspartition")
            .as_deref(),
        Some("True")
    );
    // No single partition key can scope this query (see module docs), so no
    // `x-ms-documentdb-partitionkey` header should be sent.
    assert!(request.header("x-ms-documentdb-partitionkey").is_none());
    let body = request.body_json();
    assert_eq!(body["query"], json!("SELECT * FROM c"));
}

#[tokio::test]
async fn checkpoint_list_with_workflow_filter_scopes_query() {
    let (base_url, handle) = serve_sequence(1, |_i, _request| {
        (
            200,
            "OK",
            vec![],
            json!({"_rid": "r", "Documents": [], "_count": 0}),
        )
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    storage.list(Some("wf-9")).await.unwrap();

    let requests = handle.join().unwrap();
    let body = requests[0].body_json();
    assert!(body["query"]
        .as_str()
        .unwrap()
        .contains("WHERE c.workflowId = @workflowId"));
    assert_eq!(
        body["parameters"],
        json!([{"name": "@workflowId", "value": "wf-9"}])
    );
}

#[tokio::test]
async fn checkpoint_delete_existing_does_get_then_delete() {
    let (base_url, handle) = serve_sequence(2, |i, _request| match i {
        0 => (
            200,
            "OK",
            vec![],
            json!({"id": "cp-3", "workflowId": "wf-3", "checkpoint": "{}"}),
        ),
        1 => (204, "No Content", vec![], Value::Null),
        _ => unreachable!("only 2 requests expected"),
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let existed = storage.delete("cp-3").await.unwrap();
    assert!(existed);

    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].method, "GET");
    assert_eq!(requests[1].method, "DELETE");
    assert!(requests[1].path.ends_with("/cp-3"));
}

#[tokio::test]
async fn checkpoint_delete_missing_checkpoint_only_does_get() {
    let (base_url, handle) = serve_sequence(1, |_i, _request| {
        (404, "Not Found", vec![], json!({"code": "NotFound"}))
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    let existed = storage.delete("cp-missing").await.unwrap();
    assert!(!existed);

    let requests = handle.join().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].method, "GET");
}

#[tokio::test]
async fn checkpoint_ensure_created_uses_id_partition_key() {
    let (base_url, handle) = serve_sequence(2, |i, _request| match i {
        0 => (201, "Created", vec![], json!({"id": "agent-framework"})),
        1 => (
            201,
            "Created",
            vec![],
            json!({"id": "workflow-checkpoints"}),
        ),
        _ => unreachable!("only 2 requests expected"),
    });

    let storage = CosmosCheckpointStorage::new(
        base_url,
        test_key(),
        "agent-framework",
        "workflow-checkpoints",
    )
    .unwrap();

    storage.ensure_created().await.unwrap();

    let requests = handle.join().unwrap();
    assert_eq!(requests[1].method, "POST");
    assert_eq!(requests[1].path, "/dbs/agent-framework/colls");
    let body = requests[1].body_json();
    assert_eq!(body["id"], json!("workflow-checkpoints"));
    assert_eq!(
        body["partitionKey"],
        json!({"paths": ["/id"], "kind": "Hash"})
    );
}

// endregion
