//! Hermetic loopback tests for [`AzureAISearchStore`] /
//! [`AzureAISearchCollection`] against a hand-rolled fake Azure AI Search
//! service on a bare `std::net::TcpListener` — no external process, no real
//! network.
//!
//! The unit tests next to the connector pin what it *builds* (the index
//! schema, the OData filter, the search body). These pin what it actually
//! puts on the wire and what it makes of the reply: routes, methods, the
//! api-key header, the indexing-batch envelope, a 404 read as "absent"
//! instead of an error, and a partial-success (HTTP 207) batch surfacing as a
//! failure rather than a silent one.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure_ai_search::AzureAISearchStore;
use agent_framework_core::vectors::{
    DistanceFunction, Filter, VectorCollection, VectorSearchOptions, VectorSearchResult,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use serde_json::{json, Value};

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

/// `(status, body)` — what the fake answers a given request with.
type Reply = (u16, String);

fn write_response(stream: &mut TcpStream, status: u16, body: &str) {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        207 => "Multi-Status",
        404 => "Not Found",
        _ => "Status",
    };
    let response = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).expect("write");
    stream.flush().expect("flush");
}

/// A fake Search service that answers each request from a caller-supplied
/// router, so one test can drive a whole exchange (create, upsert, search).
struct FakeSearch {
    addr: String,
    requests: Arc<Mutex<Vec<Recorded>>>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl FakeSearch {
    fn start(route: impl Fn(&str) -> Reply + Send + 'static) -> Self {
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
                        let (status, body) = route(&req.0);
                        requests_bg.lock().unwrap().push(req);
                        write_response(&mut stream, status, &body);
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

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("text").full_text_indexed().indexed(),
        VectorStoreField::data("year").with_type("int").indexed(),
        // Renamed, so the storage mapping is exercised on every path.
        VectorStoreField::vector("embedding", 3).with_storage_name("vec"),
    ])
    .unwrap()
}

#[tokio::test]
async fn ensure_collection_exists_creates_the_index_when_absent() {
    let server = FakeSearch::start(|line| {
        if line.starts_with("GET /indexes('docs')") {
            (404, r#"{"error":{"message":"not found"}}"#.into())
        } else {
            (201, r#"{"name":"docs"}"#.into())
        }
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "admin-key");
    let docs = store.get_collection("docs", definition()).unwrap();
    docs.ensure_collection_exists().await.unwrap();

    let requests = server.requests();
    assert_eq!(requests.len(), 2, "expected an existence check then a PUT");
    assert!(requests[0]
        .0
        .starts_with("GET /indexes('docs')?api-version="));
    assert!(
        requests[0]
            .1
            .to_ascii_lowercase()
            .contains("api-key: admin-key"),
        "headers: {}",
        requests[0].1
    );
    assert!(requests[1]
        .0
        .starts_with("PUT /indexes('docs')?api-version="));

    let schema: Value = serde_json::from_slice(&requests[1].2).unwrap();
    assert_eq!(schema["name"], "docs");
    let vector = schema["fields"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["name"] == "vec")
        .expect("the vector field is sent under its storage name");
    assert_eq!(vector["dimensions"], 3);
    assert_eq!(schema["vectorSearch"]["algorithms"][0]["kind"], "hnsw");
}

#[tokio::test]
async fn an_existing_index_is_not_recreated() {
    // A PUT against an existing index rewrites its schema, which can drop
    // fields a caller added out of band; "ensure" must not do that.
    let server = FakeSearch::start(|_| (200, r#"{"name":"docs"}"#.into()));
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();
    docs.ensure_collection_exists().await.unwrap();

    let requests = server.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].0.starts_with("GET /indexes('docs')"));
}

#[tokio::test]
async fn deleting_an_absent_index_succeeds() {
    let server = FakeSearch::start(|_| (404, r#"{"error":{"message":"gone"}}"#.into()));
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();
    docs.ensure_collection_deleted().await.unwrap();
    assert!(server.requests()[0]
        .0
        .starts_with("DELETE /indexes('docs')"));
}

#[tokio::test]
async fn upsert_posts_a_replacing_upload_batch_in_storage_form() {
    let server = FakeSearch::start(|_| {
        (
            200,
            r#"{"value":[{"key":"a","status":true,"statusCode":200}]}"#.into(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();

    let keys = docs
        .upsert(vec![json!({
            "id": "a",
            "text": "alpha",
            "year": 2024,
            "embedding": [1.0, 0.0, 0.0],
        })])
        .await
        .unwrap();
    assert_eq!(keys, vec![json!("a")]);

    let (line, _, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /indexes('docs')/docs/index"),
        "{line}"
    );
    let body: Value = serde_json::from_slice(&body).unwrap();
    let doc = &body["value"][0];
    // `upload` replaces the stored document; `mergeOrUpload` would keep
    // fields the new record omits, which is not what `upsert` promises.
    assert_eq!(doc["@search.action"], "upload");
    assert_eq!(doc["id"], "a");
    // Renamed on the way out, as `to_storage` defines it.
    assert_eq!(doc["vec"], json!([1.0, 0.0, 0.0]));
    assert!(doc.get("embedding").is_none());
}

#[tokio::test]
async fn a_partially_rejected_batch_is_an_error_not_a_silent_write() {
    // HTTP 207 is a success status, so a plain `is_success()` check would
    // report a half-written batch as fully written.
    let server = FakeSearch::start(|_| {
        (
            207,
            r#"{"value":[
                {"key":"a","status":true,"statusCode":200},
                {"key":"b","status":false,"statusCode":400,"errorMessage":"field 'year' is not an integer"}
            ]}"#
            .into(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();

    let err = docs
        .upsert(vec![
            json!({"id": "a", "text": "alpha", "year": 2024, "embedding": [1.0, 0.0, 0.0]}),
            json!({"id": "b", "text": "beta", "year": 2024, "embedding": [0.0, 1.0, 0.0]}),
        ])
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("rejected 1 of 2"), "{err}");
    assert!(err.contains("not an integer"), "{err}");
}

#[tokio::test]
async fn delete_addresses_documents_by_their_storage_key() {
    let server = FakeSearch::start(|_| {
        (
            200,
            r#"{"value":[{"key":"a","status":true,"statusCode":200}]}"#.into(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();
    docs.delete(vec![json!("a")]).await.unwrap();

    let body: Value = serde_json::from_slice(&server.requests()[0].2).unwrap();
    assert_eq!(body["value"][0]["@search.action"], "delete");
    assert_eq!(body["value"][0]["id"], "a");
}

#[tokio::test]
async fn get_reads_documents_and_reports_a_missing_key_as_none() {
    let server = FakeSearch::start(|line| {
        if line.contains("docs('a')") {
            (
                200,
                r#"{"id":"a","text":"alpha","year":2024,"vec":[1.0,0.0,0.0]}"#.into(),
            )
        } else {
            (404, r#"{"error":{"message":"not found"}}"#.into())
        }
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();

    let found = docs
        .get(vec![json!("a"), json!("missing")], false)
        .await
        .unwrap();
    assert_eq!(found.len(), 2, "a missing key keeps its slot");
    let record = found[0].clone().unwrap();
    assert_eq!(record["text"], "alpha");
    // Vectors were not requested, so the renamed vector field is dropped
    // rather than returned under its storage name.
    assert!(record.get("embedding").is_none());
    assert!(record.get("vec").is_none());
    assert!(found[1].is_none());
}

#[tokio::test]
async fn search_sends_the_filter_and_maps_results_back() {
    let server = FakeSearch::start(|_| {
        (
            200,
            r#"{"value":[
                {"@search.score":0.93,"id":"a","text":"alpha","year":2024,"vec":[1.0,0.0,0.0]}
            ]}"#
            .into(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();

    let hits = docs
        .search(
            vec![1.0, 0.0, 0.0],
            &VectorSearchOptions::new(5)
                .with_include_vectors(true)
                .with_filter(Filter::gte("year", 2020).unwrap()),
        )
        .await
        .unwrap();

    let (line, _, body) = server.requests().remove(0);
    assert!(
        line.starts_with("POST /indexes('docs')/docs/search"),
        "{line}"
    );
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["filter"], "year ge 2020");
    assert_eq!(body["vectorFilterMode"], "preFilter");
    assert_eq!(body["vectorQueries"][0]["fields"], "vec");

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].score, Some(0.93));
    assert_eq!(hits[0].record["id"], "a");
    // Mapped back to the logical name, and the `@search.*` annotations are
    // not carried into the record.
    assert_eq!(hits[0].record["embedding"], json!([1.0, 0.0, 0.0]));
    assert!(hits[0].record.get("@search.score").is_none());
}

#[tokio::test]
async fn a_hybrid_search_sends_the_text_alongside_the_vector() {
    let server = FakeSearch::start(|_| (200, r#"{"value":[]}"#.into()));
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    // Through the concrete type: the trait's `search` takes only a vector, so
    // hybrid retrieval is reachable via `collection`, not `get_collection`.
    let collection = store.collection("docs", definition()).unwrap();
    collection
        .search_hybrid(
            "annual report",
            vec![1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3),
        )
        .await
        .unwrap();

    let body: Value = serde_json::from_slice(&server.requests()[0].2).unwrap();
    assert_eq!(body["search"], "annual report");
    assert_eq!(body["searchFields"], "text");
    assert_eq!(body["vectorQueries"][0]["kind"], "vector");
}

#[tokio::test]
async fn list_collection_names_reads_the_index_listing() {
    let server = FakeSearch::start(|_| (200, r#"{"value":[{"name":"a"},{"name":"b"}]}"#.into()));
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    assert_eq!(store.list_collection_names().await.unwrap(), ["a", "b"]);
    assert!(server.requests()[0]
        .0
        .starts_with("GET /indexes?$select=name&api-version="));
}

#[tokio::test]
async fn a_service_error_carries_the_status_and_the_service_message() {
    let server = FakeSearch::start(|_| {
        (
            404,
            r#"{"error":{"code":"","message":"No index with the name 'docs' was found."}}"#.into(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.get_collection("docs", definition()).unwrap();
    let err = docs
        .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(1))
        .await
        .unwrap_err()
        .to_string();
    assert!(err.contains("404"), "{err}");
    assert!(err.contains("No index with the name"), "{err}");
}

#[tokio::test]
async fn a_search_says_what_kind_of_score_it_returned() {
    // Azure scores a vector query by relevance (higher is better) whatever
    // metric the profile declares, and a hybrid query by RRF — so a caller
    // reading direction from a `cosine_distance` declaration, whose
    // `higher_is_closer` is false, would rank every result backwards.
    let server = FakeSearch::start(|_| {
        (
            200,
            json!({ "value": [{ "id": "a", "text": "alpha", "@search.score": 0.87 }] }).to_string(),
        )
    });
    let store = AzureAISearchStore::with_api_key(&server.addr, "k");
    let docs = store.collection("docs", definition()).unwrap();

    let vector_hits = docs
        .search(vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(1))
        .await
        .unwrap();
    assert_eq!(
        vector_hits[0].score_kind.as_deref(),
        Some(VectorSearchResult::SCORE_KIND_RELEVANCE)
    );
    assert_eq!(
        vector_hits[0].higher_is_closer(Some(&DistanceFunction::new(
            DistanceFunction::COSINE_DISTANCE
        ))),
        Some(true),
        "the score kind overrides the declared metric, which does not describe it"
    );

    let hybrid_hits = docs
        .search_hybrid("alpha", vec![1.0, 0.0, 0.0], &VectorSearchOptions::new(1))
        .await
        .unwrap();
    assert_eq!(
        hybrid_hits[0].score_kind.as_deref(),
        Some(VectorSearchResult::SCORE_KIND_RRF)
    );
}
