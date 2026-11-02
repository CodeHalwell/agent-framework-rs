//! Hermetic loopback tests for the Qdrant store against a stateful fake
//! Qdrant REST server on a bare `std::net::TcpListener`.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use agent_framework_core::vectors::{
    DistanceFunction, Filter, IndexKind, VectorSearchOptions, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_qdrant::QdrantStore;
use serde_json::{json, Value};

#[derive(Default)]
struct Fake {
    /// Collection name -> (config, points by id string).
    collections: BTreeMap<String, (Value, BTreeMap<String, Value>)>,
    /// Every request: (method, path, api-key header, body).
    requests: Vec<(String, String, Option<String>, Value)>,
}

fn read_request(stream: &mut TcpStream) -> (String, String, Option<String>, Value) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        let n = stream.read(&mut chunk).unwrap();
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
        if n == 0 {
            break buf.len();
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let mut parts = head.lines().next().unwrap_or_default().split(' ');
    let method = parts.next().unwrap_or_default().to_string();
    let path = parts.next().unwrap_or_default().to_string();
    let header = |name: &str| {
        head.lines().find_map(|l| {
            let (k, v) = l.split_once(':')?;
            k.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    let len: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    while buf.len() < header_end + 4 + len {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = &buf[header_end + 4..header_end + 4 + len];
    let body = serde_json::from_slice(body).unwrap_or(Value::Null);
    (method, path, header("api-key"), body)
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn handle(fake: &mut Fake, method: &str, path: &str, body: &Value) -> (u16, Value) {
    let ok = |result: Value| (200, json!({"result": result, "status": "ok"}));
    let completed = || ok(json!({"operation_id": 1, "status": "completed"}));
    let path = path.split('?').next().unwrap();
    let segments: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    if let ["collections", name, _, ..] = segments.as_slice() {
        if *name != "exists" && !fake.collections.contains_key(*name) && segments[2] != "exists" {
            return (
                404,
                json!({"status": {"error": format!("Collection `{name}` doesn't exist!")}}),
            );
        }
    }
    match (method, segments.as_slice()) {
        ("GET", ["collections"]) => ok(json!({"collections": fake
            .collections
            .keys()
            .map(|n| json!({"name": n}))
            .collect::<Vec<_>>()})),
        ("GET", ["collections", name, "exists"]) => {
            ok(json!({"exists": fake.collections.contains_key(*name)}))
        }
        ("PUT", ["collections", name]) => {
            if fake.collections.contains_key(*name) {
                return (409, json!({"status": {"error": "already exists"}}));
            }
            let info = json!({"config": {"params": {"vectors": body["vectors"]}, "hnsw_config": {"m": 16}}, "payload_schema": {}});
            fake.collections
                .insert(name.to_string(), (info, BTreeMap::new()));
            ok(json!(true))
        }
        ("GET", ["collections", name]) => match fake.collections.get(*name) {
            Some((info, _)) => ok(info.clone()),
            None => (404, json!({"status": {"error": "Not found"}})),
        },
        ("DELETE", ["collections", name]) => ok(json!(fake.collections.remove(*name).is_some())),
        ("PUT", ["collections", name, "index"]) => {
            let (info, _) = fake.collections.get_mut(*name).unwrap();
            let field = body["field_name"].as_str().unwrap().to_string();
            info["payload_schema"][field] = json!({"data_type": body["field_schema"]});
            completed()
        }
        ("PUT", ["collections", name, "points"]) => {
            let (_, points) = fake.collections.get_mut(*name).unwrap();
            for p in body["points"].as_array().unwrap() {
                points.insert(p["id"].to_string(), p.clone());
            }
            completed()
        }
        ("POST", ["collections", name, "points"]) => {
            let (_, points) = &fake.collections[*name];
            let with_vector = body["with_vector"].as_bool().unwrap_or(false);
            let found: Vec<Value> = body["ids"]
                .as_array()
                .unwrap()
                .iter()
                .rev() // Qdrant does not promise request order.
                .filter_map(|id| points.get(&id.to_string()))
                .map(|p| {
                    let mut p = p.clone();
                    if !with_vector {
                        p.as_object_mut().unwrap().remove("vector");
                    }
                    p
                })
                .collect();
            ok(json!(found))
        }
        ("POST", ["collections", name, "points", "delete"]) => {
            let (_, points) = fake.collections.get_mut(*name).unwrap();
            for id in body["points"].as_array().unwrap() {
                points.remove(&id.to_string());
            }
            completed()
        }
        ("POST", ["collections", name, "points", "query"]) => {
            let (_, points) = &fake.collections[*name];
            let query = floats(&body["query"]);
            let using = body["using"].as_str().unwrap();
            let mut scored: Vec<(f64, Value)> = points
                .values()
                .map(|p| (cosine(&query, &floats(&p["vector"][using])), p.clone()))
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let offset = body["offset"].as_u64().unwrap_or(0) as usize;
            let limit = body["limit"].as_u64().unwrap() as usize;
            let hits: Vec<Value> = scored
                .into_iter()
                .skip(offset)
                .take(limit)
                .map(|(score, mut p)| {
                    p["score"] = json!(score);
                    if !body["with_vector"].as_bool().unwrap_or(false) {
                        p.as_object_mut().unwrap().remove("vector");
                    }
                    p
                })
                .collect();
            ok(json!({"points": hits}))
        }
        _ => (
            404,
            json!({"status": {"error": format!("no route {method} {path}")}}),
        ),
    }
}

fn serve() -> (String, Arc<Mutex<Fake>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let fake = Arc::new(Mutex::new(Fake::default()));
    let state = fake.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            let (method, path, key, body) = read_request(&mut stream);
            let (status, reply) = {
                let mut fake = state.lock().unwrap();
                fake.requests
                    .push((method.clone(), path.clone(), key, body.clone()));
                handle(&mut fake, &method, &path, &body)
            };
            let text = reply.to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    (url, fake)
}

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::data("year").with_type("int"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap()
}

#[tokio::test]
async fn crud_and_search_round_trip() {
    let (url, fake) = serve();
    let store = QdrantStore::new(&url).with_api_key("secret");
    let docs = store.get_collection("docs", definition()).unwrap();
    assert!(!docs.collection_exists().await.unwrap());
    docs.ensure_collection_exists().await.unwrap();
    // Idempotent: a second call validates instead of creating.
    docs.ensure_collection_exists().await.unwrap();
    assert_eq!(store.list_collection_names().await.unwrap(), ["docs"]);

    let keys = docs
        .upsert(vec![
            json!({"id": 1, "title": "north", "year": 2001, "embedding": [1.0, 0.0]}),
            json!({"id": 2, "title": "east", "year": 2010, "embedding": [0.0, 1.0]}),
            json!({"id": 3, "title": "northeast", "year": 2020, "embedding": [0.7, 0.7]}),
        ])
        .await
        .unwrap();
    assert_eq!(keys, [json!(1), json!(2), json!(3)]);

    let got = docs
        .get(vec![json!(3), json!(9), json!(1)], false)
        .await
        .unwrap();
    assert_eq!(got[0].as_ref().unwrap()["title"], "northeast");
    assert!(got[1].is_none());
    assert_eq!(got[2].as_ref().unwrap()["id"], 1);
    assert!(got[2].as_ref().unwrap().get("embedding").is_none());

    let hits = docs
        .search(
            vec![1.0, 0.1],
            &VectorSearchOptions::new(2).with_filter(Filter::gte("year", 2000).unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].record["id"], 1);
    assert!(hits[0].score.unwrap() > hits[1].score.unwrap());

    docs.delete(vec![json!(1)]).await.unwrap();
    assert!(docs.get(vec![json!(1)], false).await.unwrap()[0].is_none());
    docs.ensure_collection_deleted().await.unwrap();
    assert!(!store.collection_exists("docs").await.unwrap());

    let fake = fake.lock().unwrap();
    assert!(fake
        .requests
        .iter()
        .all(|r| r.2.as_deref() == Some("secret")));
    let index = fake
        .requests
        .iter()
        .find(|r| r.1.contains("/index"))
        .expect("payload index created");
    assert_eq!(
        index.3,
        json!({"field_name": "title", "field_schema": "keyword"})
    );
    let create = fake
        .requests
        .iter()
        .find(|r| r.0 == "PUT" && r.1 == "/collections/docs")
        .unwrap();
    assert_eq!(
        create.3,
        json!({"vectors": {"embedding": {"size": 2, "distance": "Cosine"}}})
    );
    let query = fake
        .requests
        .iter()
        .find(|r| r.1.ends_with("/points/query"))
        .unwrap();
    assert_eq!(
        query.3["filter"],
        json!({"must": [{"key": "year", "range": {"gte": 2000.0}}]})
    );
    assert_eq!(query.3["using"], "embedding");
}

#[tokio::test]
async fn an_existing_mismatched_collection_is_refused() {
    let (url, _fake) = serve();
    let store = QdrantStore::new(&url);
    store
        .get_collection("c", definition())
        .unwrap()
        .ensure_collection_exists()
        .await
        .unwrap();
    let euclid = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::vector("embedding", 2)
            .with_distance_function(DistanceFunction::new(DistanceFunction::EUCLIDEAN_DISTANCE)),
    ])
    .unwrap();
    let err = store
        .get_collection("c", euclid)
        .unwrap()
        .ensure_collection_exists()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("does not match"), "{err}");
    let flat = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::vector("embedding", 2).with_index_kind(IndexKind::new(IndexKind::FLAT)),
    ])
    .unwrap();
    let err = store
        .get_collection("c", flat)
        .unwrap()
        .ensure_collection_exists()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("flat index"), "{err}");
}

#[tokio::test]
async fn server_errors_surface() {
    let (url, _fake) = serve();
    let docs = QdrantStore::new(&url)
        .get_collection("missing", definition())
        .unwrap();
    let err = docs
        .upsert(vec![
            json!({"id": 1, "title": "t", "year": 1, "embedding": [1.0, 0.0]}),
        ])
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Qdrant returned 404: Collection `missing` doesn't exist!"),
        "{err}"
    );
}
