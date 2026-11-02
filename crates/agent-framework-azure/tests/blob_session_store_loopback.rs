//! End-to-end tests for `AzureBlobAgentSessionStore` against a stateful fake
//! Blob service on loopback: real HTTP, real request shapes, no Azure.
#![cfg(feature = "session-store")]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_azure::{AzureBlobAgentSessionStore, BlobAuth, StaticTokenCredential};
use agent_framework_core::agent::{Agent, SupportsAgentRun};
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::session_store::{AgentSessionStore, AgentSessionStoreKey};
use agent_framework_core::types::{ChatOptions, ChatResponse, Message};

/// One request as the fake service saw it.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    target: String,
    headers: String,
}

#[derive(Default)]
struct FakeState {
    container: bool,
    blobs: HashMap<String, Vec<u8>>,
    seen: Vec<Seen>,
}

struct FakeBlobService {
    url: String,
    state: Arc<Mutex<FakeState>>,
    stop: Arc<AtomicBool>,
}

fn read_request(stream: &mut TcpStream) -> (String, String, String, Vec<u8>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let end = loop {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break buf.len();
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p;
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).to_string();
    let mut first = head.lines().next().unwrap_or_default().split_whitespace();
    let method = first.next().unwrap_or_default().to_string();
    let target = first.next().unwrap_or_default().to_string();
    let len: usize = head
        .to_ascii_lowercase()
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let start = end + 4;
    while buf.len() < start + len {
        let n = stream.read(&mut chunk).unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    (method, target, head, buf[start..start + len].to_vec())
}

fn respond(stream: &mut TcpStream, status: &str, error_code: Option<&str>, body: &[u8]) {
    let code = error_code
        .map(|c| format!("x-ms-error-code: {c}\r\n"))
        .unwrap_or_default();
    let head = format!(
        "HTTP/1.1 {status}\r\n{code}Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes()).unwrap();
    stream.write_all(body).unwrap();
}

impl FakeBlobService {
    fn start(container_exists: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}/sessions", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(FakeState {
            container: container_exists,
            ..Default::default()
        }));
        let stop = Arc::new(AtomicBool::new(false));
        let (st, sp) = (state.clone(), stop.clone());
        std::thread::spawn(move || {
            while !sp.load(Ordering::SeqCst) {
                let mut stream = match listener.accept() {
                    Ok((s, _)) => s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                        continue;
                    }
                    Err(e) => panic!("{e}"),
                };
                stream.set_nonblocking(false).unwrap();
                let (method, target, headers, body) = read_request(&mut stream);
                let mut s = st.lock().unwrap();
                s.seen.push(Seen {
                    method: method.clone(),
                    target: target.clone(),
                    headers,
                });
                let path = target.split('?').next().unwrap().to_string();
                let is_container = target.contains("restype=container");
                match (method.as_str(), is_container) {
                    ("PUT", true) => {
                        if s.container {
                            respond(
                                &mut stream,
                                "409 Conflict",
                                Some("ContainerAlreadyExists"),
                                b"",
                            );
                        } else {
                            s.container = true;
                            respond(&mut stream, "201 Created", None, b"");
                        }
                    }
                    _ if !s.container => {
                        respond(&mut stream, "404 Not Found", Some("ContainerNotFound"), b"")
                    }
                    ("PUT", false) => {
                        s.blobs.insert(path, body);
                        respond(&mut stream, "201 Created", None, b"");
                    }
                    ("GET", false) => match s.blobs.get(&path).cloned() {
                        Some(b) => respond(&mut stream, "200 OK", None, &b),
                        None => respond(&mut stream, "404 Not Found", Some("BlobNotFound"), b""),
                    },
                    ("DELETE", false) => {
                        if s.blobs.remove(&path).is_some() {
                            respond(&mut stream, "202 Accepted", None, b"");
                        } else {
                            respond(&mut stream, "404 Not Found", Some("BlobNotFound"), b"");
                        }
                    }
                    _ => respond(&mut stream, "400 Bad Request", None, b""),
                }
            }
        });
        Self { url, state, stop }
    }

    fn seen(&self) -> Vec<Seen> {
        self.state.lock().unwrap().seen.clone()
    }
}

impl Drop for FakeBlobService {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

#[derive(Clone)]
struct NoModel;

#[async_trait::async_trait]
impl ChatClient for NoModel {
    async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text("unused"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        unimplemented!()
    }
}

fn agent() -> Agent {
    Agent::builder(NoModel).name("a").build()
}

fn token_store(url: &str) -> AzureBlobAgentSessionStore {
    AzureBlobAgentSessionStore::new(
        url,
        BlobAuth::Credential(Arc::new(StaticTokenCredential::new("storage-token"))),
        "my-agent",
    )
    .unwrap()
}

#[tokio::test]
async fn a_session_round_trips_through_blob_storage() {
    let svc = FakeBlobService::start(false);
    let store = token_store(&svc.url);
    let agent = agent();
    let key = AgentSessionStoreKey::new("s1").unwrap();

    assert!(store.get_session(&agent, &key).await.unwrap().is_none());

    let session = agent.create_session();
    session.state.insert("k", serde_json::json!("v"));
    store.save_session(&agent, &key, &session).await.unwrap();

    let restored = store
        .get_session(&agent, &key)
        .await
        .unwrap()
        .expect("stored");
    assert_eq!(restored.session_id(), session.session_id());
    assert_eq!(restored.state.get("k"), Some(serde_json::json!("v")));

    store.delete_session(&agent, &key).await.unwrap();
    assert!(store.get_session(&agent, &key).await.unwrap().is_none());
    // Deleting what is not there is not an error (upstream #8801).
    store.delete_session(&agent, &key).await.unwrap();
}

#[tokio::test]
async fn requests_use_the_blob_api_version_bearer_token_and_blob_type() {
    let svc = FakeBlobService::start(false);
    let store = token_store(&svc.url);
    let agent = agent();
    let key = AgentSessionStoreKey::new("s1").unwrap();
    store
        .save_session(&agent, &key, &agent.create_session())
        .await
        .unwrap();

    let seen = svc.seen();
    // The container is created first, once.
    assert_eq!(seen[0].method, "PUT");
    assert!(
        seen[0].target.contains("restype=container"),
        "{:?}",
        seen[0]
    );
    let put = &seen[1];
    let h = put.headers.to_ascii_lowercase();
    assert!(h.contains("authorization: bearer storage-token"), "{h}");
    assert!(h.contains("x-ms-version: 2023-11-03"), "{h}");
    assert!(h.contains("x-ms-blob-type: blockblob"), "{h}");
    assert!(put.target.starts_with("/sessions/v2/"), "{}", put.target);
    assert_eq!(put.target, format!("/sessions/{}", store.blob_name(&key)));

    // A second operation does not try to create the container again.
    store.get_session(&agent, &key).await.unwrap();
    assert_eq!(
        svc.seen()
            .iter()
            .filter(|s| s.target.contains("restype=container"))
            .count(),
        1
    );
}

#[tokio::test]
async fn an_existing_container_is_not_an_error() {
    let svc = FakeBlobService::start(true);
    let store = token_store(&svc.url);
    let agent = agent();
    let key = AgentSessionStoreKey::new("s").unwrap();
    store
        .save_session(&agent, &key, &agent.create_session())
        .await
        .unwrap();
}

#[tokio::test]
async fn without_container_creation_a_missing_container_is_an_error_not_an_empty_store() {
    // `get_session` must only answer `None` for a missing *blob*: a missing
    // container is a misconfiguration and has to surface.
    let svc = FakeBlobService::start(false);
    let store = token_store(&svc.url).create_container_if_not_exists(false);
    let agent = agent();
    let err = store
        .get_session(&agent, &AgentSessionStoreKey::new("s").unwrap())
        .await
        .unwrap_err();
    assert!(err.to_string().contains("ContainerNotFound"), "{err}");
    assert!(svc
        .seen()
        .iter()
        .all(|s| !s.target.contains("restype=container")));
}

#[tokio::test]
async fn a_sas_token_authenticates_by_query_string() {
    let svc = FakeBlobService::start(true);
    let store = AzureBlobAgentSessionStore::new(
        &svc.url,
        BlobAuth::Sas("?sv=2023&sig=abc".into()),
        "my-agent",
    )
    .unwrap();
    let agent = agent();
    store
        .save_session(
            &agent,
            &AgentSessionStoreKey::new("s").unwrap(),
            &agent.create_session(),
        )
        .await
        .unwrap();
    let put = svc
        .seen()
        .into_iter()
        .find(|s| s.method == "PUT" && !s.target.contains("restype"))
        .unwrap();
    assert!(put.target.ends_with("?sv=2023&sig=abc"), "{}", put.target);
    assert!(!put.headers.to_ascii_lowercase().contains("authorization:"));
}

#[tokio::test]
async fn sessions_are_isolated_by_agent_namespace_and_partition() {
    let svc = FakeBlobService::start(true);
    let a = token_store(&svc.url);
    let b = AzureBlobAgentSessionStore::new(
        &svc.url,
        BlobAuth::Credential(Arc::new(StaticTokenCredential::new("t"))),
        "other-agent",
    )
    .unwrap();
    let agent = agent();
    let key = AgentSessionStoreKey::new("s").unwrap();
    a.save_session(&agent, &key, &agent.create_session())
        .await
        .unwrap();
    assert!(b.get_session(&agent, &key).await.unwrap().is_none());
    let tenant = key.clone().with_partition("tenant", "t2").unwrap();
    assert!(a.get_session(&agent, &tenant).await.unwrap().is_none());
}
