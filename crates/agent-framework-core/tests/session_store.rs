//! End-to-end tests of the experimental session store
//! (`experimental-session-store`): saving a real `Agent`'s session, restoring
//! it, and deleting it.
//!
//! Ported from upstream's .NET `InMemoryAgentSessionStoreTests` /
//! `FileSystemAgentSessionStoreTests` (including #8801's delete cases) and
//! Python's `test_sessions.py` session-store tests.
#![cfg(feature = "experimental-session-store")]

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::session_store::*;
use async_trait::async_trait;
use serde_json::json;

/// A chat client that records what it was sent and echoes a numbered reply.
#[derive(Clone, Default)]
struct Echo {
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

#[async_trait]
impl ChatClient for Echo {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let mut seen = self.seen.lock().unwrap();
        seen.push(messages);
        Ok(ChatResponse::from_text(format!("reply {}", seen.len())))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let update = ChatResponseUpdate {
            contents: response.messages[0].contents.clone(),
            role: Some(Role::assistant()),
            ..Default::default()
        };
        Ok(Box::pin(futures::stream::iter([Ok(update)])))
    }
}

fn agent(id: &str, client: &Echo) -> Agent {
    Agent::builder(client.clone()).id(id).build()
}

fn key(id: &str) -> AgentSessionStoreKey {
    AgentSessionStoreKey::new(id).unwrap()
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages.iter().map(Message::text).collect()
}

fn temp_root() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("afr-session-store-{}", uuid::Uuid::new_v4()))
}

async fn round_trip(store: &dyn AgentSessionStore) {
    let client = Echo::default();
    let first = agent("helper", &client);
    let mut session = store
        .get_or_create_session(&first, &key("conv-1"))
        .await
        .unwrap();
    session.state.insert("theme", json!("dark"));
    first
        .run(vec![Message::user("hello")], Some(&mut session))
        .await
        .unwrap();
    store
        .save_session(&first, &key("conv-1"), &session)
        .await
        .unwrap();

    // A different agent instance with the same id picks the conversation up.
    let second = agent("helper", &client);
    let mut restored = store
        .get_session(&second, &key("conv-1"))
        .await
        .unwrap()
        .expect("stored");
    assert_eq!(restored.session_id(), session.session_id());
    assert_eq!(restored.state.get("theme"), Some(json!("dark")));
    second
        .run(vec![Message::user("again")], Some(&mut restored))
        .await
        .unwrap();
    let sent = client.seen.lock().unwrap()[1].clone();
    assert_eq!(texts(&sent), ["hello", "reply 1", "again"]);
}

#[tokio::test]
async fn in_memory_store_round_trips_state_and_history() {
    round_trip(&InMemoryAgentSessionStore::new()).await;
}

#[tokio::test]
async fn file_store_round_trips_state_and_history() {
    let root = temp_root();
    round_trip(&FileAgentSessionStore::new(&root)).await;
    let _ = std::fs::remove_dir_all(root);
}

async fn delete_semantics(store: &dyn AgentSessionStore) {
    let client = Echo::default();
    let a = agent("a", &client);
    // Upstream #8801: deleting a session that was never stored is a no-op.
    store.delete_session(&a, &key("missing")).await.unwrap();

    store
        .save_session(&a, &key("s"), &a.create_session())
        .await
        .unwrap();
    store
        .save_session(&a, &key("other"), &a.create_session())
        .await
        .unwrap();
    store.delete_session(&a, &key("s")).await.unwrap();
    assert!(store.get_session(&a, &key("s")).await.unwrap().is_none());
    assert!(store
        .get_session(&a, &key("other"))
        .await
        .unwrap()
        .is_some());
    // Deleting twice is still fine.
    store.delete_session(&a, &key("s")).await.unwrap();
}

#[tokio::test]
async fn in_memory_store_deletes() {
    delete_semantics(&InMemoryAgentSessionStore::new()).await;
}

#[tokio::test]
async fn file_store_deletes() {
    let root = temp_root();
    delete_semantics(&FileAgentSessionStore::new(&root)).await;
    // A store whose root was never created deletes as a no-op too.
    let empty = FileAgentSessionStore::new(temp_root());
    let a = agent("a", &Echo::default());
    empty.delete_session(&a, &key("s")).await.unwrap();
    let _ = std::fs::remove_dir_all(root);
}

async fn isolation(store: &dyn AgentSessionStore) {
    let client = Echo::default();
    let (a, b) = (agent("a", &client), agent("b", &client));
    let tenant1 = key("s").with_partition("tenant", "t1").unwrap();
    let tenant2 = key("s").with_partition("tenant", "t2").unwrap();
    let session = a.create_session();
    session.state.insert("owner", json!("t1"));
    store.save_session(&a, &tenant1, &session).await.unwrap();

    assert!(store.get_session(&a, &tenant2).await.unwrap().is_none());
    assert!(store.get_session(&a, &key("s")).await.unwrap().is_none());
    assert!(store.get_session(&b, &tenant1).await.unwrap().is_none());

    // Each read is an independent copy.
    let first = store.get_session(&a, &tenant1).await.unwrap().unwrap();
    first.state.insert("owner", json!("changed"));
    let second = store.get_session(&a, &tenant1).await.unwrap().unwrap();
    assert_eq!(second.state.get("owner"), Some(json!("t1")));
}

#[tokio::test]
async fn in_memory_store_isolates_agents_partitions_and_reads() {
    isolation(&InMemoryAgentSessionStore::new()).await;
}

#[tokio::test]
async fn file_store_isolates_agents_partitions_and_reads() {
    let root = temp_root();
    isolation(&FileAgentSessionStore::new(&root)).await;
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn file_store_keeps_hostile_ids_inside_the_root() {
    let root = temp_root();
    let store = FileAgentSessionStore::new(&root);
    let a = agent("../../escape", &Echo::default());
    let hostile = key("../../../etc/passwd")
        .with_partition("tenant", "/abs")
        .unwrap();
    store
        .save_session(&a, &hostile, &a.create_session())
        .await
        .unwrap();
    assert!(store.get_session(&a, &hostile).await.unwrap().is_some());
    let mut files = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(path);
            }
        }
    }
    assert_eq!(files.len(), 1);
    assert!(files[0].starts_with(&root));
    let _ = std::fs::remove_dir_all(root);
}

/// Every file under `root`, relative to it.
fn files_under(root: &std::path::Path) -> Vec<String> {
    let mut files = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else {
                files.push(
                    path.strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
    }
    files.sort();
    files
}

#[tokio::test]
async fn file_store_paths_differ_by_more_than_letter_case() {
    let root = temp_root();
    let store = FileAgentSessionStore::new(&root);
    let a = agent("Agent", &Echo::default());
    for id in ["Foo", "foo"] {
        store
            .save_session(&a, &key(id), &a.create_session())
            .await
            .unwrap();
    }
    let files = files_under(&root);
    assert_eq!(files.len(), 2);
    // Folding case leaves the paths distinct, so a case-insensitive file
    // system keeps two files too.
    let folded: std::collections::HashSet<_> = files.iter().map(|f| f.to_lowercase()).collect();
    assert_eq!(folded.len(), 2, "{files:?}");
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn a_session_id_near_the_name_limit_still_saves() {
    let root = temp_root();
    let store = FileAgentSessionStore::new(&root);
    let a = agent("agent", &Echo::default());
    // 250 bytes once ".json" is added: allowed, and the temporary file
    // must not push the write over the limit.
    let id = "s".repeat(250);
    store
        .save_session(&a, &key(&id), &a.create_session())
        .await
        .unwrap();
    assert!(store.get_session(&a, &key(&id)).await.unwrap().is_some());
    let _ = std::fs::remove_dir_all(root);
}

#[tokio::test]
async fn a_service_session_round_trips_without_history() {
    let store = InMemoryAgentSessionStore::new();
    let client = Echo::default();
    let a = agent("a", &client);
    let session = AgentSession::service("resp_123");
    store.save_session(&a, &key("s"), &session).await.unwrap();
    let restored = store.get_session(&a, &key("s")).await.unwrap().unwrap();
    assert_eq!(restored.service_session_id(), Some("resp_123"));
    assert!(
        restored.context_providers.is_empty(),
        "a local history was attached to a service session"
    );
}
