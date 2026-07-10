//! Live integration tests against a real Redis server.
//!
//! Every test resolves an endpoint independently via [`test_server`], which
//! implements both options documented for this work package:
//!
//! 1. If `REDIS_URL` is set, use it as-is. The server is assumed to be
//!    externally managed (e.g. CI service container); this test binary
//!    neither spawns nor tears it down.
//! 2. Otherwise, if a `redis-server` binary is on `PATH` (checked with
//!    `redis-server --version`), spawn a private instance on an ephemeral
//!    localhost port, wait for it to accept connections, and kill it (via a
//!    `Drop` guard) when the test's `ServerGuard` goes out of scope —
//!    including on panic/assertion failure, since unwinding still runs
//!    `Drop`.
//! 3. Otherwise, print a message to stderr and return early. The test still
//!    reports as passed (a "runtime skip", per the work package's second
//!    gating option) rather than failing a CI environment that lacks Redis.
//!
//! Every test also uses a UUID-derived key prefix / thread id so that
//! multiple tests sharing one `REDIS_URL` server (option 1) can't collide.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use agent_framework_core::memory::ContextProvider;
use agent_framework_core::threads::ChatMessageStore;
use agent_framework_core::types::ChatMessage;
use agent_framework_redis::{RedisChatMessageStore, RedisContextProvider};
use uuid::Uuid;

/// Kills the spawned `redis-server` child (if any) when dropped.
enum ServerGuard {
    Spawned(Child),
    External,
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        if let ServerGuard::Spawned(child) = self {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn redis_server_available() -> bool {
    Command::new("redis-server")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn free_ephemeral_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().expect("local_addr").port()
}

async fn wait_until_ready(url: &str) -> bool {
    for _ in 0..50 {
        if let Ok(store) = RedisChatMessageStore::new(url, None) {
            if store.ping().await {
                return true;
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    false
}

/// Resolve a Redis endpoint for this test. Returns `None` (meaning: skip)
/// if neither `REDIS_URL` nor a `redis-server` binary is available.
async fn test_server() -> Option<(String, ServerGuard)> {
    if let Ok(url) = std::env::var("REDIS_URL") {
        return Some((url, ServerGuard::External));
    }

    if !redis_server_available() {
        eprintln!(
            "skipping live Redis test: REDIS_URL is not set and no `redis-server` binary was found on PATH"
        );
        return None;
    }

    let port = free_ephemeral_port();
    let mut dir: PathBuf = std::env::temp_dir();
    dir.push(format!("agent-framework-redis-it-{}", Uuid::new_v4()));
    std::fs::create_dir_all(&dir).expect("create redis-server working dir");

    let mut child = Command::new("redis-server")
        .args([
            "--port",
            &port.to_string(),
            "--bind",
            "127.0.0.1",
            "--save",
            "",
            "--appendonly",
            "no",
            "--daemonize",
            "no",
            "--dir",
        ])
        .arg(&dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn redis-server");

    let url = format!("redis://127.0.0.1:{port}/0");
    if !wait_until_ready(&url).await {
        eprintln!("skipping live Redis test: spawned redis-server did not become ready in time");
        // Don't leave a zombie/orphaned process behind on the skip path.
        let _ = child.kill();
        let _ = child.wait();
        return None;
    }

    Some((url, ServerGuard::Spawned(child)))
}

fn unique(prefix: &str) -> String {
    format!("{prefix}-{}", Uuid::new_v4())
}

#[tokio::test]
async fn chat_message_store_add_list_clear_round_trip() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let store = RedisChatMessageStore::new(&url, Some(unique("thread"))).unwrap();

    assert!(store.list_messages().await.unwrap().is_empty());

    store
        .add_messages(vec![
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ])
        .await
        .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text(), "Hello");
    assert_eq!(messages[1].text(), "Hi there!");

    store
        .add_messages(vec![ChatMessage::user("How are you?")])
        .await
        .unwrap();
    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2].text(), "How are you?");

    store.clear().await.unwrap();
    assert!(store.list_messages().await.unwrap().is_empty());
}

#[tokio::test]
async fn chat_message_store_trims_to_max_messages() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let store = RedisChatMessageStore::new(&url, Some(unique("thread")))
        .unwrap()
        .with_max_messages(3);

    for i in 0..5 {
        store
            .add_messages(vec![ChatMessage::user(format!("message {i}"))])
            .await
            .unwrap();
    }

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 3);
    // LTRIM keeps the most recent 3: messages 2, 3, 4.
    assert_eq!(messages[0].text(), "message 2");
    assert_eq!(messages[1].text(), "message 3");
    assert_eq!(messages[2].text(), "message 4");
}

#[tokio::test]
async fn chat_message_store_auto_generated_thread_ids_are_isolated() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let key_prefix = unique("iso");
    let store_a = RedisChatMessageStore::new(&url, None)
        .unwrap()
        .with_key_prefix(&key_prefix);
    let store_b = RedisChatMessageStore::new(&url, None)
        .unwrap()
        .with_key_prefix(&key_prefix);
    assert_ne!(store_a.thread_id(), store_b.thread_id());

    store_a
        .add_messages(vec![ChatMessage::user("only in A")])
        .await
        .unwrap();

    assert_eq!(store_a.list_messages().await.unwrap().len(), 1);
    assert!(store_b.list_messages().await.unwrap().is_empty());
}

#[tokio::test]
async fn chat_message_store_survives_from_state_round_trip_against_live_server() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let thread_id = unique("thread");
    let store = RedisChatMessageStore::new(&url, Some(thread_id.clone())).unwrap();
    store
        .add_messages(vec![ChatMessage::user("persisted")])
        .await
        .unwrap();

    let state = store.serialize().await.unwrap();
    let restored = RedisChatMessageStore::from_state(&state).unwrap();
    let messages = restored.list_messages().await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "persisted");
}

#[tokio::test]
async fn context_provider_invoked_then_invoking_surfaces_matching_memory() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-1");

    provider
        .invoked(&[ChatMessage::user("I love hiking in the Cascades")], &[])
        .await
        .unwrap();

    let ctx = provider
        .invoking(&[ChatMessage::user("Tell me about hiking")])
        .await
        .unwrap();

    assert_eq!(ctx.messages.len(), 1);
    assert!(ctx.messages[0]
        .text()
        .contains("I love hiking in the Cascades"));
    assert!(ctx.messages[0]
        .text()
        .starts_with("## Memories\nConsider the following memories"));
}

#[tokio::test]
async fn context_provider_invoking_returns_empty_context_when_nothing_matches() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-2");

    provider
        .invoked(&[ChatMessage::user("I love hiking in the Cascades")], &[])
        .await
        .unwrap();

    // No overlapping token with the stored memory.
    let ctx = provider
        .invoking(&[ChatMessage::user("What is the capital of France?")])
        .await
        .unwrap();
    assert!(ctx.messages.is_empty());
}

#[tokio::test]
async fn context_provider_scopes_memories_by_user_id() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let key_prefix = unique("ctx");

    let provider_a = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-a");
    let provider_b = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-b");

    provider_a
        .invoked(
            &[ChatMessage::user("user-a's secret hobby is pottery")],
            &[],
        )
        .await
        .unwrap();

    let ctx_b = provider_b
        .invoking(&[ChatMessage::user("Tell me about pottery")])
        .await
        .unwrap();
    assert!(
        ctx_b.messages.is_empty(),
        "provider scoped to user-b must not see user-a's memories"
    );

    let ctx_a = provider_a
        .invoking(&[ChatMessage::user("Tell me about pottery")])
        .await
        .unwrap();
    assert_eq!(ctx_a.messages.len(), 1);
}

#[tokio::test]
async fn context_provider_thread_created_conflict_is_enforced_end_to_end() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-3")
        .with_scope_to_per_operation_thread_id(true);

    provider.thread_created(Some("thread-1")).await.unwrap();
    let err = provider.thread_created(Some("thread-2")).await.unwrap_err();
    assert!(err.to_string().contains("only be used with one thread"));
}
