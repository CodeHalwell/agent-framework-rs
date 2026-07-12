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
//!
//! # RediSearch (`FT.*`) coverage
//!
//! [`RedisContextProvider`]'s `FT.SEARCH`-backed retrieval path (see the
//! crate's `context_provider` module docs) only activates against a **Redis
//! Stack** server — plain `redis-server` (the only kind this file ever
//! spawns itself) has no RediSearch module loaded. So:
//!
//! - The existing tests below exercise the SCAN-fallback path exactly as
//!   before (this is what a self-spawned plain server gives you), proving
//!   fallback correctness end-to-end regardless of RediSearch availability.
//! - The `context_provider_redisearch_*` tests additionally probe the
//!   connected server for RediSearch via [`redisearch_available`] and
//!   `return` early (a graceful runtime skip, printed to stderr) unless it's
//!   present — which in practice means they only actually assert anything
//!   when `REDIS_URL` is pointed at a Redis Stack server. They are still
//!   compiled and run (as no-ops) in any environment lacking Stack, so
//!   nothing about this suite requires Stack to be installed.

use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::Message;
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

/// Probe `url` directly (independent of [`RedisContextProvider`]'s own,
/// private capability cache) for a loaded RediSearch module via `FT._LIST`.
/// Used to gate the `context_provider_redisearch_*` tests below: they only
/// assert anything when this returns `true`, which in practice means
/// `REDIS_URL` was pointed at a Redis Stack server — the plain
/// `redis-server` this file spawns itself never has RediSearch loaded.
async fn redisearch_available(url: &str) -> bool {
    let Ok(client) = redis::Client::open(url) else {
        return false;
    };
    let Ok(mut conn) = client.get_multiplexed_async_connection().await else {
        return false;
    };
    redis::cmd("FT._LIST")
        .query_async::<Vec<String>>(&mut conn)
        .await
        .is_ok()
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
            Message::user("Hello"),
            Message::assistant("Hi there!"),
        ])
        .await
        .unwrap();

    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].text(), "Hello");
    assert_eq!(messages[1].text(), "Hi there!");

    store
        .add_messages(vec![Message::user("How are you?")])
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
            .add_messages(vec![Message::user(format!("message {i}"))])
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
    assert_ne!(store_a.session_id(), store_b.session_id());

    store_a
        .add_messages(vec![Message::user("only in A")])
        .await
        .unwrap();

    assert_eq!(store_a.list_messages().await.unwrap().len(), 1);
    assert!(store_b.list_messages().await.unwrap().is_empty());
}

#[tokio::test]
async fn chat_message_store_survives_to_dict_round_trip_against_live_server() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let thread_id = unique("thread");
    let store = RedisChatMessageStore::new(&url, Some(thread_id.clone())).unwrap();
    store
        .add_messages(vec![Message::user("persisted")])
        .await
        .unwrap();

    let state = store.to_dict();
    let restored = RedisChatMessageStore::from_dict(&state).unwrap();
    let messages = restored.list_messages().await.unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].text(), "persisted");
}

#[tokio::test]
async fn chat_message_store_before_run_prepends_history_and_after_run_records_it() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let store = RedisChatMessageStore::new(&url, Some(unique("thread"))).unwrap();

    // Nothing stored yet: before_run leaves the run's own messages untouched.
    let mut ctx = SessionContext::new(vec![Message::user("first turn")]);
    store.before_run(&mut ctx).await.unwrap();
    assert!(ctx.messages.is_empty());

    // A successful run records the request + response messages.
    store
        .after_run(
            &[Message::user("first turn")],
            &[Message::assistant("first reply")],
            None,
        )
        .await
        .unwrap();

    // The next run sees the recorded turn prepended ahead of its own input.
    let mut ctx2 = SessionContext::new(vec![Message::user("second turn")]);
    store.before_run(&mut ctx2).await.unwrap();
    let texts: Vec<String> = ctx2.messages.iter().map(|m| m.text()).collect();
    assert_eq!(
        texts,
        vec!["first turn".to_string(), "first reply".to_string()]
    );

    // A failed run must not record anything.
    store
        .after_run(
            &[Message::user("second turn")],
            &[],
            Some(&agent_framework_core::error::Error::service("boom")),
        )
        .await
        .unwrap();
    let messages = store.list_messages().await.unwrap();
    assert_eq!(messages.len(), 2);

    assert!(store.is_history_provider());
}

#[tokio::test]
async fn context_provider_after_run_then_before_run_surfaces_matching_memory() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-1");

    provider
        .after_run(&[Message::user("I love hiking in the Cascades")], &[], None)
        .await
        .unwrap();

    let mut ctx = SessionContext::new(vec![Message::user("Tell me about hiking")]);
    provider.before_run(&mut ctx).await.unwrap();

    assert_eq!(ctx.messages.len(), 1);
    assert!(ctx.messages[0]
        .text()
        .contains("I love hiking in the Cascades"));
    assert!(ctx.messages[0]
        .text()
        .starts_with("## Memories\nConsider the following memories"));
}

#[tokio::test]
async fn context_provider_before_run_returns_empty_context_when_nothing_matches() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-2");

    provider
        .after_run(&[Message::user("I love hiking in the Cascades")], &[], None)
        .await
        .unwrap();

    // No overlapping token with the stored memory.
    let mut ctx = SessionContext::new(vec![Message::user("What is the capital of France?")]);
    provider.before_run(&mut ctx).await.unwrap();
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
        .after_run(
            &[Message::user("user-a's secret hobby is pottery")],
            &[],
            None,
        )
        .await
        .unwrap();

    let mut ctx_b = SessionContext::new(vec![Message::user("Tell me about pottery")]);
    provider_b.before_run(&mut ctx_b).await.unwrap();
    assert!(
        ctx_b.messages.is_empty(),
        "provider scoped to user-b must not see user-a's memories"
    );

    let mut ctx_a = SessionContext::new(vec![Message::user("Tell me about pottery")]);
    provider_a.before_run(&mut ctx_a).await.unwrap();
    assert_eq!(ctx_a.messages.len(), 1);
}

#[tokio::test]
async fn context_provider_session_id_conflict_is_enforced_end_to_end() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-it-3")
        .with_scope_to_per_operation_thread_id(true);

    let mut ctx1 = SessionContext::new(vec![]);
    ctx1.session_id = Some("thread-1".to_string());
    provider.before_run(&mut ctx1).await.unwrap();

    let mut ctx2 = SessionContext::new(vec![]);
    ctx2.session_id = Some("thread-2".to_string());
    let err = provider.before_run(&mut ctx2).await.unwrap_err();
    assert!(err.to_string().contains("only be used with one thread"));
}

/// Forcing the SCAN fallback must behave identically no matter what the
/// connected server actually supports — this is the "still asserting
/// fallback correctness end-to-end" case that always runs (never gated on
/// RediSearch availability), including when `REDIS_URL` happens to point at
/// a Redis Stack server.
#[tokio::test]
async fn context_provider_force_scan_fallback_works_regardless_of_redisearch_availability() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ctx"))
        .with_user_id("user-force-fallback")
        .with_force_scan_fallback(true);

    provider
        .after_run(&[Message::user("I love hiking in the Cascades")], &[], None)
        .await
        .unwrap();

    let mut ctx = SessionContext::new(vec![Message::user("Tell me about hiking")]);
    provider.before_run(&mut ctx).await.unwrap();

    assert_eq!(ctx.messages.len(), 1);
    assert!(ctx.messages[0]
        .text()
        .contains("I love hiking in the Cascades"));
}

#[tokio::test]
async fn context_provider_redisearch_finds_and_excludes_memories_when_available() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    if !redisearch_available(&url).await {
        eprintln!(
            "skipping RediSearch-specific test: connected Redis server has no RediSearch \
             module loaded (point REDIS_URL at a Redis Stack server to exercise FT.SEARCH)"
        );
        return;
    }

    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ftctx"))
        .with_user_id("user-ft-1");

    provider
        .after_run(&[Message::user("I love hiking in the Cascades")], &[], None)
        .await
        .unwrap();

    let mut ctx = SessionContext::new(vec![Message::user("Tell me about hiking")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert_eq!(ctx.messages.len(), 1);
    assert!(ctx.messages[0]
        .text()
        .contains("I love hiking in the Cascades"));
    assert!(ctx.messages[0]
        .text()
        .starts_with("## Memories\nConsider the following memories"));

    // No overlapping meaningful token -> FT.SEARCH finds nothing, mirroring
    // the fallback path's equivalent assertion.
    let mut ctx_empty = SessionContext::new(vec![Message::user("What is the capital of France?")]);
    provider.before_run(&mut ctx_empty).await.unwrap();
    assert!(ctx_empty.messages.is_empty());
}

#[tokio::test]
async fn context_provider_redisearch_scopes_memories_by_tag_filter_when_available() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    if !redisearch_available(&url).await {
        eprintln!("skipping RediSearch-specific test: no RediSearch module loaded");
        return;
    }

    let key_prefix = unique("ftctx");
    let provider_a = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-ft-a");
    let provider_b = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-ft-b");

    provider_a
        .after_run(
            &[Message::user("user-ft-a's secret hobby is pottery")],
            &[],
            None,
        )
        .await
        .unwrap();

    let mut ctx_b = SessionContext::new(vec![Message::user("Tell me about pottery")]);
    provider_b.before_run(&mut ctx_b).await.unwrap();
    assert!(
        ctx_b.messages.is_empty(),
        "provider scoped to user-ft-b must not see user-ft-a's memories via FT.SEARCH's TAG filter"
    );

    let mut ctx_a = SessionContext::new(vec![Message::user("Tell me about pottery")]);
    provider_a.before_run(&mut ctx_a).await.unwrap();
    assert_eq!(ctx_a.messages.len(), 1);
}

#[tokio::test]
async fn context_provider_redisearch_respects_limit_when_available() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    if !redisearch_available(&url).await {
        eprintln!("skipping RediSearch-specific test: no RediSearch module loaded");
        return;
    }

    let provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(unique("ftctx"))
        .with_user_id("user-ft-limit")
        .with_limit(2);

    for i in 0..5 {
        provider
            .after_run(
                &[Message::user(format!("apple fact number {i}"))],
                &[],
                None,
            )
            .await
            .unwrap();
    }

    let mut ctx = SessionContext::new(vec![Message::user("apple")]);
    provider.before_run(&mut ctx).await.unwrap();
    assert_eq!(ctx.messages.len(), 1);
    // DEFAULT_CONTEXT_PROMPT is itself two lines; every line after that is
    // one matched memory, so this counts how many FT.SEARCH actually
    // returned under `LIMIT 0 2`.
    let memory_line_count = ctx.messages[0].text().lines().skip(2).count();
    assert_eq!(memory_line_count, 2);
}

/// Entries written by the RediSearch path (`JSON.SET`) are a different
/// Redis value type than the SCAN fallback's plain `SET` strings — the
/// module docs call this out explicitly. Confirm it end-to-end: a memory
/// stored while RediSearch is in use is invisible to a *second*, otherwise
/// identically-scoped provider that has fallback forced on.
#[tokio::test]
async fn context_provider_redisearch_entries_are_not_visible_to_forced_scan_fallback() {
    let Some((url, _guard)) = test_server().await else {
        return;
    };
    if !redisearch_available(&url).await {
        eprintln!("skipping RediSearch-specific test: no RediSearch module loaded");
        return;
    }

    let key_prefix = unique("ftctx");
    let ft_provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-ft-cross");
    let fallback_provider = RedisContextProvider::new(&url)
        .unwrap()
        .with_key_prefix(&key_prefix)
        .with_user_id("user-ft-cross")
        .with_force_scan_fallback(true);

    ft_provider
        .after_run(&[Message::user("stored via JSON.SET")], &[], None)
        .await
        .unwrap();

    let mut ctx = SessionContext::new(vec![Message::user("Tell me about JSON.SET")]);
    fallback_provider.before_run(&mut ctx).await.unwrap();
    assert!(
        ctx.messages.is_empty(),
        "SCAN+MGET must not see a JSON.SET-backed entry (documented storage-encoding divergence)"
    );
}
