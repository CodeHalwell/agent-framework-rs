//! The harness memory provider (`experimental-harness`).
//!
//! Ported from upstream `test_harness_memory.py`.
#![cfg(feature = "experimental-harness")]

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::harness::*;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Answers each call with the next scripted text (or "ok"), recording
/// every request.
/// The scripted replies and the requests received.
type ScriptState = (VecDeque<String>, Vec<Vec<Message>>);

#[derive(Clone, Default)]
struct Scripted(Arc<Mutex<ScriptState>>);

impl Scripted {
    fn new(replies: &[&str]) -> Self {
        let s = Self::default();
        s.0.lock().unwrap().0 = replies.iter().map(|r| r.to_string()).collect();
        s
    }
    fn calls(&self) -> Vec<Vec<Message>> {
        self.0.lock().unwrap().1.clone()
    }
}

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        let mut state = self.0.lock().unwrap();
        state.1.push(messages);
        let reply = state.0.pop_front().unwrap_or_else(|| "ok".into());
        Ok(ChatResponse::from_text(reply))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatStream> {
        unimplemented!("not used")
    }
}

fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("af-memory-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn session_for(owner: &str) -> AgentSession {
    let session = AgentSession::new();
    session.state.insert("user_id", json!(owner));
    session
}

async fn prepare(
    provider: &MemoryContextProvider,
    session: &AgentSession,
    input: &str,
) -> SessionContext {
    let mut ctx = SessionContext::new(vec![Message::user(input)]);
    ctx.session = Some(session.clone());
    ctx.session_id = Some(session.session_id().to_string());
    provider.before_run(&mut ctx).await.unwrap();
    ctx
}

async fn call(
    ctx: &SessionContext,
    name: &str,
    args: Value,
) -> agent_framework_core::error::Result<String> {
    let out = ctx
        .tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tool {name}"))
        .executor
        .as_ref()
        .unwrap()
        .invoke(args)
        .await?;
    Ok(out.as_str().unwrap().to_string())
}

#[tokio::test]
async fn tools_write_read_list_delete_and_index() {
    let base = temp_dir();
    let store = Arc::new(MemoryFileStore::new(&base, "user_id"));
    let provider = MemoryContextProvider::new(store.clone());
    let session = session_for("alice");
    let ctx = prepare(&provider, &session, "hello").await;
    let names: Vec<_> = ctx.tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "list_memory_topics",
            "read_memory_topic",
            "write_memory",
            "delete_memory_topic",
            "search_memory_transcripts",
            "consolidate_memories"
        ]
    );
    assert!(ctx
        .messages
        .last()
        .unwrap()
        .text()
        .contains("### MEMORY.md\n# MEMORY\n\n- none yet"));

    let written = call(
        &ctx,
        "write_memory",
        json!({"topic": "Coffee Prefs", "memory": "Prefers  oat milk."}),
    )
    .await
    .unwrap();
    let record: Value = serde_json::from_str(&written).unwrap();
    assert_eq!(record["slug"], "coffee-prefs");
    assert_eq!(record["memories"], json!(["Prefers oat milk."]));
    call(
        &ctx,
        "write_memory",
        json!({"topic": "coffee prefs", "memory": "prefers oat milk."}),
    )
    .await
    .unwrap();

    let listed: Value =
        serde_json::from_str(&call(&ctx, "list_memory_topics", json!({})).await.unwrap()).unwrap();
    assert_eq!(listed[0]["topic"], "Coffee Prefs");
    let md = call(&ctx, "read_memory_topic", json!({"topic": "coffee-prefs"}))
        .await
        .unwrap();
    assert!(md.starts_with("# Coffee Prefs\n"), "{md}");
    assert_eq!(
        md.matches("oat milk").count(),
        2,
        "summary and one de-duplicated memory: {md}"
    );

    let root = store.memory_root(&session, "memory").unwrap();
    let index = std::fs::read_to_string(root.join("MEMORY.md")).unwrap();
    assert_eq!(
        index,
        "# MEMORY\n\n- [Coffee Prefs](topics/coffee-prefs.md): Prefers oat milk.\n"
    );

    // Another session of the same owner sees the topic and auto-loads it.
    let other = AgentSession::new();
    other.state.insert("user_id", json!("alice"));
    let next = prepare(&provider, &other, "what coffee do I like?").await;
    let memory_message = next.messages.last().unwrap().text();
    assert!(
        memory_message.contains("### topics/coffee-prefs.md\n# Coffee Prefs"),
        "{memory_message}"
    );
    // A different owner does not.
    let bob = prepare(&provider, &session_for("bob"), "coffee").await;
    assert!(bob.messages.last().unwrap().text().contains("- none yet"));

    assert_eq!(
        call(
            &ctx,
            "delete_memory_topic",
            json!({"topic": "Coffee Prefs"})
        )
        .await
        .unwrap(),
        "Deleted memory topic 'Coffee Prefs'."
    );
    assert!(
        call(&ctx, "read_memory_topic", json!({"topic": "Coffee Prefs"}))
            .await
            .is_err()
    );
    std::fs::remove_dir_all(base).unwrap();
}

#[tokio::test]
async fn an_agent_run_records_the_transcript_and_extracts_memories() {
    let base = temp_dir();
    let store = Arc::new(MemoryFileStore::new(&base, "user_id"));
    let extractor = Scripted::new(&[
        "```json\n{\"memories\": [{\"topic\": \"Pets\", \"memory\": \"Has a dog named Rex.\"}, {\"topic\": \" \", \"memory\": \"x\"}]}\n```",
    ]);
    let provider = MemoryContextProvider::builder(store.clone())
        .chat_client(Arc::new(extractor.clone()))
        .recent_turns(1)
        .build()
        .unwrap();
    let agent_client = Scripted::new(&["Nice dog!"]);
    let agent = Agent::builder(Arc::new(agent_client))
        .context_provider(Arc::new(provider.clone()))
        .build();
    let mut session = session_for("carol");
    let response = agent
        .run(
            vec![Message::user("My dog is called Rex")],
            Some(&mut session),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "Nice dog!");

    // The extractor saw the run's transcript delta.
    let calls = extractor.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0][1].text(),
        "USER: My dog is called Rex\n\nASSISTANT: Nice dog!"
    );
    let pets = store
        .get_topic(&session, "memory", "pets")
        .await
        .unwrap()
        .expect("extracted topic");
    assert_eq!(pets.memories, ["Has a dog named Rex."]);
    assert_eq!(pets.session_ids, [session.session_id().to_string()]);

    // The transcript replays as the recent turn, and is searchable.
    let ctx = prepare(&provider, &session, "anything about pets?").await;
    let texts: Vec<String> = ctx.messages.iter().map(Message::text).collect();
    assert_eq!(texts[0], "My dog is called Rex");
    assert_eq!(texts[1], "Nice dog!");
    assert!(texts[2].contains("### topics/pets.md"), "{}", texts[2]);
    let found: Value = serde_json::from_str(
        &call(&ctx, "search_memory_transcripts", json!({"query": "REX"}))
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(found.as_array().unwrap().len(), 1);
    assert_eq!(found[0]["session_id"], session.session_id());
    assert_eq!(found[0]["role"], "user");

    let state = store.read_state(&session, "memory").await.unwrap();
    assert_eq!(
        state.sessions_since_consolidation,
        [session.session_id().to_string()]
    );
    std::fs::remove_dir_all(base).unwrap();
}

#[tokio::test]
async fn consolidation_waits_for_enough_sessions_and_survives_bad_output() {
    let base = temp_dir();
    let store = Arc::new(MemoryFileStore::new(&base, "user_id"));
    let consolidator = Scripted::new(&[
        "not json",
        r#"{"summary": "Tea drinker.", "memories": ["Drinks green tea.", 7]}"#,
    ]);
    let provider = MemoryContextProvider::builder(store.clone())
        .consolidation_client(Arc::new(consolidator.clone()))
        .consolidation_min_sessions(2)
        .consolidation_interval(Duration::from_secs(3600))
        .build()
        .unwrap();
    let first = session_for("dave");
    provider
        .merge_memory(&first, "Drinks", "Drinks green tea.")
        .await
        .unwrap();
    provider
        .merge_memory(&first, "Drinks", "Likes tea.")
        .await
        .unwrap();

    // One session is not enough.
    provider
        .after_run_in_session(
            &first,
            &[Message::user("a")],
            &[Message::assistant("b")],
            None,
        )
        .await
        .unwrap();
    assert!(consolidator.calls().is_empty());

    // The second session triggers it; the malformed reply leaves the topic
    // and the window alone.
    let second = session_for("dave");
    provider
        .after_run_in_session(
            &second,
            &[Message::user("a")],
            &[Message::assistant("b")],
            None,
        )
        .await
        .unwrap();
    assert_eq!(consolidator.calls().len(), 1);
    let state = store.read_state(&second, "memory").await.unwrap();
    assert_eq!(state.sessions_since_consolidation.len(), 2);
    assert!(state.last_consolidated_at.is_none());

    // Forced consolidation applies the next reply and resets the window.
    assert_eq!(provider.consolidate(&second, true).await.unwrap(), 1);
    let drinks = store
        .get_topic(&second, "memory", "drinks")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(drinks.summary, "Tea drinker.");
    assert_eq!(drinks.memories, ["Drinks green tea."]);
    let state = store.read_state(&second, "memory").await.unwrap();
    assert!(state.sessions_since_consolidation.is_empty());
    assert!(parse_memory_timestamp(state.last_consolidated_at.as_deref().unwrap()).is_some());

    // A failed run records nothing.
    provider
        .after_run_in_session(
            &second,
            &[Message::user("q")],
            &[],
            Some(&agent_framework_core::error::Error::Service("x".into())),
        )
        .await
        .unwrap();
    assert!(store
        .read_state(&second, "memory")
        .await
        .unwrap()
        .sessions_since_consolidation
        .is_empty());
    std::fs::remove_dir_all(base).unwrap();
}

#[tokio::test]
async fn the_store_needs_an_owner_without_traversal() {
    let base = temp_dir();
    let store = MemoryFileStore::new(&base, "user_id");
    let anonymous = AgentSession::new();
    assert!(store.memory_root(&anonymous, "memory").is_err());
    assert!(store
        .memory_root(&session_for("../evil"), "memory")
        .is_err());
    let encoded = store
        .memory_root(&session_for("Alice Smith"), "memory")
        .unwrap();
    let rel = encoded.strip_prefix(&base).unwrap();
    assert_eq!(rel.components().count(), 3, "{rel:?}");
    std::fs::remove_dir_all(base).unwrap();
}
