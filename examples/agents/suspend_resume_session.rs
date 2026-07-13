//! Suspend a conversation mid-flight and resume it later: an `AgentSession`
//! serializes to a small `{session_id, service_session_id, state}` JSON blob
//! via `to_dict()` -- deliberately **without** the message history, which
//! lives in whichever `HistoryProvider` is attached and serializes itself
//! separately (`InMemoryHistoryProvider::to_dict()`/`from_dict()`). To
//! suspend, persist both blobs (here: one JSON envelope, as you might store
//! in a database row); to resume, rebuild the session with
//! `Agent::session_from_dict` and reattach a history provider restored from
//! its own blob. The free-form `session.state` bag -- where context
//! providers stash per-session data -- rides along in the session blob.
//!
//! Runs fully offline against a canned client that reports how many messages
//! each request carried, so the restored history is visible -- no API key or
//! network needed. (`thread_persistence` shows the same round-trip in its
//! minimal form; this example adds session state and the one-envelope
//! pattern.)
//!
//! ```bash
//! cargo run -p agent-framework-examples --example suspend_resume_session
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// Reports how many messages (history + new input) it was called with.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(format!(
            "(canned reply) this request carried {} message(s) of history/context.",
            messages.len()
        )))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(CannedClient).name("assistant").build();

    // An explicit history provider (keeping our own handle is what lets us
    // serialize the history below), plus some per-session state -- the kind
    // of thing a context provider records across runs.
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    session.state.insert("user_tier", json!("pro"));
    session.state.insert("turns_used", json!(1));

    println!("-- turn 1 --");
    let query = "Remember: my project is called 'skylark'.";
    let r1 = agent
        .run(vec![Message::user(query)], Some(&mut session))
        .await?;
    println!("user: {query}\nassistant: {}\n", r1.text());

    // Suspend: both blobs into one envelope. Note the session blob carries
    // the state bag but NO messages -- history is serialized separately.
    let envelope = json!({
        "session": session.to_dict(),
        "history": history.to_dict(),
    });
    assert!(envelope["session"].get("messages").is_none());
    assert_eq!(envelope["session"]["state"]["user_tier"], "pro");
    assert_eq!(envelope["history"]["messages"].as_array().unwrap().len(), 2);
    let stored = serde_json::to_string_pretty(&envelope)?;
    println!("-- suspended: the persisted envelope --\n{stored}\n");
    drop(session);
    drop(history);

    // ... process exit, days pass, the user comes back ...

    // Resume: rebuild the session from the envelope and reattach a restored
    // history provider (`from_dict` never restores providers itself).
    let envelope: serde_json::Value = serde_json::from_str(&stored)?;
    let history = Arc::new(InMemoryHistoryProvider::from_dict(&envelope["history"])?);
    let mut session = agent
        .session_from_dict(&envelope["session"])?
        .with_context_providers(vec![history as Arc<dyn ContextProvider>]);

    // The state bag survived the round-trip and stays live for further use.
    assert_eq!(session.state.get("user_tier"), Some(json!("pro")));
    session.state.insert("turns_used", json!(2));

    println!("-- turn 2, resumed --");
    let query = "What is my project called?";
    let r2 = agent
        .run(vec![Message::user(query)], Some(&mut session))
        .await?;
    println!("user: {query}\nassistant: {}", r2.text());
    println!(
        "\n(3 message(s) this time -- turn 1's exchange restored from the envelope\n\
         plus this turn's question -- and session state ({} entries) came back too.)",
        session.state.len()
    );

    Ok(())
}
