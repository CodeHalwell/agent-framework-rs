//! Session + history persistence: `AgentSession::to_dict()` produces a JSON
//! blob for the session's `{session_id, service_session_id, state}` (no
//! message history -- that lives in whichever `HistoryProvider` is attached);
//! `InMemoryHistoryProvider::to_dict()`/`from_dict()` separately
//! serializes/restores the conversation history itself. Persist both blobs
//! together to any storage layer (a file, a database row, a session cache,
//! ...) and hand them back to `AgentSession::from_dict` +
//! `InMemoryHistoryProvider::from_dict` to continue the conversation later.
//!
//! Runs fully offline against a canned client that reports how many messages
//! it was called with, so the restored history is visible in the output --
//! no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example thread_persistence
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Reports how many messages (history + new input) it was called with,
/// standing in for a real model so the restored history is observable.
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

    // A local session with an explicit `InMemoryHistoryProvider` attached --
    // keeping our own handle to it (rather than letting `create_session`
    // attach one implicitly) is what lets us serialize the history below.
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    println!("-- turn 1 --");
    let query = "My name is Ada.";
    let r1 = agent
        .run(vec![Message::user(query)], Some(&mut session))
        .await?;
    println!("user: {query}\nassistant: {}\n", r1.text());

    // Serialize the session's identity/state and the history provider's
    // messages separately -- this is what you'd persist to a file,
    // database, or session store.
    let session_state = session.to_dict();
    let history_state = history.to_dict();
    println!("-- serialized session state --");
    println!("{}\n", serde_json::to_string_pretty(&session_state)?);
    println!("-- serialized history state --");
    println!("{}\n", serde_json::to_string_pretty(&history_state)?);

    // Reconstruct the session and history provider from those blobs.
    let restored_history = Arc::new(InMemoryHistoryProvider::from_dict(&history_state)?);
    let mut restored_session = AgentSession::from_dict(&session_state)?
        .with_context_providers(vec![restored_history as Arc<dyn ContextProvider>]);

    println!("-- turn 2, continuing the restored session --");
    let query = "What is my name?";
    let r2 = agent
        .run(vec![Message::user(query)], Some(&mut restored_session))
        .await?;
    println!("user: {query}\nassistant: {}", r2.text());
    println!(
        "\n(3 message(s) this time -- 2 restored from turn 1 plus this turn's question --\n\
         confirms the restored session carried turn 1's exchange forward.)"
    );

    Ok(())
}
