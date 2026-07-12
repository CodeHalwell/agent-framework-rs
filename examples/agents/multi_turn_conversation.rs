//! One `AgentSession` reused across several `agent.run(...)` calls: each turn
//! automatically sees the full history of every prior turn (both sides --
//! user and assistant messages), accumulated by the session's attached
//! `InMemoryHistoryProvider`.
//!
//! Runs fully offline against a canned client that echoes back every user
//! message it has seen so far, so the accumulation is visible in the printed
//! output -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example multi_turn_conversation
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Echoes back every user message seen so far, standing in for a real model
/// so history accumulation is observable.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let user_texts: Vec<String> = messages
            .iter()
            .filter(|m| m.role == Role::user())
            .map(Message::text)
            .collect();
        Ok(ChatResponse::from_text(format!(
            "(canned reply) so far you've told me: {}",
            user_texts.join(" | ")
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

    // Keep our own handle to the `InMemoryHistoryProvider` (rather than
    // letting `create_session()` attach one implicitly) so we can inspect
    // the accumulated history below.
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    let turns = [
        "My favorite color is teal.",
        "I have a cat named Whiskers.",
        "What have I told you so far?",
    ];

    for (i, query) in turns.iter().enumerate() {
        let response = agent
            .run(vec![Message::user(*query)], Some(&mut session))
            .await?;
        let history_len = history.list_messages().len();
        println!("turn {}: user: {query}", i + 1);
        println!("turn {}: assistant: {}", i + 1, response.text());
        println!("  (session now holds {history_len} message(s))\n");
    }

    println!("-- full transcript from the session's history provider --");
    for m in history.list_messages() {
        println!("  {}: {}", m.role, m.text());
    }

    Ok(())
}
