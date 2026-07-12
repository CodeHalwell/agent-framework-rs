//! One `AgentThread` reused across several `agent.run(...)` calls: each turn
//! automatically sees the full history of every prior turn (both sides --
//! user and assistant messages), accumulated in the thread's message store.
//!
//! Runs fully offline against a canned client that echoes back every user
//! message it has seen so far, so the accumulation is visible in the printed
//! output -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example multi_turn_conversation
//! ```

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
    let agent = ChatAgent::builder(CannedClient).name("assistant").build();
    let mut thread = agent.get_new_thread();

    let turns = [
        "My favorite color is teal.",
        "I have a cat named Whiskers.",
        "What have I told you so far?",
    ];

    for (i, query) in turns.iter().enumerate() {
        let response = agent
            .run(vec![Message::user(*query)], Some(&mut thread))
            .await?;
        let history_len = thread.list_messages().await?.len();
        println!("turn {}: user: {query}", i + 1);
        println!("turn {}: assistant: {}", i + 1, response.text());
        println!("  (thread now holds {history_len} message(s))\n");
    }

    println!("-- full transcript from the thread's message store --");
    for m in thread.list_messages().await? {
        println!("  {}: {}", m.role, m.text());
    }

    Ok(())
}
