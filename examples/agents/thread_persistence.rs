//! Thread persistence: `thread.serialize()` produces a JSON blob you can hand
//! to any storage layer (a file, a database row, a session cache, ...);
//! `ChatAgent::deserialize_thread` rebuilds a thread from that blob, ready to
//! continue the conversation.
//!
//! Runs fully offline against a canned client that reports how many messages
//! it was called with, so the restored history is visible in the output --
//! no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example thread_persistence
//! ```

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
    let agent = ChatAgent::builder(CannedClient).name("assistant").build();
    let mut thread = agent.get_new_thread();

    println!("-- turn 1 --");
    let query = "My name is Ada.";
    let r1 = agent
        .run(vec![Message::user(query)], Some(&mut thread))
        .await?;
    println!("user: {query}\nassistant: {}\n", r1.text());

    // Serialize the thread's state -- a `type`-tagged JSON object carrying
    // either a service-managed conversation id or (as here, for a local
    // thread) the full message history. This is what you'd persist to a
    // file, database, or session store.
    let serialized = thread.serialize().await?;
    println!("-- serialized thread state --");
    println!("{}\n", serde_json::to_string_pretty(&serialized)?);

    // Reconstruct a thread from that blob. `ChatAgent::deserialize_thread`
    // populates a fresh message store (built the same way `get_new_thread`
    // would) from the serialized messages.
    let mut restored = agent.deserialize_thread(&serialized).await?;

    println!("-- turn 2, continuing the restored thread --");
    let query = "What is my name?";
    let r2 = agent
        .run(vec![Message::user(query)], Some(&mut restored))
        .await?;
    println!("user: {query}\nassistant: {}", r2.text());
    println!(
        "\n(3 message(s) this time -- 2 restored from turn 1 plus this turn's question --\n\
         confirms the restored thread carried turn 1's exchange forward.)"
    );

    Ok(())
}
