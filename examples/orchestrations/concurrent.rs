//! Concurrent orchestration: `ConcurrentBuilder` fans the same prompt out to
//! several agents in parallel and aggregates their replies into one
//! conversation. Rust analogue of the Python
//! `orchestration/concurrent_agents.py` sample.
//!
//! Runs fully offline against scripted `ChatClient`s -- no API key or
//! network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example concurrent
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// A chat client that always answers with the same scripted line -- stands
/// in for a real LLM backend so the example needs no API key or network
/// access.
struct CannedClient(&'static str);

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.0))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = resp
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

fn expert(name: &str, reply: &'static str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(CannedClient(reply)).name(name).build()) as Arc<dyn SupportsAgentRun>
}

#[tokio::main]
async fn main() -> Result<()> {
    let researcher = expert(
        "researcher",
        "Urban commuters want low upfront cost and easy charging.",
    );
    let marketer = expert("marketer", "Position it as freedom without the fuel bill.");
    let legal = expert(
        "legal",
        "Verify local e-bike class limits before advertising top speed.",
    );

    let workflow = ConcurrentBuilder::new()
        .participants(vec![researcher, marketer, legal])
        .name("product-launch-review")
        .build()?;

    let run = workflow
        .run("We're launching a budget electric bike for urban commuters.")
        .await?;

    let conversation: Vec<Message> =
        serde_json::from_value(run.last_output().unwrap_or_default()).unwrap_or_default();
    println!("Consolidated review ({} messages):", conversation.len());
    for msg in &conversation {
        let speaker = msg.author_name.as_deref().unwrap_or(msg.role.as_str());
        println!("- {speaker}: {}", msg.text());
    }
    assert_eq!(conversation.len(), 4, "prompt + 3 concurrent replies");

    Ok(())
}
