//! A sequential multi-agent workflow: a writer drafts, then an editor revises.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example sequential
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::SequentialBuilder;
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    let writer = Arc::new(
        ChatAgent::builder(client.clone())
            .name("writer")
            .instructions("You write a short first draft paragraph on the given topic.")
            .build(),
    ) as Arc<dyn Agent>;

    let editor = Arc::new(
        ChatAgent::builder(client)
            .name("editor")
            .instructions("You improve the previous draft for clarity and concision.")
            .build(),
    ) as Arc<dyn Agent>;

    let workflow = SequentialBuilder::new()
        .participants(vec![writer, editor])
        .name("write-then-edit")
        .build()?;

    let result = workflow
        .run("The benefits of Rust for systems programming")
        .await?;

    if let Some(output) = result.last_output() {
        let conversation: Vec<Message> = serde_json::from_value(output).unwrap_or_default();
        if let Some(last) = conversation.last() {
            println!("Final:\n{}", last.text());
        }
    }
    Ok(())
}
