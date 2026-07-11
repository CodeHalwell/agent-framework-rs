//! Minimal agent example.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework --example quickstart
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of France?").await?;
    println!("{}", response.text());
    Ok(())
}
