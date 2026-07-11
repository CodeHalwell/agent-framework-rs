//! The Anthropic (Claude) Messages API client.
//!
//! Requires the `anthropic` feature:
//! ```bash
//! ANTHROPIC_API_KEY=sk-ant-... cargo run -p agent-framework --example anthropic --features anthropic
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads ANTHROPIC_API_KEY (and optional ANTHROPIC_BASE_URL).
    let client = AnthropicClient::from_env("claude-sonnet-4-5-20250929")?;

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    // Note: the Messages API has no native structured-output field, so
    // `ChatOptions::response_format` is mapped by appending a strict-JSON
    // instruction (and the schema, for `ResponseFormat::JsonSchema`) to the
    // system prompt.
    Ok(())
}
