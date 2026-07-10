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

    // Note: structured output is not yet mapped for this provider (see
    // PARITY.md) -- `ChatOptions::response_format` / `ResponseFormat::JsonSchema`
    // is silently ignored by `AnthropicClient` today, unlike the OpenAI and
    // Azure providers.
    Ok(())
}
