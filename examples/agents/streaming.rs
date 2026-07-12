//! Streaming responses token-by-token.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example streaming
//! ```

use agent_framework::prelude::*;
use futures::StreamExt;
use std::io::Write;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;
    let agent = ChatAgent::builder(client)
        .instructions("You are a helpful assistant.")
        .build();

    let mut stream = agent.run_stream_once("Write a haiku about Rust.").await?;
    while let Some(update) = stream.next().await {
        let update = update?;
        print!("{}", update.text());
        std::io::stdout().flush().ok();
    }
    println!();
    Ok(())
}
