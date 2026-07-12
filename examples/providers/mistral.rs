//! [Mistral AI](https://mistral.ai)'s Chat Completions API
//! (`POST {base_url}/chat/completions`) via `MistralChatClient`.
//!
//! Requires the `mistral` feature.
//!
//! Skips gracefully unless `MISTRAL_API_KEY` is set.
//!
//! ```bash
//! MISTRAL_API_KEY=... cargo run -p agent-framework-examples --example mistral
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(_) = std::env::var("MISTRAL_API_KEY") else {
        println!("set MISTRAL_API_KEY to run this example");
        return Ok(());
    };

    // Reads MISTRAL_API_KEY (and optional MISTRAL_BASE_URL).
    let client = MistralChatClient::from_env("mistral-large-latest")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
