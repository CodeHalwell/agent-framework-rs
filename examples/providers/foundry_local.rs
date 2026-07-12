//! [Microsoft Foundry Local](https://learn.microsoft.com/azure/ai-foundry/foundry-local/)'s
//! OpenAI-compatible chat endpoint via `FoundryLocalChatClient`
//! (`POST {base_url}/chat/completions`, default `http://localhost:5273/v1`).
//!
//! Foundry Local runs models on-device behind an OpenAI-compatible server, so
//! no API key is required for a stock local install. Requires the
//! `foundry-local` feature.
//!
//! Skips gracefully unless `FOUNDRY_LOCAL_ENDPOINT` is set, so it doesn't
//! silently try to reach a server that may not be running. In real deployments
//! the port is discovered from the Foundry Local service; here it is
//! configurable via that variable (and `FOUNDRY_LOCAL_API_KEY` if the server
//! requires one).
//!
//! ```bash
//! FOUNDRY_LOCAL_ENDPOINT=http://localhost:5273/v1 \
//!   cargo run -p agent-framework-examples --example foundry_local
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(_) = std::env::var("FOUNDRY_LOCAL_ENDPOINT") else {
        println!("set FOUNDRY_LOCAL_ENDPOINT (e.g. http://localhost:5273/v1) to run this example, with a Foundry Local server reachable there");
        return Ok(());
    };

    // Reads FOUNDRY_LOCAL_ENDPOINT (base URL) and optional FOUNDRY_LOCAL_API_KEY.
    let client = FoundryLocalChatClient::from_env("phi-4")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
