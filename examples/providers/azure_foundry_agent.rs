//! Azure AI Foundry persistent agents: `AzureAIAgentClient` speaks the
//! Foundry Agents REST routes (Assistants conventions -- agents, threads,
//! runs) and plugs into `ChatAgent` like any other `ChatClient`. Entra ID
//! auth comes from the `azure` crate's credential chain -- here the Azure CLI
//! credential (`az login`), but `ClientSecretCredential`,
//! `ManagedIdentityCredential`, or a `ChainedTokenCredential` work the same.
//!
//! Skips gracefully unless AZURE_AI_PROJECT_ENDPOINT is set (e.g.
//! `https://<resource>.services.ai.azure.com/api/projects/<project>`).
//! Optional: AZURE_AI_MODEL_DEPLOYMENT_NAME (default `gpt-4o-mini`).
//!
//! ```bash
//! az login && AZURE_AI_PROJECT_ENDPOINT=https://... \
//! cargo run -p agent-framework-examples --example azure_foundry_agent
//! ```

use std::sync::Arc;

use agent_framework::azure::AzureCliCredential;
use agent_framework::azure_ai::AI_FOUNDRY_SCOPE;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(endpoint) = std::env::var("AZURE_AI_PROJECT_ENDPOINT") else {
        println!("set AZURE_AI_PROJECT_ENDPOINT (and run `az login`) to run this example");
        return Ok(());
    };
    let model = std::env::var("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());

    // Tokens are fetched per request via `az account get-access-token` for
    // the Foundry scope. `AzureAIAgentClient::with_existing_agent` targets a
    // pre-created persistent agent instead; `new` auto-creates a transient
    // one on first use and deletes it on `close()`.
    let credential = Arc::new(AzureCliCredential::new(AI_FOUNDRY_SCOPE));
    let client = AzureAIAgentClient::new(&endpoint, &model, credential)
        .with_agent_name("rust-example-agent");

    let agent = ChatAgent::builder(client.clone())
        .name("foundry-assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("Say hello from Azure AI Foundry!").await?;
    println!("{}", response.text());

    // Delete the auto-created transient agent server-side.
    client.close().await?;
    Ok(())
}
