//! Azure AI Foundry Prompt Agents: `FoundryChatClient` speaks the Foundry
//! project **Responses API** (`{endpoint}/openai/v1/responses`, path-versioned
//! -- no `?api-version=` query) and plugs into `FoundryAgent`/`Agent` like any
//! other `ChatClient`. Entra ID auth comes from the `azure` crate's credential
//! chain -- here the Azure CLI credential (`az login`), but
//! `ClientSecretCredential`, `ManagedIdentityCredential`, or a
//! `ChainedTokenCredential` work the same.
//!
//! `FoundryAgent` realizes a Prompt Agent client-side over the Responses API;
//! it does not bind to a server-hosted agent by id (the Foundry Agents
//! control plane) -- see the crate docs.
//!
//! Skips gracefully unless FOUNDRY_ENDPOINT is set (e.g.
//! `https://<resource>.services.ai.azure.com/api/projects/<project>`).
//! Optional: FOUNDRY_MODEL (default `gpt-4o-mini`).
//!
//! ```bash
//! az login && FOUNDRY_ENDPOINT=https://... \
//! cargo run -p agent-framework-examples --example azure_foundry_agent
//! ```

use std::sync::Arc;

use agent_framework::azure::AzureCliCredential;
use agent_framework::foundry::FOUNDRY_SCOPE;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(endpoint) = std::env::var("FOUNDRY_ENDPOINT") else {
        println!("set FOUNDRY_ENDPOINT (and run `az login`) to run this example");
        return Ok(());
    };
    let model = std::env::var("FOUNDRY_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    // Tokens are fetched per request via `az account get-access-token` for
    // the Foundry scope.
    let credential = Arc::new(AzureCliCredential::new(FOUNDRY_SCOPE));
    let client = FoundryChatClient::with_token_credential(&endpoint, &model, credential);

    let agent = FoundryAgent::builder(client)
        .name("foundry-assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent
        .run(
            vec![Message::user("Say hello from Azure AI Foundry!")],
            None,
        )
        .await?;
    println!("{}", response.text());

    Ok(())
}
