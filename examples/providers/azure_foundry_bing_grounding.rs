//! Bing grounding on Azure AI Foundry: `hosted_web_search().connection_id(id)`
//! builds a `bing_grounding` tool from a Bing Grounding connection
//! configured on your Foundry project. Unlike Anthropic's/OpenAI's own
//! hosted web search, Azure AI Foundry requires a connection id (or, for Bing
//! Custom Search, `.custom_connection(id, instance_name)`) -- there is no
//! keyless default.
//!
//! Entra ID auth comes from the `azure` crate's credential chain -- here the
//! Azure CLI credential (`az login`); `DefaultAzureCredential`,
//! `ClientSecretCredential`, `ManagedIdentityCredential`, or a
//! `ChainedTokenCredential` all work the same way (see
//! `providers/azure_default_credential.rs`).
//!
//! Skips gracefully unless AZURE_AI_PROJECT_ENDPOINT and BING_CONNECTION_ID
//! are both set (e.g. `AZURE_AI_PROJECT_ENDPOINT=https://<resource>.services.ai.azure.com/api/projects/<project>`).
//! Optional: AZURE_AI_MODEL_DEPLOYMENT_NAME (default `gpt-4o-mini`).
//!
//! ```bash
//! az login && AZURE_AI_PROJECT_ENDPOINT=https://... BING_CONNECTION_ID=/subscriptions/.../connections/... \
//! cargo run -p agent-framework-examples --example azure_foundry_bing_grounding
//! ```

use std::sync::Arc;

use agent_framework::azure::AzureCliCredential;
use agent_framework::azure_ai::AI_FOUNDRY_SCOPE;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(connection_id)) = (
        std::env::var("AZURE_AI_PROJECT_ENDPOINT"),
        std::env::var("BING_CONNECTION_ID"),
    ) else {
        println!(
            "set AZURE_AI_PROJECT_ENDPOINT and BING_CONNECTION_ID (and run `az login`) to run \
             this example"
        );
        return Ok(());
    };
    let model = std::env::var("AZURE_AI_MODEL_DEPLOYMENT_NAME")
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let credential = Arc::new(AzureCliCredential::new(AI_FOUNDRY_SCOPE));
    let client = AzureAIAgentClient::new(&endpoint, &model, credential)
        .with_agent_name("rust-bing-grounding-example");

    // `hosted_web_search()` is the same builder used for OpenAI's and
    // Anthropic's hosted web search; `.connection_id(...)` is what tells the
    // Azure AI Foundry converter to build a `bing_grounding` tool instead of
    // erroring for lack of a connection.
    let bing_search = hosted_web_search().connection_id(connection_id);

    let agent = ChatAgent::builder(client.clone())
        .name("bing-grounded-assistant")
        .instructions("Use web search for anything time-sensitive; cite your sources briefly.")
        .tool(bing_search)
        .build();

    let response = agent
        .run_once("What's a notable recent headline about the Rust programming language?")
        .await?;
    println!("{}", response.text());

    // Delete the auto-created transient agent server-side.
    client.close().await?;
    Ok(())
}
