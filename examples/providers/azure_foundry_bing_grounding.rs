//! Hosted web search on Azure AI Foundry's Responses API:
//! `hosted_web_search()` maps to the Responses API's generic, keyless
//! `{"type": "web_search"}` tool (`agent_framework_openai::responses::tool_to_responses_spec`
//! — the same conversion `FoundryChatClient` delegates through).
//!
//! This differs from the old Azure AI Agents threads/runs data plane (now
//! removed upstream, and rewritten in this crate onto the Responses API):
//! that older surface required a Bing Grounding **connection id**
//! (`hosted_web_search().connection_id(...)`) to build a `bing_grounding`
//! tool. The Responses API's hosted web search has no such connection-id
//! parameter -- `.connection_id(...)`/`.custom_connection(...)` are simply
//! ignored on this path. If/when Foundry's Responses API grows a
//! Bing-specific web-search variant, wire it into
//! `agent_framework_openai::responses::tool_to_responses_spec` (or a
//! Foundry-specific override) rather than here.
//!
//! Entra ID auth comes from the `azure` crate's credential chain -- here the
//! Azure CLI credential (`az login`); `DefaultAzureCredential`,
//! `ClientSecretCredential`, `ManagedIdentityCredential`, or a
//! `ChainedTokenCredential` all work the same way (see
//! `providers/azure_default_credential.rs`).
//!
//! Skips gracefully unless FOUNDRY_ENDPOINT is set (e.g.
//! `https://<resource>.services.ai.azure.com/api/projects/<project>`).
//! Optional: FOUNDRY_MODEL (default `gpt-4o-mini`).
//!
//! ```bash
//! az login && FOUNDRY_ENDPOINT=https://... \
//! cargo run -p agent-framework-examples --example azure_foundry_bing_grounding
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

    let credential = Arc::new(AzureCliCredential::new(FOUNDRY_SCOPE));
    let client = FoundryChatClient::with_token_credential(&endpoint, &model, credential);

    let agent = FoundryAgent::builder(client)
        .name("web-search-assistant")
        .instructions("Use web search for anything time-sensitive; cite your sources briefly.")
        .tool(hosted_web_search())
        .build();

    let response = agent
        .run(
            vec![Message::user(
                "What's a notable recent headline about the Rust programming language?",
            )],
            None,
        )
        .await?;
    println!("{}", response.text());

    Ok(())
}
