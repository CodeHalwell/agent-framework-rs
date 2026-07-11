//! Azure OpenAI, with both authentication modes: a static API key and
//! Microsoft Entra ID (bearer token) via `TokenCredential`.
//!
//! Requires the `azure` feature:
//! ```bash
//! AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com \
//! AZURE_OPENAI_API_KEY=... \
//! AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=my-gpt4o-deployment \
//! cargo run -p agent-framework-examples --example azure_openai
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    // AZURE_OPENAI_CHAT_DEPLOYMENT_NAME, and optional AZURE_OPENAI_API_VERSION.
    let client = AzureOpenAIClient::from_env()?;

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("Say hello in one short sentence.").await?;
    println!("{}", response.text());

    // Entra ID / bearer-token authentication instead of a static API key:
    // fetch a token however your environment does (Azure CLI, managed
    // identity, `azure_identity`'s TokenCredential, ...) and wrap it in a
    // `TokenCredential` implementation. `StaticTokenCredential` is a minimal
    // one for a pre-fetched token; `from_env` above only covers api-key auth.
    let _entra_client = AzureOpenAIClient::with_token_credential(
        "https://my-resource.openai.azure.com",
        "my-gpt4o-deployment",
        Arc::new(StaticTokenCredential::new("<a pre-fetched bearer token>")),
    );

    Ok(())
}
