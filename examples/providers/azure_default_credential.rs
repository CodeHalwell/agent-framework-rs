//! `DefaultAzureCredential`: a prebuilt Microsoft Entra ID credential chain,
//! tried in order until one succeeds (the first to work is then remembered
//! and tried first on subsequent calls):
//!
//!   1. `EnvironmentCredential`   -- `AZURE_TENANT_ID` / `AZURE_CLIENT_ID` /
//!      `AZURE_CLIENT_SECRET` (a service principal's client-secret flow).
//!      Included in the chain only when all three are set.
//!   2. `WorkloadIdentityCredential` -- Kubernetes/AKS workload identity
//!      federation (`AZURE_TENANT_ID` / `AZURE_CLIENT_ID` /
//!      `AZURE_FEDERATED_TOKEN_FILE`). Included only when all three are set.
//!   3. `ManagedIdentityCredential` -- the Azure Instance Metadata Service
//!      (IMDS) token endpoint; works unconfigured on an Azure VM/App
//!      Service/Container App with a managed identity assigned. Always
//!      present in the chain (fails at token-fetch time off-Azure).
//!   4. `AzureCliCredential`     -- shells out to `az account get-access-token`;
//!      useful for local development after `az login`. Always present.
//!
//! This mirrors (a subset of) `azure_identity`'s own `DefaultAzureCredential`
//! order; see `agent_framework_azure::credentials` for the full rationale and
//! which upstream links (shared token cache, IDE-signed-in credentials, ...)
//! are intentionally not implemented here.
//!
//! Skips gracefully unless `AZURE_OPENAI_ENDPOINT` is set. Optional:
//! `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` (default `gpt-4o-mini`).
//!
//! ```bash
//! az login && AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com \
//! cargo run -p agent-framework-examples --example azure_default_credential
//! ```

use std::sync::Arc;

use agent_framework::azure::{AzureOpenAIClient, DefaultAzureCredential};
use agent_framework::prelude::*;

/// The Entra ID scope Azure OpenAI (a Cognitive Services resource) expects.
const COGNITIVE_SERVICES_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(endpoint) = std::env::var("AZURE_OPENAI_ENDPOINT") else {
        println!("set AZURE_OPENAI_ENDPOINT (and probably run `az login`) to run this example");
        return Ok(());
    };
    let deployment = std::env::var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let credential = Arc::new(DefaultAzureCredential::new(COGNITIVE_SERVICES_SCOPE));
    let client = AzureOpenAIClient::with_token_credential(endpoint, deployment, credential);

    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("Say hello in one short sentence.").await?;
    println!("{}", response.text());

    Ok(())
}
