//! Azure AI Foundry text embeddings, over either of its two surfaces.
//!
//! A Foundry **project** endpoint -- the same one `FoundryChatClient` takes --
//! now works directly: the client derives the resource-scoped
//! `{resource}/openai/v1/embeddings` route from it, path-versioned and Entra
//! ID only. Before that derivation existed, embeddings needed a separately
//! provisioned Foundry **Models** inference endpoint, which is still
//! supported and still wins when both are configured.
//!
//! `from_env` picks the surface: `FOUNDRY_MODELS_ENDPOINT` if set, otherwise
//! `FOUNDRY_PROJECT_ENDPOINT` (or `FOUNDRY_ENDPOINT`). Skips gracefully
//! unless one of them is set. Requires FOUNDRY_EMBEDDING_MODEL.
//!
//! ```bash
//! az login && FOUNDRY_PROJECT_ENDPOINT=https://<res>.services.ai.azure.com/api/projects/<proj> \
//! FOUNDRY_EMBEDDING_MODEL=text-embedding-3-small \
//! cargo run -p agent-framework-examples --example foundry_embeddings
//! ```

use std::sync::Arc;

use agent_framework::azure::DefaultAzureCredential;
use agent_framework::foundry::embeddings::{
    openai_model_base_url, FoundryEmbeddingClient, DEFAULT_SCOPE,
};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // The derivation itself needs no credentials or network, so show it first.
    let project = "https://my-res.services.ai.azure.com/api/projects/my-proj";
    println!("project endpoint : {project}");
    println!("model route      : {}", openai_model_base_url(project)?);

    if std::env::var("FOUNDRY_MODELS_ENDPOINT").is_err()
        && std::env::var("FOUNDRY_PROJECT_ENDPOINT").is_err()
        && std::env::var("FOUNDRY_ENDPOINT").is_err()
    {
        println!("\nSet FOUNDRY_PROJECT_ENDPOINT (or FOUNDRY_MODELS_ENDPOINT) to run the rest.");
        return Ok(());
    }

    let client = FoundryEmbeddingClient::from_env(None)?;
    let inputs = ["The cat sat on the mat.", "Quarterly revenue grew by 12%."];
    let batch = client
        .get_embeddings(
            inputs.into_iter().map(Into::into).collect(),
            Some(EmbeddingGenerationOptions::new().with_dimensions(256)),
        )
        .await?;
    for (text, embedding) in inputs.iter().zip(&batch) {
        println!("{} dims  <-  {text}", embedding.dimensions());
    }

    // Constructing the project route explicitly, rather than through the
    // environment. The token audience is the Azure OpenAI data plane, because
    // the derived host is an Azure OpenAI one -- not the Foundry project
    // audience (`FOUNDRY_SCOPE`) that the Responses API needs.
    let _explicit = FoundryEmbeddingClient::with_project_endpoint(
        project,
        "text-embedding-3-small",
        Arc::new(DefaultAzureCredential::new(DEFAULT_SCOPE)),
    )?;

    Ok(())
}
