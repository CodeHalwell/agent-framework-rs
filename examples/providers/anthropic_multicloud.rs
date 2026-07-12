//! Running Anthropic's Claude models through cloud transports other than the
//! direct Anthropic API, all in the `anthropic` crate:
//!
//! - `AnthropicBedrockClient` — Claude on AWS Bedrock (SigV4-signed
//!   `InvokeModel`, `anthropic_version: bedrock-2023-05-31`).
//! - `AnthropicVertexClient` — Claude on Google Vertex AI (`:rawPredict`,
//!   `anthropic_version: vertex-2023-10-16`), with a caller-supplied OAuth
//!   access token via a `VertexTokenProvider`.
//!
//! (`AnthropicFoundryClient` for Claude on Azure AI Foundry is analogous.)
//! Requires the `anthropic` feature.
//!
//! Each transport skips gracefully unless its credentials are present:
//!
//! ```bash
//! # Bedrock: standard AWS credential env vars
//! AWS_ACCESS_KEY_ID=AKIA... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
//!   cargo run -p agent-framework-examples --example anthropic_multicloud
//!
//! # Vertex: a GCP project + an access token from `gcloud auth print-access-token`
//! GOOGLE_CLOUD_PROJECT=my-proj GOOGLE_ACCESS_TOKEN=ya29.... \
//!   cargo run -p agent-framework-examples --example anthropic_multicloud
//! ```

use std::sync::Arc;

use agent_framework::anthropic::StaticVertexToken;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let question = "What is the capital of Japan?";

    // --- Claude on AWS Bedrock -------------------------------------------
    if std::env::var("AWS_ACCESS_KEY_ID").is_ok() && std::env::var("AWS_SECRET_ACCESS_KEY").is_ok()
    {
        // Reads the standard AWS credential env vars; the model id is in the
        // request URL, not the body.
        let client = AnthropicBedrockClient::from_env("anthropic.claude-3-5-sonnet-20241022-v2:0")?;
        let agent = Agent::builder(client)
            .instructions("You are a helpful, concise assistant.")
            .build();
        println!("[bedrock] {}", agent.run_once(question).await?.text());
    } else {
        println!("[bedrock] set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY to run this transport");
    }

    // --- Claude on Google Vertex AI --------------------------------------
    match (
        std::env::var("GOOGLE_CLOUD_PROJECT"),
        std::env::var("GOOGLE_ACCESS_TOKEN"),
    ) {
        (Ok(_), Ok(token)) => {
            // The caller supplies the Google OAuth token (e.g. from
            // `gcloud auth print-access-token`); project/region come from env.
            let token_provider = Arc::new(StaticVertexToken::new(token));
            let client =
                AnthropicVertexClient::from_env("claude-3-5-sonnet-v2@20241022", token_provider)?;
            let agent = Agent::builder(client)
                .instructions("You are a helpful, concise assistant.")
                .build();
            println!("[vertex] {}", agent.run_once(question).await?.text());
        }
        _ => println!(
            "[vertex] set GOOGLE_CLOUD_PROJECT and GOOGLE_ACCESS_TOKEN to run this transport"
        ),
    }

    Ok(())
}
