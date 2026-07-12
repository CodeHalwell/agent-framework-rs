//! AWS Bedrock's [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
//! via `BedrockChatClient` (`POST bedrock-runtime.{region}.amazonaws.com/model/{model}/converse`).
//!
//! Requests are signed with AWS Signature Version 4 from the standard AWS
//! credential environment variables, so no extra SDK is needed. Requires the
//! `bedrock` feature.
//!
//! Skips gracefully unless `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are
//! set (plus `AWS_SESSION_TOKEN` for temporary/STS credentials). The region is
//! read from `AWS_REGION` (or `AWS_DEFAULT_REGION`), defaulting to `us-east-1`.
//!
//! ```bash
//! AWS_ACCESS_KEY_ID=AKIA... AWS_SECRET_ACCESS_KEY=... AWS_REGION=us-east-1 \
//!   cargo run -p agent-framework-examples --example bedrock
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("AWS_ACCESS_KEY_ID").is_err()
        || std::env::var("AWS_SECRET_ACCESS_KEY").is_err()
    {
        println!("set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to run this example");
        return Ok(());
    }

    // Reads AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN and
    // the region from AWS_REGION (or AWS_DEFAULT_REGION, default us-east-1).
    let client = BedrockChatClient::from_env("anthropic.claude-3-5-sonnet-20241022-v2:0")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
