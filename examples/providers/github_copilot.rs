//! GitHub Copilot's OpenAI-compatible chat endpoint via
//! `GitHubCopilotChatClient` (`POST https://api.githubcopilot.com/chat/completions`).
//!
//! The client exchanges a GitHub OAuth/PAT token for a short-lived Copilot
//! bearer token (`GET https://api.github.com/copilot_internal/v2/token`),
//! caches it, and refreshes it near expiry. Requires the `github-copilot`
//! feature.
//!
//! Skips gracefully unless `GITHUB_COPILOT_TOKEN` (or `GH_COPILOT_TOKEN`) is
//! set; `GITHUB_COPILOT_BASE_URL` overrides the default endpoint.
//!
//! ```bash
//! GITHUB_COPILOT_TOKEN=gho_... cargo run -p agent-framework-examples --example github_copilot
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("GITHUB_COPILOT_TOKEN").is_err() && std::env::var("GH_COPILOT_TOKEN").is_err()
    {
        println!("set GITHUB_COPILOT_TOKEN (or GH_COPILOT_TOKEN) to run this example");
        return Ok(());
    }

    // Reads GITHUB_COPILOT_TOKEN (or GH_COPILOT_TOKEN); the token exchange to a
    // Copilot bearer happens lazily on the first request.
    let client = GitHubCopilotChatClient::from_env("gpt-4o")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
