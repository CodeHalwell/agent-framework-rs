//! Google Gemini's `generateContent` REST API via `GeminiChatClient`.
//!
//! Requires the `gemini` feature.
//!
//! Skips gracefully unless `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) is set.
//!
//! ```bash
//! GEMINI_API_KEY=AIza... cargo run -p agent-framework-examples --example gemini
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("GEMINI_API_KEY").is_err() && std::env::var("GOOGLE_API_KEY").is_err() {
        println!("set GEMINI_API_KEY (or GOOGLE_API_KEY) to run this example");
        return Ok(());
    }

    // Reads GEMINI_API_KEY, falling back to GOOGLE_API_KEY.
    let client = GeminiChatClient::from_env("gemini-2.5-flash")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
