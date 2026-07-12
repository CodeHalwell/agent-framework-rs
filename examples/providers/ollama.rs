//! [Ollama](https://ollama.com)'s OpenAI-compatible chat endpoint
//! (`POST {base_url}/chat/completions`, default
//! `http://localhost:11434/v1`) via `OllamaChatClient`.
//!
//! Unlike the other provider examples, Ollama needs no API key -- a stock
//! local install answers unauthenticated requests. This example instead
//! skips gracefully unless `OLLAMA_HOST` is set (e.g. to `127.0.0.1:11434`),
//! so it doesn't silently try (and fail) to reach a server that may not be
//! running.
//!
//! ```bash
//! OLLAMA_HOST=127.0.0.1:11434 cargo run -p agent-framework-examples --example ollama
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(_) = std::env::var("OLLAMA_HOST") else {
        println!("set OLLAMA_HOST (e.g. 127.0.0.1:11434) to run this example, with an Ollama server reachable there");
        return Ok(());
    };

    // Reads OLLAMA_HOST to derive the OpenAI-compatible base URL.
    let client = OllamaChatClient::from_env("llama3.1")?;

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("What is the capital of Japan?").await?;
    println!("{}", response.text());

    Ok(())
}
