//! `OpenAIClient` against any OpenAI-Chat-Completions-compatible server --
//! llama.cpp's server, Ollama, LM Studio, vLLM, together.ai, and similar --
//! by overriding the base URL with `.with_base_url(...)`.
//!
//! Env-gated on `OPENAI_BASE_URL` (the distinguishing setting for this
//! example -- without it, this is just `agents/quickstart.rs`).
//! `OPENAI_API_KEY` is read if set but is optional here: most self-hosted
//! servers don't check it, so a placeholder is used when it's absent. A
//! hosted OpenAI-compatible provider that *does* require a real key (e.g.
//! Together.ai, Groq) will still work by setting both variables.
//! `OPENAI_MODEL` optionally names the model/deployment (default
//! `gpt-4o-mini`, though a local server usually expects its own model name).
//!
//! ```bash
//! # e.g. a local llama.cpp server: `llama-server -m model.gguf --port 8080`
//! OPENAI_BASE_URL=http://localhost:8080/v1 OPENAI_MODEL=local-model \
//! cargo run -p agent-framework-examples --example openai_compatible_endpoint
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let Ok(base_url) = std::env::var("OPENAI_BASE_URL") else {
        println!(
            "set OPENAI_BASE_URL to run this example, e.g. \
             OPENAI_BASE_URL=http://localhost:8080/v1 for a local llama.cpp/Ollama/LM Studio server"
        );
        return Ok(());
    };
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or_else(|_| "not-needed".to_string());
    let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".to_string());

    println!("connecting to {base_url} (model \"{model}\")");
    let client = OpenAIClient::new(api_key, model).with_base_url(base_url);

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are a helpful, concise assistant.")
        .build();

    let response = agent.run_once("Say hello in one short sentence.").await?;
    println!("{}", response.text());

    Ok(())
}
