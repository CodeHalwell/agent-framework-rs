//! Observability: wrap a chat client in `ObservableChatClient` to emit
//! OpenTelemetry GenAI-semantic-convention `tracing` spans (`chat`,
//! `invoke_agent`, `execute_tool`), and print them via `tracing_subscriber`'s
//! fmt layer. In production, swap the fmt layer for `tracing-opentelemetry`
//! to export these same spans to an OTel collector.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example observability
//! ```

use agent_framework::prelude::*;
use tracing_subscriber::fmt::format::FmtSpan;

#[tokio::main]
async fn main() -> Result<()> {
    // Print a line when each span opens and closes, including the `gen_ai.*`
    // fields recorded on it (operation name, model, token usage, finish
    // reason, ...).
    tracing_subscriber::fmt()
        .with_span_events(FmtSpan::NEW | FmtSpan::CLOSE)
        .with_target(false)
        .init();

    // `ObservableChatClient` wraps any `ChatClient` and tags its spans with
    // the given "system" name (the `gen_ai.system` attribute), mirroring
    // Python's OpenTelemetry instrumentation. Content capture
    // (`gen_ai.input.messages` / `gen_ai.output.messages`) stays off unless
    // you opt in with `.with_content_capture(true)`.
    let client = ObservableChatClient::new(OpenAIClient::from_env("gpt-4o-mini")?, "openai");

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("You are concise.")
        .build();

    tracing::info!("running the agent once -- watch for the 'chat' and 'invoke_agent' spans below");
    let response = agent
        .run_once("What is the tallest mountain on Earth?")
        .await?;
    println!("\n{}", response.text());

    Ok(())
}
