//! `AgentRunOptions`: per-run `ChatOptions` overrides are merged over the
//! agent's build-time defaults (temperature is replaced outright,
//! instructions are newline-concatenated -- see `ChatOptions::merge`), and
//! `additional_tools` are appended to the tool list for that call only.
//!
//! Runs fully offline against a canned client that echoes back exactly what
//! it received, so the merge is visible in the printed output -- no API key
//! or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example per_run_options
//! ```

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// A client that reports the system instructions, temperature, and tool
/// names it was actually called with, so the merge is observable. Note that
/// by the time a request reaches the client, `ChatOptions::instructions` has
/// already been consumed and turned into a leading system `Message` (see
/// `ChatAgent::prepare_request`) -- so we read it off `messages`, not
/// `options`.
#[derive(Clone)]
struct ReportingClient;

#[async_trait]
impl ChatClient for ReportingClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let system = messages
            .iter()
            .find(|m| m.role == Role::system())
            .map(Message::text);
        let tool_names: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        Ok(ChatResponse::from_text(format!(
            "[client received] system={system:?} temperature={:?} tools={tool_names:?}",
            options.temperature
        )))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // A tool available only for the second run below.
    let scratch_note = FunctionTool::new(
        "scratch_note",
        "Jot down a scratch note (per-run-only tool).",
        json!({ "type": "object", "properties": {} }),
        |_args| async move { Ok(json!("noted")) },
    )
    .into_definition();

    let agent = ChatAgent::builder(ReportingClient)
        .name("assistant")
        .instructions("Be terse.")
        .temperature(0.2)
        .build();

    println!("-- run #1: agent defaults only --");
    let r1 = agent.run_once("Hi").await?;
    println!("{}\n", r1.text());

    println!(
        "-- run #2: per-run overrides (temperature + instructions) plus a per-run-only tool --"
    );
    let overrides = AgentRunOptions::new()
        .with_chat_options(
            ChatOptions::new()
                .with_temperature(0.9)
                .with_instructions("Now be verbose and enthusiastic."),
        )
        .with_tool(scratch_note);
    let r2 = agent
        .run_with_options(vec![Message::user("Hi again")], None, overrides)
        .await?;
    println!("{}", r2.text());

    println!(
        "\nnote: temperature was replaced outright (0.2 -> 0.9); instructions were \
         newline-concatenated (agent default first, per-run override second); the \
         `scratch_note` tool was only visible on run #2."
    );

    Ok(())
}
