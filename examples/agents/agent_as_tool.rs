//! Compose agents: a specialist agent is exposed as a callable tool to an
//! orchestrator agent via `ChatAgent::as_tool`.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example agent_as_tool
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIClient::from_env("gpt-4o-mini")?;

    // The specialist: answers geography questions. `as_tool` runs it
    // statelessly (a fresh thread per call) and returns its reply text.
    let geographer = ChatAgent::builder(client.clone())
        .name("geographer")
        .description("Answers questions about world geography.")
        .instructions("Answer geography questions in one short sentence.")
        .build();

    // Wrap it as a tool with a single string "task" argument (the default;
    // `AsToolOptions::arg_name` can override it).
    let geography_tool = geographer.as_tool(
        AsToolOptions::new()
            .name("ask_geographer")
            .description("Ask the geography specialist a question."),
    );

    // The orchestrator can now call the specialist exactly like any other
    // tool -- the function-invocation loop handles the call/result round trip.
    let orchestrator = ChatAgent::builder(client)
        .name("orchestrator")
        .instructions(
            "You answer user questions. For geography questions, delegate to \
             the ask_geographer tool instead of answering directly.",
        )
        .tool(geography_tool)
        .build();

    let response = orchestrator
        .run_once("What is the highest mountain in Africa?")
        .await?;
    println!("{}", response.text());

    Ok(())
}
