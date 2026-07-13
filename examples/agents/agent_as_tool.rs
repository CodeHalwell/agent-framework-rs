//! Compose agents: a specialist agent is exposed as a callable tool to an
//! orchestrator agent via `Agent::as_tool`.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example agent_as_tool
//! ```

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;

    // The specialist: answers geography questions. `as_tool` runs it
    // statelessly (a fresh thread per call) and returns its reply text.
    let geographer = Agent::builder(client.clone())
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

    // A second specialist shares the orchestrator's session instead:
    // `propagate_session` hands it a *child* of the caller's session — same
    // identity and (shared) state bag, but an isolated server-side
    // conversation pointer.
    let historian = Agent::builder(client.clone())
        .name("historian")
        .description("Answers questions about world history.")
        .instructions("Answer history questions in one short sentence.")
        .build();
    let history_tool = historian.as_tool(
        AsToolOptions::new()
            .name("ask_historian")
            .description("Ask the history specialist a question.")
            .propagate_session(true),
    );

    // The orchestrator can now call the specialists exactly like any other
    // tool -- the function-invocation loop handles the call/result round trip.
    let orchestrator = Agent::builder(client)
        .name("orchestrator")
        .instructions(
            "You answer user questions. For geography questions, delegate to \
             the ask_geographer tool; for history questions, delegate to the \
             ask_historian tool.",
        )
        .tool(geography_tool)
        .tool(history_tool)
        .build();

    let mut session = AgentSession::new();
    let response = orchestrator
        .run(
            vec![Message::user("What is the highest mountain in Africa?")],
            Some(&mut session),
        )
        .await?;
    println!("{}", response.text());

    Ok(())
}
