//! Handoff orchestration: a triage agent routes the conversation to one of
//! several specialists via a synthetic "handoff" tool call.
//!
//! This example uses the autonomous interaction mode, which runs straight
//! through to a final answer -- the simplest, always-completing variant.
//! `HandoffInteractionMode::HumanInLoop` (the default) instead pauses after
//! every non-handoff reply so a human can supply the next message; see the
//! comment near the bottom for how that loop is driven.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example handoff
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{handoff_tool_spec, HandoffBuilder};
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;

    let billing = Arc::new(
        Agent::builder(client.clone())
            .name("billing")
            .instructions("You resolve billing and invoice questions.")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let tech_support = Arc::new(
        Agent::builder(client.clone())
            .name("tech_support")
            .instructions("You troubleshoot product and technical issues.")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    // The triage agent gets a `handoff_to_<target>` tool declared for each
    // specialist it can transfer to, so the model knows the option exists.
    // The coordinator intercepts the call rather than executing it as a real
    // tool -- see `handoff_tool_spec`'s docs for the exact mechanism.
    let triage = Arc::new(
        Agent::builder(client)
            .name("triage")
            .instructions(
                "You are the first point of contact. Hand off to 'billing' for payment \
                 questions, or 'tech_support' for technical issues.",
            )
            .tool(handoff_tool_spec(
                "billing",
                Some("Transfer to the billing specialist."),
            ))
            .tool(handoff_tool_spec(
                "tech_support",
                Some("Transfer to the technical support specialist."),
            ))
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = HandoffBuilder::new()
        .participant("triage", triage)
        .participant("billing", billing)
        .participant("tech_support", tech_support)
        .initial_agent("triage")
        .add_handoff("triage")
        .to(["billing", "tech_support"])
        .autonomous()
        .build()?;

    let run = workflow
        .run("I was charged twice for my subscription this month.")
        .await?;

    let conversation: Vec<Message> =
        serde_json::from_value(run.last_output().unwrap_or_default()).unwrap_or_default();
    for msg in &conversation {
        let speaker = msg.author_name.as_deref().unwrap_or(msg.role.as_str());
        println!("{speaker}: {}", msg.text());
    }

    // Interactive mode (the default, or explicitly `.with_user_input_request()`)
    // instead pauses with `run.state() == WorkflowRunState::IdleWithPendingRequests`
    // after each non-handoff reply, surfacing a `HandoffUserInputRequest` you
    // can drive from stdin, a chat UI, etc.:
    //
    //   let mut run = workflow.run(first_message).await?;
    //   while run.state() == WorkflowRunState::IdleWithPendingRequests {
    //       let request_id = run.pending_requests()[0].request_id.clone();
    //       let reply = read_next_message_from_a_human();
    //       run.send_response(request_id, serde_json::json!(reply)).await?;
    //   }

    Ok(())
}
