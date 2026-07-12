//! Magentic orchestration: a `StandardMagenticManager` plans a task, assigns
//! work to participants round by round via a progress ledger, and drafts a
//! final answer once the ledger says the request is satisfied.
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example magentic
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{MagenticBuilder, StandardMagenticManager};
use agent_framework_core::types::Message;

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;

    let researcher = Arc::new(
        Agent::builder(client.clone())
            .name("researcher")
            .instructions("You find and summarize relevant facts.")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let writer = Arc::new(
        Agent::builder(client.clone())
            .name("writer")
            .instructions("You turn facts into a polished, short answer.")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    // The manager is itself an LLM agent: it plans the task, picks the next
    // speaker each round, and eventually prepares the final answer. It needs
    // no special instructions -- `StandardMagenticManager` supplies its own
    // ported-from-Python planning/progress-ledger prompts.
    let manager_agent = Arc::new(Agent::builder(client).name("magentic_manager").build())
        as Arc<dyn SupportsAgentRun>;
    let manager = StandardMagenticManager::new(manager_agent)
        .max_round_count(10)
        .max_stall_count(3);

    let workflow = MagenticBuilder::new()
        .participant_described("researcher", "Finds and summarizes facts.", researcher)
        .participant_described("writer", "Writes polished final answers.", writer)
        .standard_manager(manager)
        .build()?;

    let run = workflow
        .run("What year was the Eiffel Tower completed, and why was it built?")
        .await?;

    let conversation: Vec<Message> =
        serde_json::from_value(run.last_output().unwrap_or_default()).unwrap_or_default();
    for msg in &conversation {
        let speaker = msg.author_name.as_deref().unwrap_or(msg.role.as_str());
        println!("{speaker}: {}", msg.text());
    }

    Ok(())
}
