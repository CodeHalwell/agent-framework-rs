//! Group chat orchestration: several agents collaborate in one running
//! conversation, coordinated either by round-robin turn-taking or by a
//! dedicated LLM "manager" agent that decides who speaks next (and when to
//! stop).
//!
//! ```bash
//! OPENAI_API_KEY=sk-... cargo run -p agent-framework-examples --example group_chat
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::GroupChatBuilder;
use agent_framework_core::types::Message;

fn participant(
    client: &OpenAIChatCompletionClient,
    name: &str,
    instructions: &str,
) -> Arc<dyn SupportsAgentRun> {
    Arc::new(
        Agent::builder(client.clone())
            .name(name)
            .instructions(instructions)
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = OpenAIChatCompletionClient::from_env("gpt-4o-mini")?;
    let task = "Write a one-line tagline for a coffee shop called Terra.";

    let writer = participant(&client, "writer", "You draft short, punchy marketing copy.");
    let critic = participant(
        &client,
        "critic",
        "You critique copy for clarity and accuracy in one or two sentences.",
    );

    // --- Variant 1: round-robin (the default manager) -----------------
    let round_robin = GroupChatBuilder::new()
        .participant("writer", writer.clone())
        .participant("critic", critic.clone())
        .round_robin()
        .max_rounds(4)
        .build()?;
    let run = round_robin.run(task).await?;
    print_transcript("round-robin", run.last_output());

    // --- Variant 2: an LLM manager decides who speaks next and when to
    //     finish, based on each participant's description ---------------
    let manager = participant(
        &client,
        "manager",
        "You coordinate a writer and a critic to produce a final tagline.",
    );
    let llm_managed = GroupChatBuilder::new()
        .participant_described("writer", "Drafts marketing copy.", writer)
        .participant_described("critic", "Critiques copy for clarity.", critic)
        .manager_agent(manager)
        .max_rounds(6)
        .build()?;
    let run = llm_managed.run(task).await?;
    print_transcript("LLM-managed", run.last_output());

    Ok(())
}

/// Print each message in the resulting conversation, attributed by speaker.
fn print_transcript(label: &str, output: Option<serde_json::Value>) {
    println!("\n--- {label} ---");
    let Some(output) = output else {
        println!("(no output)");
        return;
    };
    let conversation: Vec<Message> = serde_json::from_value(output).unwrap_or_default();
    for msg in &conversation {
        let speaker = msg.author_name.as_deref().unwrap_or(msg.role.as_str());
        println!("{speaker}: {}", msg.text());
    }
}
