//! workflow_as_agent: expose a built `Workflow` as an `Agent` via
//! `WorkflowAgentExt::as_agent`, then drive it exactly like any other agent
//! -- `run` with a caller-supplied `AgentThread` that accumulates history
//! across calls.
//!
//! Runs fully offline against scripted `ChatClient`s -- no API key or
//! network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example workflow_as_agent
//! ```

use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;

/// A chat client that returns each scripted reply in turn, then repeats its
/// last one -- enough determinism for a multi-turn conversation without a
/// real LLM.
#[derive(Clone)]
struct ScriptedClient {
    replies: Arc<Mutex<Vec<ChatResponse>>>,
}

impl ScriptedClient {
    fn new(replies: Vec<&str>) -> Self {
        Self {
            replies: Arc::new(Mutex::new(
                replies.into_iter().map(ChatResponse::from_text).collect(),
            )),
        }
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut queue = self.replies.lock().unwrap();
        if queue.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted replies)"))
        } else {
            Ok(queue.remove(0))
        }
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = resp
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(Box::pin(futures::stream::iter(updates)))
    }
}

fn scripted_agent(name: &str, replies: Vec<&str>) -> Arc<dyn Agent> {
    Arc::new(
        ChatAgent::builder(ScriptedClient::new(replies))
            .name(name)
            .build(),
    ) as Arc<dyn Agent>
}

#[tokio::main]
async fn main() -> Result<()> {
    let drafter = scripted_agent(
        "drafter",
        vec![
            "Draft: workflows compose small executors into a graph.",
            "Draft: sub-workflows nest a whole graph as one node.",
        ],
    );
    let editor = scripted_agent(
        "editor",
        vec![
            "Edit: workflows compose small executors into a directed graph.",
            "Edit: sub-workflows nest an entire graph inside a single node.",
        ],
    );

    let workflow = SequentialBuilder::new()
        .participants(vec![drafter, editor])
        .name("writer-pipeline")
        .build()?;

    // Exposed as an Agent -- callers never need to know it's backed by a
    // workflow under the hood.
    let agent = workflow.as_agent("writer_pipeline");
    println!("agent name: {:?}", agent.name());

    let mut thread = agent.get_new_thread();
    for topic in [
        "Explain workflows in one sentence.",
        "Now do the same for sub-workflows.",
    ] {
        let response = agent
            .run(vec![ChatMessage::user(topic)], Some(&mut thread))
            .await?;
        println!("\n> {topic}");
        for msg in &response.messages {
            println!(
                "  {}: {}",
                msg.author_name.as_deref().unwrap_or("?"),
                msg.text()
            );
        }
    }

    // Both turns' input and response messages accumulated on the same thread.
    // Per turn, `WorkflowAgent::run` writes back the raw input (1 message)
    // plus the sequential workflow's yielded output -- the full running
    // conversation (user + drafter + editor, 3 messages) -- so each turn
    // adds 4 entries: 2 turns x 4 = 8.
    let history = thread.list_messages().await?;
    println!(
        "\nthread now holds {} messages across both turns",
        history.len()
    );
    assert_eq!(history.len(), 8);

    Ok(())
}
