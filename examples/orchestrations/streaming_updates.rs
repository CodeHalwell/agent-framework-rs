//! Streaming updates from an orchestration: `run_stream` over a workflow
//! surfaces each agent's reply incrementally as
//! `WorkflowEvent::AgentRunUpdate` events -- one per streamed chunk --
//! interleaved with the workflow's own lifecycle events. This prints every
//! event as it comes off the stream.
//!
//! Runs fully offline against a scripted streaming `ChatClient` that splits
//! its canned reply into word-sized chunks -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example streaming_updates
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A chat client that streams its canned reply one word at a time instead of
/// returning it in a single chunk -- enough to demonstrate incremental
/// `AgentRunUpdate`s without a real LLM.
struct WordStreamingClient(&'static str);

#[async_trait]
impl ChatClient for WordStreamingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.0))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        let chunks: Vec<Result<ChatResponseUpdate>> = self
            .0
            .split(' ')
            .enumerate()
            .map(|(i, word)| {
                let text = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                Ok(ChatResponseUpdate::text(text))
            })
            .collect();
        Ok(Box::pin(futures::stream::iter(chunks)))
    }
}

fn streaming_agent(name: &str, reply: &'static str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(
        Agent::builder(WordStreamingClient(reply))
            .name(name)
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

#[tokio::main]
async fn main() -> Result<()> {
    let drafter = streaming_agent("drafter", "Rust workflows stream tokens incrementally.");
    let critic = streaming_agent("critic", "Concise and accurate, approved.");

    let workflow = SequentialBuilder::new()
        .participants(vec![drafter, critic])
        .name("streamed-review")
        .build()?;

    // Per executor, the runner applies buffered effects in a fixed order --
    // any workflow `Output` before the agent-run `Custom` events (Started,
    // ExecutorInvoked, AgentRunUpdate..., AgentRun) it buffered along the
    // way. So for the *last* participant (the one whose `AgentExecutor`
    // yields the workflow's output) the "workflow output" line below prints
    // before its own update chunks and "run complete" line, even though the
    // agent produced them first. Non-final participants only send a message
    // (no `Output`), so their update chunks appear in the order produced.
    let mut stream = workflow.run_stream("Summarize how streaming works in this engine.");
    let mut event_count = 0usize;
    let mut update_count = 0usize;
    while let Some(event) = stream.next().await {
        event_count += 1;
        match &event {
            WorkflowEvent::AgentRunUpdate {
                executor_id,
                update,
            } => {
                update_count += 1;
                let u: AgentResponseUpdate =
                    serde_json::from_value(update.clone()).unwrap_or_default();
                println!(
                    "{event_count:>3}  [{executor_id}] update chunk: {:?}",
                    u.text()
                );
            }
            WorkflowEvent::AgentRun {
                executor_id,
                response,
            } => {
                let r: AgentResponse = serde_json::from_value(response.clone()).unwrap_or_default();
                println!(
                    "{event_count:>3}  [{executor_id}] run complete: {:?}",
                    r.text()
                );
            }
            WorkflowEvent::Output {
                source_executor_id,
                data,
            } => {
                println!("{event_count:>3}  [{source_executor_id}] workflow output: {data}");
            }
            other => println!("{event_count:>3}  {other:?}"),
        }
    }

    // `into_run` drains any remaining buffered events (none left here, since
    // the loop above already consumed the stream) and returns the final run.
    let run = stream.into_run().await?;
    println!("\n{event_count} events observed, {update_count} of them incremental agent updates");
    println!("final state: {:?}", run.state());
    assert_eq!(run.state(), WorkflowRunState::Idle);

    Ok(())
}
