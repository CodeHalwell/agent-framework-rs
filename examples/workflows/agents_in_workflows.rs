//! agents_in_workflows: wrap `SupportsAgentRun`s as graph nodes via `AgentExecutor`, and
//! mix them with a plain `FunctionExecutor` in the same graph -- something
//! the `SequentialBuilder`/`ConcurrentBuilder` sugar doesn't expose. Each
//! `AgentExecutor` appends its agent's reply to the running conversation and
//! forwards the whole conversation downstream.
//!
//! Runs fully offline against a scripted `ChatClient` -- swap `CannedClient`
//! for a real provider client (e.g. `OpenAIChatCompletionClient::from_env(...)`) and the
//! graph is unchanged.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example agents_in_workflows
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{AgentExecutor, FunctionExecutor};
use async_trait::async_trait;
use serde_json::json;

/// A chat client that always replies with the same scripted line -- stands in
/// for a real LLM backend so the example needs no API key or network access.
struct CannedClient(&'static str);

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.0))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        // AgentExecutor always drives its agent via `run_stream`, so the
        // canned reply still needs to flow through here (as a single chunk).
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

fn canned_agent(name: &str, reply: &'static str) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(CannedClient(reply)).name(name).build()) as Arc<dyn SupportsAgentRun>
}

#[tokio::main]
async fn main() -> Result<()> {
    let drafter = AgentExecutor::new(
        "drafter",
        canned_agent("drafter", "Rust: fast, safe, fearless concurrency."),
    );
    let reviewer = AgentExecutor::new(
        "reviewer",
        canned_agent("reviewer", "Approved -- concise and on-brand."),
    );

    // A plain function executor sits downstream of the agents in the same
    // graph, pulling out just the final reply for the workflow's output.
    let extract_final = FunctionExecutor::new("extract_final", |message, ctx| async move {
        let conversation: Vec<Message> = serde_json::from_value(message)
            .map_err(|e| Error::Workflow(format!("bad conversation: {e}")))?;
        let last = conversation.last().map(Message::text).unwrap_or_default();
        ctx.yield_output(json!({ "final_reply": last, "turns": conversation.len() }))
            .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(drafter))
        .add_executor(Arc::new(reviewer))
        .add_executor(Arc::new(extract_final))
        .set_start("drafter")
        .add_chain(vec![
            "drafter".to_string(),
            "reviewer".to_string(),
            "extract_final".to_string(),
        ])
        .build()?;

    let run = workflow.run("Write a one-line tagline for Rust.").await?;
    let output = run.last_output().expect("extract_final always yields");
    println!("final reply: {}", output["final_reply"].as_str().unwrap());
    println!("conversation turns: {}", output["turns"]);
    assert_eq!(
        output["turns"],
        json!(3),
        "prompt + drafter reply + reviewer reply"
    );

    Ok(())
}
