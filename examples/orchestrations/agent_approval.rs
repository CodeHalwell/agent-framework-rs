//! Post-agent human-in-the-loop approval: `SequentialBuilder`'s
//! `.with_request_info()` inserts an `AgentApprovalExecutor` after each
//! participant, pausing the workflow *after* the agent responds so a human
//! can inspect its reply (surfaced as an `ApprovalRequest` through the
//! engine's generic `pending_requests()` / `send_response()` machinery). An
//! **empty** response approves the reply and the workflow moves on; a
//! **non-empty** response is revision feedback -- it is folded into the
//! conversation, the same agent runs again, and a fresh approval request is
//! raised (iterate-until-approved). `ConcurrentBuilder` supports the same
//! flag, pausing each participant individually.
//!
//! Runs fully offline against a scripted client -- no API key or network
//! needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example agent_approval
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::ApprovalRequest;
use async_trait::async_trait;
use serde_json::json;

/// Returns successive scripted replies (the "writer improving its draft"),
/// standing in for a real model.
struct ScriptedClient {
    replies: Vec<&'static str>,
    calls: AtomicUsize,
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        let text = self.replies.get(i).or(self.replies.last()).copied();
        Ok(ChatResponse::from_text(text.unwrap_or_default()))
    }

    // Orchestrations drive their agents through the streaming path, so the
    // scripted reply must be replayed as stream updates too.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
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

#[tokio::main]
async fn main() -> Result<()> {
    let writer = Arc::new(
        Agent::builder(ScriptedClient {
            replies: vec![
                "Announcing v2.0, with sweeping changes across the entire product surface!",
                "v2.0 is out: faster startup, a smaller install, and two breaking API changes.",
            ],
            calls: AtomicUsize::new(0),
        })
        .name("writer")
        .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = SequentialBuilder::new()
        .participants(vec![writer])
        .with_request_info() // pause for approval after each agent's reply
        .build()?;

    let mut run = workflow
        .run("Draft a one-line release announcement.")
        .await?;

    // The agent already ran; its reply is paused, awaiting review.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    let request: ApprovalRequest = serde_json::from_value(pending[0].request_data.clone())?;
    println!("-- draft 1, awaiting approval --");
    for m in &request.reply {
        println!("writer: {}", m.text());
    }

    // A non-empty response is revision feedback: the writer is re-invoked
    // with the feedback folded into its conversation, then pauses again.
    let request_id = pending[0].request_id.clone();
    let feedback = "Too vague -- name concrete changes, and mention anything breaking.";
    println!("\nreviewer: {feedback}\n");
    run.send_response(request_id, json!(feedback)).await?;

    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    let request: ApprovalRequest = serde_json::from_value(pending[0].request_data.clone())?;
    println!("-- draft 2, awaiting approval --");
    for m in &request.reply {
        println!("writer: {}", m.text());
    }
    // The feedback became part of the conversation the agent saw.
    assert!(request.conversation.iter().any(|m| m.text() == feedback));

    // An empty response approves the draft; the workflow completes.
    let request_id = pending[0].request_id.clone();
    println!("\nreviewer: (approved -- empty response)\n");
    run.send_response(request_id, json!("")).await?;

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert!(run.pending_requests().is_empty());
    let conversation: Vec<Message> =
        serde_json::from_value(run.last_output().expect("a final output"))?;
    println!("-- final conversation ({} messages) --", conversation.len());
    for m in &conversation {
        let speaker = m.author_name.as_deref().unwrap_or(m.role.as_str());
        println!("{speaker}: {}", m.text());
    }

    Ok(())
}
