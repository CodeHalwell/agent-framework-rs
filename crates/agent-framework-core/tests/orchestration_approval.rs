//! Post-agent human-approval HITL tests for the Sequential and Concurrent
//! orchestrations: `.with_request_info()` pauses *after* the agent responds
//! (an `AgentApprovalExecutor`, not the old pre-agent interceptor), routing
//! an empty response as approval and a non-empty response as revision
//! feedback that re-invokes the agent — exercised through the engine's
//! generic `pending_requests()` / `send_response()` machinery. No network.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework_core::prelude::*;
use agent_framework_core::workflow::{ApprovalRequest, ConcurrentBuilder, SequentialBuilder};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// A chat client that returns successive canned replies on each call,
/// repeating the last reply once the script is exhausted.
struct ScriptedClient {
    replies: Vec<String>,
    calls: AtomicUsize,
}

impl ScriptedClient {
    fn new(replies: Vec<&str>) -> Self {
        Self {
            replies: replies.into_iter().map(String::from).collect(),
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ChatClient for ScriptedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        let text = self
            .replies
            .get(i)
            .or_else(|| self.replies.last())
            .cloned()
            .unwrap_or_default();
        Ok(ChatResponse::from_text(text))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
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
        Ok(futures::stream::iter(updates).boxed())
    }
}

fn scripted_agent(id: &str, replies: Vec<&str>) -> Arc<dyn SupportsAgentRun> {
    Arc::new(
        Agent::builder(ScriptedClient::new(replies))
            .id(id)
            .name(id)
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

/// Extract the text of every message in a workflow output/`request_data`
/// conversation payload.
fn texts_of(value: &serde_json::Value) -> Vec<String> {
    let conv: Vec<Message> = serde_json::from_value(value.clone()).unwrap();
    conv.iter().map(|m| m.text()).collect()
}

// ----------------------------------------------------------------------------
// Sequential
// ----------------------------------------------------------------------------

#[tokio::test]
async fn sequential_with_request_info_pauses_after_agent_responds() {
    let agent = scripted_agent("a", vec!["draft-1"]);
    let workflow = SequentialBuilder::new()
        .participants(vec![agent])
        .with_request_info()
        .build()
        .unwrap();

    let run = workflow.run("do the thing").await.unwrap();

    // Paused: the agent already ran (post-agent pause), its reply is awaiting
    // approval, not the raw input.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);

    let request: ApprovalRequest = serde_json::from_value(pending[0].request_data.clone()).unwrap();
    let reply_texts: Vec<String> = request.reply.iter().map(|m| m.text()).collect();
    assert_eq!(reply_texts, vec!["draft-1".to_string()]);

    assert!(run
        .events()
        .iter()
        .any(|e| matches!(e, WorkflowEvent::RequestInfo { .. })));
}

#[tokio::test]
async fn sequential_with_request_info_empty_response_approves_and_completes() {
    let agent = scripted_agent("a", vec!["draft-1"]);
    let workflow = SequentialBuilder::new()
        .participants(vec![agent])
        .with_request_info()
        .build()
        .unwrap();

    let mut run = workflow.run("do the thing").await.unwrap();
    let request_id = run.pending_requests()[0].request_id.clone();

    // Empty response == approve.
    run.send_response(request_id, json!("")).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert!(run.pending_requests().is_empty());
    let output = run.last_output().expect("a final output");
    let texts = texts_of(&output);
    assert!(texts.contains(&"draft-1".to_string()));
}

#[tokio::test]
async fn sequential_with_request_info_revision_re_invokes_and_pauses_again() {
    let agent = scripted_agent("a", vec!["draft-1", "draft-2"]);
    let workflow = SequentialBuilder::new()
        .participants(vec![agent])
        .with_request_info()
        .build()
        .unwrap();

    let mut run = workflow.run("do the thing").await.unwrap();
    let first_id = run.pending_requests()[0].request_id.clone();

    // Non-empty response == revision feedback; the agent is re-invoked and a
    // *new* approval request is raised (iterate-until-approved), rather than
    // completing the workflow.
    run.send_response(first_id, json!("please tighten this up"))
        .await
        .unwrap();

    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    let request: ApprovalRequest = serde_json::from_value(pending[0].request_data.clone()).unwrap();
    let reply_texts: Vec<String> = request.reply.iter().map(|m| m.text()).collect();
    assert_eq!(reply_texts, vec!["draft-2".to_string()]);
    // The revision feedback was folded into the conversation the agent saw.
    let conv_texts: Vec<String> = request.conversation.iter().map(|m| m.text()).collect();
    assert!(conv_texts.iter().any(|t| t == "please tighten this up"));
    assert!(conv_texts.iter().any(|t| t == "draft-1"));

    // Approving the second round completes the workflow with draft-2.
    let second_id = pending[0].request_id.clone();
    run.send_response(second_id, json!(null)).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    let output = run.last_output().expect("a final output");
    let texts = texts_of(&output);
    assert!(texts.contains(&"draft-2".to_string()));
}

#[tokio::test]
async fn sequential_default_behavior_unchanged_without_request_info() {
    let agent = scripted_agent("a", vec!["draft-1"]);
    let workflow = SequentialBuilder::new()
        .participants(vec![agent])
        .build()
        .unwrap();

    let run = workflow.run("do the thing").await.unwrap();

    // No pause: default Sequential behavior is preserved when
    // `.with_request_info()` is never called.
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert!(run.pending_requests().is_empty());
    let output = run.last_output().expect("a final output");
    assert!(texts_of(&output).contains(&"draft-1".to_string()));
}

// ----------------------------------------------------------------------------
// Concurrent
// ----------------------------------------------------------------------------

#[tokio::test]
async fn concurrent_with_request_info_pauses_each_participant_individually() {
    let a = scripted_agent("a", vec!["from-A"]);
    let b = scripted_agent("b", vec!["from-B"]);
    let workflow = ConcurrentBuilder::new()
        .participants(vec![a, b])
        .with_request_info()
        .build()
        .unwrap();

    let run = workflow.run("question").await.unwrap();

    // Two independent per-agent pauses, not one combined pause.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 2);
}

#[tokio::test]
async fn concurrent_with_request_info_approving_all_completes_with_aggregated_output() {
    let a = scripted_agent("a", vec!["from-A"]);
    let b = scripted_agent("b", vec!["from-B"]);
    let workflow = ConcurrentBuilder::new()
        .participants(vec![a, b])
        .with_request_info()
        .build()
        .unwrap();

    let mut run = workflow.run("question").await.unwrap();
    let ids: Vec<String> = run
        .pending_requests()
        .iter()
        .map(|p| p.request_id.clone())
        .collect();
    for id in ids {
        run.send_response(id, json!("")).await.unwrap();
    }

    assert_eq!(run.state(), WorkflowRunState::Idle);
    let output = run.last_output().expect("a final output");
    let texts = texts_of(&output);
    assert!(texts.contains(&"from-A".to_string()));
    assert!(texts.contains(&"from-B".to_string()));
}

#[tokio::test]
async fn concurrent_with_request_info_revision_only_re_invokes_that_participant() {
    let a = scripted_agent("a", vec!["from-A"]);
    let b = scripted_agent("b", vec!["from-B-draft", "from-B-final"]);
    let workflow = ConcurrentBuilder::new()
        .participants(vec![a, b])
        .with_request_info()
        .build()
        .unwrap();

    let mut run = workflow.run("question").await.unwrap();
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 2);

    // Find B's pending request (the one whose reply is "from-B-draft") and
    // send revision feedback; A's request is approved outright.
    let mut b_id = None;
    let mut a_id = None;
    for p in &pending {
        let request: ApprovalRequest = serde_json::from_value(p.request_data.clone()).unwrap();
        let reply_texts: Vec<String> = request.reply.iter().map(|m| m.text()).collect();
        if reply_texts.contains(&"from-B-draft".to_string()) {
            b_id = Some(p.request_id.clone());
        } else if reply_texts.contains(&"from-A".to_string()) {
            a_id = Some(p.request_id.clone());
        }
    }
    let b_id = b_id.expect("B's pending request");
    let a_id = a_id.expect("A's pending request");

    run.send_response(a_id, json!("")).await.unwrap();
    run.send_response(b_id, json!("try again")).await.unwrap();

    // A is done; B is still pending its second round.
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    let request: ApprovalRequest = serde_json::from_value(pending[0].request_data.clone()).unwrap();
    let reply_texts: Vec<String> = request.reply.iter().map(|m| m.text()).collect();
    assert_eq!(reply_texts, vec!["from-B-final".to_string()]);

    let second_id = pending[0].request_id.clone();
    run.send_response(second_id, json!("")).await.unwrap();

    assert_eq!(run.state(), WorkflowRunState::Idle);
    let output = run.last_output().expect("a final output");
    let texts = texts_of(&output);
    assert!(texts.contains(&"from-A".to_string()));
    assert!(texts.contains(&"from-B-final".to_string()));
    assert!(!texts.contains(&"from-B-draft".to_string()));
}
