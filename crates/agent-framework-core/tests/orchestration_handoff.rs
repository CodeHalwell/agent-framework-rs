//! Handoff orchestration tests: tool-call transfer, autonomous completion, the
//! interactive request-info pause/resume path, and unknown-target handling.

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::{
    ChatResponseUpdate, Content, FinishReason, FunctionArguments, FunctionCallContent, Role,
};
use agent_framework_core::workflow::HandoffUserInputRequest;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
        }
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        _messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resps = self.responses.lock().unwrap();
        if resps.is_empty() {
            Ok(ChatResponse::from_text("(no more scripted responses)"))
        } else {
            Ok(resps.remove(0))
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
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// A response containing a single handoff tool call to `target`.
fn handoff_response(call_id: &str, target: &str) -> ChatResponse {
    let call = FunctionCallContent::new(
        call_id,
        format!("handoff_to_{target}"),
        Some(FunctionArguments::Raw("{}".into())),
    );
    ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

fn agent_with(name: &str, responses: Vec<ChatResponse>) -> Arc<dyn Agent> {
    Arc::new(
        ChatAgent::builder(MockClient::new(responses))
            .name(name)
            .build(),
    ) as Arc<dyn Agent>
}

fn conversation(value: serde_json::Value) -> Vec<ChatMessage> {
    serde_json::from_value(value).expect("output is a conversation")
}

#[tokio::test]
async fn autonomous_handoff_completes_after_transfer() {
    // A hands off to B; B answers with no handoff -> autonomous single-shot completes.
    let a = agent_with("A", vec![handoff_response("c1", "B")]);
    let b = agent_with("B", vec![ChatResponse::from_text("B handled it")]);

    let workflow = HandoffBuilder::new()
        .participant("A", a)
        .participant("B", b)
        .initial_agent("A")
        .add_handoff("A")
        .to(["B"])
        .autonomous()
        .build()
        .unwrap();

    let run = workflow.run("please help").await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::Idle);
    let conv = conversation(run.last_output().expect("autonomous run yields output"));
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("B handled it")),
        "specialist answer present: {texts:?}"
    );
}

#[tokio::test]
async fn interactive_handoff_pauses_and_resumes() {
    // Human-in-the-loop: agent answers without handoff -> request user input;
    // after a reply the agent runs again.
    let a = agent_with(
        "A",
        vec![
            ChatResponse::from_text("How can I help?"),
            ChatResponse::from_text("Thanks, all done."),
        ],
    );

    let workflow = HandoffBuilder::new()
        .participant("A", a)
        .initial_agent("A")
        .with_user_input_request()
        // Terminate only after 3 user turns so the second agent turn runs.
        .termination_condition(|conv: &[ChatMessage]| {
            conv.iter().filter(|m| m.role == Role::user()).count() >= 3
        })
        .build()
        .unwrap();

    let mut run = workflow.run("hello").await.unwrap();
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1, "one user-input request outstanding");

    // Inspect the request payload.
    let request: HandoffUserInputRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.awaiting_agent, "A");
    assert!(request
        .conversation
        .iter()
        .any(|m| m.text().contains("How can I help?")));

    // Reply -> the coordinator routes back to A, which runs a second turn.
    let req_id = pending[0].request_id.clone();
    run.send_response(req_id, json!("here is more info"))
        .await
        .unwrap();

    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);
    let pending2 = run.pending_requests();
    assert_eq!(pending2.len(), 1, "pauses again for the next user turn");
    let request2: HandoffUserInputRequest =
        serde_json::from_value(pending2[0].request_data.clone()).unwrap();
    assert!(
        request2
            .conversation
            .iter()
            .any(|m| m.text().contains("Thanks, all done.")),
        "second agent turn ran after the user reply: {:?}",
        request2.conversation
    );
}

#[tokio::test]
async fn unknown_handoff_target_is_fed_back() {
    // A requests a transfer to a non-existent specialist; the error is fed back
    // and A recovers on its next turn.
    let a = agent_with(
        "A",
        vec![
            handoff_response("c1", "ghost"),
            ChatResponse::from_text("Recovered and answered directly."),
        ],
    );

    let workflow = HandoffBuilder::new()
        .participant("A", a)
        .initial_agent("A")
        .autonomous()
        .build()
        .unwrap();

    let run = workflow.run("do it").await.unwrap();
    let conv = conversation(run.last_output().expect("run yields output"));
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("Recovered and answered directly.")),
        "agent recovered after unknown-target error: {texts:?}"
    );
}

#[tokio::test]
async fn specialist_to_specialist_handoff_chains() {
    // A -> B via handoff, then B -> C via handoff, then C answers.
    let a = agent_with("A", vec![handoff_response("c1", "B")]);
    let b = agent_with("B", vec![handoff_response("c2", "C")]);
    let c = agent_with("C", vec![ChatResponse::from_text("C final answer")]);

    let workflow = HandoffBuilder::new()
        .participant("A", a)
        .participant("B", b)
        .participant("C", c)
        .initial_agent("A")
        .add_handoff("A")
        .to(["B"])
        .add_handoff("B")
        .to(["C"])
        .autonomous()
        .build()
        .unwrap();

    let run = workflow.run("start").await.unwrap();
    let conv = conversation(run.last_output().expect("run yields output"));
    let texts: Vec<String> = conv.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("C final answer")),
        "chained: {texts:?}"
    );
}
