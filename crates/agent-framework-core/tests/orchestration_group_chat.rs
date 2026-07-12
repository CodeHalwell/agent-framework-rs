//! Group chat orchestration tests (round-robin, custom manager, LLM manager).
//! All exchanges use a scripted mock chat client — no network.

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use agent_framework_core::workflow::GroupChatDirective;
use async_trait::async_trait;
use futures::StreamExt;

/// A scripted chat client that returns queued responses in order.
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
        _messages: Vec<Message>,
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

/// Build a named agent that returns the given scripted text replies.
fn agent(name: &str, replies: Vec<&str>) -> Arc<dyn SupportsAgentRun> {
    let responses = replies.into_iter().map(ChatResponse::from_text).collect();
    Arc::new(
        Agent::builder(MockClient::new(responses))
            .name(name)
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

fn conversation(run: &WorkflowRun) -> Vec<Message> {
    let output = run.last_output().expect("group chat should yield output");
    serde_json::from_value(output).expect("output is a conversation")
}

#[tokio::test]
async fn round_robin_visits_participants_in_order() {
    let a = agent("A", vec!["a-speaks"]);
    let b = agent("B", vec!["b-speaks"]);
    let c = agent("C", vec!["c-speaks"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .participant("B", b)
        .participant("C", c)
        .round_robin()
        .max_rounds(3)
        .build()
        .unwrap();

    let run = workflow.run("kick off").await.unwrap();
    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(Message::text).collect();

    let ia = texts.iter().position(|t| t.contains("a-speaks")).unwrap();
    let ib = texts.iter().position(|t| t.contains("b-speaks")).unwrap();
    let ic = texts.iter().position(|t| t.contains("c-speaks")).unwrap();
    assert!(ia < ib && ib < ic, "round-robin order A<B<C: {texts:?}");

    // Author names are attributed to the speaking participant.
    assert!(conv
        .iter()
        .any(|m| m.author_name.as_deref() == Some("A") && m.text() == "a-speaks"));
}

#[tokio::test]
async fn custom_manager_can_finish() {
    let a = agent("writer", vec!["draft-text"]);

    let workflow = GroupChatBuilder::new()
        .participant("writer", a)
        .manager_fn(|state: &GroupChatState| {
            if state.round_index == 0 {
                GroupChatDirective::speak("writer")
            } else {
                GroupChatDirective::finish_text("all wrapped up")
            }
        })
        .max_rounds(10)
        .build()
        .unwrap();

    let run = workflow.run("write something").await.unwrap();
    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(Message::text).collect();

    assert!(texts.iter().any(|t| t.contains("draft-text")));
    assert!(
        texts.iter().any(|t| t.contains("all wrapped up")),
        "manager finish message present: {texts:?}"
    );
}

/// Build an LLM manager agent scripted to emit JSON `ManagerSelectionResponse`s.
fn manager_agent(json_responses: Vec<&str>) -> Arc<dyn SupportsAgentRun> {
    let responses = json_responses
        .into_iter()
        .map(ChatResponse::from_text)
        .collect();
    Arc::new(
        Agent::builder(MockClient::new(responses))
            .name("manager")
            .build(),
    ) as Arc<dyn SupportsAgentRun>
}

#[tokio::test]
async fn llm_manager_parses_json_selection() {
    // Round 0: select A. Round 1: finish.
    let manager = manager_agent(vec![
        r#"{"selected_participant": "A", "instruction": "please answer", "finish": false}"#,
        r#"{"finish": true, "final_message": "resolved"}"#,
    ]);
    let a = agent("A", vec!["a-answer"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .max_rounds(10)
        .build()
        .unwrap();

    let run = workflow.run("please solve").await.unwrap();
    let conv = conversation(&run);
    let texts: Vec<String> = conv.iter().map(Message::text).collect();

    assert!(
        texts.iter().any(|t| t.contains("please answer")),
        "instruction injected: {texts:?}"
    );
    assert!(texts.iter().any(|t| t.contains("a-answer")));
    assert!(
        texts.iter().any(|t| t.contains("resolved")),
        "final message: {texts:?}"
    );
}

#[tokio::test]
async fn llm_manager_malformed_json_surfaces_error() {
    // A non-JSON manager response cannot be parsed -> the run fails (matching
    // Python's `_parse_manager_selection` which raises on unparseable output).
    let manager = manager_agent(vec!["I choose nobody, sorry!"]);
    let a = agent("A", vec!["unused"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .build()
        .unwrap();

    let result = workflow.run("solve").await;
    assert!(
        result.is_err(),
        "malformed manager JSON should fail the run"
    );
}

#[tokio::test]
async fn termination_condition_halts_conversation() {
    // Terminate as soon as any assistant message mentions "STOP".
    let a = agent("A", vec!["STOP now"]);
    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .round_robin()
        .max_rounds(40)
        .termination_condition(|conv: &[Message]| conv.iter().any(|m| m.text().contains("STOP")))
        .build()
        .unwrap();

    let run = workflow.run("go").await.unwrap();
    let conv = conversation(&run);
    // A speaks once ("STOP now"), then the termination check halts the chat.
    assert_eq!(
        conv.iter().filter(|m| m.text() == "STOP now").count(),
        1,
        "participant should speak exactly once before termination"
    );
    assert_eq!(run.state(), WorkflowRunState::Idle);
}
