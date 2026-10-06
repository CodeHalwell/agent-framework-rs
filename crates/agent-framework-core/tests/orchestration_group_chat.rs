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

/// `intermediate_output_from` demotes the group chat's single final yield
/// (the finished conversation) from the workflow's terminal output to a
/// non-terminal `Intermediate` event — useful when a group chat is composed
/// as one stage of a larger pipeline. See [`GroupChatBuilder::output_from`]
/// docs for why this is whole-orchestrator-granular rather than
/// per-participant (a group chat compiles to a single executor).
#[tokio::test]
async fn intermediate_output_from_demotes_final_yield() {
    let a = agent("A", vec!["a-speaks"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .round_robin()
        .max_rounds(1)
        .intermediate_output_from(["A"])
        .build()
        .unwrap();

    let run = workflow.run("kick off").await.unwrap();

    assert!(
        run.last_output().is_none(),
        "no terminal output should be recorded once demoted to Intermediate"
    );
    let intermediate = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::Intermediate { .. }))
        .count();
    assert_eq!(
        intermediate, 1,
        "the sole yield became a non-terminal event"
    );
    let output_events = run
        .events()
        .iter()
        .filter(|e| matches!(e, WorkflowEvent::Output { .. }))
        .count();
    assert_eq!(output_events, 0);
}

/// `output_from` naming a known participant preserves the default: the
/// finished conversation remains the workflow's terminal output.
#[tokio::test]
async fn output_from_preserves_terminal_output() {
    let a = agent("A", vec!["a-speaks"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .round_robin()
        .max_rounds(1)
        .output_from(["A"])
        .build()
        .unwrap();

    let run = workflow.run("kick off").await.unwrap();
    assert!(run.last_output().is_some());
}

/// Unknown participant names are rejected at build time.
#[tokio::test]
async fn output_from_rejects_unknown_participant() {
    let a = agent("A", vec!["a-speaks"]);

    let err = match GroupChatBuilder::new()
        .participant("A", a)
        .output_from(["nobody"])
        .build()
    {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("nobody"));
}

/// Naming participants in both lists is rejected: on this single-executor
/// builder both resolve to the same underlying executor id.
#[tokio::test]
async fn output_from_and_intermediate_output_from_conflict() {
    let a = agent("A", vec!["a-speaks"]);
    let b = agent("B", vec!["b-speaks"]);

    let err = match GroupChatBuilder::new()
        .participant("A", a)
        .participant("B", b)
        .output_from(["A"])
        .intermediate_output_from(["B"])
        .build()
    {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("group_chat_orchestrator"));
}

#[tokio::test]
async fn llm_manager_parses_a_fenced_json_selection() {
    // A provider that ignores `response_format` wraps the decision in a
    // Markdown fence. The strict parse used to fail and take the run down,
    // so a `finish: true` the manager did issue was never applied.
    let manager = manager_agent(vec![
        "Thinking about who should go next.\n\n```json\n{\"selected_participant\": \"A\", \"instruction\": \"please answer\", \"finish\": false}\n```",
        "```\n{\"finish\": true, \"final_message\": \"resolved\"}\n```",
    ]);
    let a = agent("A", vec!["a-answer"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .max_rounds(10)
        .build()
        .unwrap();

    let run = workflow.run("please solve").await.unwrap();
    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("please answer")),
        "instruction injected: {texts:?}"
    );
    assert!(texts.iter().any(|t| t.contains("a-answer")), "{texts:?}");
    assert!(
        texts.iter().any(|t| t.contains("resolved")),
        "final message: {texts:?}"
    );
}

/// An inline backtick run ahead of the real block must not be taken as the
/// opening fence.
///
/// CommonMark requires an opening fence to begin its line. Without that rule
/// the inline run in the prose becomes the opening marker and closes at the
/// *real* block's opening fence, so the scanner yields one body of prose and
/// the genuine JSON is never parsed — a valid fenced decision that still
/// fails to read.
#[tokio::test]
async fn an_inline_backtick_run_does_not_open_a_fence() {
    let manager = manager_agent(vec![
        "Use ```json``` fences for structured output.\n\n```json\n{\"selected_participant\": \"A\", \"instruction\": \"please answer\", \"finish\": false}\n```",
        "Wrap it in ```these``` please.\n\n```json\n{\"finish\": true, \"final_message\": \"resolved\"}\n```",
    ]);
    let a = agent("A", vec!["a-answer"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .max_rounds(10)
        .build()
        .unwrap();

    let run = workflow.run("please solve").await.unwrap();
    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("please answer")),
        "the real fenced block should have been parsed: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("resolved")),
        "final message: {texts:?}"
    );
}

/// An indented fence still opens (CommonMark allows up to three spaces),
/// which is the other half of the line-start rule.
#[tokio::test]
async fn an_indented_fence_still_opens() {
    let manager = manager_agent(vec![
        "Here it is:\n\n   ```json\n{\"finish\": true, \"final_message\": \"resolved\"}\n   ```",
    ]);
    let a = agent("A", vec!["a-answer"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .max_rounds(10)
        .build()
        .unwrap();

    let run = workflow.run("please solve").await.unwrap();
    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("resolved")),
        "an indented fence should still parse: {texts:?}"
    );
}

#[tokio::test]
async fn llm_manager_takes_the_last_fenced_block() {
    // A model that shows a worked example first and closes with its actual
    // decision: the final block is the one that counts.
    let manager = manager_agent(vec![
        "For example:\n\n```json\n{\"finish\": true, \"final_message\": \"example\"}\n```\n\nMy decision:\n\n```json\n{\"finish\": true, \"final_message\": \"real\"}\n```",
    ]);
    let a = agent("A", vec!["unused"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .build()
        .unwrap();

    let run = workflow.run("solve").await.unwrap();
    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert!(texts.iter().any(|t| t.contains("real")), "{texts:?}");
    assert!(!texts.iter().any(|t| t.contains("example")), "{texts:?}");
}

#[tokio::test]
async fn llm_manager_keeps_backticks_inside_a_final_message() {
    // A closing fence has to end its line, so the backticks around `cargo`
    // inside the JSON string do not end the block early.
    let manager = manager_agent(vec![
        "```json\n{\"finish\": true, \"final_message\": \"run ```cargo test``` first\"}\n```",
    ]);
    let a = agent("A", vec!["unused"]);

    let workflow = GroupChatBuilder::new()
        .participant("A", a)
        .manager_agent(manager)
        .build()
        .unwrap();

    let run = workflow.run("solve").await.unwrap();
    let texts: Vec<String> = conversation(&run).iter().map(Message::text).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.contains("run ```cargo test``` first")),
        "{texts:?}"
    );
}
