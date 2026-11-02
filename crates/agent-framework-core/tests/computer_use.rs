//! Computer use (experimental `COMPUTER_USE`): the content types, result
//! validation, the tool loop stopping on a pending computer call, and the
//! workflow pause and resume.

#![cfg(feature = "experimental-computer-use")]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::{
    ComputerSafetyCheck, ComputerToolCallContent, ComputerToolResultContent, FunctionArguments,
    FunctionCallContent, UriContent,
};
use agent_framework_core::workflow::{ComputerUseRequest, SequentialBuilder};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

fn click() -> Vec<Value> {
    vec![json!({"type": "click", "x": 10, "y": 20, "button": "left"})]
}

fn check(id: &str) -> ComputerSafetyCheck {
    ComputerSafetyCheck {
        id: id.into(),
        code: Some("malicious_instructions".into()),
        message: Some("Review the page".into()),
    }
}

fn screenshot() -> Content {
    Content::Uri(UriContent {
        uri: "https://example.com/s.png".into(),
        media_type: "image/png".into(),
    })
}

#[test]
fn content_round_trips_with_upstream_tags() {
    let call = ComputerToolCallContent::new("cu_1", "call_1", click())
        .unwrap()
        .with_pending_safety_checks(vec![check("sc_1")]);
    let wire = serde_json::to_value(Content::ComputerToolCall(call.clone())).unwrap();
    assert_eq!(wire["type"], "computer_tool_call");
    assert_eq!(wire["actions"][0]["type"], "click");
    assert!(
        wire.get("user_input_request").is_none(),
        "true is the default"
    );
    assert_eq!(
        serde_json::from_value::<Content>(wire).unwrap(),
        Content::ComputerToolCall(call)
    );

    let result = ComputerToolResultContent::new("call_1")
        .unwrap()
        .with_screenshot(screenshot())
        .unwrap()
        .with_acknowledged_safety_checks(vec![check("sc_1")]);
    let wire = serde_json::to_value(Content::ComputerToolResult(result.clone())).unwrap();
    assert_eq!(wire["type"], "computer_tool_result");
    assert_eq!(wire["screenshot"]["type"], "uri");
    assert_eq!(
        serde_json::from_value::<Content>(wire).unwrap(),
        Content::ComputerToolResult(result)
    );

    assert!(ComputerToolCallContent::new("", "c", click()).is_err());
    assert!(ComputerToolCallContent::new("i", "c", vec![]).is_err());
    assert!(ComputerToolResultContent::new("").is_err());
    assert!(ComputerToolResultContent::new("c")
        .unwrap()
        .with_screenshot(Content::text("not an image"))
        .is_err());
}

#[test]
fn results_must_answer_the_call_and_acknowledge_exactly_its_checks() {
    let call = ComputerToolCallContent::new("cu_1", "call_1", click())
        .unwrap()
        .with_pending_safety_checks(vec![check("a"), check("b")]);
    let result = |call_id: &str, acks: &[&str]| {
        ComputerToolResultContent::new(call_id)
            .unwrap()
            .with_acknowledged_safety_checks(acks.iter().map(|id| check(id)).collect())
    };
    assert!(call.validate_result(&result("call_1", &["b", "a"])).is_ok());
    assert!(call.validate_result(&result("other", &["a", "b"])).is_err());
    assert!(call.validate_result(&result("call_1", &["a"])).is_err());
    assert!(call
        .validate_result(&result("call_1", &["a", "b", "c"]))
        .is_err());
}

/// Replies with `replies` in order and records every request.
struct Scripted {
    replies: Vec<Vec<Content>>,
    calls: AtomicUsize,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

#[async_trait]
impl ChatClient for Scripted {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        let i = self.calls.fetch_add(1, Ordering::SeqCst);
        let contents = self.replies[i.min(self.replies.len() - 1)].clone();
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), contents)],
            ..Default::default()
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let update = ChatResponseUpdate {
            contents: response.messages[0].contents.clone(),
            role: Some(Role::assistant()),
            ..Default::default()
        };
        Ok(futures::stream::iter(vec![Ok(update)]).boxed())
    }
}

fn scripted(replies: Vec<Vec<Content>>) -> (Scripted, Arc<Mutex<Vec<Vec<Message>>>>) {
    let seen = Arc::new(Mutex::new(Vec::new()));
    (
        Scripted {
            replies,
            calls: AtomicUsize::new(0),
            seen: seen.clone(),
        },
        seen,
    )
}

fn computer_call() -> Content {
    Content::ComputerToolCall(
        ComputerToolCallContent::new("cu_1", "call_1", click())
            .unwrap()
            .with_pending_safety_checks(vec![check("sc_1")]),
    )
}

#[tokio::test]
async fn a_pending_computer_call_returns_to_the_caller_after_local_tools_run() {
    let ran = Arc::new(AtomicUsize::new(0));
    let counter = ran.clone();
    let tool = FunctionTool::new(
        "note",
        "Take a note",
        json!({"type": "object", "properties": {}}),
        move |_args| {
            counter.fetch_add(1, Ordering::SeqCst);
            async move { Ok(json!("noted")) }
        },
    )
    .into_definition();
    let (client, seen) = scripted(vec![
        vec![
            Content::FunctionCall(FunctionCallContent::new(
                "f1",
                "note",
                Some(FunctionArguments::Raw("{}".into())),
            )),
            computer_call(),
        ],
        vec![Content::text("should not be reached")],
    ]);
    let agent = Agent::builder(client).tool(tool).build();
    let response = agent
        .run(vec![Message::user("open the page")], None)
        .await
        .unwrap();
    assert_eq!(ran.load(Ordering::SeqCst), 1, "the local tool still runs");
    assert_eq!(
        seen.lock().unwrap().len(),
        1,
        "the model is not called again"
    );
    assert!(response
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::ComputerToolCall(_))));
}

#[tokio::test]
async fn a_workflow_pauses_on_a_computer_call_and_resumes_with_the_result() {
    let (client, seen) = scripted(vec![vec![computer_call()], vec![Content::text("done")]]);
    let agent: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(client).name("operator").build());
    let workflow = SequentialBuilder::new()
        .participants(vec![agent])
        .build()
        .unwrap();
    let mut run = workflow.run("click the button").await.unwrap();
    let pending = run.pending_requests();
    assert_eq!(pending.len(), 1);
    let request: ComputerUseRequest =
        serde_json::from_value(pending[0].request_data.clone()).unwrap();
    assert_eq!(request.computer_calls[0].call_id, "call_1");
    let request_id = pending[0].request_id.clone();

    // A result that does not acknowledge the pending check is refused.
    let unacknowledged = ComputerToolResultContent::new("call_1")
        .unwrap()
        .with_screenshot(screenshot())
        .unwrap();
    let (bad_client, _) = scripted(vec![vec![computer_call()], vec![Content::text("done")]]);
    let bad_agent: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(bad_client).name("operator").build());
    let bad_workflow = SequentialBuilder::new()
        .participants(vec![bad_agent])
        .build()
        .unwrap();
    let mut bad_run = bad_workflow.run("click the button").await.unwrap();
    let bad_id = bad_run.pending_requests()[0].request_id.clone();
    assert!(bad_run
        .send_response(bad_id, serde_json::to_value(&unacknowledged).unwrap())
        .await
        .is_err());

    let result = unacknowledged.with_acknowledged_safety_checks(vec![check("sc_1")]);
    run.send_response(request_id, serde_json::to_value(&result).unwrap())
        .await
        .unwrap();
    assert!(run.pending_requests().is_empty());
    let output = run.last_output().expect("an output");
    let messages: Vec<Message> = serde_json::from_value(output).unwrap();
    assert_eq!(messages.last().unwrap().text(), "done");

    // The resumed run sent the call and its result.
    let resumed = seen.lock().unwrap().last().unwrap().clone();
    let contents: Vec<&Content> = resumed.iter().flat_map(|m| &m.contents).collect();
    assert!(contents
        .iter()
        .any(|c| matches!(c, Content::ComputerToolResult(r) if r.call_id == "call_1")));
}
