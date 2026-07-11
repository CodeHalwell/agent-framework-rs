//! WorkflowAgent tests: expose a built workflow as an `Agent`, aggregate its
//! output as the response, surface pending request-info as user-input requests,
//! and act as an `.as_tool()` target.

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::ChatResponseUpdate;
use async_trait::async_trait;
use futures::StreamExt;

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

fn agent(name: &str, replies: Vec<&str>) -> Arc<dyn Agent> {
    let responses = replies.into_iter().map(ChatResponse::from_text).collect();
    Arc::new(
        ChatAgent::builder(MockClient::new(responses))
            .name(name)
            .build(),
    ) as Arc<dyn Agent>
}

#[tokio::test]
async fn sequential_workflow_as_agent_aggregates_response() {
    let a = agent("A", vec!["step-A"]);
    let b = agent("B", vec!["step-B"]);
    let workflow = SequentialBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    // Exposed via the `WorkflowAgentExt::as_agent` extension on `Workflow`.
    let wf_agent = workflow.as_agent("pipeline");
    assert_eq!(wf_agent.name(), Some("pipeline"));

    let response = wf_agent
        .run(vec![ChatMessage::user("start")], None)
        .await
        .unwrap();
    let texts: Vec<String> = response.messages.iter().map(ChatMessage::text).collect();
    assert!(
        texts.iter().any(|t| t.contains("step-A")),
        "aggregated: {texts:?}"
    );
    assert!(
        texts.iter().any(|t| t.contains("step-B")),
        "aggregated: {texts:?}"
    );
}

#[tokio::test]
async fn workflow_agent_run_once_helper_and_as_tool() {
    let a = agent("A", vec!["only-A"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent = WorkflowAgent::new(workflow, "solo").with_description("runs a single agent");

    // `.as_tool()` yields a callable, executable tool definition.
    let tool = wf_agent.as_tool();
    assert_eq!(tool.name, "solo");
    assert_eq!(tool.description, "runs a single agent");
    assert!(
        tool.is_executable(),
        "workflow-agent tool should be executable"
    );
}

#[tokio::test]
async fn workflow_agent_surfaces_pending_request_info() {
    // An interactive handoff workflow pauses awaiting user input; the wrapping
    // WorkflowAgent surfaces that as a user-input request on the response.
    let coordinator = agent("coordinator", vec!["I need more details."]);
    let workflow = HandoffBuilder::new()
        .participant("coordinator", coordinator)
        .initial_agent("coordinator")
        .with_user_input_request()
        .build()
        .unwrap();

    let wf_agent = WorkflowAgent::new(workflow, "handoff-agent");
    let response = wf_agent
        .run(vec![ChatMessage::user("hello")], None)
        .await
        .unwrap();

    let requests = response.user_input_requests();
    assert_eq!(
        requests.len(),
        1,
        "pending request surfaced as user-input request"
    );
    assert_eq!(requests[0].function_call.name, "request_info");
}

#[tokio::test]
async fn workflow_agent_streams_agent_updates() {
    let a = agent("A", vec!["hello-from-A"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent = WorkflowAgent::new(workflow, "streamer");

    let mut stream = wf_agent.run_stream_with_thread(vec![ChatMessage::user("go")], None);
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert!(
        text.contains("hello-from-A"),
        "streamed agent update: {text}"
    );
}

// ---------------------------------------------------------------------------
// Thread write-back
// ---------------------------------------------------------------------------

#[tokio::test]
async fn workflow_agent_run_persists_input_and_response_to_thread() {
    // A single-participant sequential workflow whose (mocked) agent answers
    // "reply-1" then "reply-2" across two separate `run` calls on the same
    // thread.
    let a = agent("A", vec!["reply-1", "reply-2"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent = WorkflowAgent::new(workflow, "solo");

    let mut thread = wf_agent.get_new_thread();
    assert!(
        thread.list_messages().await.unwrap().is_empty(),
        "a fresh thread starts empty"
    );

    // --- First run ---
    let resp1 = wf_agent
        .run(vec![ChatMessage::user("first")], Some(&mut thread))
        .await
        .unwrap();
    assert!(
        resp1.messages.iter().any(|m| m.text() == "reply-1"),
        "resp1: {:?}",
        resp1.messages
    );

    let after_first = thread.list_messages().await.unwrap();
    assert!(
        !after_first.is_empty(),
        "the thread must be populated after the first run (input write-back missing)"
    );
    assert!(
        after_first.iter().any(|m| m.text() == "first"),
        "input message set 1 missing from thread: {after_first:?}"
    );
    assert!(
        after_first.iter().any(|m| m.text() == "reply-1"),
        "response message set 1 missing from thread: {after_first:?}"
    );

    // --- Second run, same thread ---
    let resp2 = wf_agent
        .run(vec![ChatMessage::user("second")], Some(&mut thread))
        .await
        .unwrap();
    assert!(
        resp2.messages.iter().any(|m| m.text() == "reply-2"),
        "resp2: {:?}",
        resp2.messages
    );

    let after_second = thread.list_messages().await.unwrap();
    assert!(
        after_second.len() > after_first.len(),
        "the second run must append to, not replace, the thread history \
         (before: {after_first:?}, after: {after_second:?})"
    );
    // Matching `ChatAgent`'s exact convention (see `agent_surfaces_and_resolves_approval_round_trip`
    // in `tests/integration.rs`): both runs' input and response message sets
    // are all present in the thread store after two runs.
    assert!(after_second.iter().any(|m| m.text() == "first"));
    assert!(after_second.iter().any(|m| m.text() == "reply-1"));
    assert!(after_second.iter().any(|m| m.text() == "second"));
    assert!(after_second.iter().any(|m| m.text() == "reply-2"));
}

#[tokio::test]
async fn workflow_agent_run_without_explicit_thread_does_not_panic() {
    // No thread supplied: `run` must create and use an ephemeral one
    // internally (mirroring `ChatAgent::run`) rather than erroring.
    let a = agent("A", vec!["only-reply"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent = WorkflowAgent::new(workflow, "solo");

    let resp = wf_agent
        .run(vec![ChatMessage::user("hi")], None)
        .await
        .unwrap();
    assert!(resp.messages.iter().any(|m| m.text() == "only-reply"));
}

#[tokio::test]
async fn workflow_agent_run_stream_with_thread_persists_messages() {
    let a = agent("A", vec!["streamed-reply"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent = WorkflowAgent::new(workflow, "streamer");

    let thread = wf_agent.get_new_thread();
    let mut stream =
        wf_agent.run_stream_with_thread(vec![ChatMessage::user("go")], Some(thread.clone()));
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert!(text.contains("streamed-reply"), "streamed text: {text}");

    // Because message stores are shared via `Arc`, the write-back that
    // happened on the internal thread clone is visible through this clone
    // too (same pattern as `ChatAgent::run_stream`).
    let history = thread.list_messages().await.unwrap();
    assert!(
        history.iter().any(|m| m.text() == "go"),
        "input missing from thread: {history:?}"
    );
    assert!(
        history.iter().any(|m| m.text() == "streamed-reply"),
        "response missing from thread: {history:?}"
    );
}

#[tokio::test]
async fn workflow_agent_trait_run_stream_yields_updates() {
    // The object-safe `Agent::run_stream` override streams the workflow's agent
    // activity (exercised through a `dyn Agent`, as hosting/orchestration do).
    let a = agent("A", vec!["hello-from-A"]);
    let workflow = SequentialBuilder::new().add(a).build().unwrap();
    let wf_agent: Arc<dyn Agent> = Arc::new(WorkflowAgent::new(workflow, "streamer"));

    let mut stream = wf_agent
        .run_stream(vec![ChatMessage::user("go")], None, None)
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert!(text.contains("hello-from-A"), "streamed via trait: {text}");
}
