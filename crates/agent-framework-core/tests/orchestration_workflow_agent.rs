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

    let mut stream = wf_agent.run_stream(vec![ChatMessage::user("go")]);
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert!(
        text.contains("hello-from-A"),
        "streamed agent update: {text}"
    );
}
