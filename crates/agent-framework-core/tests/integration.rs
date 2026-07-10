//! End-to-end tests exercising agents, the tool loop, and workflows using a
//! mock chat client (no network).

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Role};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// A scripted chat client that returns queued responses in order.
#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
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

#[tokio::test]
async fn basic_agent_run() {
    let client = MockClient::new(vec![ChatResponse::from_text("Hello there!")]);
    let agent = ChatAgent::builder(client)
        .name("assistant")
        .instructions("Be nice.")
        .build();

    let response = agent.run_once("Hi").await.unwrap();
    assert_eq!(response.text(), "Hello there!");
    assert_eq!(
        response.messages[0].author_name.as_deref(),
        Some("assistant")
    );
}

#[tokio::test]
async fn agent_streaming_updates_thread() {
    let client = MockClient::new(vec![ChatResponse::from_text("streamed reply")]);
    let agent = ChatAgent::builder(client).build();

    let mut thread = agent.get_new_thread();
    let mut stream = agent
        .run_stream("hello", Some(thread.clone()))
        .await
        .unwrap();
    let mut text = String::new();
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
    }
    assert_eq!(text, "streamed reply");
    // The shared store should now contain the user + assistant messages.
    let history = thread.list_messages().await.unwrap();
    assert_eq!(history.len(), 2);
    let _ = &mut thread;
}

#[tokio::test]
async fn tool_loop_executes_function() {
    // First response asks to call `add`; second returns the final answer.
    let call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw(json!({"a": 2, "b": 3}).to_string())),
    );
    let ask = ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("The sum is 5.");
    let client = MockClient::new(vec![ask, answer]);

    let add = AiFunction::new(
        "add",
        "Add two integers.",
        json!({
            "type": "object",
            "properties": { "a": {"type":"integer"}, "b": {"type":"integer"} },
            "required": ["a","b"]
        }),
        |args| async move {
            let a = args["a"].as_i64().unwrap_or(0);
            let b = args["b"].as_i64().unwrap_or(0);
            Ok(json!(a + b))
        },
    )
    .into_definition();

    let agent = ChatAgent::builder(client).tool(add).build();
    let response = agent.run_once("What is 2 + 3?").await.unwrap();
    assert!(response.text().contains("5"), "got: {}", response.text());
    // The response should include the tool interaction messages.
    assert!(response.messages.iter().any(|m| m.role == Role::tool()
        && m.contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(_)))));
}

#[tokio::test]
async fn sequential_workflow_chains_agents() {
    let a = Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text("step-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn Agent>;
    let b = Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text("step-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn Agent>;

    let workflow = agent_framework_core::workflow::SequentialBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("start").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<ChatMessage> = serde_json::from_value(output).unwrap();
    let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
    assert!(texts.contains(&"step-A".to_string()));
    assert!(texts.contains(&"step-B".to_string()));
}

#[tokio::test]
async fn concurrent_workflow_fans_out() {
    let a = Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text("from-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn Agent>;
    let b = Arc::new(
        ChatAgent::builder(MockClient::new(vec![ChatResponse::from_text("from-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn Agent>;

    let workflow = agent_framework_core::workflow::ConcurrentBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("question").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<ChatMessage> = serde_json::from_value(output).unwrap();
    let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
    assert!(texts.iter().any(|t| t == "from-A"));
    assert!(texts.iter().any(|t| t == "from-B"));
}

#[tokio::test]
async fn workflow_function_executor() {
    use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};

    let doubler = FunctionExecutor::new("double", |msg, ctx| async move {
        let n = msg.as_i64().unwrap_or(0);
        ctx.send_message(json!(n * 2)).await?;
        Ok(())
    });
    let printer = FunctionExecutor::new("out", |msg, ctx| async move {
        ctx.yield_output(msg).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(doubler))
        .add_executor(Arc::new(printer))
        .set_start("double")
        .add_edge("double", "out")
        .build()
        .unwrap();

    let result = workflow.run(json!(21)).await.unwrap();
    assert_eq!(result.last_output(), Some(json!(42)));
}

#[test]
fn chat_options_merge() {
    let base = ChatOptions::new()
        .with_temperature(0.2)
        .with_instructions("base");
    let over = ChatOptions::new()
        .with_temperature(0.9)
        .with_instructions("more");
    let merged = base.merge(over);
    assert_eq!(merged.temperature, Some(0.9));
    assert_eq!(merged.instructions.as_deref(), Some("base\nmore"));
}
