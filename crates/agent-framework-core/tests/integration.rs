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

#[test]
fn function_call_merge_does_not_duplicate_name() {
    // A provider that repeats the full name in a continuation chunk must not
    // produce "addadd".
    let mut base =
        FunctionCallContent::new("c1", "add", Some(FunctionArguments::Raw("{\"a\":".into())));
    let cont = FunctionCallContent::new("", "add", Some(FunctionArguments::Raw("1}".into())));
    base.merge(&cont).unwrap();
    assert_eq!(base.name, "add");
    match base.arguments {
        Some(FunctionArguments::Raw(s)) => assert_eq!(s, "{\"a\":1}"),
        other => panic!("unexpected args: {other:?}"),
    }
}

/// Agent middleware that appends a suffix to every assistant message.
struct SuffixMiddleware;

#[async_trait]
impl Middleware<AgentRunContext> for SuffixMiddleware {
    async fn process(
        &self,
        ctx: AgentRunContext,
        next: Next<AgentRunContext>,
    ) -> Result<AgentRunContext> {
        let mut ctx = next.run(ctx).await?;
        if let Some(resp) = ctx.result.as_mut() {
            for m in &mut resp.messages {
                m.contents.push(Content::text(" [checked]"));
            }
        }
        Ok(ctx)
    }
}

#[tokio::test]
async fn middleware_applies_on_streaming_path() {
    let client = MockClient::new(vec![ChatResponse::from_text("answer")]);
    let agent = ChatAgent::builder(client)
        .middleware(Arc::new(SuffixMiddleware))
        .build();

    // Streaming must honor the middleware just like `run` does.
    let mut stream = agent.run_stream("hi", None).await.unwrap();
    let mut text = String::new();
    while let Some(u) = stream.next().await {
        text.push_str(&u.unwrap().text());
    }
    assert!(text.contains("answer"), "got: {text}");
    assert!(
        text.contains("[checked]"),
        "middleware not applied on stream: {text}"
    );
}

#[tokio::test]
async fn tool_loop_reports_invalid_arguments() {
    // The model asks to call `add` with malformed JSON arguments; the loop must
    // report a tool error rather than invoking with null input.
    let bad_call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw("{ not json".into())),
    );
    let ask = ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(bad_call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("done");

    let invoked = Arc::new(Mutex::new(false));
    let invoked_clone = invoked.clone();
    let add = AiFunction::new(
        "add",
        "Add.",
        json!({"type":"object","properties":{}}),
        move |_args| {
            let invoked = invoked_clone.clone();
            async move {
                *invoked.lock().unwrap() = true;
                Ok(json!(0))
            }
        },
    )
    .into_definition();

    let agent = ChatAgent::builder(MockClient::new(vec![ask, answer]))
        .tool(add)
        .build();
    let response = agent.run_once("add please").await.unwrap();

    // The tool must NOT have been invoked with bogus arguments.
    assert!(
        !*invoked.lock().unwrap(),
        "tool should not run on invalid args"
    );
    // A tool-error result should be present in the conversation.
    assert!(response.messages.iter().any(|m| m
        .contents
        .iter()
        .any(|c| matches!(c, Content::FunctionResult(fr) if fr.exception.is_some()))));
}

/// A context provider that records whether `invoked` fired and injects a tool.
struct RecordingProvider {
    invoked: Arc<Mutex<bool>>,
}

#[async_trait]
impl ContextProvider for RecordingProvider {
    async fn invoking(&self, _messages: &[ChatMessage]) -> Result<Context> {
        Ok(Context::new().with_instructions("remember: be brief"))
    }
    async fn invoked(&self, _request: &[ChatMessage], _response: &[ChatMessage]) -> Result<()> {
        *self.invoked.lock().unwrap() = true;
        Ok(())
    }
}

#[tokio::test]
async fn context_provider_invoked_hook_fires() {
    let invoked = Arc::new(Mutex::new(false));
    let provider = RecordingProvider {
        invoked: invoked.clone(),
    };
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));

    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let agent = ChatAgent::builder(client)
        .context_provider(aggregate)
        .build();

    let _ = agent.run_once("hi").await.unwrap();
    assert!(
        *invoked.lock().unwrap(),
        "invoked hook was not called after run"
    );
}

#[tokio::test]
async fn streaming_tool_replay_preserves_message_boundaries() {
    // Tool call, then final answer — two assistant messages that must NOT be
    // merged when the streamed updates are re-aggregated.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("final answer");

    let noop = AiFunction::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("done")) },
    )
    .into_definition();

    let agent = ChatAgent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    // Re-aggregate exactly as a downstream consumer would.
    let aggregated = AgentRunResponse::from_updates(updates);
    // The final answer must appear as its own assistant message, not merged
    // into the earlier tool-call message.
    let final_msg = aggregated.messages.last().unwrap();
    assert_eq!(final_msg.text(), "final answer");
    assert!(
        final_msg
            .contents
            .iter()
            .all(|c| !matches!(c, Content::FunctionCall(_))),
        "final message was merged with the tool-call message"
    );
}

#[tokio::test]
async fn workflow_errors_on_max_iterations() {
    use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};

    // A single executor that sends to itself forever.
    let looper = FunctionExecutor::new("loop", |_msg, ctx| async move {
        ctx.send_message(json!(1)).await?;
        Ok(())
    });
    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(looper))
        .set_start("loop")
        .add_edge("loop", "loop")
        .set_max_iterations(5)
        .build()
        .unwrap();

    let result = workflow.run(json!(1)).await;
    assert!(
        result.is_err(),
        "expected a workflow error on iteration limit"
    );
}
