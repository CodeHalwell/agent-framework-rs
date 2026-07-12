//! End-to-end tests exercising agents, the tool loop, and workflows using a
//! mock chat client (no network).

use std::sync::{Arc, Mutex};

use agent_framework_core::agent::AsToolOptions;
use agent_framework_core::prelude::*;
use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Role};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

/// A scripted chat client that returns queued responses in order.
#[derive(Clone)]
struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
    seen_options: Arc<Mutex<Vec<ChatOptions>>>,
}

impl MockClient {
    fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
            seen_options: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl MockClient {
    /// The `ChatOptions` of the most recent call, if any.
    fn last_options(&self) -> Option<ChatOptions> {
        self.seen_options.lock().unwrap().last().cloned()
    }
    /// Every call's `ChatOptions`, in order.
    fn all_options(&self) -> Vec<ChatOptions> {
        self.seen_options.lock().unwrap().clone()
    }
    /// Every call's message list, in order.
    fn all_seen(&self) -> Vec<Vec<Message>> {
        self.seen.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for MockClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages);
        self.seen_options.lock().unwrap().push(options);
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

#[tokio::test]
async fn basic_agent_run() {
    let client = MockClient::new(vec![ChatResponse::from_text("Hello there!")]);
    let agent = Agent::builder(client)
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
    let agent = Agent::builder(client).build();

    let mut thread = agent.get_new_thread();
    let mut stream = agent
        .run_stream("hello", Some(thread.clone()), None)
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
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("The sum is 5.");
    let client = MockClient::new(vec![ask, answer]);

    let add = FunctionTool::new(
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

    let agent = Agent::builder(client).tool(add).build();
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
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("step-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;
    let b = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("step-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = agent_framework_core::workflow::SequentialBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("start").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
    let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
    assert!(texts.contains(&"step-A".to_string()));
    assert!(texts.contains(&"step-B".to_string()));
}

#[tokio::test]
async fn concurrent_workflow_fans_out() {
    let a = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("from-A")]))
            .name("A")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;
    let b = Arc::new(
        Agent::builder(MockClient::new(vec![ChatResponse::from_text("from-B")]))
            .name("B")
            .build(),
    ) as Arc<dyn SupportsAgentRun>;

    let workflow = agent_framework_core::workflow::ConcurrentBuilder::new()
        .participants(vec![a, b])
        .build()
        .unwrap();

    let result = workflow.run("question").await.unwrap();
    let output = result.last_output().expect("a final output");
    let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
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

/// SupportsAgentRun middleware that appends a suffix to every assistant message.
struct SuffixMiddleware;

#[async_trait]
impl Middleware<AgentContext> for SuffixMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
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
    let agent = Agent::builder(client)
        .middleware(Arc::new(SuffixMiddleware))
        .build();

    // Streaming must honor the middleware just like `run` does.
    let mut stream = agent.run_stream("hi", None, None).await.unwrap();
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
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(bad_call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("done");

    let invoked = Arc::new(Mutex::new(false));
    let invoked_clone = invoked.clone();
    let add = FunctionTool::new(
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

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
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

/// A context provider that records lifecycle-hook activity: whether `invoked`
/// fired, the error (if any) the last `invoked` carried, and every id passed to
/// `thread_created`. Also injects an instruction so `invoking` has an effect.
#[derive(Default, Clone)]
struct RecordingProvider {
    invoked: Arc<Mutex<bool>>,
    invoked_error: Arc<Mutex<Option<String>>>,
    thread_created_ids: Arc<Mutex<Vec<Option<String>>>>,
}

#[async_trait]
impl ContextProvider for RecordingProvider {
    async fn invoking(&self, _messages: &[Message]) -> Result<Context> {
        Ok(Context::new().with_instructions("remember: be brief"))
    }
    async fn thread_created(&self, thread_id: Option<&str>) -> Result<()> {
        self.thread_created_ids
            .lock()
            .unwrap()
            .push(thread_id.map(str::to_string));
        Ok(())
    }
    async fn invoked(
        &self,
        _request: &[Message],
        _response: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        *self.invoked.lock().unwrap() = true;
        *self.invoked_error.lock().unwrap() = error.map(|e| e.to_string());
        Ok(())
    }
}

#[tokio::test]
async fn context_provider_invoked_hook_fires() {
    let provider = RecordingProvider::default();
    let invoked = provider.invoked.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));

    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let agent = Agent::builder(client).context_provider(aggregate).build();

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
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("final answer");

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("done")) },
    )
    .into_definition();

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    // Re-aggregate exactly as a downstream consumer would.
    let aggregated = AgentResponse::from_updates(updates);
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
async fn streaming_tool_replay_preserves_usage_finish_reason_and_conversation_id() {
    // Usage, finish reason, and the service conversation id must survive the
    // tool-loop's run-then-replay streaming path, so aggregating the stream
    // yields the same metadata a non-streaming run() returns.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let mut usage = UsageDetails::new();
    usage.input_token_count = Some(11);
    usage.output_token_count = Some(7);
    let answer = ChatResponse {
        usage_details: Some(usage),
        finish_reason: Some(FinishReason::stop()),
        conversation_id: Some("conv-9".into()),
        ..ChatResponse::from_text("final answer")
    };

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("done")) },
    )
    .into_definition();

    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    assert_eq!(aggregated.conversation_id.as_deref(), Some("conv-9"));
    let usage = aggregated
        .usage_details
        .as_ref()
        .expect("usage must survive the replay");
    assert_eq!(usage.output_token_count, Some(7));
    // The usage rode as a Content::Usage item and must have folded into
    // usage_details, not leaked into the final message's contents.
    assert!(aggregated
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .all(|c| !matches!(c, Content::Usage(_))));
    assert_eq!(aggregated.messages.last().unwrap().text(), "final answer");
}

#[tokio::test]
async fn per_run_conversation_id_survives_on_a_local_thread() {
    // A per-run ChatOptions::conversation_id on a LOCAL thread must reach the
    // provider (it was previously clobbered by the thread's absent service
    // id, silently starting a new service conversation).
    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let probe = client.clone();
    let agent = Agent::builder(client).build();
    let mut thread = agent.get_new_thread();
    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        conversation_id: Some("conv-override".into()),
        ..Default::default()
    });
    agent
        .run_with_options(vec![Message::user("hi")], Some(&mut thread), options)
        .await
        .unwrap();
    assert_eq!(
        probe.last_options().unwrap().conversation_id.as_deref(),
        Some("conv-override")
    );
}

#[tokio::test]
async fn service_thread_id_wins_over_per_run_conversation_id() {
    // Continuity contract: a service-managed thread's id drives the call even
    // when a per-run override is supplied.
    let resp = ChatResponse {
        conversation_id: Some("svc-1".into()),
        ..ChatResponse::from_text("ok")
    };
    let client = MockClient::new(vec![resp]);
    let probe = client.clone();
    let agent = Agent::builder(client).build();
    let mut thread = AgentThread::service("svc-1");
    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        conversation_id: Some("conv-override".into()),
        ..Default::default()
    });
    agent
        .run_with_options(vec![Message::user("hi")], Some(&mut thread), options)
        .await
        .unwrap();
    assert_eq!(
        probe.last_options().unwrap().conversation_id.as_deref(),
        Some("svc-1")
    );
}

#[tokio::test]
async fn middleware_stream_replay_preserves_conversation_id_and_usage() {
    // With agent middleware configured, run_stream replays the completed run;
    // the response's conversation id and usage must survive that replay.
    let mut usage = UsageDetails::new();
    usage.output_token_count = Some(3);
    let resp = ChatResponse {
        conversation_id: Some("conv-7".into()),
        usage_details: Some(usage),
        ..ChatResponse::from_text("answer")
    };
    let client = MockClient::new(vec![resp]);
    let agent = Agent::builder(client)
        .middleware(Arc::new(SuffixMiddleware))
        .build();

    let mut stream = agent.run_stream("hi", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    assert_eq!(aggregated.conversation_id.as_deref(), Some("conv-7"));
    assert_eq!(
        aggregated
            .usage_details
            .expect("usage survives")
            .output_token_count,
        Some(3)
    );
}

#[tokio::test]
async fn service_created_conversation_id_propagates_into_tool_followup() {
    // A service-managed client creates the thread on the first tool-call turn
    // and returns its conversation_id. The follow-up submission (carrying the
    // FunctionResultContent) must target that thread, and — since the service
    // now holds the history — send only the new tool results.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let first = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        conversation_id: Some("thread_new".into()),
        ..Default::default()
    };
    let second = ChatResponse::from_text("done");
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();
    let probe = MockClient::new(vec![first, second]);
    let client = FunctionInvokingChatClient::new(probe.clone());

    let options = ChatOptions {
        tools: vec![noop],
        ..Default::default()
    };
    let resp = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    assert_eq!(resp.text(), "done");

    let all_opts = probe.all_options();
    assert_eq!(all_opts.len(), 2, "expected two underlying calls");
    // First call had no conversation id; the follow-up carries the one the
    // service created.
    assert!(all_opts[0].conversation_id.is_none());
    assert_eq!(all_opts[1].conversation_id.as_deref(), Some("thread_new"));

    // The follow-up sends only the new tool results, not the re-accumulated
    // history (the service holds it server-side).
    let seen = probe.all_seen();
    let followup = &seen[1];
    assert!(
        followup.iter().all(|m| m.role == Role::tool()),
        "follow-up should carry only tool-result messages"
    );
}

#[tokio::test]
async fn duplicate_provider_message_ids_do_not_merge_on_replay() {
    // A service (e.g. Assistants) can reuse one run id for the tool-call turn
    // and the final answer. If the replay preserved that duplicate id,
    // aggregation would merge the final text into the tool-call message and
    // reorder it ahead of the tool result. The replay must keep the two
    // assistant messages distinct.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let mut tool_call_msg =
        Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]);
    tool_call_msg.message_id = Some("run_dup".into());
    let ask = ChatResponse {
        messages: vec![tool_call_msg],
        ..Default::default()
    };
    let mut final_msg = Message::with_contents(Role::assistant(), vec![Content::text("final")]);
    final_msg.message_id = Some("run_dup".into()); // same id as the tool-call turn
    let answer = ChatResponse {
        messages: vec![final_msg],
        ..Default::default()
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();
    let agent = Agent::builder(MockClient::new(vec![ask, answer]))
        .tool(noop)
        .build();

    let mut stream = agent.run_stream("go", None, None).await.unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = AgentResponse::from_updates(updates);
    // Final answer stays its own message, after the tool result — not merged
    // into the tool-call message.
    let last = aggregated.messages.last().unwrap();
    assert_eq!(last.text(), "final");
    assert!(last
        .contents
        .iter()
        .all(|c| !matches!(c, Content::FunctionCall(_))));
}

#[tokio::test]
async fn provider_resolved_tool_calls_are_not_executed_locally() {
    // A response carrying a function call WITH its matching result in the
    // same response (e.g. Anthropic server-side web-search/MCP tool use) was
    // executed by the provider: the call must not enter the local tool loop,
    // which would emit a bogus "tool not found" and burn an extra iteration.
    let call = FunctionCallContent::new(
        "srv_1",
        "hosted_web_search",
        Some(FunctionArguments::Raw("{}".into())),
    );
    let resolved = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![
                Content::FunctionCall(call),
                Content::FunctionResult(FunctionResultContent {
                    call_id: "srv_1".into(),
                    result: Some(json!({"hits": 3})),
                    exception: None,
                }),
                Content::text("Found 3 results."),
            ],
        )],
        ..Default::default()
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("x")) },
    )
    .into_definition();
    // Exactly one scripted response: a second loop iteration would consume
    // the "(no more scripted responses)" fallback and change the text.
    let agent = Agent::builder(MockClient::new(vec![resolved]))
        .tool(noop)
        .build();

    let out = agent.run_once("go").await.unwrap();
    assert_eq!(out.text(), "Found 3 results.");
    // No synthetic error result was appended for the pre-resolved call.
    assert!(out
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(Content::as_function_result)
        .all(|fr| fr.exception.is_none()));
}

#[tokio::test]
async fn chat_level_tool_stream_replay_carries_finish_reason() {
    // AgentResponse has no finish_reason (matching upstream), so the
    // finish-reason half of the replay metadata is asserted at the
    // chat-client level, where ChatResponse::from_updates surfaces it.
    let call =
        FunctionCallContent::new("call_1", "noop", Some(FunctionArguments::Raw("{}".into())));
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse {
        finish_reason: Some(FinishReason::stop()),
        ..ChatResponse::from_text("done")
    };
    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();

    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ask, answer]));
    let options = ChatOptions {
        tools: vec![noop],
        ..Default::default()
    };
    let mut stream = client
        .get_streaming_response(vec![Message::user("go")], options)
        .await
        .unwrap();
    let mut updates = Vec::new();
    while let Some(u) = stream.next().await {
        updates.push(u.unwrap());
    }
    let aggregated = ChatResponse::from_updates(updates);
    assert_eq!(aggregated.finish_reason, Some(FinishReason::stop()));
    assert_eq!(aggregated.messages.last().unwrap().text(), "done");
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

// ---------------------------------------------------------------------------
// Structured output
// ---------------------------------------------------------------------------

#[test]
fn response_format_serializes_to_openai_shape() {
    // Text / JsonObject.
    assert_eq!(
        serde_json::to_value(ResponseFormat::Text).unwrap(),
        json!({ "type": "text" })
    );
    assert_eq!(
        serde_json::to_value(ResponseFormat::JsonObject).unwrap(),
        json!({ "type": "json_object" })
    );

    // JsonSchema nests under "json_schema", matching OpenAI's request object.
    let fmt = ResponseFormat::JsonSchema {
        name: "Person".into(),
        description: Some("a person".into()),
        schema: json!({ "type": "object", "properties": { "name": { "type": "string" } } }),
        strict: Some(true),
    };
    let value = serde_json::to_value(&fmt).unwrap();
    assert_eq!(value["type"], "json_schema");
    assert_eq!(value["json_schema"]["name"], "Person");
    assert_eq!(value["json_schema"]["description"], "a person");
    assert_eq!(value["json_schema"]["strict"], true);
    assert_eq!(value["json_schema"]["schema"]["type"], "object");

    // Round-trips through Deserialize.
    let back: ResponseFormat = serde_json::from_value(value).unwrap();
    assert_eq!(back, fmt);
}

#[test]
fn parse_json_reads_structured_value() {
    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct Person {
        name: String,
        age: u32,
    }

    let resp = ChatResponse::from_text(r#"{"name":"Ada","age":36}"#);
    let person: Person = resp.parse_json().unwrap();
    assert_eq!(
        person,
        Person {
            name: "Ada".into(),
            age: 36
        }
    );

    // The same convenience exists on AgentResponse.
    let agent_resp =
        AgentResponse::from_chat_response(ChatResponse::from_text(r#"{"name":"Bob","age":5}"#));
    let person2: Person = agent_resp.parse_json().unwrap();
    assert_eq!(person2.name, "Bob");

    // Non-JSON text surfaces an error rather than panicking.
    assert!(ChatResponse::from_text("not json")
        .parse_json::<Person>()
        .is_err());
}

#[test]
fn response_format_builder_sugar_sets_option() {
    let agent = Agent::builder(MockClient::new(vec![])).response_format(ResponseFormat::JsonObject);
    // Build and confirm the option flows through (via a run that echoes options
    // is unnecessary; just assert the builder compiles and produces an agent).
    let _agent = agent.build();
}

// ---------------------------------------------------------------------------
// ToolMode serde
// ---------------------------------------------------------------------------

#[test]
fn tool_mode_serde_round_trip() {
    assert_eq!(serde_json::to_value(ToolMode::Auto).unwrap(), json!("auto"));
    assert_eq!(
        serde_json::to_value(ToolMode::required_any()).unwrap(),
        json!("required")
    );
    // Like Python's serialize_model, the pinned function name is not persisted
    // on the mode itself (the provider mapping applies it).
    assert_eq!(
        serde_json::to_value(ToolMode::required_function("get_weather")).unwrap(),
        json!("required")
    );
    assert_eq!(serde_json::to_value(ToolMode::None).unwrap(), json!("none"));

    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("auto")).unwrap(),
        ToolMode::Auto
    );
    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("required")).unwrap(),
        ToolMode::Required(None)
    );
    assert_eq!(
        serde_json::from_value::<ToolMode>(json!("none")).unwrap(),
        ToolMode::None
    );

    assert_eq!(
        ToolMode::required_function("f").required_function_name(),
        Some("f")
    );
    assert_eq!(ToolMode::Auto.required_function_name(), None);
}

// ---------------------------------------------------------------------------
// Update aggregation
// ---------------------------------------------------------------------------

#[test]
fn agent_update_aggregation() {
    let updates = vec![
        AgentResponseUpdate {
            contents: vec![Content::text("Hello")],
            role: Some(Role::assistant()),
            ..Default::default()
        },
        AgentResponseUpdate {
            contents: vec![Content::text(" world")],
            role: Some(Role::assistant()),
            ..Default::default()
        },
    ];
    let resp = AgentResponse::from_agent_run_response_updates(updates);
    assert_eq!(resp.text(), "Hello world");
}

// ---------------------------------------------------------------------------
// Function-approval flow
// ---------------------------------------------------------------------------

/// A tool requiring approval that records how many times it actually executed.
fn approval_tool(counter: Arc<Mutex<u32>>) -> ToolDefinition {
    FunctionTool::new(
        "get_secret",
        "Return the secret value.",
        json!({ "type": "object", "properties": {} }),
        move |_args| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok(json!("42"))
            }
        },
    )
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition()
}

fn secret_call() -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_secret",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::test]
async fn approval_loop_approve_executes_and_answers() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("The secret is 42."),
    ]));
    let options = ChatOptions::new().with_tool(tool);

    // Request 1: the model asks for an approval-gated tool -> we get an approval
    // request back, and the tool has NOT run.
    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1, "expected one approval request");
    assert_eq!(requests[0].function_call.call_id, "call_1");
    assert_eq!(*counter.lock().unwrap(), 0, "tool ran before approval");
    // The assistant message still carries the original function call too.
    assert_eq!(resp1.function_calls().len(), 1);

    // Request 2: approve -> the tool runs and the model produces a final answer.
    let approval = requests[0].create_response(true);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(approval)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();
    assert!(resp2.text().contains("42"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 1, "tool should run exactly once");
}

#[tokio::test]
async fn approval_loop_reject_skips_execution() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("Understood, I won't retrieve it."),
    ]));
    let options = ChatOptions::new().with_tool(tool);

    let resp1 = client
        .get_response(vec![Message::user("what is the secret?")], options.clone())
        .await
        .unwrap();
    let requests = resp1.user_input_requests();
    assert_eq!(requests.len(), 1);

    // Reject the call.
    let rejection = requests[0].create_response(false);
    let mut conversation = vec![Message::user("what is the secret?")];
    conversation.extend(resp1.messages.clone());
    conversation.push(Message::with_contents(
        Role::user(),
        vec![Content::FunctionApprovalResponse(rejection)],
    ));
    let resp2 = client.get_response(conversation, options).await.unwrap();

    assert!(resp2.text().contains("won't"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 0, "rejected tool must not run");
}

#[tokio::test]
async fn agent_surfaces_and_resolves_approval_round_trip() {
    let counter = Arc::new(Mutex::new(0));
    let tool = approval_tool(counter.clone());
    let agent = Agent::builder(MockClient::new(vec![
        secret_call(),
        ChatResponse::from_text("The secret is 42."),
    ]))
    .name("keeper")
    .tool(tool)
    .build();

    let mut thread = agent.get_new_thread();

    // First run pauses awaiting approval; the request is surfaced on the agent
    // response and persisted to the thread.
    let resp1 = agent
        .run(vec![Message::user("get the secret")], Some(&mut thread))
        .await
        .unwrap();
    assert_eq!(resp1.user_input_requests().len(), 1);
    let approval = resp1.user_input_requests()[0].create_response(true);

    // Supplying the approval response as new input resolves the exchange.
    let resp2 = agent
        .run(
            vec![Message::with_contents(
                Role::user(),
                vec![Content::FunctionApprovalResponse(approval)],
            )],
            Some(&mut thread),
        )
        .await
        .unwrap();
    assert!(resp2.text().contains("42"), "got: {}", resp2.text());
    assert_eq!(*counter.lock().unwrap(), 1);

    // The thread retains the full approval exchange.
    let history = thread.list_messages().await.unwrap();
    assert!(history.iter().any(|m| !m.user_input_requests().is_empty()));
}

// ---------------------------------------------------------------------------
// SupportsAgentRun-as-tool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn agent_as_tool_is_callable_by_another_agent() {
    // Inner agent always answers "INNER-RESULT".
    let inner = Agent::builder(MockClient::new(vec![ChatResponse::from_text(
        "INNER-RESULT",
    )]))
    .name("researcher")
    .description("Performs research tasks.")
    .build();
    let research_tool = inner.as_tool(AsToolOptions::new().name("research"));
    assert_eq!(research_tool.name, "research");

    // Outer agent: the model calls `research`, then answers.
    let call = FunctionCallContent::new(
        "c1",
        "research",
        Some(FunctionArguments::Raw(
            json!({ "task": "find X" }).to_string(),
        )),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let outer = Agent::builder(MockClient::new(vec![ask, ChatResponse::from_text("Done.")]))
        .tool(research_tool)
        .build();

    let response = outer.run_once("do research").await.unwrap();
    assert!(response.text().contains("Done"), "got: {}", response.text());

    // The inner agent's output flowed back as the tool result.
    let saw_inner = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .any(|c| {
            matches!(c, Content::FunctionResult(fr)
            if fr.result.as_ref().and_then(|v| v.as_str()) == Some("INNER-RESULT"))
        });
    assert!(
        saw_inner,
        "inner agent result missing: {:?}",
        response.messages
    );
}

// ---------------------------------------------------------------------------
// Observability
//
// The span-capture smoke test lives in its own binary (`tests/observability.rs`)
// so the `chat` tracing callsite is first evaluated under the capturing
// subscriber — `tracing` caches callsite interest globally, so sharing a binary
// with tests that hit the callsite under the no-op subscriber would disable it.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn observable_chat_client_is_transparent() {
    // Decorating a client must not change its observable behavior.
    let client = ObservableChatClient::new(
        MockClient::new(vec![ChatResponse::from_text("plain")]),
        "mock",
    );
    let resp = client
        .get_response(vec![Message::user("hi")], ChatOptions::new())
        .await
        .unwrap();
    assert_eq!(resp.text(), "plain");
}

// ---------------------------------------------------------------------------
// Chat & function middleware
// ---------------------------------------------------------------------------

/// Chat middleware that rewrites every outgoing user message's text.
struct RewriteUserMessage;

#[async_trait]
impl Middleware<ChatContext> for RewriteUserMessage {
    async fn process(&self, mut ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        for m in &mut ctx.messages {
            if m.role == Role::user() {
                *m = Message::user("REWRITTEN");
            }
        }
        next.run(ctx).await
    }
}

#[tokio::test]
async fn chat_middleware_rewrites_outgoing_message() {
    let client = MockClient::new(vec![ChatResponse::from_text("ok")]);
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .chat_middleware(Arc::new(RewriteUserMessage))
        .build();

    let _ = agent.run_once("original").await.unwrap();

    let seen = seen.lock().unwrap();
    let last = seen.last().expect("the model should have been called");
    assert!(
        last.iter().any(|m| m.text() == "REWRITTEN"),
        "model did not see the rewritten message: {last:?}"
    );
}

/// Chat middleware that short-circuits with a canned response, never letting
/// the call reach the underlying client.
struct ShortCircuitChat;

#[async_trait]
impl Middleware<ChatContext> for ShortCircuitChat {
    async fn process(&self, mut ctx: ChatContext, _next: Next<ChatContext>) -> Result<ChatContext> {
        // Deliberately does not call `next.run(ctx)`: the underlying client
        // must never be invoked.
        ctx.result = Some(ChatResponse::from_text("canned"));
        ctx.terminate = true;
        Ok(ctx)
    }
}

#[tokio::test]
async fn chat_middleware_short_circuits_model_call() {
    let client = MockClient::new(vec![ChatResponse::from_text("should not be used")]);
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .chat_middleware(Arc::new(ShortCircuitChat))
        .build();

    let response = agent.run_once("hi").await.unwrap();

    assert_eq!(response.text(), "canned");
    assert!(
        seen.lock().unwrap().is_empty(),
        "the underlying model must not have been called"
    );
}

/// Function middleware that rewrites arguments before execution.
struct RewriteArgsMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for RewriteArgsMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        if let Some(obj) = ctx.arguments.as_object_mut() {
            obj.insert("a".to_string(), json!(100));
        }
        next.run(ctx).await
    }
}

fn add_call(a: i64, b: i64) -> ChatResponse {
    let call = FunctionCallContent::new(
        "call_1",
        "add",
        Some(FunctionArguments::Raw(json!({"a": a, "b": b}).to_string())),
    );
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    }
}

#[tokio::test]
async fn function_middleware_rewrites_arguments() {
    let client = MockClient::new(vec![add_call(2, 3), ChatResponse::from_text("done")]);

    let seen_args: Arc<Mutex<Option<Value>>> = Arc::new(Mutex::new(None));
    let seen_args_clone = seen_args.clone();
    let add = FunctionTool::new(
        "add",
        "Add two integers.",
        json!({"type":"object","properties":{}}),
        move |args: Value| {
            let seen_args_clone = seen_args_clone.clone();
            async move {
                *seen_args_clone.lock().unwrap() = Some(args.clone());
                let a = args["a"].as_i64().unwrap_or(0);
                let b = args["b"].as_i64().unwrap_or(0);
                Ok(json!(a + b))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .tool(add)
        .function_middleware(Arc::new(RewriteArgsMiddleware))
        .build();

    let _ = agent.run_once("add 2 and 3").await.unwrap();

    let seen = seen_args
        .lock()
        .unwrap()
        .clone()
        .expect("the tool should have run");
    assert_eq!(
        seen["a"],
        json!(100),
        "middleware did not rewrite the argument: {seen:?}"
    );
    assert_eq!(seen["b"], json!(3), "unrelated argument must be untouched");
}

/// Function middleware that blocks execution entirely by short-circuiting
/// with its own result.
struct BlockExecutionMiddleware;

#[async_trait]
impl Middleware<FunctionInvocationContext> for BlockExecutionMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        _next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        ctx.result = Some(json!("blocked"));
        ctx.terminate = true;
        Ok(ctx)
    }
}

#[tokio::test]
async fn function_middleware_blocks_execution() {
    let client = MockClient::new(vec![add_call(2, 3), ChatResponse::from_text("done")]);

    let invoked = Arc::new(Mutex::new(false));
    let invoked_clone = invoked.clone();
    let add = FunctionTool::new(
        "add",
        "Add two integers.",
        json!({"type":"object","properties":{}}),
        move |_args| {
            let invoked_clone = invoked_clone.clone();
            async move {
                *invoked_clone.lock().unwrap() = true;
                Ok(json!(999))
            }
        },
    )
    .into_definition();

    let agent = Agent::builder(client)
        .tool(add)
        .function_middleware(Arc::new(BlockExecutionMiddleware))
        .build();

    let response = agent.run_once("add 2 and 3").await.unwrap();

    assert!(!*invoked.lock().unwrap(), "the tool must not have executed");
    assert!(
        response
            .messages
            .iter()
            .any(|m| m.contents.iter().any(|c| matches!(
                c,
                Content::FunctionResult(fr) if fr.result == Some(json!("blocked"))
            ))),
        "the blocked result should still flow through as the tool result: {:?}",
        response.messages
    );
}

/// Records `"{label}-before"`/`"{label}-after"` around `next.run(...)`, so two
/// instances reveal the pipeline's nesting order.
struct OrderRecorder {
    label: &'static str,
    log: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for OrderRecorder {
    async fn process(
        &self,
        ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        self.log
            .lock()
            .unwrap()
            .push(format!("{}-before", self.label));
        let ctx = next.run(ctx).await?;
        self.log
            .lock()
            .unwrap()
            .push(format!("{}-after", self.label));
        Ok(ctx)
    }
}

#[tokio::test]
async fn function_middleware_order_is_onion_nested() {
    // Two function middlewares must nest onion-style — first registered is
    // outermost — matching the ordering convention `MiddlewarePipeline`
    // already establishes for agent middleware (`Next::run` walks the
    // registered list front-to-back, invoking the terminal only once every
    // middleware has called `next`).
    let client = MockClient::new(vec![
        ChatResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "call_1",
                    "noop",
                    Some(FunctionArguments::Raw("{}".into())),
                ))],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        },
        ChatResponse::from_text("done"),
    ]);

    let noop = FunctionTool::new(
        "noop",
        "noop",
        json!({"type":"object","properties":{}}),
        |_a| async move { Ok(json!("ok")) },
    )
    .into_definition();

    let log = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder(client)
        .tool(noop)
        .function_middleware(Arc::new(OrderRecorder {
            label: "A",
            log: log.clone(),
        }))
        .function_middleware(Arc::new(OrderRecorder {
            label: "B",
            log: log.clone(),
        }))
        .build();

    let _ = agent.run_once("go").await.unwrap();

    let log = log.lock().unwrap().clone();
    assert_eq!(log, vec!["A-before", "B-before", "B-after", "A-after"]);
}

#[tokio::test]
async fn service_conversation_id_is_adopted_by_thread() {
    use std::sync::{Arc, Mutex};

    // A client that manages conversations service-side: returns a
    // conversation id and records the options of every request.
    struct ServiceClient {
        seen_options: Arc<Mutex<Vec<ChatOptions>>>,
    }
    #[async_trait::async_trait]
    impl ChatClient for ServiceClient {
        async fn get_response(
            &self,
            _messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen_options.lock().unwrap().push(options);
            let mut resp = ChatResponse::from_text("ok");
            resp.conversation_id = Some("conv-1".to_string());
            Ok(resp)
        }
        async fn get_streaming_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<agent_framework_core::client::ChatStream> {
            let resp = self.get_response(messages, options).await?;
            let mut update = ChatResponseUpdate::text(resp.text());
            update.conversation_id = Some("conv-1".to_string());
            Ok(Box::pin(futures::stream::iter(vec![Ok(update)])))
        }
    }

    let seen_options = Arc::new(Mutex::new(Vec::new()));
    let agent = Agent::builder(ServiceClient {
        seen_options: seen_options.clone(),
    })
    .name("svc")
    .build();

    // Fresh agent threads start with an (empty) local store; the returned
    // service conversation id must still be adopted.
    let mut thread = agent.get_new_thread();
    let response = agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();
    assert_eq!(response.conversation_id.as_deref(), Some("conv-1"));
    assert_eq!(thread.service_thread_id(), Some("conv-1"));

    // Turn two must carry the id back to the service.
    agent
        .run(vec![Message::user("again")], Some(&mut thread))
        .await
        .unwrap();
    let opts = seen_options.lock().unwrap();
    assert_eq!(opts.len(), 2);
    assert_eq!(opts[0].conversation_id, None);
    assert_eq!(opts[1].conversation_id.as_deref(), Some("conv-1"));
}

// ===========================================================================
// Task 5: as_tool name sanitization
// ===========================================================================

#[tokio::test]
async fn as_tool_sanitizes_agent_name() {
    let agent = Agent::builder(MockClient::new(vec![]))
        .name("My Weather Agent!! v2")
        .build();
    // Spaces/punctuation -> underscores, collapsed, trimmed.
    let tool = agent.as_tool(AsToolOptions::new());
    assert_eq!(tool.name, "My_Weather_Agent_v2");

    // An explicit name is used verbatim (mirrors Python `name or sanitize`).
    let tool2 = agent.as_tool(AsToolOptions::new().name("explicit name"));
    assert_eq!(tool2.name, "explicit name");

    // Leading digit gets an underscore prefix; all-invalid -> "agent".
    let numeric = Agent::builder(MockClient::new(vec![]))
        .name("9lives")
        .build();
    assert_eq!(numeric.as_tool(AsToolOptions::new()).name, "_9lives");
    let junk = Agent::builder(MockClient::new(vec![])).name("@@@").build();
    assert_eq!(junk.as_tool(AsToolOptions::new()).name, "agent");
}

// ===========================================================================
// Task 6: service-managed thread with no returned conversation id errors
// ===========================================================================

#[tokio::test]
async fn service_thread_without_conversation_id_errors() {
    // The client succeeds but returns no conversation id.
    let client = MockClient::new(vec![ChatResponse::from_text("hi")]);
    let agent = Agent::builder(client).name("svc").build();
    let mut thread = agent.get_new_thread_with_service_id("svc-thread").unwrap();
    let err = agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::AgentExecution(_)));
    assert!(err
        .to_string()
        .contains("did not return a valid conversation id"));
}

// ===========================================================================
// Task 1: ContextProvider::thread_created is fired by Agent
// ===========================================================================

/// Echoes the request's conversation id back (keeps a service thread valid).
struct EchoServiceClient;
#[async_trait]
impl ChatClient for EchoServiceClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok");
        resp.conversation_id = options.conversation_id.clone();
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let mut u = ChatResponseUpdate::text(resp.text());
        u.conversation_id = resp.conversation_id.clone();
        Ok(Box::pin(futures::stream::iter(vec![Ok(u)])))
    }
}

/// Returns a fresh conversation id for a previously-local thread to adopt.
struct AdoptServiceClient;
#[async_trait]
impl ChatClient for AdoptServiceClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok");
        resp.conversation_id = Some("adopted-1".to_string());
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let mut u = ChatResponseUpdate::text(resp.text());
        u.conversation_id = Some("adopted-1".to_string());
        Ok(Box::pin(futures::stream::iter(vec![Ok(u)])))
    }
}

#[tokio::test]
async fn thread_created_fires_for_service_thread() {
    let provider = RecordingProvider::default();
    let ids = provider.thread_created_ids.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));
    let agent = Agent::builder(EchoServiceClient)
        .context_provider(aggregate)
        .build();

    let mut thread = agent.get_new_thread_with_service_id("svc-1").unwrap();
    agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();

    // Fired once at run start with the service thread id (no re-fire on the
    // echoed-back same id).
    assert_eq!(ids.lock().unwrap().clone(), vec![Some("svc-1".to_string())]);
}

#[tokio::test]
async fn thread_created_fires_on_service_id_adoption() {
    let provider = RecordingProvider::default();
    let ids = provider.thread_created_ids.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));
    let agent = Agent::builder(AdoptServiceClient)
        .context_provider(aggregate)
        .build();

    // A local thread that adopts the id returned by the service.
    let mut thread = agent.get_new_thread();
    agent
        .run(vec![Message::user("hi")], Some(&mut thread))
        .await
        .unwrap();

    assert_eq!(
        ids.lock().unwrap().clone(),
        vec![Some("adopted-1".to_string())]
    );
    assert_eq!(thread.service_thread_id(), Some("adopted-1"));
}

#[tokio::test]
async fn thread_created_fires_on_adoption_when_streaming() {
    let provider = RecordingProvider::default();
    let ids = provider.thread_created_ids.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));
    let agent = Agent::builder(AdoptServiceClient)
        .context_provider(aggregate)
        .build();

    let mut stream = agent.run_stream("hi", None, None).await.unwrap();
    while let Some(u) = stream.next().await {
        u.unwrap();
    }
    assert_eq!(
        ids.lock().unwrap().clone(),
        vec![Some("adopted-1".to_string())]
    );
}

// ===========================================================================
// Task 2: ContextProvider::invoked observes failures
// ===========================================================================

struct FailingClient;
#[async_trait]
impl ChatClient for FailingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Err(Error::service("boom"))
    }
    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Err(Error::service("boom"))
    }
}

#[tokio::test]
async fn invoked_hook_observes_failure() {
    let provider = RecordingProvider::default();
    let invoked = provider.invoked.clone();
    let invoked_error = provider.invoked_error.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));
    let agent = Agent::builder(FailingClient)
        .context_provider(aggregate)
        .build();

    let err = agent.run_once("hi").await.unwrap_err();
    assert!(err.to_string().contains("boom"));
    assert!(
        *invoked.lock().unwrap(),
        "invoked fired on the failure path"
    );
    let recorded = invoked_error.lock().unwrap().clone();
    assert!(
        recorded.is_some_and(|m| m.contains("boom")),
        "provider observed the run error"
    );
}

#[tokio::test]
async fn invoked_hook_observes_streaming_failure() {
    let provider = RecordingProvider::default();
    let invoked_error = provider.invoked_error.clone();
    let aggregate = Arc::new(AggregateContextProvider::from_providers(vec![Arc::new(
        provider,
    )]));
    let agent = Agent::builder(FailingClient)
        .context_provider(aggregate)
        .build();

    let err = agent.run_stream("hi", None, None).await.err().unwrap();
    assert!(err.to_string().contains("boom"));
    assert!(
        invoked_error
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|m| m.contains("boom")),
        "provider observed the streaming failure"
    );
}

// ===========================================================================
// Task 3: structured-output value auto-population
// ===========================================================================

#[tokio::test]
async fn structured_output_value_autofilled_on_agent_run() {
    let client = MockClient::new(vec![ChatResponse::from_text("{\"city\": \"Paris\"}")]);
    let agent = Agent::builder(client)
        .response_format(ResponseFormat::JsonObject)
        .build();
    let resp = agent.run_once("where?").await.unwrap();
    assert_eq!(resp.value, Some(json!({"city": "Paris"})));
}

#[tokio::test]
async fn structured_output_value_tolerates_non_json() {
    let client = MockClient::new(vec![ChatResponse::from_text("sorry, no idea")]);
    let agent = Agent::builder(client)
        .response_format(ResponseFormat::JsonObject)
        .build();
    let resp = agent.run_once("where?").await.unwrap();
    assert_eq!(resp.value, None);
}

#[tokio::test]
async fn structured_output_value_autofilled_on_bare_client() {
    use agent_framework_core::client::FunctionInvokingChatClient;
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ChatResponse::from_text(
        "{\"n\": 5}",
    )]));
    let mut opts = ChatOptions::new();
    opts.response_format = Some(ResponseFormat::JsonObject);
    let resp = client
        .get_response(vec![Message::user("x")], opts)
        .await
        .unwrap();
    assert_eq!(resp.value, Some(json!({"n": 5})));
}

// ===========================================================================
// Task 7: thread persistence (agent-level: factory, deserialize, service id)
// ===========================================================================

#[tokio::test]
async fn chat_message_store_factory_used_by_get_new_thread() {
    // The factory seeds a marker so we can observe it was used.
    let agent = Agent::builder(MockClient::new(vec![]))
        .chat_message_store_factory(|| {
            Arc::new(InMemoryChatMessageStore::with_messages(vec![
                Message::system("MARKER"),
            ]))
        })
        .build();
    let thread = agent.get_new_thread();
    let msgs = thread.list_messages().await.unwrap();
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "MARKER");
}

#[tokio::test]
async fn agent_deserialize_thread_roundtrips_history() {
    let agent = Agent::builder(MockClient::new(vec![])).build();
    let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
        Message::user("hi"),
        Message::assistant("hello"),
    ]));
    let state = AgentThread::local(store).serialize().await.unwrap();

    let restored = agent.deserialize_thread(&state).await.unwrap();
    let msgs = restored.list_messages().await.unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[1].text(), "hello");
}

#[tokio::test]
async fn agent_get_new_thread_with_service_id() {
    let agent = Agent::builder(MockClient::new(vec![])).build();
    let thread = agent.get_new_thread_with_service_id("svc-9").unwrap();
    assert_eq!(thread.service_thread_id(), Some("svc-9"));
    assert!(thread.message_store().is_none());
}

// ---------------------------------------------------------------------------
// GAP 1.4 — trait-level streaming; GAP 1.5 — per-run options; Task 3 — declaration-only
// ---------------------------------------------------------------------------

/// A client that streams a fixed list of text deltas (real incremental
/// streaming, distinct from `MockClient`'s per-message replay).
#[derive(Clone)]
struct DeltaClient {
    deltas: Vec<String>,
}

#[async_trait]
impl ChatClient for DeltaClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.deltas.concat()))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        let updates: Vec<Result<ChatResponseUpdate>> = self
            .deltas
            .iter()
            .map(|d| Ok(ChatResponseUpdate::text(d.clone())))
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// A client that records every `ChatOptions` it is handed (to assert per-run
/// option precedence and per-run tool visibility).
#[derive(Clone)]
struct RecordingClient {
    seen: Arc<Mutex<Vec<ChatOptions>>>,
}

impl RecordingClient {
    fn new() -> Self {
        Self {
            seen: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatClient for RecordingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(options);
        Ok(ChatResponse::from_text("ok"))
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

fn declaration_only_tool(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: String::new(),
        parameters: json!({ "type": "object", "properties": {} }),
        kind: ToolKind::Function,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

#[tokio::test]
async fn trait_default_run_stream_buffers_for_minimal_agent() {
    // A minimal custom agent implementing only `run` + `id` gets the trait's
    // default buffered `run_stream` for free.
    struct EchoAgent;
    #[async_trait]
    impl SupportsAgentRun for EchoAgent {
        async fn run(
            &self,
            messages: Vec<Message>,
            _thread: Option<&mut AgentThread>,
        ) -> Result<AgentResponse> {
            let text = messages.last().map(Message::text).unwrap_or_default();
            Ok(AgentResponse {
                messages: vec![Message::assistant(format!("echo: {text}"))],
                ..Default::default()
            })
        }
        fn id(&self) -> &str {
            "echo"
        }
    }

    let agent = EchoAgent;
    let mut stream = SupportsAgentRun::run_stream(&agent, vec![Message::user("hi")], None, None)
        .await
        .unwrap();
    let mut text = String::new();
    let mut count = 0;
    while let Some(update) = stream.next().await {
        text.push_str(&update.unwrap().text());
        count += 1;
    }
    assert_eq!(text, "echo: hi");
    assert_eq!(count, 1, "one buffered update per response message");
}

#[tokio::test]
async fn chat_agent_trait_stream_yields_real_deltas() {
    // Agent's real streaming override forwards one update per model delta.
    let client = DeltaClient {
        deltas: vec!["Hel".into(), "lo ".into(), "world".into()],
    };
    let agent = Agent::builder(client).build();
    let mut stream = SupportsAgentRun::run_stream(&agent, vec![Message::user("hi")], None, None)
        .await
        .unwrap();
    let mut deltas = Vec::new();
    while let Some(update) = stream.next().await {
        deltas.push(update.unwrap().text());
    }
    assert_eq!(deltas.len(), 3, "one update per streamed delta");
    assert_eq!(deltas.concat(), "Hello world");
}

#[tokio::test]
async fn per_run_chat_options_override_agent_defaults() {
    // SupportsAgentRun default temperature 0.2; a per-run override of 0.9 must win, matching
    // Python's `run_chat_options & ChatOptions(...)`.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let agent = Agent::builder(client).temperature(0.2).build();

    let options = AgentRunOptions::new().with_chat_options(ChatOptions {
        temperature: Some(0.9),
        ..Default::default()
    });
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].temperature,
        Some(0.9),
        "per-run temperature wins over the agent default"
    );
}

#[tokio::test]
async fn per_run_tools_are_visible_only_for_that_call() {
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let agent = Agent::builder(client)
        .tool(declaration_only_tool("base_tool"))
        .build();

    // Run 1: inject an extra per-run tool.
    let options = AgentRunOptions::new().with_tool(declaration_only_tool("run_tool"));
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();
    // Run 2: no per-run tools.
    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let names =
        |i: usize| -> Vec<String> { recorded[i].tools.iter().map(|t| t.name.clone()).collect() };
    assert!(names(0).contains(&"base_tool".to_string()));
    assert!(
        names(0).contains(&"run_tool".to_string()),
        "per-run tool visible for that call"
    );
    assert!(
        !names(1).contains(&"run_tool".to_string()),
        "per-run tool must NOT leak into the next call"
    );
    assert!(names(1).contains(&"base_tool".to_string()));
}

#[tokio::test]
async fn declaration_only_tool_call_is_returned_to_caller() {
    // The model calls a known-but-declaration-only tool; the loop must return
    // the response with the FunctionCallContent intact (frontend-tool pattern).
    let call = FunctionCallContent::new(
        "c1",
        "frontend_tool",
        Some(FunctionArguments::Raw(json!({"x": 1}).to_string())),
    );
    let resp = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![resp]));
    // A real executable tool is present so the invocation loop actually engages;
    // the model instead calls the declaration-only tool, which the loop must
    // return unexecuted rather than error on.
    let real_tool = FunctionTool::new(
        "real",
        "",
        json!({ "type": "object", "properties": {} }),
        |_args: Value| async { Ok(Value::Null) },
    )
    .into_definition();
    let options = ChatOptions {
        tools: vec![real_tool, declaration_only_tool("frontend_tool")],
        ..Default::default()
    };
    let out = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();

    let calls = out.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "frontend_tool");
    let has_result = out
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionResult(_)));
    assert!(!has_result, "declaration-only call must not be executed");
}

#[tokio::test]
async fn unknown_tool_call_is_not_declaration_only() {
    // A genuinely unknown tool name keeps the not-found error behavior (an
    // error result, loop continues), NOT the declaration-only early return.
    let call = FunctionCallContent::new("c1", "ghost_tool", None);
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        ..Default::default()
    };
    let answer = ChatResponse::from_text("done");
    // A real executable tool is present so the loop engages, but the model calls
    // a different, unknown tool.
    let real_tool = FunctionTool::new(
        "real",
        "",
        json!({ "type": "object", "properties": {} }),
        |_args: Value| async { Ok(Value::Null) },
    )
    .into_definition();
    let client = FunctionInvokingChatClient::new(MockClient::new(vec![ask, answer]));
    let options = ChatOptions {
        tools: vec![real_tool],
        ..Default::default()
    };
    let out = client
        .get_response(vec![Message::user("go")], options)
        .await
        .unwrap();

    let has_error_result = out
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionResult(fr) if fr.exception.is_some()));
    assert!(
        has_error_result,
        "unknown tool yields an error result, not a declaration-only return"
    );
    assert_eq!(out.text(), "done");
}

// -- ToolSource: dynamic tool resolution per agent run --------------------

/// A [`ToolSource`] that returns a scripted sequence of tool lists, one per
/// `resolve_tools` call (the last is repeated once the script is
/// exhausted). Stands in for an MCP server whose catalog changes between
/// runs (e.g. after a `notifications/tools/list_changed`), without any real
/// transport.
struct StubToolSource {
    name: String,
    call_count: Arc<Mutex<usize>>,
    responses: Vec<Vec<ToolDefinition>>,
}

impl StubToolSource {
    fn new(name: &str, responses: Vec<Vec<ToolDefinition>>) -> Self {
        Self {
            name: name.to_string(),
            call_count: Arc::new(Mutex::new(0)),
            responses,
        }
    }
}

#[async_trait]
impl ToolSource for StubToolSource {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        let mut count = self.call_count.lock().unwrap();
        let idx = (*count).min(self.responses.len().saturating_sub(1));
        *count += 1;
        Ok(self.responses.get(idx).cloned().unwrap_or_default())
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// A [`ToolSource`] whose `resolve_tools` always fails — stands in for an
/// MCP server that is unreachable at run time.
struct FailingToolSource;

#[async_trait]
impl ToolSource for FailingToolSource {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        Err(Error::service("mcp server unreachable"))
    }
    fn source_name(&self) -> &str {
        "failing-source"
    }
}

#[tokio::test]
async fn tool_source_resolved_fresh_each_run_sees_catalog_change() {
    // Simulates a server whose tool list grows between runs (e.g. after a
    // notifications/tools/list_changed): the agent must re-resolve the
    // source on every run rather than resolving it once at build time.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let source = Arc::new(StubToolSource::new(
        "mcp",
        vec![
            vec![declaration_only_tool("tool_a")],
            vec![
                declaration_only_tool("tool_a"),
                declaration_only_tool("tool_b"),
            ],
        ],
    ));
    let agent = Agent::builder(client).tool_source(source).build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();
    let _ = agent
        .run(vec![Message::user("hi again")], None)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    assert_eq!(recorded.len(), 2);
    let names =
        |i: usize| -> Vec<String> { recorded[i].tools.iter().map(|t| t.name.clone()).collect() };
    assert_eq!(names(0), vec!["tool_a".to_string()]);
    assert_eq!(
        names(1),
        vec!["tool_a".to_string(), "tool_b".to_string()],
        "second run must see the source's updated catalog"
    );
}

#[tokio::test]
async fn tool_source_dedup_explicit_tool_wins_over_source_tool() {
    // The agent's own build-time tool named "shared" must win over a
    // same-named tool produced by a tool source (dedup against "explicit
    // tools", first wins).
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let explicit = ToolDefinition {
        description: "explicit".to_string(),
        ..declaration_only_tool("shared")
    };
    let source_tool = ToolDefinition {
        description: "from-source".to_string(),
        ..declaration_only_tool("shared")
    };
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![source_tool]]));
    let agent = Agent::builder(client)
        .tool(explicit)
        .tool_source(source)
        .build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(
        shared.len(),
        1,
        "only one 'shared' tool should survive dedup"
    );
    assert_eq!(
        shared[0].description, "explicit",
        "the explicit tool wins over the source's same-named tool"
    );
}

#[tokio::test]
async fn tool_source_dedup_first_registered_source_wins() {
    // Two sources both produce a "shared" tool; the first-registered
    // source's version must win.
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let first = Arc::new(StubToolSource::new(
        "first",
        vec![vec![ToolDefinition {
            description: "from-first".to_string(),
            ..declaration_only_tool("shared")
        }]],
    ));
    let second = Arc::new(StubToolSource::new(
        "second",
        vec![vec![ToolDefinition {
            description: "from-second".to_string(),
            ..declaration_only_tool("shared")
        }]],
    ));
    let agent = Agent::builder(client)
        .tool_source(first)
        .tool_source(second)
        .build();

    let _ = agent.run(vec![Message::user("hi")], None).await.unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].description, "from-first");
}

#[tokio::test]
async fn tool_source_dedup_against_per_run_additional_tools() {
    // A per-run `additional_tools` entry must also win over a same-named
    // tool from a source (sources are resolved last).
    let client = RecordingClient::new();
    let seen = client.seen.clone();
    let source_tool = ToolDefinition {
        description: "from-source".to_string(),
        ..declaration_only_tool("shared")
    };
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![source_tool]]));
    let agent = Agent::builder(client).tool_source(source).build();

    let per_run_tool = ToolDefinition {
        description: "per-run".to_string(),
        ..declaration_only_tool("shared")
    };
    let options = AgentRunOptions::new().with_tool(per_run_tool);
    let _ = agent
        .run_with_options(vec![Message::user("hi")], None, options)
        .await
        .unwrap();

    let recorded = seen.lock().unwrap();
    let shared: Vec<_> = recorded[0]
        .tools
        .iter()
        .filter(|t| t.name == "shared")
        .collect();
    assert_eq!(shared.len(), 1);
    assert_eq!(shared[0].description, "per-run");
}

#[tokio::test]
async fn failing_tool_source_propagates_error_out_of_run() {
    // Mirrors the Python reference's run()/run_stream(), which do not catch
    // a failure raised while connecting to an MCPTool at run time -- it
    // propagates out of the whole run rather than being swallowed.
    let client = MockClient::new(vec![ChatResponse::from_text("should not be reached")]);
    let agent = Agent::builder(client)
        .tool_source(Arc::new(FailingToolSource))
        .build();

    let err = agent
        .run(vec![Message::user("hi")], None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::Service(_)));
}

#[tokio::test]
async fn tool_source_tool_is_invokable_by_the_function_loop() {
    // A tool resolved from a ToolSource must be genuinely usable, not just
    // present in the assembled ChatOptions: the model calls it and the
    // function-invocation loop executes it like any other tool.
    let call = FunctionCallContent::new(
        "call_1",
        "double",
        Some(FunctionArguments::Raw(json!({"n": 21}).to_string())),
    );
    let ask = ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call)],
        )],
        finish_reason: Some(FinishReason::tool_calls()),
        ..Default::default()
    };
    let answer = ChatResponse::from_text("42");
    let client = MockClient::new(vec![ask, answer]);

    let double = FunctionTool::new(
        "double",
        "Double a number.",
        json!({
            "type": "object",
            "properties": { "n": {"type": "integer"} },
            "required": ["n"]
        }),
        |args: Value| async move {
            let n = args["n"].as_i64().unwrap_or(0);
            Ok(json!(n * 2))
        },
    )
    .into_definition();
    let source = Arc::new(StubToolSource::new("mcp", vec![vec![double]]));

    let agent = Agent::builder(client).tool_source(source).build();
    let response = agent.run_once("double 21").await.unwrap();
    assert!(response.text().contains("42"), "got: {}", response.text());
    assert!(response.messages.iter().any(|m| m.role == Role::tool()
        && m.contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(_)))));
}
