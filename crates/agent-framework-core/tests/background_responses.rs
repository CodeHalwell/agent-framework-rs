//! Background responses (long-running operations resumed with a
//! continuation token): the options reach the chat client, the agent rejects
//! what upstream rejects, an unfinished reply is not recorded as history, and
//! the tool loop stops retrieving an operation once it has finished.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::history::InMemoryHistoryProvider;
use agent_framework_core::prelude::*;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::{ContinuationToken, FunctionArguments, OperationState};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// One recorded request: its messages and options.
type Request = (Vec<Message>, ChatOptions);

/// Plays scripted replies (and streams) and records every request.
#[derive(Clone, Default)]
struct Script {
    replies: Arc<Mutex<VecDeque<ChatResponse>>>,
    streams: Arc<Mutex<VecDeque<Vec<ChatResponseUpdate>>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Script {
    fn replies(replies: Vec<ChatResponse>) -> Self {
        let script = Self::default();
        *script.replies.lock().unwrap() = replies.into();
        script
    }

    fn streams(streams: Vec<Vec<ChatResponseUpdate>>) -> Self {
        let script = Self::default();
        *script.streams.lock().unwrap() = streams.into();
        script
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for Script {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.requests.lock().unwrap().push((messages, options));
        Ok(self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| ChatResponse::from_text("default")))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        self.requests.lock().unwrap().push((messages, options));
        let updates = self.streams.lock().unwrap().pop_front().unwrap_or_default();
        Ok(futures::stream::iter(updates.into_iter().map(Ok)).boxed())
    }
}

fn token(id: &str) -> ContinuationToken {
    ContinuationToken::new(id)
}

/// A reply from an operation that is still running.
fn pending(id: &str, partial: Option<&str>) -> ChatResponse {
    ChatResponse {
        messages: partial
            .map(|text| vec![Message::assistant(text)])
            .unwrap_or_default(),
        continuation_token: Some(token(id)),
        ..Default::default()
    }
}

fn session_with(history: &InMemoryHistoryProvider) -> AgentSession {
    let mut session = AgentSession::new();
    session.context_providers.push(Arc::new(history.clone()));
    session
}

fn texts(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .map(|m| format!("{}:{}", m.role, m.text()))
        .collect()
}

#[tokio::test]
async fn background_and_continuation_token_reach_the_chat_client() {
    let script = Script::replies(vec![
        pending("op-1", None),
        ChatResponse::from_text("finished"),
    ]);
    let agent = Agent::builder(script.clone()).build();
    let mut session = agent.create_session();

    let first = agent
        .run_with_options(
            vec![Message::user("start")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    assert_eq!(first.continuation_token, Some(token("op-1")));

    let done = agent
        .run_with_options(
            Vec::new(),
            Some(&mut session),
            AgentRunOptions::new().with_continuation_token(token("op-1")),
        )
        .await
        .unwrap();
    assert_eq!(done.text(), "finished");
    assert!(done.continuation_token.is_none());

    let requests = script.requests();
    assert_eq!(requests[0].1.background, Some(true));
    assert!(requests[0].1.continuation_token.is_none());
    // The poll carries the token and sends no messages, not even history.
    assert_eq!(requests[1].1.continuation_token, Some(token("op-1")));
    assert!(requests[1].0.is_empty(), "got {:?}", texts(&requests[1].0));
}

#[tokio::test]
async fn chat_options_carry_background_through_the_merge() {
    let script = Script::replies(vec![pending("op-2", None)]);
    let agent = Agent::builder(script.clone())
        .chat_options(ChatOptions::new().with_background(true))
        .build();
    let mut session = agent.create_session();
    let response = agent
        .run_with_options(
            vec![Message::user("go")],
            Some(&mut session),
            AgentRunOptions::new().with_chat_options(ChatOptions::new().with_temperature(0.1)),
        )
        .await
        .unwrap();
    assert_eq!(response.continuation_token, Some(token("op-2")));
    assert_eq!(script.requests()[0].1.background, Some(true));

    // The run option wins over the chat option.
    let merged = ChatOptions::new()
        .with_background(true)
        .merge(ChatOptions::new().with_background(false));
    assert_eq!(merged.background, Some(false));
}

#[tokio::test]
async fn a_continuation_token_with_new_messages_is_rejected() {
    let script = Script::default();
    let agent = Agent::builder(script.clone()).build();
    let mut session = agent.create_session();
    let err = agent
        .run_with_options(
            vec![Message::user("more")],
            Some(&mut session),
            AgentRunOptions::new().with_continuation_token(token("op")),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("continuation token"), "{err}");

    let err = match agent
        .run_stream(
            vec![Message::user("more")],
            Some(session.clone()),
            Some(AgentRunOptions::new().with_continuation_token(token("op"))),
        )
        .await
    {
        Err(e) => e,
        Ok(_) => panic!("a streamed run must reject it too"),
    };
    assert!(err.to_string().contains("continuation token"), "{err}");
    assert!(script.requests().is_empty());
}

#[tokio::test]
async fn a_background_run_needs_a_session() {
    let script = Script::default();
    let agent = Agent::builder(script.clone()).build();
    let err = agent
        .run_with_options(
            vec![Message::user("go")],
            None,
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap_err();
    assert!(err.to_string().contains("session"), "{err}");
    assert!(script.requests().is_empty());
}

#[tokio::test]
async fn an_unfinished_reply_is_not_recorded_until_the_operation_completes() {
    let script = Script::replies(vec![
        pending("op-3", Some("half an ans")),
        pending("op-3", Some("most of an answ")),
        ChatResponse::from_text("the whole answer"),
    ]);
    let agent = Agent::builder(script.clone()).build();
    let history = InMemoryHistoryProvider::new();
    let mut session = session_with(&history);

    agent
        .run_with_options(
            vec![Message::user("question")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    // The input is recorded by the run that sent it; the partial reply is not.
    assert_eq!(texts(&history.list_messages()), ["user:question"]);

    for _ in 0..2 {
        agent
            .run_with_options(
                Vec::new(),
                Some(&mut session),
                AgentRunOptions::new().with_continuation_token(token("op-3")),
            )
            .await
            .unwrap();
    }
    assert_eq!(
        texts(&history.list_messages()),
        ["user:question", "assistant:the whole answer"]
    );
}

#[tokio::test]
async fn background_runs_fall_back_from_per_call_history_to_the_end_of_the_run() {
    let script = Script::replies(vec![pending("op-4", Some("partial"))]);
    let agent = Agent::builder(script.clone())
        .tool(lookup())
        .require_per_service_call_history_persistence(true)
        .build();
    let history = InMemoryHistoryProvider::new();
    let mut session = session_with(&history);
    agent
        .run_with_options(
            vec![Message::user("question")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    assert_eq!(texts(&history.list_messages()), ["user:question"]);
}

fn stream_update(text: &str, id: &str, state: OperationState) -> ChatResponseUpdate {
    ChatResponseUpdate {
        contents: if text.is_empty() {
            Vec::new()
        } else {
            vec![Content::text(text)]
        },
        role: Some(Role::assistant()),
        continuation_token: (state == OperationState::InProgress).then(|| token(id)),
        operation_state: Some(state),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_streamed_background_run_drops_its_token_once_it_finishes() {
    let finished = vec![
        stream_update("", "op-5@1", OperationState::InProgress),
        stream_update("Hel", "op-5@2", OperationState::InProgress),
        stream_update("lo", "op-5@3", OperationState::InProgress),
        stream_update("", "", OperationState::Terminal),
    ];
    let interrupted = finished[..3].to_vec();
    let script = Script::streams(vec![finished, interrupted]);
    let agent = Agent::builder(script.clone()).build();
    let history = InMemoryHistoryProvider::new();
    let session = session_with(&history);

    let run = |input: &str| {
        agent.run_stream(
            vec![Message::user(input)],
            Some(session.clone()),
            Some(AgentRunOptions::new().with_background(true)),
        )
    };
    let updates: Vec<_> = run("first")
        .await
        .unwrap()
        .map(|u| u.unwrap())
        .collect()
        .await;
    assert_eq!(
        updates[2].continuation_token,
        Some(token("op-5@3")),
        "every in-progress update can resume the stream"
    );
    let response = AgentResponse::from_updates(updates);
    assert_eq!(response.text(), "Hello");
    assert!(response.continuation_token.is_none());
    assert_eq!(
        texts(&history.list_messages()),
        ["user:first", "assistant:Hello"]
    );

    // A stream that ends before the operation does keeps the latest token,
    // and its partial reply stays out of the history.
    let updates: Vec<_> = run("second")
        .await
        .unwrap()
        .map(|u| u.unwrap())
        .collect()
        .await;
    let response = AgentResponse::from_updates(updates);
    assert_eq!(response.continuation_token, Some(token("op-5@3")));
    assert_eq!(
        texts(&history.list_messages()),
        ["user:first", "assistant:Hello", "user:second"]
    );
}

fn lookup() -> ToolDefinition {
    FunctionTool::new(
        "lookup",
        "Look something up.",
        json!({"type": "object", "properties": {}}),
        |_| async move { Ok(json!("found")) },
    )
    .into_definition()
}

fn ask(call_id: &str) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                call_id,
                "lookup",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        ..Default::default()
    }
}

#[tokio::test]
async fn the_tool_loop_submits_results_instead_of_retrieving_a_finished_operation() {
    // Polling the finished operation returns a tool call; the loop runs the
    // tool and must then send its result, not retrieve the operation again.
    let script = Script::replies(vec![ask("c1"), ChatResponse::from_text("answered")]);
    let agent = Agent::builder(script.clone()).tool(lookup()).build();
    let mut session = agent.create_session();
    let response = agent
        .run_with_options(
            Vec::new(),
            Some(&mut session),
            AgentRunOptions::new()
                .with_background(true)
                .with_continuation_token(token("op-6")),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "answered");
    let requests = script.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].1.continuation_token, Some(token("op-6")));
    assert!(requests[1].1.continuation_token.is_none());
    assert_eq!(requests[1].1.background, Some(true));
}

#[tokio::test]
async fn the_tool_loop_hands_back_an_operation_that_is_still_running() {
    let script = Script::replies(vec![pending("op-7", None)]);
    let agent = Agent::builder(script.clone()).tool(lookup()).build();
    let mut session = agent.create_session();
    let response = agent
        .run_with_options(
            vec![Message::user("go")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    assert_eq!(response.continuation_token, Some(token("op-7")));
    assert_eq!(script.requests().len(), 1);

    // The streamed tool loop replays the result with its token.
    let script = Script::replies(vec![pending("op-8", Some("thinking"))]);
    let agent = Agent::builder(script.clone()).tool(lookup()).build();
    let updates: Vec<_> = agent
        .run_stream(
            vec![Message::user("go")],
            Some(agent.create_session()),
            Some(AgentRunOptions::new().with_background(true)),
        )
        .await
        .unwrap()
        .map(|u| u.unwrap())
        .collect()
        .await;
    let response = AgentResponse::from_updates(updates);
    assert_eq!(response.continuation_token, Some(token("op-8")));
}

#[tokio::test]
async fn the_call_budget_carries_across_background_continuations() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let runs = Arc::new(AtomicUsize::new(0));
    let counted = runs.clone();
    let tool = FunctionTool::new(
        "lookup",
        "Look something up.",
        json!({"type": "object", "properties": {}}),
        move |_| {
            let counted = counted.clone();
            async move {
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(json!("found"))
            }
        },
    )
    .into_definition();
    // Request 1 runs the tool once (spending the whole budget of one call),
    // then the operation is still running. Request 2 resumes it and the
    // model asks for the tool again: the budget is already spent.
    let script = Script::replies(vec![
        ask("c1"),
        pending("op-9", None),
        ask("c2"),
        ChatResponse::from_text("done"),
    ]);
    let agent = Agent::builder(script.clone())
        .tool(tool)
        .function_invocation_config(FunctionInvocationConfig {
            max_function_calls: Some(1),
            ..Default::default()
        })
        .build();
    let mut session = agent.create_session();
    let first = agent
        .run_with_options(
            vec![Message::user("go")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    let token = first.continuation_token.clone().expect("still running");
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    agent
        .run_with_options(
            Vec::new(),
            Some(&mut session),
            AgentRunOptions::new()
                .with_background(true)
                .with_continuation_token(token),
        )
        .await
        .unwrap();
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "a continuation must not get a fresh call budget"
    );
}

/// A tool that counts its executions.
fn counting_tool(name: &str, runs: Arc<std::sync::atomic::AtomicUsize>) -> ToolDefinition {
    FunctionTool::new(
        name,
        "Counts its runs.",
        json!({"type": "object", "properties": {}}),
        move |_| {
            let runs = runs.clone();
            async move {
                runs.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(json!("ran"))
            }
        },
    )
    .into_definition()
}

fn ask_tool(call_id: &str, name: &str) -> ChatResponse {
    ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        ..Default::default()
    }
}

#[tokio::test]
async fn the_iteration_limit_carries_across_background_continuations() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let runs = Arc::new(AtomicUsize::new(0));
    // Request 1 uses both iterations (one tool call, then a pending poll).
    // The continuation must go straight to the tools-disabled failsafe.
    let script = Script::replies(vec![
        ask_tool("c1", "lookup"),
        pending("op-10", None),
        ask_tool("c2", "lookup"),
        ask_tool("c3", "lookup"),
        ChatResponse::from_text("done"),
    ]);
    let agent = Agent::builder(script.clone())
        .tool(counting_tool("lookup", runs.clone()))
        .function_invocation_config(FunctionInvocationConfig {
            max_iterations: 2,
            ..Default::default()
        })
        .build();
    let mut session = agent.create_session();
    let first = agent
        .run_with_options(
            vec![Message::user("go")],
            Some(&mut session),
            AgentRunOptions::new().with_background(true),
        )
        .await
        .unwrap();
    let token = first.continuation_token.clone().expect("still running");
    agent
        .run_with_options(
            Vec::new(),
            Some(&mut session),
            AgentRunOptions::new()
                .with_background(true)
                .with_continuation_token(token),
        )
        .await
        .unwrap();
    assert_eq!(
        runs.load(Ordering::SeqCst),
        1,
        "a continuation must not get fresh iterations"
    );
}

/// Injects a tool, plus a message that a continuation must not send.
struct ToolProvider(ToolDefinition);

#[async_trait]
impl agent_framework_core::memory::ContextProvider for ToolProvider {
    async fn before_run(
        &self,
        ctx: &mut agent_framework_core::memory::SessionContext,
    ) -> Result<()> {
        ctx.tools.push(self.0.clone());
        ctx.messages.push(Message::user("provider context"));
        Ok(())
    }
}

#[tokio::test]
async fn provider_tools_are_available_to_a_continuation() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let runs = Arc::new(AtomicUsize::new(0));
    let script = Script::replies(vec![
        ask_tool("c1", "provided"),
        ChatResponse::from_text("done"),
    ]);
    let agent = Agent::builder(script.clone())
        .context_provider(Arc::new(ToolProvider(counting_tool(
            "provided",
            runs.clone(),
        ))))
        .build();
    let mut session = agent.create_session();
    let response = agent
        .run_with_options(
            Vec::new(),
            Some(&mut session),
            AgentRunOptions::new()
                .with_background(true)
                .with_continuation_token(token("op-11")),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(runs.load(Ordering::SeqCst), 1, "the provider's tool ran");
    let requests = script.requests();
    assert!(
        requests[0].1.tools.iter().any(|t| t.name == "provided"),
        "the continuation offers the provider's tool"
    );
    assert!(
        requests[0].0.is_empty(),
        "no provider context is sent with a continuation"
    );
}
