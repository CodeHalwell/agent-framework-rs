//! Per-model-call history persistence
//! (`require_per_service_call_history_persistence`): each model call of the
//! tool loop records what it added, so a run that fails midway keeps the
//! turns it completed, and a run that succeeds is recorded exactly once.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::InMemoryHistoryProvider;
use agent_framework_core::prelude::*;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::FunctionArguments;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::json;

/// Plays a script of replies; `Err` entries fail that call.
#[derive(Clone)]
struct Script(Arc<Mutex<VecDeque<Result<ChatResponse>>>>);

#[async_trait]
impl ChatClient for Script {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.0
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok(ChatResponse::from_text("done")))
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

fn agent(script: Vec<Result<ChatResponse>>, per_call: bool) -> Agent {
    let lookup = FunctionTool::new(
        "lookup",
        "Look something up.",
        json!({"type": "object", "properties": {}}),
        |_| async move { Ok(json!("found")) },
    )
    .into_definition();
    Agent::builder(Script(Arc::new(Mutex::new(script.into()))))
        .name("worker")
        .tool(lookup)
        .require_per_service_call_history_persistence(per_call)
        .build()
}

fn session_with(history: &InMemoryHistoryProvider) -> AgentSession {
    let mut session = AgentSession::new();
    session.context_providers.push(Arc::new(history.clone()));
    session
}

fn roles(messages: &[Message]) -> Vec<String> {
    messages.iter().map(|m| m.role.to_string()).collect()
}

#[tokio::test]
async fn a_run_failing_midway_keeps_its_completed_turns() {
    let script = || {
        vec![
            Ok(ask("c1")),
            Ok(ask("c2")),
            Err(Error::service("model unavailable")),
        ]
    };

    let history = InMemoryHistoryProvider::new();
    let mut session = session_with(&history);
    let failed = agent(script(), true)
        .run(vec![Message::user("research this")], Some(&mut session))
        .await;
    assert!(failed.is_err());
    let kept = history.list_messages();
    // The input with the first call, then the first tool result with the
    // second call; the third call failed, so the last result is not kept.
    assert_eq!(roles(&kept), ["user", "assistant", "tool", "assistant"]);
    assert_eq!(kept[0].text(), "research this");
    assert_eq!(kept[1].author_name.as_deref(), Some("worker"));

    // Without the setting a failed run records nothing.
    let history = InMemoryHistoryProvider::new();
    let mut session = session_with(&history);
    assert!(agent(script(), false)
        .run(vec![Message::user("research this")], Some(&mut session))
        .await
        .is_err());
    assert!(history.list_messages().is_empty());
}

#[tokio::test]
async fn a_successful_run_is_recorded_once_as_without_the_setting() {
    let script = || vec![Ok(ask("c1")), Ok(ChatResponse::from_text("the answer"))];
    let mut recorded = Vec::new();
    for per_call in [true, false] {
        let history = InMemoryHistoryProvider::new();
        let mut session = session_with(&history);
        agent(script(), per_call)
            .run(vec![Message::user("q")], Some(&mut session))
            .await
            .unwrap();
        recorded.push(roles(&history.list_messages()));
    }
    assert_eq!(recorded[0], ["user", "assistant", "tool", "assistant"]);
    assert_eq!(recorded[0], recorded[1]);
}

#[tokio::test]
async fn a_streamed_run_without_tool_calls_is_still_recorded() {
    let history = InMemoryHistoryProvider::new();
    let session = session_with(&history);
    let mut stream = agent(vec![Ok(ChatResponse::from_text("hi"))], true)
        .run_stream(vec![Message::user("hello")], Some(session), None)
        .await
        .unwrap();
    while let Some(update) = stream.next().await {
        update.unwrap();
    }
    assert_eq!(roles(&history.list_messages()), ["user", "assistant"]);
}

#[tokio::test]
async fn a_second_run_does_not_rerecord_the_first() {
    let history = InMemoryHistoryProvider::new();
    let mut session = session_with(&history);
    let agent = agent(
        vec![
            Ok(ask("c1")),
            Ok(ChatResponse::from_text("one")),
            Ok(ask("c2")),
            Ok(ChatResponse::from_text("two")),
        ],
        true,
    );
    agent
        .run(vec![Message::user("first")], Some(&mut session))
        .await
        .unwrap();
    agent
        .run(vec![Message::user("second")], Some(&mut session))
        .await
        .unwrap();
    let texts: Vec<String> = history
        .list_messages()
        .iter()
        .filter(|m| m.role == Role::user() || m.role == Role::assistant())
        .map(Message::text)
        .filter(|t| !t.is_empty())
        .collect();
    assert_eq!(texts, ["first", "one", "second", "two"]);
}
