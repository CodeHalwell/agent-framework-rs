//! `BackgroundSource::Provider`: a `background: true` response is submitted
//! to the agent's provider as a background response and polled with the
//! continuation token it returns, until it finishes.

mod common;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::prelude::*;
use agent_framework_core::tools::FunctionTool;
use agent_framework_core::types::{ContinuationToken, FunctionArguments};
use agent_framework_foundry_hosting::{BackgroundSource, HistorySource, ResponsesHostServer};
use async_trait::async_trait;
use axum::Router;
use common::*;
use futures::StreamExt;
use serde_json::{json, Value};

/// One recorded request: its messages and options.
type Request = (Vec<Message>, ChatOptions);

/// Plays scripted replies and records each request's messages and options.
#[derive(Clone, Default)]
struct Provider {
    replies: Arc<Mutex<VecDeque<ChatResponse>>>,
    requests: Arc<Mutex<Vec<Request>>>,
}

impl Provider {
    fn new(replies: Vec<ChatResponse>) -> Self {
        let provider = Self::default();
        *provider.replies.lock().unwrap() = replies.into();
        provider
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl ChatClient for Provider {
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
            .unwrap_or_else(|| ChatResponse::from_text("unscripted")))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let update = ChatResponseUpdate {
            contents: response
                .messages
                .into_iter()
                .flat_map(|m| m.contents)
                .collect(),
            role: Some(Role::assistant()),
            conversation_id: response.conversation_id,
            ..Default::default()
        };
        Ok(futures::stream::iter([Ok(update)]).boxed())
    }
}

fn with_service(mut response: ChatResponse) -> ChatResponse {
    response.conversation_id = Some("svc_1".into());
    response
}

fn pending(messages: Vec<Message>) -> ChatResponse {
    with_service(ChatResponse {
        messages,
        continuation_token: Some(ContinuationToken::new("tok")),
        ..Default::default()
    })
}

fn build(provider: Provider, history: HistorySource, tools: Vec<ToolDefinition>) -> Router {
    let agent = Agent::builder(provider).name("bg").tools(tools).build();
    ResponsesHostServer::agent(Arc::new(agent))
        .with_history_source(history)
        .with_background_source(BackgroundSource::Provider)
        .with_background_poll_interval(Duration::from_millis(1))
        .into_router()
}

async fn finished(router: &Router, id: &str) -> Value {
    for _ in 0..200 {
        let (_, polled) = json(router, get(&format!("/responses/{id}"))).await;
        if polled["status"] != "queued" && polled["status"] != "in_progress" {
            return polled;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("the response did not finish");
}

#[tokio::test]
async fn a_background_response_is_polled_from_the_provider_until_it_finishes() {
    let provider = Provider::new(vec![
        pending(Vec::new()),
        // A poll's partial output is not recorded; polling repeats it.
        pending(vec![Message::assistant("the answ")]),
        with_service(ChatResponse::from_text("the answer")),
    ]);
    let router = build(provider.clone(), HistorySource::Service, Vec::new());
    let (_, queued) = json(
        &router,
        post(
            "/responses",
            json!({"input": "question", "background": true}),
        ),
    )
    .await;
    assert_eq!(queued["status"], "queued");
    let done = finished(&router, queued["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "completed", "{done}");
    assert_eq!(done["output"].as_array().unwrap().len(), 1, "{done}");
    assert_eq!(done["output"][0]["content"][0]["text"], "the answer");

    let requests = provider.requests();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].1.background, Some(true));
    assert!(requests[0].1.continuation_token.is_none());
    assert_eq!(requests[0].0.last().unwrap().text(), "question");
    for (messages, options) in &requests[1..] {
        assert_eq!(options.background, Some(true));
        assert_eq!(
            options.continuation_token,
            Some(ContinuationToken::new("tok"))
        );
        assert!(messages.is_empty());
    }
}

#[tokio::test]
async fn a_streamed_background_response_emits_the_polled_output() {
    let provider = Provider::new(vec![
        pending(Vec::new()),
        with_service(ChatResponse::from_text("streamed answer")),
    ]);
    let router = build(provider.clone(), HistorySource::Service, Vec::new());
    let (_, _, body) = send(
        &router,
        post(
            "/responses",
            json!({"input": "q", "background": true, "stream": true}),
        ),
    )
    .await;
    let events = sse_events(&body);
    assert_eq!(events.last().unwrap().0, "response.completed", "{body}");
    assert!(body.contains("streamed answer"), "{body}");
    assert_eq!(provider.requests().len(), 2);
}

#[tokio::test]
async fn tool_results_completed_ahead_of_an_unfinished_response_are_kept() {
    let ask = with_service(ChatResponse {
        messages: vec![Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "c1",
                "lookup",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        )],
        ..Default::default()
    });
    let provider = Provider::new(vec![
        ask,
        pending(vec![Message::assistant("partial")]),
        with_service(ChatResponse::from_text("found it")),
    ]);
    let lookup = FunctionTool::new(
        "lookup",
        "Look something up.",
        json!({"type": "object", "properties": {}}),
        |_| async move { Ok(json!("result")) },
    )
    .into_definition();
    let router = build(provider.clone(), HistorySource::Service, vec![lookup]);
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "q", "background": true})),
    )
    .await;
    let done = finished(&router, queued["id"].as_str().unwrap()).await;
    let kinds: Vec<_> = done["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item["type"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(
        kinds,
        ["function_call", "function_call_output", "message"],
        "{done}"
    );
    assert_eq!(done["output"][2]["content"][0]["text"], "found it");
}

#[tokio::test]
async fn provider_background_requires_service_history() {
    let provider = Provider::new(vec![pending(Vec::new())]);
    let router = build(provider.clone(), HistorySource::AgentServer, Vec::new());
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "q", "background": true})),
    )
    .await;
    let done = finished(&router, queued["id"].as_str().unwrap()).await;
    assert_eq!(done["status"], "failed");
    assert!(done["error"]["message"]
        .as_str()
        .unwrap()
        .contains("HistorySource::Service"));
    assert!(provider.requests().is_empty());

    // Without `background`, the provider is called as usual.
    let provider = Provider::new(vec![ChatResponse::from_text("plain")]);
    let router = build(provider.clone(), HistorySource::Service, Vec::new());
    let (_, plain) = json(&router, post("/responses", json!({"input": "q"}))).await;
    assert_eq!(plain["output_text"], "plain");
    assert!(provider.requests()[0].1.background.is_none());
}
