//! Computer use through the Responses host (experimental `COMPUTER_USE`):
//! an agent's computer call is emitted as a `computer_call` item, and the
//! client's `computer_call_output` reply reaches the agent as a computer
//! result.

#![cfg(feature = "experimental-computer-use")]

mod common;

use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::prelude::*;
use agent_framework_core::types::{ComputerSafetyCheck, ComputerToolCallContent};
use agent_framework_foundry_hosting::ResponsesHostServer;
use async_trait::async_trait;
use axum::http::StatusCode;
use common::*;
use futures::StreamExt;
use serde_json::json;

/// Asks for a click until it sees a screenshot, then reports it.
#[derive(Clone, Default)]
struct Operator {
    seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

#[async_trait]
impl ChatClient for Operator {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        self.seen.lock().unwrap().push(messages.clone());
        let screenshot = messages
            .iter()
            .flat_map(|m| &m.contents)
            .find_map(|c| match c {
                Content::ComputerToolResult(r) => Some(r.clone()),
                _ => None,
            });
        let content = match screenshot {
            Some(result) => Content::text(format!("clicked {}", result.call_id)),
            None => Content::ComputerToolCall(
                ComputerToolCallContent::new(
                    "cu_model",
                    "call_1",
                    vec![json!({"type": "click", "x": 3, "y": 4, "button": "left"})],
                )
                .unwrap()
                .with_pending_safety_checks(vec![ComputerSafetyCheck {
                    id: "sc_1".into(),
                    code: Some("irrelevant_domain".into()),
                    message: Some("check the domain".into()),
                }]),
            ),
        };
        Ok(ChatResponse {
            messages: vec![Message::with_contents(Role::assistant(), vec![content])],
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

#[tokio::test]
async fn computer_calls_go_out_and_their_screenshots_come_back() {
    let operator = Operator::default();
    let agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(operator.clone()).build());
    let router = ResponsesHostServer::agent(agent).into_router();

    let (status, first) = json(&router, post("/responses", json!({"input": "click it"}))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let item = &first["output"][0];
    assert_eq!(item["type"], "computer_call");
    assert!(item["id"].as_str().unwrap().starts_with("cu_"), "{item}");
    assert_eq!(item["call_id"], "call_1");
    assert_eq!(item["actions"][0]["type"], "click");
    assert_eq!(item["pending_safety_checks"][0]["id"], "sc_1");
    assert_eq!(item["status"], "completed");

    let (status, second) = json(
        &router,
        post(
            "/responses",
            json!({
                "previous_response_id": first["id"],
                "input": [{
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {"type": "computer_screenshot", "image_url": "https://x/after.png"},
                    "acknowledged_safety_checks": [{"id": "sc_1"}]
                }]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["output"][0]["content"][0]["text"], "clicked call_1");

    // The agent received the call (replayed) and the client's screenshot.
    let last = operator.seen.lock().unwrap().last().unwrap().clone();
    let contents: Vec<&Content> = last.iter().flat_map(|m| &m.contents).collect();
    assert!(contents
        .iter()
        .any(|c| matches!(c, Content::ComputerToolCall(_))));
    assert!(contents.iter().any(|c| matches!(c, Content::ComputerToolResult(r)
        if r.acknowledged_safety_checks.as_ref().is_some_and(|a| a[0].id == "sc_1")
            && matches!(r.screenshot.as_deref(), Some(Content::Uri(u)) if u.uri == "https://x/after.png"))));

    // A reply without a screenshot is refused.
    let (status, body) = json(
        &router,
        post(
            "/responses",
            json!({
                "previous_response_id": first["id"],
                "input": [{"type": "computer_call_output", "call_id": "call_1", "output": {"type": "text"}}]
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}
