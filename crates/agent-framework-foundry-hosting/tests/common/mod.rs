//! Shared helpers: a scripted chat client behind a real `Agent`, and
//! oneshot HTTP helpers.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::Result;
use agent_framework_core::prelude::*;
use async_trait::async_trait;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use axum::Router;
use futures::StreamExt;
use http_body_util::BodyExt;
use serde_json::Value;
use tower::ServiceExt;

/// Replies `"<n> messages, last: <text>"` and records each request.
#[derive(Clone, Default)]
pub struct Echo {
    pub seen: Arc<Mutex<Vec<Vec<Message>>>>,
}

impl Echo {
    fn reply(&self, messages: &[Message]) -> String {
        self.seen.lock().unwrap().push(messages.to_vec());
        let last = messages.last().map(Message::text).unwrap_or_default();
        format!("{} messages, last: {last}", messages.len())
    }
}

#[async_trait]
impl ChatClient for Echo {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(self.reply(&messages)))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        _: ChatOptions,
    ) -> Result<ChatStream> {
        let text = self.reply(&messages);
        let (a, b) = text.split_at(text.len() / 2);
        let updates = [a, b].map(|t| {
            Ok(ChatResponseUpdate {
                contents: vec![Content::text(t)],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates).boxed())
    }
}

pub fn agent(echo: Echo) -> Arc<dyn SupportsAgentRun> {
    Arc::new(Agent::builder(echo).name("echo").build())
}

pub async fn send(
    router: &Router,
    request: Request<Body>,
) -> (StatusCode, axum::http::HeaderMap, String) {
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body().collect().await.unwrap().to_bytes();
    (status, headers, String::from_utf8_lossy(&body).to_string())
}

pub fn post(path: &str, body: Value) -> Request<Body> {
    Request::post(path)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

pub fn get(path: &str) -> Request<Body> {
    Request::get(path).body(Body::empty()).unwrap()
}

pub async fn json(router: &Router, request: Request<Body>) -> (StatusCode, Value) {
    let (status, _, body) = send(router, request).await;
    (status, serde_json::from_str(&body).unwrap_or(Value::Null))
}

/// `(event, data)` pairs of an SSE body.
pub fn sse_events(body: &str) -> Vec<(String, Value)> {
    body.split("\n\n")
        .filter_map(|frame| {
            let mut event = None;
            let mut data = None;
            for line in frame.lines() {
                if let Some(e) = line.strip_prefix("event: ") {
                    event = Some(e.to_string());
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data = serde_json::from_str(d).ok();
                }
            }
            Some((event?, data?))
        })
        .collect()
}
