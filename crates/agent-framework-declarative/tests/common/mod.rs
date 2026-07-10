//! Shared test helpers: a scripted mock chat client (no network), copied from
//! the core crate's integration-test pattern.
//!
//! Each integration-test binary includes this module and uses a different
//! subset, so unused-item warnings here are expected.
#![allow(dead_code)]

use std::sync::{Arc, Mutex};

use agent_framework_core::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A scripted chat client that returns queued responses in order.
#[derive(Clone)]
pub struct MockClient {
    responses: Arc<Mutex<Vec<ChatResponse>>>,
    /// Every message list the client was asked to respond to.
    pub seen: Arc<Mutex<Vec<Vec<ChatMessage>>>>,
    model: Option<String>,
}

impl MockClient {
    /// Create a mock returning `responses` in order (then a filler response).
    pub fn new(responses: Vec<ChatResponse>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses)),
            seen: Arc::new(Mutex::new(Vec::new())),
            model: None,
        }
    }

    /// A mock that always replies with the same text.
    pub fn always(text: &str) -> Self {
        Self::new(vec![ChatResponse::from_text(text)])
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
        } else if resps.len() == 1 {
            Ok(resps[0].clone())
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

    fn model_id(&self) -> Option<&str> {
        self.model.as_deref()
    }
}

/// Build an agent registry entry backed by a mock that always replies `text`.
pub fn mock_agent(name: &str, text: &str) -> ChatAgent {
    ChatAgent::builder(MockClient::always(text))
        .name(name)
        .build()
}
