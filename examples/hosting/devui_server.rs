//! Serve agents over HTTP with `AgentHost`: a DevUI-style API with entity
//! discovery and OpenAI-Responses-flavored execution (JSON or SSE).
//!
//! Uses a real OpenAI-backed agent when OPENAI_API_KEY is set, otherwise a
//! canned offline mock -- so the server itself runs without any credentials.
//!
//! Requires the `hosting` feature:
//! ```bash
//! cargo run -p agent-framework-examples --example hosting_server
//! ```
//!
//! Then, from another terminal:
//! ```bash
//! curl http://127.0.0.1:8080/health
//! curl http://127.0.0.1:8080/v1/entities
//! curl http://127.0.0.1:8080/v1/entities/assistant/info
//! curl -X POST http://127.0.0.1:8080/v1/responses \
//!   -H 'content-type: application/json' \
//!   -d '{"model": "assistant", "input": "Hello!"}'
//! # SSE streaming: add "stream": true to the body above.
//! ```

use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// A tiny offline stand-in for a model, used when OPENAI_API_KEY is unset.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let last = messages.last().map(ChatMessage::text).unwrap_or_default();
        Ok(ChatResponse::from_text(format!(
            "(canned reply -- set OPENAI_API_KEY for a real model) You said: {last}"
        )))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates = resp.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates.collect::<Vec<_>>()).boxed())
    }
}

fn build_agent() -> ChatAgent {
    let instructions = "You are a helpful, concise assistant.";
    match OpenAIClient::from_env("gpt-4o-mini") {
        Ok(client) => ChatAgent::builder(client)
            .name("assistant")
            .description("General-purpose assistant served over HTTP.")
            .instructions(instructions)
            .build(),
        Err(_) => ChatAgent::builder(CannedClient)
            .name("assistant")
            .description("Offline canned assistant (no OPENAI_API_KEY).")
            .instructions(instructions)
            .build(),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // `AgentHost` also takes workflows via `.workflow(name, workflow)`, and
    // `into_router()` yields a plain axum::Router if you'd rather nest this
    // into a larger app (or merge the A2A / OpenAI-compatible routers from
    // `agent_framework::hosting::{a2a, openai_compat}` alongside it).
    let host = AgentHost::new().agent("assistant", build_agent());

    println!("serving on http://127.0.0.1:8080 (Ctrl-C to stop)");
    println!("try: curl http://127.0.0.1:8080/v1/entities");
    host.serve(([127, 0, 0, 1], 8080)).await
}
