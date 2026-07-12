//! Serve an agent over the OpenAI Chat Completions API: `OpenAiRouter` exposes
//! `POST /v1/chat/completions` (JSON or SSE), so any OpenAI-Chat client can
//! talk to an `agent-framework-rs` agent without knowing it isn't OpenAI.
//!
//! Uses a real OpenAI-backed agent when OPENAI_API_KEY is set, otherwise a
//! canned offline mock -- so the server itself runs without any credentials
//! (same pattern as `hosting/devui_server.rs`).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example openai_compat_server
//! ```
//!
//! Then, from another terminal:
//! ```bash
//! curl -X POST http://127.0.0.1:8082/v1/chat/completions \
//!   -H 'content-type: application/json' \
//!   -d '{"model": "assistant", "messages": [{"role": "user", "content": "Hello!"}]}'
//!
//! # SSE streaming: add "stream": true to the body above.
//! curl -N -X POST http://127.0.0.1:8082/v1/chat/completions \
//!   -H 'content-type: application/json' \
//!   -d '{"model": "assistant", "messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
//! ```

use agent_framework::hosting::openai_compat::OpenAiRouter;
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
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let last = messages.last().map(Message::text).unwrap_or_default();
        Ok(ChatResponse::from_text(format!(
            "(canned reply -- set OPENAI_API_KEY for a real model) You said: {last}"
        )))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
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

fn build_agent() -> Agent {
    let instructions = "You are a helpful, concise assistant.";
    match OpenAIChatCompletionClient::from_env("gpt-4o-mini") {
        Ok(client) => Agent::builder(client)
            .name("assistant")
            .description("General-purpose assistant served over the OpenAI-compatible API.")
            .instructions(instructions)
            .build(),
        Err(_) => Agent::builder(CannedClient)
            .name("assistant")
            .description("Offline canned assistant (no OPENAI_API_KEY).")
            .instructions(instructions)
            .build(),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // `OpenAiRouter::into_router()` is a plain `axum::Router`, so it nests or
    // merges freely with the DevUI (`AgentHost`) and A2A (`A2ARouter`) routers
    // -- see `hosting/devui_server.rs`'s doc comment for a combined example.
    let app = OpenAiRouter::for_agent("assistant", build_agent()).into_router();

    println!("serving on http://127.0.0.1:8082 (Ctrl-C to stop)");
    println!("try: curl -X POST http://127.0.0.1:8082/v1/chat/completions -H 'content-type: application/json' -d '{{\"model\": \"assistant\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8082)).await?;
    axum::serve(listener, app).await
}
