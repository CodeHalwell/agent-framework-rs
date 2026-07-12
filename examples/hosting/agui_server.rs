//! Serve an agent over the AG-UI protocol (CopilotKit's SupportsAgentRun-User
//! Interaction protocol): `POST /` takes a RunAgentInput body and streams
//! camelCase SSE events (RUN_STARTED, TEXT_MESSAGE_*, TOOL_CALL_*,
//! RUN_FINISHED) that AG-UI frontends consume directly.
//!
//! Uses a real OpenAI-backed agent when OPENAI_API_KEY is set, otherwise a
//! canned offline mock -- the server itself runs without credentials.
//!
//! Requires the `hosting` feature:
//! ```bash
//! cargo run -p agent-framework-examples --example agui_server
//! ```
//!
//! Then, from another terminal:
//! ```bash
//! curl -N -X POST http://127.0.0.1:8081/ \
//!   -H 'content-type: application/json' \
//!   -d '{"threadId":"t1","runId":"r1","messages":[{"role":"user","content":"Hello!"}]}'
//! ```

use agent_framework::hosting::agui::AgUiRouter;
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
    match OpenAIClient::from_env("gpt-4o-mini") {
        Ok(client) => Agent::builder(client)
            .name("assistant")
            .instructions(instructions)
            .build(),
        Err(_) => Agent::builder(CannedClient)
            .name("assistant")
            .instructions(instructions)
            .build(),
    }
}

#[tokio::main]
async fn main() -> std::io::Result<()> {
    // `.agent(...)`/`.path(...)` add more agents at distinct paths;
    // `into_router()` is a plain axum::Router, so it nests/merges freely with
    // the AgentHost (DevUI), A2A, and OpenAI-compatible routers.
    let app = AgUiRouter::for_agent("assistant", build_agent()).into_router();

    println!("AG-UI endpoint on http://127.0.0.1:8081/ (Ctrl-C to stop)");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8081)).await?;
    axum::serve(listener, app).await
}
