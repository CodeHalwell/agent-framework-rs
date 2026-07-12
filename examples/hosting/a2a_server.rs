//! Serve an agent over the Agent2Agent (A2A) protocol: `A2ARouter` exposes
//! the agent card at `GET /.well-known/agent-card.json` and a JSON-RPC 2.0
//! endpoint at `POST /` (`message/send`, `tasks/get`, `tasks/cancel`).
//!
//! Uses a real OpenAI-backed agent when OPENAI_API_KEY is set, otherwise a
//! canned offline mock -- so the server itself runs without any credentials
//! (same pattern as `hosting/devui_server.rs`). Point `a2a/a2a_client.rs`
//! (`A2A_AGENT_URL=http://127.0.0.1:8083/`) at this server to talk to it as a
//! local `SupportsAgentRun`.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example a2a_server
//! ```
//!
//! Then, from another terminal:
//! ```bash
//! curl http://127.0.0.1:8083/.well-known/agent-card.json
//!
//! curl -X POST http://127.0.0.1:8083/ \
//!   -H 'content-type: application/json' \
//!   -d '{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{
//!         "kind":"message","role":"user","messageId":"m1",
//!         "parts":[{"kind":"text","text":"Hello!"}]}}}'
//! ```

use agent_framework::hosting::a2a::A2ARouter;
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
            .description("General-purpose assistant served over A2A.")
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
    // `base_url` is the JSON-RPC endpoint clients POST to; it's echoed back
    // verbatim in the served agent card's `url` field. The skill list
    // defaults to one skill derived from the agent's name/description --
    // `.skill(..)`/`.add_skill(..)` override or extend it.
    let app =
        A2ARouter::for_agent("assistant", build_agent(), "http://127.0.0.1:8083/").into_router();

    println!("serving on http://127.0.0.1:8083 (Ctrl-C to stop)");
    println!("try: curl http://127.0.0.1:8083/.well-known/agent-card.json");
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8083)).await?;
    axum::serve(listener, app).await
}
