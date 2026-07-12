//! Chat middleware: wraps the underlying `ChatClient` call itself, one level
//! below agent middleware (`agents/agent_middleware.rs`). It can rewrite the
//! outgoing `ctx.messages` / `ctx.chat_options` before calling
//! `next.run(ctx)`, observe `ctx.result` afterward, or short-circuit by
//! setting `ctx.result` and `ctx.terminate = true` without calling `next` --
//! the underlying client is then never invoked at all.
//!
//! Runs fully offline against a canned client -- no API key or network
//! needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example chat_middleware
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Forces `temperature` to `0.0` before the call (a "pin down determinism"
/// policy), and logs the response afterward.
struct PinTemperatureMiddleware;

#[async_trait]
impl Middleware<ChatContext> for PinTemperatureMiddleware {
    async fn process(&self, mut ctx: ChatContext, next: Next<ChatContext>) -> Result<ChatContext> {
        println!(
            "  [chat-middleware] outgoing temperature={:?} -> forcing 0.0",
            ctx.chat_options.temperature
        );
        ctx.chat_options.temperature = Some(0.0);
        let ctx = next.run(ctx).await?;
        println!(
            "  [chat-middleware] incoming: {:?}",
            ctx.result.as_ref().map(ChatResponse::text)
        );
        Ok(ctx)
    }
}

/// Serves a canned response without ever calling `next` -- the underlying
/// client is skipped entirely. Useful for caching, mocking in tests, or
/// policy-based refusals at the chat-client boundary.
struct CachingMiddleware {
    cached: ChatResponse,
}

#[async_trait]
impl Middleware<ChatContext> for CachingMiddleware {
    async fn process(&self, mut ctx: ChatContext, _next: Next<ChatContext>) -> Result<ChatContext> {
        println!("  [cache] short-circuiting -- serving a cached response, client not called");
        ctx.result = Some(self.cached.clone());
        ctx.terminate = true;
        Ok(ctx)
    }
}

/// Reports the temperature it was called with, and counts how many times it
/// was actually invoked (to prove the caching scenario below skips it).
#[derive(Clone, Default)]
struct CountingClient {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatClient for CountingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(ChatResponse::from_text(format!(
            "(live call) temperature was {:?}",
            options.temperature
        )))
    }

    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Scenario 1: pre-call rewrite + post-call observation.
    let client = CountingClient::default();
    let calls = client.calls.clone();
    let agent = ChatAgent::builder(client)
        .name("assistant")
        .temperature(0.9)
        .chat_middleware(Arc::new(PinTemperatureMiddleware))
        .build();

    println!("-- scenario 1: request rewriting --");
    let r1 = agent.run_once("Hi").await?;
    println!("final: {}\n", r1.text());
    println!(
        "(underlying client was called {} time(s))\n",
        calls.load(Ordering::SeqCst)
    );

    // Scenario 2: short-circuiting -- the underlying client is never reached.
    let client2 = CountingClient::default();
    let calls2 = client2.calls.clone();
    let agent2 = ChatAgent::builder(client2)
        .name("assistant")
        .chat_middleware(Arc::new(CachingMiddleware {
            cached: ChatResponse::from_text("(cached) no need to call the model for this."),
        }))
        .build();

    println!("-- scenario 2: short-circuiting --");
    let r2 = agent2.run_once("Hi again").await?;
    println!("final: {}", r2.text());
    println!(
        "(underlying client was called {} time(s) -- proves the short-circuit skipped it)",
        calls2.load(Ordering::SeqCst)
    );

    Ok(())
}
