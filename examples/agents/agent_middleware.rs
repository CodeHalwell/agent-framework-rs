//! SupportsAgentRun middleware: wraps a whole `agent.run(...)` call. A middleware
//! receives an owned `AgentContext` and a `Next` continuation -- call
//! `next.run(ctx)` to continue the chain and observe/rewrite the result, or
//! return without calling it (optionally setting `ctx.terminate = true`) to
//! short-circuit the run entirely, before the underlying model is ever
//! called.
//!
//! Two middleware are composed here: a logging middleware that wraps every
//! run, and a blocked-words middleware nested inside it that can terminate
//! the run early with a canned refusal.
//!
//! Runs fully offline against a canned client -- no API key or network
//! needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example agent_middleware
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Logs every run's start and finish. Registered first, so it wraps
/// everything else in the pipeline (including `BlockedWordsMiddleware`
/// below): it observes the final result whether or not an inner middleware
/// short-circuited the run.
struct LoggingMiddleware;

#[async_trait]
impl Middleware<AgentContext> for LoggingMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        println!(
            "  [logging] -> run starting ({} input message(s))",
            ctx.messages.len()
        );
        let ctx = next.run(ctx).await?;
        println!(
            "  [logging] <- run finished: {:?}",
            ctx.result.as_ref().map(AgentResponse::text)
        );
        Ok(ctx)
    }
}

/// Terminates the run before the model is ever called if the latest message
/// contains a blocked word, substituting a canned refusal. Registered second,
/// so it runs nested inside `LoggingMiddleware`.
struct BlockedWordsMiddleware {
    blocked: Vec<String>,
}

#[async_trait]
impl Middleware<AgentContext> for BlockedWordsMiddleware {
    async fn process(
        &self,
        mut ctx: AgentContext,
        next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        let last_text = ctx.messages.last().map(Message::text).unwrap_or_default();
        let hit = self
            .blocked
            .iter()
            .find(|word| last_text.to_lowercase().contains(word.as_str()));

        if let Some(word) = hit {
            println!("  [blocked-words] refusing request containing '{word}' -- model not called");
            ctx.result = Some(AgentResponse {
                messages: vec![Message::assistant(format!(
                    "Sorry, I can't help with requests containing '{word}'."
                ))],
                ..Default::default()
            });
            ctx.terminate = true;
            return Ok(ctx);
        }

        // No blocked word: continue the chain (the terminal handler, which
        // calls the underlying model, runs next).
        next.run(ctx).await
    }
}

/// A minimal offline stand-in for a model.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Ok(ChatResponse::from_text(
            "Sure, here's a normal, unblocked answer.",
        ))
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
    let agent = Agent::builder(CannedClient)
        .name("assistant")
        .middleware(Arc::new(LoggingMiddleware))
        .middleware(Arc::new(BlockedWordsMiddleware {
            blocked: vec!["classified".to_string()],
        }))
        .build();

    println!("-- run #1: an ordinary query --");
    let r1 = agent.run_once("What's a good pasta recipe?").await?;
    println!("final: {}\n", r1.text());

    println!("-- run #2: a query containing a blocked word --");
    let r2 = agent
        .run_once("Tell me something classified about the project.")
        .await?;
    println!("final: {}", r2.text());

    Ok(())
}
