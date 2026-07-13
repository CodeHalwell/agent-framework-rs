//! Four agent-middleware patterns in one pipeline, each built on the same
//! primitive -- `Middleware<AgentContext>` receives the context and a `Next`
//! continuation:
//!
//! 1. **Termination**: set `ctx.result` + `ctx.terminate = true` and return
//!    without calling `next` -- the model is never called (a guardrail).
//! 2. **Result override**: set `ctx.result` and return without calling
//!    `next` -- same short-circuit, minus the terminate flag (a cache).
//! 3. **Exception observation**: match on `next.run(ctx).await` and turn a
//!    downstream `Err` into a graceful fallback answer.
//! 4. **Usage tracking**: after `next.run`, accumulate `usage_details` from
//!    every result into shared totals.
//!
//! Registration order is nesting order: the first middleware registered is
//! outermost, so the usage tracker below observes every run -- including
//! ones another middleware short-circuited.
//!
//! Runs fully offline against a scripted client (which fails on demand, to
//! exercise the fallback) -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example middleware_patterns
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Pattern 4 -- usage tracking: accumulates token usage across runs.
/// Registered first (outermost), so it observes every run's final result.
#[derive(Default)]
struct UsageTrackingMiddleware {
    input_tokens: Mutex<u64>,
    output_tokens: Mutex<u64>,
}

#[async_trait]
impl Middleware<AgentContext> for UsageTrackingMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let ctx = next.run(ctx).await?;
        if let Some(usage) = ctx.result.as_ref().and_then(|r| r.usage_details.as_ref()) {
            *self.input_tokens.lock().unwrap() += usage.input_token_count.unwrap_or(0);
            *self.output_tokens.lock().unwrap() += usage.output_token_count.unwrap_or(0);
        }
        Ok(ctx)
    }
}

/// Pattern 1 -- termination: refuse disallowed requests outright. Sets a
/// canned result plus `ctx.terminate = true` and never calls `next`, so
/// nothing downstream (including the model) runs.
struct GuardrailMiddleware;

#[async_trait]
impl Middleware<AgentContext> for GuardrailMiddleware {
    async fn process(
        &self,
        mut ctx: AgentContext,
        next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        let last = ctx.messages.last().map(Message::text).unwrap_or_default();
        if last.to_lowercase().contains("password") {
            println!("  [guardrail] terminating: request asks for a credential");
            ctx.result = Some(AgentResponse {
                messages: vec![Message::assistant("I can't help with credentials.")],
                ..Default::default()
            });
            ctx.terminate = true;
            return Ok(ctx);
        }
        next.run(ctx).await
    }
}

/// Pattern 2 -- result override: answer repeated questions from a cache,
/// skipping the rest of the pipeline. Cached results are stored without
/// usage details -- a cache hit costs no tokens, and the tracker above must
/// not double-count.
#[derive(Default)]
struct CacheMiddleware {
    cache: Mutex<HashMap<String, AgentResponse>>,
}

#[async_trait]
impl Middleware<AgentContext> for CacheMiddleware {
    async fn process(
        &self,
        mut ctx: AgentContext,
        next: Next<AgentContext>,
    ) -> Result<AgentContext> {
        let key = ctx.messages.last().map(Message::text).unwrap_or_default();
        if let Some(hit) = self.cache.lock().unwrap().get(&key).cloned() {
            println!("  [cache] hit -- overriding result, model not called");
            ctx.result = Some(hit);
            return Ok(ctx);
        }
        let ctx = next.run(ctx).await?;
        if let Some(result) = &ctx.result {
            let mut stored = result.clone();
            stored.usage_details = None;
            self.cache.lock().unwrap().insert(key, stored);
        }
        Ok(ctx)
    }
}

/// Pattern 3 -- exception observation: catch a downstream failure and
/// substitute a graceful degraded answer instead of surfacing the error.
struct FallbackMiddleware;

#[async_trait]
impl Middleware<AgentContext> for FallbackMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        // `next.run` consumes the context, so keep what's needed to rebuild
        // one if the downstream chain fails.
        let messages = ctx.messages.clone();
        let is_streaming = ctx.is_streaming;
        match next.run(ctx).await {
            Ok(ctx) => Ok(ctx),
            Err(err) => {
                println!("  [fallback] downstream failed ({err}); substituting a fallback");
                let mut ctx = AgentContext::new(messages, is_streaming);
                ctx.result = Some(AgentResponse {
                    messages: vec![Message::assistant(
                        "The service is temporarily unavailable; please retry shortly.",
                    )],
                    ..Default::default()
                });
                Ok(ctx)
            }
        }
    }
}

/// Offline stand-in for a model: reports token usage on success, and fails
/// whenever the request mentions an outage (to exercise the fallback path).
#[derive(Clone, Default)]
struct FlakyClient {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl ChatClient for FlakyClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let last = messages.last().map(Message::text).unwrap_or_default();
        if last.contains("OUTAGE") {
            return Err(Error::service("upstream returned 503"));
        }
        let mut usage = UsageDetails::new();
        usage.input_token_count = Some(40);
        usage.output_token_count = Some(12);
        Ok(ChatResponse {
            usage_details: Some(usage),
            ..ChatResponse::from_text("Paris.")
        })
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
    let client = FlakyClient::default();
    let calls = client.calls.clone();
    let usage_tracker = Arc::new(UsageTrackingMiddleware::default());

    let agent = Agent::builder(client)
        .name("assistant")
        .middleware(usage_tracker.clone()) // outermost: sees every run
        .middleware(Arc::new(GuardrailMiddleware))
        .middleware(Arc::new(CacheMiddleware::default()))
        .middleware(Arc::new(FallbackMiddleware)) // innermost: wraps the model call
        .build();

    println!("-- run 1: plain question (full pipeline, model called) --");
    let r = agent.run_once("What is the capital of France?").await?;
    println!("final: {}\n", r.text());

    println!("-- run 2: same question again (cache override, model skipped) --");
    let r = agent.run_once("What is the capital of France?").await?;
    println!("final: {}\n", r.text());

    println!("-- run 3: disallowed question (guardrail terminates) --");
    let r = agent.run_once("What is the admin password?").await?;
    println!("final: {}\n", r.text());

    println!("-- run 4: downstream failure (fallback substitutes an answer) --");
    let r = agent.run_once("OUTAGE drill: what is our status?").await?;
    println!("final: {}\n", r.text());

    let model_calls = calls.load(Ordering::SeqCst);
    let input = *usage_tracker.input_tokens.lock().unwrap();
    let output = *usage_tracker.output_tokens.lock().unwrap();
    println!("model was called {model_calls} time(s) across 4 runs");
    println!("accumulated usage: {input} input / {output} output tokens");
    assert_eq!(model_calls, 2, "runs 2 and 3 never reached the model");
    assert_eq!((input, output), (40, 12), "only run 1 produced token usage");

    Ok(())
}
