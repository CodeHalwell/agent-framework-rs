//! Automatic retries with `RetryingChatClient`: wrap any `ChatClient` in a
//! `RetryPolicy` (exponential backoff + jitter, `Retry-After` honored, only
//! transient statuses retried). Runs fully offline against a scripted flaky
//! client -- no API key or network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example retry_policy
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// A client that fails twice with retryable statuses, then succeeds --
/// standing in for a rate-limited / briefly unavailable provider.
#[derive(Clone)]
struct FlakyClient {
    calls: Arc<AtomicUsize>,
    script: Arc<Mutex<Vec<Result<ChatResponse>>>>,
}

#[async_trait]
impl ChatClient for FlakyClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
        let outcome = self.script.lock().unwrap().pop();
        match outcome {
            Some(outcome) => {
                println!(
                    "  attempt {n}: {}",
                    if outcome.is_ok() {
                        "success"
                    } else {
                        "transient failure"
                    }
                );
                outcome
            }
            None => Ok(ChatResponse::from_text("(script exhausted)")),
        }
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
    // Scripted outcomes, popped from the back: 503, then a 429 carrying a
    // server-provided Retry-After of 1s (which overrides the computed
    // backoff), then success. `Error::service_status` is what the shipped
    // providers produce for HTTP failures, including the parsed Retry-After.
    let flaky = FlakyClient {
        calls: Arc::new(AtomicUsize::new(0)),
        script: Arc::new(Mutex::new(vec![
            Ok(ChatResponse::from_text("Hello after two retries!")),
            Err(Error::service_status(429, "rate limited", Some(1.0))),
            Err(Error::service_status(503, "unavailable", None)),
        ])),
    };

    // The policy: up to 4 retries, 200ms initial backoff doubling per
    // attempt, capped at 2s. Non-retryable statuses (400, 401, ...) fail
    // immediately; `RetryOn::predicate` can override what counts as
    // transient. Jitter is on by default.
    let policy = RetryPolicy::with_max_retries(4)
        .initial_delay(Duration::from_millis(200))
        .max_delay(Duration::from_secs(2))
        .backoff_multiplier(2.0);

    let client = RetryingChatClient::new(flaky).with_policy(policy);

    // Works standalone or as the client behind a ChatAgent -- retries happen
    // beneath the function-invocation loop either way.
    let agent = ChatAgent::builder(client).name("resilient").build();

    println!("running (watch the attempts):");
    let response = agent.run_once("Hi!").await?;
    println!("final answer: {}", response.text());

    Ok(())
}
