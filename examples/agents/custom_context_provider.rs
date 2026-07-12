//! A custom `ContextProvider`: injects extra instructions before every run
//! (`before_run`, mutating a `&mut SessionContext` in place), and observes
//! `after_run`, which fires after every run on both the success and failure
//! paths (`after_run`'s `error` argument carries the failure on the latter).
//!
//! Runs fully offline against three small canned clients -- no API key or
//! network needed.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example custom_context_provider
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Injects a turn-counter instruction on every `before_run` call, and prints
/// whenever either `ContextProvider` hook fires.
#[derive(Default)]
struct DemoProvider {
    turn: AtomicUsize,
}

#[async_trait]
impl ContextProvider for DemoProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let n = self.turn.fetch_add(1, Ordering::SeqCst) + 1;
        println!("  [context] before_run (call #{n}) -- injecting an instruction");
        ctx.add_instructions(format!("This is invocation #{n}."));
        Ok(())
    }

    async fn after_run(
        &self,
        request: &[Message],
        response: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        match error {
            Some(e) => println!("  [context] after_run: run FAILED: {e}"),
            None => println!(
                "  [context] after_run: run OK ({} request message(s), {} response message(s))",
                request.len(),
                response.len()
            ),
        }
        Ok(())
    }
}

/// Echoes back whatever `conversation_id` it was given, keeping a
/// service-managed thread valid across turns.
#[derive(Clone)]
struct EchoingClient;

#[async_trait]
impl ChatClient for EchoingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok, using the existing service thread");
        resp.conversation_id = options.conversation_id;
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Always mints a fresh conversation id, so a plain local thread adopts it as
/// a service-managed thread.
#[derive(Clone)]
struct AdoptingClient;

#[async_trait]
impl ChatClient for AdoptingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut resp = ChatResponse::from_text("ok, minted a new service thread");
        resp.conversation_id = Some("adopted-1".to_string());
        Ok(resp)
    }
    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

/// Always fails, to demonstrate `after_run`'s `error` argument.
#[derive(Clone)]
struct FailingClient;

#[async_trait]
impl ChatClient for FailingClient {
    async fn get_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        Err(Error::service("simulated outage"))
    }
    async fn get_streaming_response(
        &self,
        _messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatStream> {
        Err(Error::service("simulated outage"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let provider: Arc<dyn ContextProvider> = Arc::new(DemoProvider::default());

    println!("-- scenario 1: an already service-managed thread --");
    let agent1 = Agent::builder(EchoingClient)
        .context_provider(provider.clone())
        .build();
    let mut thread1 = agent1.get_new_thread_with_service_id("demo-thread")?;
    let r1 = agent1
        .run(vec![Message::user("hi")], Some(&mut thread1))
        .await?;
    println!("agent: {}\n", r1.text());

    println!("-- scenario 2: a local thread adopts a service id from the response --");
    let agent2 = Agent::builder(AdoptingClient)
        .context_provider(provider.clone())
        .build();
    let mut thread2 = agent2.get_new_thread();
    let r2 = agent2
        .run(vec![Message::user("hi")], Some(&mut thread2))
        .await?;
    println!(
        "agent: {} (thread adopted id: {:?})\n",
        r2.text(),
        thread2.service_thread_id()
    );

    println!("-- scenario 3: after_run observes a failed run --");
    let agent3 = Agent::builder(FailingClient)
        .context_provider(provider)
        .build();
    match agent3.run_once("hi").await {
        Ok(_) => unreachable!("FailingClient always errors"),
        Err(e) => println!("run failed as expected (propagated to the caller too): {e}"),
    }

    Ok(())
}
