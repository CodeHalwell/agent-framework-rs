//! Human-in-the-loop workflows: `RequestInfoExecutor` pauses a run pending
//! external input; the caller inspects `pending_requests()`, supplies an
//! answer via `send_response`, and the run resumes to completion.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes, so it
//! needs no API key or network access.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example workflow_hitl
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{FunctionExecutor, RequestResponse};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    // "asker" forwards a fresh question on to the request-info node. When the
    // human's answer later comes back to it as a `RequestResponse` message,
    // it yields that answer as the workflow's output instead of forwarding.
    let asker = FunctionExecutor::new("asker", |message, ctx| async move {
        match RequestResponse::from_message(&message) {
            Some(response) => {
                ctx.yield_output(json!({ "answer": response.data })).await?;
            }
            None => {
                ctx.send_message(message).await?;
            }
        }
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(asker))
        .add_executor(Arc::new(RequestInfoExecutor::new("ask_human")))
        .set_start("asker")
        .add_edge("asker", "ask_human")
        .build()?;

    let mut run = workflow
        .run(json!("What is your favorite programming language?"))
        .await?;
    println!("state after first pass: {:?}", run.state());
    assert_eq!(run.state(), WorkflowRunState::IdleWithPendingRequests);

    // Inspect what's pending -- in a real application this is where you'd
    // surface a prompt to a human (CLI, chat UI, ticket queue, ...).
    let pending = run.pending_requests();
    for req in &pending {
        println!("pending request {}: {:?}", req.request_id, req.request_data);
    }
    assert_eq!(pending.len(), 1);

    // Answer it and resume. `send_response` routes the answer back to
    // whichever executor originally made the request ("asker" here).
    let request_id = pending[0].request_id.clone();
    run.send_response(request_id, json!("Rust")).await?;

    println!("state after response: {:?}", run.state());
    println!("final output: {:?}", run.last_output());
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!({ "answer": "Rust" })));

    Ok(())
}
