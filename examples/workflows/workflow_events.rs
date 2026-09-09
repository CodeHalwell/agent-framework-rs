//! The workflow event stream: every `WorkflowEvent` the engine emits, what
//! each one means, and the three ways an executor can put data on the stream.
//!
//! `workflow.run(..)` buffers events and hands them back on the finished
//! `WorkflowRun` (`run.events()`). `workflow.run_stream(..)` yields them as
//! they happen, which is what you want behind a progress bar, a log, or an
//! SSE endpoint. Both see the same events; this example uses the streaming
//! form so the ordering is visible.
//!
//! The three ways to emit:
//!
//! - `ctx.yield_output(v)` from an executor named by `output_from` (or from
//!   any executor when no designation is configured) -> `Output`, which is
//!   **terminal**: it becomes part of `run.last_output()`.
//! - `ctx.yield_output(v)` from an executor named by
//!   `intermediate_output_from` -> `Intermediate`, a progress signal that is
//!   never recorded as the run's result.
//! - `ctx.add_event(WorkflowEvent::Custom(v))` -> `Custom`, an arbitrary
//!   application event the engine passes through untouched. This is the hook
//!   for domain telemetry ("scored 12 candidates", "cache hit") that does not
//!   belong in the data flowing along the edges.
//!
//! Runs fully offline.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example workflow_events
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use futures::StreamExt;
use serde_json::{json, Value};

/// A one-line rendering of an event, so the stream reads as a trace.
fn render(event: &WorkflowEvent) -> String {
    match event {
        WorkflowEvent::Started => "Started".into(),
        WorkflowEvent::Status(state) => format!("Status({state:?})"),
        WorkflowEvent::SuperStepStarted(n) => format!("SuperStepStarted({n})"),
        WorkflowEvent::SuperStepCompleted(n) => format!("SuperStepCompleted({n})"),
        WorkflowEvent::ExecutorInvoked { executor_id } => format!("ExecutorInvoked({executor_id})"),
        WorkflowEvent::ExecutorCompleted { executor_id } => {
            format!("ExecutorCompleted({executor_id})")
        }
        WorkflowEvent::ExecutorFailed { executor_id, error } => {
            format!("ExecutorFailed({executor_id}): {error}")
        }
        WorkflowEvent::AgentRunUpdate { executor_id, .. } => {
            format!("AgentRunUpdate({executor_id})  <- one streamed chunk")
        }
        WorkflowEvent::AgentRun { executor_id, .. } => format!("AgentRun({executor_id})"),
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => format!("Output from {source_executor_id}: {data}   [TERMINAL]"),
        WorkflowEvent::Intermediate {
            data,
            source_executor_id,
        } => format!("Intermediate from {source_executor_id}: {data}"),
        WorkflowEvent::Custom(data) => format!("Custom: {data}"),
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            ..
        } => format!("RequestInfo({request_id}) from {source_executor_id}"),
        WorkflowEvent::Failed { error } => format!("Failed: {error}"),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Stage 1: reports progress as it goes, but its yields are *not* the
    // workflow's answer -- they are designated intermediate below.
    let extract = FunctionExecutor::new("extract", |message: Value, ctx| async move {
        let text = message.as_str().unwrap_or_default().to_string();
        let words: Vec<&str> = text.split_whitespace().collect();

        // A domain event: not data on an edge, not an output -- telemetry.
        ctx.add_event(WorkflowEvent::Custom(json!({
            "stage": "extract",
            "word_count": words.len(),
        })));

        // A progress signal a UI can render while the run is still going.
        ctx.yield_output(json!(format!("extracted {} word(s)", words.len())))
            .await?;

        ctx.send_message(json!({ "words": words })).await?;
        Ok(())
    });

    // Stage 2: the node that actually produces the answer.
    let summarise = FunctionExecutor::new("summarise", |message: Value, ctx| async move {
        let words = message["words"].as_array().cloned().unwrap_or_default();
        let longest = words
            .iter()
            .filter_map(Value::as_str)
            .max_by_key(|w| w.len())
            .unwrap_or("");

        ctx.add_event(WorkflowEvent::Custom(json!({
            "stage": "summarise",
            "longest_word": longest,
        })));

        ctx.yield_output(json!({ "longest_word": longest, "total": words.len() }))
            .await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .name("event-tour")
        .add_executor(Arc::new(extract))
        .add_executor(Arc::new(summarise))
        .set_start("extract")
        .add_edge("extract", "summarise")
        // The designation that splits terminal from non-terminal yields.
        // Without these two lines every yield would be an `Output`.
        .output_from(["summarise"])
        .intermediate_output_from(["extract"])
        .build()?;

    println!("== streaming the event trace ==\n");

    let mut stream = workflow.run_stream(json!("the quick brown fox jumps over the lazy dog"));
    while let Some(event) = stream.next().await {
        println!("  {}", render(&event));
    }

    // `into_run` finishes the stream and hands back the same completed run
    // `workflow.run(..)` would have produced.
    let run = stream.into_run().await?;

    println!("\n== what survived as the run's result ==\n");
    println!("  state:       {:?}", run.state());
    println!("  outputs:     {:?}", run.outputs());
    println!("  last_output: {:?}", run.last_output());
    println!(
        "\n  Note `extracted 9 word(s)` is nowhere in `outputs()`: it was an\n  \
         Intermediate. Only `summarise`'s yield counts, because only it was\n  \
         named in `output_from`."
    );

    println!("\n== picking events back out of a completed run ==\n");

    // The buffered form: identical events, available after the fact. Handy
    // for tests and for post-hoc analysis of a run you did not stream.
    let run = workflow
        .run(json!("a much shorter sentence here"))
        .await?;

    let customs: Vec<&Value> = run
        .events()
        .iter()
        .filter_map(|e| match e {
            WorkflowEvent::Custom(v) => Some(v),
            _ => None,
        })
        .collect();
    println!("  custom events: {customs:?}");

    let executors_run: Vec<&str> = run
        .events()
        .iter()
        .filter_map(|e| match e {
            WorkflowEvent::ExecutorCompleted { executor_id } => Some(executor_id.as_str()),
            _ => None,
        })
        .collect();
    println!("  executors that completed: {executors_run:?}");
    println!("  total events buffered: {}", run.events().len());

    println!(
        "\nnote: `ExecutorFailed` and `Failed` appear when a node returns an\n\
         error; `RequestInfo` when one pauses for human input (see the\n\
         `workflow_hitl` example); `AgentRunUpdate`/`AgentRun` when a node\n\
         wraps an `Agent` (see `agents_in_workflows` and `streaming_updates`)."
    );

    Ok(())
}
