//! loops_and_max_iterations: a self-looping executor forms a cycle in the
//! graph (the engine only requires every node be reachable from the start
//! node -- cycles are fine). `set_max_iterations` bounds how many supersteps
//! a run may take before it fails, which is what keeps a stuck loop from
//! running forever.
//!
//! Runs fully offline (no LLM calls) using a single `FunctionExecutor`.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example loops_and_max_iterations
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

const TARGET: i64 = 5;

/// A counter that increments on every visit and loops back to itself until it
/// reaches `TARGET`, at which point it yields the final count instead.
fn build(max_iterations: usize) -> Result<Workflow> {
    let counter = FunctionExecutor::new("counter", |message, ctx| async move {
        let n = message.as_i64().unwrap_or(0) + 1;
        println!("  superstep: counter = {n}");
        if n >= TARGET {
            ctx.yield_output(json!({ "final": n })).await?;
        } else {
            ctx.send_message(json!(n)).await?; // feeds back into "counter"
        }
        Ok(())
    });

    WorkflowBuilder::new()
        .add_executor(Arc::new(counter))
        .set_start("counter")
        .add_edge("counter", "counter")
        .set_max_iterations(max_iterations)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("-- enough headroom (max_iterations = 10) --");
    let workflow = build(10)?;
    let run = workflow.run(json!(0)).await?;
    println!("reached: {:?}\n", run.last_output());
    assert_eq!(run.last_output(), Some(json!({ "final": TARGET })));

    println!("-- too tight a bound (max_iterations = 3) --");
    let bounded = build(3)?;
    match bounded.run(json!(0)).await {
        Ok(_) => unreachable!("expected this run to hit max_iterations first"),
        Err(e) => println!("run failed as expected: {e}"),
    }

    Ok(())
}
