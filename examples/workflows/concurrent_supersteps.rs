//! concurrent_supersteps: executors scheduled in the same superstep run
//! concurrently -- the runner drives a superstep's invocations with
//! `futures::future::join_all`, not one after another. Two ~300ms "workers"
//! fanned out from one source finish in about one sleep's worth of
//! wall-clock time, not two.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example concurrent_supersteps
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

const SLEEP: Duration = Duration::from_millis(300);

/// A "slow" worker that blocks its own superstep for `SLEEP` before replying.
fn sleepy_worker(id: &'static str) -> Arc<dyn Executor> {
    Arc::new(FunctionExecutor::new(id, move |message, ctx| async move {
        tokio::time::sleep(SLEEP).await;
        ctx.send_message(json!({ "worker": id, "input": message }))
            .await?;
        Ok(())
    }))
}

#[tokio::main]
async fn main() -> Result<()> {
    let dispatch = FunctionExecutor::new("dispatch", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let join = FunctionExecutor::new("join", |message, ctx| async move {
        ctx.yield_output(message).await?;
        Ok(())
    });

    let workflow = WorkflowBuilder::new()
        .add_executor(Arc::new(dispatch))
        .add_executor(sleepy_worker("worker_a"))
        .add_executor(sleepy_worker("worker_b"))
        .add_executor(Arc::new(join))
        .set_start("dispatch")
        .add_fan_out(
            "dispatch",
            vec!["worker_a".to_string(), "worker_b".to_string()],
        )
        .add_fan_in(vec!["worker_a".to_string(), "worker_b".to_string()], "join")
        .build()?;

    let started = Instant::now();
    let run = workflow.run(json!("go")).await?;
    let elapsed = started.elapsed();

    println!("both workers reported in: {:?}", run.last_output());
    println!("wall clock for the whole run: {elapsed:?} (one {SLEEP:?} sleep, not two)");
    let sequential_estimate = SLEEP * 2;
    assert!(
        elapsed < Duration::from_millis(550),
        "worker_a and worker_b share a superstep and should run concurrently \
         (~{SLEEP:?} total), not sequentially (~{sequential_estimate:?}); took {elapsed:?}"
    );

    Ok(())
}
