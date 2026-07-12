//! custom_executors: implement the `Executor` trait by hand -- overriding
//! `snapshot_state`/`restore_state` -- versus a `FunctionExecutor` closure.
//! Both can hold running state across messages within one process, but only
//! the hand-written executor's state survives a checkpoint/restore round
//! trip onto a *fresh* executor instance: `FunctionExecutor` always reports
//! "nothing to snapshot".
//!
//! Runs fully offline using `InMemoryCheckpointStorage`.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example custom_executors
//! ```

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use async_trait::async_trait;
use serde_json::{json, Value};

/// A hand-written executor that keeps a running total across every message
/// it sees, and knows how to save/restore that total for checkpointing.
struct RunningTotal {
    id: String,
    total: Mutex<i64>,
}

impl RunningTotal {
    fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            total: Mutex::new(0),
        }
    }
}

#[async_trait]
impl Executor for RunningTotal {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let n = message.as_i64().unwrap_or(0);
        let total = {
            let mut guard = self.total.lock().unwrap();
            *guard += n;
            *guard
        };
        ctx.yield_output(json!({ "total": total })).await?;
        Ok(())
    }

    // Overriding these two is what makes this executor's state
    // checkpoint/restore-able; a `FunctionExecutor` cannot do this.
    async fn snapshot_state(&self) -> Option<Value> {
        Some(json!(*self.total.lock().unwrap()))
    }

    async fn restore_state(&self, state: Value) -> Result<()> {
        *self.total.lock().unwrap() = state.as_i64().unwrap_or(0);
        Ok(())
    }
}

/// The same running-total behavior as a `FunctionExecutor` closure. It can
/// mutate its captured state just as freely within one process, but has no
/// `restore_state` hook to plug into, so `snapshot_state` defaults to `None`
/// and the state is invisible to checkpointing.
fn function_running_total(id: &str) -> FunctionExecutor {
    let total = Arc::new(Mutex::new(0i64));
    FunctionExecutor::new(id.to_string(), move |message, ctx| {
        let total = total.clone();
        async move {
            let n = message.as_i64().unwrap_or(0);
            let current = {
                let mut guard = total.lock().unwrap();
                *guard += n;
                *guard
            };
            ctx.yield_output(json!({ "total": current })).await?;
            Ok(())
        }
    })
}

fn build_custom(storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    WorkflowBuilder::new()
        .add_executor(Arc::new(RunningTotal::new("counter")))
        .set_start("counter")
        .with_checkpointing(storage)
        .build()
}

fn build_function(storage: Arc<dyn CheckpointStorage>) -> Result<Workflow> {
    WorkflowBuilder::new()
        .add_executor(Arc::new(function_running_total("counter")))
        .set_start("counter")
        .with_checkpointing(storage)
        .build()
}

/// Run `workflow` once per value in `inputs`, returning the id of the
/// checkpoint written by the *last* run. Diffs the storage's checkpoint set
/// before/after each run instead of trusting timestamp ordering, since
/// several fast, in-memory runs can land in the same millisecond.
async fn run_and_track_latest_checkpoint(
    workflow: &Workflow,
    storage: &Arc<dyn CheckpointStorage>,
    inputs: [i64; 3],
    label: &str,
) -> Result<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut latest = String::new();
    for n in inputs {
        let run = workflow.run(json!(n)).await?;
        println!("{label}: +{n} -> total {:?}", run.last_output());
        for cp in storage.list(None).await? {
            if seen.insert(cp.checkpoint_id.clone()) {
                latest = cp.checkpoint_id;
            }
        }
    }
    Ok(latest)
}

#[tokio::main]
async fn main() -> Result<()> {
    let inputs = [10, 20, 5]; // running total ends at 35

    // --- Manual `Executor` impl: state survives a checkpoint/restore. -----
    let custom_storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let custom = build_custom(custom_storage.clone())?;
    let last_checkpoint =
        run_and_track_latest_checkpoint(&custom, &custom_storage, inputs, "custom executor")
            .await?;

    let fresh_custom = build_custom(custom_storage.clone())?; // brand-new RunningTotal, total = 0
    fresh_custom
        .run_from_checkpoint(&last_checkpoint, custom_storage)
        .await?;
    let after_resume = fresh_custom.run(json!(5)).await?; // continues from the restored total
    println!(
        "custom executor after restore + 5: {:?} (state survived the round trip)",
        after_resume.last_output()
    );
    assert_eq!(after_resume.last_output(), Some(json!({ "total": 40 })));

    // --- `FunctionExecutor` closure: state does NOT survive. ---------------
    let fn_storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let function = build_function(fn_storage.clone())?;
    let last_checkpoint =
        run_and_track_latest_checkpoint(&function, &fn_storage, inputs, "function executor")
            .await?;

    let fresh_function = build_function(fn_storage.clone())?; // brand-new closure state, total = 0
    fresh_function
        .run_from_checkpoint(&last_checkpoint, fn_storage)
        .await?;
    let after_resume = fresh_function.run(json!(5)).await?; // nothing was captured to restore
    println!(
        "function executor after restore + 5: {:?} (state was lost -- no snapshot to restore)",
        after_resume.last_output()
    );
    assert_eq!(after_resume.last_output(), Some(json!({ "total": 5 })));

    Ok(())
}
