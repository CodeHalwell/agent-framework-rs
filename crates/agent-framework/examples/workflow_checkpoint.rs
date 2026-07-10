//! Workflow checkpointing: `FileCheckpointStorage` persists run state after
//! every superstep; `list` inspects what was saved; `run_from_checkpoint`
//! resumes a *paused-in-the-middle* run on a brand-new `Workflow` instance,
//! continuing from the in-flight messages rather than re-running from the
//! start.
//!
//! Runs fully offline (no LLM calls) using `FunctionExecutor` nodes, so it
//! needs no API key or network access.
//!
//! ```bash
//! cargo run -p agent-framework --example workflow_checkpoint
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::{get_checkpoint_summary, FunctionExecutor};
use serde_json::{json, Value};

/// A three-step pipeline: double -> add ten -> report. Each step appends its
/// name to a `log` entry in `SharedState`, so the log -- like the in-flight
/// message queue -- is part of what gets checkpointed and later restored.
///
/// `storage` is optional so the same builder can produce both the original
/// (checkpointing) workflow and the fresh workflow we resume into later.
fn build_pipeline(storage: Option<Arc<dyn CheckpointStorage>>) -> Result<Workflow> {
    let double = FunctionExecutor::new("double", |message, ctx| async move {
        let n = message.as_i64().unwrap_or(0);
        log_step(&ctx, "double").await;
        ctx.send_message(json!(n * 2)).await?;
        Ok(())
    });
    let add_ten = FunctionExecutor::new("add_ten", |message, ctx| async move {
        let n = message.as_i64().unwrap_or(0);
        log_step(&ctx, "add_ten").await;
        ctx.send_message(json!(n + 10)).await?;
        Ok(())
    });
    let report = FunctionExecutor::new("report", |message, ctx| async move {
        log_step(&ctx, "report").await;
        let log = ctx.shared_state().get("log").await.unwrap_or(json!([]));
        ctx.yield_output(json!({ "result": message, "log": log }))
            .await?;
        Ok(())
    });

    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(double))
        .add_executor(Arc::new(add_ten))
        .add_executor(Arc::new(report))
        .set_start("double")
        .add_chain(vec![
            "double".to_string(),
            "add_ten".to_string(),
            "report".to_string(),
        ]);
    if let Some(storage) = storage {
        builder = builder.with_checkpointing(storage);
    }
    builder.build()
}

/// Append `name` to the run-scoped "log" list kept in `SharedState`.
async fn log_step(ctx: &WorkflowContext, name: &str) {
    ctx.shared_state()
        .update("log", move |current| {
            let mut log: Vec<Value> = current
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            log.push(json!(name));
            json!(log)
        })
        .await;
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("agent-framework-example-checkpoints");
    let _ = std::fs::remove_dir_all(&dir); // start from a clean slate each run
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);
    println!("checkpoints written to {}", dir.display());

    let workflow = build_pipeline(Some(storage.clone()))?;
    let run = workflow.run(json!(3)).await?;
    println!("full run output: {:?}", run.last_output());

    // Three supersteps ran (double -> add_ten -> report), so three
    // checkpoints were written -- one per superstep boundary.
    let mut checkpoints = storage.list(Some(workflow.id())).await?;
    checkpoints.sort_by_key(|cp| cp.iteration_count);
    println!("saved {} checkpoint(s):", checkpoints.len());
    for cp in &checkpoints {
        let summary = get_checkpoint_summary(cp);
        println!(
            "  checkpoint {} @ superstep {} ({})",
            summary.checkpoint_id, summary.iteration_count, summary.status
        );
    }

    // Resume from the *first* checkpoint (right after "double" ran, with
    // "add_ten" still pending) on a brand-new Workflow/executor set. It
    // should reach the same final result without ever re-running "double".
    let checkpoint_after_double = &checkpoints[0];
    let fresh_workflow = build_pipeline(None)?;
    let resumed = fresh_workflow
        .run_from_checkpoint(&checkpoint_after_double.checkpoint_id, storage.clone())
        .await?;
    println!(
        "resumed from mid-pipeline -> state: {:?}, output: {:?}",
        resumed.state(),
        resumed.last_output()
    );
    assert_eq!(resumed.last_output(), run.last_output());

    Ok(())
}
