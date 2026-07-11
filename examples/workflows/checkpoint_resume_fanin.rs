//! checkpoint_resume_fanin: take a checkpoint while a fan-in barrier is only
//! partially satisfied -- one source has delivered, the other is still an
//! extra hop away -- then resume on a fresh `Workflow` and confirm the
//! fan-in still fires once the missing source arrives. Exercises the same
//! `fanin_state` capture the engine uses for any checkpoint taken mid-barrier.
//!
//! Runs fully offline using `FunctionExecutor` nodes and `FileCheckpointStorage`
//! in a temp directory (see `checkpoint.rs` for the plain, non-fan-in case).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example checkpoint_resume_fanin
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::workflow::FunctionExecutor;
use serde_json::json;

/// `start` fans out to `left` and `extra_hop`. `left` reports straight to
/// the fan-in; `extra_hop` adds one more hop through `right` before *it*
/// reports to the fan-in -- so the two fan-in sources delivered a superstep
/// apart, and a checkpoint taken in between catches the barrier
/// half-satisfied.
fn build(storage: Option<Arc<dyn CheckpointStorage>>) -> Result<Workflow> {
    let start = FunctionExecutor::new("start", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let left = FunctionExecutor::new("left", |message, ctx| async move {
        ctx.send_message(json!({ "side": "left", "value": message }))
            .await?;
        Ok(())
    });
    let extra_hop = FunctionExecutor::new("extra_hop", |message, ctx| async move {
        ctx.send_message(message).await?;
        Ok(())
    });
    let right = FunctionExecutor::new("right", |message, ctx| async move {
        ctx.send_message(json!({ "side": "right", "value": message }))
            .await?;
        Ok(())
    });
    let joiner = FunctionExecutor::new("joiner", |message, ctx| async move {
        ctx.yield_output(json!({ "joined": message })).await?;
        Ok(())
    });

    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(start))
        .add_executor(Arc::new(left))
        .add_executor(Arc::new(extra_hop))
        .add_executor(Arc::new(right))
        .add_executor(Arc::new(joiner))
        .set_start("start")
        .add_fan_out("start", vec!["left".to_string(), "extra_hop".to_string()])
        .add_edge("extra_hop", "right")
        .add_fan_in(vec!["left".to_string(), "right".to_string()], "joiner");
    if let Some(storage) = storage {
        builder = builder.with_checkpointing(storage);
    }
    builder.build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("agent-framework-example-checkpoint-resume-fanin");
    let _ = std::fs::remove_dir_all(&dir); // start from a clean slate each run
    let storage: Arc<dyn CheckpointStorage> = Arc::new(FileCheckpointStorage::new(&dir)?);
    println!("checkpoints written to {}", dir.display());

    let workflow = build(Some(storage.clone()))?;
    let full_run = workflow.run(json!(1)).await?;
    println!("full run output: {:?}", full_run.last_output());

    // Exactly one checkpoint catches the fan-in half satisfied: "left" has
    // already delivered to the "joiner" barrier, but "right"'s trigger
    // message is still one hop away (via "extra_hop"), so it has not.
    let checkpoints = storage.list(Some(workflow.id())).await?;
    let partial = checkpoints
        .iter()
        .find(|cp| !cp.fanin_state.is_empty())
        .expect("one checkpoint should catch the fan-in half-satisfied");
    let buffered = &partial.fanin_state["joiner"];
    println!(
        "checkpoint {} @ superstep {}: fan-in has {}/2 sources buffered ({:?})",
        partial.checkpoint_id,
        partial.iteration_count,
        buffered.len(),
        buffered.keys().collect::<Vec<_>>()
    );
    assert_eq!(
        buffered.len(),
        1,
        "only \"left\" should have reported in yet"
    );
    assert!(buffered.contains_key("left"));

    // Resume on a brand-new Workflow/executor set from that half-satisfied
    // checkpoint. "right"'s message is still queued, so this next superstep
    // completes the barrier and "joiner" fires -- exactly as it would have
    // in the uninterrupted run.
    let fresh = build(None)?;
    let resumed = fresh
        .run_from_checkpoint(&partial.checkpoint_id, storage)
        .await?;
    println!(
        "resumed from the half-satisfied fan-in -> state: {:?}, output: {:?}",
        resumed.state(),
        resumed.last_output()
    );
    assert_eq!(resumed.state(), WorkflowRunState::Idle);
    assert_eq!(resumed.last_output(), full_run.last_output());

    Ok(())
}
