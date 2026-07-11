//! Checkpoint graph-signature validation: a resumed checkpoint must come from
//! a graph whose topology matches the resuming workflow. Covers a same-graph
//! roundtrip, a changed-graph rejection (with an actionable message naming both
//! signatures), the `run_from_checkpoint_unchecked` override, and legacy
//! signatureless checkpoints loading with a warning. No network.

use std::sync::Arc;

use agent_framework_core::workflow::{
    CheckpointStorage, FunctionExecutor, InMemoryCheckpointStorage, Workflow, WorkflowBuilder,
    WorkflowCheckpoint, WorkflowRunState,
};
use serde_json::json;

/// A three-stage pipeline `p1 -> p2 -> p3` that accumulates into shared state
/// and yields the sum at `p3`. `extra` adds a fourth node `p4` (fed by `p3`)
/// to produce a *different* graph with the same core routing.
fn build_pipeline(storage: Option<Arc<dyn CheckpointStorage>>, extra: bool) -> Workflow {
    let p1 = FunctionExecutor::new("p1", |msg, ctx| async move {
        let n = msg.as_i64().unwrap_or(0);
        ctx.shared_state()
            .update("sum", move |cur| {
                json!(cur.and_then(|v| v.as_i64()).unwrap_or(0) + n)
            })
            .await;
        ctx.send_message(json!(n)).await?;
        Ok(())
    });
    let p2 = FunctionExecutor::new("p2", |msg, ctx| async move {
        ctx.send_message(msg).await?;
        Ok(())
    });
    let p3 = FunctionExecutor::new("p3", |_msg, ctx| async move {
        let sum = ctx.shared_state().get("sum").await.unwrap_or(json!(0));
        ctx.yield_output(sum).await?;
        Ok(())
    });

    let mut builder = WorkflowBuilder::new()
        .add_executor(Arc::new(p1))
        .add_executor(Arc::new(p2))
        .add_executor(Arc::new(p3))
        .set_start("p1")
        .add_edge("p1", "p2")
        .add_edge("p2", "p3");

    if extra {
        let p4 = FunctionExecutor::new("p4", |_msg, _ctx| async move { Ok(()) });
        builder = builder.add_executor(Arc::new(p4)).add_edge("p3", "p4");
    }
    if let Some(s) = storage {
        builder = builder.with_checkpointing(s);
    }
    builder.build().unwrap()
}

/// Run the base pipeline once and return a mid-run checkpoint (one in-flight
/// message, `iteration_count == 1`).
async fn mid_run_checkpoint(storage: &Arc<dyn CheckpointStorage>) -> WorkflowCheckpoint {
    let workflow = build_pipeline(Some(storage.clone()), false);
    let run = workflow.run(json!(10)).await.unwrap();
    assert_eq!(run.last_output(), Some(json!(10)));
    storage
        .list(None)
        .await
        .unwrap()
        .into_iter()
        .find(|c| c.iteration_count == 1)
        .expect("a mid-run checkpoint")
}

#[test]
fn signature_is_deterministic_and_topology_sensitive() {
    // Two independent builds of the same graph agree; adding a node/edge does
    // not.
    let a = build_pipeline(None, false);
    let b = build_pipeline(None, false);
    let extended = build_pipeline(None, true);

    assert!(!a.graph_signature().is_empty());
    assert_eq!(
        a.graph_signature(),
        b.graph_signature(),
        "identical graphs share a signature regardless of build instance"
    );
    assert_ne!(
        a.graph_signature(),
        extended.graph_signature(),
        "adding a node + edge changes the signature"
    );
}

#[tokio::test]
async fn same_graph_roundtrip_passes() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let cp = mid_run_checkpoint(&storage).await;
    assert!(
        !cp.graph_signature.is_empty(),
        "checkpoint records a signature"
    );

    let resumed = build_pipeline(Some(storage.clone()), false);
    let run = resumed
        .run_from_checkpoint(&cp.checkpoint_id, storage.clone())
        .await
        .expect("resuming an identical graph succeeds");
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!(10)));
}

#[tokio::test]
async fn changed_graph_is_rejected_with_actionable_message() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let cp = mid_run_checkpoint(&storage).await;

    let changed = build_pipeline(Some(storage.clone()), true);
    let err = match changed
        .run_from_checkpoint(&cp.checkpoint_id, storage.clone())
        .await
    {
        Ok(_) => panic!("resuming a changed graph must fail"),
        Err(e) => e,
    };

    let msg = err.to_string();
    assert!(msg.contains("graph signature mismatch"), "message: {msg}");
    // Both signatures are named so the operator can see what changed.
    assert!(
        msg.contains(&cp.graph_signature),
        "names checkpoint sig: {msg}"
    );
    assert!(
        msg.contains(changed.graph_signature()),
        "names workflow sig: {msg}"
    );
    assert!(
        msg.contains("run_from_checkpoint_unchecked"),
        "points at the override: {msg}"
    );
}

#[tokio::test]
async fn unchecked_override_bypasses_validation() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let cp = mid_run_checkpoint(&storage).await;

    // The extended graph is a strict superset for routing purposes, so the
    // in-flight message still drives to the same output when forced through.
    let changed = build_pipeline(Some(storage.clone()), true);
    let run = changed
        .run_from_checkpoint_unchecked(&cp.checkpoint_id, storage.clone())
        .await
        .expect("unchecked resume ignores the signature mismatch");
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!(10)));
}

#[tokio::test]
async fn legacy_signatureless_checkpoint_loads() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let cp = mid_run_checkpoint(&storage).await;

    // Simulate a checkpoint written before signatures existed: strip the field
    // from the JSON entirely, so it deserializes back with the serde default
    // (empty signature).
    let mut value = serde_json::to_value(&cp).unwrap();
    value.as_object_mut().unwrap().remove("graph_signature");
    let legacy: WorkflowCheckpoint = serde_json::from_value(value).unwrap();
    assert!(legacy.graph_signature.is_empty());

    let legacy_storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let legacy_id = legacy_storage.save(legacy).await.unwrap();

    // A legacy checkpoint resumes (with an internal warning) rather than erroring.
    let resumed = build_pipeline(Some(storage.clone()), false);
    let run = resumed
        .run_from_checkpoint(&legacy_id, legacy_storage.clone())
        .await
        .expect("a signatureless checkpoint still loads");
    assert_eq!(run.state(), WorkflowRunState::Idle);
    assert_eq!(run.last_output(), Some(json!(10)));
}
