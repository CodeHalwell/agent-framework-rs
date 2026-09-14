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

/// Executor ids are only required to be non-empty, so a signature that
/// renders them by joining with separators is not injective: a caller whose
/// ids happen to contain `,` or `->` can produce two structurally different
/// graphs that hash identically. A checkpoint from either would then be
/// accepted for the other — resumed onto a topology it was never written
/// for, which is exactly what the signature exists to prevent.
#[test]
fn ids_containing_separators_do_not_collide() {
    // Two different graphs, distinguishable only by where the separator falls.
    let node = |id: &str| Arc::new(FunctionExecutor::new(id, |_m, _c| async move { Ok(()) }));
    // A fans out to two executors; B fans out to one whose id happens to
    // contain the separator. Under a joined rendering both the node list
    // ("s,x,y") and the edge descriptor ("fanout:s->[x,y]") come out
    // character-for-character identical.
    let a = WorkflowBuilder::new()
        .add_executor(node("s"))
        .add_executor(node("x"))
        .add_executor(node("y"))
        .set_start("s")
        .add_fan_out("s", vec!["x".to_string(), "y".to_string()])
        .build()
        .expect("build a");
    let b = WorkflowBuilder::new()
        .add_executor(node("s"))
        .add_executor(node("x,y"))
        .set_start("s")
        .add_fan_out("s", vec!["x,y".to_string()])
        .build()
        .expect("build b");

    assert_ne!(
        a.graph_signature(),
        b.graph_signature(),
        "a fan-out to two targets must not sign the same as one target named after both"
    );
}

/// A checkpoint written by an older signature *scheme* has not necessarily
/// been written for a different graph — the encoding changed under it. The
/// mismatch must say so, rather than sending the reader looking for a change
/// to their own graph that is not there.
#[tokio::test]
async fn an_older_signature_scheme_is_reported_as_such() {
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let cp = mid_run_checkpoint(&storage).await;

    let mut value = serde_json::to_value(&cp).unwrap();
    value
        .as_object_mut()
        .unwrap()
        .insert("graph_signature".into(), json!("v1-0123456789abcdef"));
    let old_scheme: WorkflowCheckpoint = serde_json::from_value(value).unwrap();
    let old_storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let id = old_storage.save(old_scheme).await.unwrap();

    let err = match build_pipeline(Some(storage.clone()), false)
        .run_from_checkpoint(&id, old_storage.clone())
        .await
    {
        Ok(_) => panic!("an older scheme cannot be compared"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("scheme mismatch"), "{err}");
    assert!(
        err.contains("may well be identical"),
        "the message must not claim the graph changed: {err}"
    );

    // And the override still resumes it, which is the way out the message
    // points at.
    let run = match build_pipeline(Some(storage), false)
        .run_from_checkpoint_unchecked(&id, old_storage)
        .await
    {
        Ok(run) => run,
        Err(e) => panic!("unchecked resume ignores the scheme mismatch: {e}"),
    };
    assert_eq!(run.state(), WorkflowRunState::Idle);
}
