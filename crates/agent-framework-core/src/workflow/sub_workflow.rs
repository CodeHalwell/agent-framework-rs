//! Sub-workflow composition: [`WorkflowExecutor`] wraps a [`Workflow`] as an
//! [`Executor`] node in a parent workflow.
//!
//! Rust equivalent of Python's `_workflow_executor.py`. The wrapper runs a child
//! workflow to completion (or until it pauses awaiting input), forwards the
//! child's outputs onward as messages (or yields them directly), and intercepts
//! the child's requests — re-emitting them into the parent so the parent's
//! caller can answer via its own `send_responses`, which routes the response
//! back into the child.
//!
//! Divergences from Python (documented): the parent forwards child requests via
//! the standard `request_info` mechanism (reusing the child's `request_id`)
//! rather than Python's `SubWorkflowRequestMessage`/`SubWorkflowResponseMessage`
//! wrappers; a child failure propagates as an executor failure rather than a
//! `WorkflowErrorEvent`. Concurrent invocations are isolated by a per-invocation
//! run id, matching Python's per-execution isolation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;

use super::checkpoint::WorkflowCheckpoint;
use super::context::WorkflowContext;
use super::executor::Executor;
use super::request_info::RequestResponse;
use super::runner::{Workflow, WorkflowRun};
use crate::error::Result;

/// A single in-flight child execution: the paused run plus how many of its
/// outputs have already been forwarded to the parent.
struct ChildExecution {
    run: WorkflowRun,
    forwarded_outputs: usize,
}

#[derive(Default)]
struct WrapperState {
    /// run_id -> paused child execution.
    runs: HashMap<String, ChildExecution>,
    /// forwarded child request_id -> run_id (for routing responses back).
    request_map: HashMap<String, String>,
}

/// An [`Executor`] that runs a child [`Workflow`], enabling hierarchical
/// composition.
pub struct WorkflowExecutor {
    id: String,
    child: Workflow,
    allow_direct_output: bool,
    state: Mutex<WrapperState>,
}

impl WorkflowExecutor {
    /// Wrap `workflow` as an executor with the given `id`.
    ///
    /// By default the child's outputs are sent onward as messages from this
    /// node. Use [`WorkflowExecutor::with_direct_output`] to instead yield them
    /// directly as parent workflow outputs.
    pub fn new(id: impl Into<String>, workflow: Workflow) -> Self {
        let id = id.into();
        // A sub-workflow's state is checkpointed *by its parent*, embedded in
        // the parent's checkpoint (see `snapshot_state`). Its own storage would
        // write a second, independent series of checkpoints that nothing ever
        // resumes from — two sources of truth for one run. Upstream refuses the
        // configuration outright (#7097); detaching reaches the same outcome
        // without failing a constructor.
        if workflow.has_checkpoint_storage() {
            tracing::warn!(
                executor_id = %id,
                "detaching the sub-workflow's own checkpoint storage: a sub-workflow is \
                 checkpointed by its parent, embedded in the parent's checkpoint"
            );
        }
        Self {
            id,
            child: workflow.without_checkpoint_storage(),
            allow_direct_output: false,
            state: Mutex::new(WrapperState::default()),
        }
    }

    /// Yield the child's outputs directly to the parent's output stream instead
    /// of forwarding them as messages.
    pub fn with_direct_output(mut self, allow: bool) -> Self {
        self.allow_direct_output = allow;
        self
    }

    /// The wrapped child workflow.
    pub fn workflow(&self) -> &Workflow {
        &self.child
    }

    /// Forward new outputs and intercept new requests from a (possibly resumed)
    /// child execution.
    async fn process(
        &self,
        run_id: String,
        mut child: ChildExecution,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        // Forward only outputs produced since the last time we processed.
        let all_outputs = child.run.outputs();
        let new_outputs: Vec<Value> = all_outputs
            .into_iter()
            .skip(child.forwarded_outputs)
            .collect();
        child.forwarded_outputs += new_outputs.len();
        if self.allow_direct_output {
            for out in new_outputs {
                ctx.yield_output(out).await?;
            }
        } else {
            for out in new_outputs {
                ctx.send_message(out).await?;
            }
        }

        // Determine which child requests are newly outstanding.
        let pending = child.run.pending_requests();
        let mut to_forward = Vec::new();
        {
            let mut state = self.state.lock().unwrap();
            for pr in &pending {
                if !state.request_map.contains_key(&pr.request_id) {
                    state
                        .request_map
                        .insert(pr.request_id.clone(), run_id.clone());
                    to_forward.push((pr.request_id.clone(), pr.request_data.clone()));
                }
            }
            if pending.is_empty() {
                state.runs.remove(&run_id);
            } else {
                state.runs.insert(run_id.clone(), child);
            }
        }

        // Re-emit each intercepted request into the parent, reusing the child's
        // request id so responses correlate. The reply routes back to this node.
        let wrapper_id = ctx.executor_id().to_string();
        for (request_id, request_data) in to_forward {
            ctx.record_request_with_id(request_id, wrapper_id.clone(), request_data);
        }
        Ok(())
    }

    async fn handle_response(&self, resp: RequestResponse, ctx: &WorkflowContext) -> Result<()> {
        let request_id = resp.request_id.clone();
        let run_id = self.state.lock().unwrap().request_map.remove(&request_id);
        let Some(run_id) = run_id else {
            return Ok(());
        };
        let child = self.state.lock().unwrap().runs.remove(&run_id);
        let Some(mut child) = child else {
            return Ok(());
        };
        child.run.send_response(request_id, resp.data).await?;
        self.process(run_id, child, ctx).await
    }
}

/// One entry of [`WorkflowExecutor`]'s checkpointed state: a child run's
/// captured checkpoint plus how many of its outputs the parent already
/// forwarded.
#[derive(Serialize, Deserialize)]
struct CheckpointedChild {
    checkpoint: WorkflowCheckpoint,
    forwarded_outputs: usize,
}

/// [`WorkflowExecutor`]'s full checkpoint payload.
#[derive(Default, Serialize, Deserialize)]
struct CheckpointedState {
    /// run_id -> the child execution paused under it.
    runs: HashMap<String, CheckpointedChild>,
    /// forwarded child request_id -> run_id.
    request_map: HashMap<String, String>,
}

#[async_trait]
impl Executor for WorkflowExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    /// Embed each paused child run's own checkpoint in the parent's.
    ///
    /// Without this the parent's checkpoint recorded nothing about its
    /// sub-workflows, so a resumed parent met a child that had forgotten
    /// everything: its executors' state, its in-flight messages, its position
    /// in the superstep loop. Anything mid-progress was silently lost, and a
    /// response arriving for a forwarded request had no run to route back to.
    /// Mirrors upstream's `WorkflowExecutor.on_checkpoint_save` (#7097).
    async fn snapshot_state(&self) -> Option<Value> {
        // `capture_checkpoint_object` is async and the guard is not `Send`, so
        // the paused runs are moved out of the lock, captured, then moved back.
        // Taking ownership also means no borrow of `self.state` spans an await.
        let (mut owned_runs, request_map) = {
            let mut state = self.state.lock().unwrap();
            if state.runs.is_empty() && state.request_map.is_empty() {
                return None;
            }
            (std::mem::take(&mut state.runs), state.request_map.clone())
        };

        let mut runs = HashMap::new();
        for (run_id, child) in &owned_runs {
            match child.run.capture_checkpoint_object().await {
                Ok(checkpoint) => {
                    runs.insert(
                        run_id.clone(),
                        CheckpointedChild {
                            checkpoint,
                            forwarded_outputs: child.forwarded_outputs,
                        },
                    );
                }
                Err(e) => tracing::warn!(
                    executor_id = %self.id,
                    run_id = %run_id,
                    "skipping sub-workflow run in checkpoint: {e}"
                ),
            }
        }

        // Put the runs back. Anything inserted meanwhile wins — dropping a
        // newer run to restore a stale snapshot of it would lose work.
        {
            let mut state = self.state.lock().unwrap();
            for (run_id, child) in owned_runs.drain() {
                state.runs.entry(run_id).or_insert(child);
            }
        }

        serde_json::to_value(CheckpointedState { runs, request_map }).ok()
    }

    /// Rebuild each paused child run from its embedded checkpoint.
    ///
    /// A run whose checkpoint no longer matches the child graph is dropped with
    /// a warning rather than failing the whole restore: the rest of the parent
    /// is still resumable, and a mismatched child would misroute messages.
    /// Mirrors upstream's `WorkflowExecutor.on_checkpoint_restore` (#7097).
    async fn restore_state(&self, state: Value) -> Result<()> {
        let saved: CheckpointedState = serde_json::from_value(state).map_err(|e| {
            crate::error::Error::Workflow(format!(
                "sub-workflow executor '{}': malformed checkpoint state: {e}",
                self.id
            ))
        })?;

        let mut restored = WrapperState {
            runs: HashMap::new(),
            request_map: saved.request_map,
        };
        for (run_id, child) in saved.runs {
            match self
                .child
                .restore_run_from_checkpoint_object(child.checkpoint)
                .await
            {
                Ok(run) => {
                    restored.runs.insert(
                        run_id,
                        ChildExecution {
                            run,
                            forwarded_outputs: child.forwarded_outputs,
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        executor_id = %self.id,
                        run_id = %run_id,
                        "dropping unrestorable sub-workflow run: {e}"
                    );
                }
            }
        }
        // Requests whose run did not survive can never be answered; drop them
        // so a later response is ignored rather than routed into nothing.
        restored
            .request_map
            .retain(|_, run_id| restored.runs.contains_key(run_id));

        *self.state.lock().unwrap() = restored;
        Ok(())
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // A response to a previously-forwarded request?
        if let Some(resp) = RequestResponse::from_message(&message) {
            let known = self
                .state
                .lock()
                .unwrap()
                .request_map
                .contains_key(&resp.request_id);
            if known {
                return self.handle_response(resp, &ctx).await;
            }
        }

        // Otherwise treat the message as fresh input for a new child run.
        let run_id = uuid::Uuid::new_v4().to_string();
        let child_run = self.child.run(message).await?;
        self.process(
            run_id,
            ChildExecution {
                run: child_run,
                forwarded_outputs: 0,
            },
            &ctx,
        )
        .await
    }
}
