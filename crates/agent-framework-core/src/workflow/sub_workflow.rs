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
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Mutex;

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
        Self {
            id: id.into(),
            child: workflow,
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

#[async_trait]
impl Executor for WorkflowExecutor {
    fn id(&self) -> &str {
        &self.id
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
