//! Workflow events emitted during execution.

use serde_json::Value;

/// The run state of a workflow, mirroring `WorkflowRunState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowRunState {
    Started,
    InProgress,
    InProgressPendingRequests,
    Idle,
    IdleWithPendingRequests,
    Failed,
    Cancelled,
}

/// An event observed while a workflow runs.
///
/// Corresponds to the `WorkflowEvent` hierarchy in the Python engine. Only
/// workflow-level events are observable; inter-executor messages are internal.
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// The run has begun.
    Started,
    /// A run-state transition.
    Status(WorkflowRunState),
    /// A superstep has started (with its iteration index).
    SuperStepStarted(usize),
    /// A superstep has completed.
    SuperStepCompleted(usize),
    /// An executor began processing a message.
    ExecutorInvoked { executor_id: String },
    /// An executor finished processing.
    ExecutorCompleted { executor_id: String },
    /// An executor failed.
    ExecutorFailed { executor_id: String, error: String },
    /// A workflow-level output was yielded.
    Output {
        data: Value,
        source_executor_id: String,
    },
    /// A custom event added by an executor.
    Custom(Value),
    /// A request for external input (human-in-the-loop).
    RequestInfo {
        request_id: String,
        source_executor_id: String,
        request_data: Value,
    },
    /// The run failed terminally.
    Failed { error: String },
}

impl WorkflowEvent {
    /// If this is an [`WorkflowEvent::Output`], return the data.
    pub fn as_output(&self) -> Option<&Value> {
        match self {
            WorkflowEvent::Output { data, .. } => Some(data),
            _ => None,
        }
    }
}
