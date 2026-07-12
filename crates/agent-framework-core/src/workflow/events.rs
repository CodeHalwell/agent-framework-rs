//! Workflow events emitted during execution.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The run state of a workflow, mirroring Python's `WorkflowRunState`.
///
/// The string forms match the Python enum values (`SCREAMING_SNAKE_CASE`) so
/// serialized status is interchangeable with the reference implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum WorkflowRunState {
    /// Run has been initiated; pre-work phase.
    Started,
    /// The workflow is actively executing.
    InProgress,
    /// Active execution while one or more info requests are outstanding.
    InProgressPendingRequests,
    /// Quiescent with no outstanding requests and no more work; terminal.
    Idle,
    /// Paused awaiting external input; non-terminal (resumable).
    IdleWithPendingRequests,
    /// Finished with an error; terminal.
    Failed,
    /// Finished due to cancellation; terminal.
    Cancelled,
}

/// An event observed while a workflow runs.
///
/// Corresponds to the `WorkflowEvent` hierarchy in the Python engine. Only
/// workflow-level events are observable; inter-executor messages are internal.
/// Variant names map onto Python types as noted (e.g. [`WorkflowEvent::Started`]
/// is `WorkflowStartedEvent`, [`WorkflowEvent::Output`] is `WorkflowOutputEvent`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkflowEvent {
    /// The run has begun (`WorkflowStartedEvent`).
    Started,
    /// A run-state transition (`WorkflowStatusEvent`).
    Status(WorkflowRunState),
    /// A superstep has started, with its 1-based iteration index.
    SuperStepStarted(usize),
    /// A superstep has completed, with its 1-based iteration index.
    SuperStepCompleted(usize),
    /// An executor began processing a message (`ExecutorInvokedEvent`).
    ExecutorInvoked { executor_id: String },
    /// An executor finished processing (`ExecutorCompletedEvent`).
    ExecutorCompleted { executor_id: String },
    /// An executor failed (`ExecutorFailedEvent`).
    ExecutorFailed { executor_id: String, error: String },
    /// An agent produced an incremental streaming update (`AgentRunUpdateEvent`).
    AgentRunUpdate { executor_id: String, update: Value },
    /// An agent completed a run (`AgentRunEvent`).
    AgentRun {
        executor_id: String,
        response: Value,
    },
    /// A workflow-level output was yielded (`WorkflowOutputEvent`).
    ///
    /// Terminal: recorded as (part of) the run's final output. Emitted for
    /// yields from an executor named by
    /// [`WorkflowBuilder::output_from`](super::WorkflowBuilder::output_from),
    /// or from any executor when no output designation is configured at all
    /// (see [`WorkflowEvent::Intermediate`] for the non-terminal counterpart).
    Output {
        data: Value,
        source_executor_id: String,
    },
    /// A non-terminal, workflow-level progress signal.
    ///
    /// Mirrors upstream's `intermediate` yield classification: emitted for
    /// yields from an executor named by
    /// [`WorkflowBuilder::intermediate_output_from`](super::WorkflowBuilder::intermediate_output_from),
    /// and (when an `output_from` allowlist is configured) for yields from any
    /// executor *not* on that allowlist, as a safe non-terminal fallback. Never
    /// recorded as the run's final output ([`WorkflowRun::last_output`](super::WorkflowRun::last_output)
    /// ignores it).
    Intermediate {
        data: Value,
        source_executor_id: String,
    },
    /// A custom event added by an executor.
    Custom(Value),
    /// A request for external input, human-in-the-loop (`RequestInfoEvent`).
    RequestInfo {
        request_id: String,
        /// The executor that surfaced the request (the emitter).
        source_executor_id: String,
        request_data: Value,
    },
    /// The run failed terminally (`WorkflowFailedEvent`).
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

    /// If this is a [`WorkflowEvent::Intermediate`], return the data.
    pub fn as_intermediate(&self) -> Option<&Value> {
        match self {
            WorkflowEvent::Intermediate { data, .. } => Some(data),
            _ => None,
        }
    }

    /// If this is a [`WorkflowEvent::RequestInfo`], return `(request_id, data)`.
    pub fn as_request_info(&self) -> Option<(&str, &Value)> {
        match self {
            WorkflowEvent::RequestInfo {
                request_id,
                request_data,
                ..
            } => Some((request_id, request_data)),
            _ => None,
        }
    }

    /// If this is a [`WorkflowEvent::Status`], return the state.
    pub fn as_status(&self) -> Option<WorkflowRunState> {
        match self {
            WorkflowEvent::Status(s) => Some(*s),
            _ => None,
        }
    }
}
