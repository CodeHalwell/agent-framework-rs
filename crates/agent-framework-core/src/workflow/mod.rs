//! Graph-based multi-agent workflow orchestration.
//!
//! Rust equivalent of `agent_framework._workflows`. A [`Workflow`] is a directed
//! graph of [`Executor`] nodes connected by [`EdgeGroup`]s, executed in
//! supersteps (Pregel/BSP style): messages sent during a superstep are buffered
//! and delivered at its end. Execution runs until no messages remain or
//! `max_iterations` is hit.
//!
//! Beyond the core superstep engine this module provides:
//! - a [`WorkflowRun`] handle that can pause and resume, plus streaming via
//!   [`Workflow::run_stream`];
//! - human-in-the-loop requests through [`RequestInfoExecutor`] and the run
//!   handle's `send_responses`;
//! - run-scoped [`SharedState`] visible to every executor;
//! - [checkpointing](CheckpointStorage) at superstep boundaries with in-memory
//!   and file-backed storage;
//! - graph [validation](validate_workflow_graph) at build time;
//! - [visualization](WorkflowViz) as Mermaid and Graphviz DOT;
//! - sub-workflow composition via [`WorkflowExecutor`].

mod checkpoint;
mod context;
mod edge;
mod events;
mod executor;
mod orchestration;
mod request_info;
mod runner;
mod shared_state;
mod sub_workflow;
mod validation;
mod viz;

pub use checkpoint::{
    get_checkpoint_summary, CheckpointStorage, FileCheckpointStorage, InMemoryCheckpointStorage,
    WorkflowCheckpoint, WorkflowCheckpointSummary,
};
pub use context::{WorkflowContext, WorkflowMessage};
pub use edge::{Case, Condition, Default, EdgeGroup, Selection};
pub use events::{WorkflowEvent, WorkflowRunState};
pub use executor::{Executor, FunctionExecutor};
pub use orchestration::{AgentExecutor, ConcurrentBuilder, SequentialBuilder};
pub use request_info::{PendingRequest, RequestInfoExecutor, RequestResponse};
pub use runner::{Workflow, WorkflowBuilder, WorkflowRun, WorkflowRunStream};
pub use shared_state::SharedState;
pub use sub_workflow::WorkflowExecutor;
pub use validation::{validate_workflow_graph, ValidationType, WorkflowValidationError};
pub use viz::WorkflowViz;
