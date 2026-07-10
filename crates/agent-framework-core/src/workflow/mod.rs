//! Graph-based multi-agent workflow orchestration.
//!
//! Rust equivalent of `agent_framework._workflows`. A [`Workflow`] is a directed
//! graph of [`Executor`] nodes connected by [`EdgeGroup`]s, executed in
//! supersteps (Pregel/BSP style): messages sent during a superstep are buffered
//! and delivered at its end. Execution runs until no messages remain or
//! `max_iterations` is hit.

mod context;
mod edge;
mod events;
mod executor;
mod orchestration;
mod runner;

pub use context::{WorkflowContext, WorkflowMessage};
pub use edge::{Case, Condition, Default, EdgeGroup, Selection};
pub use events::{WorkflowEvent, WorkflowRunState};
pub use executor::{Executor, FunctionExecutor};
pub use orchestration::{ConcurrentBuilder, SequentialBuilder};
pub use runner::{Workflow, WorkflowBuilder, WorkflowRunResult};
