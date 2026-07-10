//! Sequential orchestration: a pipeline of agents that each see and extend the
//! running conversation. Rust analogue of `_sequential.py`.

use std::sync::Arc;

use super::AgentExecutor;
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::workflow::{Workflow, WorkflowBuilder};

/// Builder for a sequential pipeline of agents. Rust analogue of
/// `SequentialBuilder`. Each participant sees the running conversation and
/// appends its reply; the final conversation is yielded as output.
#[derive(Default)]
pub struct SequentialBuilder {
    participants: Vec<Arc<dyn Agent>>,
    name: Option<String>,
}

impl SequentialBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ordered list of participants.
    pub fn participants(mut self, agents: impl IntoIterator<Item = Arc<dyn Agent>>) -> Self {
        self.participants = agents.into_iter().collect();
        self
    }

    /// Append a participant to the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, agent: Arc<dyn Agent>) -> Self {
        self.participants.push(agent);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validate and build the sequential workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "sequential workflow needs at least one participant".into(),
            ));
        }
        let mut builder = WorkflowBuilder::new();
        let last = self.participants.len() - 1;
        let mut ids = Vec::new();
        for (i, agent) in self.participants.into_iter().enumerate() {
            let id = format!("seq_{i}");
            let exec = AgentExecutor::new(id.clone(), agent).with_output(i == last);
            builder = builder.add_executor(Arc::new(exec));
            ids.push(id);
        }
        builder = builder.set_start(ids[0].clone()).add_chain(ids);
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}
