//! Concurrent orchestration: fan out an input to several agents and fan their
//! replies back in. Rust analogue of `_concurrent.py`.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::{parse_conversation, AgentExecutor};
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::types::Message;
use crate::workflow::{Executor, Workflow, WorkflowBuilder, WorkflowContext};

/// A dispatcher that broadcasts its input to all concurrent participants.
struct DispatchExecutor {
    id: String,
}

#[async_trait]
impl Executor for DispatchExecutor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let conversation = parse_conversation(&message)?;
        let payload = serde_json::to_value(&conversation)
            .map_err(|e| Error::Workflow(format!("serialize error: {e}")))?;
        ctx.send_message(payload).await?;
        Ok(())
    }
}

/// The default aggregator: collects each participant's final conversation and
/// yields the union of the initial prompt plus each agent's last reply.
struct AggregateExecutor {
    id: String,
}

#[async_trait]
impl Executor for AggregateExecutor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // message is an array of conversations (one per participant).
        let conversations = match &message {
            Value::Array(items) => items,
            _ => return Err(Error::Workflow("aggregator expected an array".into())),
        };
        let mut merged: Vec<Message> = Vec::new();
        let mut seeded = false;
        for conv_value in conversations {
            let conv = parse_conversation(conv_value)?;
            if !seeded {
                // Seed with everything except the last (the shared prompt).
                if conv.len() > 1 {
                    merged.extend(conv[..conv.len() - 1].iter().cloned());
                }
                seeded = true;
            }
            if let Some(last) = conv.last() {
                merged.push(last.clone());
            }
        }
        let payload = serde_json::to_value(&merged)
            .map_err(|e| Error::Workflow(format!("serialize error: {e}")))?;
        ctx.yield_output(payload).await?;
        Ok(())
    }
}

/// Builder for a concurrent fan-out/fan-in over agents. Rust analogue of
/// `ConcurrentBuilder`.
#[derive(Default)]
pub struct ConcurrentBuilder {
    participants: Vec<Arc<dyn SupportsAgentRun>>,
    name: Option<String>,
}

impl ConcurrentBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the participants that run concurrently.
    pub fn participants(
        mut self,
        agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>,
    ) -> Self {
        self.participants = agents.into_iter().collect();
        self
    }

    /// Add a participant.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, agent: Arc<dyn SupportsAgentRun>) -> Self {
        self.participants.push(agent);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validate and build the concurrent workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "concurrent workflow needs at least one participant".into(),
            ));
        }
        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(DispatchExecutor {
                id: "dispatch".into(),
            }))
            .add_executor(Arc::new(AggregateExecutor {
                id: "aggregate".into(),
            }))
            .set_start("dispatch");

        let mut agent_ids = Vec::new();
        for (i, agent) in self.participants.into_iter().enumerate() {
            let id = format!("agent_{i}");
            builder = builder.add_executor(Arc::new(AgentExecutor::new(id.clone(), agent)));
            agent_ids.push(id);
        }
        builder = builder.add_fan_out("dispatch", agent_ids.clone());
        builder = builder.add_fan_in(agent_ids, "aggregate");
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}
