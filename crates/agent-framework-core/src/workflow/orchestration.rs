//! Prebuilt orchestration patterns: sequential and concurrent.
//!
//! These wrap [`Agent`]s as workflow [`Executor`]s that pass a shared
//! conversation (`Vec<ChatMessage>`, carried as JSON) between participants.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;

use super::context::WorkflowContext;
use super::executor::Executor;
use super::runner::{Workflow, WorkflowBuilder};
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::types::ChatMessage;

fn parse_conversation(value: &Value) -> Result<Vec<ChatMessage>> {
    match value {
        Value::String(s) => Ok(vec![ChatMessage::user(s.clone())]),
        Value::Array(_) => serde_json::from_value(value.clone())
            .map_err(|e| Error::Workflow(format!("invalid conversation: {e}"))),
        Value::Object(_) => {
            let msg: ChatMessage = serde_json::from_value(value.clone())
                .map_err(|e| Error::Workflow(format!("invalid message: {e}")))?;
            Ok(vec![msg])
        }
        _ => Err(Error::Workflow("unsupported workflow input".into())),
    }
}

/// An [`Executor`] that runs an [`Agent`] over the incoming conversation and
/// appends the agent's reply. Rust analogue of `AgentExecutor`.
pub struct AgentExecutor {
    id: String,
    agent: Arc<dyn Agent>,
    /// When true, yield the conversation as workflow output instead of sending.
    emit_output: bool,
}

impl AgentExecutor {
    pub fn new(id: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        Self {
            id: id.into(),
            agent,
            emit_output: false,
        }
    }

    pub fn with_output(mut self, emit_output: bool) -> Self {
        self.emit_output = emit_output;
        self
    }
}

#[async_trait]
impl Executor for AgentExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let mut conversation = parse_conversation(&message)?;
        let response = self.agent.run(conversation.clone(), None).await?;
        conversation.extend(response.messages);
        let payload = serde_json::to_value(&conversation)
            .map_err(|e| Error::Workflow(format!("failed to serialize conversation: {e}")))?;
        if self.emit_output {
            ctx.yield_output(payload).await?;
        } else {
            ctx.send_message(payload).await?;
        }
        Ok(())
    }
}

/// Builder for a sequential pipeline of agents. Rust analogue of
/// `SequentialBuilder`. Each participant sees the running conversation and
/// appends its reply; the final conversation is yielded as output.
#[derive(Default)]
pub struct SequentialBuilder {
    participants: Vec<Arc<dyn Agent>>,
    name: Option<String>,
}

impl SequentialBuilder {
    pub fn new() -> Self {
        Self::default()
    }

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

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

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
        let mut merged: Vec<ChatMessage> = Vec::new();
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
    participants: Vec<Arc<dyn Agent>>,
    name: Option<String>,
}

impl ConcurrentBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn participants(mut self, agents: impl IntoIterator<Item = Arc<dyn Agent>>) -> Self {
        self.participants = agents.into_iter().collect();
        self
    }

    /// Add a participant.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, agent: Arc<dyn Agent>) -> Self {
        self.participants.push(agent);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

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
