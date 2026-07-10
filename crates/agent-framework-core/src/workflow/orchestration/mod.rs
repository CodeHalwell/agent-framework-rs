//! Prebuilt orchestration patterns built on the workflow engine.
//!
//! Each pattern wraps [`Agent`]s as workflow [`Executor`] nodes that pass a
//! shared conversation (`Vec<ChatMessage>`, carried as JSON) between
//! participants:
//!
//! - [`SequentialBuilder`] / [`ConcurrentBuilder`] — pipeline and fan-out/fan-in.
//! - [`GroupChatBuilder`] — a manager coordinates a multi-agent conversation
//!   (round-robin, a custom [`GroupChatManager`], or an LLM manager agent).
//! - [`HandoffBuilder`] — agents transfer control via synthetic handoff tool
//!   calls, optionally requesting fresh user input between turns.
//! - [`MagenticBuilder`] — Magentic-One style planning + progress-ledger
//!   orchestration driven by a [`MagenticManager`].
//! - [`WorkflowAgent`] — expose a built [`Workflow`] as an [`Agent`].
//!
//! Rust equivalents of `agent_framework._workflows` (`_sequential`,
//! `_concurrent`, `_group_chat`, `_handoff`, `_magentic`, `_agent`).
//!
//! ## Design note (divergence from Python)
//!
//! The Python orchestrators build a multi-node graph in which the orchestrator
//! and every participant are separate executors that exchange envelope messages.
//! This Rust port keeps each orchestrator as a **single** [`Executor`] that
//! drives its participants by calling [`Agent::run`] directly (the same approach
//! the built-in [`AgentExecutor`] already uses), looping internally within one
//! superstep. Human-in-the-loop patterns (handoff interactive mode) still pause
//! and resume across supersteps via the engine's
//! [`request_info`](WorkflowContext::request_info) machinery. This preserves each
//! pattern's *semantics* while building entirely on the existing engine.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::context::WorkflowContext;
use super::events::WorkflowEvent;
use super::executor::Executor;
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::types::{AgentRunResponse, ChatMessage};

mod concurrent;
mod group_chat;
mod handoff;
mod magentic;
mod sequential;
mod workflow_agent;

pub use concurrent::ConcurrentBuilder;
pub use group_chat::{
    GroupChatBuilder, GroupChatDirective, GroupChatManager, GroupChatState,
    ManagerSelectionResponse, RoundRobinManager, DEFAULT_GROUP_CHAT_MAX_ITERATIONS,
    DEFAULT_MANAGER_INSTRUCTIONS, DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT,
};
pub use handoff::{
    handoff_tool_spec, HandoffBuilder, HandoffEdgeBuilder, HandoffInteractionMode,
    HandoffUserInputRequest,
};
pub use magentic::{
    MagenticBuilder, MagenticContext, MagenticManager, MagenticPlanReviewDecision,
    MagenticPlanReviewRequest, MagenticProgressLedger, MagenticProgressLedgerItem,
    MagenticTaskLedger, StandardMagenticManager, MAGENTIC_MANAGER_NAME,
    ORCHESTRATOR_FINAL_ANSWER_PROMPT, ORCHESTRATOR_PROGRESS_LEDGER_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT, ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT, ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
};
pub use sequential::SequentialBuilder;
pub use workflow_agent::{WorkflowAgent, WorkflowAgentExt};

/// Normalize a workflow input value into a conversation.
///
/// Accepts a bare string (→ one user message), an array of messages, or a
/// single message object. Shared by every orchestration executor.
pub(crate) fn parse_conversation(value: &Value) -> Result<Vec<ChatMessage>> {
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

/// Ensure a message carries an author name, defaulting to `name` when unset.
///
/// Mirrors Python's `ensure_author`: participants and orchestrators tag their
/// messages so the running transcript attributes each turn to its speaker.
pub(crate) fn ensure_author(mut message: ChatMessage, name: &str) -> ChatMessage {
    if message.author_name.is_none() {
        message.author_name = Some(name.to_string());
    }
    message
}

/// Run an agent over a conversation and surface its activity on the event
/// stream (an [`AgentRunUpdate`](WorkflowEvent::AgentRunUpdate) per message and a
/// final [`AgentRun`](WorkflowEvent::AgentRun)), returning the response with
/// every message attributed to `author`.
pub(crate) async fn run_agent_and_emit(
    agent: &Arc<dyn Agent>,
    conversation: Vec<ChatMessage>,
    executor_id: &str,
    author: &str,
    ctx: &WorkflowContext,
) -> Result<AgentRunResponse> {
    let mut response = agent.run(conversation, None).await?;
    for msg in &mut response.messages {
        if msg.author_name.is_none() {
            msg.author_name = Some(author.to_string());
        }
    }
    for msg in &response.messages {
        if let Ok(update) = serde_json::to_value(msg) {
            ctx.add_event(WorkflowEvent::AgentRunUpdate {
                executor_id: executor_id.to_string(),
                update,
            });
        }
    }
    if let Ok(resp_value) = serde_json::to_value(&response) {
        ctx.add_event(WorkflowEvent::AgentRun {
            executor_id: executor_id.to_string(),
            response: resp_value,
        });
    }
    Ok(response)
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
    /// Wrap `agent` as an executor with the given `id`.
    pub fn new(id: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        Self {
            id: id.into(),
            agent,
            emit_output: false,
        }
    }

    /// When set, the executor yields the running conversation as a workflow
    /// output instead of forwarding it downstream.
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
        let response =
            run_agent_and_emit(&self.agent, conversation.clone(), &self.id, &self.id, &ctx).await?;

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
