//! Prebuilt orchestration patterns built on the workflow engine.
//!
//! Each pattern wraps [`SupportsAgentRun`]s as workflow [`Executor`] nodes that pass a
//! shared conversation (`Vec<Message>`, carried as JSON) between
//! participants:
//!
//! - [`SequentialBuilder`] / [`ConcurrentBuilder`] — pipeline and fan-out/fan-in.
//! - [`GroupChatBuilder`] — a manager coordinates a multi-agent conversation
//!   (round-robin, a custom [`GroupChatManager`], or an LLM manager agent).
//! - [`HandoffBuilder`] — agents transfer control via synthetic handoff tool
//!   calls, optionally requesting fresh user input between turns.
//! - [`MagenticBuilder`] — Magentic-One style planning + progress-ledger
//!   orchestration driven by a [`MagenticManager`].
//! - [`WorkflowAgent`] — expose a built [`Workflow`] as an [`SupportsAgentRun`].
//!
//! Rust equivalents of `agent_framework._workflows` (`_sequential`,
//! `_concurrent`, `_group_chat`, `_handoff`, `_magentic`, `_agent`).
//!
//! ## Design note (divergence from Python)
//!
//! The Python orchestrators build a multi-node graph in which the orchestrator
//! and every participant are separate executors that exchange envelope messages.
//! This Rust port keeps each orchestrator as a **single** [`Executor`] that
//! drives its participants by calling [`SupportsAgentRun::run`] directly (the same approach
//! the built-in [`AgentExecutor`] already uses), looping internally within one
//! superstep. Human-in-the-loop patterns (handoff interactive mode) still pause
//! and resume across supersteps via the engine's
//! [`request_info`](WorkflowContext::request_info) machinery. This preserves each
//! pattern's *semantics* while building entirely on the existing engine.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use super::context::WorkflowContext;
use super::events::WorkflowEvent;
use super::executor::Executor;
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::types::{AgentResponse, AgentResponseUpdate, Message};

mod approval;
mod concurrent;
mod group_chat;
mod handoff;
mod magentic;
mod sequential;
mod workflow_agent;

pub use approval::{AgentApprovalExecutor, ApprovalRequest};
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
    MagenticStallInterventionDecision, MagenticStallInterventionRequest, MagenticTaskLedger,
    StandardMagenticManager, MAGENTIC_MANAGER_NAME, ORCHESTRATOR_FINAL_ANSWER_PROMPT,
    ORCHESTRATOR_PROGRESS_LEDGER_PROMPT, ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT, ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT,
    ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT, ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT,
};
pub use sequential::SequentialBuilder;
pub use workflow_agent::{WorkflowAgent, WorkflowAgentExt};

/// Normalize a workflow input value into a conversation.
///
/// Accepts a bare string (→ one user message), an array of messages, or a
/// single message object. Shared by every orchestration executor.
pub(crate) fn parse_conversation(value: &Value) -> Result<Vec<Message>> {
    match value {
        Value::String(s) => Ok(vec![Message::user(s.clone())]),
        Value::Array(_) => serde_json::from_value(value.clone())
            .map_err(|e| Error::Workflow(format!("invalid conversation: {e}"))),
        Value::Object(_) => {
            let msg: Message = serde_json::from_value(value.clone())
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
pub(crate) fn ensure_author(mut message: Message, name: &str) -> Message {
    if message.author_name.is_none() {
        message.author_name = Some(name.to_string());
    }
    message
}

/// Run an agent over a conversation and surface its activity on the event
/// stream incrementally: an [`AgentRunUpdate`](WorkflowEvent::AgentRunUpdate)
/// per streamed [`AgentResponseUpdate`] as it arrives, then a final
/// aggregated [`AgentRun`](WorkflowEvent::AgentRun). Returns the response with
/// every message attributed to `author`.
///
/// Mirrors upstream `AgentExecutor`'s streaming (`_agent_executor.py:268-360`),
/// which drives the agent via `run_stream` and emits an `AgentRunUpdateEvent`
/// per update before the terminal `AgentRunEvent`. The updates are aggregated
/// back into an [`AgentResponse`] via [`AgentResponse::from_updates`].
pub(crate) async fn run_agent_and_emit(
    agent: &Arc<dyn SupportsAgentRun>,
    conversation: Vec<Message>,
    executor_id: &str,
    author: &str,
    ctx: &WorkflowContext,
) -> Result<AgentResponse> {
    let mut stream = agent.run_stream(conversation, None, None).await?;
    let mut updates: Vec<AgentResponseUpdate> = Vec::new();
    while let Some(item) = stream.next().await {
        let mut update = item?;
        if update.author_name.is_none() {
            update.author_name = Some(author.to_string());
        }
        if let Ok(update_value) = serde_json::to_value(&update) {
            ctx.add_event(WorkflowEvent::AgentRunUpdate {
                executor_id: executor_id.to_string(),
                update: update_value,
            });
        }
        updates.push(update);
    }

    let mut response = AgentResponse::from_updates(updates);
    for msg in &mut response.messages {
        if msg.author_name.is_none() {
            msg.author_name = Some(author.to_string());
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

/// An [`Executor`] that runs an [`SupportsAgentRun`] over the incoming conversation and
/// appends the agent's reply. Rust analogue of `AgentExecutor`.
pub struct AgentExecutor {
    id: String,
    agent: Arc<dyn SupportsAgentRun>,
    /// When true, yield the conversation as a workflow output (in addition to
    /// sending it downstream when [`Self::also_send`] is set).
    emit_output: bool,
    /// When true, forward the conversation downstream even though
    /// `emit_output` is also set. Lets a builder designate an interior
    /// participant (not just the terminal one) as an output/intermediate
    /// source — see [`SequentialBuilder::output_from`](super::SequentialBuilder::output_from)
    /// and [`ConcurrentBuilder::output_from`](super::ConcurrentBuilder::output_from).
    also_send: bool,
}

impl AgentExecutor {
    /// Wrap `agent` as an executor with the given `id`.
    pub fn new(id: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self {
            id: id.into(),
            agent,
            emit_output: false,
            also_send: false,
        }
    }

    /// When set, the executor yields the running conversation as a workflow
    /// output. Unless [`Self::with_also_send`] is also set, this replaces
    /// forwarding the conversation downstream (the pre-designation default:
    /// a pipeline's terminal executor only yields, it has nothing downstream
    /// to send to).
    pub fn with_output(mut self, emit_output: bool) -> Self {
        self.emit_output = emit_output;
        self
    }

    /// When set, the executor forwards the conversation downstream via
    /// [`WorkflowContext::send_message`] even if [`Self::with_output`] is
    /// also set (rather than the exclusive-or default). Needed whenever an
    /// interior (non-terminal) participant is designated as an output or
    /// intermediate-output source and must still hand the conversation to
    /// the next stage / the fan-in barrier.
    pub fn with_also_send(mut self, also_send: bool) -> Self {
        self.also_send = also_send;
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
            ctx.yield_output(payload.clone()).await?;
        }
        if !self.emit_output || self.also_send {
            ctx.send_message(payload).await?;
        }
        Ok(())
    }
}
