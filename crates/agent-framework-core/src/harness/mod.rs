//! The experimental agent harness: an agent loop, a todo list, standing tool
//! approvals and operating modes.
//!
//! Rust port of four pieces of upstream's `agent_framework._harness` package
//! (Python `ExperimentalFeature.HARNESS`, .NET `Microsoft.Agents.AI.Harness`
//! `[Experimental]`). Everything here sits behind the `experimental-harness`
//! cargo feature and carries no semver promise; see `docs/feature-stages.md`.
//!
//! | Upstream (Python) | Here |
//! |---|---|
//! | `_loop.py`: `AgentLoopMiddleware`, `JudgeVerdict`, `todos_remaining`, `todos_remaining_message` | [`LoopAgent`], [`Judge`], [`JudgeVerdict`], [`todos_remaining`], [`todos_remaining_message`] |
//! | `_todo.py`: `TodoProvider`, `TodoStore`, `TodoSessionStore`, `TodoItem`, … | [`TodoProvider`], [`TodoStore`], [`TodoSessionStore`], [`TodoItem`], … |
//! | `_tool_approval.py`: `ToolApprovalMiddleware`, `ToolApprovalRule`, `ToolApprovalState`, `create_always_approve_*` | [`ToolApprovalAgent`], [`ToolApprovalRule`], [`ToolApprovalState`], [`ToolApprovalAgent::always_approve_tool_response`], … |
//! | `_mode.py`: `AgentModeProvider`, `get_agent_mode`, `set_agent_mode` | [`AgentModeProvider`], [`get_agent_mode`], [`set_agent_mode`] |
//!
//! # Not ported here
//!
//! - **Background agents** (`_background_agents.py`) have no Rust
//!   counterpart, so the loop's `background_tasks_running` /
//!   `background_tasks_running_message` helpers are not ported either.
//! - **File access, file memory and memory** (`_file_access.py`,
//!   `_file_memory.py`, `_memory.py`) and the full `create_harness_agent`
//!   assembly are out of scope for now.
//! - `TodoFileStore` is not ported: [`TodoStore`] is a public trait, so a
//!   file-backed store can be supplied by the application.
//!
//! # Divergences from upstream
//!
//! **The loop and tool approval are agent wrappers, not middleware.**
//! Upstream Python implements both as `AgentMiddleware` that calls
//! `call_next()` repeatedly. In this port an agent middleware's
//! [`Next`](crate::middleware::Next) continuation is single-shot, and agent
//! middleware runs *inside* the context-provider and history pass, so it
//! cannot re-run a whole turn. Both pieces therefore take the shape .NET
//! gives them: a [`SupportsAgentRun`](crate::agent::SupportsAgentRun) that
//! wraps another agent ([`LoopAgent`], which is .NET's `LoopAgent` name for
//! Python's `AgentLoopMiddleware`, and [`ToolApprovalAgent`]). Every
//! iteration is a full run of the wrapped agent, so its context providers
//! and history provider run on each pass. Python's turn-scoped
//! `after_run_once_per_turn` deferral has no counterpart.
//!
//! **Providers are passed explicitly.** Python's loop helpers find the
//! `TodoProvider` / `AgentModeProvider` on `agent.context_providers`. Rust
//! agents do not expose their providers, so [`todos_remaining`] and
//! [`todos_remaining_message`] take the provider handle (an `Arc`) the
//! application also registered on the agent.
//!
//! **Context providers see the session.** Upstream's `before_run` receives
//! the `AgentSession`; this port adds the same thing as
//! [`SessionContext::session`](crate::memory::SessionContext::session), which
//! the todo and mode providers use to read their state.
//!
//! See each submodule's documentation for the divergences specific to it.

pub mod agent_loop;
pub mod mode;
pub mod todo;
pub mod tool_approval;

pub use agent_loop::{
    async_loop_callback, judge_feedback_message, todos_remaining, todos_remaining_message,
    AsyncLoopCallback, Judge, JudgeVerdict, LoopAgent, LoopAgentBuilder, LoopCallback, LoopContext,
    LoopDecision, TodosRemaining, TodosRemainingMessage, VerdictParser, CRITERIA_PLACEHOLDER,
    DEFAULT_JUDGE_INSTRUCTIONS, DEFAULT_JUDGE_MAX_ITERATIONS, DEFAULT_MAX_ITERATIONS,
    DEFAULT_NEXT_MESSAGE, JUDGE_VERDICT_DONE, JUDGE_VERDICT_MORE,
};
pub use mode::{
    get_agent_mode, set_agent_mode, AgentModeProvider, AgentModeProviderBuilder,
    DEFAULT_MODE_CHANGE_NOTIFICATION, DEFAULT_MODE_INSTRUCTIONS, DEFAULT_MODE_SOURCE_ID,
    EXECUTE_MODE_INSTRUCTIONS, PLAN_MODE_INSTRUCTIONS,
};
pub use todo::{
    TodoCompleteInput, TodoInput, TodoItem, TodoProvider, TodoSessionStore, TodoStore,
    DEFAULT_TODO_INSTRUCTIONS, DEFAULT_TODO_SOURCE_ID,
};
pub use tool_approval::{
    ToolApprovalAgent, ToolApprovalRule, ToolApprovalScope, ToolApprovalState,
    ToolAutoApprovalRule, DEFAULT_TOOL_APPROVAL_SOURCE_ID,
};

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use serde_json::{Map, Value};
use tokio::sync::mpsc;

use crate::agent::AgentRunStream;
use crate::error::{Error, Result};
use crate::session::AgentSession;
use crate::types::AgentResponseUpdate;

/// Read the JSON object a harness component keeps under `source_id` in the
/// session's state bag. An absent key reads as an empty object; any other
/// non-object value is corrupt state and errors.
pub(crate) fn read_state_object(
    session: &AgentSession,
    source_id: &str,
) -> Result<Map<String, Value>> {
    match session.state.get(source_id) {
        None | Some(Value::Null) => Ok(Map::new()),
        Some(Value::Object(map)) => Ok(map),
        Some(other) => Err(Error::Serialization(format!(
            "session state for source_id {source_id:?} must be a JSON object; got {}",
            json_type_name(&other)
        ))),
    }
}

/// The JSON type name of `value`, for error messages.
pub(crate) fn json_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// The sending half handed to a [`channel_stream`] producer.
pub(crate) struct UpdateSink(mpsc::Sender<Result<AgentResponseUpdate>>);

impl UpdateSink {
    /// Send one update. Returns `false` once the consumer has dropped the
    /// stream, so the producer can stop early.
    pub(crate) async fn send(&self, update: AgentResponseUpdate) -> bool {
        self.0.send(Ok(update)).await.is_ok()
    }
}

/// Build an [`AgentRunStream`] from a producer future that pushes updates
/// into an [`UpdateSink`]. The producer runs on a spawned task; an `Err` it
/// returns becomes the stream's final item, and dropping the stream aborts
/// the task.
pub(crate) fn channel_stream<Fut>(make: impl FnOnce(UpdateSink) -> Fut) -> AgentRunStream
where
    Fut: Future<Output = Result<()>> + Send + 'static,
{
    let (tx, rx) = mpsc::channel(16);
    let error_tx = tx.clone();
    let producer = make(UpdateSink(tx));
    let handle = tokio::spawn(async move {
        if let Err(e) = producer.await {
            let _ = error_tx.send(Err(e)).await;
        }
    });
    Box::pin(AbortOnDrop {
        rx,
        handle: Some(handle),
    })
}

struct AbortOnDrop {
    rx: mpsc::Receiver<Result<AgentResponseUpdate>>,
    handle: Option<tokio::task::JoinHandle<()>>,
}

impl Stream for AbortOnDrop {
    type Item = Result<AgentResponseUpdate>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            handle.abort();
        }
    }
}
