//! Background agents: a [`ContextProvider`] that lets an agent delegate work
//! to sub-agents that run concurrently, then wait for and collect results.
//!
//! Port of upstream `_harness/_background_agents.py`
//! (`BackgroundAgentsProvider`). The provider adds its instructions (with a
//! list of the available agents), a `user` message listing the current tasks
//! when there are any, and six tools:
//!
//! | Tool | Does |
//! |---|---|
//! | `background_agents_start_task` | Start a task on a named agent, in a fresh session of that agent. |
//! | `background_agents_wait_for_first_completion` | Wait until the first of some tasks finishes, or the wait timeout passes. |
//! | `background_agents_get_task_results` | A finished task's text output, or its error. |
//! | `background_agents_get_all_tasks` | Every task with its status. |
//! | `background_agents_continue_task` | Send follow-up input to a finished task's session. |
//! | `background_agents_clear_completed_task` | Forget a finished task and its session. |
//!
//! Task metadata persists in the parent session's state bag under the
//! provider's `source_id` as `{"next_task_id": n, "tasks": [...]}`. The
//! running tasks and the sub-agents' sessions are process state held by the
//! provider: a task whose runtime is gone (for example after a restart, or
//! a session restored elsewhere) is reported as [`BackgroundTaskStatus::Lost`].
//! Call [`BackgroundAgentsProvider::release_session`] when a session ends so
//! its tasks and sub-sessions are dropped.
//!
//! [`background_tasks_running`] and [`background_tasks_running_message`] are
//! [`LoopAgent`](super::LoopAgent) callbacks that keep a loop going while
//! tasks run.
//!
//! # Security
//!
//! The parent sends the sub-agents text (which may carry content from its
//! own untrusted context) and reads back whatever they produce. Only supply
//! agents you trust with that data, since a compromised one can exfiltrate
//! its input or answer with a prompt injection.
//!
//! # Divergences
//!
//! - Tasks run on the Tokio runtime ([`tokio::spawn`]), so the agents must be
//!   `'static` ([`Arc<dyn SupportsAgentRun>`]).
//! - The parent run must have a session ([`SessionContext::session`]);
//!   `before_run` fails without one.
//! - Upstream's loop helpers find the provider on the agent; here they take
//!   the provider, as [`todos_remaining`](super::todos_remaining) does.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::Message;

use super::agent_loop::{LoopCallback, LoopContext, LoopDecision};
use super::read_state_object;

/// The default `source_id` of a [`BackgroundAgentsProvider`].
pub const DEFAULT_BACKGROUND_AGENTS_SOURCE_ID: &str = "background_agents";

/// How long `background_agents_wait_for_first_completion` blocks by default.
pub const DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT: Duration = Duration::from_secs(300);

/// The default instructions. `{background_agents}` is replaced with the list
/// of available agents.
pub const DEFAULT_BACKGROUND_AGENTS_INSTRUCTIONS: &str = "## Background Agents

You have access to background agents that can perform work on your behalf.

- Use the `background_agents_*` tools to start tasks on background agents and check their results.
- Creating a background task does not block, and background tasks run concurrently.
- Important: Always wait for outstanding tasks to finish before you finish processing.
- Important: After retrieving results from a completed task, clear it with \
background_agents_clear_completed_task to free memory, unless you plan to continue it with \
background_agents_continue_task.

{background_agents}";

/// The status of a background task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackgroundTaskStatus {
    /// Still running.
    Running,
    /// Finished with a response.
    Completed,
    /// Finished with an error, or was cancelled.
    Failed,
    /// Its runtime is gone, so its outcome cannot be known.
    Lost,
}

impl BackgroundTaskStatus {
    /// The wire name: `running`, `completed`, `failed` or `lost`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Lost => "lost",
        }
    }
}

/// The persisted metadata of one background task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackgroundTaskInfo {
    /// The task id, unique within the parent session.
    pub id: u64,
    /// The name of the agent running it, as the model gave it.
    pub agent_name: String,
    /// The model's description of the task.
    pub description: String,
    /// The task's status.
    pub status: BackgroundTaskStatus,
    /// The response text of a completed task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result_text: Option<String>,
    /// The error of a failed task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_text: Option<String>,
}

/// How a task's run ended.
#[derive(Debug, Clone)]
enum Outcome {
    Completed(String),
    Failed(String),
}

/// One execution of a task (a start or a continuation).
struct InFlight {
    done: watch::Receiver<bool>,
    outcome: Arc<Mutex<Option<Outcome>>>,
    abort: tokio::task::AbortHandle,
}

impl InFlight {
    fn is_done(&self) -> bool {
        *self.done.borrow()
    }

    fn outcome(&self) -> Option<Outcome> {
        lock(&self.outcome).clone()
    }

    async fn wait(&self) {
        let mut done = self.done.clone();
        // An Err means the task was dropped (aborted) without signalling.
        let _ = done.wait_for(|d| *d).await;
    }
}

#[derive(Default)]
struct RuntimeInner {
    in_flight: HashMap<u64, Arc<InFlight>>,
    sessions: HashMap<u64, AgentSession>,
    closed: bool,
}

/// The process state of one parent session's tasks.
#[derive(Default)]
struct Runtime {
    inner: Mutex<RuntimeInner>,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A [`ContextProvider`] that lets an agent run sub-agents in the
/// background. See the [module docs](self).
#[derive(Clone)]
pub struct BackgroundAgentsProvider {
    shared: Arc<Shared>,
}

struct Shared {
    /// Keyed by lowercased name, in the order given.
    agents: Vec<(String, Arc<dyn SupportsAgentRun>)>,
    source_id: String,
    instructions: String,
    wait_timeout: Duration,
    runtimes: Mutex<HashMap<String, Arc<Runtime>>>,
}

impl std::fmt::Debug for BackgroundAgentsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackgroundAgentsProvider")
            .field("source_id", &self.shared.source_id)
            .field(
                "agents",
                &self
                    .shared
                    .agents
                    .iter()
                    .map(|(k, _)| k)
                    .collect::<Vec<_>>(),
            )
            .field("wait_timeout", &self.shared.wait_timeout)
            .finish_non_exhaustive()
    }
}

/// Builder for a [`BackgroundAgentsProvider`].
pub struct BackgroundAgentsProviderBuilder {
    agents: Vec<Arc<dyn SupportsAgentRun>>,
    source_id: String,
    instructions: Option<String>,
    wait_timeout: Duration,
}

impl BackgroundAgentsProviderBuilder {
    /// Set the `source_id` the task state is stored under.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Replace the default instructions. A `{background_agents}` placeholder
    /// is replaced with the list of agents.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// How long the wait tool blocks before reporting that nothing finished
    /// (default five minutes).
    pub fn wait_timeout(mut self, timeout: Duration) -> Self {
        self.wait_timeout = timeout;
        self
    }

    /// Build the provider. Fails when there are no agents, an agent has no
    /// name, two names are equal ignoring case, or the wait timeout is zero.
    pub fn build(self) -> Result<BackgroundAgentsProvider> {
        if self.agents.is_empty() {
            return Err(Error::Configuration(
                "At least one background agent must be provided.".into(),
            ));
        }
        if self.wait_timeout.is_zero() {
            return Err(Error::Configuration(
                "wait_timeout must be a positive duration.".into(),
            ));
        }
        let mut agents: Vec<(String, Arc<dyn SupportsAgentRun>)> = Vec::new();
        for agent in self.agents {
            let name = agent.name().map(str::trim).unwrap_or_default();
            if name.is_empty() {
                return Err(Error::Configuration(
                    "All background agents must have a non-empty name.".into(),
                ));
            }
            let key = agent.name().unwrap_or_default().to_lowercase();
            if agents.iter().any(|(k, _)| *k == key) {
                return Err(Error::Configuration(format!(
                    "Duplicate background agent name: '{}'. Agent names must be unique (case-insensitive).",
                    agent.name().unwrap_or_default()
                )));
            }
            agents.push((key, agent));
        }
        let mut list = vec!["Available background agents:".to_string()];
        for (_, agent) in &agents {
            let mut line = format!("- {}", agent.name().unwrap_or_default());
            if let Some(description) = agent.description().filter(|d| !d.is_empty()) {
                line.push_str(&format!(": {description}"));
            }
            list.push(line);
        }
        let instructions = self
            .instructions
            .as_deref()
            .unwrap_or(DEFAULT_BACKGROUND_AGENTS_INSTRUCTIONS)
            .replace("{background_agents}", &list.join("\n"));
        Ok(BackgroundAgentsProvider {
            shared: Arc::new(Shared {
                agents,
                source_id: self.source_id,
                instructions,
                wait_timeout: self.wait_timeout,
                runtimes: Mutex::new(HashMap::new()),
            }),
        })
    }
}

impl BackgroundAgentsProvider {
    /// A builder over `agents`, each of which needs a unique name.
    pub fn builder(
        agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>,
    ) -> BackgroundAgentsProviderBuilder {
        BackgroundAgentsProviderBuilder {
            agents: agents.into_iter().collect(),
            source_id: DEFAULT_BACKGROUND_AGENTS_SOURCE_ID.into(),
            instructions: None,
            wait_timeout: DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT,
        }
    }

    /// A provider over `agents` with the default settings.
    pub fn new(agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>) -> Result<Self> {
        Self::builder(agents).build()
    }

    /// The provider's `source_id`.
    pub fn source_id(&self) -> &str {
        &self.shared.source_id
    }

    /// The instructions, with the agent list filled in.
    pub fn instructions(&self) -> &str {
        &self.shared.instructions
    }

    /// The tasks persisted for `session`, as last refreshed. Does not look at
    /// the running tasks; [`tasks`](Self::tasks) does.
    pub fn persisted_tasks(&self, session: &AgentSession) -> Result<Vec<BackgroundTaskInfo>> {
        Ok(load_state(session, &self.shared.source_id)?.1)
    }

    /// The tasks of `session`, with finished ones' outcomes recorded first.
    pub fn tasks(&self, session: &AgentSession) -> Result<Vec<BackgroundTaskInfo>> {
        let runtime = self.runtime(session);
        refresh(session, &self.shared.source_id, &runtime)
    }

    /// The runtime of `session`, created if absent or closed.
    fn runtime(&self, session: &AgentSession) -> Arc<Runtime> {
        let mut runtimes = lock(&self.shared.runtimes);
        let entry = runtimes
            .entry(session.session_id().to_string())
            .or_default();
        if lock(&entry.inner).closed {
            *entry = Arc::default();
        }
        entry.clone()
    }

    /// Drop the running tasks and sub-sessions of `session`.
    ///
    /// With `cancel_running`, running tasks are aborted and awaited for up to
    /// `timeout` (`None` waits as long as it takes); without it, the call
    /// fails while any task is running. Tools of a released session refuse
    /// to start, continue, wait or clear.
    pub async fn release_session(
        &self,
        session: &AgentSession,
        cancel_running: bool,
        timeout: Option<Duration>,
    ) -> Result<()> {
        let session_id = session.session_id().to_string();
        let Some(runtime) = lock(&self.shared.runtimes).get(&session_id).cloned() else {
            return Ok(());
        };
        let pending: Vec<Arc<InFlight>> = {
            let mut inner = lock(&runtime.inner);
            if inner.closed {
                return Ok(());
            }
            let pending: Vec<_> = inner
                .in_flight
                .values()
                .filter(|t| !t.is_done())
                .cloned()
                .collect();
            if !pending.is_empty() && !cancel_running {
                return Err(Error::Other(format!(
                    "Cannot release session {session_id}: {} tasks still running.",
                    pending.len()
                )));
            }
            inner.closed = true;
            pending
        };
        for task in &pending {
            task.abort.abort();
        }
        let drained = futures::future::join_all(pending.iter().map(|t| t.wait()));
        match timeout {
            Some(limit) => {
                if tokio::time::timeout(limit, drained).await.is_err() {
                    tracing::warn!(
                        tasks = pending.len(),
                        "background session release timed out; abandoning the remaining tasks"
                    );
                }
            }
            None => {
                drained.await;
            }
        }
        {
            let mut inner = lock(&runtime.inner);
            inner.in_flight.clear();
            inner.sessions.clear();
        }
        let mut runtimes = lock(&self.shared.runtimes);
        if runtimes
            .get(&session_id)
            .is_some_and(|r| Arc::ptr_eq(r, &runtime))
        {
            runtimes.remove(&session_id);
        }
        Ok(())
    }

    fn agent(&self, name: &str) -> Option<&Arc<dyn SupportsAgentRun>> {
        let key = name.to_lowercase();
        self.shared
            .agents
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, a)| a)
    }

    fn available(&self) -> String {
        self.shared
            .agents
            .iter()
            .map(|(_, a)| a.name().unwrap_or_default().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The six tools, bound to `session`.
    pub fn tools(&self, session: &AgentSession) -> Vec<ToolDefinition> {
        let bound = Bound {
            provider: self.clone(),
            session: session.clone(),
            runtime: self.runtime(session),
        };
        vec![
            bound.start_tool(),
            bound.wait_tool(),
            bound.results_tool(),
            bound.all_tasks_tool(),
            bound.continue_tool(),
            bound.clear_tool(),
        ]
    }
}

/// A provider bound to one parent session for one run.
#[derive(Clone)]
struct Bound {
    provider: BackgroundAgentsProvider,
    session: AgentSession,
    runtime: Arc<Runtime>,
}

impl Bound {
    fn source_id(&self) -> &str {
        &self.provider.shared.source_id
    }

    /// Start `input` on `agent` in `sub_session` as task `id`, recording it
    /// as the task's current execution.
    fn spawn(
        &self,
        inner: &mut RuntimeInner,
        id: u64,
        agent: Arc<dyn SupportsAgentRun>,
        input: String,
        mut sub_session: AgentSession,
    ) {
        let (tx, rx) = watch::channel(false);
        let outcome = Arc::new(Mutex::new(None));
        let (slot, runtime) = (outcome.clone(), Arc::downgrade(&self.runtime));
        let handle = tokio::spawn(async move {
            let result = agent
                .run(vec![Message::user(input)], Some(&mut sub_session))
                .await;
            *lock(&slot) = Some(match result {
                Ok(response) => Outcome::Completed(response.text()),
                Err(e) => Outcome::Failed(e.to_string()),
            });
            // Hand the session back for continuations.
            if let Some(runtime) = runtime.upgrade() {
                let mut inner = lock(&runtime.inner);
                if !inner.closed {
                    inner.sessions.insert(id, sub_session);
                }
            }
            let _ = tx.send(true);
        });
        let in_flight = Arc::new(InFlight {
            done: rx,
            outcome,
            abort: handle.abort_handle(),
        });
        inner.in_flight.insert(id, in_flight);
    }

    fn start_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name of the background agent to run the task.
            agent_name: String,
            /// The text input for the agent.
            input: String,
            /// A short description of the task.
            description: String,
        }
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_start_task",
            "Start a background task on a named agent. Returns a confirmation with the task ID.",
            move |args: Args| {
                let bound = bound.clone();
                async move {
                    Ok(Value::String(bound.start(
                        args.agent_name,
                        args.input,
                        args.description,
                    )?))
                }
            },
        )
        .into_definition()
    }

    fn start(&self, agent_name: String, input: String, description: String) -> Result<String> {
        let Some(agent) = self.provider.agent(&agent_name).cloned() else {
            return Ok(format!(
                "Error: No background agent found with name '{agent_name}'. Available agents: {}",
                self.provider.available()
            ));
        };
        let mut inner = lock(&self.runtime.inner);
        if inner.closed {
            return Ok(
                "Error: Session is being released; cannot start a new background task.".into(),
            );
        }
        let (next_id, mut tasks) = load_state(&self.session, self.source_id())?;
        let id = next_id;
        let sub_session = agent.create_session();
        self.spawn(&mut inner, id, agent, input, sub_session);
        tasks.push(BackgroundTaskInfo {
            id,
            agent_name: agent_name.clone(),
            description,
            status: BackgroundTaskStatus::Running,
            result_text: None,
            error_text: None,
        });
        save_state(&self.session, self.source_id(), id + 1, &tasks)?;
        Ok(format!(
            "Background task {id} started on agent '{agent_name}'."
        ))
    }

    fn wait_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// IDs of the tasks to wait for.
            task_ids: Vec<u64>,
        }
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_wait_for_first_completion",
            "Wait until the first of the specified tasks completes or the configured timeout expires. Returns \
             the completed task's ID. On timeout, the tasks remain running and this tool can be called again to \
             continue waiting.",
            move |args: Args| {
                let bound = bound.clone();
                async move { Ok(Value::String(bound.wait(args.task_ids).await?)) }
            },
        )
        .into_definition()
    }

    async fn wait(&self, task_ids: Vec<u64>) -> Result<String> {
        let waitable: Vec<(u64, Arc<InFlight>)> = {
            let inner = lock(&self.runtime.inner);
            if inner.closed {
                return Ok(
                    "Error: Session is being released; cannot wait for background tasks.".into(),
                );
            }
            if task_ids.is_empty() {
                return Ok("Error: No task IDs provided.".into());
            }
            task_ids
                .iter()
                .filter_map(|id| inner.in_flight.get(id).map(|t| (*id, t.clone())))
                .collect()
        };
        if waitable.is_empty() {
            let tasks = refresh(&self.session, self.source_id(), &self.runtime)?;
            if let Some(done) = tasks
                .iter()
                .find(|t| task_ids.contains(&t.id) && t.status != BackgroundTaskStatus::Running)
            {
                return Ok(format!(
                    "Task {} is not running; current status: {}.",
                    done.id,
                    done.status.as_str()
                ));
            }
            return Ok("Error: None of the specified task IDs correspond to running tasks.".into());
        }
        let first = futures::future::select_all(
            waitable
                .iter()
                .map(|(_, t)| Box::pin(t.wait()) as futures::future::BoxFuture<'_, ()>),
        );
        let timeout = self.provider.shared.wait_timeout;
        let index = match tokio::time::timeout(timeout, first).await {
            Ok((_, index, _)) => index,
            Err(_) => {
                return Ok(format!(
                    "No background task completed within {} seconds. The tasks are still running; call this \
                     tool again if you wish to continue waiting.",
                    timeout.as_secs()
                ))
            }
        };
        let (id, finished) = &waitable[index];
        let status = {
            let mut inner = lock(&self.runtime.inner);
            let (next_id, mut tasks) = load_state(&self.session, self.source_id())?;
            let current = inner
                .in_flight
                .get(id)
                .is_some_and(|t| Arc::ptr_eq(t, finished));
            match tasks.iter_mut().find(|t| t.id == *id) {
                Some(info) => {
                    finalize(info, finished);
                    let status = info.status.as_str().to_string();
                    // Report the waited-on execution without overwriting a
                    // newer continuation of the same task.
                    if current {
                        inner.in_flight.remove(id);
                        save_state(&self.session, self.source_id(), next_id, &tasks)?;
                    }
                    status
                }
                None => "Unknown".to_string(),
            }
        };
        Ok(format!("Task {id} finished with status: {status}."))
    }

    fn results_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// ID of the task.
            task_id: u64,
        }
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_get_task_results",
            "Get the text output of a background task by its ID.",
            move |args: Args| {
                let bound = bound.clone();
                async move {
                    let tasks = refresh(&bound.session, bound.source_id(), &bound.runtime)?;
                    let id = args.task_id;
                    let Some(info) = tasks.iter().find(|t| t.id == id) else {
                        return Ok(Value::String(format!("Error: No task found with ID {id}.")));
                    };
                    Ok(Value::String(match info.status {
                        BackgroundTaskStatus::Completed => info
                            .result_text
                            .clone()
                            .filter(|t| !t.is_empty())
                            .unwrap_or_else(|| "(no output)".into()),
                        BackgroundTaskStatus::Failed => format!(
                            "Task failed: {}",
                            info.error_text
                                .as_deref()
                                .filter(|e| !e.is_empty())
                                .unwrap_or("Unknown error")
                        ),
                        BackgroundTaskStatus::Lost => {
                            "Task state was lost (reference unavailable).".into()
                        }
                        BackgroundTaskStatus::Running => format!("Task {id} is still running."),
                    }))
                }
            },
        )
        .into_definition()
    }

    fn all_tasks_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {}
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_get_all_tasks",
            "List all background tasks with their IDs, statuses, agent names, and descriptions.",
            move |_: Args| {
                let bound = bound.clone();
                async move {
                    let tasks = refresh(&bound.session, bound.source_id(), &bound.runtime)?;
                    if tasks.is_empty() {
                        return Ok(Value::String("No tasks.".into()));
                    }
                    let mut lines = vec!["Tasks:".to_string()];
                    lines.extend(tasks.iter().map(task_line));
                    Ok(Value::String(lines.join("\n")))
                }
            },
        )
        .into_definition()
    }

    fn continue_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// ID of the task to continue.
            task_id: u64,
            /// The follow-up input.
            text: String,
        }
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_continue_task",
            "Send follow-up input to a completed or failed task to resume its work.",
            move |args: Args| {
                let bound = bound.clone();
                async move { Ok(Value::String(bound.continue_task(args.task_id, args.text)?)) }
            },
        )
        .into_definition()
    }

    fn continue_task(&self, id: u64, text: String) -> Result<String> {
        if lock(&self.runtime.inner).closed {
            return Ok(
                "Error: Session is being released; cannot continue a background task.".into(),
            );
        }
        let tasks = refresh(&self.session, self.source_id(), &self.runtime)?;
        let Some(info) = tasks.iter().find(|t| t.id == id) else {
            return Ok(format!("Error: No task found with ID {id}."));
        };
        match info.status {
            BackgroundTaskStatus::Lost => {
                return Ok(format!(
                    "Error: Task {id} cannot be continued because its session was lost. Start a new task instead."
                ))
            }
            BackgroundTaskStatus::Running => {
                return Ok(format!(
                    "Error: Task {id} is still running. Wait for it to complete before continuing."
                ))
            }
            _ => {}
        }
        let Some(agent) = self.provider.agent(&info.agent_name).cloned() else {
            return Ok(format!(
                "Error: Agent '{}' is no longer available.",
                info.agent_name
            ));
        };
        let mut inner = lock(&self.runtime.inner);
        if inner.closed {
            return Ok(
                "Error: Session is being released; cannot continue a background task.".into(),
            );
        }
        let Some(sub_session) = inner.sessions.remove(&id) else {
            return Ok(format!(
                "Error: Session for task {id} is no longer available."
            ));
        };
        self.spawn(&mut inner, id, agent, text, sub_session);
        let (next_id, mut tasks) = load_state(&self.session, self.source_id())?;
        if let Some(info) = tasks.iter_mut().find(|t| t.id == id) {
            info.status = BackgroundTaskStatus::Running;
            info.result_text = None;
            info.error_text = None;
        }
        save_state(&self.session, self.source_id(), next_id, &tasks)?;
        Ok(format!("Task {id} continued with new input."))
    }

    fn clear_tool(&self) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// ID of the task to clear.
            task_id: u64,
        }
        let bound = self.clone();
        FunctionTool::typed(
            "background_agents_clear_completed_task",
            "Remove a completed or failed task and release its session to free memory.",
            move |args: Args| {
                let bound = bound.clone();
                async move {
                    let id = args.task_id;
                    if lock(&bound.runtime.inner).closed {
                        return Ok(Value::String(
                            "Error: Session is being released; cannot clear tasks.".into(),
                        ));
                    }
                    let tasks = refresh(&bound.session, bound.source_id(), &bound.runtime)?;
                    let Some(info) = tasks.iter().find(|t| t.id == id) else {
                        return Ok(Value::String(format!("Error: No task found with ID {id}.")));
                    };
                    if info.status == BackgroundTaskStatus::Running {
                        return Ok(Value::String(format!(
                            "Error: Task {id} is still running. Wait for it to complete before clearing."
                        )));
                    }
                    let mut inner = lock(&bound.runtime.inner);
                    let (next_id, mut tasks) = load_state(&bound.session, bound.source_id())?;
                    tasks.retain(|t| t.id != id);
                    inner.in_flight.remove(&id);
                    inner.sessions.remove(&id);
                    save_state(&bound.session, bound.source_id(), next_id, &tasks)?;
                    Ok(Value::String(format!("Task {id} cleared.")))
                }
            },
        )
        .into_definition()
    }
}

#[async_trait]
impl ContextProvider for BackgroundAgentsProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = ctx.session.clone().ok_or_else(|| {
            Error::Configuration(
                "BackgroundAgentsProvider requires a session to track its tasks.".into(),
            )
        })?;
        ctx.add_instructions(self.shared.instructions.clone());
        ctx.tools.extend(self.tools(&session));
        // Refresh first so tasks that finished between turns show their
        // outcome.
        let tasks = self.tasks(&session)?;
        if !tasks.is_empty() {
            let mut lines = vec!["### Current background tasks".to_string()];
            lines.extend(tasks.iter().map(task_line));
            ctx.messages.push(Message::user(lines.join("\n")));
        }
        Ok(())
    }
}

fn task_line(t: &BackgroundTaskInfo) -> String {
    format!(
        "- Task {} [{}] ({}): {}",
        t.id,
        t.status.as_str(),
        t.agent_name,
        t.description
    )
}

/// The persisted `(next_task_id, tasks)` of `session`.
fn load_state(session: &AgentSession, source_id: &str) -> Result<(u64, Vec<BackgroundTaskInfo>)> {
    let state = read_state_object(session, source_id)?;
    let next_id = state
        .get("next_task_id")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let tasks = match state.get("tasks") {
        None | Some(Value::Null) => Vec::new(),
        Some(raw) => serde_json::from_value(raw.clone()).map_err(|e| {
            Error::Serialization(format!(
                "background task state for source_id {source_id:?} is malformed: {e}"
            ))
        })?,
    };
    Ok((next_id, tasks))
}

fn save_state(
    session: &AgentSession,
    source_id: &str,
    next_id: u64,
    tasks: &[BackgroundTaskInfo],
) -> Result<()> {
    let mut state = read_state_object(session, source_id)?;
    state.insert("next_task_id".into(), json!(next_id));
    state.insert("tasks".into(), serde_json::to_value(tasks)?);
    session.state.insert(source_id, Value::Object(state));
    Ok(())
}

/// Record a finished execution's outcome on `info`.
fn finalize(info: &mut BackgroundTaskInfo, finished: &InFlight) {
    match finished.outcome() {
        Some(Outcome::Completed(text)) => {
            info.status = BackgroundTaskStatus::Completed;
            info.result_text = Some(text);
        }
        Some(Outcome::Failed(error)) => {
            info.status = BackgroundTaskStatus::Failed;
            info.error_text = Some(error);
        }
        None => {
            info.status = BackgroundTaskStatus::Failed;
            info.error_text = Some("Task was canceled.".into());
        }
    }
}

/// Mark running tasks that have finished, or whose runtime is gone, and
/// return the updated list.
fn refresh(
    session: &AgentSession,
    source_id: &str,
    runtime: &Runtime,
) -> Result<Vec<BackgroundTaskInfo>> {
    let mut inner = lock(&runtime.inner);
    let (next_id, mut tasks) = load_state(session, source_id)?;
    let mut changed = false;
    for info in tasks
        .iter_mut()
        .filter(|t| t.status == BackgroundTaskStatus::Running)
    {
        match inner.in_flight.get(&info.id).cloned() {
            None => {
                info.status = BackgroundTaskStatus::Lost;
                changed = true;
            }
            Some(task) if task.is_done() => {
                finalize(info, &task);
                inner.in_flight.remove(&info.id);
                changed = true;
            }
            Some(_) => {}
        }
    }
    if changed {
        save_state(session, source_id, next_id, &tasks)?;
    }
    Ok(tasks)
}

/// A loop condition that continues while `provider` has running tasks for
/// the session; see [`background_tasks_running`].
#[derive(Debug, Clone)]
pub struct BackgroundTasksRunning {
    provider: BackgroundAgentsProvider,
}

/// A [`LoopAgent`](super::LoopAgent) condition that continues while any of
/// `provider`'s tasks is marked running in the session's persisted state.
/// Pair it with a maximum iteration count.
pub fn background_tasks_running(provider: &BackgroundAgentsProvider) -> BackgroundTasksRunning {
    BackgroundTasksRunning {
        provider: provider.clone(),
    }
}

fn running(provider: &BackgroundAgentsProvider, session: &AgentSession) -> Vec<BackgroundTaskInfo> {
    provider
        .persisted_tasks(session)
        .unwrap_or_default()
        .into_iter()
        .filter(|t| t.status == BackgroundTaskStatus::Running)
        .collect()
}

#[async_trait]
impl LoopCallback<LoopDecision> for BackgroundTasksRunning {
    async fn call(&self, ctx: &LoopContext) -> Result<LoopDecision> {
        Ok(LoopDecision::from(
            !running(&self.provider, &ctx.session).is_empty(),
        ))
    }
}

/// A next-message callback listing the running tasks; see
/// [`background_tasks_running_message`].
#[derive(Debug, Clone)]
pub struct BackgroundTasksRunningMessage {
    provider: BackgroundAgentsProvider,
}

/// A next-message callback that reminds the agent which background tasks
/// are still running, to pair with [`background_tasks_running`]. Yields
/// `None` (the loop's default handling) when none is.
pub fn background_tasks_running_message(
    provider: &BackgroundAgentsProvider,
) -> BackgroundTasksRunningMessage {
    BackgroundTasksRunningMessage {
        provider: provider.clone(),
    }
}

#[async_trait]
impl LoopCallback<Option<Vec<Message>>> for BackgroundTasksRunningMessage {
    async fn call(&self, ctx: &LoopContext) -> Result<Option<Vec<Message>>> {
        let running = running(&self.provider, &ctx.session);
        if running.is_empty() {
            return Ok(None);
        }
        let lines: Vec<String> = running
            .iter()
            .map(|t| format!("- #{} ({}): {}", t.id, t.agent_name, t.description))
            .collect();
        Ok(Some(vec![Message::user(format!(
            "You still have {} background task(s) running that must finish before you can complete the \
             work:\n{}\n\nWait for these tasks to complete, retrieve their results, and incorporate them. Only \
             stop once every background task has finished.",
            running.len(),
            lines.join("\n")
        ))]))
    }
}
