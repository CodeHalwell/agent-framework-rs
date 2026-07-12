//! Group chat orchestration: a manager coordinates a multi-agent conversation
//! by selecting who speaks next. Rust analogue of `_group_chat.py`,
//! `_base_group_chat_orchestrator.py`, and `_orchestrator_helpers.py`.
//!
//! Three manager flavours are supported:
//! - [`RoundRobinManager`] (the default) cycles through participants in order;
//! - a **custom** manager — any [`GroupChatManager`] implementation or a sync
//!   closure via [`GroupChatBuilder::manager_fn`] — decides the next speaker or
//!   finishes;
//! - an **LLM manager** — a [`ChatAgent`](crate::agent::ChatAgent) (any
//!   [`Agent`]) prompted with [`DEFAULT_MANAGER_INSTRUCTIONS`] that returns a
//!   [`ManagerSelectionResponse`] as JSON.
//!
//! Divergence from Python: the reference builds a graph of orchestrator +
//! participant nodes. Here a single orchestrator [`Executor`] drives the loop,
//! calling participants via [`Agent::run`] directly. Python's group chat has no
//! built-in round-robin manager (it requires `set_manager` or
//! `set_select_speakers_func`); round-robin is a Rust convenience. Python's
//! manager rounds default to unlimited (bounded only by the workflow's
//! `DEFAULT_MAX_ITERATIONS = 100`); here [`GroupChatBuilder`] applies
//! [`DEFAULT_GROUP_CHAT_MAX_ITERATIONS`] as a safety bound when no
//! [`max_rounds`](GroupChatBuilder::max_rounds) is set.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ensure_author, parse_conversation, run_agent_and_emit};
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::types::Message;
use crate::workflow::{Executor, Workflow, WorkflowBuilder, WorkflowContext};

/// Safety bound on manager rounds applied by [`GroupChatBuilder`] when no
/// explicit [`max_rounds`](GroupChatBuilder::max_rounds) is configured.
///
/// Python leaves manager rounds unbounded (capped only by the workflow's
/// `DEFAULT_MAX_ITERATIONS = 100`); this Rust default prevents an unbounded
/// round-robin or LLM manager from looping forever.
pub const DEFAULT_GROUP_CHAT_MAX_ITERATIONS: usize = 40;

/// Default instructions for an LLM group-chat manager. Ported verbatim from
/// Python's `DEFAULT_MANAGER_INSTRUCTIONS`.
pub const DEFAULT_MANAGER_INSTRUCTIONS: &str =
    "You are coordinating a team conversation to solve the user's task.
Your role is to orchestrate collaboration between multiple participants by selecting who speaks next.
Leverage each participant's unique expertise as described in their descriptions.
Have participants build on each other's contributions - earlier participants gather information,
later ones refine and synthesize.
Only finish the task after multiple relevant participants have contributed their expertise.";

/// The structured-output contract described to an LLM manager. Ported verbatim
/// from Python's `DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT` (the
/// `ManagerDirectiveModel` field naming).
pub const DEFAULT_MANAGER_STRUCTURED_OUTPUT_PROMPT: &str =
    "Return your decision using the following structure:
- next_agent: name of the participant who should act next (use null when finish is true)
- message: instruction for that participant (empty string if not needed)
- finish: boolean indicating if the task is complete
- final_response: when finish is true, provide the final answer to the user";

/// The JSON schema instruction appended to the manager prompt so the model
/// emits a [`ManagerSelectionResponse`] the orchestrator can parse.
const MANAGER_SELECTION_SCHEMA_PROMPT: &str = "Respond with ONLY a JSON object of the form:
{\"selected_participant\": <participant name, or null to finish>, \"instruction\": <string, may be empty>, \
\"finish\": <true or false>, \"final_message\": <final answer string when finish is true, else null>}";

/// An instruction emitted by a [`GroupChatManager`]: either route to a
/// participant, or finish the conversation. Rust analogue of
/// `GroupChatDirective`.
#[derive(Debug, Clone)]
pub enum GroupChatDirective {
    /// Route the conversation to `participant`, optionally with an instruction.
    Speak {
        /// The participant that should speak next.
        participant: String,
        /// An optional instruction appended as a manager message before the turn.
        instruction: Option<String>,
    },
    /// Finish the conversation, optionally with a final message.
    Finish {
        /// The final message to append (defaults to a completion notice).
        final_message: Option<Message>,
    },
}

impl GroupChatDirective {
    /// Route to `participant` with no instruction.
    pub fn speak(participant: impl Into<String>) -> Self {
        GroupChatDirective::Speak {
            participant: participant.into(),
            instruction: None,
        }
    }

    /// Route to `participant` with an instruction.
    pub fn speak_with(participant: impl Into<String>, instruction: impl Into<String>) -> Self {
        GroupChatDirective::Speak {
            participant: participant.into(),
            instruction: Some(instruction.into()),
        }
    }

    /// Finish with a default completion message.
    pub fn finish() -> Self {
        GroupChatDirective::Finish {
            final_message: None,
        }
    }

    /// Finish with an explicit final message.
    pub fn finish_with(message: Message) -> Self {
        GroupChatDirective::Finish {
            final_message: Some(message),
        }
    }

    /// Finish with a plain-text final answer.
    pub fn finish_text(text: impl Into<String>) -> Self {
        GroupChatDirective::Finish {
            final_message: Some(Message::assistant(text)),
        }
    }
}

/// A snapshot of orchestration state handed to a [`GroupChatManager`] for its
/// speaker-selection decision. Rust analogue of `GroupChatStateSnapshot`.
#[derive(Debug, Clone)]
pub struct GroupChatState {
    /// The original task message, if any.
    pub task: Option<Message>,
    /// Registered participants as `(name, description)` pairs, in order.
    pub participants: Vec<(String, String)>,
    /// The full running conversation transcript.
    pub conversation: Vec<Message>,
    /// The number of participant turns taken so far.
    pub round_index: usize,
}

/// The decision-making interface for a group chat. Implementations pick the
/// next speaker or finish the conversation. Rust analogue of the manager
/// callable / `set_manager` agent in Python.
#[async_trait]
pub trait GroupChatManager: Send + Sync {
    /// Decide the next [`GroupChatDirective`] from the current state.
    async fn select(&self, state: &GroupChatState) -> Result<GroupChatDirective>;

    /// A stable display name used to attribute manager-authored messages.
    fn name(&self) -> &str {
        "group_chat_manager"
    }
}

/// The default manager: cycles through participants in registration order.
///
/// Not present in Python (which requires an explicit manager); provided here as
/// an ergonomic default. Termination is enforced by the orchestrator's
/// `max_rounds` bound.
pub struct RoundRobinManager {
    name: String,
}

impl RoundRobinManager {
    /// Create a round-robin manager.
    pub fn new() -> Self {
        Self {
            name: "round_robin_manager".to_string(),
        }
    }
}

impl Default for RoundRobinManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl GroupChatManager for RoundRobinManager {
    async fn select(&self, state: &GroupChatState) -> Result<GroupChatDirective> {
        if state.participants.is_empty() {
            return Ok(GroupChatDirective::finish());
        }
        let idx = state.round_index % state.participants.len();
        Ok(GroupChatDirective::speak(state.participants[idx].0.clone()))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A [`GroupChatManager`] backed by a synchronous closure. Created by
/// [`GroupChatBuilder::manager_fn`].
struct FnManager<F> {
    f: F,
    name: String,
}

#[async_trait]
impl<F> GroupChatManager for FnManager<F>
where
    F: Fn(&GroupChatState) -> GroupChatDirective + Send + Sync,
{
    async fn select(&self, state: &GroupChatState) -> Result<GroupChatDirective> {
        Ok((self.f)(state))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// The structured decision an LLM manager returns. Ported from Python's
/// `ManagerSelectionResponse`; the orchestrator parses the manager agent's JSON
/// output into this shape.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ManagerSelectionResponse {
    /// The participant to speak next (`None` when finishing).
    #[serde(default)]
    pub selected_participant: Option<String>,
    /// An optional instruction for the selected participant.
    #[serde(default)]
    pub instruction: Option<String>,
    /// Whether the conversation should complete.
    #[serde(default)]
    pub finish: bool,
    /// The final answer text when finishing.
    #[serde(default)]
    pub final_message: Option<String>,
}

impl ManagerSelectionResponse {
    /// Convert the parsed selection into a [`GroupChatDirective`].
    fn into_directive(self) -> Result<GroupChatDirective> {
        if self.finish {
            let msg = self.final_message.map(Message::assistant);
            return Ok(GroupChatDirective::Finish { final_message: msg });
        }
        match self.selected_participant {
            Some(p) if !p.is_empty() => Ok(GroupChatDirective::Speak {
                participant: p,
                instruction: self.instruction.filter(|s| !s.is_empty()),
            }),
            _ => Err(Error::Workflow(
                "manager selection missing selected_participant when finish=false".into(),
            )),
        }
    }
}

/// An LLM-driven manager: a [`ChatAgent`](crate::agent::ChatAgent) is prompted
/// with the participant roster and running conversation and returns a
/// [`ManagerSelectionResponse`] as JSON. Created by
/// [`GroupChatBuilder::manager_agent`].
struct LlmGroupChatManager {
    agent: Arc<dyn Agent>,
    name: String,
}

impl LlmGroupChatManager {
    /// Build the system context message listing participants (mirrors Python's
    /// `_build_manager_context_message`), plus the manager instructions and the
    /// JSON schema contract.
    fn system_message(&self, state: &GroupChatState) -> Message {
        let roster = state
            .participants
            .iter()
            .map(|(name, desc)| format!("- {name}: {desc}"))
            .collect::<Vec<_>>()
            .join("\n");
        let text = format!(
            "{DEFAULT_MANAGER_INSTRUCTIONS}\n\nAvailable participants:\n{roster}\n\n\
IMPORTANT: Choose only from these exact participant names (case-sensitive).\n\n{MANAGER_SELECTION_SCHEMA_PROMPT}"
        );
        Message::system(text)
    }

    /// Parse the manager agent's response into a [`ManagerSelectionResponse`],
    /// mirroring Python's `_parse_manager_selection`: prefer the structured
    /// `value`, then fall back to parsing the message text as JSON.
    fn parse(response: &crate::types::AgentResponse) -> Result<ManagerSelectionResponse> {
        if let Some(value) = &response.value {
            if let Ok(sel) = serde_json::from_value::<ManagerSelectionResponse>(value.clone()) {
                return Ok(sel);
            }
        }
        let text = response.text();
        serde_json::from_str::<ManagerSelectionResponse>(text.trim()).map_err(|e| {
            Error::Workflow(format!(
                "manager response did not contain valid selection data ({e}). \
                 Ensure the manager agent returns JSON matching ManagerSelectionResponse."
            ))
        })
    }
}

#[async_trait]
impl GroupChatManager for LlmGroupChatManager {
    async fn select(&self, state: &GroupChatState) -> Result<GroupChatDirective> {
        let mut conversation = vec![self.system_message(state)];
        conversation.extend(state.conversation.iter().cloned());
        let response = self.agent.run(conversation, None).await?;
        let selection = Self::parse(&response)?;
        selection.into_directive()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

type TerminationCondition = Arc<dyn Fn(&[Message]) -> bool + Send + Sync>;

/// The single executor that drives a group chat conversation.
struct GroupChatOrchestrator {
    id: String,
    manager: Arc<dyn GroupChatManager>,
    participants: Vec<(String, Arc<dyn Agent>)>,
    descriptions: Vec<(String, String)>,
    manager_name: String,
    max_rounds: Option<usize>,
    termination: Option<TerminationCondition>,
}

impl GroupChatOrchestrator {
    fn find(&self, name: &str) -> Option<&Arc<dyn Agent>> {
        self.participants
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
    }
}

#[async_trait]
impl Executor for GroupChatOrchestrator {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let mut conversation = parse_conversation(&message)?;
        let task = conversation.last().cloned();
        let mut round_index = 0usize;

        loop {
            if let Some(max) = self.max_rounds {
                if round_index >= max {
                    conversation.push(ensure_author(
                        Message::assistant(
                            "Conversation halted after reaching manager round limit.",
                        ),
                        &self.manager_name,
                    ));
                    break;
                }
            }
            if let Some(term) = &self.termination {
                if term(&conversation) {
                    conversation.push(ensure_author(
                        Message::assistant(
                            "Conversation halted after termination condition was met.",
                        ),
                        &self.manager_name,
                    ));
                    break;
                }
            }

            let state = GroupChatState {
                task: task.clone(),
                participants: self.descriptions.clone(),
                conversation: conversation.clone(),
                round_index,
            };
            let directive = self.manager.select(&state).await?;

            match directive {
                GroupChatDirective::Finish { final_message } => {
                    let msg = final_message
                        .unwrap_or_else(|| Message::assistant("Conversation completed."));
                    conversation.push(ensure_author(msg, &self.manager_name));
                    break;
                }
                GroupChatDirective::Speak {
                    participant,
                    instruction,
                } => {
                    let agent = self.find(&participant).ok_or_else(|| {
                        Error::Workflow(format!(
                            "manager selected unknown participant '{participant}'"
                        ))
                    })?;
                    if let Some(instr) = instruction {
                        if !instr.is_empty() {
                            conversation
                                .push(ensure_author(Message::assistant(instr), &self.manager_name));
                        }
                    }
                    let response = run_agent_and_emit(
                        agent,
                        conversation.clone(),
                        &self.id,
                        &participant,
                        &ctx,
                    )
                    .await?;
                    conversation.extend(response.messages);
                    round_index += 1;
                }
            }
        }

        let payload = serde_json::to_value(&conversation)
            .map_err(|e| Error::Workflow(format!("failed to serialize conversation: {e}")))?;
        ctx.yield_output(payload).await?;
        Ok(())
    }
}

/// Which manager strategy the builder should install.
enum ManagerChoice {
    RoundRobin,
    Custom(Arc<dyn GroupChatManager>),
    Agent(Arc<dyn Agent>),
}

/// Builder for a group chat workflow. Rust analogue of `GroupChatBuilder`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::prelude::*;
/// # use agent_framework_core::workflow::GroupChatBuilder;
/// # fn demo(writer: Arc<dyn Agent>, reviewer: Arc<dyn Agent>) -> Result<()> {
/// let workflow = GroupChatBuilder::new()
///     .participant("writer", writer)
///     .participant("reviewer", reviewer)
///     .round_robin()
///     .max_rounds(4)
///     .build()?;
/// # let _ = workflow;
/// # Ok(())
/// # }
/// ```
pub struct GroupChatBuilder {
    participants: Vec<(String, String, Arc<dyn Agent>)>,
    manager: ManagerChoice,
    manager_name: Option<String>,
    max_rounds: Option<usize>,
    termination: Option<TerminationCondition>,
    name: Option<String>,
}

impl Default for GroupChatBuilder {
    fn default() -> Self {
        Self {
            participants: Vec::new(),
            manager: ManagerChoice::RoundRobin,
            manager_name: None,
            max_rounds: None,
            termination: None,
            name: None,
        }
    }
}

impl GroupChatBuilder {
    /// Create an empty builder (round-robin manager by default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a participant (its description defaults to its name).
    pub fn participant(mut self, name: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        let name = name.into();
        self.participants.push((name.clone(), name, agent));
        self
    }

    /// Register a participant with an explicit description (shown to an LLM
    /// manager for selection).
    pub fn participant_described(
        mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<dyn Agent>,
    ) -> Self {
        self.participants
            .push((name.into(), description.into(), agent));
        self
    }

    /// Register several participants as `(name, agent)` pairs.
    pub fn participants(
        mut self,
        participants: impl IntoIterator<Item = (String, Arc<dyn Agent>)>,
    ) -> Self {
        for (name, agent) in participants {
            self.participants.push((name.clone(), name, agent));
        }
        self
    }

    /// Use the built-in round-robin manager (the default).
    pub fn round_robin(mut self) -> Self {
        self.manager = ManagerChoice::RoundRobin;
        self
    }

    /// Use a custom [`GroupChatManager`].
    pub fn manager(mut self, manager: Arc<dyn GroupChatManager>) -> Self {
        self.manager = ManagerChoice::Custom(manager);
        self
    }

    /// Use a custom synchronous closure to select the next speaker or finish.
    pub fn manager_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&GroupChatState) -> GroupChatDirective + Send + Sync + 'static,
    {
        self.manager = ManagerChoice::Custom(Arc::new(FnManager {
            f,
            name: "custom_manager".to_string(),
        }));
        self
    }

    /// Use an LLM agent as the manager (prompted with
    /// [`DEFAULT_MANAGER_INSTRUCTIONS`] and asked for a
    /// [`ManagerSelectionResponse`] as JSON).
    pub fn manager_agent(mut self, agent: Arc<dyn Agent>) -> Self {
        self.manager = ManagerChoice::Agent(agent);
        self
    }

    /// Cap the number of manager rounds. When unset, the builder applies
    /// [`DEFAULT_GROUP_CHAT_MAX_ITERATIONS`].
    pub fn max_rounds(mut self, max_rounds: usize) -> Self {
        self.max_rounds = Some(max_rounds);
        self
    }

    /// Set a termination condition evaluated against the running conversation
    /// before each manager turn.
    pub fn termination_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&[Message]) -> bool + Send + Sync + 'static,
    {
        self.termination = Some(Arc::new(condition));
        self
    }

    /// Override the manager's display name (used to attribute its messages).
    pub fn manager_name(mut self, name: impl Into<String>) -> Self {
        self.manager_name = Some(name.into());
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validate and build the group chat workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "group chat needs at least one participant".into(),
            ));
        }

        let manager: Arc<dyn GroupChatManager> = match self.manager {
            ManagerChoice::RoundRobin => Arc::new(RoundRobinManager::new()),
            ManagerChoice::Custom(m) => m,
            ManagerChoice::Agent(agent) => Arc::new(LlmGroupChatManager {
                agent,
                name: "manager".to_string(),
            }),
        };
        let manager_name = self
            .manager_name
            .unwrap_or_else(|| manager.name().to_string());

        let descriptions: Vec<(String, String)> = self
            .participants
            .iter()
            .map(|(n, d, _)| (n.clone(), d.clone()))
            .collect();
        let participants: Vec<(String, Arc<dyn Agent>)> = self
            .participants
            .into_iter()
            .map(|(n, _, a)| (n, a))
            .collect();

        let orchestrator = GroupChatOrchestrator {
            id: "group_chat_orchestrator".to_string(),
            manager,
            participants,
            descriptions,
            manager_name,
            max_rounds: Some(self.max_rounds.unwrap_or(DEFAULT_GROUP_CHAT_MAX_ITERATIONS)),
            termination: self.termination,
        };

        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(orchestrator))
            .set_start("group_chat_orchestrator");
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}
