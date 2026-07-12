//! Handoff orchestration: agents transfer control to one another via synthetic
//! `handoff_to_<name>` tool calls. Rust analogue of `_handoff.py`.
//!
//! A coordinator runs the current agent over the full conversation; if the
//! agent emits a handoff tool call, a synthetic tool result acknowledging the
//! transfer is appended and control moves to the target. Otherwise the turn's
//! response is the interaction result:
//! - in [`HandoffInteractionMode::Autonomous`] (single-shot) the workflow
//!   completes, yielding the conversation;
//! - in [`HandoffInteractionMode::HumanInLoop`] the coordinator emits a
//!   [`request_info`](WorkflowContext::request_info) asking the caller for the
//!   next user message, then resumes when the response arrives.
//!
//! Divergences from Python (documented): (1) because a built
//! [`ChatAgent`](crate::agent::ChatAgent)'s tool list is fixed and the [`Agent`]
//! trait exposes no tool mutation, the coordinator detects handoffs by
//! **inspecting** the agent's response for a handoff-shaped function call
//! (matching Python's `_resolve_handoff_target`) rather than injecting an
//! auto-handoff middleware; [`handoff_tool_spec`] generates the marker tool
//! specs so callers can attach them to their agents. (2) Autonomous mode
//! completes on the first non-handoff answer (a true single-shot) rather than
//! re-prompting the same agent until a turn limit as Python does.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{parse_conversation, run_agent_and_emit};
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::tools::{ApprovalMode, ToolDefinition, ToolKind};
use crate::types::{Content, FunctionCallContent, FunctionResultContent, Message, Role};
use crate::workflow::{Executor, RequestResponse, Workflow, WorkflowBuilder, WorkflowContext};

/// Whether the workflow pauses for fresh user input between agent turns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandoffInteractionMode {
    /// After an agent answers without handing off, request the next user
    /// message via the engine's request-info machinery (the default; mirrors
    /// Python's `human_in_loop`).
    #[default]
    HumanInLoop,
    /// Complete the workflow on the first non-handoff answer (single-shot).
    Autonomous,
}

/// The payload of the request-info event emitted when the workflow needs fresh
/// user input. Rust analogue of Python's `HandoffUserInputRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandoffUserInputRequest {
    /// The conversation so far (cleaned of tool-call plumbing).
    pub conversation: Vec<Message>,
    /// The id of the agent awaiting the user's reply.
    pub awaiting_agent: String,
    /// A human-facing prompt describing what input is needed.
    pub prompt: String,
}

/// The default handoff turn limit (Python's `_DEFAULT_AUTONOMOUS_TURN_LIMIT`).
const DEFAULT_TURN_LIMIT: usize = 50;

/// Build the marker tool spec for handing off to `target`. Rust analogue of
/// Python's `_create_handoff_tool`.
///
/// The returned [`ToolDefinition`] is a **non-executable marker** (no local
/// executor): the coordinator intercepts calls to it rather than running it.
/// Attach it to an agent's tools so its model can request the transfer.
pub fn handoff_tool_spec(target: &str, description: Option<&str>) -> ToolDefinition {
    let name = format!("handoff_to_{}", sanitize_identifier(target));
    let desc = description
        .map(str::to_string)
        .unwrap_or_else(|| format!("Handoff to the {target} agent."));
    ToolDefinition {
        name,
        description: desc,
        parameters: json!({
            "type": "object",
            "properties": {
                "context": { "type": "string", "description": "Optional context for the handoff." }
            }
        }),
        kind: ToolKind::Function,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// Deterministic, lowercase identifier derived from `value`. Rust analogue of
/// Python's `sanitize_identifier`.
fn sanitize_identifier(value: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
            prev_underscore = ch == '_';
        } else if !prev_underscore {
            out.push('_');
            prev_underscore = true;
        }
    }
    let mut cleaned = out.trim_matches('_').to_string();
    if cleaned.is_empty() {
        cleaned = "agent".to_string();
    }
    if cleaned
        .chars()
        .next()
        .map(|c| c.is_ascii_digit())
        .unwrap_or(false)
    {
        cleaned = format!("agent_{cleaned}");
    }
    cleaned.to_lowercase()
}

/// The outcome of inspecting an agent response for a handoff request.
enum HandoffResolution {
    /// No handoff call present.
    None,
    /// A handoff to a known participant.
    Known {
        target: String,
        call: FunctionCallContent,
    },
    /// A handoff to an unregistered target (fed back as an error).
    Unknown {
        name: String,
        call: FunctionCallContent,
    },
}

/// Parse a handoff target from a tool name (`handoff_to_x` / `transfer to x`).
fn target_from_tool_name(name: &str) -> Option<String> {
    let lower = name.to_lowercase();
    for prefix in ["handoff", "transfer"] {
        if let Some(rest) = lower.strip_prefix(prefix) {
            let rest = rest.trim_start_matches(['_', ' ', '-']);
            if let Some(after_to) = rest.strip_prefix("to") {
                let target = after_to.trim_start_matches(['_', ' ', '-']);
                if !target.is_empty() {
                    return Some(target.to_string());
                }
            }
        }
    }
    None
}

/// Extract the handoff target encoded by a function call (tool name or a
/// `handoff_to` argument). Rust analogue of `_target_from_function_call`.
fn target_from_call(call: &FunctionCallContent) -> Option<String> {
    if let Some(t) = target_from_tool_name(&call.name) {
        return Some(t);
    }
    if let Ok(args) = call.parse_arguments() {
        if let Some(Value::String(v)) = args.get("handoff_to") {
            let v = v.trim();
            if !v.is_empty() {
                return Some(v.to_string());
            }
        }
    }
    None
}

impl HandoffResolution {
    /// Detect a handoff from an agent response. The first handoff-shaped call
    /// wins; additional ones are warned about (Python: "first wins, warn").
    fn detect(
        response: &crate::types::AgentResponse,
        tool_targets: &HashMap<String, String>,
    ) -> HandoffResolution {
        let mut calls: Vec<FunctionCallContent> = Vec::new();
        for msg in &response.messages {
            for content in &msg.contents {
                match content {
                    Content::FunctionCall(fc) => calls.push(fc.clone()),
                    Content::FunctionApprovalRequest(req) => calls.push(req.function_call.clone()),
                    _ => {}
                }
            }
        }
        let mut handoff_calls: Vec<(String, FunctionCallContent)> = calls
            .into_iter()
            .filter_map(|c| target_from_call(&c).map(|t| (t, c)))
            .collect();
        if handoff_calls.is_empty() {
            return HandoffResolution::None;
        }
        if handoff_calls.len() > 1 {
            tracing::warn!(
                "agent emitted {} handoff calls; using the first",
                handoff_calls.len()
            );
        }
        let (candidate, call) = handoff_calls.remove(0);
        match tool_targets.get(&candidate.to_lowercase()) {
            Some(target) => HandoffResolution::Known {
                target: target.clone(),
                call,
            },
            None => HandoffResolution::Unknown {
                name: candidate,
                call,
            },
        }
    }
}

/// Remove tool-call plumbing from a conversation for clean display / routing.
/// Rust analogue of `clean_conversation_for_handoff`.
fn clean_conversation(conversation: &[Message]) -> Vec<Message> {
    let mut cleaned = Vec::new();
    for msg in conversation {
        if msg.role == Role::tool() {
            continue;
        }
        let has_tool_content = msg.contents.iter().any(|c| {
            matches!(
                c,
                Content::FunctionCall(_) | Content::FunctionApprovalRequest(_)
            )
        });
        if !has_tool_content {
            cleaned.push(msg.clone());
            continue;
        }
        let text = msg.text();
        if !text.trim().is_empty() {
            let mut fresh = Message::new(msg.role.clone(), text);
            fresh.author_name = msg.author_name.clone();
            fresh.additional_properties = msg.additional_properties.clone();
            cleaned.push(fresh);
        }
    }
    cleaned
}

/// Coerce an arbitrary request-info response payload into user messages. Rust
/// analogue of `_as_user_messages`.
fn as_user_messages(value: &Value) -> Vec<Message> {
    match value {
        Value::String(s) => vec![Message::user(s.clone())],
        Value::Array(_) => {
            if let Ok(msgs) = serde_json::from_value::<Vec<Message>>(value.clone()) {
                return msgs
                    .into_iter()
                    .map(|m| {
                        if m.role == Role::user() {
                            m
                        } else {
                            Message::user(m.text())
                        }
                    })
                    .collect();
            }
            vec![Message::user(value.to_string())]
        }
        Value::Object(_) => {
            if let Ok(m) = serde_json::from_value::<Message>(value.clone()) {
                return vec![if m.role == Role::user() {
                    m
                } else {
                    Message::user(m.text())
                }];
            }
            if let Some(Value::String(text)) = value.get("text").or_else(|| value.get("content")) {
                return vec![Message::user(text.clone())];
            }
            vec![Message::user(value.to_string())]
        }
        _ => vec![Message::user(value.to_string())],
    }
}

type TerminationCondition = Arc<dyn Fn(&[Message]) -> bool + Send + Sync>;

/// Default termination: stop after 10 user messages (Python's default).
fn default_termination(conversation: &[Message]) -> bool {
    conversation
        .iter()
        .filter(|m| m.role == Role::user())
        .count()
        >= 10
}

/// Persisted coordinator state carried across a request-info pause.
#[derive(Default, Serialize, Deserialize)]
struct HandoffPersisted {
    conversation: Vec<Message>,
    current_agent: String,
}

const HANDOFF_STATE_KEY: &str = "_handoff_state";

/// The single executor that coordinates agent-to-agent handoffs.
struct HandoffCoordinator {
    id: String,
    agents: Vec<(String, Arc<dyn Agent>)>,
    initial_agent: String,
    tool_targets: HashMap<String, String>,
    interaction_mode: HandoffInteractionMode,
    turn_limit: usize,
    termination: TerminationCondition,
    prompt: String,
}

impl HandoffCoordinator {
    fn find(&self, name: &str) -> Option<&Arc<dyn Agent>> {
        self.agents.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }

    /// Append a synthetic tool result acknowledging the resolved handoff target.
    fn append_tool_ack(conversation: &mut Vec<Message>, call: &FunctionCallContent, target: &str) {
        if call.call_id.is_empty() {
            return;
        }
        let result = FunctionResultContent {
            call_id: call.call_id.clone(),
            result: Some(json!({ "handoff_to": target })),
            exception: None,
        };
        let mut msg = Message::with_contents(Role::tool(), vec![Content::FunctionResult(result)]);
        msg.author_name = Some(call.name.clone());
        conversation.push(msg);
    }

    /// Append a tool result reporting an unknown handoff target (fed back to the
    /// agent so it can correct).
    fn append_error_ack(conversation: &mut Vec<Message>, call: &FunctionCallContent, name: &str) {
        let result = FunctionResultContent {
            call_id: call.call_id.clone(),
            result: None,
            exception: Some(format!("Error: unknown handoff target '{name}'.")),
        };
        let mut msg = Message::with_contents(Role::tool(), vec![Content::FunctionResult(result)]);
        msg.author_name = Some(call.name.clone());
        conversation.push(msg);
    }

    async fn save_state(&self, ctx: &WorkflowContext, conversation: &[Message], current: &str) {
        let state = HandoffPersisted {
            conversation: conversation.to_vec(),
            current_agent: current.to_string(),
        };
        if let Ok(value) = serde_json::to_value(&state) {
            ctx.shared_state().set(HANDOFF_STATE_KEY, value).await;
        }
    }

    async fn load_state(&self, ctx: &WorkflowContext) -> HandoffPersisted {
        ctx.shared_state()
            .get(HANDOFF_STATE_KEY)
            .await
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    /// Drive agents starting from `current` over `conversation` until a handoff
    /// terminus (completion or a user-input pause).
    async fn run_loop(
        &self,
        mut conversation: Vec<Message>,
        mut current: String,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        let mut turns = 0usize;
        loop {
            turns += 1;
            if turns > self.turn_limit {
                let out = clean_conversation(&conversation);
                ctx.yield_output(serialize(&out)?).await?;
                return Ok(());
            }

            let agent = self.find(&current).ok_or_else(|| {
                Error::Workflow(format!("handoff routed to unknown agent '{current}'"))
            })?;
            let response =
                run_agent_and_emit(agent, conversation.clone(), &self.id, &current, ctx).await?;
            conversation.extend(response.messages.clone());

            match HandoffResolution::detect(&response, &self.tool_targets) {
                HandoffResolution::Known { target, call } => {
                    Self::append_tool_ack(&mut conversation, &call, &target);
                    current = target;
                    continue;
                }
                HandoffResolution::Unknown { name, call } => {
                    Self::append_error_ack(&mut conversation, &call, &name);
                    continue;
                }
                HandoffResolution::None => {
                    if (self.termination)(&conversation) {
                        let out = clean_conversation(&conversation);
                        ctx.yield_output(serialize(&out)?).await?;
                        return Ok(());
                    }
                    match self.interaction_mode {
                        HandoffInteractionMode::Autonomous => {
                            let out = clean_conversation(&conversation);
                            ctx.yield_output(serialize(&out)?).await?;
                            return Ok(());
                        }
                        HandoffInteractionMode::HumanInLoop => {
                            self.save_state(ctx, &conversation, &current).await;
                            let request = HandoffUserInputRequest {
                                conversation: clean_conversation(&conversation),
                                awaiting_agent: current.clone(),
                                prompt: self.prompt.clone(),
                            };
                            ctx.request_info(serialize(&request)?).await?;
                            return Ok(());
                        }
                    }
                }
            }
        }
    }
}

fn serialize<T: Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(|e| Error::Workflow(format!("serialize error: {e}")))
}

#[async_trait]
impl Executor for HandoffCoordinator {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // A response to a user-input request resumes the paused conversation.
        if let Some(resp) = RequestResponse::from_message(&message) {
            let mut state = self.load_state(&ctx).await;
            let user_msgs = as_user_messages(&resp.data);
            state.conversation.extend(user_msgs);
            if (self.termination)(&state.conversation) {
                let out = clean_conversation(&state.conversation);
                ctx.yield_output(serialize(&out)?).await?;
                return Ok(());
            }
            // Default (return-to-coordinator): route new input to the initial agent.
            let start = self.initial_agent.clone();
            return self.run_loop(state.conversation, start, &ctx).await;
        }

        // Fresh input: start from the initial agent.
        let conversation = parse_conversation(&message)?;
        let start = self.initial_agent.clone();
        self.run_loop(conversation, start, &ctx).await
    }
}

/// Intermediate builder returned by [`HandoffBuilder::add_handoff`] to express
/// `add_handoff(source).to([targets])` fluently.
pub struct HandoffEdgeBuilder {
    builder: HandoffBuilder,
    source: String,
}

impl HandoffEdgeBuilder {
    /// Complete the edge: `source` may hand off to each of `targets`.
    pub fn to<I, S>(mut self, targets: I) -> HandoffBuilder
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let list: Vec<String> = targets.into_iter().map(Into::into).collect();
        self.builder
            .handoff_map
            .entry(self.source)
            .or_default()
            .extend(list);
        self.builder
    }
}

/// Builder for a handoff workflow. Rust analogue of `HandoffBuilder`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::prelude::*;
/// # use agent_framework_core::workflow::HandoffBuilder;
/// # fn demo(triage: Arc<dyn Agent>, billing: Arc<dyn Agent>) -> Result<()> {
/// let workflow = HandoffBuilder::new()
///     .participant("triage", triage)
///     .participant("billing", billing)
///     .initial_agent("triage")
///     .add_handoff("triage").to(["billing"])
///     .autonomous()
///     .build()?;
/// # let _ = workflow;
/// # Ok(())
/// # }
/// ```
pub struct HandoffBuilder {
    participants: Vec<(String, Arc<dyn Agent>)>,
    initial_agent: Option<String>,
    handoff_map: HashMap<String, Vec<String>>,
    interaction_mode: HandoffInteractionMode,
    turn_limit: usize,
    termination: Option<TerminationCondition>,
    prompt: Option<String>,
    name: Option<String>,
}

impl Default for HandoffBuilder {
    fn default() -> Self {
        Self {
            participants: Vec::new(),
            initial_agent: None,
            handoff_map: HashMap::new(),
            interaction_mode: HandoffInteractionMode::default(),
            turn_limit: DEFAULT_TURN_LIMIT,
            termination: None,
            prompt: None,
            name: None,
        }
    }
}

impl HandoffBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a participant by name.
    pub fn participant(mut self, name: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        self.participants.push((name.into(), agent));
        self
    }

    /// Register several participants as `(name, agent)` pairs.
    pub fn participants(
        mut self,
        participants: impl IntoIterator<Item = (String, Arc<dyn Agent>)>,
    ) -> Self {
        self.participants.extend(participants);
        self
    }

    /// Designate the entry-point agent that first receives user input.
    pub fn initial_agent(mut self, name: impl Into<String>) -> Self {
        self.initial_agent = Some(name.into());
        self
    }

    /// Alias for [`HandoffBuilder::initial_agent`] (Python's `set_coordinator`).
    pub fn coordinator(self, name: impl Into<String>) -> Self {
        self.initial_agent(name)
    }

    /// Begin a handoff edge from `source`; complete it with
    /// [`HandoffEdgeBuilder::to`].
    pub fn add_handoff(self, source: impl Into<String>) -> HandoffEdgeBuilder {
        HandoffEdgeBuilder {
            source: source.into(),
            builder: self,
        }
    }

    /// Set the interaction mode.
    pub fn interaction_mode(mut self, mode: HandoffInteractionMode) -> Self {
        self.interaction_mode = mode;
        self
    }

    /// Run autonomously (single-shot): complete on the first non-handoff answer.
    pub fn autonomous(mut self) -> Self {
        self.interaction_mode = HandoffInteractionMode::Autonomous;
        self
    }

    /// Request fresh user input between turns (the default).
    pub fn with_user_input_request(mut self) -> Self {
        self.interaction_mode = HandoffInteractionMode::HumanInLoop;
        self
    }

    /// Cap the number of agent turns per user message (Python's autonomous turn
    /// limit; default 50).
    pub fn max_iterations(mut self, limit: usize) -> Self {
        self.turn_limit = limit.max(1);
        self
    }

    /// Set a termination condition evaluated against the conversation.
    pub fn termination_condition<F>(mut self, condition: F) -> Self
    where
        F: Fn(&[Message]) -> bool + Send + Sync + 'static,
    {
        self.termination = Some(Arc::new(condition));
        self
    }

    /// Set the prompt shown when requesting user input.
    pub fn request_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Validate and build the handoff workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "handoff workflow needs at least one participant".into(),
            ));
        }
        let initial_agent = self
            .initial_agent
            .clone()
            .unwrap_or_else(|| self.participants[0].0.clone());
        if !self.participants.iter().any(|(n, _)| n == &initial_agent) {
            return Err(Error::Workflow(format!(
                "initial agent '{initial_agent}' is not a registered participant"
            )));
        }

        // Build the resolution table: any registered participant is a valid
        // handoff target, addressable by name or `handoff_to_<name>`.
        let mut tool_targets: HashMap<String, String> = HashMap::new();
        for (name, _) in &self.participants {
            let sanitized = sanitize_identifier(name);
            tool_targets.insert(format!("handoff_to_{sanitized}"), name.clone());
            tool_targets.insert(sanitized, name.clone());
            tool_targets.insert(name.to_lowercase(), name.clone());
        }

        let coordinator = HandoffCoordinator {
            id: "handoff_coordinator".to_string(),
            agents: self.participants,
            initial_agent,
            tool_targets,
            interaction_mode: self.interaction_mode,
            turn_limit: self.turn_limit,
            termination: self
                .termination
                .unwrap_or_else(|| Arc::new(default_termination)),
            prompt: self
                .prompt
                .unwrap_or_else(|| "Provide your next input for the conversation.".to_string()),
        };

        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(coordinator))
            .set_start("handoff_coordinator");
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}
