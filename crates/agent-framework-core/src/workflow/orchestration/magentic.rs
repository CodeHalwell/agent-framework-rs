//! Magentic-One style orchestration: a manager gathers facts and a plan, then
//! coordinates participants round by round via a structured progress ledger,
//! replanning on stalls. Rust analogue of `_magentic.py`.
//!
//! The [`StandardMagenticManager`] drives an LLM ([`Agent`]) with the ported
//! Magentic-One prompts to plan, produce a JSON progress ledger each round,
//! replan on stall, and synthesize the final answer. The [`MagenticManager`]
//! trait lets callers supply a fully custom manager.
//!
//! Divergences from Python (documented): a single orchestrator [`Executor`]
//! drives the loop and calls participants via [`Agent::run`] directly (Python
//! wires a graph of agent nodes); the progress-ledger retry loop has no
//! backoff sleep.
//!
//! ## Human-in-the-loop plan review
//!
//! [`MagenticBuilder::with_plan_review`] enables a pause after the initial
//! plan (mirrors Python's `MagenticOrchestratorExecutor(require_plan_signoff=True)`,
//! wired from `MagenticBuilder.with_plan_review()`): the orchestrator emits a
//! [`MagenticPlanReviewRequest`] via [`WorkflowContext::request_info`] and
//! suspends; [`crate::workflow::WorkflowRun::send_response`] with a
//! [`MagenticPlanReviewDecision`] resumes it. The loop semantics mirror
//! `_handle_plan_review_response` in `_magentic.py` exactly:
//!
//! - `Approve { edited_plan: Some(text), .. }` adopts `text` as the plan
//!   directly (no LLM call), re-renders the combined ledger from the
//!   unchanged facts, and proceeds.
//! - `Approve { comments: Some(text), .. }` (no `edited_plan`) records `text`
//!   as human feedback and calls [`MagenticManager::replan`] (one LLM call)
//!   before proceeding.
//! - `Approve { edited_plan: None, comments: None }` proceeds with the
//!   ledger unchanged.
//! - `Revise` repeats the same edited-plan/comments handling but, instead of
//!   proceeding, re-sends another `MagenticPlanReviewRequest` (looping until
//!   an `Approve`) — *unless* the review round count exceeds
//!   [`MagenticBuilder::max_plan_review_rounds`] (Python default, and this
//!   port's default: 10 *revise* rounds), in which case it force-proceeds
//!   with whatever plan is current, appending a notice message, exactly as
//!   Python does.
//! - Plan review only fires once, right after the initial plan; stall-
//!   triggered replans later in the run never re-open it (Python doesn't
//!   either — `_reset_and_replan` never calls `_send_plan_review_request`).
//!
//! ## Human-in-the-loop stall intervention
//!
//! [`MagenticBuilder::with_stall_intervention`] enables a pause when the round
//! loop detects a stall (mirrors Python's
//! `MagenticBuilder.with_human_input_on_stall()` /
//! `MagenticHumanInterventionKind.STALL`): instead of silently auto-replanning
//! once `stall_count` exceeds `max_stall_count`, the orchestrator emits a
//! [`MagenticStallInterventionRequest`] (task, stall reason/details, counts,
//! round, resets-so-far, current facts/plan, last agent) via
//! [`WorkflowContext::request_info`] and suspends. A
//! [`MagenticStallInterventionDecision`] resumes it:
//!
//! - `Continue` clears the stall counter and resumes the round loop as-is
//!   (Python's `CONTINUE`).
//! - `Replan { guidance }` takes the existing reset/replan path (Python's
//!   `REPLAN`); when `guidance` is present it is appended to the history after
//!   the fresh task ledger as `"Human guidance to help with stall: …"` so the
//!   manager sees it on the next round (folding in Python's separate
//!   `GUIDANCE` decision, whose free-text comments this port routes through
//!   the same variant).
//! - `Abort` stops coordinating and synthesizes the final answer from whatever
//!   history exists (a small superset of Python's stall decisions, which offer
//!   no explicit abort — documented divergence).
//!
//! With stall intervention disabled the behavior is unchanged: stalls always
//! auto-replan. Like plan review, the paused round-loop state
//! ([`MagenticContext`]) is persisted through [`WorkflowContext::shared_state`]
//! so the orchestrator executor stays stateless across the pause.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{ensure_author, parse_conversation, run_agent_and_emit};
use crate::agent::Agent;
use crate::error::{Error, Result};
use crate::types::{AgentRunResponse, ChatMessage, Role};
use crate::workflow::{
    Executor, RequestResponse, Workflow, WorkflowBuilder, WorkflowContext, WorkflowEvent,
};

/// The author name used for orchestrator/manager-generated messages.
pub const MAGENTIC_MANAGER_NAME: &str = "magentic_manager";

// region Magentic One prompts (ported verbatim from `_magentic.py`)

/// Facts pre-survey prompt. Placeholder: `{task}`.
pub const ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT: &str = r#"Below I will present you a request.

Before we begin addressing the request, please answer the following pre-survey to the best of your ability.
Keep in mind that you are Ken Jennings-level with trivia, and Mensa-level with puzzles, so there should be
a deep well to draw from.

Here is the request:

{task}

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the request itself. It is possible that
       there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found.
       In some cases, authoritative sources are mentioned in the request itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc.
Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"#;

/// Plan prompt. Placeholder: `{team}`.
pub const ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT: &str = r#"Fantastic. To address this request we have assembled the following team:

{team}

Based on the team composition, and known and unknown facts, please devise a short bullet-point plan for addressing the
original request. Remember, there is no requirement to involve all team members. A team member's particular expertise
may not be needed for this task.
"#;

/// Combined task-ledger render. Placeholders: `{task}`, `{team}`, `{facts}`, `{plan}`.
pub const ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT: &str = r#"
We are working to address the following user request:

{task}


To answer this request we have assembled the following team:

{team}


Here is an initial fact sheet to consider:

{facts}


Here is the plan to follow as best as possible:

{plan}
"#;

/// Facts-update prompt used on replan. Placeholders: `{task}`, `{old_facts}`.
pub const ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT: &str = r#"As a reminder, we are working to solve the following task:

{task}

It is clear we are not making as much progress as we would like, but we may have learned something new.
Please rewrite the following fact sheet, updating it to include anything new we have learned that may be helpful.

Example edits can include (but are not limited to) adding new guesses, moving educated guesses to verified facts
if appropriate, etc. Updates may be made to any section of the fact sheet, and more than one section of the fact
sheet can be edited. This is an especially good time to update educated guesses, so please at least add or update
one educated guess or hunch, and explain your reasoning.

Here is the old fact sheet:

{old_facts}
"#;

/// Plan-update prompt used on replan. Placeholder: `{team}`.
pub const ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT: &str = r#"Please briefly explain what went wrong on this last run
(the root cause of the failure), and then come up with a new plan that takes steps and includes hints to overcome prior
challenges and especially avoids repeating the same mistakes. As before, the new plan should be concise, expressed in
bullet-point form, and consider the following team composition:

{team}
"#;

/// Progress-ledger prompt requesting structured JSON. Placeholders: `{task}`,
/// `{team}`, `{names}`.
pub const ORCHESTRATOR_PROGRESS_LEDGER_PROMPT: &str = r#"
Recall we are working on the following request:

{task}

And we have assembled the following team:

{team}

To make progress on the request, please answer the following questions, including necessary reasoning:

    - Is the request fully satisfied? (True if complete, or False if the original request has yet to be
      SUCCESSFULLY and FULLY addressed)
    - Are we in a loop where we are repeating the same requests and or getting the same responses as before?
      Loops can span multiple turns, and can include repeated actions like scrolling up or down more than a
      handful of times.
    - Are we making forward progress? (True if just starting, or recent messages are adding value. False if recent
      messages show evidence of being stuck in a loop or if there is evidence of significant barriers to success
      such as the inability to read from a required file)
    - Who should speak next? (select from: {names})
    - What instruction or question would you give this team member? (Phrase as if speaking directly to them, and
      include any specific information they may need)

Please output an answer in pure JSON format according to the following schema. The JSON object must be parsable as-is.
DO NOT OUTPUT ANYTHING OTHER THAN JSON, AND DO NOT DEVIATE FROM THIS SCHEMA:

{
    "is_request_satisfied": {

        "reason": string,
        "answer": boolean
    },
    "is_in_loop": {
        "reason": string,
        "answer": boolean
    },
    "is_progress_being_made": {
        "reason": string,
        "answer": boolean
    },
    "next_speaker": {
        "reason": string,
        "answer": string (select from: {names})
    },
    "instruction_or_question": {
        "reason": string,
        "answer": string
    }
}
"#;

/// Final-answer synthesis prompt. Placeholder: `{task}`.
pub const ORCHESTRATOR_FINAL_ANSWER_PROMPT: &str = r#"
We are working on the following task:
{task}

We have completed the task.

The above messages contain the conversation that took place to complete the task.

Based on the information gathered, provide the final answer to the original request.
The answer should be phrased as if you were speaking to the user.
"#;

// endregion

/// Render a participant roster as `- name: description` lines. Rust analogue of
/// `_team_block`.
fn team_block(participants: &[(String, String)]) -> String {
    participants
        .iter()
        .map(|(name, desc)| format!("- {name}: {desc}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Find the most recent assistant message. Rust analogue of `_first_assistant`.
fn first_assistant(messages: &[ChatMessage]) -> Option<ChatMessage> {
    messages
        .iter()
        .rev()
        .find(|m| m.role == Role::assistant())
        .cloned()
}

/// Extract the first balanced JSON object from model output. Rust analogue of
/// `_extract_json` (fenced blocks are handled implicitly by scanning for the
/// first `{...}`).
fn extract_json(text: &str) -> Result<Value> {
    let start = text
        .find('{')
        .ok_or_else(|| Error::Workflow("no JSON object found in model output".into()))?;
    let mut depth = 0usize;
    let mut end = None;
    for (i, ch) in text[start..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(start + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end.ok_or_else(|| Error::Workflow("unbalanced JSON braces".into()))?;
    let candidate = &text[start..end];
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(candidate) {
        return Ok(v);
    }
    // Tolerate Python-style literals.
    let fixed = candidate
        .replace("True", "true")
        .replace("False", "false")
        .replace("None", "null");
    match serde_json::from_str::<Value>(&fixed) {
        Ok(v @ Value::Object(_)) => Ok(v),
        _ => Err(Error::Workflow(
            "unable to parse JSON from model output".into(),
        )),
    }
}

/// A single progress-ledger field: a reason plus a boolean or string answer.
/// Rust analogue of `_MagenticProgressLedgerItem`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticProgressLedgerItem {
    /// The model's reasoning for this field.
    #[serde(default)]
    pub reason: String,
    /// The answer, either a boolean (for the yes/no fields) or a string.
    #[serde(default)]
    pub answer: Value,
}

impl MagenticProgressLedgerItem {
    /// Interpret the answer as a boolean (defaulting to `false`).
    pub fn answer_bool(&self) -> bool {
        match &self.answer {
            Value::Bool(b) => *b,
            Value::String(s) => matches!(s.trim().to_lowercase().as_str(), "true" | "yes"),
            _ => false,
        }
    }

    /// Interpret the answer as a string (defaulting to empty).
    pub fn answer_str(&self) -> String {
        match &self.answer {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }
}

/// The structured progress ledger produced each round. Rust analogue of
/// `_MagenticProgressLedger`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticProgressLedger {
    /// Whether the original request is fully satisfied.
    pub is_request_satisfied: MagenticProgressLedgerItem,
    /// Whether the team is stuck repeating itself.
    pub is_in_loop: MagenticProgressLedgerItem,
    /// Whether forward progress is being made.
    pub is_progress_being_made: MagenticProgressLedgerItem,
    /// Who should speak next.
    pub next_speaker: MagenticProgressLedgerItem,
    /// The instruction or question for the next speaker.
    pub instruction_or_question: MagenticProgressLedgerItem,
}

/// Facts + plan captured for a run. Rust analogue of `_MagenticTaskLedger`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticTaskLedger {
    /// The fact sheet message.
    pub facts: ChatMessage,
    /// The plan message.
    pub plan: ChatMessage,
}

/// Mutable state threaded through the Magentic manager and orchestrator. Rust
/// analogue of `MagenticContext`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticContext {
    /// The task message.
    pub task: ChatMessage,
    /// The running conversation history.
    pub chat_history: Vec<ChatMessage>,
    /// Participants as `(name, description)` pairs.
    pub participant_descriptions: Vec<(String, String)>,
    /// The number of coordination rounds taken.
    pub round_count: usize,
    /// The consecutive stall count.
    pub stall_count: usize,
    /// The number of replans performed.
    pub reset_count: usize,
}

impl MagenticContext {
    /// Create a fresh context for `task` and `participants`.
    pub fn new(task: ChatMessage, participant_descriptions: Vec<(String, String)>) -> Self {
        Self {
            task,
            chat_history: Vec::new(),
            participant_descriptions,
            round_count: 0,
            stall_count: 0,
            reset_count: 0,
        }
    }

    /// Clear the chat history and reset the stall count, incrementing the reset
    /// count. Preserves the task, round count, and participants. Rust analogue
    /// of `MagenticContext.reset`.
    pub fn reset(&mut self) {
        self.chat_history.clear();
        self.stall_count = 0;
        self.reset_count += 1;
    }
}

/// The Magentic manager interface: planning, replanning, progress evaluation,
/// and final-answer synthesis. Rust analogue of `MagenticManagerBase`.
#[async_trait]
pub trait MagenticManager: Send + Sync {
    /// Gather facts and produce the initial plan (returns the combined ledger).
    async fn plan(&self, context: &MagenticContext) -> Result<ChatMessage>;

    /// Update facts and plan after a stall (returns the combined ledger).
    async fn replan(&self, context: &MagenticContext) -> Result<ChatMessage>;

    /// Produce the structured progress ledger for the current round.
    async fn create_progress_ledger(
        &self,
        context: &MagenticContext,
    ) -> Result<MagenticProgressLedger>;

    /// Synthesize the final answer addressed to the user.
    async fn prepare_final_answer(&self, context: &MagenticContext) -> Result<ChatMessage>;

    /// The stall threshold that triggers a replan.
    fn max_stall_count(&self) -> usize {
        3
    }

    /// The maximum number of replans, if bounded.
    fn max_reset_count(&self) -> Option<usize> {
        None
    }

    /// The maximum number of rounds, if bounded.
    fn max_round_count(&self) -> Option<usize> {
        None
    }

    /// The manager's current decomposed task ledger (facts + plan), if it
    /// tracks one separately from the combined message [`Self::plan`] /
    /// [`Self::replan`] return.
    ///
    /// Used only by plan review, to surface separate `facts`/`plan` text in
    /// [`MagenticPlanReviewRequest`] and to re-render the combined ledger
    /// after a direct human edit without an LLM call. Default `None`;
    /// [`StandardMagenticManager`] overrides it. Rust analogue of Python's
    /// `getattr(manager, "task_ledger", None)` escape hatch in
    /// `_send_plan_review_request` / `_handle_plan_review_response`.
    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        None
    }
}

/// The standard LLM-driven manager. Rust analogue of `StandardMagenticManager`.
pub struct StandardMagenticManager {
    agent: Arc<dyn Agent>,
    task_ledger: Mutex<Option<MagenticTaskLedger>>,
    max_stall_count: usize,
    max_reset_count: Option<usize>,
    max_round_count: Option<usize>,
    progress_ledger_retry_count: usize,
}

impl StandardMagenticManager {
    /// Create a manager driven by `agent`.
    pub fn new(agent: Arc<dyn Agent>) -> Self {
        Self {
            agent,
            task_ledger: Mutex::new(None),
            max_stall_count: 3,
            max_reset_count: None,
            max_round_count: None,
            progress_ledger_retry_count: 3,
        }
    }

    /// Set the stall threshold (default 3).
    pub fn max_stall_count(mut self, n: usize) -> Self {
        self.max_stall_count = n;
        self
    }

    /// Set the maximum number of replans (default unbounded).
    pub fn max_reset_count(mut self, n: usize) -> Self {
        self.max_reset_count = Some(n);
        self
    }

    /// Set the maximum number of rounds (default unbounded).
    pub fn max_round_count(mut self, n: usize) -> Self {
        self.max_round_count = Some(n);
        self
    }

    /// Set the progress-ledger parse retry budget (default 3).
    pub fn progress_ledger_retry_count(mut self, n: usize) -> Self {
        self.progress_ledger_retry_count = n.max(1);
        self
    }

    /// The current task ledger, if planning has run.
    pub fn task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.task_ledger.lock().unwrap().clone()
    }

    /// Run the underlying agent and return the last message, tagged as the
    /// manager. Rust analogue of `_complete`.
    async fn complete(&self, messages: Vec<ChatMessage>) -> Result<ChatMessage> {
        let response = self.agent.run(messages, None).await?;
        if let Some(last) = response.messages.last() {
            Ok(ChatMessage {
                role: last.role.clone(),
                contents: vec![crate::types::Content::text(last.text())],
                author_name: last
                    .author_name
                    .clone()
                    .or_else(|| Some(MAGENTIC_MANAGER_NAME.to_string())),
                message_id: None,
                additional_properties: Default::default(),
            })
        } else {
            Ok(ensure_author(
                ChatMessage::assistant("No output produced."),
                MAGENTIC_MANAGER_NAME,
            ))
        }
    }
}

#[async_trait]
impl MagenticManager for StandardMagenticManager {
    async fn plan(&self, context: &MagenticContext) -> Result<ChatMessage> {
        let task_text = context.task.text();
        let team_text = team_block(&context.participant_descriptions);

        let facts_user =
            ChatMessage::user(ORCHESTRATOR_TASK_LEDGER_FACTS_PROMPT.replace("{task}", &task_text));
        let mut facts_msgs = context.chat_history.clone();
        facts_msgs.push(facts_user.clone());
        let facts_msg = self.complete(facts_msgs).await?;

        let plan_user =
            ChatMessage::user(ORCHESTRATOR_TASK_LEDGER_PLAN_PROMPT.replace("{team}", &team_text));
        let mut plan_msgs = context.chat_history.clone();
        plan_msgs.extend([facts_user, facts_msg.clone(), plan_user]);
        let plan_msg = self.complete(plan_msgs).await?;

        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: facts_msg.clone(),
            plan: plan_msg.clone(),
        });

        let combined = ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
            .replace("{task}", &task_text)
            .replace("{team}", &team_text)
            .replace("{facts}", &facts_msg.text())
            .replace("{plan}", &plan_msg.text());
        Ok(ensure_author(
            ChatMessage::assistant(combined),
            MAGENTIC_MANAGER_NAME,
        ))
    }

    async fn replan(&self, context: &MagenticContext) -> Result<ChatMessage> {
        let ledger = self
            .task_ledger
            .lock()
            .unwrap()
            .clone()
            .ok_or_else(|| Error::Workflow("replan() called before plan()".into()))?;
        let task_text = context.task.text();
        let team_text = team_block(&context.participant_descriptions);

        let facts_update_user = ChatMessage::user(
            ORCHESTRATOR_TASK_LEDGER_FACTS_UPDATE_PROMPT
                .replace("{task}", &task_text)
                .replace("{old_facts}", &ledger.facts.text()),
        );
        let mut facts_msgs = context.chat_history.clone();
        facts_msgs.push(facts_update_user.clone());
        let updated_facts = self.complete(facts_msgs).await?;

        let plan_update_user = ChatMessage::user(
            ORCHESTRATOR_TASK_LEDGER_PLAN_UPDATE_PROMPT.replace("{team}", &team_text),
        );
        let mut plan_msgs = context.chat_history.clone();
        plan_msgs.extend([facts_update_user, updated_facts.clone(), plan_update_user]);
        let updated_plan = self.complete(plan_msgs).await?;

        *self.task_ledger.lock().unwrap() = Some(MagenticTaskLedger {
            facts: updated_facts.clone(),
            plan: updated_plan.clone(),
        });

        let combined = ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
            .replace("{task}", &task_text)
            .replace("{team}", &team_text)
            .replace("{facts}", &updated_facts.text())
            .replace("{plan}", &updated_plan.text());
        Ok(ensure_author(
            ChatMessage::assistant(combined),
            MAGENTIC_MANAGER_NAME,
        ))
    }

    async fn create_progress_ledger(
        &self,
        context: &MagenticContext,
    ) -> Result<MagenticProgressLedger> {
        let names: Vec<String> = context
            .participant_descriptions
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        if names.is_empty() {
            return Err(Error::Workflow(
                "no participants configured; cannot determine next speaker".into(),
            ));
        }
        let names_csv = names.join(", ");
        let team_text = team_block(&context.participant_descriptions);
        let prompt = ORCHESTRATOR_PROGRESS_LEDGER_PROMPT
            .replace("{task}", &context.task.text())
            .replace("{team}", &team_text)
            .replace("{names}", &names_csv);
        let user = ChatMessage::user(prompt);

        let mut last_error = None;
        for _ in 0..self.progress_ledger_retry_count {
            let mut msgs = context.chat_history.clone();
            msgs.push(user.clone());
            let raw = self.complete(msgs).await?;
            match extract_json(&raw.text()).and_then(|v| {
                serde_json::from_value::<MagenticProgressLedger>(v)
                    .map_err(|e| Error::Workflow(format!("invalid progress ledger: {e}")))
            }) {
                Ok(ledger) => return Ok(ledger),
                Err(e) => last_error = Some(e),
            }
        }
        Err(Error::Workflow(format!(
            "progress ledger parse failed after {} attempt(s): {}",
            self.progress_ledger_retry_count,
            last_error
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown".into())
        )))
    }

    async fn prepare_final_answer(&self, context: &MagenticContext) -> Result<ChatMessage> {
        let prompt = ORCHESTRATOR_FINAL_ANSWER_PROMPT.replace("{task}", &context.task.text());
        let mut msgs = context.chat_history.clone();
        msgs.push(ChatMessage::user(prompt));
        let response = self.complete(msgs).await?;
        Ok(ensure_author(
            ChatMessage::assistant(response.text()),
            MAGENTIC_MANAGER_NAME,
        ))
    }

    fn max_stall_count(&self) -> usize {
        self.max_stall_count
    }
    fn max_reset_count(&self) -> Option<usize> {
        self.max_reset_count
    }
    fn max_round_count(&self) -> Option<usize> {
        self.max_round_count
    }
    fn current_task_ledger(&self) -> Option<MagenticTaskLedger> {
        self.task_ledger()
    }
}

/// The payload of the request-info event emitted when plan review is enabled
/// and the orchestrator needs a human decision on the task ledger. Rust
/// analogue of Python's `_MagenticHumanInterventionRequest` narrowed to its
/// `kind=PLAN_REVIEW` fields (`task_text`, `facts_text`, `plan_text`,
/// `round_index`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticPlanReviewRequest {
    /// The original task text.
    pub task: String,
    /// The current fact sheet text.
    pub facts: String,
    /// The current plan text.
    pub plan: String,
    /// How many *revise* rounds have happened so far (0 for the first request).
    pub round: u32,
}

/// A human's decision on a [`MagenticPlanReviewRequest`]. Rust analogue of
/// Python's `_MagenticHumanInterventionReply` narrowed to plan review, with
/// `MagenticHumanInterventionDecision.{APPROVE,REVISE}` as the two variants.
///
/// Both variants accept an optional `edited_plan` and/or `comments`, mirroring
/// Python exactly: either decision may carry a direct plan edit (applied
/// verbatim, no LLM call) or free-text feedback (fed to
/// [`MagenticManager::replan`]); `edited_plan` wins if both are set. See the
/// module-level docs for the full loop semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum MagenticPlanReviewDecision {
    /// Accept the plan and proceed into the coordination round loop.
    Approve {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        edited_plan: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        comments: Option<String>,
    },
    /// Ask for another round of planning; the orchestrator re-sends a
    /// [`MagenticPlanReviewRequest`] (unless the round limit is exceeded, in
    /// which case it force-proceeds instead).
    Revise {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        edited_plan: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        comments: Option<String>,
    },
}

impl MagenticPlanReviewDecision {
    /// Approve the plan as-is.
    pub fn approve() -> Self {
        Self::Approve {
            edited_plan: None,
            comments: None,
        }
    }

    /// Approve, replacing the plan text directly (no LLM call).
    pub fn approve_with_edited_plan(edited_plan: impl Into<String>) -> Self {
        Self::Approve {
            edited_plan: Some(edited_plan.into()),
            comments: None,
        }
    }

    /// Approve, first asking the manager to replan with this feedback.
    pub fn approve_with_comments(comments: impl Into<String>) -> Self {
        Self::Approve {
            edited_plan: None,
            comments: Some(comments.into()),
        }
    }

    /// Ask for a revision, replacing the plan text directly (no LLM call)
    /// and re-requesting review.
    pub fn revise_with_edited_plan(edited_plan: impl Into<String>) -> Self {
        Self::Revise {
            edited_plan: Some(edited_plan.into()),
            comments: None,
        }
    }

    /// Ask for a revision, having the manager replan with this feedback
    /// before re-requesting review.
    pub fn revise_with_comments(comments: impl Into<String>) -> Self {
        Self::Revise {
            edited_plan: None,
            comments: Some(comments.into()),
        }
    }
}

/// The payload of the request-info event emitted when stall intervention is
/// enabled and the round loop detects a stall. Rust analogue of Python's
/// `_MagenticHumanInterventionRequest` narrowed to its `kind=STALL` fields
/// (`stall_count`, `max_stall_count`, `task_text`, `facts_text`, `plan_text`,
/// `last_agent`, `stall_reason`), plus `round`/`resets_so_far` for context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MagenticStallInterventionRequest {
    /// The original task text.
    pub task: String,
    /// A human-readable description of why progress stalled (no forward
    /// progress and/or looping). Python's `stall_reason`.
    pub reason: String,
    /// The consecutive stall count that tripped the threshold.
    pub stall_count: usize,
    /// The configured stall threshold (`max_stall_count`).
    pub max_stall_count: usize,
    /// The coordination round the stall was detected on.
    pub round: usize,
    /// How many replans (resets) have already happened this run.
    pub resets_so_far: usize,
    /// The current fact sheet text (empty when the manager tracks no ledger).
    pub facts: String,
    /// The current plan text (empty when the manager tracks no ledger).
    pub plan: String,
    /// The agent the ledger last nominated to speak. Python's `last_agent`.
    pub last_agent: String,
}

/// A human's decision on a [`MagenticStallInterventionRequest`]. Rust analogue
/// of Python's `_MagenticHumanInterventionReply` narrowed to the stall subset
/// of `MagenticHumanInterventionDecision` (`CONTINUE` / `REPLAN` / `GUIDANCE`).
///
/// Python exposes three stall decisions; this port folds `REPLAN` and
/// `GUIDANCE` into [`Replan { guidance }`](Self::Replan) — a forced replan
/// that optionally carries the human's free-text guidance — and adds an
/// explicit [`Abort`](Self::Abort) (which Python's stall decisions lack). See
/// the module-level docs for the exact wiring.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum MagenticStallInterventionDecision {
    /// Proceed as-is: clear the stall counter and resume the round loop
    /// (Python's `CONTINUE`).
    Continue,
    /// Force a replan via the existing reset path (Python's `REPLAN`);
    /// `guidance`, if present, is appended to the history after the fresh task
    /// ledger as `"Human guidance to help with stall: …"` (Python's
    /// `GUIDANCE`).
    Replan {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        guidance: Option<String>,
    },
    /// Stop coordinating and synthesize the final answer from the current
    /// history.
    Abort,
}

impl MagenticStallInterventionDecision {
    /// Continue as-is (clear the stall counter and proceed).
    pub fn continue_as_is() -> Self {
        Self::Continue
    }

    /// Force a replan with no extra guidance.
    pub fn replan() -> Self {
        Self::Replan { guidance: None }
    }

    /// Force a replan, appending the given human guidance to the history.
    pub fn replan_with_guidance(guidance: impl Into<String>) -> Self {
        Self::Replan {
            guidance: Some(guidance.into()),
        }
    }

    /// Abort and produce the final answer from the current history.
    pub fn abort() -> Self {
        Self::Abort
    }
}

/// Orchestrator state persisted across a plan-review pause via
/// [`WorkflowContext::shared_state`] (the executor instance itself must stay
/// stateless between `execute()` calls — see `HandoffCoordinator`'s
/// save_state/load_state in `handoff.rs` for the same pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MagenticPlanReviewState {
    mctx: MagenticContext,
    combined_ledger: ChatMessage,
    facts_text: String,
    plan_text: String,
    round: u32,
}

/// The shared-state key [`MagenticPlanReviewState`] is stored under.
const PLAN_REVIEW_STATE_KEY: &str = "_magentic_plan_review_state";

/// The shared-state key the paused round-loop [`MagenticContext`] is stored
/// under while a stall-intervention request is outstanding.
const STALL_STATE_KEY: &str = "_magentic_stall_state";

/// The single executor that drives the Magentic loop.
struct MagenticOrchestrator {
    id: String,
    manager: Arc<dyn MagenticManager>,
    participants: Vec<(String, Arc<dyn Agent>)>,
    descriptions: Vec<(String, String)>,
    max_stall_count: usize,
    max_reset_count: Option<usize>,
    max_round_count: Option<usize>,
    require_plan_signoff: bool,
    max_plan_review_rounds: u32,
    enable_stall_intervention: bool,
}

impl MagenticOrchestrator {
    fn find(&self, name: &str) -> Option<&Arc<dyn Agent>> {
        self.participants
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, a)| a)
    }

    fn emit_orchestrator_message(&self, ctx: &WorkflowContext, message: &ChatMessage) {
        if let Ok(update) = serde_json::to_value(message) {
            ctx.add_event(WorkflowEvent::AgentRunUpdate {
                executor_id: self.id.clone(),
                update,
            });
        }
    }

    async fn reset_and_replan(
        &self,
        mctx: &mut MagenticContext,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        mctx.reset();
        let task_ledger = self.manager.replan(mctx).await?;
        mctx.chat_history.push(task_ledger.clone());
        self.emit_orchestrator_message(ctx, &task_ledger);
        Ok(())
    }

    async fn yield_messages(
        &self,
        ctx: &WorkflowContext,
        messages: Vec<ChatMessage>,
    ) -> Result<()> {
        let payload = serde_json::to_value(&messages)
            .map_err(|e| Error::Workflow(format!("serialize error: {e}")))?;
        ctx.yield_output(payload).await
    }

    /// Split a manager's current ledger into separate facts/plan text for a
    /// [`MagenticPlanReviewRequest`], falling back to the combined message's
    /// text as `plan` (facts left empty) for managers that don't track a
    /// decomposed ledger. Rust analogue of Python's
    /// `getattr(manager, "task_ledger", None)` access pattern.
    fn decompose_ledger(&self, combined: &ChatMessage) -> (String, String) {
        match self.manager.current_task_ledger() {
            Some(ledger) => (ledger.facts.text(), ledger.plan.text()),
            None => (String::new(), combined.text()),
        }
    }

    /// Re-render the combined ledger message from `state`'s current
    /// facts/plan text, for applying a human-edited plan without an LLM
    /// call. Uses the same template `StandardMagenticManager` renders with.
    fn render_edited_ledger(&self, state: &MagenticPlanReviewState) -> ChatMessage {
        let team_text = team_block(&state.mctx.participant_descriptions);
        let combined = ORCHESTRATOR_TASK_LEDGER_FULL_PROMPT
            .replace("{task}", &state.mctx.task.text())
            .replace("{team}", &team_text)
            .replace("{facts}", &state.facts_text)
            .replace("{plan}", &state.plan_text);
        ensure_author(ChatMessage::assistant(combined), MAGENTIC_MANAGER_NAME)
    }

    async fn save_review_state(&self, ctx: &WorkflowContext, state: &MagenticPlanReviewState) {
        if let Ok(value) = serde_json::to_value(state) {
            ctx.shared_state().set(PLAN_REVIEW_STATE_KEY, value).await;
        }
    }

    async fn load_review_state(&self, ctx: &WorkflowContext) -> Result<MagenticPlanReviewState> {
        ctx.shared_state()
            .get(PLAN_REVIEW_STATE_KEY)
            .await
            .and_then(|v| serde_json::from_value(v).ok())
            .ok_or_else(|| {
                Error::Workflow(
                    "magentic plan-review response received with no pending review state".into(),
                )
            })
    }

    /// Emit a [`MagenticPlanReviewRequest`] built from `state` and pause.
    async fn send_plan_review_request(
        &self,
        ctx: &WorkflowContext,
        state: &MagenticPlanReviewState,
    ) -> Result<()> {
        let request = MagenticPlanReviewRequest {
            task: state.mctx.task.text(),
            facts: state.facts_text.clone(),
            plan: state.plan_text.clone(),
            round: state.round,
        };
        ctx.request_info(serialize(&request)?).await
    }

    /// Handle a [`MagenticPlanReviewDecision`] delivered back as a
    /// [`RequestResponse`]. Rust analogue of `_handle_plan_review_response`
    /// — see the module-level docs for the full semantics.
    async fn handle_plan_review_response(
        &self,
        resp: RequestResponse,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        let decision: MagenticPlanReviewDecision = serde_json::from_value(resp.data)
            .map_err(|e| Error::Workflow(format!("invalid plan-review decision: {e}")))?;
        let mut state = self.load_review_state(ctx).await?;

        match decision {
            MagenticPlanReviewDecision::Approve {
                edited_plan,
                comments,
            } => {
                if let Some(edited) = edited_plan {
                    state.plan_text = edited;
                    state.combined_ledger = self.render_edited_ledger(&state);
                } else if let Some(comments) = comments {
                    state.mctx.chat_history.push(ChatMessage::user(format!(
                        "Human plan feedback: {comments}"
                    )));
                    state.combined_ledger = self.manager.replan(&state.mctx).await?;
                    let (facts_text, plan_text) = self.decompose_ledger(&state.combined_ledger);
                    state.facts_text = facts_text;
                    state.plan_text = plan_text;
                }
                state.mctx.chat_history.push(state.combined_ledger.clone());
                self.emit_orchestrator_message(ctx, &state.combined_ledger);
                ctx.shared_state().delete(PLAN_REVIEW_STATE_KEY).await;
                self.run_round_loop(state.mctx, ctx).await
            }
            MagenticPlanReviewDecision::Revise {
                edited_plan,
                comments,
            } => {
                state.round += 1;
                if state.round > self.max_plan_review_rounds {
                    // Mirrors Python: the over-the-limit response's edits are
                    // discarded and whatever plan currently stands is used.
                    let notice = ensure_author(
                        ChatMessage::assistant(
                            "Plan review closed after max rounds. Proceeding with the current \
                             plan and will no longer prompt for plan approval."
                                .to_string(),
                        ),
                        MAGENTIC_MANAGER_NAME,
                    );
                    state.mctx.chat_history.push(notice.clone());
                    self.emit_orchestrator_message(ctx, &notice);
                    state.mctx.chat_history.push(state.combined_ledger.clone());
                    self.emit_orchestrator_message(ctx, &state.combined_ledger);
                    ctx.shared_state().delete(PLAN_REVIEW_STATE_KEY).await;
                    return self.run_round_loop(state.mctx, ctx).await;
                }

                if let Some(edited) = edited_plan {
                    state.plan_text = edited;
                    state.combined_ledger = self.render_edited_ledger(&state);
                } else {
                    if let Some(comments) = comments {
                        state.mctx.chat_history.push(ChatMessage::user(format!(
                            "Human plan feedback: {comments}"
                        )));
                    }
                    state.combined_ledger = self.manager.replan(&state.mctx).await?;
                    let (facts_text, plan_text) = self.decompose_ledger(&state.combined_ledger);
                    state.facts_text = facts_text;
                    state.plan_text = plan_text;
                }
                self.save_review_state(ctx, &state).await;
                self.send_plan_review_request(ctx, &state).await
            }
        }
    }

    /// Whether a stall-intervention request is currently outstanding (its
    /// paused round-loop state is persisted in shared state).
    async fn has_stall_state(&self, ctx: &WorkflowContext) -> bool {
        ctx.shared_state().has(STALL_STATE_KEY).await
    }

    async fn save_stall_state(&self, ctx: &WorkflowContext, mctx: &MagenticContext) {
        if let Ok(value) = serde_json::to_value(mctx) {
            ctx.shared_state().set(STALL_STATE_KEY, value).await;
        }
    }

    async fn load_stall_state(&self, ctx: &WorkflowContext) -> Result<MagenticContext> {
        ctx.shared_state()
            .get(STALL_STATE_KEY)
            .await
            .and_then(|v| serde_json::from_value(v).ok())
            .ok_or_else(|| {
                Error::Workflow(
                    "magentic stall-intervention response received with no pending stall state"
                        .into(),
                )
            })
    }

    /// Persist the round-loop state and emit a [`MagenticStallInterventionRequest`],
    /// suspending the run. Rust analogue of the `_enable_stall_intervention`
    /// branch in Python's `_run_inner_loop_exclusive`.
    async fn pause_for_stall(
        &self,
        mctx: MagenticContext,
        ledger: &MagenticProgressLedger,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        // Python reads facts/plan straight off the manager's task ledger,
        // leaving them empty when the manager tracks none (unlike plan review,
        // which falls back to the combined ledger text).
        let (facts, plan) = match self.manager.current_task_ledger() {
            Some(ledger) => (ledger.facts.text(), ledger.plan.text()),
            None => (String::new(), String::new()),
        };
        let request = MagenticStallInterventionRequest {
            task: mctx.task.text(),
            reason: stall_reason(ledger),
            stall_count: mctx.stall_count,
            max_stall_count: self.max_stall_count,
            round: mctx.round_count,
            resets_so_far: mctx.reset_count,
            facts,
            plan,
            last_agent: ledger.next_speaker.answer_str(),
        };
        self.save_stall_state(ctx, &mctx).await;
        ctx.request_info(serialize(&request)?).await
    }

    /// Handle a [`MagenticStallInterventionDecision`] delivered back as a
    /// [`RequestResponse`]. Rust analogue of
    /// `_handle_stall_intervention_response` — see the module-level docs.
    async fn handle_stall_intervention_response(
        &self,
        resp: RequestResponse,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        let decision: MagenticStallInterventionDecision = serde_json::from_value(resp.data)
            .map_err(|e| Error::Workflow(format!("invalid stall-intervention decision: {e}")))?;
        let mut mctx = self.load_stall_state(ctx).await?;
        ctx.shared_state().delete(STALL_STATE_KEY).await;

        match decision {
            MagenticStallInterventionDecision::Continue => {
                // Clear the stall counter and resume coordinating as-is.
                mctx.stall_count = 0;
                self.run_round_loop(mctx, ctx).await
            }
            MagenticStallInterventionDecision::Replan { guidance } => {
                // The existing reset path clears history + stall count and
                // replans; guidance is appended afterward so it survives the
                // reset and reaches the manager on the next round.
                self.reset_and_replan(&mut mctx, ctx).await?;
                if let Some(guidance) = guidance {
                    let msg =
                        ChatMessage::user(format!("Human guidance to help with stall: {guidance}"));
                    mctx.chat_history.push(msg.clone());
                    self.emit_orchestrator_message(ctx, &msg);
                }
                self.run_round_loop(mctx, ctx).await
            }
            MagenticStallInterventionDecision::Abort => {
                let final_answer = self.manager.prepare_final_answer(&mctx).await?;
                self.yield_messages(ctx, vec![final_answer]).await
            }
        }
    }

    /// Run the coordination round loop (Python's "inner loop") starting from
    /// `mctx`. Entry point both for a fresh run (plan review disabled, or
    /// already approved) and for resuming after a plan-review approval.
    async fn run_round_loop(&self, mut mctx: MagenticContext, ctx: &WorkflowContext) -> Result<()> {
        loop {
            // Limit checks (before incrementing the round, mirroring Python).
            let hit_round = self.max_round_count.is_some_and(|m| mctx.round_count >= m);
            let hit_reset = self.max_reset_count.is_some_and(|m| mctx.reset_count >= m);
            if hit_round || hit_reset {
                let limit = if hit_round { "round" } else { "reset" };
                let partial = first_assistant(&mctx.chat_history).unwrap_or_else(|| {
                    ensure_author(
                        ChatMessage::assistant(format!(
                            "Stopped due to {limit} limit. No partial result available."
                        )),
                        MAGENTIC_MANAGER_NAME,
                    )
                });
                return self.yield_messages(ctx, vec![partial]).await;
            }

            mctx.round_count += 1;

            let ledger = match self.manager.create_progress_ledger(&mctx).await {
                Ok(l) => l,
                Err(_) => {
                    self.reset_and_replan(&mut mctx, ctx).await?;
                    continue;
                }
            };

            if ledger.is_request_satisfied.answer_bool() {
                let final_answer = self.manager.prepare_final_answer(&mctx).await?;
                return self.yield_messages(ctx, vec![final_answer]).await;
            }

            if !ledger.is_progress_being_made.answer_bool() || ledger.is_in_loop.answer_bool() {
                mctx.stall_count += 1;
            } else {
                mctx.stall_count = mctx.stall_count.saturating_sub(1);
            }

            if mctx.stall_count > self.max_stall_count {
                if self.enable_stall_intervention {
                    // Pause for a human decision instead of auto-replanning.
                    return self.pause_for_stall(mctx, &ledger, ctx).await;
                }
                self.reset_and_replan(&mut mctx, ctx).await?;
                continue;
            }

            let next = ledger.next_speaker.answer_str();
            if self.find(&next).is_none() {
                let final_answer = self.manager.prepare_final_answer(&mctx).await?;
                return self.yield_messages(ctx, vec![final_answer]).await;
            }

            let instruction = ledger.instruction_or_question.answer_str();
            let instr_msg =
                ensure_author(ChatMessage::assistant(instruction), MAGENTIC_MANAGER_NAME);
            mctx.chat_history.push(instr_msg.clone());
            self.emit_orchestrator_message(ctx, &instr_msg);

            let agent = self.find(&next).expect("checked above");
            let response =
                run_agent_and_emit(agent, mctx.chat_history.clone(), &self.id, &next, ctx).await?;
            if let Some(body) = last_message(&response) {
                if body.role != Role::user() {
                    let author = body.author_name.clone().unwrap_or_else(|| next.clone());
                    mctx.chat_history
                        .push(ChatMessage::user(format!("Transferred to {author}")));
                }
                mctx.chat_history.push(body);
            }
        }
    }
}

#[async_trait]
impl Executor for MagenticOrchestrator {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // A response to an outstanding request resumes the paused orchestrator.
        // A pending stall intervention (persisted round-loop state) takes
        // precedence; otherwise it is a plan-review response. The two never
        // overlap: plan-review state is cleared before the round loop starts,
        // and stall pauses only happen inside the round loop.
        if let Some(resp) = RequestResponse::from_message(&message) {
            if self.has_stall_state(&ctx).await {
                return self.handle_stall_intervention_response(resp, &ctx).await;
            }
            return self.handle_plan_review_response(resp, &ctx).await;
        }

        let input = parse_conversation(&message)?;
        let task = input
            .last()
            .cloned()
            .ok_or_else(|| Error::Workflow("magentic requires a task message".into()))?;

        let mut mctx = MagenticContext::new(task.clone(), self.descriptions.clone());
        mctx.chat_history = input;
        self.emit_orchestrator_message(&ctx, &task);

        // Initial planning.
        let task_ledger = self.manager.plan(&mctx).await?;

        if self.require_plan_signoff {
            // Withhold the ledger from chat_history until approved, mirroring
            // Python's `handle_start_message` (which only appends it after
            // `_send_plan_review_request` is skipped or resolved).
            let (facts_text, plan_text) = self.decompose_ledger(&task_ledger);
            let state = MagenticPlanReviewState {
                mctx,
                combined_ledger: task_ledger,
                facts_text,
                plan_text,
                round: 0,
            };
            self.save_review_state(&ctx, &state).await;
            return self.send_plan_review_request(&ctx, &state).await;
        }

        mctx.chat_history.push(task_ledger.clone());
        self.emit_orchestrator_message(&ctx, &task_ledger);

        self.run_round_loop(mctx, &ctx).await
    }
}

fn last_message(response: &AgentRunResponse) -> Option<ChatMessage> {
    response.messages.last().cloned()
}

/// Build the human-readable stall reason from a progress ledger, mirroring
/// Python's `_run_inner_loop_exclusive`: "No progress being made" when progress
/// stalled and/or "Agents appear to be in a loop" when looping, joined by
/// "; " when both hold.
fn stall_reason(ledger: &MagenticProgressLedger) -> String {
    let mut reason = if ledger.is_progress_being_made.answer_bool() {
        String::new()
    } else {
        "No progress being made".to_string()
    };
    if ledger.is_in_loop.answer_bool() {
        let loop_msg = "Agents appear to be in a loop";
        reason = if reason.is_empty() {
            loop_msg.to_string()
        } else {
            format!("{reason}; {loop_msg}")
        };
    }
    reason
}

fn serialize<T: Serialize>(value: &T) -> Result<Value> {
    serde_json::to_value(value).map_err(|e| Error::Workflow(format!("serialize error: {e}")))
}

/// Builder for a Magentic workflow. Rust analogue of `MagenticBuilder`.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::prelude::*;
/// # use agent_framework_core::workflow::{MagenticBuilder, StandardMagenticManager};
/// # fn demo(coder: Arc<dyn Agent>, researcher: Arc<dyn Agent>, manager_agent: Arc<dyn Agent>) -> Result<()> {
/// let manager = StandardMagenticManager::new(manager_agent);
/// let workflow = MagenticBuilder::new()
///     .participant("coder", coder)
///     .participant("researcher", researcher)
///     .standard_manager(manager)
///     .max_round_count(10)
///     .build()?;
/// # let _ = workflow;
/// # Ok(())
/// # }
/// ```
#[derive(Default)]
pub struct MagenticBuilder {
    participants: Vec<(String, String, Arc<dyn Agent>)>,
    manager: Option<Arc<dyn MagenticManager>>,
    max_round_count: Option<usize>,
    max_stall_count: Option<usize>,
    max_reset_count: Option<usize>,
    name: Option<String>,
    enable_plan_review: bool,
    max_plan_review_rounds: Option<u32>,
    enable_stall_intervention: bool,
}

/// Default cap on plan-review *revise* rounds before the orchestrator
/// force-proceeds with the current plan (matches Python's
/// `MagenticOrchestratorExecutor(max_plan_review_rounds=10)` default — not
/// itself exposed on Python's `MagenticBuilder`, but the underlying value it
/// always constructs the executor with).
const DEFAULT_MAX_PLAN_REVIEW_ROUNDS: u32 = 10;

impl MagenticBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a participant (its description defaults to its name).
    pub fn participant(mut self, name: impl Into<String>, agent: Arc<dyn Agent>) -> Self {
        let name = name.into();
        self.participants.push((name.clone(), name, agent));
        self
    }

    /// Register a participant with an explicit description (shown to the manager).
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

    /// Use a custom [`MagenticManager`].
    pub fn manager(mut self, manager: Arc<dyn MagenticManager>) -> Self {
        self.manager = Some(manager);
        self
    }

    /// Use a [`StandardMagenticManager`] (convenience wrapper around
    /// [`MagenticBuilder::manager`]).
    pub fn standard_manager(mut self, manager: StandardMagenticManager) -> Self {
        self.manager = Some(Arc::new(manager));
        self
    }

    /// Override the maximum number of rounds.
    pub fn max_round_count(mut self, n: usize) -> Self {
        self.max_round_count = Some(n);
        self
    }

    /// Override the stall threshold.
    pub fn max_stall_count(mut self, n: usize) -> Self {
        self.max_stall_count = Some(n);
        self
    }

    /// Override the maximum number of replans.
    pub fn max_reset_count(mut self, n: usize) -> Self {
        self.max_reset_count = Some(n);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Require a human to review and approve (or send back for revision) the
    /// initial plan before the coordination round loop starts. See the
    /// module-level docs for the full request/response loop semantics. Rust
    /// analogue of Python's `MagenticBuilder.with_plan_review(enable=True)`.
    pub fn with_plan_review(mut self) -> Self {
        self.enable_plan_review = true;
        self
    }

    /// Override the maximum number of plan-review *revise* rounds before the
    /// orchestrator force-proceeds with whatever plan is current (default 10,
    /// matching Python). Only meaningful with [`Self::with_plan_review`].
    pub fn max_plan_review_rounds(mut self, n: u32) -> Self {
        self.max_plan_review_rounds = Some(n);
        self
    }

    /// Pause for a human decision when the round loop detects a stall, instead
    /// of silently auto-replanning. When `stall_count` exceeds
    /// `max_stall_count` the orchestrator emits a
    /// [`MagenticStallInterventionRequest`] via
    /// [`WorkflowContext::request_info`] and suspends until a
    /// [`MagenticStallInterventionDecision`] is supplied. See the module-level
    /// docs for the full loop semantics. Rust analogue of Python's
    /// `MagenticBuilder.with_human_input_on_stall(enable=True)`.
    pub fn with_stall_intervention(mut self) -> Self {
        self.enable_stall_intervention = true;
        self
    }

    /// Validate and build the Magentic workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "magentic workflow needs at least one participant".into(),
            ));
        }
        let manager = self
            .manager
            .ok_or_else(|| Error::Workflow("magentic workflow requires a manager".into()))?;

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

        let orchestrator = MagenticOrchestrator {
            id: "magentic_orchestrator".to_string(),
            max_stall_count: self
                .max_stall_count
                .unwrap_or_else(|| manager.max_stall_count()),
            max_reset_count: self.max_reset_count.or_else(|| manager.max_reset_count()),
            max_round_count: self.max_round_count.or_else(|| manager.max_round_count()),
            manager,
            participants,
            descriptions,
            require_plan_signoff: self.enable_plan_review,
            max_plan_review_rounds: self
                .max_plan_review_rounds
                .unwrap_or(DEFAULT_MAX_PLAN_REVIEW_ROUNDS),
            enable_stall_intervention: self.enable_stall_intervention,
        };

        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(orchestrator))
            .set_start("magentic_orchestrator");
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}
