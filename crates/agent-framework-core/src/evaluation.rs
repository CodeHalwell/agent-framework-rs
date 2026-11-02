//! Provider-agnostic evaluation: run an agent or workflow over test queries
//! and score the results with local checks or evaluation services.
//!
//! Port of upstream `agent_framework._evaluation` (`ExperimentalFeature.EVALS`),
//! behind the `experimental-evals` feature.
//!
//! - [`EvalItem`] is one query/response interaction, derived from a
//!   conversation by a [`ConversationSplit`] or a custom splitter.
//! - An [`Evaluator`] scores a batch of items into [`EvalResults`];
//!   [`LocalEvaluator`] runs [`EvalCheck`]s in process, for CI smoke tests.
//! - Built-in checks: [`keyword_check`], [`tool_called_check`],
//!   [`tool_calls_present`], [`tool_call_args_match`]; [`evaluator`] wraps a
//!   plain function whose result is a `bool`, a score, a JSON object or a
//!   [`CheckResult`].
//! - [`evaluate_agent`] runs an agent over queries (or takes existing
//!   responses) and evaluates them; [`evaluate_workflow`] evaluates each
//!   agent in a workflow run and the overall output.
//! - [`EvalResults::raise_for_status`] and the `assert_*` methods turn
//!   results into a CI gate.
//!
//! ```
//! use agent_framework_core::evaluation::*;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() {
//! let item = EvalItem::new(vec![
//!     Message::user("What's the weather?"),
//!     Message::assistant("Sunny, with a high of 20 degrees."),
//! ]);
//! let local = LocalEvaluator::new(vec![
//!     keyword_check(["sunny"], false),
//!     evaluator("short", |item: &EvalItem| item.response().len() < 200),
//! ]);
//! let results = local.evaluate(&[item], "smoke").await.unwrap();
//! results.raise_for_status(None).unwrap();
//! # }
//! ```
//!
//! # Divergences
//!
//! - **Tool definitions are explicit.** Upstream reads them from response
//!   metadata or the agent's options; a Rust agent is a trait object, so
//!   pass them with [`EvaluateAgent::tools`] (an [`Agent`](crate::agent::Agent)
//!   lists its own with [`Agent::tools`](crate::agent::Agent::tools)).
//! - **Function evaluators** take the whole [`EvalItem`] instead of
//!   parameters matched by name.
//! - **Workflow evaluation** reads each agent's input and response from the
//!   [`WorkflowEvent::AgentRun`] events the built-in agent executors emit,
//!   and the overall query from the first of them.
//! - The deprecated `AgentEvalConverter` is not ported.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::{join_all, BoxFuture};
use serde_json::{Map, Value};

use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::{AgentResponse, Content, FunctionArguments, Message, Role};
use crate::workflow::{Workflow, WorkflowEvent, WorkflowRun};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Evaluation results contained failures. Returned by
/// [`EvalResults::raise_for_status`] and the `assert_*` methods.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalNotPassedError(pub String);

impl fmt::Display for EvalNotPassedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EvalNotPassedError {}

impl From<EvalNotPassedError> for Error {
    fn from(e: EvalNotPassedError) -> Self {
        Error::Other(e.0)
    }
}

// ---------------------------------------------------------------------------
// Splitting and items
// ---------------------------------------------------------------------------

/// Splits a conversation into `(query messages, response messages)`.
pub type ConversationSplitter =
    Arc<dyn Fn(&[Message]) -> (Vec<Message>, Vec<Message>) + Send + Sync>;

/// The built-in split strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConversationSplit {
    /// Split at the last user message: everything up to and including it is
    /// the query, the rest the response. Evaluates the latest answer.
    #[default]
    LastTurn,
    /// Split after the first user message: the whole remainder is the
    /// response. Evaluates the full trajectory.
    Full,
}

impl ConversationSplit {
    /// Apply the strategy.
    pub fn split(self, conversation: &[Message]) -> (Vec<Message>, Vec<Message>) {
        let mut users = conversation
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role.as_str() == Role::USER)
            .map(|(i, _)| i);
        let at = match self {
            Self::LastTurn => users.next_back(),
            Self::Full => users.next(),
        };
        match at {
            Some(i) => (conversation[..=i].to_vec(), conversation[i + 1..].to_vec()),
            None => (Vec::new(), conversation.to_vec()),
        }
    }

    /// The strategy as a [`ConversationSplitter`].
    pub fn splitter(self) -> ConversationSplitter {
        Arc::new(move |c| self.split(c))
    }
}

/// A tool call an agent is expected to make. The evaluator decides how to
/// match it.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpectedToolCall {
    /// The tool name.
    pub name: String,
    /// The expected arguments; `None` means "do not check".
    pub arguments: Option<Map<String, Value>>,
}

impl ExpectedToolCall {
    /// Expect a call to `name`, with any arguments.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: None,
        }
    }

    /// Expect these arguments (a subset match for [`tool_call_args_match`]).
    /// A non-object value is ignored.
    pub fn with_arguments(mut self, arguments: Value) -> Self {
        if let Value::Object(map) = arguments {
            self.arguments = Some(map);
        }
        self
    }
}

/// One interaction to evaluate. The conversation is the source of truth;
/// [`query`](Self::query) and [`response`](Self::response) are derived from
/// it by the split strategy.
#[derive(Clone, Default)]
pub struct EvalItem {
    /// The full conversation.
    pub conversation: Vec<Message>,
    /// Tool definitions, for evaluators that check tool use.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Grounding context.
    pub context: Option<String>,
    /// The expected output, for ground-truth comparison.
    pub expected_output: Option<String>,
    /// The expected tool calls.
    pub expected_tool_calls: Option<Vec<ExpectedToolCall>>,
    /// How to split the conversation; [`ConversationSplit::LastTurn`] when
    /// unset.
    pub split_strategy: Option<ConversationSplitter>,
}

impl fmt::Debug for EvalItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EvalItem")
            .field("conversation", &self.conversation)
            .field("context", &self.context)
            .field("expected_output", &self.expected_output)
            .field("expected_tool_calls", &self.expected_tool_calls)
            .finish_non_exhaustive()
    }
}

impl EvalItem {
    /// An item over `conversation`.
    pub fn new(conversation: Vec<Message>) -> Self {
        Self {
            conversation,
            ..Default::default()
        }
    }

    /// The conversation split by `split`, else the item's strategy, else
    /// [`ConversationSplit::LastTurn`].
    pub fn split_messages(
        &self,
        split: Option<&ConversationSplitter>,
    ) -> (Vec<Message>, Vec<Message>) {
        match split.or(self.split_strategy.as_ref()) {
            Some(splitter) => splitter(&self.conversation),
            None => ConversationSplit::LastTurn.split(&self.conversation),
        }
    }

    /// The user text on the query side of the split.
    pub fn query(&self) -> String {
        joined_text(&self.split_messages(None).0, Role::USER)
    }

    /// The assistant text on the response side of the split.
    pub fn response(&self) -> String {
        joined_text(&self.split_messages(None).1, Role::ASSISTANT)
    }

    /// One item per user turn, each with the conversation up to the next
    /// user message, so every response is evaluated with its full context.
    pub fn per_turn_items(
        conversation: &[Message],
        tools: Option<Vec<ToolDefinition>>,
        context: Option<String>,
    ) -> Vec<EvalItem> {
        let users: Vec<usize> = conversation
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role.as_str() == Role::USER)
            .map(|(i, _)| i)
            .collect();
        (0..users.len())
            .map(|turn| {
                let end = users.get(turn + 1).copied().unwrap_or(conversation.len());
                EvalItem {
                    conversation: conversation[..end].to_vec(),
                    tools: tools.clone(),
                    context: context.clone(),
                    ..Default::default()
                }
            })
            .collect()
    }

    /// `(name, arguments)` for every function call in the conversation.
    /// Arguments that are not a JSON object are `None`.
    pub fn tool_calls(&self) -> Vec<(String, Option<Map<String, Value>>)> {
        let mut calls = Vec::new();
        for message in &self.conversation {
            for content in &message.contents {
                let Content::FunctionCall(call) = content else {
                    continue;
                };
                if call.name.is_empty() {
                    continue;
                }
                let args = match &call.arguments {
                    None => Some(Map::new()),
                    Some(FunctionArguments::Object(map)) => {
                        Some(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                    }
                    Some(FunctionArguments::Raw(raw)) if raw.trim().is_empty() => Some(Map::new()),
                    Some(FunctionArguments::Raw(raw)) => match serde_json::from_str(raw) {
                        Ok(Value::Object(map)) => Some(map),
                        _ => None,
                    },
                };
                calls.push((call.name.clone(), args));
            }
        }
        calls
    }
}

fn joined_text(messages: &[Message], role: &str) -> String {
    messages
        .iter()
        .filter(|m| m.role.as_str() == role)
        .map(Message::text)
        .filter(|t| !t.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// One dimension's score from a rubric evaluator.
#[derive(Debug, Clone, PartialEq)]
pub struct RubricScore {
    /// The dimension id.
    pub id: String,
    /// The score, or `None` when the dimension did not apply.
    pub score: Option<i64>,
    /// Whether the dimension applied to the item.
    pub applicable: bool,
    /// The dimension's weight.
    pub weight: i64,
    /// The evaluator's rationale.
    pub reason: String,
}

/// One evaluator's result on one item.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalScoreResult {
    /// The evaluator name.
    pub name: String,
    /// The numeric score.
    pub score: f64,
    /// Whether the item passed this evaluator.
    pub passed: Option<bool>,
    /// Raw evaluator output (rationale, metadata).
    pub sample: Option<Value>,
    /// Per-dimension scores, for rubric evaluators.
    pub dimensions: Option<Vec<RubricScore>>,
}

/// One item's result.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalItemResult {
    /// The provider's item id.
    pub item_id: String,
    /// `pass`, `fail` or `error`.
    pub status: String,
    /// Per-evaluator scores.
    pub scores: Vec<EvalScoreResult>,
    /// The error category of an errored item.
    pub error_code: Option<String>,
    /// The error detail.
    pub error_message: Option<String>,
    /// A Responses API response id, if any.
    pub response_id: Option<String>,
    /// The evaluated query.
    pub input_text: Option<String>,
    /// The evaluated response.
    pub output_text: Option<String>,
    /// Token counts.
    pub token_usage: Option<BTreeMap<String, i64>>,
    /// Provider-specific data.
    pub metadata: Option<Value>,
}

impl EvalItemResult {
    /// The item errored (an infrastructure failure, not a quality one).
    pub fn is_error(&self) -> bool {
        matches!(self.status.as_str(), "error" | "errored")
    }
    /// The item passed every evaluator.
    pub fn is_passed(&self) -> bool {
        self.status == "pass"
    }
    /// The item failed at least one evaluator.
    pub fn is_failed(&self) -> bool {
        self.status == "fail"
    }
}

/// One provider's results for an evaluation run.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EvalResults {
    /// The provider that produced them.
    pub provider: String,
    /// The provider's evaluation id.
    pub eval_id: String,
    /// The provider's run id.
    pub run_id: String,
    /// `completed`, `failed`, `canceled`, `timeout` or `partial`.
    pub status: String,
    /// `passed` / `failed` / `errored` counts.
    pub result_counts: Option<BTreeMap<String, i64>>,
    /// A link to the results in the provider's portal.
    pub report_url: Option<String>,
    /// Error detail when the run failed.
    pub error: Option<String>,
    /// Counts per evaluator.
    pub per_evaluator: BTreeMap<String, BTreeMap<String, i64>>,
    /// Per-item results.
    pub items: Vec<EvalItemResult>,
    /// Per-agent results of a workflow evaluation, by executor id.
    pub sub_results: BTreeMap<String, EvalResults>,
}

impl EvalResults {
    /// Empty, completed results for `provider`.
    pub fn new(provider: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            status: "completed".into(),
            ..Default::default()
        }
    }

    fn count(&self, key: &str) -> i64 {
        self.result_counts
            .as_ref()
            .and_then(|c| c.get(key))
            .copied()
            .unwrap_or(0)
    }

    /// Passing results.
    pub fn passed(&self) -> i64 {
        self.count("passed")
    }
    /// Failing results.
    pub fn failed(&self) -> i64 {
        self.count("failed")
    }
    /// Passing plus failing results.
    pub fn total(&self) -> i64 {
        self.passed() + self.failed()
    }

    /// The run completed with results, none failed or errored, and every
    /// workflow sub-result passed too.
    pub fn all_passed(&self) -> bool {
        if self.status != "completed" {
            return false;
        }
        let clean = self.failed() == 0 && self.count("errored") == 0 && self.total() > 0;
        if !self.sub_results.is_empty() {
            let own = if self.result_counts.is_some() {
                clean
            } else {
                true
            };
            return own && self.sub_results.values().all(EvalResults::all_passed);
        }
        clean
    }

    /// Fail when anything failed or errored, with a summary (or `msg`).
    pub fn raise_for_status(
        &self,
        msg: Option<&str>,
    ) -> std::result::Result<(), EvalNotPassedError> {
        if self.all_passed() {
            return Ok(());
        }
        let errored = self.count("errored");
        let mut detail = msg.map(str::to_string).unwrap_or_else(|| {
            format!(
                "Eval run {} {}: {} passed, {} failed.",
                self.run_id,
                self.status,
                self.passed(),
                self.failed()
            )
        });
        if errored > 0 {
            detail.push_str(&format!(" {errored} errored."));
        }
        if let Some(url) = &self.report_url {
            detail.push_str(&format!(" See {url} for details."));
        }
        if let Some(error) = &self.error {
            detail.push_str(&format!(" Error: {error}"));
        }
        let failed: Vec<&str> = self
            .sub_results
            .iter()
            .filter(|(_, s)| !s.all_passed())
            .map(|(n, _)| n.as_str())
            .collect();
        if !failed.is_empty() {
            detail.push_str(&format!(" Failed: {}.", failed.join(", ")));
        }
        let errored_items: Vec<String> = self
            .items
            .iter()
            .filter(|i| i.is_error())
            .map(|i| {
                format!(
                    "{}: {}",
                    i.item_id,
                    i.error_code.as_deref().unwrap_or("unknown")
                )
            })
            .collect();
        if !errored_items.is_empty() {
            detail.push_str(&format!(" Errored items: {}.", errored_items.join(", ")));
        }
        Err(EvalNotPassedError(detail))
    }

    fn walk<'a>(&'a self, f: &mut dyn FnMut(&'a EvalItemResult)) {
        for item in &self.items {
            f(item);
        }
        for sub in self.sub_results.values() {
            sub.walk(f);
        }
    }

    /// Fail when any item's score (from `evaluator`, if given) is below
    /// `min_score`, sub-results included.
    pub fn assert_score_at_least(
        &self,
        min_score: f64,
        evaluator: Option<&str>,
        msg: Option<&str>,
    ) -> std::result::Result<(), EvalNotPassedError> {
        let mut offenders = Vec::new();
        self.walk(&mut |item| {
            for score in &item.scores {
                if evaluator.is_some_and(|e| e != score.name) {
                    continue;
                }
                if score.score < min_score {
                    offenders.push(format!(
                        "{}/{}={:.3}",
                        item.item_id, score.name, score.score
                    ));
                }
            }
        });
        if offenders.is_empty() {
            return Ok(());
        }
        Err(EvalNotPassedError(msg.map(str::to_string).unwrap_or_else(
            || {
                format!(
                    "{} score(s) below threshold {min_score}{}: {}{}",
                    offenders.len(),
                    evaluator.map(|e| format!(" for {e}")).unwrap_or_default(),
                    offenders
                        .iter()
                        .take(5)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", "),
                    more(offenders.len())
                )
            },
        )))
    }

    /// Fail when a rubric dimension scores below `min_score` on any item
    /// where it applies; with `require_applicable`, also when it applies to
    /// none of an item's scores.
    pub fn assert_dimension_score_at_least(
        &self,
        dimension_id: &str,
        min_score: f64,
        evaluator: Option<&str>,
        require_applicable: bool,
        msg: Option<&str>,
    ) -> std::result::Result<(), EvalNotPassedError> {
        let mut offenders = Vec::new();
        let mut missing = Vec::new();
        self.walk(&mut |item| {
            let mut found = false;
            for score in &item.scores {
                if evaluator.is_some_and(|e| e != score.name) {
                    continue;
                }
                for dimension in score.dimensions.iter().flatten() {
                    if dimension.id != dimension_id || !dimension.applicable {
                        continue;
                    }
                    found = true;
                    if dimension.score.is_none_or(|s| (s as f64) < min_score) {
                        offenders.push(format!(
                            "{}/{}/{dimension_id}={}",
                            item.item_id,
                            score.name,
                            dimension
                                .score
                                .map_or("None".to_string(), |s| s.to_string())
                        ));
                    }
                }
            }
            if require_applicable && !found {
                missing.push(item.item_id.clone());
            }
        });
        let mut problems = Vec::new();
        if !offenders.is_empty() {
            problems.push(format!(
                "{} dimension score(s) for '{dimension_id}' below {min_score}: {}{}",
                offenders.len(),
                offenders
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
                more(offenders.len())
            ));
        }
        if !missing.is_empty() {
            problems.push(format!(
                "Dimension '{dimension_id}' not applicable on {} item(s): {}",
                missing.len(),
                missing
                    .iter()
                    .take(5)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if problems.is_empty() {
            return Ok(());
        }
        Err(EvalNotPassedError(
            msg.map(str::to_string)
                .unwrap_or_else(|| problems.join("; ")),
        ))
    }

    /// Fail when any item failed or errored, sub-results included.
    pub fn assert_no_failed_items(
        &self,
        msg: Option<&str>,
    ) -> std::result::Result<(), EvalNotPassedError> {
        let mut bad = Vec::new();
        self.walk(&mut |item| {
            if item.is_failed() || item.is_error() {
                bad.push(format!("{}:{}", item.item_id, item.status));
            }
        });
        if bad.is_empty() {
            return Ok(());
        }
        Err(EvalNotPassedError(msg.map(str::to_string).unwrap_or_else(
            || {
                format!(
                    "{} item(s) failed or errored: {}{}",
                    bad.len(),
                    bad.iter().take(5).cloned().collect::<Vec<_>>().join(", "),
                    more(bad.len())
                )
            },
        )))
    }
}

fn more(n: usize) -> String {
    if n > 5 {
        format!(" (+{} more)", n - 5)
    } else {
        String::new()
    }
}

// ---------------------------------------------------------------------------
// Evaluators and checks
// ---------------------------------------------------------------------------

/// An evaluation provider: scores a batch of items.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// The provider's display name.
    fn name(&self) -> &str;

    /// Score `items` as a run named `eval_name`.
    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults>;
}

/// One check's verdict on one item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckResult {
    /// Whether the check passed.
    pub passed: bool,
    /// Why.
    pub reason: String,
    /// The check's name.
    pub check_name: String,
}

impl CheckResult {
    fn new(passed: bool, reason: impl Into<String>, check_name: &str) -> Self {
        Self {
            passed,
            reason: reason.into(),
            check_name: check_name.to_string(),
        }
    }
}

/// A check a [`LocalEvaluator`] applies to each item.
#[async_trait]
pub trait EvalCheck: Send + Sync {
    /// Check one item.
    async fn check(&self, item: &EvalItem) -> Result<CheckResult>;
}

#[async_trait]
impl<F> EvalCheck for F
where
    F: Fn(&EvalItem) -> CheckResult + Send + Sync,
{
    async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
        Ok(self(item))
    }
}

/// A shared [`EvalCheck`].
pub type SharedCheck = Arc<dyn EvalCheck>;

/// Pass when the response contains every keyword.
pub fn keyword_check<S: Into<String>>(
    keywords: impl IntoIterator<Item = S>,
    case_sensitive: bool,
) -> SharedCheck {
    let keywords: Vec<String> = keywords.into_iter().map(Into::into).collect();
    Arc::new(move |item: &EvalItem| {
        let fold = |s: &str| {
            if case_sensitive {
                s.to_string()
            } else {
                s.to_lowercase()
            }
        };
        let text = fold(&item.response());
        let missing: Vec<&String> = keywords
            .iter()
            .filter(|k| !text.contains(&fold(k)))
            .collect();
        if missing.is_empty() {
            CheckResult::new(true, "All keywords found", "keyword_check")
        } else {
            CheckResult::new(
                false,
                format!("Missing keywords: {missing:?}"),
                "keyword_check",
            )
        }
    })
}

/// How [`tool_called_check`] combines its tool names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolCallMode {
    /// Every tool must be called.
    #[default]
    All,
    /// At least one must be.
    Any,
}

/// Pass when the conversation called the named tools.
pub fn tool_called_check<S: Into<String>>(
    tool_names: impl IntoIterator<Item = S>,
    mode: ToolCallMode,
) -> SharedCheck {
    let names: Vec<String> = tool_names.into_iter().map(Into::into).collect();
    Arc::new(move |item: &EvalItem| {
        let called: HashSet<String> = item.tool_calls().into_iter().map(|(n, _)| n).collect();
        let mut sorted: Vec<&String> = called.iter().collect();
        sorted.sort();
        match mode {
            ToolCallMode::All => {
                let missing: Vec<&String> = names.iter().filter(|n| !called.contains(*n)).collect();
                if missing.is_empty() {
                    CheckResult::new(
                        true,
                        format!("All expected tools called: {sorted:?}"),
                        "tool_called",
                    )
                } else {
                    CheckResult::new(
                        false,
                        format!("Expected tools not called: {missing:?} (called: {sorted:?})"),
                        "tool_called",
                    )
                }
            }
            ToolCallMode::Any => {
                let mut found: Vec<&String> =
                    names.iter().filter(|n| called.contains(*n)).collect();
                found.sort();
                if found.is_empty() {
                    CheckResult::new(
                        false,
                        format!("None of expected tools called: {names:?} (called: {sorted:?})"),
                        "tool_called",
                    )
                } else {
                    CheckResult::new(
                        true,
                        format!("Expected tool found: {found:?}"),
                        "tool_called",
                    )
                }
            }
        }
    })
}

/// Pass when every [`EvalItem::expected_tool_calls`] name was called
/// (unordered; extra calls and arguments are not checked).
pub fn tool_calls_present() -> SharedCheck {
    Arc::new(|item: &EvalItem| {
        let expected = item.expected_tool_calls.clone().unwrap_or_default();
        if expected.is_empty() {
            return CheckResult::new(
                true,
                "No expected tool calls specified.",
                "tool_calls_present",
            );
        }
        let actual: HashSet<String> = item.tool_calls().into_iter().map(|(n, _)| n).collect();
        let mut sorted: Vec<&String> = actual.iter().collect();
        sorted.sort();
        let (found, missing): (Vec<&String>, Vec<&String>) = expected
            .iter()
            .map(|e| &e.name)
            .partition(|n| actual.contains(*n));
        if missing.is_empty() {
            CheckResult::new(
                true,
                format!("All expected tools called: {found:?} (called: {sorted:?})"),
                "tool_calls_present",
            )
        } else {
            CheckResult::new(
                false,
                format!("Missing tool calls: {missing:?} (called: {sorted:?})"),
                "tool_calls_present",
            )
        }
    })
}

/// Pass when every expected tool call was made with (at least) the expected
/// arguments.
pub fn tool_call_args_match() -> SharedCheck {
    Arc::new(|item: &EvalItem| {
        let expected = item.expected_tool_calls.clone().unwrap_or_default();
        if expected.is_empty() {
            return CheckResult::new(
                true,
                "No expected tool calls specified.",
                "tool_call_args_match",
            );
        }
        let actual = item.tool_calls();
        let mut matched = 0;
        let mut details = Vec::new();
        for exp in &expected {
            let matching: Vec<&Option<Map<String, Value>>> = actual
                .iter()
                .filter(|(n, _)| *n == exp.name)
                .map(|(_, a)| a)
                .collect();
            if matching.is_empty() {
                details.push(format!("  {}: not called", exp.name));
                continue;
            }
            let Some(want) = &exp.arguments else {
                matched += 1;
                details.push(format!("  {}: called (args not checked)", exp.name));
                continue;
            };
            let found = matching.iter().any(|args| {
                args.as_ref()
                    .is_some_and(|a| want.iter().all(|(k, v)| a.get(k) == Some(v)))
            });
            if found {
                matched += 1;
                details.push(format!("  {}: args match", exp.name));
            } else {
                let shown: Vec<Value> = matching
                    .iter()
                    .map(|a| (*a).clone().map_or(Value::Null, Value::Object))
                    .collect();
                details.push(format!(
                    "  {}: args mismatch (actual: {})",
                    exp.name,
                    Value::Array(shown)
                ));
            }
        }
        CheckResult::new(
            matched == expected.len(),
            format!(
                "Tool call args match: {matched}/{}\n{}",
                expected.len(),
                details.join("\n")
            ),
            "tool_call_args_match",
        )
    })
}

/// What a function evaluator may return; see [`evaluator`].
#[derive(Debug, Clone, PartialEq)]
pub enum CheckOutcome {
    /// Pass or fail.
    Bool(bool),
    /// A score: 0.5 or more passes.
    Score(f64),
    /// A JSON object with `score` (and optional `threshold`, `passed`,
    /// `reason`) or `passed` (and optional `reason`).
    Json(Value),
    /// A ready verdict.
    Result(CheckResult),
}

impl From<bool> for CheckOutcome {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<f64> for CheckOutcome {
    fn from(v: f64) -> Self {
        Self::Score(v)
    }
}
impl From<Value> for CheckOutcome {
    fn from(v: Value) -> Self {
        Self::Json(v)
    }
}
impl From<CheckResult> for CheckOutcome {
    fn from(v: CheckResult) -> Self {
        Self::Result(v)
    }
}

impl CheckOutcome {
    /// The verdict, named `check_name`.
    pub fn into_result(self, check_name: &str) -> Result<CheckResult> {
        Ok(match self {
            Self::Result(r) => r,
            Self::Bool(b) => CheckResult::new(b, if b { "passed" } else { "failed" }, check_name),
            Self::Score(s) => CheckResult::new(s >= 0.5, format!("score={s:.3}"), check_name),
            Self::Json(Value::Object(map)) => {
                if let Some(raw) = map.get("score") {
                    let score = raw.as_f64().ok_or_else(|| {
                        Error::Configuration(format!(
                            "Function evaluator '{check_name}' returned dict with non-numeric 'score' value: {raw}"
                        ))
                    })?;
                    let passed = match map.get("passed") {
                        Some(p) => truthy(p),
                        None => {
                            score >= map.get("threshold").and_then(Value::as_f64).unwrap_or(0.5)
                        }
                    };
                    let reason = map
                        .get("reason")
                        .map(display)
                        .unwrap_or_else(|| format!("score={score:.3}"));
                    CheckResult::new(passed, reason, check_name)
                } else if let Some(p) = map.get("passed") {
                    if !(p.is_boolean() || p.is_i64() || p.is_u64()) {
                        return Err(Error::Configuration(format!(
                            "Function evaluator '{check_name}' returned dict with non-boolean 'passed' value: {p}"
                        )));
                    }
                    let passed = truthy(p);
                    let reason = map
                        .get("reason")
                        .map(display)
                        .unwrap_or_else(|| if passed { "passed" } else { "failed" }.into());
                    CheckResult::new(passed, reason, check_name)
                } else {
                    return Err(unsupported(
                        check_name,
                        "object without 'score' or 'passed'",
                    ));
                }
            }
            Self::Json(other) => return Err(unsupported(check_name, json_kind(&other))),
        })
    }
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::Null => false,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn display(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn json_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

fn unsupported(check_name: &str, kind: &str) -> Error {
    Error::Configuration(format!(
        "Function evaluator '{check_name}' returned unsupported type {kind}. Expected bool, float, dict, or CheckResult."
    ))
}

/// Wrap a function of the item as a named check. It may return a `bool`, an
/// `f64` score (0.5 passes), a JSON object (`{"score": .., "threshold": ..}`
/// or `{"passed": ..}`, each with an optional `reason`) or a
/// [`CheckResult`].
pub fn evaluator<F, R>(name: impl Into<String>, f: F) -> SharedCheck
where
    F: Fn(&EvalItem) -> R + Send + Sync + 'static,
    R: Into<CheckOutcome>,
{
    let name = name.into();
    struct Wrapped<F> {
        name: String,
        f: F,
    }
    #[async_trait]
    impl<F, R> EvalCheck for Wrapped<F>
    where
        F: Fn(&EvalItem) -> R + Send + Sync,
        R: Into<CheckOutcome>,
    {
        async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
            (self.f)(item).into().into_result(&self.name)
        }
    }
    Arc::new(Wrapped { name, f })
}

/// [`evaluator`] for an async function, such as an LLM judge.
pub fn async_evaluator<F, R>(name: impl Into<String>, f: F) -> SharedCheck
where
    F: for<'a> Fn(&'a EvalItem) -> BoxFuture<'a, Result<R>> + Send + Sync + 'static,
    R: Into<CheckOutcome> + 'static,
{
    struct Wrapped<F> {
        name: String,
        f: F,
    }
    #[async_trait]
    impl<F, R> EvalCheck for Wrapped<F>
    where
        F: for<'a> Fn(&'a EvalItem) -> BoxFuture<'a, Result<R>> + Send + Sync,
        R: Into<CheckOutcome> + 'static,
    {
        async fn check(&self, item: &EvalItem) -> Result<CheckResult> {
            (self.f)(item).await?.into().into_result(&self.name)
        }
    }
    Arc::new(Wrapped {
        name: name.into(),
        f,
    })
}

/// An [`Evaluator`] that runs [`EvalCheck`]s locally, with no service.
///
/// An item passes only when at least one check ran and every check passed;
/// an item with no checks fails, since a pass would carry no evidence.
#[derive(Clone)]
pub struct LocalEvaluator {
    checks: Vec<SharedCheck>,
}

impl fmt::Debug for LocalEvaluator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocalEvaluator")
            .field("checks", &self.checks.len())
            .finish()
    }
}

impl LocalEvaluator {
    /// An evaluator over `checks`.
    pub fn new(checks: Vec<SharedCheck>) -> Self {
        Self { checks }
    }
}

#[async_trait]
impl Evaluator for LocalEvaluator {
    fn name(&self) -> &str {
        "Local"
    }

    async fn evaluate(&self, items: &[EvalItem], eval_name: &str) -> Result<EvalResults> {
        let (mut passed, mut failed) = (0, 0);
        let mut per_check: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
        let mut failures = Vec::new();
        let mut result_items = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let results = join_all(self.checks.iter().map(|c| c.check(item)))
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            let mut item_passed = !results.is_empty();
            let mut scores = Vec::new();
            for result in results {
                let counts = per_check
                    .entry(result.check_name.clone())
                    .or_insert_with(|| {
                        [("passed", 0), ("failed", 0), ("errored", 0)]
                            .into_iter()
                            .map(|(k, v)| (k.to_string(), v))
                            .collect()
                    });
                if result.passed {
                    *counts.get_mut("passed").expect("seeded") += 1;
                } else {
                    *counts.get_mut("failed").expect("seeded") += 1;
                    item_passed = false;
                    failures.push(format!("{}: {}", result.check_name, result.reason));
                }
                scores.push(EvalScoreResult {
                    name: result.check_name.clone(),
                    score: if result.passed { 1.0 } else { 0.0 },
                    passed: Some(result.passed),
                    sample: (!result.reason.is_empty())
                        .then(|| serde_json::json!({"reason": result.reason})),
                    dimensions: None,
                });
            }
            if item_passed {
                passed += 1;
            } else {
                failed += 1;
            }
            result_items.push(EvalItemResult {
                item_id: index.to_string(),
                status: if item_passed { "pass" } else { "fail" }.into(),
                scores,
                input_text: Some(item.query()),
                output_text: Some(item.response()),
                ..Default::default()
            });
        }
        Ok(EvalResults {
            provider: self.name().into(),
            eval_id: "local".into(),
            run_id: eval_name.into(),
            status: "completed".into(),
            result_counts: Some(
                [("passed", passed), ("failed", failed), ("errored", 0)]
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), v))
                    .collect(),
            ),
            per_evaluator: per_check,
            items: result_items,
            error: (!failures.is_empty()).then(|| failures.join("; ")),
            ..Default::default()
        })
    }
}

/// Evaluators, or checks to run in a [`LocalEvaluator`]. Consecutive checks
/// are grouped into one local evaluator, as upstream does with bare
/// callables.
#[derive(Clone)]
pub enum EvaluatorSpec {
    /// A provider.
    Evaluator(Arc<dyn Evaluator>),
    /// A local check.
    Check(SharedCheck),
}

impl From<Arc<dyn Evaluator>> for EvaluatorSpec {
    fn from(e: Arc<dyn Evaluator>) -> Self {
        Self::Evaluator(e)
    }
}
impl From<SharedCheck> for EvaluatorSpec {
    fn from(c: SharedCheck) -> Self {
        Self::Check(c)
    }
}
impl From<LocalEvaluator> for EvaluatorSpec {
    fn from(e: LocalEvaluator) -> Self {
        Self::Evaluator(Arc::new(e))
    }
}

fn resolve_evaluators(specs: Vec<EvaluatorSpec>) -> Result<Vec<Arc<dyn Evaluator>>> {
    let mut resolved: Vec<Arc<dyn Evaluator>> = Vec::new();
    let mut pending: Vec<SharedCheck> = Vec::new();
    for spec in specs {
        match spec {
            EvaluatorSpec::Evaluator(e) => {
                if !pending.is_empty() {
                    resolved.push(Arc::new(LocalEvaluator::new(std::mem::take(&mut pending))));
                }
                resolved.push(e);
            }
            EvaluatorSpec::Check(c) => pending.push(c),
        }
    }
    if !pending.is_empty() {
        resolved.push(Arc::new(LocalEvaluator::new(pending)));
    }
    if resolved.is_empty() {
        return Err(Error::Configuration(
            "Provide at least one evaluator.".into(),
        ));
    }
    Ok(resolved)
}

async fn run_evaluators(
    evaluators: &[Arc<dyn Evaluator>],
    items: &[EvalItem],
    eval_name: &str,
) -> Result<Vec<EvalResults>> {
    let several = evaluators.len() > 1;
    join_all(evaluators.iter().map(|ev| {
        let name = if several {
            format!("{eval_name} ({})", ev.name())
        } else {
            eval_name.to_string()
        };
        async move { ev.evaluate(items, &name).await }
    }))
    .await
    .into_iter()
    .collect()
}

fn to_eval_item(
    query: Vec<Message>,
    response: &AgentResponse,
    tools: Option<Vec<ToolDefinition>>,
    context: Option<String>,
) -> EvalItem {
    let mut conversation = query;
    conversation.extend(response.messages.iter().cloned());
    EvalItem {
        conversation,
        tools: tools.filter(|t| !t.is_empty()),
        context,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// evaluate_agent
// ---------------------------------------------------------------------------

/// Configures an agent evaluation; see [`evaluate_agent`].
pub struct EvaluateAgent {
    agent: Option<Arc<dyn SupportsAgentRun>>,
    queries: Vec<String>,
    expected_output: Option<Vec<String>>,
    expected_tool_calls: Option<Vec<Vec<ExpectedToolCall>>>,
    responses: Option<Vec<AgentResponse>>,
    tools: Option<Vec<ToolDefinition>>,
    evaluators: Vec<EvaluatorSpec>,
    eval_name: Option<String>,
    context: Option<String>,
    conversation_split: Option<ConversationSplitter>,
    num_repetitions: usize,
}

/// Evaluate an agent: run it over queries (or take existing responses) and
/// score the interactions with every evaluator. Returns one
/// [`EvalResults`] per evaluator.
pub fn evaluate_agent() -> EvaluateAgent {
    EvaluateAgent {
        agent: None,
        queries: Vec::new(),
        expected_output: None,
        expected_tool_calls: None,
        responses: None,
        tools: None,
        evaluators: Vec::new(),
        eval_name: None,
        context: None,
        conversation_split: None,
        num_repetitions: 1,
    }
}

impl EvaluateAgent {
    /// The agent to run.
    pub fn agent(mut self, agent: Arc<dyn SupportsAgentRun>) -> Self {
        self.agent = Some(agent);
        self
    }
    /// The test queries.
    pub fn queries<S: Into<String>>(mut self, queries: impl IntoIterator<Item = S>) -> Self {
        self.queries = queries.into_iter().map(Into::into).collect();
        self
    }
    /// The expected output, one per query.
    pub fn expected_output<S: Into<String>>(
        mut self,
        expected: impl IntoIterator<Item = S>,
    ) -> Self {
        self.expected_output = Some(expected.into_iter().map(Into::into).collect());
        self
    }
    /// The expected tool calls, one list per query.
    pub fn expected_tool_calls(mut self, expected: Vec<Vec<ExpectedToolCall>>) -> Self {
        self.expected_tool_calls = Some(expected);
        self
    }
    /// Evaluate these responses (one per query) instead of running the agent.
    pub fn responses(mut self, responses: Vec<AgentResponse>) -> Self {
        self.responses = Some(responses);
        self
    }
    /// Tool definitions for every item (not passed to the agent). An empty
    /// list means none.
    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }
    /// Add an evaluator, or a check for a local evaluator.
    pub fn evaluator(mut self, evaluator: impl Into<EvaluatorSpec>) -> Self {
        self.evaluators.push(evaluator.into());
        self
    }
    /// The run's display name (default `Eval: {agent name}`).
    pub fn eval_name(mut self, name: impl Into<String>) -> Self {
        self.eval_name = Some(name.into());
        self
    }
    /// Grounding context for every item.
    pub fn context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
    /// The split strategy for every item.
    pub fn conversation_split(mut self, split: ConversationSplitter) -> Self {
        self.conversation_split = Some(split);
        self
    }
    /// Run each query this many times (default 1), to measure consistency.
    pub fn num_repetitions(mut self, n: usize) -> Self {
        self.num_repetitions = n;
        self
    }

    /// Run the evaluation.
    pub async fn run(self) -> Result<Vec<EvalResults>> {
        if self.num_repetitions < 1 {
            return Err(Error::Configuration(format!(
                "num_repetitions must be >= 1, got {}.",
                self.num_repetitions
            )));
        }
        let n = self.queries.len();
        if let Some(e) = &self.expected_output {
            if e.len() != n {
                return Err(Error::Configuration(format!(
                    "Got {n} queries but {} expected_output values.",
                    e.len()
                )));
            }
        }
        if let Some(e) = &self.expected_tool_calls {
            if e.len() != n {
                return Err(Error::Configuration(format!(
                    "Got {n} queries but {} expected_tool_calls lists.",
                    e.len()
                )));
            }
        }
        let evaluators = resolve_evaluators(self.evaluators)?;
        let mut items = Vec::new();
        match (&self.responses, &self.agent) {
            (Some(responses), _) => {
                if n == 0 {
                    return Err(Error::Configuration(
                        "Provide 'queries' alongside 'responses' so the conversation can be constructed for evaluation."
                            .into(),
                    ));
                }
                if responses.len() != n {
                    return Err(Error::Configuration(format!(
                        "Got {n} queries but {} responses.",
                        responses.len()
                    )));
                }
                for (q, r) in self.queries.iter().zip(responses) {
                    items.push(to_eval_item(
                        vec![Message::user(q.clone())],
                        r,
                        self.tools.clone(),
                        self.context.clone(),
                    ));
                }
            }
            (None, Some(agent)) if n > 0 => {
                for _ in 0..self.num_repetitions {
                    for q in &self.queries {
                        let response = agent.run(vec![Message::user(q.clone())], None).await?;
                        items.push(to_eval_item(
                            vec![Message::user(q.clone())],
                            &response,
                            self.tools.clone(),
                            self.context.clone(),
                        ));
                    }
                }
            }
            (None, None) if n > 0 => {
                return Err(Error::Configuration(
                    "Provide 'agent' when using 'queries' to run the agent. To evaluate pre-existing responses \
                     without an agent, use 'responses' instead."
                        .into(),
                ))
            }
            _ => {
                return Err(Error::Configuration(
                    "Provide either 'queries' (with 'agent') or 'responses' (or both).".into(),
                ))
            }
        }
        for (i, item) in items.iter_mut().enumerate() {
            if let Some(e) = &self.expected_output {
                item.expected_output = Some(e[i % e.len()].clone());
            }
            if let Some(e) = &self.expected_tool_calls {
                item.expected_tool_calls = Some(e[i % e.len()].clone());
            }
            if let Some(split) = &self.conversation_split {
                item.split_strategy = Some(split.clone());
            }
        }
        let name = self.eval_name.unwrap_or_else(|| {
            let agent = self
                .agent
                .as_ref()
                .map(|a| a.display_name())
                .unwrap_or_else(|| "agent".into());
            format!("Eval: {agent}")
        });
        run_evaluators(&evaluators, &items, &name).await
    }
}

// ---------------------------------------------------------------------------
// evaluate_workflow
// ---------------------------------------------------------------------------

/// Configures a workflow evaluation; see [`evaluate_workflow`].
pub struct EvaluateWorkflow<'a> {
    workflow: &'a Workflow,
    workflow_run: Option<&'a WorkflowRun>,
    queries: Vec<String>,
    expected_output: Option<Vec<String>>,
    evaluators: Vec<EvaluatorSpec>,
    eval_name: Option<String>,
    include_overall: bool,
    include_per_agent: bool,
    conversation_split: Option<ConversationSplitter>,
    num_repetitions: usize,
}

/// Evaluate a multi-agent workflow: each agent's interactions (as
/// sub-results, by executor id) and, optionally, the workflow's final
/// output. Pass a finished run with
/// [`workflow_run`](EvaluateWorkflow::workflow_run), or queries to run.
pub fn evaluate_workflow(workflow: &Workflow) -> EvaluateWorkflow<'_> {
    EvaluateWorkflow {
        workflow,
        workflow_run: None,
        queries: Vec::new(),
        expected_output: None,
        evaluators: Vec::new(),
        eval_name: None,
        include_overall: true,
        include_per_agent: true,
        conversation_split: None,
        num_repetitions: 1,
    }
}

/// One agent interaction recovered from a workflow run.
struct AgentInteraction {
    executor_id: String,
    query: Vec<Message>,
    response: AgentResponse,
}

/// The [`WorkflowEvent::AgentRun`] interactions of a run, skipping internal
/// executors.
fn agent_interactions(run: &WorkflowRun) -> Vec<AgentInteraction> {
    run.events()
        .iter()
        .filter_map(|event| {
            let WorkflowEvent::AgentRun {
                executor_id,
                response,
                input,
            } = event
            else {
                return None;
            };
            let internal = executor_id.starts_with('_')
                || matches!(
                    executor_id.to_lowercase().as_str(),
                    "input-conversation" | "end-conversation" | "end"
                );
            if internal {
                return None;
            }
            let response: AgentResponse = serde_json::from_value(response.clone()).ok()?;
            let input: Vec<Message> = serde_json::from_value(input.clone()).unwrap_or_default();
            let users: Vec<Message> = input
                .iter()
                .filter(|m| m.role.as_str() == Role::USER)
                .cloned()
                .collect();
            let query = if users.is_empty() { input } else { users };
            if query.is_empty() {
                return None;
            }
            Some(AgentInteraction {
                executor_id: executor_id.clone(),
                query,
                response,
            })
        })
        .collect()
}

/// The item for a run's final output against `query`.
fn overall_item(query: &str, run: &WorkflowRun) -> Option<EvalItem> {
    let output = run.outputs().pop()?;
    let messages: Option<Vec<Message>> = serde_json::from_value(output.clone()).ok();
    let response = match messages {
        Some(msgs) if !msgs.is_empty() => AgentResponse {
            messages: vec![Message::assistant(joined_text(&msgs, Role::ASSISTANT))],
            ..Default::default()
        },
        _ => match serde_json::from_value::<AgentResponse>(output.clone()) {
            Ok(r) if !r.messages.is_empty() => r,
            _ => AgentResponse {
                messages: vec![Message::assistant(match &output {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })],
                ..Default::default()
            },
        },
    };
    Some(to_eval_item(
        vec![Message::user(query)],
        &response,
        None,
        None,
    ))
}

impl<'a> EvaluateWorkflow<'a> {
    /// Evaluate this finished run instead of running queries.
    pub fn workflow_run(mut self, run: &'a WorkflowRun) -> Self {
        self.workflow_run = Some(run);
        self
    }
    /// Run the workflow over these queries.
    pub fn queries<S: Into<String>>(mut self, queries: impl IntoIterator<Item = S>) -> Self {
        self.queries = queries.into_iter().map(Into::into).collect();
        self
    }
    /// The expected final output, one per query.
    pub fn expected_output<S: Into<String>>(
        mut self,
        expected: impl IntoIterator<Item = S>,
    ) -> Self {
        self.expected_output = Some(expected.into_iter().map(Into::into).collect());
        self
    }
    /// Add an evaluator, or a check for a local evaluator.
    pub fn evaluator(mut self, evaluator: impl Into<EvaluatorSpec>) -> Self {
        self.evaluators.push(evaluator.into());
        self
    }
    /// The run's display name.
    pub fn eval_name(mut self, name: impl Into<String>) -> Self {
        self.eval_name = Some(name.into());
        self
    }
    /// Whether to evaluate the final output (default true).
    pub fn include_overall(mut self, include: bool) -> Self {
        self.include_overall = include;
        self
    }
    /// Whether to evaluate each agent (default true).
    pub fn include_per_agent(mut self, include: bool) -> Self {
        self.include_per_agent = include;
        self
    }
    /// The split strategy for every item.
    pub fn conversation_split(mut self, split: ConversationSplitter) -> Self {
        self.conversation_split = Some(split);
        self
    }
    /// Run each query this many times (default 1).
    pub fn num_repetitions(mut self, n: usize) -> Self {
        self.num_repetitions = n;
        self
    }

    /// Run the evaluation.
    pub async fn run(self) -> Result<Vec<EvalResults>> {
        if self.workflow_run.is_none() && self.queries.is_empty() {
            return Err(Error::Configuration(
                "Provide either 'workflow_run' or 'queries'.".into(),
            ));
        }
        if let Some(e) = &self.expected_output {
            if self.queries.is_empty() {
                return Err(Error::Configuration(
                    "Provide 'queries' when using 'expected_output'; 'expected_output' is not supported with \
                     'workflow_run' only."
                        .into(),
                ));
            }
            if e.len() != self.queries.len() {
                return Err(Error::Configuration(format!(
                    "Got {} queries but {} expected_output values.",
                    self.queries.len(),
                    e.len()
                )));
            }
        }
        if self.num_repetitions < 1 {
            return Err(Error::Configuration(format!(
                "num_repetitions must be >= 1, got {}.",
                self.num_repetitions
            )));
        }
        let evaluators = resolve_evaluators(self.evaluators)?;
        let name = self
            .eval_name
            .unwrap_or_else(|| "Workflow Eval: Workflow".to_string());

        let mut interactions = Vec::new();
        let mut overall = Vec::new();
        if self.queries.is_empty() {
            let run = self.workflow_run.expect("checked above");
            let found = agent_interactions(run);
            if self.include_overall {
                let query = found.first().map(|i| joined_text(&i.query, Role::USER));
                if let Some(item) = query
                    .filter(|q| !q.is_empty())
                    .and_then(|q| overall_item(&q, run))
                {
                    overall.push(item);
                }
            }
            interactions = found;
        } else {
            for _ in 0..self.num_repetitions {
                for (i, q) in self.queries.iter().enumerate() {
                    let run = self.workflow.run(q.clone()).await?;
                    interactions.extend(agent_interactions(&run));
                    if self.include_overall {
                        if let Some(mut item) = overall_item(q, &run) {
                            if let Some(e) = &self.expected_output {
                                item.expected_output = Some(e[i].clone());
                            }
                            overall.push(item);
                        }
                    }
                }
            }
        }

        let mut by_agent: BTreeMap<String, Vec<EvalItem>> = BTreeMap::new();
        if self.include_per_agent {
            for interaction in interactions {
                by_agent
                    .entry(interaction.executor_id)
                    .or_default()
                    .push(to_eval_item(
                        interaction.query,
                        &interaction.response,
                        None,
                        None,
                    ));
            }
        }
        if by_agent.is_empty() && overall.is_empty() {
            return Err(Error::Configuration(
                "No agent executor data found in the workflow result. Ensure the workflow uses \
                 AgentExecutor-based agents."
                    .into(),
            ));
        }
        if let Some(split) = &self.conversation_split {
            for item in by_agent.values_mut().flatten().chain(overall.iter_mut()) {
                item.split_strategy = Some(split.clone());
            }
        }

        let several = evaluators.len() > 1;
        let mut all = Vec::new();
        for ev in &evaluators {
            let suffix = if several {
                format!(" ({})", ev.name())
            } else {
                String::new()
            };
            let mut subs = BTreeMap::new();
            for (executor_id, items) in &by_agent {
                subs.insert(
                    executor_id.clone(),
                    ev.evaluate(items, &format!("{name} \u{2014} {executor_id}{suffix}"))
                        .await?,
                );
            }
            let mut result = if self.include_overall && !overall.is_empty() {
                ev.evaluate(&overall, &format!("{name} \u{2014} overall{suffix}"))
                    .await?
            } else {
                let completed = subs.values().all(|s: &EvalResults| s.status == "completed");
                EvalResults {
                    provider: ev.name().into(),
                    eval_id: "aggregate".into(),
                    run_id: "aggregate".into(),
                    status: if completed { "completed" } else { "partial" }.into(),
                    result_counts: Some(
                        [
                            (
                                "passed".to_string(),
                                subs.values().map(EvalResults::passed).sum(),
                            ),
                            (
                                "failed".to_string(),
                                subs.values().map(EvalResults::failed).sum(),
                            ),
                        ]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }
            };
            result.sub_results = subs;
            all.push(result);
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convo() -> Vec<Message> {
        vec![
            Message::system("sys"),
            Message::user("first"),
            Message::assistant("a1"),
            Message::user("second"),
            Message::assistant("a2"),
        ]
    }

    #[test]
    fn splits_and_derived_text() {
        let (q, r) = ConversationSplit::LastTurn.split(&convo());
        assert_eq!((q.len(), r.len()), (4, 1));
        let (q, r) = ConversationSplit::Full.split(&convo());
        assert_eq!((q.len(), r.len()), (2, 3));
        let item = EvalItem::new(convo());
        assert_eq!(item.query(), "first second");
        assert_eq!(item.response(), "a2");
        let mut full = EvalItem::new(convo());
        full.split_strategy = Some(ConversationSplit::Full.splitter());
        assert_eq!(full.response(), "a1 a2");
        assert_eq!(full.query(), "first");
        assert_eq!(EvalItem::per_turn_items(&convo(), None, None).len(), 2);
    }

    #[test]
    fn outcomes_coerce_like_upstream() {
        let r = |o: CheckOutcome| o.into_result("c").unwrap();
        assert!(r(true.into()).passed);
        assert!(!r(0.4.into()).passed);
        assert_eq!(r(0.75.into()).reason, "score=0.750");
        assert!(!r(serde_json::json!({"score": 0.6, "threshold": 0.7}).into()).passed);
        assert!(r(serde_json::json!({"score": 0.1, "passed": true}).into()).passed);
        assert_eq!(
            r(serde_json::json!({"passed": 0, "reason": "nope"}).into()).reason,
            "nope"
        );
        assert!(CheckOutcome::from(serde_json::json!("x"))
            .into_result("c")
            .is_err());
        assert!(CheckOutcome::from(serde_json::json!({"score": "high"}))
            .into_result("c")
            .is_err());
    }

    #[test]
    fn all_passed_needs_results() {
        let mut r = EvalResults::new("p");
        assert!(!r.all_passed(), "no results is not a pass");
        r.result_counts = Some([("passed".to_string(), 2)].into_iter().collect());
        assert!(r.all_passed());
        let mut sub = EvalResults::new("p");
        sub.result_counts = Some([("failed".to_string(), 1)].into_iter().collect());
        r.sub_results.insert("writer".into(), sub);
        let err = r.raise_for_status(None).unwrap_err();
        assert!(err.0.contains("Failed: writer."), "{err}");
    }
}
