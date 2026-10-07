//! Standing tool approvals and one-at-a-time approval prompts.
//!
//! Port of upstream `_harness/_tool_approval.py` (`ToolApprovalMiddleware`),
//! shaped like .NET's `ToolApprovalAgent`. [`ToolApprovalAgent`] wraps
//! another agent and, using state kept in the session:
//!
//! - **records standing rules**: an approval created with
//!   [`ToolApprovalAgent::always_approve_tool_response`] (or the
//!   `_with_arguments_` variant) adds a rule, and later calls to that tool
//!   (with any arguments, or the exact same arguments) are approved without
//!   asking;
//! - **auto-approves** requests matching a standing rule or one of the
//!   configured [`ToolAutoApprovalRule`]s, re-running the inner agent with
//!   the approvals so the caller never sees them;
//! - **queues** the rest, so the caller is asked about one tool call at a
//!   time. The approvals are collected and sent to the inner agent together
//!   once the queue is empty.
//!
//! A batch that also asks the user for something else (an OAuth consent
//! request) is returned whole rather than split.
//!
//! The agent requires a session: calling it without one is an error.
//!
//! # Divergences
//!
//! - **A wrapper agent, not middleware**; see the
//!   [module docs](super#divergences-from-upstream).
//! - **How "always approve" is signalled.** Upstream marks the approval
//!   response with an `additional_properties` entry. The Rust approval
//!   content has no property bag, so
//!   [`ToolApprovalAgent::always_approve_tool_response`] records the intent
//!   in the session against the request's id and returns an ordinary
//!   approval. The rule is added when that approval comes back to the agent
//!   still approved and still for the same call; the rule is derived from the
//!   call recorded when the response was created, so a response edited on
//!   the way back cannot widen it.
//! - **No hosted-server boundary.** Upstream rules carry an optional
//!   `server_label` for hosted MCP approvals. Approval requests in this port
//!   are for local function calls only, so rules match by tool name (and
//!   arguments) alone.
//! - Upstream rebinds each inbound approval response to the pending request
//!   it recorded for the session; this port's function-invocation loop does
//!   its own matching of responses to requests, so responses are forwarded
//!   as received.
//! - The function-invocation budget is shared across auto-approval
//!   re-runs by the function-invocation loop itself (it parks the budget in
//!   the session), so there is no extra plumbing for it here. Re-runs are
//!   also capped by [`ToolApprovalAgent::with_max_auto_approval_iterations`]
//!   (default [`DEFAULT_MAX_AUTO_APPROVAL_ITERATIONS`]), as in .NET: each
//!   re-run is a fresh inner run, so the inner per-run iteration limit
//!   restarts every time and cannot bound the chain.
//! - If the inner run fails, the approvals collected for it are put back, so
//!   retrying the last answer still sends the whole batch.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use crate::error::{Error, Result};
use crate::session::AgentSession;
use crate::types::{
    AgentResponse, AgentResponseUpdate, Content, FunctionApprovalRequestContent,
    FunctionApprovalResponseContent, FunctionCallContent, Message, Role, UsageDetails,
};

use super::{channel_stream, json_type_name};

/// The default `source_id` (state key) of a [`ToolApprovalAgent`].
pub const DEFAULT_TOOL_APPROVAL_SOURCE_ID: &str = "tool_approval";

/// The default cap on how many times a [`ToolApprovalAgent`] re-runs its
/// inner agent within one run because every approval request was
/// auto-approved. Matches .NET's `DefaultMaxAutoApprovalIterations`.
pub const DEFAULT_MAX_AUTO_APPROVAL_ITERATIONS: usize = 40;

/// What a standing approval covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolApprovalScope {
    /// Every future call to the tool.
    Tool,
    /// Future calls to the tool with exactly the same arguments.
    ToolWithArguments,
}

/// A standing rule that approves future matching tool calls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolApprovalRule {
    /// The tool the rule applies to.
    pub tool_name: String,
    /// Canonical JSON of each argument value, or `None` for a rule covering
    /// every call to the tool. An empty map matches only calls without
    /// arguments; it is never a wildcard.
    #[serde(default)]
    pub arguments: Option<BTreeMap<String, String>>,
}

impl ToolApprovalRule {
    /// A rule approving every call to `tool_name`.
    pub fn tool(tool_name: impl Into<String>) -> Result<Self> {
        let tool_name = tool_name.into().trim().to_string();
        if tool_name.is_empty() {
            return Err(Error::Configuration(
                "Tool approval rule tool_name must be a non-empty string.".into(),
            ));
        }
        Ok(Self {
            tool_name,
            arguments: None,
        })
    }

    /// A rule approving calls to the same tool with exactly `call`'s
    /// arguments.
    pub fn tool_with_arguments(call: &FunctionCallContent) -> Result<Self> {
        Ok(Self {
            arguments: Some(canonical_arguments(call)?),
            ..Self::tool(call.name.clone())?
        })
    }

    /// Whether the rule approves `call`.
    pub fn matches(&self, call: &FunctionCallContent) -> bool {
        if self.tool_name != call.name {
            return false;
        }
        match &self.arguments {
            None => true,
            Some(expected) => canonical_arguments(call).is_ok_and(|actual| actual == *expected),
        }
    }
}

/// Canonical, order-independent serialization of a call's arguments: each
/// value as compact JSON with object keys sorted.
fn canonical_arguments(call: &FunctionCallContent) -> Result<BTreeMap<String, String>> {
    Ok(call
        .parse_arguments()?
        .into_iter()
        .map(|(k, v)| (k, canonical_json(&v)))
        .collect())
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, String> =
                map.iter().map(|(k, v)| (k, canonical_json(v))).collect();
            let body: Vec<String> = sorted
                .into_iter()
                .map(|(k, v)| format!("{}:{v}", Value::String(k.clone())))
                .collect();
            format!("{{{}}}", body.join(","))
        }
        Value::Array(items) => {
            let body: Vec<String> = items.iter().map(canonical_json).collect();
            format!("[{}]", body.join(","))
        }
        other => other.to_string(),
    }
}

/// An "always approve" choice made for a request, waiting for its approval
/// to come back.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct PendingStandingApproval {
    scope: ToolApprovalScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    function_call: FunctionCallContent,
}

/// The session-backed state of a [`ToolApprovalAgent`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolApprovalState {
    /// Standing approval rules.
    #[serde(default)]
    pub rules: Vec<ToolApprovalRule>,
    /// Approval requests not yet shown to the caller.
    #[serde(default)]
    pub queued_approval_requests: Vec<FunctionApprovalRequestContent>,
    /// Approvals held back until every queued request is answered.
    #[serde(default)]
    pub collected_approval_responses: Vec<FunctionApprovalResponseContent>,
    /// "Always approve" choices keyed by approval-request id.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pending_standing_approvals: BTreeMap<String, PendingStandingApproval>,
}

impl ToolApprovalState {
    fn add_rule_if_missing(&mut self, rule: ToolApprovalRule) {
        if !self.rules.contains(&rule) {
            self.rules.push(rule);
        }
    }
}

/// A heuristic that may approve a tool call without asking the caller.
///
/// Implemented for any `Fn(&FunctionCallContent) -> bool`; implement the
/// trait directly for an asynchronous check.
///
/// **Security:** a rule that matches by tool name approves *any* tool with
/// that name, including one registered later for an unrelated purpose. Make
/// sure no other tool shares a name a rule approves, or match on the
/// arguments as well.
#[async_trait]
pub trait ToolAutoApprovalRule: Send + Sync {
    /// Whether to approve `call` without asking.
    async fn approves(&self, call: &FunctionCallContent) -> bool;
}

#[async_trait]
impl<F> ToolAutoApprovalRule for F
where
    F: Fn(&FunctionCallContent) -> bool + Send + Sync,
{
    async fn approves(&self, call: &FunctionCallContent) -> bool {
        self(call)
    }
}

/// Wraps an agent with standing tool approvals, auto-approval rules and
/// one-at-a-time approval prompts. See the [module docs](self).
#[derive(Clone)]
pub struct ToolApprovalAgent {
    inner: Arc<dyn SupportsAgentRun>,
    source_id: String,
    auto_approval_rules: Vec<Arc<dyn ToolAutoApprovalRule>>,
    max_auto_approval_iterations: usize,
}

impl std::fmt::Debug for ToolApprovalAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolApprovalAgent")
            .field("inner", &self.inner.id())
            .field("source_id", &self.source_id)
            .field("auto_approval_rules", &self.auto_approval_rules.len())
            .field(
                "max_auto_approval_iterations",
                &self.max_auto_approval_iterations,
            )
            .finish()
    }
}

impl ToolApprovalAgent {
    /// Wrap `inner`.
    pub fn new(inner: Arc<dyn SupportsAgentRun>) -> Self {
        Self {
            inner,
            source_id: DEFAULT_TOOL_APPROVAL_SOURCE_ID.to_string(),
            auto_approval_rules: Vec::new(),
            max_auto_approval_iterations: DEFAULT_MAX_AUTO_APPROVAL_ITERATIONS,
        }
    }

    /// Builder: keep state under `source_id` (default `"tool_approval"`).
    pub fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Builder: add a rule that may approve calls without asking. See the
    /// security note on [`ToolAutoApprovalRule`].
    pub fn with_auto_approval_rule(mut self, rule: impl ToolAutoApprovalRule + 'static) -> Self {
        self.auto_approval_rules.push(Arc::new(rule));
        self
    }

    /// Builder: cap how many times one run re-invokes the inner agent because
    /// every approval request it surfaced was auto-approved (default
    /// [`DEFAULT_MAX_AUTO_APPROVAL_ITERATIONS`]). On reaching the cap the
    /// agent takes one final inner turn without auto-approving, and returns
    /// it as-is, so any approval request in it goes to the caller (possibly
    /// several at once). Errors when `max` is zero.
    ///
    /// Counterpart of .NET `ToolApprovalAgentOptions.MaxAutoApprovalIterations`.
    pub fn with_max_auto_approval_iterations(mut self, max: usize) -> Result<Self> {
        if max == 0 {
            return Err(Error::Configuration(
                "max_auto_approval_iterations must be at least 1.".into(),
            ));
        }
        self.max_auto_approval_iterations = max;
        Ok(self)
    }

    /// The cap on auto-approval re-runs within one run.
    pub fn max_auto_approval_iterations(&self) -> usize {
        self.max_auto_approval_iterations
    }

    /// The wrapped agent.
    pub fn inner(&self) -> &Arc<dyn SupportsAgentRun> {
        &self.inner
    }

    /// The state key this agent uses.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// The approval state stored in `session`.
    pub fn state(&self, session: &AgentSession) -> Result<ToolApprovalState> {
        match session.state.get(&self.source_id) {
            None | Some(Value::Null) => Ok(ToolApprovalState::default()),
            Some(value @ Value::Object(_)) => serde_json::from_value(value).map_err(|e| {
                Error::Serialization(format!(
                    "invalid tool approval state under {:?}: {e}",
                    self.source_id
                ))
            }),
            Some(other) => Err(Error::Serialization(format!(
                "session state for {:?} must be a JSON object; got {}",
                self.source_id,
                json_type_name(&other)
            ))),
        }
    }

    fn save_state(&self, session: &AgentSession, state: &ToolApprovalState) -> Result<()> {
        let mut serialized = match serde_json::to_value(state)? {
            Value::Object(map) => map,
            _ => unreachable!("ToolApprovalState serializes to an object"),
        };
        // Keep any keys someone else stored alongside ours.
        if let Some(Value::Object(existing)) = session.state.get(&self.source_id) {
            for (key, value) in existing {
                serialized.entry(key).or_insert(value);
            }
        }
        session
            .state
            .insert(self.source_id.clone(), Value::Object(serialized));
        Ok(())
    }

    /// Approve `request` and, once the approval reaches this agent, approve
    /// every later call to the same tool without asking.
    ///
    /// Counterpart of upstream `create_always_approve_tool_response`.
    pub fn always_approve_tool_response(
        &self,
        session: &AgentSession,
        request: &FunctionApprovalRequestContent,
        reason: Option<&str>,
    ) -> Result<FunctionApprovalResponseContent> {
        self.always_approve(session, request, ToolApprovalScope::Tool, reason)
    }

    /// Approve `request` and, once the approval reaches this agent, approve
    /// later calls to the same tool **with the same arguments** without
    /// asking.
    ///
    /// Counterpart of upstream
    /// `create_always_approve_tool_with_arguments_response`.
    pub fn always_approve_tool_with_arguments_response(
        &self,
        session: &AgentSession,
        request: &FunctionApprovalRequestContent,
        reason: Option<&str>,
    ) -> Result<FunctionApprovalResponseContent> {
        self.always_approve(
            session,
            request,
            ToolApprovalScope::ToolWithArguments,
            reason,
        )
    }

    fn always_approve(
        &self,
        session: &AgentSession,
        request: &FunctionApprovalRequestContent,
        scope: ToolApprovalScope,
        reason: Option<&str>,
    ) -> Result<FunctionApprovalResponseContent> {
        let mut state = self.state(session)?;
        state.pending_standing_approvals.insert(
            request.id.clone(),
            PendingStandingApproval {
                scope,
                reason: reason.map(str::to_string),
                function_call: request.function_call.clone(),
            },
        );
        self.save_state(session, &state)?;
        Ok(request.create_response(true))
    }

    /// Strip approval responses out of the caller's input into
    /// `collected_approval_responses`, recording standing rules on the way.
    fn prepare_inbound(
        &self,
        messages: Vec<Message>,
        state: &mut ToolApprovalState,
    ) -> Vec<Message> {
        let mut prepared = Vec::with_capacity(messages.len());
        for mut message in messages {
            let mut changed = false;
            let mut kept = Vec::with_capacity(message.contents.len());
            for content in std::mem::take(&mut message.contents) {
                match content {
                    Content::FunctionApprovalResponse(response) => {
                        changed = true;
                        if let Some(pending) = state.pending_standing_approvals.remove(&response.id)
                        {
                            if response.approved
                                && pending
                                    .function_call
                                    .same_invocation(&response.function_call)
                            {
                                let rule = match pending.scope {
                                    ToolApprovalScope::Tool => {
                                        ToolApprovalRule::tool(pending.function_call.name.clone())
                                    }
                                    ToolApprovalScope::ToolWithArguments => {
                                        ToolApprovalRule::tool_with_arguments(
                                            &pending.function_call,
                                        )
                                    }
                                };
                                if let Ok(rule) = rule {
                                    state.add_rule_if_missing(rule);
                                }
                            }
                        }
                        // A retry after a failed inner run resends an answer
                        // that is already collected; keep one copy.
                        state
                            .collected_approval_responses
                            .retain(|existing| existing.id != response.id);
                        state.collected_approval_responses.push(response);
                    }
                    other => kept.push(other),
                }
            }
            // Drop a message only when it held nothing but approval responses.
            if !changed || !kept.is_empty() {
                message.contents = kept;
                prepared.push(message);
            }
        }
        prepared
    }

    async fn is_auto_approved(
        &self,
        request: &FunctionApprovalRequestContent,
        state: &ToolApprovalState,
    ) -> bool {
        let call = &request.function_call;
        if state.rules.iter().any(|rule| rule.matches(call)) {
            return true;
        }
        for rule in &self.auto_approval_rules {
            if rule.approves(call).await {
                return true;
            }
        }
        false
    }

    /// Move queued requests that a rule now approves into the collected
    /// approvals.
    async fn drain_queue(&self, state: &mut ToolApprovalState) {
        let queued = std::mem::take(&mut state.queued_approval_requests);
        for request in queued {
            if self.is_auto_approved(&request, state).await {
                state
                    .collected_approval_responses
                    .push(request.create_response(true));
            } else {
                state.queued_approval_requests.push(request);
            }
        }
    }

    /// Prepend the collected approvals to `messages` as one `user` message.
    fn inject_collected(messages: Vec<Message>, state: &mut ToolApprovalState) -> Vec<Message> {
        if state.collected_approval_responses.is_empty() {
            return messages;
        }
        let approvals: Vec<Content> = state
            .collected_approval_responses
            .drain(..)
            .map(Content::FunctionApprovalResponse)
            .collect();
        let mut out = Vec::with_capacity(messages.len() + 1);
        out.push(Message::with_contents(Role::user(), approvals));
        out.extend(messages);
        out
    }

    /// Decide what happens to an outbound batch of approval requests.
    /// Returns the indices to remove from the batch (auto-approved or
    /// queued) and whether every request was auto-approved.
    async fn decide(
        &self,
        requests: &[FunctionApprovalRequestContent],
        state: &mut ToolApprovalState,
        preserve_batch: bool,
    ) -> (HashSet<usize>, bool) {
        if requests.is_empty() {
            return (HashSet::new(), false);
        }
        let mut removed = HashSet::new();
        let mut unresolved = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            if self.is_auto_approved(request, state).await {
                state
                    .collected_approval_responses
                    .push(request.create_response(true));
                removed.insert(index);
            } else {
                unresolved.push(index);
            }
        }
        if removed.is_empty() && (preserve_batch || unresolved.len() <= 1) {
            return (HashSet::new(), false);
        }
        if !preserve_batch {
            for &index in unresolved.iter().skip(1) {
                state.queued_approval_requests.push(requests[index].clone());
                removed.insert(index);
            }
        }
        (removed, unresolved.is_empty())
    }

    fn queued_response(request: FunctionApprovalRequestContent) -> AgentResponse {
        AgentResponse {
            messages: vec![Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionApprovalRequest(request)],
            )],
            ..Default::default()
        }
    }

    async fn run_impl(
        &self,
        messages: Vec<Message>,
        session: &mut AgentSession,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let mut state = self.state(session)?;
        let mut messages = self.prepare_inbound(messages, &mut state);
        self.drain_queue(&mut state).await;
        if !state.queued_approval_requests.is_empty() {
            let next = state.queued_approval_requests.remove(0);
            self.save_state(session, &state)?;
            return Ok(Self::queued_response(next));
        }
        let mut iteration = 0usize;
        // Usage of every inner run this call made: an auto-approved pass is
        // discarded, but its tokens were still spent.
        let mut usage: Option<UsageDetails> = None;
        loop {
            let collected = state.collected_approval_responses.clone();
            messages = Self::inject_collected(messages, &mut state);
            self.save_state(session, &state)?;

            let result = self
                .inner
                .run_with_options(messages, Some(&mut *session), options.clone())
                .await;
            let mut response = match result {
                Ok(response) => response,
                Err(error) => {
                    // Keep the batch so a retry can still send it.
                    state.collected_approval_responses = collected;
                    self.save_state(session, &state)?;
                    return Err(error);
                }
            };
            if let Some(run_usage) = response.usage_details.take() {
                usage = Some(match usage {
                    Some(total) => total + run_usage,
                    None => run_usage,
                });
            }
            response.usage_details = usage.clone();
            if iteration >= self.max_auto_approval_iterations {
                // Cap reached: return this turn as-is, without auto-approving
                // again, so the caller decides on any request in it.
                return Ok(response);
            }
            iteration += 1;

            let preserve_batch =
                has_other_user_input(response.messages.iter().flat_map(|m| &m.contents));
            let positions: Vec<(usize, usize)> = approval_positions(&response.messages);
            let requests: Vec<FunctionApprovalRequestContent> = positions
                .iter()
                .map(|&(m, c)| approval_at(&response.messages, m, c).clone())
                .collect();
            let (removed, all_auto_approved) =
                self.decide(&requests, &mut state, preserve_batch).await;
            if !removed.is_empty() {
                let drop: HashSet<(usize, usize)> = removed.iter().map(|&i| positions[i]).collect();
                response.messages = remove_contents(response.messages, &drop);
            }
            self.save_state(session, &state)?;
            if !all_auto_approved || preserve_batch {
                return Ok(response);
            }
            messages = Vec::new();
        }
    }

    async fn stream_impl(
        self,
        messages: Vec<Message>,
        mut session: AgentSession,
        options: Option<AgentRunOptions>,
        sink: super::UpdateSink,
    ) -> Result<()> {
        let mut state = self.state(&session)?;
        let mut messages = self.prepare_inbound(messages, &mut state);
        self.drain_queue(&mut state).await;
        if !state.queued_approval_requests.is_empty() {
            let next = state.queued_approval_requests.remove(0);
            self.save_state(&session, &state)?;
            sink.send(AgentResponseUpdate {
                contents: vec![Content::FunctionApprovalRequest(next)],
                role: Some(Role::assistant()),
                ..Default::default()
            })
            .await;
            return Ok(());
        }
        let mut iteration = 0usize;
        loop {
            let collected = state.collected_approval_responses.clone();
            messages = Self::inject_collected(messages, &mut state);
            self.save_state(&session, &state)?;
            // On reaching the cap this pass streams through as-is.
            let capped = iteration >= self.max_auto_approval_iterations;
            iteration += 1;

            // Keep the batch so a retry can still send it.
            let restore = |state: &mut ToolApprovalState, session: &AgentSession| {
                state.collected_approval_responses = collected.clone();
                self.save_state(session, state)
            };
            let mut inner = match self
                .inner
                .run_stream(messages, Some(session.clone()), options.clone())
                .await
            {
                Ok(inner) => inner,
                Err(error) => {
                    restore(&mut state, &session)?;
                    return Err(error);
                }
            };
            // Stream until the first approval request, then buffer the rest
            // so auto-approved or queued requests never reach the caller.
            let mut buffered: Vec<AgentResponseUpdate> = Vec::new();
            let mut saw_other_input = false;
            let mut conversation_id: Option<String> = None;
            while let Some(update) = inner.next().await {
                let update = match update {
                    Ok(update) => update,
                    Err(error) => {
                        restore(&mut state, &session)?;
                        return Err(error);
                    }
                };
                if let Some(cid) = &update.conversation_id {
                    conversation_id = Some(cid.clone());
                }
                let has_request = update
                    .contents
                    .iter()
                    .any(|c| matches!(c, Content::FunctionApprovalRequest(_)));
                if capped || (buffered.is_empty() && !has_request) {
                    saw_other_input |= has_other_user_input(update.contents.iter());
                    if !sink.send(update).await {
                        return Ok(());
                    }
                    continue;
                }
                buffered.push(update);
            }
            // The inner run adopted any service conversation id on its own
            // copy of the session; carry it to ours for the next pass.
            if let Some(cid) = conversation_id {
                session.try_adopt_service_session_id(&cid);
            }
            if buffered.is_empty() {
                return Ok(());
            }

            let preserve_batch =
                saw_other_input || has_other_user_input(buffered.iter().flat_map(|u| &u.contents));
            let positions: Vec<(usize, usize)> = buffered
                .iter()
                .enumerate()
                .flat_map(|(u, update)| {
                    update
                        .contents
                        .iter()
                        .enumerate()
                        .filter_map(move |(c, content)| {
                            matches!(content, Content::FunctionApprovalRequest(_)).then_some((u, c))
                        })
                })
                .collect();
            let requests: Vec<FunctionApprovalRequestContent> = positions
                .iter()
                .map(|&(u, c)| match &buffered[u].contents[c] {
                    Content::FunctionApprovalRequest(r) => r.clone(),
                    _ => unreachable!("positions index approval requests"),
                })
                .collect();
            let (removed, all_auto_approved) =
                self.decide(&requests, &mut state, preserve_batch).await;
            self.save_state(&session, &state)?;
            let drop: HashSet<(usize, usize)> = removed.iter().map(|&i| positions[i]).collect();
            for (u, mut update) in buffered.into_iter().enumerate() {
                let before = update.contents.len();
                update.contents = update
                    .contents
                    .into_iter()
                    .enumerate()
                    .filter(|(c, _)| !drop.contains(&(u, *c)))
                    .map(|(_, content)| content)
                    .collect();
                if before > 0 && update.contents.is_empty() {
                    continue;
                }
                if !sink.send(update).await {
                    return Ok(());
                }
            }
            if !all_auto_approved || preserve_batch {
                return Ok(());
            }
            messages = Vec::new();
        }
    }
}

/// Whether `contents` asks the user for something other than a tool approval.
fn has_other_user_input<'a>(contents: impl IntoIterator<Item = &'a Content>) -> bool {
    contents
        .into_iter()
        .any(|c| matches!(c, Content::OauthConsentRequest(_)))
}

fn approval_positions(messages: &[Message]) -> Vec<(usize, usize)> {
    messages
        .iter()
        .enumerate()
        .flat_map(|(m, message)| {
            message
                .contents
                .iter()
                .enumerate()
                .filter_map(move |(c, content)| {
                    matches!(content, Content::FunctionApprovalRequest(_)).then_some((m, c))
                })
        })
        .collect()
}

fn approval_at(messages: &[Message], m: usize, c: usize) -> &FunctionApprovalRequestContent {
    match &messages[m].contents[c] {
        Content::FunctionApprovalRequest(r) => r,
        _ => unreachable!("positions index approval requests"),
    }
}

/// Remove the contents at `drop` positions, dropping messages left empty.
fn remove_contents(messages: Vec<Message>, drop: &HashSet<(usize, usize)>) -> Vec<Message> {
    messages
        .into_iter()
        .enumerate()
        .filter_map(|(m, mut message)| {
            let before = message.contents.len();
            message.contents = std::mem::take(&mut message.contents)
                .into_iter()
                .enumerate()
                .filter(|(c, _)| !drop.contains(&(m, *c)))
                .map(|(_, content)| content)
                .collect();
            (before == message.contents.len() || !message.contents.is_empty()).then_some(message)
        })
        .collect()
}

#[async_trait]
impl SupportsAgentRun for ToolApprovalAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.run_with_options(messages, session, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let session = session.ok_or_else(|| {
            Error::Configuration("ToolApprovalAgent requires an AgentSession.".into())
        })?;
        self.run_impl(messages, session, options).await
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let session = session.ok_or_else(|| {
            Error::Configuration("ToolApprovalAgent requires an AgentSession.".into())
        })?;
        let this = self.clone();
        Ok(channel_stream(move |sink| {
            this.stream_impl(messages, session, options, sink)
        }))
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn create_session(&self) -> AgentSession {
        self.inner.create_session()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionArguments;
    use serde_json::json;

    fn call(name: &str, args: Value) -> FunctionCallContent {
        let map = args.as_object().cloned().unwrap_or_default();
        FunctionCallContent::new(
            format!("call_{name}"),
            name,
            Some(FunctionArguments::Object(map.into_iter().collect())),
        )
    }

    #[test]
    fn canonical_json_sorts_keys_recursively() {
        assert_eq!(
            canonical_json(&json!({"b": [1, {"d": 1, "c": 2}], "a": "x"})),
            r#"{"a":"x","b":[1,{"c":2,"d":1}]}"#
        );
    }

    #[test]
    fn rules_match_by_name_and_exact_arguments() {
        let one = call("write", json!({"path": "a", "opts": {"x": 1, "y": 2}}));
        let reordered = call("write", json!({"opts": {"y": 2, "x": 1}, "path": "a"}));
        let other = call("write", json!({"path": "b"}));

        let tool_rule = ToolApprovalRule::tool("write").unwrap();
        assert!(tool_rule.matches(&one) && tool_rule.matches(&other));
        assert!(!tool_rule.matches(&call("read", json!({}))));

        let args_rule = ToolApprovalRule::tool_with_arguments(&one).unwrap();
        assert!(args_rule.matches(&reordered));
        assert!(!args_rule.matches(&other));

        // An empty-arguments rule only matches no-argument calls.
        let empty = ToolApprovalRule::tool_with_arguments(&call("write", json!({}))).unwrap();
        assert_eq!(empty.arguments, Some(BTreeMap::new()));
        assert!(empty.matches(&FunctionCallContent::new("c", "write", None)));
        assert!(!empty.matches(&other));

        assert!(ToolApprovalRule::tool("  ").is_err());
    }

    #[test]
    fn state_round_trips_through_json() {
        let mut state = ToolApprovalState::default();
        state.rules.push(ToolApprovalRule::tool("t").unwrap());
        state
            .queued_approval_requests
            .push(FunctionApprovalRequestContent {
                id: "r1".into(),
                function_call: call("t", json!({"a": 1})),
            });
        let value = serde_json::to_value(&state).unwrap();
        let restored: ToolApprovalState = serde_json::from_value(value).unwrap();
        assert_eq!(restored, state);
    }
}
