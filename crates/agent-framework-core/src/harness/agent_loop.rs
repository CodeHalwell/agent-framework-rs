//! The agent loop: re-run an agent until a condition says the work is done.
//!
//! Port of upstream `_harness/_loop.py`. [`LoopAgent`] wraps another agent
//! and, after every run, asks a [`LoopCallback`] whether to go again. The
//! same type covers both upstream patterns:
//!
//! 1. **A hand-written condition** ([`LoopAgent::builder`]): loop while a
//!    marker is missing, while [`todos_remaining`] says the
//!    [`TodoProvider`] has open items, and so on. Each pass adds an entry to
//!    a **progress log** (by default the response text, or whatever
//!    [`LoopAgentBuilder::record_feedback`] returns), which callbacks see as
//!    [`LoopContext::progress`] and which is injected into the next input.
//! 2. **A chat-client judge** ([`LoopAgent::with_judge`]): a second model
//!    decides whether the original request has been answered, and its
//!    reasoning is fed back to the agent when it has not.
//!
//! The input of each further pass comes from
//! [`LoopAgentBuilder::next_message`] (default: [`DEFAULT_NEXT_MESSAGE`]).
//! [`max_iterations`](LoopAgentBuilder::max_iterations) caps the number of
//! runs (default [`DEFAULT_MAX_ITERATIONS`], or
//! [`DEFAULT_JUDGE_MAX_ITERATIONS`] for a judge loop); `None` removes the
//! cap. A run that ends with a pending tool-approval request stops the loop
//! so the caller can answer it.
//!
//! A non-streaming run returns every pass's messages with the injected loop
//! messages between them, and the usage summed over all passes
//! ([`return_final_only`](LoopAgentBuilder::return_final_only) returns just
//! the last pass). A streaming run yields every pass's updates and the
//! injected messages as `user` updates between passes.
//!
//! # Fresh context
//!
//! With [`fresh_context`](LoopAgentBuilder::fresh_context) each pass starts
//! from the original input plus the progress log instead of the accumulated
//! conversation. The session's state bag and service conversation id are
//! snapshotted before the loop and restored before every further pass, so
//! working state from earlier passes is discarded. Conversation history in
//! this port lives in a [`HistoryProvider`](crate::history::HistoryProvider)
//! rather than the state bag, so the loop also swaps the session's history
//! providers for a scratch in-memory history seeded with the pre-loop
//! transcript for the duration of the loop. Afterwards the original history
//! providers are put back and the final pass's input and response are
//! recorded on them, which leaves the session's history where upstream
//! leaves it: the pre-loop transcript plus the final pass. History providers
//! registered on the wrapped *agent* (rather than the session) are not
//! swapped.
//!
//! # Divergences
//!
//! - A wrapper agent rather than `AgentMiddleware`; see the
//!   [module docs](super#divergences-from-upstream).
//! - Callbacks receive a [`LoopContext`] instead of keyword arguments, and
//!   there is no `agent` entry: helpers that need a provider take it
//!   explicitly ([`todos_remaining`], [`todos_remaining_message`]).
//! - The loop always runs against a session. When the caller passes none,
//!   one is created with the wrapped agent's
//!   [`create_session`](SupportsAgentRun::create_session), and the session
//!   gets an in-memory history provider unless it is service-managed (as
//!   [`Agent`](crate::agent::Agent) would attach). Upstream injects the
//!   whole progress log when there is no session; here only the latest
//!   entry is injected outside fresh-context mode, because earlier entries
//!   are already in the session's history.
//! - The aggregated non-streaming transcript and the streamed loop messages
//!   contain only what the loop synthesized (the progress message and the
//!   next-message input); re-sent caller input is not repeated, as in .NET.
//! - The judge's verdict parser is synchronous, and a judge reply whose
//!   text is a JSON verdict is accepted even when the client did not fill in
//!   [`ChatResponse::value`].
//! - `background_tasks_running` is not ported: there are no background
//!   agents in this port.

use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use crate::client::ChatClient;
use crate::error::{Error, Result};
use crate::history::{ensure_history_provider, InMemoryHistoryProvider};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::types::{
    AgentResponse, AgentResponseUpdate, ChatOptions, ChatResponse, Content, Message,
    ResponseFormat, UsageDetails,
};

use super::{channel_stream, AgentModeProvider, TodoProvider, UpdateSink};

/// The default input of every pass after the first.
pub const DEFAULT_NEXT_MESSAGE: &str = "Continue working on the task. If it is complete, say so.";

/// The default cap on the number of runs of a [`LoopAgent`].
pub const DEFAULT_MAX_ITERATIONS: usize = 10;

/// The default cap on the number of runs of a judge loop
/// ([`LoopAgent::with_judge`]). Judged loops are costly, so the cap is lower.
pub const DEFAULT_JUDGE_MAX_ITERATIONS: usize = 5;

/// Replaced with the rendered criteria in judge instructions. Instructions
/// without it get no criteria.
pub const CRITERIA_PLACEHOLDER: &str = "{{criteria}}";

/// The marker a judge without structured output ends with when the request
/// has been addressed.
pub const JUDGE_VERDICT_DONE: &str = "VERDICT: DONE";

/// The marker a judge without structured output ends with when more work is
/// needed. It wins over [`JUDGE_VERDICT_DONE`] when both appear.
pub const JUDGE_VERDICT_MORE: &str = "VERDICT: MORE";

/// The default judge instructions.
pub const DEFAULT_JUDGE_INSTRUCTIONS: &str = "You are an evaluator. You are given a user's \
original request and an agent's latest response. Decide whether the agent has fully addressed \
the original request. Set 'answered' to true if the request has been fully addressed, or false \
if more work is still required, and use 'reasoning' to briefly justify your decision. If you \
cannot return structured output, end your reply with a line reading exactly 'VERDICT: DONE' when \
the request has been fully addressed or 'VERDICT: MORE' when more work is still required.\
{{criteria}}";

/// What a [`LoopAgent`] callback sees after a pass.
#[derive(Debug, Clone)]
pub struct LoopContext {
    /// The number of completed passes (1 after the first).
    pub iteration: usize,
    /// The response of the pass that just completed.
    pub last_result: AgentResponse,
    /// The input of the pass that just completed.
    pub messages: Vec<Message>,
    /// The input of the first pass (including any additional instructions).
    pub original_messages: Vec<Message>,
    /// The session the loop runs against. Clones share its state bag.
    pub session: AgentSession,
    /// The progress log so far. While the condition is evaluated and the
    /// feedback is recorded it holds the earlier passes' entries; the
    /// next-message callback also sees this pass's entry.
    pub progress: Vec<String>,
    /// The feedback the condition returned for this pass, if any.
    pub feedback: Option<String>,
}

/// A loop condition's answer: whether to run again, and optional feedback
/// for the next-message and record-feedback callbacks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoopDecision {
    /// Whether to run the agent again.
    pub should_continue: bool,
    /// Feedback surfaced as [`LoopContext::feedback`].
    pub feedback: Option<String>,
}

impl LoopDecision {
    /// Stop looping.
    pub fn stop() -> Self {
        Self::default()
    }

    /// Run again.
    pub fn run_again() -> Self {
        Self {
            should_continue: true,
            feedback: None,
        }
    }

    /// Attach feedback.
    pub fn with_feedback(mut self, feedback: impl Into<String>) -> Self {
        self.feedback = Some(feedback.into());
        self
    }
}

impl From<bool> for LoopDecision {
    fn from(should_continue: bool) -> Self {
        Self {
            should_continue,
            feedback: None,
        }
    }
}

impl From<(bool, Option<String>)> for LoopDecision {
    fn from((should_continue, feedback): (bool, Option<String>)) -> Self {
        Self {
            should_continue,
            feedback,
        }
    }
}

/// A [`LoopAgent`] callback producing a `T` from the loop state.
///
/// Implemented for synchronous closures `Fn(&LoopContext) -> R` where
/// `R: Into<T>` (annotate the parameter: `|ctx: &LoopContext| ...`). Use
/// [`async_loop_callback`] for an asynchronous closure, or implement the
/// trait directly. The three callback kinds are:
///
/// - the condition, `LoopCallback<LoopDecision>` (closures may return a
///   `bool` or a `(bool, Option<String>)`);
/// - the next message, `LoopCallback<Option<Vec<Message>>>`: `None` re-sends
///   the previous pass's input unchanged (without progress injection), or
///   the default nudge in fresh-context mode;
/// - the feedback recorder, `LoopCallback<Option<String>>`: the entry to
///   append to the progress log, or `None` for none.
#[async_trait]
pub trait LoopCallback<T>: Send + Sync {
    /// Produce the value for the current loop state.
    async fn call(&self, ctx: &LoopContext) -> Result<T>;
}

#[async_trait]
impl<T, F, R> LoopCallback<T> for F
where
    T: Send + 'static,
    F: Fn(&LoopContext) -> R + Send + Sync,
    R: Into<T>,
{
    async fn call(&self, ctx: &LoopContext) -> Result<T> {
        Ok(self(ctx).into())
    }
}

/// A [`LoopCallback`] built from an asynchronous closure; see
/// [`async_loop_callback`].
pub struct AsyncLoopCallback<F>(F);

/// Adapt an asynchronous closure, which receives an owned [`LoopContext`],
/// into a [`LoopCallback`].
///
/// ```
/// # use agent_framework_core::harness::{async_loop_callback, LoopContext, LoopDecision};
/// let condition = async_loop_callback(|ctx: LoopContext| async move {
///     Ok(LoopDecision::from(!ctx.last_result.text().contains("DONE")))
/// });
/// # let _ = condition;
/// ```
pub fn async_loop_callback<T, F, Fut>(f: F) -> AsyncLoopCallback<F>
where
    F: Fn(LoopContext) -> Fut + Send + Sync,
    Fut: Future<Output = Result<T>> + Send,
{
    AsyncLoopCallback(f)
}

#[async_trait]
impl<T, F, Fut> LoopCallback<T> for AsyncLoopCallback<F>
where
    T: Send + 'static,
    F: Fn(LoopContext) -> Fut + Send + Sync,
    Fut: Future<Output = Result<T>> + Send,
{
    async fn call(&self, ctx: &LoopContext) -> Result<T> {
        (self.0)(ctx.clone()).await
    }
}

type Condition = Arc<dyn LoopCallback<LoopDecision>>;
type NextMessage = Arc<dyn LoopCallback<Option<Vec<Message>>>>;
type RecordFeedback = Arc<dyn LoopCallback<Option<String>>>;

struct LoopConfig {
    should_continue: Condition,
    max_iterations: Option<usize>,
    next_message: Option<NextMessage>,
    record_feedback: Option<RecordFeedback>,
    inject_progress: bool,
    fresh_context: bool,
    return_final_only: bool,
    additional_instructions: Option<String>,
}

/// Re-runs a wrapped agent until a condition says to stop. See the
/// [module docs](self).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::prelude::*;
/// # use agent_framework_core::harness::{LoopAgent, LoopContext};
/// # async fn demo(agent: Agent) -> Result<()> {
/// let looping = LoopAgent::builder(Arc::new(agent), |ctx: &LoopContext| {
///     !ctx.last_result.text().contains("DONE")
/// })
/// .max_iterations(Some(5))
/// .build()?;
/// let response = looping.run(vec![Message::user("Write the report.")], None).await?;
/// # let _ = response;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct LoopAgent {
    inner: Arc<dyn SupportsAgentRun>,
    config: Arc<LoopConfig>,
}

impl std::fmt::Debug for LoopAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopAgent")
            .field("inner", &self.inner.id())
            .field("max_iterations", &self.config.max_iterations)
            .field("fresh_context", &self.config.fresh_context)
            .finish_non_exhaustive()
    }
}

/// Configures a [`LoopAgent`].
pub struct LoopAgentBuilder {
    inner: Arc<dyn SupportsAgentRun>,
    should_continue: Condition,
    max_iterations: Option<usize>,
    next_message: Option<NextMessage>,
    record_feedback: Option<RecordFeedback>,
    inject_progress: bool,
    fresh_context: bool,
    return_final_only: bool,
    additional_instructions: Option<String>,
}

impl LoopAgentBuilder {
    /// The cap on the number of runs (default [`DEFAULT_MAX_ITERATIONS`]).
    /// `None` loops until the condition stops it, so make sure it can.
    pub fn max_iterations(mut self, max_iterations: Option<usize>) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Produce the input of each further pass (default: one `user` message
    /// with [`DEFAULT_NEXT_MESSAGE`]).
    pub fn next_message(
        mut self,
        next_message: impl LoopCallback<Option<Vec<Message>>> + 'static,
    ) -> Self {
        self.next_message = Some(Arc::new(next_message));
        self
    }

    /// Produce each pass's progress-log entry (default: the response text).
    /// Prefer a terse summary for long loops.
    pub fn record_feedback(
        mut self,
        record_feedback: impl LoopCallback<Option<String>> + 'static,
    ) -> Self {
        self.record_feedback = Some(Arc::new(record_feedback));
        self
    }

    /// Whether to inject the progress log into the next input as a `user`
    /// message reading "Progress so far: ..." (default `true`). Outside
    /// fresh-context mode only the latest entry is injected.
    pub fn inject_progress(mut self, inject: bool) -> Self {
        self.inject_progress = inject;
        self
    }

    /// Start every further pass from the original input plus the progress
    /// log, resetting the session in between (default `false`). See
    /// [fresh context](self#fresh-context).
    pub fn fresh_context(mut self, fresh: bool) -> Self {
        self.fresh_context = fresh;
        self
    }

    /// Make a non-streaming run return only the final pass's response
    /// (default `false`: the whole transcript). Streaming is unaffected.
    pub fn return_final_only(mut self, final_only: bool) -> Self {
        self.return_final_only = final_only;
        self
    }

    /// An extra instruction prepended to the input as a `system` message, so
    /// it is part of the original input every pass sees.
    pub fn additional_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.additional_instructions = Some(instructions.into());
        self
    }

    /// Build the loop. Errors when `max_iterations` is `Some(0)`.
    pub fn build(self) -> Result<LoopAgent> {
        if self.max_iterations == Some(0) {
            return Err(Error::Configuration(
                "max_iterations must be None or a positive integer (>= 1).".into(),
            ));
        }
        Ok(LoopAgent {
            inner: self.inner,
            config: Arc::new(LoopConfig {
                should_continue: self.should_continue,
                max_iterations: self.max_iterations,
                next_message: self.next_message,
                record_feedback: self.record_feedback,
                inject_progress: self.inject_progress,
                fresh_context: self.fresh_context,
                return_final_only: self.return_final_only,
                additional_instructions: self.additional_instructions,
            }),
        })
    }
}

impl LoopAgent {
    /// Loop `inner` while `should_continue` says so.
    pub fn builder(
        inner: Arc<dyn SupportsAgentRun>,
        should_continue: impl LoopCallback<LoopDecision> + 'static,
    ) -> LoopAgentBuilder {
        LoopAgentBuilder {
            inner,
            should_continue: Arc::new(should_continue),
            max_iterations: Some(DEFAULT_MAX_ITERATIONS),
            next_message: None,
            record_feedback: None,
            inject_progress: true,
            fresh_context: false,
            return_final_only: false,
            additional_instructions: None,
        }
    }

    /// Loop `inner` until `judge` decides the original request has been
    /// answered.
    ///
    /// The judge's reasoning is fed back as the next input
    /// ([`judge_feedback_message`]), the cap defaults to
    /// [`DEFAULT_JUDGE_MAX_ITERATIONS`], and any [`Judge::criteria`] are
    /// also given to the agent as an additional instruction. All of these
    /// can still be overridden on the returned builder.
    ///
    /// **Security:** the judge is sent the original request and the agent's
    /// latest response on every pass, and its reasoning is fed back to the
    /// agent. Only use a judge client you trust as much as the agent's own
    /// model: a hostile one can exfiltrate the conversation or steer the
    /// agent through its feedback.
    pub fn with_judge(inner: Arc<dyn SupportsAgentRun>, judge: Judge) -> LoopAgentBuilder {
        let agent_instructions = judge.agent_instructions();
        let mut builder = Self::builder(inner, judge)
            .max_iterations(Some(DEFAULT_JUDGE_MAX_ITERATIONS))
            .next_message(judge_feedback_message);
        builder.additional_instructions = agent_instructions;
        builder
    }

    /// The wrapped agent.
    pub fn inner(&self) -> &Arc<dyn SupportsAgentRun> {
        &self.inner
    }

    async fn run_loop(
        &self,
        input: Vec<Message>,
        session: &mut AgentSession,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        ensure_history_provider(session);
        let mut run = LoopRun::start(self.config.clone(), input, session).await?;
        let mut messages = run.original.clone();
        let mut transcript = Vec::new();
        let mut usage: Option<UsageDetails> = None;
        let last = loop {
            let response = match self
                .inner
                .run_with_options(messages.clone(), Some(&mut *session), options.clone())
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    run.abort(session);
                    return Err(e);
                }
            };
            if let Some(u) = &response.usage_details {
                usage = Some(match usage {
                    Some(acc) => acc + u.clone(),
                    None => u.clone(),
                });
            }
            transcript.extend(response.messages.iter().cloned());
            match run.after_iteration(session, &messages, &response).await {
                Err(e) => {
                    run.abort(session);
                    return Err(e);
                }
                Ok(Step::Stop) => break response,
                Ok(Step::Continue {
                    messages: next,
                    surfaced,
                }) => {
                    transcript.extend(surfaced);
                    messages = next;
                }
            }
        };
        run.finish(session, &messages, &last).await?;
        if self.config.return_final_only {
            return Ok(last);
        }
        Ok(AgentResponse {
            messages: transcript,
            usage_details: usage,
            ..last
        })
    }

    async fn stream_loop(
        self,
        input: Vec<Message>,
        mut session: AgentSession,
        options: Option<AgentRunOptions>,
        sink: UpdateSink,
    ) -> Result<()> {
        ensure_history_provider(&mut session);
        let mut run = LoopRun::start(self.config.clone(), input, &mut session).await?;
        let mut messages = run.original.clone();
        loop {
            let mut stream = match self
                .inner
                .run_stream(messages.clone(), Some(session.clone()), options.clone())
                .await
            {
                Ok(s) => s,
                Err(e) => {
                    run.abort(&mut session);
                    return Err(e);
                }
            };
            let mut updates = Vec::new();
            while let Some(update) = stream.next().await {
                let update = match update {
                    Ok(u) => u,
                    Err(e) => {
                        run.abort(&mut session);
                        return Err(e);
                    }
                };
                updates.push(update.clone());
                if !sink.send(update).await {
                    run.abort(&mut session);
                    return Ok(());
                }
            }
            let response = AgentResponse::from_updates(updates);
            // The inner run adopted any service conversation id on its own
            // copy of the session; carry it to ours.
            if let Some(cid) = &response.conversation_id {
                session.try_adopt_service_session_id(cid);
            }
            match run
                .after_iteration(&mut session, &messages, &response)
                .await
            {
                Err(e) => {
                    run.abort(&mut session);
                    return Err(e);
                }
                Ok(Step::Stop) => {
                    return run.finish(&mut session, &messages, &response).await;
                }
                Ok(Step::Continue {
                    messages: next,
                    surfaced,
                }) => {
                    for message in surfaced {
                        if !sink.send(message_to_update(message)).await {
                            run.abort(&mut session);
                            return Ok(());
                        }
                    }
                    messages = next;
                }
            }
        }
    }
}

#[async_trait]
impl SupportsAgentRun for LoopAgent {
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
        match session {
            Some(session) => self.run_loop(messages, session, options).await,
            None => {
                let mut session = self.inner.create_session();
                self.run_loop(messages, &mut session, options).await
            }
        }
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let session = session.unwrap_or_else(|| self.inner.create_session());
        let this = self.clone();
        Ok(channel_stream(move |sink| {
            this.stream_loop(messages, session, options, sink)
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

/// Whether `response` carries a pending tool-approval request.
fn has_pending_approval_request(response: &AgentResponse) -> bool {
    response
        .messages
        .iter()
        .flat_map(|m| &m.contents)
        .any(|c| matches!(c, Content::FunctionApprovalRequest(_)))
}

fn message_to_update(message: Message) -> AgentResponseUpdate {
    AgentResponseUpdate {
        contents: message.contents,
        role: Some(message.role),
        author_name: message.author_name,
        message_id: Some(
            message
                .message_id
                .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string()),
        ),
        ..Default::default()
    }
}

fn render_progress(entries: &[String]) -> Message {
    let body: Vec<String> = entries.iter().map(|e| format!("- {e}")).collect();
    Message::user(format!("Progress so far:\n{}", body.join("\n")))
}

enum Step {
    Stop,
    Continue {
        /// The next pass's full input.
        messages: Vec<Message>,
        /// The part of it the loop synthesized, surfaced to the caller.
        surfaced: Vec<Message>,
    },
}

/// The per-run state of a loop.
struct LoopRun {
    config: Arc<LoopConfig>,
    original: Vec<Message>,
    progress: Vec<String>,
    iteration: usize,
    fresh: Option<FreshContext>,
}

impl LoopRun {
    async fn start(
        config: Arc<LoopConfig>,
        mut input: Vec<Message>,
        session: &mut AgentSession,
    ) -> Result<Self> {
        if let Some(instructions) = &config.additional_instructions {
            input.insert(0, Message::system(instructions.clone()));
        }
        let fresh = if config.fresh_context {
            let fresh = FreshContext::capture(session).await?;
            fresh.install(session);
            Some(fresh)
        } else {
            None
        };
        Ok(Self {
            config,
            original: input,
            progress: Vec::new(),
            iteration: 0,
            fresh,
        })
    }

    async fn after_iteration(
        &mut self,
        session: &mut AgentSession,
        messages_used: &[Message],
        result: &AgentResponse,
    ) -> Result<Step> {
        self.iteration += 1;
        // Escape hatch: a pending approval goes back to the caller.
        if has_pending_approval_request(result) {
            return Ok(Step::Stop);
        }
        let mut ctx = LoopContext {
            iteration: self.iteration,
            last_result: result.clone(),
            messages: messages_used.to_vec(),
            original_messages: self.original.clone(),
            session: session.clone(),
            progress: self.progress.clone(),
            feedback: None,
        };
        // The cap short-circuits before the (possibly costly) condition.
        let decision = if self
            .config
            .max_iterations
            .is_some_and(|max| self.iteration >= max)
        {
            LoopDecision::stop()
        } else {
            self.config.should_continue.call(&ctx).await?
        };
        ctx.feedback = decision.feedback.clone();

        let entry = match &self.config.record_feedback {
            Some(record) => record.call(&ctx).await?,
            None => Some(result.text().trim().to_string()),
        };
        if let Some(entry) = entry.filter(|e| !e.is_empty()) {
            self.progress.push(entry);
            ctx.progress = self.progress.clone();
        }

        if !decision.should_continue {
            return Ok(Step::Stop);
        }
        if let Some(fresh) = &self.fresh {
            fresh.reset(session)?;
        }
        self.resolve_next(&ctx, messages_used).await
    }

    async fn resolve_next(&self, ctx: &LoopContext, messages_used: &[Message]) -> Result<Step> {
        let fresh = self.config.fresh_context;
        let next = match &self.config.next_message {
            None => vec![Message::user(DEFAULT_NEXT_MESSAGE)],
            Some(callback) => match callback.call(ctx).await? {
                Some(messages) => messages,
                None if !fresh => {
                    return Ok(Step::Continue {
                        messages: messages_used.to_vec(),
                        surfaced: Vec::new(),
                    })
                }
                None => vec![Message::user(DEFAULT_NEXT_MESSAGE)],
            },
        };
        let mut surfaced = Vec::new();
        if self.config.inject_progress && !self.progress.is_empty() {
            let entries = if fresh {
                &self.progress[..]
            } else {
                &self.progress[self.progress.len() - 1..]
            };
            surfaced.push(render_progress(entries));
        }
        surfaced.extend(next);
        let messages = if fresh {
            self.original
                .iter()
                .cloned()
                .chain(surfaced.clone())
                .collect()
        } else {
            surfaced.clone()
        };
        Ok(Step::Continue { messages, surfaced })
    }

    async fn finish(
        &mut self,
        session: &mut AgentSession,
        last_input: &[Message],
        last_response: &AgentResponse,
    ) -> Result<()> {
        match self.fresh.take() {
            Some(fresh) => fresh.finish(session, last_input, last_response).await,
            None => Ok(()),
        }
    }

    fn abort(&mut self, session: &mut AgentSession) {
        if let Some(fresh) = self.fresh.take() {
            session.context_providers = fresh.providers;
        }
    }
}

/// What fresh-context mode needs to reset a session between passes.
struct FreshContext {
    snapshot: Value,
    providers: Vec<Arc<dyn ContextProvider>>,
    /// The session's transcript before the loop, when it has a history
    /// provider to read it from.
    pre_loop_history: Option<Vec<Message>>,
}

impl FreshContext {
    async fn capture(session: &AgentSession) -> Result<Self> {
        let providers = session.context_providers.clone();
        // Plain loops rather than iterator adapters: a closure held across
        // an `.await` trips the higher-ranked `Send` check of the caller.
        let mut history: Vec<Arc<dyn ContextProvider>> = Vec::new();
        for provider in &providers {
            if provider.is_history_provider() {
                history.push(provider.clone());
            }
        }
        let pre_loop_history = if history.is_empty() {
            None
        } else {
            let mut ctx = SessionContext::new(Vec::new());
            ctx.session_id = Some(session.session_id().to_string());
            ctx.session = Some(session.clone());
            for provider in &history {
                provider.before_run(&mut ctx).await?;
            }
            Some(ctx.messages)
        };
        Ok(Self {
            snapshot: session.to_dict(),
            providers,
            pre_loop_history,
        })
    }

    /// Point the session at a scratch history seeded with the pre-loop
    /// transcript, keeping its other providers.
    fn install(&self, session: &mut AgentSession) {
        let mut providers = Vec::with_capacity(self.providers.len());
        let mut scratch = self.pre_loop_history.as_ref().map(|history| {
            Arc::new(InMemoryHistoryProvider::with_messages(history.clone()))
                as Arc<dyn ContextProvider>
        });
        for provider in &self.providers {
            if provider.is_history_provider() {
                if let Some(scratch) = scratch.take() {
                    providers.push(scratch);
                }
            } else {
                providers.push(provider.clone());
            }
        }
        session.context_providers = providers;
    }

    fn reset(&self, session: &mut AgentSession) -> Result<()> {
        session.restore_snapshot(&self.snapshot)?;
        self.install(session);
        Ok(())
    }

    async fn finish(
        self,
        session: &mut AgentSession,
        last_input: &[Message],
        last_response: &AgentResponse,
    ) -> Result<()> {
        session.context_providers = self.providers.clone();
        for provider in &self.providers {
            if provider.is_history_provider() {
                provider
                    .after_run(last_input, &last_response.messages, None)
                    .await?;
            }
        }
        Ok(())
    }
}

/// The structured verdict a [`Judge`] asks for.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JudgeVerdict {
    /// Whether the agent fully addressed the original request.
    pub answered: bool,
    /// A brief justification.
    #[serde(default)]
    pub reasoning: String,
}

impl JudgeVerdict {
    /// The JSON-schema response format requesting a verdict.
    pub fn response_format() -> ResponseFormat {
        ResponseFormat::JsonSchema {
            name: "JudgeVerdict".into(),
            description: Some("Structured verdict returned by the judge.".into()),
            schema: json!({
                "type": "object",
                "properties": {
                    "answered": {
                        "type": "boolean",
                        "description": "True if the agent has fully addressed the original \
                            request and it adheres to the other judging standards, otherwise False."
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief justification for the verdict."
                    }
                },
                "required": ["answered", "reasoning"],
                "additionalProperties": false
            }),
            strict: Some(true),
        }
    }
}

/// Converts a judge's [`ChatResponse`] into a [`JudgeVerdict`]; see
/// [`Judge::verdict_parser`].
pub type VerdictParser = Arc<dyn Fn(&ChatResponse) -> Result<JudgeVerdict> + Send + Sync>;

/// A chat-client judge for [`LoopAgent::with_judge`]. Used as a loop
/// condition it continues while the original request is not answered, with
/// the judge's reasoning as feedback.
///
/// The judge is called directly (no tools, session or middleware) with the
/// instructions, the original input and the agent's latest response, and
/// asked for a [`JudgeVerdict`]. When the client ignores structured output,
/// the reply's [`JUDGE_VERDICT_MORE`] / [`JUDGE_VERDICT_DONE`] marker decides
/// (more work wins, and so does a reply with neither).
#[derive(Clone)]
pub struct Judge {
    client: Arc<dyn ChatClient>,
    criteria: Vec<String>,
    instructions: Option<String>,
    response_format: Option<ResponseFormat>,
    verdict_parser: Option<VerdictParser>,
}

impl std::fmt::Debug for Judge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Judge")
            .field("criteria", &self.criteria)
            .field("instructions", &self.instructions)
            .field("response_format", &self.response_format)
            .field("verdict_parser", &self.verdict_parser.is_some())
            .finish_non_exhaustive()
    }
}

impl Judge {
    /// A judge backed by `client`, asking for a [`JudgeVerdict`].
    pub fn new(client: Arc<dyn ChatClient>) -> Self {
        Self {
            client,
            criteria: Vec::new(),
            instructions: None,
            response_format: Some(JudgeVerdict::response_format()),
            verdict_parser: None,
        }
    }

    /// Criteria the response must satisfy. They are rendered into the judge
    /// instructions at [`CRITERIA_PLACEHOLDER`] and, through
    /// [`LoopAgent::with_judge`], given to the agent as an extra instruction.
    pub fn criteria<S: Into<String>>(mut self, criteria: impl IntoIterator<Item = S>) -> Self {
        self.criteria = criteria.into_iter().map(Into::into).collect();
        self
    }

    /// Replace [`DEFAULT_JUDGE_INSTRUCTIONS`]. Include
    /// [`CRITERIA_PLACEHOLDER`] where the criteria should go.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// The response format sent to the judge (default:
    /// [`JudgeVerdict::response_format`]). A provider-specific format
    /// usually needs a [`verdict_parser`](Self::verdict_parser).
    pub fn response_format(mut self, format: Option<ResponseFormat>) -> Self {
        self.response_format = format;
        self
    }

    /// Convert the judge's response yourself. The parser owns the
    /// interpretation: its errors surface, with no marker fallback.
    pub fn verdict_parser(
        mut self,
        parser: impl Fn(&ChatResponse) -> Result<JudgeVerdict> + Send + Sync + 'static,
    ) -> Self {
        self.verdict_parser = Some(Arc::new(parser));
        self
    }

    /// The judge's system instructions with the criteria substituted.
    pub fn rendered_instructions(&self) -> String {
        let block = if self.criteria.is_empty() {
            String::new()
        } else {
            format!(
                "\n\nThe response must satisfy all of the following criteria:\n{}",
                bullets(&self.criteria)
            )
        };
        self.instructions
            .as_deref()
            .unwrap_or(DEFAULT_JUDGE_INSTRUCTIONS)
            .replace(CRITERIA_PLACEHOLDER, &block)
    }

    /// The instruction telling the agent about the criteria, if any.
    pub fn agent_instructions(&self) -> Option<String> {
        (!self.criteria.is_empty()).then(|| {
            format!(
                "Your response must satisfy all of the following criteria:\n{}",
                bullets(&self.criteria)
            )
        })
    }

    /// Ask the judge about the loop's latest pass.
    pub async fn verdict(&self, ctx: &LoopContext) -> Result<JudgeVerdict> {
        let mut messages = vec![
            Message::system(self.rendered_instructions()),
            Message::user("Evaluate the agent's work. The user's original request follows:"),
        ];
        messages.extend(ctx.original_messages.iter().cloned());
        messages.push(Message::user("The agent's latest response was:"));
        messages.extend(ctx.last_result.messages.iter().cloned());
        messages.push(Message::user(
            "Has the original request been fully addressed?",
        ));
        let mut options = ChatOptions::new();
        options.response_format = self.response_format.clone();
        let response = self.client.get_response(messages, options).await?;

        if let Some(parser) = &self.verdict_parser {
            return parser(&response);
        }
        if let Some(verdict) = response
            .value
            .clone()
            .and_then(|v| serde_json::from_value::<JudgeVerdict>(v).ok())
        {
            return Ok(verdict);
        }
        let text = response.text();
        if let Ok(verdict) = serde_json::from_str::<JudgeVerdict>(text.trim()) {
            return Ok(verdict);
        }
        let upper = text.to_uppercase();
        let answered = !upper.contains(JUDGE_VERDICT_MORE) && upper.contains(JUDGE_VERDICT_DONE);
        Ok(JudgeVerdict {
            answered,
            reasoning: text.trim().to_string(),
        })
    }
}

fn bullets(items: &[String]) -> String {
    items
        .iter()
        .map(|i| format!("- {i}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[async_trait]
impl LoopCallback<LoopDecision> for Judge {
    async fn call(&self, ctx: &LoopContext) -> Result<LoopDecision> {
        let verdict = self.verdict(ctx).await?;
        Ok(LoopDecision {
            should_continue: !verdict.answered,
            feedback: Some(verdict.reasoning).filter(|r| !r.is_empty()),
        })
    }
}

/// The default next message of a judge loop: relays the judge's reasoning
/// ([`LoopContext::feedback`]) to the agent, or falls back to
/// [`DEFAULT_NEXT_MESSAGE`].
pub fn judge_feedback_message(ctx: &LoopContext) -> Option<Vec<Message>> {
    let text = match ctx.feedback.as_deref().filter(|f| !f.is_empty()) {
        Some(feedback) => format!(
            "An evaluator reviewed your previous response and judged that it does not yet fully \
             address the original request.\n\nEvaluator feedback: {feedback}\n\n\
             Revise and continue so the original request is fully addressed."
        ),
        None => DEFAULT_NEXT_MESSAGE.to_string(),
    };
    Some(vec![Message::user(text)])
}

/// A loop condition that continues while a [`TodoProvider`] has open items;
/// see [`todos_remaining`].
#[derive(Debug, Clone)]
pub struct TodosRemaining {
    todos: TodoProvider,
    modes: Option<(AgentModeProvider, Vec<String>)>,
}

/// A loop condition that continues while `todos` (the provider registered
/// on the wrapped agent) has open items for the session. Counterpart of
/// upstream `todos_remaining` and .NET's `TodoCompletionLoopEvaluator`.
pub fn todos_remaining(todos: &TodoProvider) -> TodosRemaining {
    TodosRemaining {
        todos: todos.clone(),
        modes: None,
    }
}

impl TodosRemaining {
    /// Only continue while `mode_provider` reports one of `modes`
    /// (case-insensitive), for example to keep looping in an `execute` mode
    /// but stay interactive while planning. Errors when `modes` is empty.
    pub fn looping_modes<S: Into<String>>(
        mut self,
        mode_provider: &AgentModeProvider,
        modes: impl IntoIterator<Item = S>,
    ) -> Result<Self> {
        let modes: Vec<String> = modes
            .into_iter()
            .map(|m| m.into().trim().to_lowercase())
            .collect();
        if modes.is_empty() {
            return Err(Error::Configuration(
                "looping_modes must be a non-empty sequence of mode names.".into(),
            ));
        }
        self.modes = Some((mode_provider.clone(), modes));
        Ok(self)
    }
}

#[async_trait]
impl LoopCallback<LoopDecision> for TodosRemaining {
    async fn call(&self, ctx: &LoopContext) -> Result<LoopDecision> {
        if let Some((provider, modes)) = &self.modes {
            let current = provider.mode(&ctx.session)?;
            if !modes.contains(&current.trim().to_lowercase()) {
                return Ok(LoopDecision::stop());
            }
        }
        let open = self.todos.remaining(&ctx.session).await?;
        Ok(LoopDecision::from(!open.is_empty()))
    }
}

/// A next-message callback listing a [`TodoProvider`]'s open items; see
/// [`todos_remaining_message`].
#[derive(Debug, Clone)]
pub struct TodosRemainingMessage {
    todos: TodoProvider,
}

/// A next-message callback that reminds the agent which todos are still
/// open, to pair with [`todos_remaining`]. Yields `None` (the loop's default
/// handling) when nothing is open.
pub fn todos_remaining_message(todos: &TodoProvider) -> TodosRemainingMessage {
    TodosRemainingMessage {
        todos: todos.clone(),
    }
}

#[async_trait]
impl LoopCallback<Option<Vec<Message>>> for TodosRemainingMessage {
    async fn call(&self, ctx: &LoopContext) -> Result<Option<Vec<Message>>> {
        let open = self.todos.remaining(&ctx.session).await?;
        if open.is_empty() {
            return Ok(None);
        }
        let lines: Vec<String> = open.iter().map(|i| format!("- {}", i.title)).collect();
        Ok(Some(vec![Message::user(format!(
            "You still have {} open todo item(s) that must be addressed before you can \
             finish:\n{}\n\nContinue working through them now. Mark each todo complete as you \
             finish it, and only stop once every todo item is complete.",
            open.len(),
            lines.join("\n")
        ))]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(session: AgentSession) -> LoopContext {
        LoopContext {
            iteration: 1,
            last_result: AgentResponse::default(),
            messages: vec![],
            original_messages: vec![],
            session,
            progress: vec![],
            feedback: None,
        }
    }

    #[test]
    fn decisions_convert_from_bools_and_tuples() {
        assert_eq!(LoopDecision::from(true), LoopDecision::run_again());
        assert_eq!(
            LoopDecision::from((false, Some("why".to_string()))),
            LoopDecision::stop().with_feedback("why")
        );
    }

    #[test]
    fn judge_renders_criteria_into_instructions() {
        struct Never;
        #[async_trait]
        impl ChatClient for Never {
            async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
                unreachable!()
            }
            async fn get_streaming_response(
                &self,
                _: Vec<Message>,
                _: ChatOptions,
            ) -> Result<crate::client::ChatStream> {
                unreachable!()
            }
        }
        let judge = Judge::new(Arc::new(Never));
        assert!(!judge.rendered_instructions().contains(CRITERIA_PLACEHOLDER));
        assert!(judge.agent_instructions().is_none());

        let judge = judge.criteria(["cite sources", "be brief"]);
        let rendered = judge.rendered_instructions();
        assert!(rendered.ends_with(
            "\n\nThe response must satisfy all of the following criteria:\n- cite sources\n- be brief"
        ));
        assert_eq!(
            judge.agent_instructions().unwrap(),
            "Your response must satisfy all of the following criteria:\n- cite sources\n- be brief"
        );

        let custom = judge.instructions("Judge strictly.{{criteria}} End.");
        assert_eq!(
            custom.rendered_instructions(),
            "Judge strictly.\n\nThe response must satisfy all of the following criteria:\n\
             - cite sources\n- be brief End."
        );
    }

    #[test]
    fn judge_feedback_message_relays_feedback() {
        let mut c = ctx(AgentSession::new());
        assert_eq!(
            judge_feedback_message(&c).unwrap()[0].text(),
            DEFAULT_NEXT_MESSAGE
        );
        c.feedback = Some("missing the summary".into());
        assert!(judge_feedback_message(&c).unwrap()[0]
            .text()
            .contains("Evaluator feedback: missing the summary"));
    }

    #[tokio::test]
    async fn todos_remaining_reflects_store_state_and_modes() {
        let todos = TodoProvider::new();
        let modes = AgentModeProvider::new();
        let session = AgentSession::new();
        let c = ctx(session.clone());
        let condition = todos_remaining(&todos);
        assert!(!condition.call(&c).await.unwrap().should_continue);
        assert!(todos_remaining_message(&todos)
            .call(&c)
            .await
            .unwrap()
            .is_none());

        session.state.insert(
            "todo",
            json!({"items": [
                {"id": 1, "title": "Open", "description": null, "is_complete": false},
                {"id": 2, "title": "Done", "description": null, "is_complete": true}
            ], "next_id": 3}),
        );
        assert!(condition.call(&c).await.unwrap().should_continue);
        let message = todos_remaining_message(&todos)
            .call(&c)
            .await
            .unwrap()
            .unwrap();
        let text = message[0].text();
        assert!(text.starts_with("You still have 1 open todo item(s)"));
        assert!(text.contains("- Open") && !text.contains("- Done"));

        // Mode gating: only loop in execute mode.
        let gated = todos_remaining(&todos)
            .looping_modes(&modes, ["Execute"])
            .unwrap();
        assert!(!gated.call(&c).await.unwrap().should_continue);
        modes.set_mode(&session, "execute").unwrap();
        assert!(gated.call(&c).await.unwrap().should_continue);

        assert!(todos_remaining(&todos)
            .looping_modes(&modes, Vec::<String>::new())
            .is_err());
    }
}
