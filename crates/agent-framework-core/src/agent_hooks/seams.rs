//! The enforcement seams: the agent decorator, the output agent middleware,
//! the model-call chat-client decorator and the tool function middleware,
//! sharing one per-run state through a task-local (the counterpart of
//! upstream's `ContextVar` / .NET's `AsyncLocal`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{Map, Value};

use super::codecs;
use super::protocol::{
    EnforcementMode, InterceptionBlocked, InterceptionContextBuilder, InterceptionEmitter,
    InterceptionPoint, Interceptor, RecordSink, ShutdownReason, DEFAULT_TIMEOUT,
};
use crate::agent::{Agent, AgentBuilder, AgentRunOptions, AgentRunStream, SupportsAgentRun};
use crate::client::{ChatClient, ChatStream};
use crate::error::{Error, Result};
use crate::middleware::{AgentContext, FunctionInvocationContext, Middleware, Next};
use crate::session::AgentSession;
use crate::types::{
    AgentResponse, AgentResponseUpdate, ChatOptions, ChatResponse, ChatResponseUpdate, Content,
    Message, UsageContent,
};

/// The `agent.framework` value this host reports.
const FRAMEWORK: &str = "agent-framework";

tokio::task_local! {
    static RUN_STATE: Arc<RunState>;
}

/// Options for [`AgentHooks::new`]: the interceptors and how they are run.
///
/// Mirrors .NET's `AgentHooksOptions`. At least one interceptor is required.
#[derive(Clone, Default)]
pub struct AgentHooksOptions {
    interceptors: Vec<(Option<String>, Arc<dyn Interceptor>)>,
    mode: EnforcementMode,
    timeout: Option<Option<Duration>>,
    record_sink: Option<RecordSink>,
}

impl std::fmt::Debug for AgentHooksOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHooksOptions")
            .field("interceptors", &self.interceptors.len())
            .field("mode", &self.mode)
            .field("timeout", &self.timeout)
            .field("record_sink", &self.record_sink.is_some())
            .finish()
    }
}

impl AgentHooksOptions {
    /// Empty options (`enforce` mode, the 5-second default timeout).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an interceptor. Interceptors run in registration order.
    pub fn interceptor(mut self, interceptor: Arc<dyn Interceptor>) -> Self {
        self.interceptors.push((None, interceptor));
        self
    }

    /// Register an interceptor under a payload-free name recorded on the
    /// interception records.
    pub fn named_interceptor(
        mut self,
        name: impl Into<String>,
        interceptor: Arc<dyn Interceptor>,
    ) -> Self {
        self.interceptors.push((Some(name.into()), interceptor));
        self
    }

    /// Enforce verdicts (default) or only record them.
    pub fn mode(mut self, mode: EnforcementMode) -> Self {
        self.mode = mode;
        self
    }

    /// The per-interceptor timeout; `None` disables it.
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// A callback receiving every interception record.
    pub fn record_sink(mut self, sink: RecordSink) -> Self {
        self.record_sink = Some(sink);
        self
    }
}

/// What one [`AgentHooks`] bundle was configured with. Its `Arc` identity is
/// the ownership token: a seam only binds to a run state created by its own
/// bundle, so nested guarded agents can never misroute emissions.
struct Config {
    options: AgentHooksOptions,
    /// A host-owned session: when set, every run emits on this pair and the
    /// host owns `agent_startup` / `agent_shutdown`.
    host: Option<(Arc<InterceptionEmitter>, Arc<InterceptionContextBuilder>)>,
}

/// Why a run must not egress, recorded by a seam that cannot fail the run
/// directly.
#[derive(Clone)]
enum Halt {
    Blocked(Box<InterceptionBlocked>),
    Failure(String),
}

/// Per-run enforcement state shared by the seams.
struct RunState {
    emitter: Arc<InterceptionEmitter>,
    builder: Arc<InterceptionContextBuilder>,
    session_scoped: bool,
    config: Arc<Config>,
    halted: Mutex<Option<Halt>>,
    /// Tool invocations in flight, keyed by the token the outer tool seam
    /// puts in the invocation's metadata; the inner seam records there
    /// whether (and with which arguments) the call was dispatched.
    tool_calls: Mutex<HashMap<String, ToolTrack>>,
}

/// What the inner tool seam observed for one invocation.
enum ToolTrack {
    /// The inner seam has not run (yet): no `pre_tool_call` was emitted.
    Pending,
    /// `pre_tool_call` blocked the call; no `post_tool_call` follows (§6.2).
    Blocked,
    /// The call was dispatched with these (post-transform) arguments.
    Dispatched(Map<String, Value>),
}

impl RunState {
    /// Record a halt (the first one wins) and return the fail-closed error
    /// that carries it out of the function-invocation loop.
    fn halt(&self, halt: Halt) -> Error {
        let message = match &halt {
            Halt::Blocked(b) => format!("agent-hooks halted the run: {b}"),
            Halt::Failure(m) => m.clone(),
        };
        self.halted.lock().unwrap().get_or_insert(halt);
        Error::middleware_failure(message)
    }

    /// The error a halted run surfaces at the run boundary.
    fn halt_error(&self) -> Option<Error> {
        self.halted.lock().unwrap().as_ref().map(|h| match h {
            Halt::Blocked(b) => Error::InterceptionBlocked(b.clone()),
            Halt::Failure(m) => Error::middleware_failure(m.clone()),
        })
    }

    /// Record a host projection failure (§10.3) and return the error that
    /// fails the guarded action closed.
    fn projection_failure(&self, point: InterceptionPoint, error: Error) -> Error {
        self.emitter.record_host_failure(
            point,
            "projection failed",
            self.builder.session_id(),
            Some(self.builder.next_sequence()),
        );
        error
    }
}

fn blocked(b: Box<InterceptionBlocked>) -> Error {
    Error::InterceptionBlocked(b)
}

/// The run state for `config`, failing closed when there is none (a seam
/// used outside its guarded agent) or it belongs to another bundle.
fn current_state(config: &Arc<Config>, seam: &str) -> Result<Arc<RunState>> {
    let state = RUN_STATE.try_with(Arc::clone).map_err(|_| {
        Error::middleware_failure(format!(
            "the agent-hooks {seam} seam was invoked without an active agent-hooks run; \
             the seams are installed as one unit by AgentHooks::build_agent"
        ))
    })?;
    if !Arc::ptr_eq(&state.config, config) {
        return Err(Error::middleware_failure(format!(
            "the agent-hooks {seam} seam found an active run owned by a different \
             agent-hooks bundle; install exactly one bundle per agent"
        )));
    }
    Ok(state)
}

/// The AGENT-HOOKS-0.1 enforcement bundle.
///
/// One indivisible unit: [`AgentHooks::build_agent`] installs every seam
/// together on an [`AgentBuilder`] and returns the guarded
/// [`AgentHooksAgent`]. The seams themselves are private, so a partial
/// install (which would enforce only part of the contract) is impossible by
/// construction.
pub struct AgentHooks {
    config: Arc<Config>,
}

impl std::fmt::Debug for AgentHooks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHooks")
            .field("options", &self.config.options)
            .field("host_owned_session", &self.config.host.is_some())
            .finish()
    }
}

impl AgentHooks {
    /// A bundle running one agent-hooks session per agent run: a fresh
    /// emitter and session id per run, with `agent_startup` and
    /// `agent_shutdown` bracketing it.
    ///
    /// Errors when `options` has no interceptor: an emitter with none denies
    /// every emission, so the agent could never run.
    pub fn new(options: AgentHooksOptions) -> Result<Self> {
        if options.interceptors.is_empty() {
            return Err(Error::Configuration(
                "agent-hooks enforcement requires at least one interceptor (an emitter with \
                 none fails closed on every emission); register an explicit allow-all \
                 interceptor for a deliberate passthrough"
                    .into(),
            ));
        }
        Ok(Self {
            config: Arc::new(Config {
                options,
                host: None,
            }),
        })
    }

    /// A bundle bound to a host-owned session: every run emits on `emitter`
    /// with contexts from `builder`, only the per-run points (`input`
    /// through `output`) are emitted, and the host emits `agent_startup` and
    /// `agent_shutdown` itself.
    pub fn from_emitter(
        emitter: Arc<InterceptionEmitter>,
        builder: Arc<InterceptionContextBuilder>,
    ) -> Self {
        Self {
            config: Arc::new(Config {
                options: AgentHooksOptions::default(),
                host: Some((emitter, builder)),
            }),
        }
    }

    /// Install every seam on `builder` and build the guarded agent.
    ///
    /// The bundle's agent middleware and the `post_tool_call` half of its
    /// function seam go first in their lists, whatever was added before:
    /// middleware outside the bundle would run outside the enforcement
    /// boundary. The `pre_tool_call` half goes last, directly around the
    /// tool, so it judges the arguments the tool actually receives. Its
    /// model-call seam wraps the
    /// supplied chat client directly, below the function-invocation loop,
    /// so every model service call is bracketed individually.
    pub fn build_agent(self, builder: AgentBuilder) -> AgentHooksAgent {
        let config = self.config;
        let chat_config = config.clone();
        let (inner, tool_names) = builder.install_agent_hooks(
            Arc::new(OutputMiddleware {
                config: config.clone(),
            }),
            Arc::new(ToolPostMiddleware {
                config: config.clone(),
            }),
            Arc::new(ToolPreMiddleware {
                config: config.clone(),
            }),
            move |client| {
                Arc::new(GuardedChatClient {
                    inner: client,
                    config: chat_config,
                })
            },
        );
        AgentHooksAgent {
            inner,
            config,
            tool_names,
        }
    }
}

/// An [`Agent`] guarded by an [`AgentHooks`] bundle.
///
/// Emits `agent_startup` and `input` before the inner run starts (before
/// context providers load history, so `input` sees exactly the caller's
/// request and a transformed input is what history records), and
/// `agent_shutdown` after it ends. `output` is emitted from inside the inner
/// run, before history is persisted. Streaming runs are fully buffered:
/// nothing is released before the `output` verdict.
pub struct AgentHooksAgent {
    inner: Agent,
    config: Arc<Config>,
    /// The agent's own tools, for `agent_startup.tools_registered`.
    tool_names: Vec<String>,
}

impl std::fmt::Debug for AgentHooksAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHooksAgent")
            .field("id", &self.inner.id())
            .field("name", &self.inner.name())
            .finish()
    }
}

impl AgentHooksAgent {
    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.inner.description()
    }

    fn new_run_state(&self) -> Arc<RunState> {
        let config = &self.config;
        let (emitter, builder, session_scoped) = match &config.host {
            Some((emitter, builder)) => (emitter.clone(), builder.clone(), true),
            None => {
                let options = &config.options;
                let mut emitter = InterceptionEmitter::new()
                    .with_mode(options.mode)
                    .with_timeout(options.timeout.unwrap_or(Some(DEFAULT_TIMEOUT)));
                if let Some(sink) = &options.record_sink {
                    emitter = emitter.with_record_sink(sink.clone());
                }
                for (name, interceptor) in &options.interceptors {
                    emitter = emitter.register(interceptor.clone(), name.clone());
                }
                let mut builder = InterceptionContextBuilder::new(
                    self.inner.id(),
                    FRAMEWORK,
                    uuid::Uuid::new_v4().simple().to_string(),
                );
                if let Some(name) = self.inner.name() {
                    builder = builder.with_agent_name(name);
                }
                (Arc::new(emitter), Arc::new(builder), false)
            }
        };
        Arc::new(RunState {
            emitter,
            builder,
            session_scoped,
            config: config.clone(),
            halted: Mutex::new(None),
            tool_calls: Mutex::new(HashMap::new()),
        })
    }

    /// The run-start tool snapshot: the agent's tools plus this run's.
    /// Tools added later (context providers, tool sources) appear in each
    /// `pre_model_call`'s `tools`.
    fn tools_registered(&self, options: &AgentRunOptions) -> Vec<String> {
        let mut names = self.tool_names.clone();
        let run_tools = options
            .chat_options
            .iter()
            .flat_map(|o| o.tools.iter())
            .chain(options.additional_tools.iter());
        for tool in run_tools {
            if !names.contains(&tool.name) {
                names.push(tool.name.clone());
            }
        }
        names
    }

    /// Emit `agent_startup` (per-run sessions) and `input`; apply input
    /// transforms to `messages`.
    async fn start(
        &self,
        state: &RunState,
        messages: &mut Vec<Message>,
        options: &AgentRunOptions,
    ) -> Result<()> {
        if !state.session_scoped {
            state
                .emitter
                .emit(state.builder.agent_startup(self.tools_registered(options)))
                .await
                .map_err(blocked)?;
        }
        let (content, role) = codecs::input_to_wire(messages)
            .map_err(|e| state.projection_failure(InterceptionPoint::Input, e))?;
        let context = state.builder.input(content, role);
        let before = context.target().clone();
        let outcome = state.emitter.emit(context).await.map_err(blocked)?;
        codecs::input_write_back(messages, &before, &outcome.target)
    }

    /// Surface any halt, then emit `agent_shutdown` (per-run sessions).
    async fn finish<T>(&self, state: &RunState, result: Result<T>) -> Result<T> {
        let result = match state.halt_error() {
            Some(halt) => Err(halt),
            None => result,
        };
        if !state.session_scoped {
            let reason = if result.is_ok() {
                ShutdownReason::Completed
            } else {
                ShutdownReason::Error
            };
            // Best-effort trail closure; a block here is record-only (§6.1a).
            state
                .emitter
                .emit_unchecked(state.builder.agent_shutdown(reason))
                .await;
        }
        result
    }
}

#[async_trait]
impl SupportsAgentRun for AgentHooksAgent {
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
        mut messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let state = self.new_run_state();
        RUN_STATE
            .scope(state.clone(), async {
                let result = async {
                    self.start(&state, &mut messages, &options).await?;
                    self.inner
                        .run_with_options(messages, session, options)
                        .await
                }
                .await;
                self.finish(&state, result).await
            })
            .await
    }

    async fn run_stream(
        &self,
        mut messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let options = options.unwrap_or_default();
        let state = self.new_run_state();
        RUN_STATE
            .scope(state.clone(), async {
                let result = async {
                    self.start(&state, &mut messages, &options).await?;
                    // The inner agent always has agent middleware (the
                    // bundle's), so its stream replays an already-verdicted
                    // response; draining it here keeps everything inside the
                    // run state and releases nothing before the verdict.
                    let stream =
                        SupportsAgentRun::run_stream(&self.inner, messages, session, Some(options))
                            .await?;
                    stream
                        .collect::<Vec<_>>()
                        .await
                        .into_iter()
                        .collect::<Result<Vec<AgentResponseUpdate>>>()
                }
                .await;
                let updates = self.finish(&state, result).await?;
                Ok(futures::stream::iter(updates.into_iter().map(Ok)).boxed())
            })
            .await
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

/// Agent seam: `output`, emitted over the run's final response after every
/// inner middleware (so a substituted or short-circuited result is guarded
/// too) and before the agent persists history.
struct OutputMiddleware {
    config: Arc<Config>,
}

#[async_trait]
impl Middleware<AgentContext> for OutputMiddleware {
    async fn process(&self, ctx: AgentContext, next: Next<AgentContext>) -> Result<AgentContext> {
        let state = current_state(&self.config, "agent")?;
        let mut ctx = next.run(ctx).await?;
        if let Some(halt) = state.halt_error() {
            // Something above the tool loop swallowed a halt: still nothing
            // egresses or persists.
            return Err(halt);
        }
        if let Some(response) = ctx.result.as_mut() {
            let before = codecs::output_to_wire(response)
                .map_err(|e| state.projection_failure(InterceptionPoint::Output, e))?;
            let outcome = state
                .emitter
                .emit(state.builder.output(before.clone()))
                .await
                .map_err(blocked)?;
            codecs::output_write_back(response, &before, &outcome.target)?;
        }
        Ok(ctx)
    }
}

/// Chat seam: `pre_model_call` / `post_model_call` around each model service
/// call. Wraps the supplied client below the function-invocation loop, so a
/// denied response never reaches the loop (its tool calls never run).
struct GuardedChatClient {
    inner: Arc<dyn ChatClient>,
    config: Arc<Config>,
}

impl GuardedChatClient {
    fn model_id(&self, options: &ChatOptions) -> String {
        options
            .model
            .clone()
            .or_else(|| self.inner.model().map(str::to_string))
            .unwrap_or_else(|| "unknown".to_string())
    }

    async fn pre(
        &self,
        state: &RunState,
        model_id: &str,
        messages: Vec<Message>,
        options: &ChatOptions,
    ) -> Result<Vec<Message>> {
        let before = codecs::request_to_wire(&messages)
            .map_err(|e| state.projection_failure(InterceptionPoint::PreModelCall, e))?;
        let context = state.builder.pre_model_call(
            model_id,
            before.clone(),
            codecs::tools_to_wire(&options.tools),
        );
        let outcome = state.emitter.emit(context).await.map_err(blocked)?;
        codecs::request_write_back(messages, &before, &outcome.target)
    }

    async fn post(
        &self,
        state: &RunState,
        model_id: &str,
        response: &mut ChatResponse,
    ) -> Result<bool> {
        let before = codecs::response_to_wire(response)
            .map_err(|e| state.projection_failure(InterceptionPoint::PostModelCall, e))?;
        let tool_calls = before["tool_calls"].as_array().cloned().unwrap_or_default();
        let context = state.builder.post_model_call(
            response.model.as_deref().unwrap_or(model_id),
            before["content"].clone(),
            tool_calls,
            before["finish_reason"].as_str().unwrap_or("stop"),
            codecs::usage_to_wire(response.usage_details.as_ref()),
        );
        let outcome = state.emitter.emit(context).await.map_err(blocked)?;
        codecs::response_write_back(response, &before, &outcome.target)
    }
}

#[async_trait]
impl ChatClient for GuardedChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let state = current_state(&self.config, "chat")?;
        let model_id = self.model_id(&options);
        let messages = self.pre(&state, &model_id, messages, &options).await?;
        let mut response = self.inner.get_response(messages, options).await?;
        self.post(&state, &model_id, &mut response).await?;
        Ok(response)
    }

    /// Fail-closed by buffering (§12.1): the model stream is fully consumed,
    /// `post_model_call` judges the assembled response, and only then are
    /// updates released — the buffered ones when untouched, otherwise ones
    /// re-derived from the verdicted response.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let state = current_state(&self.config, "chat")?;
        let model_id = self.model_id(&options);
        let messages = self.pre(&state, &model_id, messages, &options).await?;
        let stream = self.inner.get_streaming_response(messages, options).await?;
        let buffered = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<ChatResponseUpdate>>>()?;
        let mut response = ChatResponse::from_updates(buffered.clone());
        let changed = self.post(&state, &model_id, &mut response).await?;
        let updates = if changed {
            chat_response_to_updates(response)
        } else {
            buffered
        };
        Ok(futures::stream::iter(updates.into_iter().map(Ok)).boxed())
    }

    fn model(&self) -> Option<&str> {
        self.inner.model()
    }
}

/// Re-derive stream updates from a (transformed) response: one per message,
/// with the response-level metadata (ids, model, `created_at`) on each and
/// the finish reason, usage, continuation token and `additional_properties`
/// on the last, so [`ChatResponse::from_updates`] reassembles the same
/// metadata the buffered stream carried.
fn chat_response_to_updates(response: ChatResponse) -> Vec<ChatResponseUpdate> {
    let last = response.messages.len().saturating_sub(1);
    let mut updates: Vec<ChatResponseUpdate> = response
        .messages
        .into_iter()
        .enumerate()
        .map(|(i, m)| ChatResponseUpdate {
            contents: m.contents,
            role: Some(m.role),
            author_name: m.author_name,
            message_id: Some(m.message_id.unwrap_or_else(|| format!("msg-{i}"))),
            response_id: response.response_id.clone(),
            conversation_id: response.conversation_id.clone(),
            model: response.model.clone(),
            created_at: response.created_at.clone(),
            ..Default::default()
        })
        .collect();
    if updates.is_empty() {
        updates.push(ChatResponseUpdate {
            role: Some(crate::types::Role::assistant()),
            response_id: response.response_id.clone(),
            conversation_id: response.conversation_id.clone(),
            model: response.model.clone(),
            created_at: response.created_at.clone(),
            ..Default::default()
        });
    }
    let tail_index = last.min(updates.len() - 1);
    let tail = &mut updates[tail_index];
    tail.finish_reason = response.finish_reason;
    tail.continuation_token = response.continuation_token;
    tail.additional_properties = response.additional_properties;
    if let Some(details) = response.usage_details {
        tail.contents.push(Content::Usage(UsageContent { details }));
    }
    updates
}

/// The metadata key carrying the outer tool seam's per-invocation token.
const TOOL_TOKEN_KEY: &str = "agent_hooks.invocation";

/// Function seam, outer half: `post_tool_call`.
///
/// The function seam is two function middleware. [`ToolPreMiddleware`] is
/// **last** in the agent's list, directly around the tool, so
/// `pre_tool_call` judges (and its transform rewrites) the arguments the
/// tool actually receives, after every other middleware has rewritten them.
/// This one is **first**, so `post_tool_call` judges the result the function
/// loop actually uses, after every other middleware has rewritten it, and
/// reports the arguments the inner half saw dispatched (§4.2).
///
/// A policy deny blocks the call (the tool is not run, or its result is
/// discarded) and fails it with [`Error::ToolRejected`] carrying the
/// blocked-call payload, so the loop continues as if the call had failed
/// (§6.1-§6.2): it is an error result, counts toward the consecutive-error
/// limit, and the model sees the payload whatever `include_detailed_errors`
/// says. A `host_error:*` deny, or a failure of this seam itself,
/// halts the run instead: the loop absorbs ordinary errors into tool results
/// (fail open for an enforcement failure), so the seam returns
/// [`Error::MiddlewareFailure`], the loop's one fail-closed escape, and the
/// agent decorator surfaces the recorded block at the run boundary.
///
/// A middleware between the halves that answers without calling on reaches
/// no tool: nothing is emitted for it here, and its result reaches the model
/// through the next `pre_model_call`.
struct ToolPostMiddleware {
    config: Arc<Config>,
}

/// Function seam, inner half: `pre_tool_call`; see [`ToolPostMiddleware`].
struct ToolPreMiddleware {
    config: Arc<Config>,
}

/// The model's call id for an invocation (a fresh one when it has none).
fn invocation_call_id(ctx: &FunctionInvocationContext) -> String {
    ctx.metadata
        .get("call_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string())
}

/// The tool-error payload a policy deny hands the model.
fn blocked_payload(b: &InterceptionBlocked) -> Value {
    let mut payload = serde_json::json!({
        "error": format!("Tool call blocked by agent-hooks at {}.", b.point()),
        "reason": b.reason().unwrap_or("deny"),
    });
    if let Some(message) = &b.verdict.message {
        payload["message"] = Value::String(message.clone());
    }
    payload
}

/// Enforce a tool-seam deny: a host error halts the run; a policy deny fails
/// the call with the blocked-call payload as its model-visible error.
fn block_tool(state: &RunState, b: Box<InterceptionBlocked>) -> Error {
    if b.is_host_error() {
        return state.halt(Halt::Blocked(b));
    }
    Error::ToolRejected(blocked_payload(&b).to_string())
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for ToolPostMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let state = current_state(&self.config, "function")?;
        let call_id = invocation_call_id(&ctx);
        // The inner half reports the same id even when the model gave none.
        ctx.metadata
            .insert("call_id".into(), Value::String(call_id.clone()));
        let token = uuid::Uuid::new_v4().simple().to_string();
        state
            .tool_calls
            .lock()
            .unwrap()
            .insert(token.clone(), ToolTrack::Pending);
        ctx.metadata
            .insert(TOOL_TOKEN_KEY.into(), Value::String(token.clone()));
        let name = ctx.function_name.clone();

        let result = next.run(ctx).await;
        let track = state.tool_calls.lock().unwrap().remove(&token);
        let args = match track {
            Some(ToolTrack::Dispatched(args)) => args,
            // Blocked at pre_tool_call (§6.2: no post_tool_call), or no tool
            // was reached at all.
            Some(ToolTrack::Blocked | ToolTrack::Pending) | None => {
                return result.map(|mut ctx| {
                    ctx.metadata.remove(TOOL_TOKEN_KEY);
                    ctx
                });
            }
        };

        match result {
            Err(error) => {
                // The invocation was dispatched and errored; the contract
                // still brackets it (§3), whatever kind of error it was.
                let value = Value::String(crate::observability::error_type(&error));
                let post = state
                    .builder
                    .post_tool_call(&call_id, &name, args, value.clone(), true);
                let emitted = state.emitter.emit(post).await;
                // A halt (the executor's, or another fail-closed
                // middleware's) stays a halt: the post hook observes it but
                // cannot turn it back into a model-facing result.
                if error.is_middleware_failure() {
                    return Err(error);
                }
                match emitted {
                    // A transform rewrites the error the loop hands the
                    // model; it is the interceptor's model-facing text, so
                    // it reaches the model even without detailed errors.
                    Ok(outcome) if outcome.target != value => {
                        let text = match outcome.target {
                            Value::String(s) => s,
                            other => other.to_string(),
                        };
                        Err(Error::ToolRejected(text))
                    }
                    Ok(_) => Err(error),
                    // A host error halts the run; a policy deny over an
                    // errored call changes nothing.
                    Err(b) if b.is_host_error() => Err(state.halt(Halt::Blocked(b))),
                    Err(_) => Err(error),
                }
            }
            Ok(mut ctx) => {
                ctx.metadata.remove(TOOL_TOKEN_KEY);
                let value = ctx.result.clone().unwrap_or(Value::Null);
                let post =
                    state
                        .builder
                        .post_tool_call(&call_id, &name, args, value.clone(), false);
                match state.emitter.emit(post).await {
                    Ok(outcome) => {
                        if outcome.target != value {
                            ctx.result = Some(outcome.target);
                        }
                        Ok(ctx)
                    }
                    // §6.1: the result is discarded as if the call errored.
                    Err(b) => Err(block_tool(&state, b)),
                }
            }
        }
    }
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for ToolPreMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let state = current_state(&self.config, "function")?;
        // The outer half's token. A middleware that dropped it, or an
        // invocation that bypassed the outer half, leaves the call
        // unbracketed: fail closed.
        let token = match ctx.metadata.remove(TOOL_TOKEN_KEY) {
            Some(Value::String(token)) if state.tool_calls.lock().unwrap().contains_key(&token) => {
                token
            }
            _ => {
                return Err(state.halt(Halt::Failure(
                    "agent-hooks: tool invocation not bracketed by the tool seam".into(),
                )));
            }
        };
        let call_id = invocation_call_id(&ctx);
        let name = ctx.function_name.clone();
        let mut args = codecs::tool_args_to_wire(&ctx.arguments);
        let pre = state.builder.pre_tool_call(&call_id, &name, args.clone());
        match state.emitter.emit(pre).await {
            Ok(outcome) => match codecs::tool_args_write_back(&args, &outcome.target) {
                Ok(Some(rewritten)) => {
                    ctx.arguments = Value::Object(rewritten.clone());
                    args = rewritten;
                }
                Ok(None) => {}
                Err(e) => return Err(state.halt(Halt::Failure(e.to_string()))),
            },
            // §6.2: not dispatched, and no post_tool_call.
            Err(b) => {
                state
                    .tool_calls
                    .lock()
                    .unwrap()
                    .insert(token, ToolTrack::Blocked);
                return Err(block_tool(&state, b));
            }
        }
        state
            .tool_calls
            .lock()
            .unwrap()
            .insert(token, ToolTrack::Dispatched(args));
        next.run(ctx).await
    }
}

impl AgentBuilder {
    /// Build the agent guarded by `hooks`; shorthand for
    /// [`AgentHooks::build_agent`].
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental-agent-hooks")))]
    pub fn build_with_agent_hooks(self, hooks: AgentHooks) -> AgentHooksAgent {
        hooks.build_agent(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rederived_updates_keep_response_metadata() {
        let mut response = ChatResponse::from_text("transformed");
        response.created_at = Some("2026-10-07T00:00:00Z".into());
        response
            .additional_properties
            .insert("provider".into(), Value::String("p".into()));
        let back = ChatResponse::from_updates(chat_response_to_updates(response.clone()));
        assert_eq!(back.created_at, response.created_at);
        assert_eq!(back.additional_properties, response.additional_properties);
    }

    struct Unreachable;

    #[async_trait]
    impl ChatClient for Unreachable {
        async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            panic!("the guarded client must not be reached")
        }
        async fn get_streaming_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatStream> {
            panic!("the guarded client must not be reached")
        }
    }

    fn config() -> Arc<Config> {
        Arc::new(Config {
            options: AgentHooksOptions::default(),
            host: None,
        })
    }

    fn run_state(config: Arc<Config>) -> Arc<RunState> {
        Arc::new(RunState {
            emitter: Arc::new(InterceptionEmitter::new()),
            builder: Arc::new(InterceptionContextBuilder::new("a", FRAMEWORK, "s")),
            session_scoped: false,
            config,
            halted: Mutex::new(None),
            tool_calls: Mutex::new(HashMap::new()),
        })
    }

    #[tokio::test]
    async fn seams_without_their_run_fail_closed() {
        let client = GuardedChatClient {
            inner: Arc::new(Unreachable),
            config: config(),
        };
        // No run at all.
        let err = client
            .get_response(vec![Message::user("x")], ChatOptions::new())
            .await
            .unwrap_err();
        assert!(err.is_middleware_failure(), "{err}");
        // A run owned by another bundle.
        let err = RUN_STATE
            .scope(
                run_state(config()),
                client.get_response(vec![Message::user("x")], ChatOptions::new()),
            )
            .await
            .unwrap_err();
        assert!(err.is_middleware_failure(), "{err}");
        assert!(err.to_string().contains("different"), "{err}");
    }

    #[test]
    fn rederived_updates_carry_response_metadata() {
        let response = ChatResponse {
            messages: vec![Message::assistant("a"), Message::assistant("b")],
            response_id: Some("r".into()),
            finish_reason: Some(crate::types::FinishReason::stop()),
            ..Default::default()
        };
        let updates = chat_response_to_updates(response);
        assert_eq!(updates.len(), 2);
        assert!(updates[0].finish_reason.is_none());
        assert_eq!(updates[1].finish_reason.as_ref().unwrap().as_str(), "stop");
        let back = ChatResponse::from_updates(updates);
        let texts: Vec<String> = back.messages.iter().map(Message::text).collect();
        assert_eq!(texts, vec!["a", "b"]);
        assert_eq!(back.response_id.as_deref(), Some("r"));
    }

    /// Streams `"part one"` + `" part two"`.
    struct TwoChunks;

    #[async_trait]
    impl ChatClient for TwoChunks {
        async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse::from_text("part one part two"))
        }
        async fn get_streaming_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatStream> {
            let chunk = |t: &str| {
                Ok(ChatResponseUpdate {
                    contents: vec![Content::text(t)],
                    role: Some(crate::types::Role::assistant()),
                    message_id: Some("m".into()),
                    ..Default::default()
                })
            };
            Ok(futures::stream::iter(vec![chunk("part one"), chunk(" part two")]).boxed())
        }
    }

    async fn stream_with(
        verdict: super::super::protocol::Verdict,
    ) -> Result<Vec<ChatResponseUpdate>> {
        let config = config();
        let emitter = InterceptionEmitter::new().register(
            Arc::new(super::super::protocol::interceptor_fn(move |ctx| {
                let verdict = verdict.clone();
                async move {
                    Ok(if ctx.point() == InterceptionPoint::PostModelCall {
                        verdict
                    } else {
                        super::super::protocol::Verdict::allow()
                    })
                }
            })),
            None,
        );
        let state = Arc::new(RunState {
            emitter: Arc::new(emitter),
            ..Arc::try_unwrap(run_state(config.clone())).ok().unwrap()
        });
        let client = GuardedChatClient {
            inner: Arc::new(TwoChunks),
            config,
        };
        RUN_STATE
            .scope(state, async {
                let stream = client
                    .get_streaming_response(vec![Message::user("x")], ChatOptions::new())
                    .await?;
                stream.collect::<Vec<_>>().await.into_iter().collect()
            })
            .await
    }

    #[tokio::test]
    async fn streamed_model_calls_are_buffered_behind_post_model_call() {
        use super::super::protocol::Verdict;
        // Untouched: the buffered chunks replay as they arrived.
        let updates = stream_with(Verdict::allow()).await.unwrap();
        assert_eq!(updates.len(), 2);
        // Denied: the stream never opens.
        let err = stream_with(Verdict::deny("unsafe")).await.unwrap_err();
        assert_eq!(err.interception_blocked().unwrap().reason(), Some("unsafe"));
        // Transformed: updates are re-derived from the verdicted response.
        let updates = stream_with(Verdict::transform(
            "$target.content",
            serde_json::json!("safe"),
        ))
        .await
        .unwrap();
        assert_eq!(ChatResponse::from_updates(updates).text(), "safe");
    }
}
