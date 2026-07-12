//! Agents: the [`Agent`] trait and the concrete [`ChatAgent`].
//!
//! Rust equivalent of `agent_framework._agents`.

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use tracing::Instrument;
use uuid::Uuid;

use crate::client::{ChatClient, FunctionInvokingChatClient};
use crate::error::{Error, Result};
use crate::memory::{AggregateContextProvider, ContextProvider};
use crate::middleware::{AgentRunContext, ChatContext, MiddlewarePipeline, Terminal};
use crate::threads::{AgentThread, ChatMessageStore, InMemoryChatMessageStore};
use crate::tools::{AiFunction, ToolDefinition, ToolSource};
use crate::types::{
    prepare_messages, AgentRunResponse, AgentRunResponseUpdate, ChatMessage, ChatOptions,
    ChatResponse, IntoMessages, ResponseFormat,
};

/// A boxed stream of agent run updates.
pub type AgentRunStream = Pin<Box<dyn Stream<Item = Result<AgentRunResponseUpdate>> + Send>>;

/// Per-run option overrides for a single [`Agent::run_with_options`] /
/// [`Agent::run_stream`] call, merged over the agent's build-time defaults.
///
/// Mirrors upstream `run`/`run_stream` per-call keyword arguments and .NET
/// `AgentRunOptions`: the per-run [`ChatOptions`] take precedence over the
/// agent's defaults (via [`ChatOptions::merge`], matching Python's
/// `run_chat_options & ChatOptions(...)`), and
/// [`additional_tools`](Self::additional_tools) are appended to the tool list
/// for that call only.
#[derive(Debug, Clone, Default)]
pub struct AgentRunOptions {
    /// Chat-option overrides merged over the agent's defaults (per-run wins).
    pub chat_options: Option<ChatOptions>,
    /// Extra tools available only for this run, appended to the agent's tools.
    ///
    /// Declaration-only tools (no executor) surface their calls back to the
    /// caller instead of being executed locally — this is how a hosting
    /// frontend injects client-side tools (see the AG-UI router).
    pub additional_tools: Vec<ToolDefinition>,
}

impl AgentRunOptions {
    /// Empty options (no overrides).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the per-run chat-option overrides (merged over the agent defaults,
    /// per-run winning).
    pub fn with_chat_options(mut self, options: ChatOptions) -> Self {
        self.chat_options = Some(options);
        self
    }

    /// Append a tool available only for this run.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.additional_tools.push(tool);
        self
    }

    /// Append multiple tools available only for this run.
    pub fn with_tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.additional_tools.extend(tools);
        self
    }

    /// Whether these options carry no overrides at all. Used by the default
    /// [`Agent::run_with_options`] to decide whether to warn about ignoring
    /// options it cannot honor.
    pub fn is_empty(&self) -> bool {
        self.chat_options.is_none() && self.additional_tools.is_empty()
    }
}

/// Map a completed run into buffered agent updates — one update per message,
/// each with a distinct `message_id` so that re-aggregation via
/// [`AgentRunResponse::from_updates`] keeps the message boundaries. Shared by
/// the default [`Agent::run_stream`] and [`ChatAgent`]'s middleware-path
/// replay.
///
/// Response-level metadata survives the replay: `response_id` and
/// `conversation_id` ride on every update, and `usage_details` rides the
/// final update as a [`Content::Usage`] item (aggregation folds it back into
/// [`AgentRunResponse::usage_details`], never into message contents) — the
/// same contract as the tool-loop replay in `FunctionInvokingChatClient`.
pub(crate) fn response_to_updates(
    response: AgentRunResponse,
) -> Vec<Result<AgentRunResponseUpdate>> {
    let AgentRunResponse {
        messages,
        response_id,
        conversation_id,
        usage_details,
        ..
    } = response;
    let last = messages.len().saturating_sub(1);
    let mut updates: Vec<Result<AgentRunResponseUpdate>> = messages
        .into_iter()
        .enumerate()
        .map(|(i, m)| {
            let message_id = m.message_id.clone().or_else(|| Some(format!("msg-{i}")));
            let mut contents = m.contents;
            if i == last {
                if let Some(usage) = usage_details.clone() {
                    contents.push(crate::types::Content::Usage(crate::types::UsageContent {
                        details: usage,
                    }));
                }
            }
            Ok(AgentRunResponseUpdate {
                contents,
                role: Some(m.role),
                author_name: m.author_name,
                message_id,
                response_id: response_id.clone(),
                conversation_id: conversation_id.clone(),
                ..Default::default()
            })
        })
        .collect();
    if updates.is_empty() && (usage_details.is_some() || response_id.is_some()) {
        let contents = usage_details
            .map(|u| {
                vec![crate::types::Content::Usage(crate::types::UsageContent {
                    details: u,
                })]
            })
            .unwrap_or_default();
        updates.push(Ok(AgentRunResponseUpdate {
            contents,
            role: Some(crate::types::Role::assistant()),
            response_id,
            conversation_id,
            ..Default::default()
        }));
    }
    updates
}

/// A factory that builds a fresh [`ChatMessageStore`] for each new local
/// thread, mirroring Python's `chat_message_store_factory`
/// (`_agents.py:1088-1092`). Configured via
/// [`ChatAgentBuilder::chat_message_store_factory`] and used by
/// [`ChatAgent::get_new_thread`] / [`ChatAgent::deserialize_thread`].
pub type ChatMessageStoreFactory = Arc<dyn Fn() -> Arc<dyn ChatMessageStore> + Send + Sync>;

/// Sanitize an agent name into a valid tool/function identifier, mirroring
/// Python's `_sanitize_agent_name` (`_agents.py:53-87`).
///
/// Every character that is not ASCII alphanumeric or `_` is replaced with `_`;
/// runs of `_` are collapsed to one; leading/trailing `_` are trimmed. An
/// all-invalid name (e.g. `"@@@"`) becomes `"agent"`, and a name that would
/// start with a digit is prefixed with `_`. `None` maps to `None`.
fn sanitize_agent_name(agent_name: Option<&str>) -> Option<String> {
    let name = agent_name?;
    let replaced: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    // Collapse consecutive underscores into one.
    let mut collapsed = String::with_capacity(replaced.len());
    let mut prev_underscore = false;
    for c in replaced.chars() {
        if c == '_' {
            if !prev_underscore {
                collapsed.push('_');
            }
            prev_underscore = true;
        } else {
            collapsed.push(c);
            prev_underscore = false;
        }
    }
    let trimmed = collapsed.trim_matches('_');
    if trimmed.is_empty() {
        return Some("agent".to_string());
    }
    let mut result = trimmed.to_string();
    if result.starts_with(|c: char| c.is_ascii_digit()) {
        result.insert(0, '_');
    }
    Some(result)
}

/// The common interface implemented by all agents.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run the agent to completion.
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse>;

    /// Run the agent to completion, applying per-run [`AgentRunOptions`] over
    /// the agent's build-time defaults.
    ///
    /// The default implementation ignores `options` and delegates to
    /// [`Agent::run`], emitting a `tracing::warn!` when non-empty options are
    /// supplied — mirroring upstream agents that silently drop kwargs they do
    /// not understand. Agents that support per-run overrides (notably
    /// [`ChatAgent`]) override this.
    async fn run_with_options(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
        options: AgentRunOptions,
    ) -> Result<AgentRunResponse> {
        if !options.is_empty() {
            tracing::warn!(
                agent = %self.id(),
                "agent does not support per-run options; ignoring them"
            );
        }
        self.run(messages, thread).await
    }

    /// Run the agent and stream incremental [`AgentRunResponseUpdate`]s.
    ///
    /// The default implementation is a **buffered fallback**: it runs to
    /// completion via [`Agent::run_with_options`] and yields the response's
    /// messages as updates. Agents with a real streaming backend (notably
    /// [`ChatAgent`], [`WorkflowAgent`](crate::workflow::WorkflowAgent), and the
    /// A2A client agent) override this to stream incrementally.
    ///
    /// `thread` is taken **by value**: the returned stream owns it and writes
    /// the conversation back once the stream is fully consumed. When the
    /// thread's message store is shared (as [`ChatAgent`]'s is, via `Arc`), the
    /// write-back is observable through a clone taken before streaming.
    async fn run_stream(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<AgentThread>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let mut owned = thread;
        let response = self
            .run_with_options(messages, owned.as_mut(), options.unwrap_or_default())
            .await?;
        Ok(futures::stream::iter(response_to_updates(response)).boxed())
    }

    /// A stable identifier for this agent.
    fn id(&self) -> &str;

    /// The optional human-readable name.
    fn name(&self) -> Option<&str> {
        None
    }

    /// The display name: `name` if set, else `id`.
    fn display_name(&self) -> String {
        self.name()
            .map(str::to_string)
            .unwrap_or_else(|| self.id().to_string())
    }

    /// A fresh thread for a new conversation.
    fn get_new_thread(&self) -> AgentThread {
        AgentThread::new()
    }
}

/// The primary concrete agent: pairs a chat client with instructions, default
/// options, tools, context providers, and middleware.
///
/// Cheaply cloneable (the client, context providers, and middleware are shared
/// via `Arc`), which is what makes [`ChatAgent::as_tool`] possible.
#[derive(Clone)]
pub struct ChatAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_provider: Option<Arc<AggregateContextProvider>>,
    /// Factory for a new local thread's message store. When unset,
    /// [`InMemoryChatMessageStore`] is used.
    chat_message_store_factory: Option<ChatMessageStoreFactory>,
    agent_middleware: MiddlewarePipeline<AgentRunContext>,
    /// Middleware run around the underlying chat-client call (mirrors
    /// Python's `use_chat_middleware`). See [`ChatAgent::call_chat_client`].
    chat_middleware: MiddlewarePipeline<ChatContext>,
    /// Dynamic tool sources (e.g. MCP servers), resolved fresh on every run
    /// and appended after the agent's static/context/per-run tools. See
    /// [`ToolSource`] and [`ChatAgent::prepare_request`].
    tool_sources: Vec<Arc<dyn ToolSource>>,
}

/// Options for [`ChatAgent::as_tool`].
#[derive(Debug, Clone, Default)]
pub struct AsToolOptions {
    /// The tool name. Defaults to the agent's name (else its id).
    pub name: Option<String>,
    /// The tool description. Defaults to the agent's description (else empty).
    pub description: Option<String>,
    /// The single string argument's name. Defaults to `"task"`.
    pub arg_name: Option<String>,
    /// The argument's description. Defaults to `"Task for {tool_name}"`.
    pub arg_description: Option<String>,
}

impl AsToolOptions {
    pub fn new() -> Self {
        Self::default()
    }
    /// Set the tool name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// Set the tool description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    /// Set the argument name (default `"task"`).
    pub fn arg_name(mut self, arg_name: impl Into<String>) -> Self {
        self.arg_name = Some(arg_name.into());
        self
    }
    /// Set the argument description.
    pub fn arg_description(mut self, arg_description: impl Into<String>) -> Self {
        self.arg_description = Some(arg_description.into());
        self
    }
}

impl ChatAgent {
    /// Start building an agent from a chat client. The client is automatically
    /// wrapped with [`FunctionInvokingChatClient`] so local tools are executed.
    pub fn builder(client: impl ChatClient + 'static) -> ChatAgentBuilder {
        ChatAgentBuilder::new(client)
    }

    /// The agent's default instructions.
    pub fn instructions(&self) -> Option<&str> {
        self.chat_options.instructions.as_deref()
    }

    /// Run and stream incremental updates — an ergonomic wrapper over the
    /// object-safe [`Agent::run_stream`] trait method (the real streaming
    /// implementation), accepting `impl IntoMessages`.
    ///
    /// The thread's history is updated when the stream completes; because
    /// message stores are shared via `Arc`, updates are visible on the original
    /// thread once the returned stream is fully consumed. Pass per-run
    /// [`AgentRunOptions`] to override the agent's defaults for this call only.
    pub async fn run_stream(
        &self,
        messages: impl IntoMessages,
        thread: Option<AgentThread>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        Agent::run_stream(self, messages.into_messages(), thread, options).await
    }

    /// Ergonomic streaming run with a fresh thread and no per-run options
    /// (mirrors [`ChatAgent::run_once`]).
    pub async fn run_stream_once(&self, messages: impl IntoMessages) -> Result<AgentRunStream> {
        Agent::run_stream(self, messages.into_messages(), None, None).await
    }

    /// Ergonomic run without an explicit thread.
    pub async fn run_once(&self, messages: impl IntoMessages) -> Result<AgentRunResponse> {
        self.run(messages.into_messages(), None).await
    }

    /// The real streaming implementation, shared by the [`Agent::run_stream`]
    /// trait impl. Kept as an inherent helper so the trait method stays a thin
    /// forwarder.
    async fn run_stream_impl(
        &self,
        input: Vec<ChatMessage>,
        thread: Option<AgentThread>,
        run_options: AgentRunOptions,
    ) -> Result<AgentRunStream> {
        let mut thread = thread.unwrap_or_else(|| self.get_new_thread());
        let (final_messages, options) = self
            .prepare_request(&input, &mut thread, &run_options)
            .await?;

        // When agent middleware is configured, route the run through the same
        // pipeline as `run` (so guardrails/rewrites/termination apply) and then
        // emit the resulting messages as updates. Token-level streaming is only
        // used when there is no agent middleware to honor.
        if self.has_middleware() {
            let response = match self.run_core(final_messages, options, true).await {
                Ok(r) => r,
                Err(e) => {
                    if let Some(cp) = self.resolve_provider(&thread) {
                        let _ = cp.invoked(&input, &[], Some(&e)).await;
                    }
                    return Err(e);
                }
            };
            self.update_thread_conversation_id(&mut thread, response.conversation_id.as_deref())
                .await?;
            thread.on_new_messages(input.clone()).await?;
            thread.on_new_messages(response.messages.clone()).await?;
            if let Some(cp) = self.resolve_provider(&thread) {
                cp.invoked(&input, &response.messages, None).await?;
            }
            // Distinct message ids keep boundaries when re-aggregated; the
            // response's conversation/response ids and usage ride along so
            // service-managed continuity survives the middleware replay.
            return Ok(futures::stream::iter(response_to_updates(response)).boxed());
        }

        let (final_messages, options) = self
            .apply_chat_middleware_pre_call(final_messages, options)
            .await?;
        // Capture the structured-output format before `options` is consumed, so
        // the stream's terminal aggregation can auto-populate `value` (task 3).
        let response_format = options.response_format.clone();
        let agent_name = self.name.clone();
        let provider = self.resolve_provider(&thread);
        let inner = match self
            .client
            .get_streaming_response(final_messages, options)
            .await
        {
            Ok(s) => s,
            Err(e) => {
                // Failure before the stream opens: let providers observe it.
                if let Some(cp) = &provider {
                    let _ = cp.invoked(&input, &[], Some(&e)).await;
                }
                return Err(e);
            }
        };

        // Wrap the inner stream: forward mapped updates, then update the thread.
        let stream =
            async_stream_forward(inner, agent_name, thread, input, provider, response_format);
        Ok(stream.boxed())
    }

    /// Assemble the final message list and options for a request, applying
    /// thread history and context providers.
    async fn prepare_request(
        &self,
        input: &[ChatMessage],
        thread: &mut AgentThread,
        run_options: &AgentRunOptions,
    ) -> Result<(Vec<ChatMessage>, ChatOptions)> {
        // Fire the context-provider `thread_created` hook when a run uses a
        // service-managed thread (mirrors `_agents.py:1264-1265`). The id
        // argument is the service thread id.
        let service_thread_id = thread.service_thread_id().map(str::to_string);
        if let Some(id) = &service_thread_id {
            if let Some(cp) = self.resolve_provider(thread) {
                cp.thread_created(Some(id)).await?;
            }
        }

        // Merge per-run chat-option overrides over the agent's defaults, with
        // the per-run side winning (mirrors Python's `run_chat_options &
        // ChatOptions(...)`, whose right-hand side takes precedence — see
        // `ChatOptions::merge`).
        let mut options = match &run_options.chat_options {
            Some(overrides) => self.chat_options.clone().merge(overrides.clone()),
            None => self.chat_options.clone(),
        };
        // A service-managed thread's id drives continuity and wins; on a
        // local thread, a per-run / agent-default `conversation_id` override
        // survives (previously it was unconditionally cleared here, silently
        // starting a new service conversation despite the documented per-run
        // precedence).
        options.conversation_id = service_thread_id.or(options.conversation_id.take());

        let mut history = thread.list_messages().await?;

        // Context provider injection.
        let provider = thread
            .context_provider
            .clone()
            .or_else(|| self.context_provider.clone());
        if let Some(cp) = provider {
            let ctx = cp.invoking(input).await?;
            if let Some(instr) = ctx.instructions {
                options.instructions = Some(match options.instructions.take() {
                    Some(base) => format!("{base}\n{instr}"),
                    None => instr,
                });
            }
            history.extend(ctx.messages);
            // Deduplicate by name: a tool may be defined on the agent and also
            // injected by a context provider. Providers rejecting duplicate
            // tool names would otherwise fail the request.
            for t in ctx.tools {
                if !options.tools.iter().any(|existing| existing.name == t.name) {
                    options.tools.push(t);
                }
            }
        }

        // Append per-run additional tools (deduplicated by name), available for
        // this call only. Declaration-only tools (no executor) surface their
        // calls back to the caller (frontend-tool pattern) via the
        // function-invocation loop's declaration-only handling in `client.rs`.
        for t in &run_options.additional_tools {
            if !options.tools.iter().any(|existing| existing.name == t.name) {
                options.tools.push(t.clone());
            }
        }

        // Resolve dynamic tool sources (e.g. MCP servers) fresh for this run,
        // appended after every tool assembled above (dedup by name against
        // those tools plus any earlier source already appended in this same
        // loop; first registrant wins) — mirrors the Python reference's
        // `existing_names` skip when (re)loading MCP tools/prompts
        // (`_mcp.py:654,696`). A source's failure propagates out of the whole
        // run; see [`ToolSource::resolve_tools`].
        for source in &self.tool_sources {
            let resolved = source.resolve_tools().await?;
            for t in resolved {
                if options.tools.iter().any(|existing| existing.name == t.name) {
                    tracing::warn!(
                        source = source.source_name(),
                        tool = %t.name,
                        "tool source produced a tool whose name collides with an existing \
                         tool; skipping"
                    );
                    continue;
                }
                options.tools.push(t);
            }
        }

        history.extend(input.iter().cloned());
        let instructions = options.instructions.take();
        let final_messages = prepare_messages(history, instructions.as_deref());
        Ok((final_messages, options))
    }

    /// Run the agent middleware pipeline with a terminal that calls the chat
    /// client, returning the aggregated response. Shared by `run` and the
    /// middleware path of `run_stream` (thread updates are handled by callers).
    ///
    /// Wrapped in an `invoke_agent` span (OTel GenAI semconv). The plain
    /// token-streaming path does not go through here; that path is observed at
    /// the chat-client decorator level (see [`crate::observability`]).
    async fn run_core(
        &self,
        final_messages: Vec<ChatMessage>,
        options: ChatOptions,
        is_streaming: bool,
    ) -> Result<AgentRunResponse> {
        let span = crate::observability::agent_span(
            self.name.as_deref().unwrap_or(self.id.as_str()),
            &self.id,
        );
        async move {
            let result = self
                .run_core_inner(final_messages, options, is_streaming)
                .await;
            let span = tracing::Span::current();
            match &result {
                Ok(response) => {
                    if let Some(usage) = &response.usage_details {
                        if let Some(input) = usage.input_token_count {
                            span.record(crate::observability::attr::INPUT_TOKENS, input);
                        }
                        if let Some(output) = usage.output_token_count {
                            span.record(crate::observability::attr::OUTPUT_TOKENS, output);
                        }
                    }
                }
                Err(err) => {
                    crate::observability::record_error(&span, err);
                }
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn run_core_inner(
        &self,
        final_messages: Vec<ChatMessage>,
        options: ChatOptions,
        is_streaming: bool,
    ) -> Result<AgentRunResponse> {
        let client = self.client.clone();
        let chat_middleware = self.chat_middleware.clone();
        let terminal: Terminal<AgentRunContext> = Box::new(move |mut ctx: AgentRunContext| {
            let client = client.clone();
            let options = options.clone();
            let chat_middleware = chat_middleware.clone();
            Box::pin(async move {
                if ctx.terminate {
                    return Ok(ctx);
                }
                let response = Self::call_chat_client(
                    &client,
                    &chat_middleware,
                    ctx.messages.clone(),
                    options,
                    ctx.is_streaming,
                )
                .await?;
                ctx.result = Some(AgentRunResponse::from_chat_response(response));
                Ok(ctx)
            }) as crate::tools::BoxFuture<Result<AgentRunContext>>
        });

        let ctx = AgentRunContext::new(final_messages, is_streaming);
        let ctx = self.agent_middleware.execute(ctx, terminal).await?;
        let mut response = ctx.result.ok_or_else(|| {
            crate::error::Error::AgentExecution("agent produced no result".into())
        })?;

        if let Some(name) = &self.name {
            for m in &mut response.messages {
                if m.author_name.is_none() {
                    m.author_name = Some(name.clone());
                }
            }
        }
        Ok(response)
    }

    /// Invoke the chat client once, routed through the chat-middleware
    /// pipeline (mirrors Python's `use_chat_middleware`).
    ///
    /// Middleware may mutate `messages`/`chat_options` before the call, then
    /// observe (or override, via [`ChatContext::result`]) the response after
    /// calling `next.run(...)`. A middleware that sets `terminate = true`
    /// without invoking `next` short-circuits the call entirely: the
    /// underlying client is never invoked, and [`ChatContext::result`] (if
    /// set) becomes the returned response.
    async fn call_chat_client(
        client: &Arc<dyn ChatClient>,
        chat_middleware: &MiddlewarePipeline<ChatContext>,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
        is_streaming: bool,
    ) -> Result<ChatResponse> {
        if chat_middleware.is_empty() {
            return client.get_response(messages, options).await;
        }
        let client = client.clone();
        let terminal: Terminal<ChatContext> = Box::new(move |mut ctx: ChatContext| {
            let client = client.clone();
            Box::pin(async move {
                if ctx.terminate {
                    return Ok(ctx);
                }
                let response = client
                    .get_response(ctx.messages.clone(), ctx.chat_options.clone())
                    .await?;
                ctx.result = Some(response);
                Ok(ctx)
            }) as crate::tools::BoxFuture<Result<ChatContext>>
        });
        let ctx = ChatContext::new(messages, options, is_streaming);
        let ctx = chat_middleware.execute(ctx, terminal).await?;
        ctx.result.ok_or_else(|| {
            crate::error::Error::AgentExecution("chat middleware produced no result".into())
        })
    }

    /// Apply chat middleware to a *streaming* call's `messages`/`chat_options`
    /// before the real network call.
    ///
    /// Unlike [`ChatAgent::call_chat_client`], this only honors *pre-call*
    /// mutation: a real token stream can't flow back through
    /// [`ChatContext::result`] (typed for a complete [`ChatResponse`]), so any
    /// middleware logic placed *after* `next.run(...)` observes
    /// `ctx.result == None` and cannot post-process individual streamed
    /// tokens, and `terminate`/`result` short-circuiting is not honored here.
    /// This mirrors upstream Python's `use_chat_middleware`, whose streaming
    /// path likewise hands middleware an unconsumed async generator rather
    /// than driving it through the pipeline. Full interception (including
    /// short-circuiting) for chat middleware is available via
    /// [`ChatAgent::run`]/[`ChatAgent::run_once`], and via `run_stream` too
    /// when at least one agent middleware is also configured (that path
    /// funnels through [`ChatAgent::run_core`] and replays the result as
    /// updates).
    async fn apply_chat_middleware_pre_call(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<(Vec<ChatMessage>, ChatOptions)> {
        if self.chat_middleware.is_empty() {
            return Ok((messages, options));
        }
        let terminal: Terminal<ChatContext> = Box::new(|ctx| Box::pin(async move { Ok(ctx) }));
        let ctx = ChatContext::new(messages, options, true);
        let ctx = self.chat_middleware.execute(ctx, terminal).await?;
        Ok((ctx.messages, ctx.chat_options))
    }

    /// Whether this agent has any agent-level middleware configured.
    fn has_middleware(&self) -> bool {
        !self.agent_middleware.is_empty()
    }

    /// The effective context provider for a run: the thread's, else the agent's.
    fn resolve_provider(&self, thread: &AgentThread) -> Option<Arc<AggregateContextProvider>> {
        thread
            .context_provider
            .clone()
            .or_else(|| self.context_provider.clone())
    }

    /// Build a fresh message store from the configured factory, or an
    /// [`InMemoryChatMessageStore`] when none is set.
    fn new_message_store(&self) -> Arc<dyn ChatMessageStore> {
        match &self.chat_message_store_factory {
            Some(factory) => factory(),
            None => Arc::new(InMemoryChatMessageStore::new()),
        }
    }

    /// Reconcile a run's conversation id with the thread, mirroring Python's
    /// `_update_thread_with_type_and_conversation_id` (`_agents.py:1204-1234`).
    ///
    /// * No id returned while the thread *is* service-managed → the service
    ///   doesn't support service-managed threads for this request, so surface
    ///   an [`Error::AgentExecution`] (matches Python raising
    ///   `AgentExecutionException`, GAP item 14).
    /// * An id returned that the thread newly adopts → fire the
    ///   `thread_created` context-provider hook with the adopted id (task 1,
    ///   `_agents.py:1228-1229`).
    async fn update_thread_conversation_id(
        &self,
        thread: &mut AgentThread,
        response_conversation_id: Option<&str>,
    ) -> Result<()> {
        match response_conversation_id {
            None => {
                if thread.service_thread_id().is_some() {
                    return Err(Error::AgentExecution(
                        "Service did not return a valid conversation id when using a service \
                         managed thread."
                            .into(),
                    ));
                }
                Ok(())
            }
            Some(cid) => {
                if thread.try_adopt_service_thread_id(cid).await? {
                    if let Some(cp) = self.resolve_provider(thread) {
                        cp.thread_created(Some(cid)).await?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// State carried while forwarding a chat stream as agent updates.
type ForwardFinish = Option<(
    AgentThread,
    Vec<ChatMessage>,
    Option<Arc<AggregateContextProvider>>,
    Option<ResponseFormat>,
)>;

/// Forward an inner chat stream as agent updates and update the thread on end.
fn async_stream_forward(
    inner: crate::client::ChatStream,
    agent_name: Option<String>,
    thread: AgentThread,
    input: Vec<ChatMessage>,
    provider: Option<Arc<AggregateContextProvider>>,
    response_format: Option<ResponseFormat>,
) -> impl Stream<Item = Result<AgentRunResponseUpdate>> + Send {
    let finish: ForwardFinish = Some((thread, input, provider, response_format));
    futures::stream::unfold(
        (
            inner,
            Vec::<crate::types::ChatResponseUpdate>::new(),
            false,
            finish,
        ),
        move |(mut inner, mut collected, done, mut finish)| {
            let agent_name = agent_name.clone();
            async move {
                if done {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(update)) => {
                        collected.push(update.clone());
                        let mut au = AgentRunResponseUpdate::from_chat_update(&update);
                        if au.author_name.is_none() {
                            au.author_name = agent_name.clone();
                        }
                        Some((Ok(au), (inner, collected, false, finish)))
                    }
                    Some(Err(e)) => {
                        // Failure mid-stream: let context providers observe the
                        // error (task 2) before surfacing it. The stream error
                        // takes precedence, so the hook result is discarded.
                        if let Some((_thread, input, Some(cp), _rf)) = finish.take() {
                            let _ = cp.invoked(&input, &[], Some(&e)).await;
                        }
                        Some((Err(e), (inner, collected, true, None)))
                    }
                    None => {
                        // Stream finished: reconcile the conversation id, update
                        // the thread history, and fire the completion hook.
                        // Surface any failure as the final item rather than
                        // dropping it.
                        if let Some((mut thread, input, provider, response_format)) = finish.take()
                        {
                            let response = ChatResponse::from_updates_with_format(
                                collected.clone(),
                                response_format.as_ref(),
                            );
                            match response.conversation_id.as_deref() {
                                None => {
                                    if thread.service_thread_id().is_some() {
                                        return Some((
                                            Err(Error::AgentExecution(
                                                "Service did not return a valid conversation id \
                                                 when using a service managed thread."
                                                    .into(),
                                            )),
                                            (inner, collected, true, None),
                                        ));
                                    }
                                }
                                Some(cid) => match thread.try_adopt_service_thread_id(cid).await {
                                    Ok(true) => {
                                        if let Some(cp) = &provider {
                                            if let Err(e) = cp.thread_created(Some(cid)).await {
                                                return Some((
                                                    Err(e),
                                                    (inner, collected, true, None),
                                                ));
                                            }
                                        }
                                    }
                                    Ok(false) => {}
                                    Err(e) => {
                                        return Some((Err(e), (inner, collected, true, None)))
                                    }
                                },
                            }
                            if let Err(e) = thread.on_new_messages(input.clone()).await {
                                return Some((Err(e), (inner, collected, true, None)));
                            }
                            if let Err(e) = thread.on_new_messages(response.messages.clone()).await
                            {
                                return Some((Err(e), (inner, collected, true, None)));
                            }
                            if let Some(cp) = provider {
                                if let Err(e) = cp.invoked(&input, &response.messages, None).await {
                                    return Some((Err(e), (inner, collected, true, None)));
                                }
                            }
                        }
                        None
                    }
                }
            }
        },
    )
}

#[async_trait]
impl Agent for ChatAgent {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        self.run_with_options(messages, thread, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
        options: AgentRunOptions,
    ) -> Result<AgentRunResponse> {
        let mut owned_thread;
        let thread: &mut AgentThread = match thread {
            Some(t) => t,
            None => {
                owned_thread = self.get_new_thread();
                &mut owned_thread
            }
        };

        let (final_messages, chat_options) =
            self.prepare_request(&messages, thread, &options).await?;
        let response = match self.run_core(final_messages, chat_options, false).await {
            Ok(r) => r,
            Err(e) => {
                // Failure path: let context providers observe the error
                // (task 2). The run's error takes precedence over any hook
                // failure, so the hook result is intentionally discarded.
                if let Some(cp) = self.resolve_provider(thread) {
                    let _ = cp.invoked(&messages, &[], Some(&e)).await;
                }
                return Err(e);
            }
        };

        // Persist / validate the service-managed conversation id before
        // touching local history (tasks: service-thread adoption + missing-id
        // error). Fires `thread_created` on newly adopted ids.
        self.update_thread_conversation_id(thread, response.conversation_id.as_deref())
            .await?;
        // Update thread history.
        thread.on_new_messages(messages.clone()).await?;
        thread.on_new_messages(response.messages.clone()).await?;

        // Fire the context provider's success completion hook.
        if let Some(cp) = self.resolve_provider(thread) {
            cp.invoked(&messages, &response.messages, None).await?;
        }

        Ok(response)
    }

    async fn run_stream(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<AgentThread>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        self.run_stream_impl(messages, thread, options.unwrap_or_default())
            .await
    }

    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_new_thread(&self) -> AgentThread {
        // A service-managed conversation id yields a service thread; otherwise
        // create a local thread with a shared store (from the configured
        // factory, else in-memory) so history is observable across clones
        // (e.g. during streaming).
        let mut thread = match &self.chat_options.conversation_id {
            Some(id) => AgentThread::service(id.clone()),
            None => AgentThread::local(self.new_message_store()),
        };
        if let Some(cp) = &self.context_provider {
            thread = thread.with_context_provider(cp.clone());
        }
        thread
    }
}

/// Builder for [`ChatAgent`].
pub struct ChatAgentBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    /// The raw, caller-supplied client. Wrapping in [`FunctionInvokingChatClient`]
    /// is deferred to [`ChatAgentBuilder::build`] so that builder-collected
    /// function middleware can be threaded into the wrapper's constructor.
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_provider: Option<Arc<AggregateContextProvider>>,
    chat_message_store_factory: Option<ChatMessageStoreFactory>,
    agent_middleware: Vec<Arc<crate::middleware::AgentMiddleware>>,
    chat_middleware: Vec<Arc<crate::middleware::ChatMiddleware>>,
    function_middleware: Vec<Arc<crate::middleware::FunctionMiddleware>>,
    tool_sources: Vec<Arc<dyn ToolSource>>,
}

impl ChatAgentBuilder {
    fn new(client: impl ChatClient + 'static) -> Self {
        Self {
            id: None,
            name: None,
            description: None,
            instructions: None,
            client: Arc::new(client),
            chat_options: ChatOptions::new(),
            context_provider: None,
            chat_message_store_factory: None,
            agent_middleware: Vec::new(),
            chat_middleware: Vec::new(),
            function_middleware: Vec::new(),
            tool_sources: Vec::new(),
        }
    }

    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
    pub fn model(mut self, model_id: impl Into<String>) -> Self {
        self.chat_options.model_id = Some(model_id.into());
        self
    }
    pub fn temperature(mut self, temperature: f32) -> Self {
        self.chat_options.temperature = Some(temperature);
        self
    }
    pub fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.chat_options.max_tokens = Some(max_tokens);
        self
    }
    /// Request a structured-output response format (e.g. a JSON schema).
    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.chat_options.response_format = Some(format);
        self
    }
    /// Add a tool available to the agent.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.chat_options.tools.push(tool);
        self
    }
    /// Add multiple tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.chat_options.tools.extend(tools);
        self
    }
    /// Register a dynamic tool source (e.g. an MCP server wrapper), resolved
    /// fresh on every run and appended after the agent's static/context/
    /// per-run tools (dedup by name against those and any earlier-registered
    /// source; first registrant wins). Call repeatedly to register more than
    /// one source. See [`ToolSource`], resolved internally by every
    /// [`ChatAgent`] run.
    pub fn tool_source(mut self, source: Arc<dyn ToolSource>) -> Self {
        self.tool_sources.push(source);
        self
    }
    /// Set the context provider(s).
    pub fn context_provider(mut self, provider: Arc<AggregateContextProvider>) -> Self {
        self.context_provider = Some(provider);
        self
    }
    /// Set a factory for the message store of each new local thread, mirroring
    /// Python's `chat_message_store_factory`. Used by
    /// [`ChatAgent::get_new_thread`] and [`ChatAgent::deserialize_thread`]
    /// instead of hardcoding [`InMemoryChatMessageStore`].
    pub fn chat_message_store_factory<F>(mut self, factory: F) -> Self
    where
        F: Fn() -> Arc<dyn ChatMessageStore> + Send + Sync + 'static,
    {
        self.chat_message_store_factory = Some(Arc::new(factory));
        self
    }
    /// Add an agent middleware.
    pub fn middleware(mut self, mw: Arc<crate::middleware::AgentMiddleware>) -> Self {
        self.agent_middleware.push(mw);
        self
    }
    /// Add a chat middleware, run around the underlying chat-client call on
    /// every request (repeatable, like [`ChatAgentBuilder::middleware`]).
    /// See the chat-client call pipeline for exactly what it can observe
    /// and mutate.
    pub fn chat_middleware(mut self, mw: Arc<crate::middleware::ChatMiddleware>) -> Self {
        self.chat_middleware.push(mw);
        self
    }
    /// Add a function-invocation middleware, run around every local tool call
    /// (repeatable). Plumbed down into the [`FunctionInvokingChatClient`]
    /// this builder wraps the underlying client with.
    pub fn function_middleware(mut self, mw: Arc<crate::middleware::FunctionMiddleware>) -> Self {
        self.function_middleware.push(mw);
        self
    }
    /// Override the whole chat options object (advanced).
    pub fn chat_options(mut self, options: ChatOptions) -> Self {
        // Preserve tools/instructions collected so far by merging.
        self.chat_options = options.merge(self.chat_options);
        self
    }

    /// Build the agent.
    pub fn build(mut self) -> ChatAgent {
        if let Some(instr) = self.instructions.take() {
            self.chat_options.instructions = Some(match self.chat_options.instructions.take() {
                Some(existing) => format!("{instr}\n{existing}"),
                None => instr,
            });
        }
        if self.chat_options.model_id.is_none() {
            self.chat_options.model_id = self.client.model_id().map(str::to_string);
        }
        // Wrap the raw client in `FunctionInvokingChatClient` now that all
        // builder-collected function middleware is known.
        let client: Arc<dyn ChatClient> = Arc::new(
            FunctionInvokingChatClient::new(self.client)
                .with_function_middleware(self.function_middleware),
        );
        ChatAgent {
            id: self.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            name: self.name,
            description: self.description,
            client,
            chat_options: self.chat_options,
            context_provider: self.context_provider,
            chat_message_store_factory: self.chat_message_store_factory,
            agent_middleware: MiddlewarePipeline::new(self.agent_middleware),
            chat_middleware: MiddlewarePipeline::new(self.chat_middleware),
            tool_sources: self.tool_sources,
        }
    }
}

impl ChatAgent {
    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Create a new **service-managed** thread bound to `service_thread_id`,
    /// mirroring Python's `get_new_thread(service_thread_id=…)`
    /// (`_agents.py:1078-1082`).
    ///
    /// The agent's context provider (if any) is attached. Returns an error if
    /// the requested combination is invalid (a service id plus a local store);
    /// this constructor only sets the service id, so it always succeeds, but
    /// routing through [`AgentThread::try_from_parts`] keeps the service-xor-
    /// store invariant in one place.
    pub fn get_new_thread_with_service_id(
        &self,
        service_thread_id: impl Into<String>,
    ) -> Result<AgentThread> {
        let mut thread = AgentThread::try_from_parts(Some(service_thread_id.into()), None)?;
        if let Some(cp) = &self.context_provider {
            thread = thread.with_context_provider(cp.clone());
        }
        Ok(thread)
    }

    /// Reconstruct a thread from serialized state (as produced by
    /// [`AgentThread::serialize`]), mirroring Python's
    /// `BaseAgent.deserialize_thread` (`_agents.py:378-392`).
    ///
    /// A local thread's store is built via the agent's
    /// `chat_message_store_factory` (or an [`InMemoryChatMessageStore`]) and
    /// populated from the state; the agent's context provider is attached.
    pub async fn deserialize_thread(&self, serialized_thread: &Value) -> Result<AgentThread> {
        let mut thread =
            AgentThread::deserialize(serialized_thread, Some(self.new_message_store())).await?;
        if let Some(cp) = &self.context_provider {
            thread = thread.with_context_provider(cp.clone());
        }
        Ok(thread)
    }

    /// Wrap this agent as a [`ToolDefinition`] usable by another agent's
    /// `.tool(...)`. Mirrors Python `BaseAgent.as_tool`.
    ///
    /// The tool takes a single string argument (default name `"task"`) and, on
    /// each call, runs this agent **statelessly** (a fresh thread per call),
    /// returning the response text.
    ///
    /// ```no_run
    /// # use agent_framework_core::prelude::*;
    /// # use agent_framework_core::agent::AsToolOptions;
    /// # fn demo(researcher: ChatAgent, coordinator_client: impl ChatClient + 'static) {
    /// let research_tool = researcher.as_tool(AsToolOptions::new().name("research"));
    /// let coordinator = ChatAgent::builder(coordinator_client)
    ///     .tool(research_tool)
    ///     .build();
    /// # let _ = coordinator;
    /// # }
    /// ```
    pub fn as_tool(&self, options: AsToolOptions) -> ToolDefinition {
        let agent = Arc::new(self.clone());
        // Mirror Python `name or _sanitize_agent_name(self.name)`: an explicit
        // name is used verbatim; a derived name is sanitized into a valid tool
        // identifier. Falls back to the agent id when no name is available.
        let tool_name = options
            .name
            .or_else(|| sanitize_agent_name(self.name.as_deref()))
            .unwrap_or_else(|| self.id.clone());
        let description = options
            .description
            .or_else(|| self.description.clone())
            .unwrap_or_default();
        let arg_name = options.arg_name.unwrap_or_else(|| "task".to_string());
        let arg_description = options
            .arg_description
            .unwrap_or_else(|| format!("Task for {tool_name}"));
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                arg_name.clone(): { "type": "string", "description": arg_description }
            },
            "required": [arg_name.clone()],
        });
        let arg_key = arg_name;
        AiFunction::new(tool_name, description, schema, move |args: Value| {
            let agent = agent.clone();
            let arg_key = arg_key.clone();
            async move {
                let task = args
                    .get(&arg_key)
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let response = agent.run_once(task).await?;
                Ok(Value::String(response.text()))
            }
        })
        .into_definition()
    }
}
