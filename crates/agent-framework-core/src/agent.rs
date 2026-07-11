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
use crate::error::Result;
use crate::memory::{AggregateContextProvider, ContextProvider};
use crate::middleware::{AgentRunContext, ChatContext, MiddlewarePipeline, Terminal};
use crate::threads::AgentThread;
use crate::tools::{AiFunction, ToolDefinition};
use crate::types::{
    prepare_messages, AgentRunResponse, AgentRunResponseUpdate, ChatMessage, ChatOptions,
    ChatResponse, IntoMessages, ResponseFormat,
};

/// A boxed stream of agent run updates.
pub type AgentRunStream = Pin<Box<dyn Stream<Item = Result<AgentRunResponseUpdate>> + Send>>;

/// The common interface implemented by all agents.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Run the agent to completion.
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse>;

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
    agent_middleware: MiddlewarePipeline<AgentRunContext>,
    /// Middleware run around the underlying chat-client call (mirrors
    /// Python's `use_chat_middleware`). See [`ChatAgent::call_chat_client`].
    chat_middleware: MiddlewarePipeline<ChatContext>,
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

    /// Run and stream incremental updates.
    ///
    /// The thread's history is updated when the stream completes; because
    /// message stores are shared via `Arc`, updates are visible on the original
    /// thread once the returned stream is fully consumed.
    pub async fn run_stream(
        &self,
        messages: impl IntoMessages,
        thread: Option<AgentThread>,
    ) -> Result<AgentRunStream> {
        let input = messages.into_messages();
        let mut thread = thread.unwrap_or_else(|| self.get_new_thread());
        let (final_messages, options) = self.prepare_request(&input, &mut thread).await?;

        // When agent middleware is configured, route the run through the same
        // pipeline as `run` (so guardrails/rewrites/termination apply) and then
        // emit the resulting messages as updates. Token-level streaming is only
        // used when there is no agent middleware to honor.
        if self.has_middleware() {
            let response = self.run_core(final_messages, options, true).await?;
            if let Some(cid) = response.conversation_id.as_deref() {
                thread.try_adopt_service_thread_id(cid).await?;
            }
            thread.on_new_messages(input.clone()).await?;
            thread.on_new_messages(response.messages.clone()).await?;
            if let Some(cp) = self.resolve_provider(&thread) {
                cp.invoked(&input, &response.messages).await?;
            }
            let updates: Vec<Result<AgentRunResponseUpdate>> = response
                .messages
                .into_iter()
                .enumerate()
                .map(|(i, m)| {
                    // Distinct message ids keep boundaries when re-aggregated.
                    let message_id = m.message_id.clone().or_else(|| Some(format!("msg-{i}")));
                    Ok(AgentRunResponseUpdate {
                        contents: m.contents,
                        role: Some(m.role),
                        author_name: m.author_name,
                        message_id,
                        ..Default::default()
                    })
                })
                .collect();
            return Ok(futures::stream::iter(updates).boxed());
        }

        let (final_messages, options) = self
            .apply_chat_middleware_pre_call(final_messages, options)
            .await?;
        let inner = self
            .client
            .get_streaming_response(final_messages, options)
            .await?;
        let agent_name = self.name.clone();
        let provider = self.resolve_provider(&thread);

        // Wrap the inner stream: forward mapped updates, then update the thread.
        let stream = async_stream_forward(inner, agent_name, thread, input, provider);
        Ok(stream.boxed())
    }

    /// Ergonomic run without an explicit thread.
    pub async fn run_once(&self, messages: impl IntoMessages) -> Result<AgentRunResponse> {
        self.run(messages.into_messages(), None).await
    }

    /// Assemble the final message list and options for a request, applying
    /// thread history and context providers.
    async fn prepare_request(
        &self,
        input: &[ChatMessage],
        thread: &mut AgentThread,
    ) -> Result<(Vec<ChatMessage>, ChatOptions)> {
        let mut options = self.chat_options.clone();
        options.conversation_id = thread.service_thread_id();

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
                    span.record(
                        crate::observability::attr::ERROR_TYPE,
                        crate::observability::error_type(err).as_str(),
                    );
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
}

/// State carried while forwarding a chat stream as agent updates.
type ForwardFinish = Option<(
    AgentThread,
    Vec<ChatMessage>,
    Option<Arc<AggregateContextProvider>>,
)>;

/// Forward an inner chat stream as agent updates and update the thread on end.
fn async_stream_forward(
    inner: crate::client::ChatStream,
    agent_name: Option<String>,
    thread: AgentThread,
    input: Vec<ChatMessage>,
    provider: Option<Arc<AggregateContextProvider>>,
) -> impl Stream<Item = Result<AgentRunResponseUpdate>> + Send {
    let finish: ForwardFinish = Some((thread, input, provider));
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
                    Some(Err(e)) => Some((Err(e), (inner, collected, true, finish))),
                    None => {
                        // Stream finished: update the thread history and fire the
                        // provider hook. Surface any failure as the final item
                        // rather than dropping it.
                        if let Some((mut thread, input, provider)) = finish.take() {
                            let response = ChatResponse::from_updates(collected.clone());
                            if let Some(cid) = response.conversation_id.as_deref() {
                                if let Err(e) = thread.try_adopt_service_thread_id(cid).await {
                                    return Some((Err(e), (inner, collected, true, None)));
                                }
                            }
                            if let Err(e) = thread.on_new_messages(input.clone()).await {
                                return Some((Err(e), (inner, collected, true, None)));
                            }
                            if let Err(e) = thread.on_new_messages(response.messages.clone()).await
                            {
                                return Some((Err(e), (inner, collected, true, None)));
                            }
                            if let Some(cp) = provider {
                                if let Err(e) = cp.invoked(&input, &response.messages).await {
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
        let mut owned_thread;
        let thread: &mut AgentThread = match thread {
            Some(t) => t,
            None => {
                owned_thread = self.get_new_thread();
                &mut owned_thread
            }
        };

        let (final_messages, options) = self.prepare_request(&messages, thread).await?;
        let response = self.run_core(final_messages, options, false).await?;

        // Persist a service-managed conversation id before touching local
        // history, so service-backed threads stay service-backed.
        if let Some(cid) = response.conversation_id.as_deref() {
            thread.try_adopt_service_thread_id(cid).await?;
        }
        // Update thread history.
        thread.on_new_messages(messages.clone()).await?;
        thread.on_new_messages(response.messages.clone()).await?;

        // Fire the context provider's completion hook.
        if let Some(cp) = self.resolve_provider(thread) {
            cp.invoked(&messages, &response.messages).await?;
        }

        Ok(response)
    }

    fn id(&self) -> &str {
        &self.id
    }
    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_new_thread(&self) -> AgentThread {
        // A service-managed conversation id yields a service thread; otherwise
        // create a local thread with a shared in-memory store so history is
        // observable across clones (e.g. during streaming).
        let mut thread = match &self.chat_options.conversation_id {
            Some(id) => AgentThread::service(id.clone()),
            None => AgentThread::local(std::sync::Arc::new(
                crate::threads::InMemoryChatMessageStore::new(),
            )),
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
    agent_middleware: Vec<Arc<crate::middleware::AgentMiddleware>>,
    chat_middleware: Vec<Arc<crate::middleware::ChatMiddleware>>,
    function_middleware: Vec<Arc<crate::middleware::FunctionMiddleware>>,
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
            agent_middleware: Vec::new(),
            chat_middleware: Vec::new(),
            function_middleware: Vec::new(),
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
    /// Set the context provider(s).
    pub fn context_provider(mut self, provider: Arc<AggregateContextProvider>) -> Self {
        self.context_provider = Some(provider);
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
            agent_middleware: MiddlewarePipeline::new(self.agent_middleware),
            chat_middleware: MiddlewarePipeline::new(self.chat_middleware),
        }
    }
}

impl ChatAgent {
    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
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
        let tool_name = options
            .name
            .or_else(|| self.name.clone())
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
