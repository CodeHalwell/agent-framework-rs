//! Agents: the [`Agent`] trait and the concrete [`ChatAgent`].
//!
//! Rust equivalent of `agent_framework._agents`.

use async_trait::async_trait;
use futures::stream::{Stream, StreamExt};
use std::pin::Pin;
use std::sync::Arc;
use uuid::Uuid;

use crate::client::{ChatClient, FunctionInvokingChatClient};
use crate::error::Result;
use crate::memory::{AggregateContextProvider, ContextProvider};
use crate::middleware::{AgentRunContext, MiddlewarePipeline, Terminal};
use crate::threads::AgentThread;
use crate::tools::ToolDefinition;
use crate::types::{
    prepare_messages, AgentRunResponse, AgentRunResponseUpdate, ChatMessage, ChatOptions,
    IntoMessages,
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
pub struct ChatAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_provider: Option<Arc<AggregateContextProvider>>,
    agent_middleware: MiddlewarePipeline<AgentRunContext>,
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

        let inner = self
            .client
            .get_streaming_response(final_messages, options)
            .await?;
        let agent_name = self.name.clone();
        let input_for_thread = input;

        // Wrap the inner stream: forward mapped updates, then update the thread.
        let stream = async_stream_forward(inner, agent_name, thread, input_for_thread);
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
        options.conversation_id = thread.service_thread_id().map(str::to_string);

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
            options.tools.extend(ctx.tools);
        }

        history.extend(input.iter().cloned());
        let instructions = options.instructions.take();
        let final_messages = prepare_messages(history, instructions.as_deref());
        Ok((final_messages, options))
    }
}

/// Forward an inner chat stream as agent updates and update the thread on end.
fn async_stream_forward(
    inner: crate::client::ChatStream,
    agent_name: Option<String>,
    thread: AgentThread,
    input: Vec<ChatMessage>,
) -> impl Stream<Item = Result<AgentRunResponseUpdate>> + Send {
    use crate::types::ChatResponse;
    futures::stream::unfold(
        (
            inner,
            Vec::<crate::types::ChatResponseUpdate>::new(),
            false,
            Some((thread, input)),
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
                        // Stream finished: update the thread history.
                        if let Some((thread, input)) = finish.take() {
                            let mut thread = thread;
                            let response = ChatResponse::from_updates(collected.clone());
                            let _ = thread.on_new_messages(input).await;
                            let _ = thread.on_new_messages(response.messages).await;
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

        // Run through the agent middleware pipeline (terminal = LLM call).
        let client = self.client.clone();
        let terminal: Terminal<AgentRunContext> = Box::new(move |mut ctx: AgentRunContext| {
            let client = client.clone();
            let options = options.clone();
            Box::pin(async move {
                if ctx.terminate {
                    return Ok(ctx);
                }
                let response = client.get_response(ctx.messages.clone(), options).await?;
                ctx.result = Some(AgentRunResponse::from_chat_response(response));
                Ok(ctx)
            }) as crate::tools::BoxFuture<Result<AgentRunContext>>
        });

        let ctx = AgentRunContext::new(final_messages, false);
        let ctx = self.agent_middleware.execute(ctx, terminal).await?;
        let mut response = ctx.result.ok_or_else(|| {
            crate::error::Error::AgentExecution("agent produced no result".into())
        })?;

        // Fill author names.
        if let Some(name) = &self.name {
            for m in &mut response.messages {
                if m.author_name.is_none() {
                    m.author_name = Some(name.clone());
                }
            }
        }

        // Update thread history.
        thread.on_new_messages(messages).await?;
        thread.on_new_messages(response.messages.clone()).await?;

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
    client: Arc<dyn ChatClient>,
    chat_options: ChatOptions,
    context_provider: Option<Arc<AggregateContextProvider>>,
    agent_middleware: Vec<Arc<crate::middleware::AgentMiddleware>>,
}

impl ChatAgentBuilder {
    fn new(client: impl ChatClient + 'static) -> Self {
        Self {
            id: None,
            name: None,
            description: None,
            instructions: None,
            client: Arc::new(FunctionInvokingChatClient::new(client)),
            chat_options: ChatOptions::new(),
            context_provider: None,
            agent_middleware: Vec::new(),
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
        ChatAgent {
            id: self.id.unwrap_or_else(|| Uuid::new_v4().to_string()),
            name: self.name,
            description: self.description,
            client: self.client,
            chat_options: self.chat_options,
            context_provider: self.context_provider,
            agent_middleware: MiddlewarePipeline::new(self.agent_middleware),
        }
    }
}

impl ChatAgent {
    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}
