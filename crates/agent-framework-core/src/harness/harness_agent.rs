//! The harness agent: a pre-assembled agent with the harness pieces wired in.
//!
//! Port of upstream `_harness/_agent.py` (`create_harness_agent`). Python
//! takes some forty keyword arguments; here they are methods of
//! [`HarnessAgentBuilder`], started with [`create_harness_agent`]. By
//! default the agent gets:
//!
//! - [`DEFAULT_HARNESS_INSTRUCTIONS`], followed by any agent instructions;
//! - a [`TodoProvider`], an [`AgentModeProvider`] and a
//!   [`FileMemoryProvider`] over a [`FileSystemAgentFileStore`] at
//!   `{cwd}/agent-file-memory`;
//! - token-budget compaction when both token limits are set;
//! - a [`ToolApprovalAgent`] around the agent, so standing "always approve"
//!   answers and auto-approval rules apply.
//!
//! Opt in to a [`FileAccessProvider`] with
//! [`file_access_store`](HarnessAgentBuilder::file_access_store), a
//! [`BackgroundAgentsProvider`] with
//! [`background_agents`](HarnessAgentBuilder::background_agents), skills,
//! a shell tool, hosted web search, and a [`LoopAgent`] with
//! [`loop_should_continue`](HarnessAgentBuilder::loop_should_continue).
//!
//! The result is a [`HarnessAgent`]: the outermost wrapper as a
//! [`SupportsAgentRun`], plus handles on the providers it created, which the
//! loop callbacks ([`todos_remaining`](super::todos_remaining),
//! [`background_tasks_running`](super::background_tasks_running)) take.
//!
//! # Divergences
//!
//! - **Wrappers, not middleware.** Tool approval and the loop are agent
//!   wrappers in this port (see the [harness docs](super)), so the result
//!   nests `LoopAgent(ToolApprovalAgent(Agent))` instead of one agent with
//!   middleware. Message injection needs no wiring: it is built into every
//!   agent's function-invocation loop (see
//!   [`enqueue_messages`](crate::session::enqueue_messages)).
//! - **History.** History stays on the session, as everywhere in this
//!   port: a session from [`HarnessAgent::create_session`] gets an
//!   in-memory history provider. Attach another history provider to the
//!   session to persist elsewhere. Upstream persists history after every
//!   model call (`require_per_service_call_history_persistence`); this port
//!   persists after each run.
//! - **Compaction** runs before each run, over the history-prepended
//!   messages ([`AgentBuilder::with_compaction`]); there is no after-run
//!   compaction of the stored history.
//! - **Web search and shell are explicit.** Upstream adds the client's web
//!   search tool when the client advertises one and wires the shell tool
//!   from `agent-framework-tools`. A Rust client does not advertise hosted
//!   tools, and core cannot depend on the tools crate, so
//!   [`web_search`](HarnessAgentBuilder::web_search) adds the hosted tool on
//!   request and [`shell`](HarnessAgentBuilder::shell) takes the shell tool
//!   and its environment provider ready-made.
//! - **Skills** take a [`SkillsProvider`]; file-based skill discovery
//!   (`skills_paths`) is not ported.
//! - `otel_provider_name` has no counterpart.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use crate::agent::{Agent, AgentRunOptions, AgentRunStream, SupportsAgentRun};
use crate::client::ChatClient;
use crate::compaction::{CompactionStrategy, TokenBudget};
use crate::error::{Error, Result};
use crate::memory::ContextProvider;
use crate::session::AgentSession;
use crate::skills::SkillsProvider;
use crate::tools::{hosted_web_search, ToolDefinition};
use crate::types::{AgentResponse, ChatOptions, FunctionCallContent, Message};

use super::agent_loop::{LoopAgent, LoopCallback, LoopDecision, DEFAULT_MAX_ITERATIONS};
use super::background_agents::{BackgroundAgentsProvider, DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT};
use super::file_access::FileAccessProvider;
use super::file_memory::FileMemoryProvider;
use super::file_store::{AgentFileStore, FileSystemAgentFileStore};
use super::mode::AgentModeProvider;
use super::todo::TodoProvider;
use super::tool_approval::ToolApprovalAgent;

/// The default harness-level instructions.
pub const DEFAULT_HARNESS_INSTRUCTIONS: &str = "\
You are a helpful AI assistant that uses tools to complete tasks.

## General guidelines

- Think through the task before acting. Break complex work into clear steps.
- Use the tools available to you to gather information, perform actions, and verify results.
- Explain your reasoning and thought process as you work through tasks.
- Explain what you learned and what you are going to do next between tool calls, \
so the user can follow along with your thought process.
- Avoid making more than 4 tool calls in a row without explaining what you are doing.
- If a tool call fails or returns unexpected results, adapt your approach rather than \
repeating the same call.
- When you have completed the task, present a clear and concise summary of what you did \
and what you found.
";

type ApprovalRule = Arc<dyn Fn(&FunctionCallContent) -> bool + Send + Sync>;
type NextMessage = Arc<dyn LoopCallback<Option<Vec<Message>>>>;

/// Start building a harness agent over `client`. See the
/// [module docs](self).
pub fn create_harness_agent(client: Arc<dyn ChatClient>) -> HarnessAgentBuilder {
    HarnessAgentBuilder {
        client,
        id: None,
        name: None,
        description: None,
        harness_instructions: None,
        agent_instructions: None,
        tools: Vec::new(),
        max_context_window_tokens: None,
        max_output_tokens: None,
        disable_compaction: false,
        compaction_strategy: None,
        disable_todo: false,
        todo_provider: None,
        disable_mode: false,
        mode_provider: None,
        disable_file_memory: false,
        file_memory_store: None,
        file_access_store: None,
        file_access_session_scoped: false,
        file_access_disable_write_tools: false,
        file_access_disable_readonly_tool_approval: false,
        file_access_disable_write_tool_approval: false,
        skills_provider: None,
        background_agents: Vec::new(),
        background_agents_instructions: None,
        background_agents_wait_timeout: DEFAULT_BACKGROUND_AGENTS_WAIT_TIMEOUT,
        shell: None,
        web_search: false,
        disable_tool_auto_approval: false,
        auto_approval_rules: Vec::new(),
        loop_should_continue: None,
        loop_next_message: None,
        loop_max_iterations: Some(DEFAULT_MAX_ITERATIONS),
        context_providers: Vec::new(),
        chat_options: None,
    }
}

/// Configures a [`HarnessAgent`]; see [`create_harness_agent`].
pub struct HarnessAgentBuilder {
    client: Arc<dyn ChatClient>,
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    harness_instructions: Option<String>,
    agent_instructions: Option<String>,
    tools: Vec<ToolDefinition>,
    max_context_window_tokens: Option<usize>,
    max_output_tokens: Option<usize>,
    disable_compaction: bool,
    compaction_strategy: Option<Box<dyn CompactionStrategy>>,
    disable_todo: bool,
    todo_provider: Option<TodoProvider>,
    disable_mode: bool,
    mode_provider: Option<AgentModeProvider>,
    disable_file_memory: bool,
    file_memory_store: Option<Arc<dyn AgentFileStore>>,
    file_access_store: Option<Arc<dyn AgentFileStore>>,
    file_access_session_scoped: bool,
    file_access_disable_write_tools: bool,
    file_access_disable_readonly_tool_approval: bool,
    file_access_disable_write_tool_approval: bool,
    skills_provider: Option<SkillsProvider>,
    background_agents: Vec<Arc<dyn SupportsAgentRun>>,
    background_agents_instructions: Option<String>,
    background_agents_wait_timeout: Duration,
    shell: Option<(ToolDefinition, Option<Arc<dyn ContextProvider>>)>,
    web_search: bool,
    disable_tool_auto_approval: bool,
    auto_approval_rules: Vec<ApprovalRule>,
    loop_should_continue: Option<Arc<dyn LoopCallback<LoopDecision>>>,
    loop_next_message: Option<NextMessage>,
    loop_max_iterations: Option<usize>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    chat_options: Option<ChatOptions>,
}

/// Forwards to a shared loop callback, which `LoopAgentBuilder` takes by value.
struct Shared<T>(Arc<dyn LoopCallback<T>>);

#[async_trait]
impl<T: Send + 'static> LoopCallback<T> for Shared<T> {
    async fn call(&self, ctx: &super::agent_loop::LoopContext) -> Result<T> {
        self.0.call(ctx).await
    }
}

impl HarnessAgentBuilder {
    /// The agent id (a UUID when unset).
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
    /// The agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    /// The agent description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
    /// Replace [`DEFAULT_HARNESS_INSTRUCTIONS`]; `""` omits them.
    pub fn harness_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.harness_instructions = Some(instructions.into());
        self
    }
    /// Task-specific instructions, after the harness instructions.
    pub fn agent_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.agent_instructions = Some(instructions.into());
        self
    }
    /// Add a tool.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }
    /// Add tools.
    pub fn tools(mut self, tools: impl IntoIterator<Item = ToolDefinition>) -> Self {
        self.tools.extend(tools);
        self
    }
    /// The model's context window, in tokens. With
    /// [`max_output_tokens`](Self::max_output_tokens) it enables the default
    /// token-budget compaction.
    pub fn max_context_window_tokens(mut self, tokens: usize) -> Self {
        self.max_context_window_tokens = Some(tokens);
        self
    }
    /// The maximum output per response, in tokens; also the default
    /// `max_tokens` option.
    pub fn max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = Some(tokens);
        self
    }
    /// Turn compaction off.
    pub fn disable_compaction(mut self) -> Self {
        self.disable_compaction = true;
        self
    }
    /// Compact with `strategy` instead of the token-budget default (which
    /// then needs no token limits).
    pub fn compaction_strategy(mut self, strategy: impl CompactionStrategy + 'static) -> Self {
        self.compaction_strategy = Some(Box::new(strategy));
        self
    }
    /// Leave out the todo provider.
    pub fn disable_todo(mut self) -> Self {
        self.disable_todo = true;
        self
    }
    /// Use this todo provider (keep a clone for
    /// [`todos_remaining`](super::todos_remaining)).
    pub fn todo_provider(mut self, provider: TodoProvider) -> Self {
        self.todo_provider = Some(provider);
        self
    }
    /// Leave out the mode provider.
    pub fn disable_mode(mut self) -> Self {
        self.disable_mode = true;
        self
    }
    /// Use this mode provider.
    pub fn mode_provider(mut self, provider: AgentModeProvider) -> Self {
        self.mode_provider = Some(provider);
        self
    }
    /// Leave out file memory.
    pub fn disable_file_memory(mut self) -> Self {
        self.disable_file_memory = true;
        self
    }
    /// Back file memory with `store` instead of `{cwd}/agent-file-memory`.
    pub fn file_memory_store(mut self, store: Arc<dyn AgentFileStore>) -> Self {
        self.file_memory_store = Some(store);
        self
    }
    /// Give the agent file access tools over `store`.
    pub fn file_access_store(mut self, store: Arc<dyn AgentFileStore>) -> Self {
        self.file_access_store = Some(store);
        self
    }
    /// Confine file access to a per-session workspace.
    pub fn file_access_session_scoped(mut self) -> Self {
        self.file_access_session_scoped = true;
        self
    }
    /// Advertise only the read-only file access tools.
    pub fn file_access_disable_write_tools(mut self) -> Self {
        self.file_access_disable_write_tools = true;
        self
    }
    /// Run the read-only file access tools without approval.
    pub fn file_access_disable_readonly_tool_approval(mut self) -> Self {
        self.file_access_disable_readonly_tool_approval = true;
        self
    }
    /// Run the writing file access tools without approval.
    pub fn file_access_disable_write_tool_approval(mut self) -> Self {
        self.file_access_disable_write_tool_approval = true;
        self
    }
    /// Add a skills provider.
    pub fn skills_provider(mut self, provider: SkillsProvider) -> Self {
        self.skills_provider = Some(provider);
        self
    }
    /// Agents the harness agent may delegate background tasks to. Only
    /// supply agents you trust with what it may send them.
    pub fn background_agents(
        mut self,
        agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>,
    ) -> Self {
        self.background_agents.extend(agents);
        self
    }
    /// Replace the background agents' instructions (`{background_agents}`
    /// is replaced with the list).
    pub fn background_agents_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.background_agents_instructions = Some(instructions.into());
        self
    }
    /// How long the background wait tool blocks (default five minutes).
    pub fn background_agents_wait_timeout(mut self, timeout: Duration) -> Self {
        self.background_agents_wait_timeout = timeout;
        self
    }
    /// Add a shell tool and, optionally, the context provider describing its
    /// environment (both from `agent-framework-tools`).
    pub fn shell(
        mut self,
        tool: ToolDefinition,
        environment_provider: Option<Arc<dyn ContextProvider>>,
    ) -> Self {
        self.shell = Some((tool, environment_provider));
        self
    }
    /// Add the hosted web search tool. Only for a client whose service runs
    /// it.
    pub fn web_search(mut self) -> Self {
        self.web_search = true;
        self
    }
    /// Do not wrap the agent in a [`ToolApprovalAgent`].
    pub fn disable_tool_auto_approval(mut self) -> Self {
        self.disable_tool_auto_approval = true;
        self
    }
    /// A rule that approves a call that would otherwise need approval
    /// (for example [`FileAccessProvider::read_only_tools_auto_approval_rule`]).
    pub fn auto_approval_rule(
        mut self,
        rule: impl Fn(&FunctionCallContent) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.auto_approval_rules.push(Arc::new(rule));
        self
    }
    /// Re-run the agent while `condition` says so, in a [`LoopAgent`]
    /// outside tool approval.
    pub fn loop_should_continue(
        mut self,
        condition: impl LoopCallback<LoopDecision> + 'static,
    ) -> Self {
        self.loop_should_continue = Some(Arc::new(condition));
        self
    }
    /// The loop's next-message callback.
    pub fn loop_next_message(
        mut self,
        next: impl LoopCallback<Option<Vec<Message>>> + 'static,
    ) -> Self {
        self.loop_next_message = Some(Arc::new(next));
        self
    }
    /// The loop's iteration cap (`None` for none).
    pub fn loop_max_iterations(mut self, max: Option<usize>) -> Self {
        self.loop_max_iterations = max;
        self
    }
    /// Add context providers after the built-in ones.
    pub fn context_providers(
        mut self,
        providers: impl IntoIterator<Item = Arc<dyn ContextProvider>>,
    ) -> Self {
        self.context_providers.extend(providers);
        self
    }
    /// Default chat options (temperature and so on). `max_tokens` defaults
    /// to [`max_output_tokens`](Self::max_output_tokens) when unset here.
    pub fn chat_options(mut self, options: ChatOptions) -> Self {
        self.chat_options = Some(options);
        self
    }

    /// Assemble the agent.
    ///
    /// Fails when a token limit is zero, the output limit is not below the
    /// context window, the background agents are invalid, the default file
    /// memory directory cannot be used, or the loop configuration is invalid.
    pub fn build(self) -> Result<HarnessAgent> {
        if self.max_context_window_tokens == Some(0) {
            return Err(Error::Configuration(
                "max_context_window_tokens must be positive.".into(),
            ));
        }
        if self.max_output_tokens == Some(0) {
            return Err(Error::Configuration(
                "max_output_tokens must be positive.".into(),
            ));
        }
        if let (Some(window), Some(output)) =
            (self.max_context_window_tokens, self.max_output_tokens)
        {
            if output >= window {
                return Err(Error::Configuration(
                    "max_output_tokens must be less than max_context_window_tokens.".into(),
                ));
            }
        }

        let harness = self
            .harness_instructions
            .unwrap_or_else(|| DEFAULT_HARNESS_INSTRUCTIONS.to_string());
        let instructions = format!(
            "{harness}\n\n{}",
            self.agent_instructions.unwrap_or_default()
        )
        .trim()
        .to_string();

        let mut builder = Agent::builder(self.client.clone());
        if let Some(id) = self.id {
            builder = builder.id(id);
        }
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        if let Some(description) = self.description {
            builder = builder.description(description);
        }
        let mut options = self.chat_options.unwrap_or_default();
        if !instructions.is_empty() {
            options.instructions = Some(match options.instructions.take() {
                Some(extra) if !extra.is_empty() => format!("{instructions}\n\n{extra}"),
                _ => instructions,
            });
        }
        if options.max_tokens.is_none() {
            options.max_tokens = self
                .max_output_tokens
                .map(|t| u32::try_from(t).unwrap_or(u32::MAX));
        }
        builder = builder.chat_options(options);

        // Compaction first among the agent's providers, so it sees the
        // history the session's provider prepended and nothing the harness
        // providers add after it.
        if !self.disable_compaction {
            match (
                self.compaction_strategy,
                self.max_context_window_tokens,
                self.max_output_tokens,
            ) {
                (Some(strategy), _, _) => {
                    builder = builder.with_compaction(BoxedStrategy(strategy));
                }
                (None, Some(window), Some(output)) => {
                    builder = builder.with_compaction(TokenBudget::new(window - output));
                }
                _ => {}
            }
        }

        let todo = (!self.disable_todo).then(|| self.todo_provider.unwrap_or_default());
        if let Some(todo) = &todo {
            builder = builder.context_provider(Arc::new(todo.clone()));
        }
        let mode = (!self.disable_mode).then(|| self.mode_provider.unwrap_or_default());
        if let Some(mode) = &mode {
            builder = builder.context_provider(Arc::new(mode.clone()));
        }
        if !self.disable_file_memory {
            let store = match self.file_memory_store {
                Some(store) => store,
                None => {
                    let root: PathBuf = std::env::current_dir()
                        .map_err(|e| Error::Configuration(format!("no working directory: {e}")))?
                        .join("agent-file-memory");
                    std::fs::create_dir_all(&root).map_err(|e| {
                        Error::Configuration(format!("cannot create {}: {e}", root.display()))
                    })?;
                    Arc::new(FileSystemAgentFileStore::new(root)?)
                }
            };
            builder = builder.context_provider(Arc::new(FileMemoryProvider::new(store)));
        }
        let file_access = self.file_access_store.map(|store| {
            let mut p = FileAccessProvider::new(store);
            if self.file_access_disable_write_tools {
                p = p.disable_write_tools();
            }
            if self.file_access_disable_readonly_tool_approval {
                p = p.disable_readonly_tool_approval();
            }
            if self.file_access_disable_write_tool_approval {
                p = p.disable_write_tool_approval();
            }
            if self.file_access_session_scoped {
                p = p.session_scoped();
            }
            p
        });
        if let Some(p) = &file_access {
            builder = builder.context_provider(Arc::new(p.clone()));
        }
        if let Some(skills) = self.skills_provider {
            builder = builder.context_provider(Arc::new(skills));
        }
        let background = if self.background_agents.is_empty() {
            None
        } else {
            let mut b = BackgroundAgentsProvider::builder(self.background_agents)
                .wait_timeout(self.background_agents_wait_timeout);
            if let Some(instructions) = self.background_agents_instructions {
                b = b.instructions(instructions);
            }
            Some(b.build()?)
        };
        if let Some(p) = &background {
            builder = builder.context_provider(Arc::new(p.clone()));
        }

        let mut tools = Vec::new();
        if self.web_search {
            tools.push(hosted_web_search());
        }
        if let Some((tool, provider)) = self.shell {
            tools.push(tool);
            if let Some(provider) = provider {
                builder = builder.context_provider(provider);
            }
        }
        tools.extend(self.tools);
        builder = builder.tools(tools);
        for provider in self.context_providers {
            builder = builder.context_provider(provider);
        }

        let agent = Arc::new(builder.build());
        let mut outer: Arc<dyn SupportsAgentRun> = agent.clone();
        if !self.disable_tool_auto_approval {
            let mut approval = ToolApprovalAgent::new(outer);
            for rule in self.auto_approval_rules {
                approval =
                    approval.with_auto_approval_rule(move |call: &FunctionCallContent| rule(call));
            }
            outer = Arc::new(approval);
        }
        if let Some(condition) = self.loop_should_continue {
            let mut b = LoopAgent::builder(outer, Shared(condition))
                .max_iterations(self.loop_max_iterations);
            if let Some(next) = self.loop_next_message {
                b = b.next_message(Shared(next));
            }
            outer = Arc::new(b.build()?);
        }
        Ok(HarnessAgent {
            agent,
            outer,
            todo,
            mode,
            file_access,
            background,
        })
    }
}

/// A [`CompactionStrategy`] behind a box, for the builder's type-erased one.
struct BoxedStrategy(Box<dyn CompactionStrategy>);

impl CompactionStrategy for BoxedStrategy {
    fn compact(
        &self,
        messages: &[Message],
        tokenizer: &dyn crate::compaction::Tokenizer,
    ) -> Vec<Message> {
        self.0.compact(messages, tokenizer)
    }
}

/// An assembled harness agent; see [`create_harness_agent`].
#[derive(Clone)]
pub struct HarnessAgent {
    agent: Arc<Agent>,
    outer: Arc<dyn SupportsAgentRun>,
    todo: Option<TodoProvider>,
    mode: Option<AgentModeProvider>,
    file_access: Option<FileAccessProvider>,
    background: Option<BackgroundAgentsProvider>,
}

impl std::fmt::Debug for HarnessAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HarnessAgent")
            .field("id", &self.agent.id())
            .finish_non_exhaustive()
    }
}

impl HarnessAgent {
    /// The inner [`Agent`], without the approval and loop wrappers.
    pub fn agent(&self) -> &Arc<Agent> {
        &self.agent
    }
    /// The outermost wrapper, which [`run`](SupportsAgentRun::run) uses.
    pub fn outer(&self) -> &Arc<dyn SupportsAgentRun> {
        &self.outer
    }
    /// The todo provider, unless disabled.
    pub fn todo_provider(&self) -> Option<&TodoProvider> {
        self.todo.as_ref()
    }
    /// The mode provider, unless disabled.
    pub fn mode_provider(&self) -> Option<&AgentModeProvider> {
        self.mode.as_ref()
    }
    /// The file access provider, when a store was given.
    pub fn file_access_provider(&self) -> Option<&FileAccessProvider> {
        self.file_access.as_ref()
    }
    /// The background agents provider, when agents were given.
    pub fn background_agents_provider(&self) -> Option<&BackgroundAgentsProvider> {
        self.background.as_ref()
    }
}

#[async_trait]
impl SupportsAgentRun for HarnessAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.outer.run(messages, session).await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        self.outer
            .run_with_options(messages, session, options)
            .await
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        self.outer.run_stream(messages, session, options).await
    }

    fn id(&self) -> &str {
        self.agent.id()
    }

    fn name(&self) -> Option<&str> {
        SupportsAgentRun::name(&*self.agent)
    }

    fn description(&self) -> Option<&str> {
        SupportsAgentRun::description(&*self.agent)
    }

    fn create_session(&self) -> AgentSession {
        self.outer.create_session()
    }
}
