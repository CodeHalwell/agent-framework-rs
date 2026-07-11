//! High-level MCP tools: [`McpStdioTool`], [`McpStreamableHttpTool`], and
//! [`McpWebsocketTool`].
//!
//! Each owns a lazily-connected, shared [`McpClient`] and turns the server's
//! tool catalog into ready-to-use [`ToolDefinition`]s whose executors call
//! back into that shared session.
//!
//! Two ways to wire one into a [`agent_framework_core::agent::ChatAgent`]:
//!
//! - **Static** (frozen at build time): `mcp.tool_definitions().await` once,
//!   up front, and hand the result to `ChatAgent::builder().tools(..)`. The
//!   agent never notices a later server-side tool-catalog change.
//! - **Dynamic** (resolved on every run): all three types implement
//!   [`agent_framework_core::tools::ToolSource`], so
//!   `ChatAgent::builder().tool_source(Arc::new(mcp))` connects lazily on
//!   the agent's first run and re-resolves the tool list on every
//!   subsequent run from a cache that self-invalidates on the server's
//!   `notifications/tools/list_changed` (see [`McpClient::list_tools_cached`]).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::OnceCell;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::{ApprovalMode, Tool, ToolDefinition, ToolSource};
use agent_framework_core::types::ChatMessage;

use crate::client::McpClient;
use crate::protocol::{
    normalize_mcp_name, role_and_content_to_chat_message, PromptDescriptor, ToolDescriptor,
};
use crate::sampling::{Root, SamplingHandler};
use crate::transport::{McpStdioTransport, McpStreamableHttpTransport, McpWebsocketTransport};

/// The `clientInfo.name` this crate sends during `initialize`.
const CLIENT_NAME: &str = "agent-framework-rs";

/// Map an MCP `prompts/get` message into a core [`ChatMessage`] — mirrors
/// the Python reference's `_mcp_prompt_message_to_chat_message`.
fn prompt_message_to_chat_message(msg: &crate::protocol::PromptMessage) -> ChatMessage {
    role_and_content_to_chat_message(&msg.role, &msg.content)
}
/// The `clientInfo.version` this crate sends during `initialize`.
const CLIENT_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Approval policy for tools produced from an MCP server.
///
/// Mirrors the Python reference's `approval_mode`, which accepts
/// `"always_require"`, `"never_require"`, or a per-tool-name mapping. Rust's
/// [`ApprovalMode`] has no "unset" state (it defaults to
/// [`ApprovalMode::NeverRequire`]), so [`McpApprovalMode::NeverRequireAll`] is
/// this type's default too.
///
/// Note: Python's per-tool config accepts both an `always_require_approval`
/// and a `never_require_approval` name set, because its underlying
/// `approval_mode` can also be *unset* (`None`), which a fuller "never
/// listed" set can be distinguished from. Rust's [`ApprovalMode`] is a plain
/// two-value enum with no unset state, so "explicitly never" and "not
/// listed" already collapse to the same [`ApprovalMode::NeverRequire`]
/// result — [`McpApprovalMode::PerTool`] therefore only needs the
/// `always_require` set.
#[derive(Debug, Clone, Default)]
pub enum McpApprovalMode {
    /// No tool produced by this server requires approval (default).
    #[default]
    NeverRequireAll,
    /// Every tool produced by this server requires approval before it runs.
    AlwaysRequireAll,
    /// Approval is required only for tools whose local (normalized) name is
    /// in this set; every other tool resolves to [`ApprovalMode::NeverRequire`].
    PerTool { always_require: HashSet<String> },
}

impl McpApprovalMode {
    /// Require approval for every tool.
    pub fn always_require_all() -> Self {
        Self::AlwaysRequireAll
    }

    /// Require approval for no tool (the default).
    pub fn never_require_all() -> Self {
        Self::NeverRequireAll
    }

    /// Require approval only for the named tools (by local/normalized name).
    pub fn per_tool<I>(always_require: I) -> Self
    where
        I: IntoIterator<Item = &'static str>,
    {
        Self::PerTool {
            always_require: always_require.into_iter().map(str::to_string).collect(),
        }
    }

    fn resolve(&self, local_name: &str) -> ApprovalMode {
        match self {
            McpApprovalMode::NeverRequireAll => ApprovalMode::NeverRequire,
            McpApprovalMode::AlwaysRequireAll => ApprovalMode::AlwaysRequire,
            McpApprovalMode::PerTool { always_require } => {
                if always_require.contains(local_name) {
                    ApprovalMode::AlwaysRequire
                } else {
                    ApprovalMode::NeverRequire
                }
            }
        }
    }
}

/// An executable [`Tool`] that forwards `invoke` to `tools/call` on a shared
/// [`McpClient`], using the server's original (un-normalized) tool name.
struct McpToolExecutor {
    local_name: String,
    description: String,
    parameters: Value,
    remote_name: String,
    client: Arc<McpClient>,
}

#[async_trait]
impl Tool for McpToolExecutor {
    fn name(&self) -> &str {
        &self.local_name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        self.client
            .call_tool_value(&self.remote_name, arguments)
            .await
    }
}

/// Build one [`ToolDefinition`] per (filtered) tool descriptor, wired to call
/// back through `client`.
///
/// A tool whose normalized name collides with one already built from this
/// same listing is skipped (first occurrence wins), with a `tracing::warn!`
/// — mirrors the Python reference's `existing_names` skip when loading tools
/// (`_mcp.py:654`).
fn build_tool_definitions(
    client: Arc<McpClient>,
    tools: &[ToolDescriptor],
    allowed_tools: Option<&HashSet<String>>,
    approval_mode: &McpApprovalMode,
) -> Vec<ToolDefinition> {
    let mut seen_names: HashSet<String> = HashSet::new();
    let mut definitions = Vec::new();
    for descriptor in tools {
        let local_name = normalize_mcp_name(&descriptor.name);
        if let Some(allowed) = allowed_tools {
            if !allowed.contains(local_name.as_str()) {
                continue;
            }
        }
        if !seen_names.insert(local_name.clone()) {
            tracing::warn!(
                tool = %descriptor.name,
                local_name = %local_name,
                "MCP tool name collides (after normalization) with another tool from the \
                 same server; skipping the later one"
            );
            continue;
        }
        let executor: Arc<dyn Tool> = Arc::new(McpToolExecutor {
            local_name: local_name.clone(),
            description: descriptor.description.clone().unwrap_or_default(),
            parameters: descriptor.input_schema.clone(),
            remote_name: descriptor.name.clone(),
            client: client.clone(),
        });
        let mut definition = ToolDefinition::from_tool(executor);
        definition.approval_mode = approval_mode.resolve(&local_name);
        definitions.push(definition);
    }
    definitions
}

/// Drop any prompt whose normalized name collides with an earlier one in the
/// same listing (first occurrence wins), warning on each skip — the prompts
/// counterpart of [`build_tool_definitions`]'s dedup, mirroring the Python
/// reference's `existing_names` skip when loading prompts (`_mcp.py:696`).
///
/// Unlike tools, this port doesn't convert prompts into invokable
/// [`ToolDefinition`]s (a caller instead calls e.g. [`McpStdioTool::get_prompt`]
/// with the server's own, un-normalized name), so [`normalize_mcp_name`] is
/// used here only to decide what collides, not as an identifier the caller
/// ends up using — the returned [`PromptDescriptor`]s keep their original
/// `name`.
fn dedup_prompts_by_normalized_name(prompts: Vec<PromptDescriptor>) -> Vec<PromptDescriptor> {
    let mut seen_names: HashSet<String> = HashSet::new();
    prompts
        .into_iter()
        .filter(|p| {
            let local_name = normalize_mcp_name(&p.name);
            if seen_names.insert(local_name.clone()) {
                true
            } else {
                tracing::warn!(
                    prompt = %p.name,
                    local_name = %local_name,
                    "MCP prompt name collides (after normalization) with another prompt from \
                     the same server; skipping the later one"
                );
                false
            }
        })
        .collect()
}

/// An MCP tool backed by a stdio-connected child process.
///
/// ```no_run
/// # use agent_framework_mcp::McpStdioTool;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let mcp = McpStdioTool::new("filesystem", "npx")
///     .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
///     .description("Local filesystem access");
/// let tools = mcp.tool_definitions().await?;
/// # let _ = tools;
/// # Ok(())
/// # }
/// ```
pub struct McpStdioTool {
    name: String,
    description: Option<String>,
    command: String,
    args: Vec<String>,
    env: Option<HashMap<String, String>>,
    cwd: Option<PathBuf>,
    allowed_tools: Option<HashSet<String>>,
    approval_mode: McpApprovalMode,
    sampling_handler: Option<SamplingHandler>,
    roots: Option<Vec<Root>>,
    load_tools: bool,
    load_prompts: bool,
    request_timeout: Option<Duration>,
    session: OnceCell<Arc<McpClient>>,
}

impl McpStdioTool {
    /// Create a tool that spawns `command` (with no arguments) as an MCP
    /// server over stdio when connected.
    pub fn new(name: impl Into<String>, command: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            command: command.into(),
            args: Vec::new(),
            env: None,
            cwd: None,
            allowed_tools: None,
            approval_mode: McpApprovalMode::default(),
            sampling_handler: None,
            roots: None,
            load_tools: true,
            load_prompts: true,
            request_timeout: None,
            session: OnceCell::new(),
        }
    }

    /// Set the command-line arguments passed to the server process.
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Add environment variables for the server process (on top of the
    /// inherited parent environment).
    pub fn env<I, K, V>(mut self, env: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env = Some(env.into_iter().map(|(k, v)| (k.into(), v.into())).collect());
        self
    }

    /// Set the server process's working directory.
    pub fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Set a human-readable description for this tool source.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Restrict [`Self::tool_definitions`] to these (local/normalized) tool names.
    pub fn allowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Set the approval policy applied to produced [`ToolDefinition`]s.
    pub fn approval_mode(mut self, mode: McpApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Register the handler for server-initiated `sampling/createMessage`
    /// requests, applied when the underlying session is created. See
    /// [`McpClient::sampling_handler`].
    pub fn sampling_handler(mut self, handler: SamplingHandler) -> Self {
        self.sampling_handler = Some(handler);
        self
    }

    /// Register a static list of filesystem roots, applied when the
    /// underlying session is created. See [`McpClient::roots`].
    pub fn roots<I>(mut self, roots: I) -> Self
    where
        I: IntoIterator<Item = Root>,
    {
        self.roots = Some(roots.into_iter().collect());
        self
    }

    /// Whether to load tools from the server (default `true`). When `false`,
    /// [`ToolSource::resolve_tools`] returns an empty list without
    /// connecting or performing any round trip — mirrors the Python
    /// reference's `load_tools` constructor flag (`_mcp.py:400-403`). Does
    /// not affect [`Self::tool_definitions`], which always performs a live
    /// fetch regardless of this flag.
    pub fn load_tools(mut self, load_tools: bool) -> Self {
        self.load_tools = load_tools;
        self
    }

    /// Whether to load prompts from the server (default `true`). When
    /// `false`, [`Self::prompts`] returns an empty list without connecting
    /// or performing any round trip — mirrors the Python reference's
    /// `load_prompts` constructor flag (`_mcp.py:400-403`).
    pub fn load_prompts(mut self, load_prompts: bool) -> Self {
        self.load_prompts = load_prompts;
        self
    }

    /// Set a per-request timeout applied while awaiting a response to any
    /// JSON-RPC request sent to this server (`initialize`, `tools/list`,
    /// `tools/call`, ...). Mirrors the Python reference's `request_timeout`
    /// constructor parameter (`_mcp.py:400-403`, there in whole seconds;
    /// here a full [`Duration`]) and [`McpStreamableHttpTool::timeout`]'s
    /// shape for the HTTP transport. Unset (the default) waits indefinitely.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Connect to the server and perform the `initialize` handshake.
    ///
    /// Idempotent and safe to call concurrently: the first caller performs
    /// the handshake, later/concurrent callers await and reuse its result.
    pub async fn connect(&self) -> Result<()> {
        self.session
            .get_or_try_init(|| async {
                let mut transport = McpStdioTransport::spawn(
                    &self.command,
                    &self.args,
                    self.env.as_ref(),
                    self.cwd.as_deref(),
                )
                .await?;
                if let Some(timeout) = self.request_timeout {
                    transport = transport.with_request_timeout(timeout);
                }
                let mut client = McpClient::new(Arc::new(transport));
                if let Some(handler) = &self.sampling_handler {
                    client = client.sampling_handler(handler.clone());
                }
                if let Some(roots) = &self.roots {
                    client = client.roots(roots.clone());
                }
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
    ///
    /// Always performs a live `tools/list` round trip (ignores
    /// [`Self::load_tools`] and any cache) — for a run-time-resolved,
    /// cached, `load_tools`-aware alternative, use this type as a
    /// [`ToolSource`] (its [`ToolSource::resolve_tools`] impl) instead.
    pub async fn tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    /// Connect (if not already connected) and list the server's prompts,
    /// cached after the first successful call until invalidated by a
    /// `notifications/prompts/list_changed` notification from the server
    /// (see [`McpClient::list_prompts_cached`]). Returns an empty list
    /// without a round trip if [`Self::load_prompts`] is `false`, or if the
    /// server didn't declare the `prompts` capability — see
    /// [`McpClient::list_prompts`].
    pub async fn prompts(&self) -> Result<Vec<PromptDescriptor>> {
        if !self.load_prompts {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let prompts = client.list_prompts_cached().await?;
        Ok(dedup_prompts_by_normalized_name(prompts))
    }

    /// Connect (if not already connected) and fetch a rendered prompt's
    /// messages, mapped into core [`ChatMessage`]s — mirrors Python's
    /// `MCPTool.get_prompt`.
    pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<Vec<ChatMessage>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let result = client.get_prompt(name, arguments).await?;
        Ok(result
            .messages
            .iter()
            .map(prompt_message_to_chat_message)
            .collect())
    }

    /// The configured tool-source name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The configured description, if any.
    pub fn description_text(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Close the underlying session, if connected (best effort, idempotent).
    pub async fn close(&self) -> Result<()> {
        if let Some(client) = self.session.get() {
            client.close().await?;
        }
        Ok(())
    }
}

/// Resolved per agent run: lazily connects on first call (reusing
/// [`Self::connect`]) and serves tools from [`McpClient::list_tools_cached`],
/// which is invalidated automatically by a
/// `notifications/tools/list_changed` notification from the server. Returns
/// an empty list without connecting if [`Self::load_tools`] is `false`.
/// Connection/listing failures propagate — see [`ToolSource::resolve_tools`].
#[async_trait]
impl ToolSource for McpStdioTool {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        if !self.load_tools {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools_cached().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// An MCP tool backed by a streamable-HTTP server.
///
/// ```no_run
/// # use agent_framework_mcp::McpStreamableHttpTool;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let mcp = McpStreamableHttpTool::new("web-api", "https://api.example.com/mcp")
///     .headers([("Authorization", "Bearer token")])
///     .description("Web API operations");
/// let tools = mcp.tool_definitions().await?;
/// # let _ = tools;
/// # Ok(())
/// # }
/// ```
pub struct McpStreamableHttpTool {
    name: String,
    description: Option<String>,
    url: String,
    headers: Vec<(String, String)>,
    timeout: Option<Duration>,
    allowed_tools: Option<HashSet<String>>,
    approval_mode: McpApprovalMode,
    sampling_handler: Option<SamplingHandler>,
    roots: Option<Vec<Root>>,
    load_tools: bool,
    load_prompts: bool,
    session: OnceCell<Arc<McpClient>>,
}

impl McpStreamableHttpTool {
    /// Create a tool that talks to the MCP server at `url` over streamable HTTP.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            url: url.into(),
            headers: Vec::new(),
            timeout: None,
            allowed_tools: None,
            approval_mode: McpApprovalMode::default(),
            sampling_handler: None,
            roots: None,
            load_tools: true,
            load_prompts: true,
            session: OnceCell::new(),
        }
    }

    /// Add custom headers (e.g. `Authorization`) sent with every request.
    pub fn headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers
            .extend(headers.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Set a per-request timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a human-readable description for this tool source.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Restrict [`Self::tool_definitions`] to these (local/normalized) tool names.
    pub fn allowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Set the approval policy applied to produced [`ToolDefinition`]s.
    pub fn approval_mode(mut self, mode: McpApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Register the handler for server-initiated `sampling/createMessage`
    /// requests, applied when the underlying session is created. See
    /// [`McpClient::sampling_handler`].
    pub fn sampling_handler(mut self, handler: SamplingHandler) -> Self {
        self.sampling_handler = Some(handler);
        self
    }

    /// Register a static list of filesystem roots, applied when the
    /// underlying session is created. See [`McpClient::roots`].
    pub fn roots<I>(mut self, roots: I) -> Self
    where
        I: IntoIterator<Item = Root>,
    {
        self.roots = Some(roots.into_iter().collect());
        self
    }

    /// Whether to load tools from the server (default `true`). When `false`,
    /// [`ToolSource::resolve_tools`] returns an empty list without
    /// connecting or performing any round trip — mirrors the Python
    /// reference's `load_tools` constructor flag (`_mcp.py:400-403`). Does
    /// not affect [`Self::tool_definitions`], which always performs a live
    /// fetch regardless of this flag.
    pub fn load_tools(mut self, load_tools: bool) -> Self {
        self.load_tools = load_tools;
        self
    }

    /// Whether to load prompts from the server (default `true`). When
    /// `false`, [`Self::prompts`] returns an empty list without connecting
    /// or performing any round trip — mirrors the Python reference's
    /// `load_prompts` constructor flag (`_mcp.py:400-403`).
    pub fn load_prompts(mut self, load_prompts: bool) -> Self {
        self.load_prompts = load_prompts;
        self
    }

    /// Connect to the server and perform the `initialize` handshake.
    ///
    /// Idempotent and safe to call concurrently: the first caller performs
    /// the handshake, later/concurrent callers await and reuse its result.
    pub async fn connect(&self) -> Result<()> {
        self.session
            .get_or_try_init(|| async {
                let header_map = McpStreamableHttpTransport::header_map(&self.headers)?;
                let transport =
                    McpStreamableHttpTransport::new(self.url.clone(), header_map, self.timeout);
                let mut client = McpClient::new(Arc::new(transport));
                if let Some(handler) = &self.sampling_handler {
                    client = client.sampling_handler(handler.clone());
                }
                if let Some(roots) = &self.roots {
                    client = client.roots(roots.clone());
                }
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
    ///
    /// Always performs a live `tools/list` round trip (ignores
    /// [`Self::load_tools`] and any cache) — for a run-time-resolved,
    /// cached, `load_tools`-aware alternative, use this type as a
    /// [`ToolSource`] (its [`ToolSource::resolve_tools`] impl) instead.
    pub async fn tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    /// Connect (if not already connected) and list the server's prompts,
    /// cached after the first successful call until invalidated by a
    /// `notifications/prompts/list_changed` notification from the server
    /// (see [`McpClient::list_prompts_cached`]). Returns an empty list
    /// without a round trip if [`Self::load_prompts`] is `false`, or if the
    /// server didn't declare the `prompts` capability — see
    /// [`McpClient::list_prompts`].
    pub async fn prompts(&self) -> Result<Vec<PromptDescriptor>> {
        if !self.load_prompts {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let prompts = client.list_prompts_cached().await?;
        Ok(dedup_prompts_by_normalized_name(prompts))
    }

    /// Connect (if not already connected) and fetch a rendered prompt's
    /// messages, mapped into core [`ChatMessage`]s — mirrors Python's
    /// `MCPTool.get_prompt`.
    pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<Vec<ChatMessage>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let result = client.get_prompt(name, arguments).await?;
        Ok(result
            .messages
            .iter()
            .map(prompt_message_to_chat_message)
            .collect())
    }

    /// The configured tool-source name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The configured description, if any.
    pub fn description_text(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Close the underlying session, if connected (best effort, idempotent).
    pub async fn close(&self) -> Result<()> {
        if let Some(client) = self.session.get() {
            client.close().await?;
        }
        Ok(())
    }
}

/// Resolved per agent run: lazily connects on first call (reusing
/// [`Self::connect`]) and serves tools from [`McpClient::list_tools_cached`],
/// which is invalidated automatically by a
/// `notifications/tools/list_changed` notification from the server. Returns
/// an empty list without connecting if [`Self::load_tools`] is `false`.
/// Connection/listing failures propagate — see [`ToolSource::resolve_tools`].
#[async_trait]
impl ToolSource for McpStreamableHttpTool {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        if !self.load_tools {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools_cached().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

/// An MCP tool backed by a WebSocket-connected server.
///
/// ```no_run
/// # use agent_framework_mcp::McpWebsocketTool;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let mcp = McpWebsocketTool::new("realtime-service", "wss://service.example.com/mcp")
///     .headers([("Authorization", "Bearer token")])
///     .description("Real-time service operations");
/// let tools = mcp.tool_definitions().await?;
/// # let _ = tools;
/// # Ok(())
/// # }
/// ```
pub struct McpWebsocketTool {
    name: String,
    description: Option<String>,
    url: String,
    headers: Vec<(String, String)>,
    allowed_tools: Option<HashSet<String>>,
    approval_mode: McpApprovalMode,
    sampling_handler: Option<SamplingHandler>,
    roots: Option<Vec<Root>>,
    load_tools: bool,
    load_prompts: bool,
    request_timeout: Option<Duration>,
    session: OnceCell<Arc<McpClient>>,
}

impl McpWebsocketTool {
    /// Create a tool that talks to the MCP server at `url` (`ws://` or
    /// `wss://`) over a WebSocket.
    pub fn new(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: None,
            url: url.into(),
            headers: Vec::new(),
            allowed_tools: None,
            approval_mode: McpApprovalMode::default(),
            sampling_handler: None,
            roots: None,
            load_tools: true,
            load_prompts: true,
            request_timeout: None,
            session: OnceCell::new(),
        }
    }

    /// Add custom headers (e.g. `Authorization`) sent on the WebSocket upgrade
    /// request.
    pub fn headers<I, K, V>(mut self, headers: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.headers
            .extend(headers.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Set a human-readable description for this tool source.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Restrict [`Self::tool_definitions`] to these (local/normalized) tool names.
    pub fn allowed_tools<I, S>(mut self, tools: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Set the approval policy applied to produced [`ToolDefinition`]s.
    pub fn approval_mode(mut self, mode: McpApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Register the handler for server-initiated `sampling/createMessage`
    /// requests, applied when the underlying session is created. See
    /// [`McpClient::sampling_handler`].
    pub fn sampling_handler(mut self, handler: SamplingHandler) -> Self {
        self.sampling_handler = Some(handler);
        self
    }

    /// Register a static list of filesystem roots, applied when the
    /// underlying session is created. See [`McpClient::roots`].
    pub fn roots<I>(mut self, roots: I) -> Self
    where
        I: IntoIterator<Item = Root>,
    {
        self.roots = Some(roots.into_iter().collect());
        self
    }

    /// Whether to load tools from the server (default `true`). When `false`,
    /// [`ToolSource::resolve_tools`] returns an empty list without
    /// connecting or performing any round trip — mirrors the Python
    /// reference's `load_tools` constructor flag (`_mcp.py:400-403`). Does
    /// not affect [`Self::tool_definitions`], which always performs a live
    /// fetch regardless of this flag.
    pub fn load_tools(mut self, load_tools: bool) -> Self {
        self.load_tools = load_tools;
        self
    }

    /// Whether to load prompts from the server (default `true`). When
    /// `false`, [`Self::prompts`] returns an empty list without connecting
    /// or performing any round trip — mirrors the Python reference's
    /// `load_prompts` constructor flag (`_mcp.py:400-403`).
    pub fn load_prompts(mut self, load_prompts: bool) -> Self {
        self.load_prompts = load_prompts;
        self
    }

    /// Set a per-request timeout applied while awaiting a response to any
    /// JSON-RPC request sent to this server (`initialize`, `tools/list`,
    /// `tools/call`, ...). Mirrors the Python reference's `request_timeout`
    /// constructor parameter (`_mcp.py:400-403`, there in whole seconds;
    /// here a full [`Duration`]) and [`McpStreamableHttpTool::timeout`]'s
    /// shape for the HTTP transport. Unset (the default) waits indefinitely.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Connect to the server and perform the `initialize` handshake.
    ///
    /// Idempotent and safe to call concurrently: the first caller performs
    /// the handshake, later/concurrent callers await and reuse its result.
    pub async fn connect(&self) -> Result<()> {
        self.session
            .get_or_try_init(|| async {
                let mut transport =
                    McpWebsocketTransport::connect(&self.url, &self.headers).await?;
                if let Some(timeout) = self.request_timeout {
                    transport = transport.with_request_timeout(timeout);
                }
                let mut client = McpClient::new(Arc::new(transport));
                if let Some(handler) = &self.sampling_handler {
                    client = client.sampling_handler(handler.clone());
                }
                if let Some(roots) = &self.roots {
                    client = client.roots(roots.clone());
                }
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
    ///
    /// Always performs a live `tools/list` round trip (ignores
    /// [`Self::load_tools`] and any cache) — for a run-time-resolved,
    /// cached, `load_tools`-aware alternative, use this type as a
    /// [`ToolSource`] (its [`ToolSource::resolve_tools`] impl) instead.
    pub async fn tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    /// Connect (if not already connected) and list the server's prompts,
    /// cached after the first successful call until invalidated by a
    /// `notifications/prompts/list_changed` notification from the server
    /// (see [`McpClient::list_prompts_cached`]). Returns an empty list
    /// without a round trip if [`Self::load_prompts`] is `false`, or if the
    /// server didn't declare the `prompts` capability — see
    /// [`McpClient::list_prompts`].
    pub async fn prompts(&self) -> Result<Vec<PromptDescriptor>> {
        if !self.load_prompts {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let prompts = client.list_prompts_cached().await?;
        Ok(dedup_prompts_by_normalized_name(prompts))
    }

    /// Connect (if not already connected) and fetch a rendered prompt's
    /// messages, mapped into core [`ChatMessage`]s — mirrors Python's
    /// `MCPTool.get_prompt`.
    pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<Vec<ChatMessage>> {
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let result = client.get_prompt(name, arguments).await?;
        Ok(result
            .messages
            .iter()
            .map(prompt_message_to_chat_message)
            .collect())
    }

    /// The configured tool-source name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The configured description, if any.
    pub fn description_text(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Close the underlying session, if connected (best effort, idempotent).
    pub async fn close(&self) -> Result<()> {
        if let Some(client) = self.session.get() {
            client.close().await?;
        }
        Ok(())
    }
}

/// Resolved per agent run: lazily connects on first call (reusing
/// [`Self::connect`]) and serves tools from [`McpClient::list_tools_cached`],
/// which is invalidated automatically by a
/// `notifications/tools/list_changed` notification from the server. Returns
/// an empty list without connecting if [`Self::load_tools`] is `false`.
/// Connection/listing failures propagate — see [`ToolSource::resolve_tools`].
#[async_trait]
impl ToolSource for McpWebsocketTool {
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        if !self.load_tools {
            return Ok(Vec::new());
        }
        self.connect().await?;
        let client = self
            .session
            .get()
            .expect("connect() initializes the session or returns an error")
            .clone();
        let tools = client.list_tools_cached().await?;
        Ok(build_tool_definitions(
            client,
            &tools,
            self.allowed_tools.as_ref(),
            &self.approval_mode,
        ))
    }

    fn source_name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{PromptDescriptor, ToolDescriptor};
    use serde_json::json;

    fn descriptor(name: &str) -> ToolDescriptor {
        ToolDescriptor {
            name: name.to_string(),
            description: Some(format!("{name} tool")),
            input_schema: json!({"type": "object", "properties": {}}),
            output_schema: None,
        }
    }

    fn prompt_descriptor(name: &str) -> PromptDescriptor {
        PromptDescriptor {
            name: name.to_string(),
            description: Some(format!("{name} prompt")),
            arguments: None,
        }
    }

    fn dummy_client() -> Arc<McpClient> {
        // A transport that is never actually called in these tests (they only
        // exercise definition-building, not invocation).
        struct Unreachable;
        #[async_trait]
        impl crate::transport::McpTransport for Unreachable {
            async fn call(&self, _method: &str, _params: Value) -> Result<Value> {
                unreachable!("not exercised in these tests")
            }
            async fn notify(&self, _method: &str, _params: Value) -> Result<()> {
                Ok(())
            }
            async fn close(&self) -> Result<()> {
                Ok(())
            }
        }
        Arc::new(McpClient::new(Arc::new(Unreachable)))
    }

    #[test]
    fn approval_mode_defaults_to_never_require() {
        let mode = McpApprovalMode::default();
        assert_eq!(mode.resolve("anything"), ApprovalMode::NeverRequire);
    }

    #[test]
    fn approval_mode_always_require_all_applies_to_every_tool() {
        let mode = McpApprovalMode::always_require_all();
        assert_eq!(mode.resolve("a"), ApprovalMode::AlwaysRequire);
        assert_eq!(mode.resolve("b"), ApprovalMode::AlwaysRequire);
    }

    #[test]
    fn approval_mode_per_tool_resolves_each_name() {
        let mode = McpApprovalMode::per_tool(["delete_file"]);
        assert_eq!(mode.resolve("delete_file"), ApprovalMode::AlwaysRequire);
        assert_eq!(mode.resolve("read_file"), ApprovalMode::NeverRequire);
        assert_eq!(mode.resolve("unlisted"), ApprovalMode::NeverRequire);
    }

    #[test]
    fn build_tool_definitions_applies_allowed_tools_filter() {
        let client = dummy_client();
        let tools = vec![descriptor("echo"), descriptor("add"), descriptor("delete")];
        let allowed: HashSet<String> = ["echo", "add"].into_iter().map(String::from).collect();
        let defs =
            build_tool_definitions(client, &tools, Some(&allowed), &McpApprovalMode::default());
        let names: HashSet<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, HashSet::from(["echo", "add"]));
    }

    #[test]
    fn build_tool_definitions_without_filter_returns_all() {
        let client = dummy_client();
        let tools = vec![descriptor("echo"), descriptor("add")];
        let defs = build_tool_definitions(client, &tools, None, &McpApprovalMode::default());
        assert_eq!(defs.len(), 2);
    }

    #[test]
    fn build_tool_definitions_normalizes_names_and_sets_approval() {
        let client = dummy_client();
        let tools = vec![descriptor("weather/get current")];
        let mode = McpApprovalMode::always_require_all();
        let defs = build_tool_definitions(client, &tools, None, &mode);
        assert_eq!(defs[0].name, "weather-get-current");
        assert_eq!(defs[0].approval_mode, ApprovalMode::AlwaysRequire);
        assert!(defs[0].is_executable());
    }

    #[test]
    fn build_tool_definitions_dedups_normalized_name_collision_first_wins() {
        let client = dummy_client();
        // "weather/get" and "weather-get" both normalize to "weather-get" --
        // the second must be dropped, keeping the first's description.
        let tools = vec![descriptor("weather/get"), descriptor("weather-get")];
        let defs = build_tool_definitions(client, &tools, None, &McpApprovalMode::default());
        assert_eq!(defs.len(), 1, "the colliding second tool must be skipped");
        assert_eq!(defs[0].name, "weather-get");
        assert_eq!(defs[0].description, "weather/get tool");
    }

    #[test]
    fn build_tool_definitions_dedup_runs_after_allowed_tools_filter() {
        let client = dummy_client();
        // Both normalize to "echo", but only "echo/one" passes the filter --
        // it must still be produced (the filtered-out "echo two" doesn't
        // count as a prior occurrence).
        let tools = vec![descriptor("echo two"), descriptor("echo/one")];
        let allowed: HashSet<String> = ["echo-one"].into_iter().map(String::from).collect();
        let defs =
            build_tool_definitions(client, &tools, Some(&allowed), &McpApprovalMode::default());
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo-one");
    }

    #[test]
    fn dedup_prompts_by_normalized_name_skips_collision_first_wins() {
        let prompts = vec![
            prompt_descriptor("greet/user"),
            prompt_descriptor("greet-user"),
            prompt_descriptor("farewell"),
        ];
        let deduped = dedup_prompts_by_normalized_name(prompts);
        // Original (un-normalized) names are preserved on the survivors.
        let names: Vec<&str> = deduped.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["greet/user", "farewell"]);
    }

    #[test]
    fn dedup_prompts_by_normalized_name_no_collision_keeps_all() {
        let prompts = vec![prompt_descriptor("greet"), prompt_descriptor("farewell")];
        let deduped = dedup_prompts_by_normalized_name(prompts);
        assert_eq!(deduped.len(), 2);
    }

    #[tokio::test]
    async fn stdio_tool_builder_stores_configuration() {
        let tool = McpStdioTool::new("fs", "npx")
            .args(["-y", "server-filesystem"])
            .env([("KEY", "value")])
            .description("desc")
            .allowed_tools(["read_file"]);
        assert_eq!(tool.name(), "fs");
        assert_eq!(tool.description_text(), Some("desc"));
        assert_eq!(
            tool.args,
            vec!["-y".to_string(), "server-filesystem".to_string()]
        );
        assert_eq!(tool.env.as_ref().unwrap().get("KEY").unwrap(), "value");
        assert!(tool.allowed_tools.as_ref().unwrap().contains("read_file"));
    }

    #[tokio::test]
    async fn http_tool_builder_stores_configuration() {
        let tool = McpStreamableHttpTool::new("api", "https://example.com/mcp")
            .headers([("Authorization", "Bearer x")])
            .timeout(Duration::from_secs(5))
            .description("desc");
        assert_eq!(tool.name(), "api");
        assert_eq!(tool.description_text(), Some("desc"));
        assert_eq!(
            tool.headers,
            vec![("Authorization".to_string(), "Bearer x".to_string())]
        );
        assert_eq!(tool.timeout, Some(Duration::from_secs(5)));
    }

    #[tokio::test]
    async fn websocket_tool_builder_stores_configuration() {
        let tool = McpWebsocketTool::new("realtime", "wss://example.com/mcp")
            .headers([("Authorization", "Bearer x")])
            .description("desc")
            .allowed_tools(["echo"]);
        assert_eq!(tool.name(), "realtime");
        assert_eq!(tool.description_text(), Some("desc"));
        assert_eq!(
            tool.headers,
            vec![("Authorization".to_string(), "Bearer x".to_string())]
        );
        assert!(tool.allowed_tools.as_ref().unwrap().contains("echo"));
    }

    // -- load_tools / load_prompts / request_timeout ----------------------

    #[test]
    fn all_three_wrappers_default_load_tools_and_load_prompts_to_true() {
        let stdio = McpStdioTool::new("s", "cmd");
        assert!(stdio.load_tools);
        assert!(stdio.load_prompts);
        assert!(stdio.request_timeout.is_none());

        let http = McpStreamableHttpTool::new("h", "https://example.com/mcp");
        assert!(http.load_tools);
        assert!(http.load_prompts);

        let ws = McpWebsocketTool::new("w", "wss://example.com/mcp");
        assert!(ws.load_tools);
        assert!(ws.load_prompts);
        assert!(ws.request_timeout.is_none());
    }

    #[test]
    fn stdio_tool_builder_stores_load_flags_and_request_timeout() {
        let tool = McpStdioTool::new("s", "cmd")
            .load_tools(false)
            .load_prompts(false)
            .request_timeout(Duration::from_secs(3));
        assert!(!tool.load_tools);
        assert!(!tool.load_prompts);
        assert_eq!(tool.request_timeout, Some(Duration::from_secs(3)));
    }

    #[test]
    fn http_tool_builder_stores_load_flags() {
        let tool = McpStreamableHttpTool::new("h", "https://example.com/mcp")
            .load_tools(false)
            .load_prompts(false);
        assert!(!tool.load_tools);
        assert!(!tool.load_prompts);
    }

    #[test]
    fn websocket_tool_builder_stores_load_flags_and_request_timeout() {
        let tool = McpWebsocketTool::new("w", "wss://example.com/mcp")
            .load_tools(false)
            .load_prompts(false)
            .request_timeout(Duration::from_secs(7));
        assert!(!tool.load_tools);
        assert!(!tool.load_prompts);
        assert_eq!(tool.request_timeout, Some(Duration::from_secs(7)));
    }

    // -- load_tools=false / load_prompts=false short-circuit (no connect) -
    //
    // `session` is a private `OnceCell`, checked directly (accessible from
    // this child module, same as the field reads in the builder-storage
    // tests above) to prove the short-circuit never attempts a connection
    // at all -- not just that it returns an empty list.

    #[tokio::test]
    async fn stdio_resolve_tools_short_circuits_when_load_tools_false() {
        let tool = McpStdioTool::new("s", "does-not-exist-binary").load_tools(false);
        let resolved = ToolSource::resolve_tools(&tool).await.unwrap();
        assert!(resolved.is_empty());
        assert!(
            tool.session.get().is_none(),
            "load_tools=false must skip connecting entirely"
        );
    }

    #[tokio::test]
    async fn http_resolve_tools_short_circuits_when_load_tools_false() {
        let tool = McpStreamableHttpTool::new("h", "http://127.0.0.1:0/unused").load_tools(false);
        let resolved = ToolSource::resolve_tools(&tool).await.unwrap();
        assert!(resolved.is_empty());
        assert!(tool.session.get().is_none());
    }

    #[tokio::test]
    async fn websocket_resolve_tools_short_circuits_when_load_tools_false() {
        let tool = McpWebsocketTool::new("w", "ws://127.0.0.1:0/unused").load_tools(false);
        let resolved = ToolSource::resolve_tools(&tool).await.unwrap();
        assert!(resolved.is_empty());
        assert!(tool.session.get().is_none());
    }

    #[tokio::test]
    async fn stdio_prompts_short_circuits_when_load_prompts_false() {
        let tool = McpStdioTool::new("s", "does-not-exist-binary").load_prompts(false);
        let prompts = tool.prompts().await.unwrap();
        assert!(prompts.is_empty());
        assert!(
            tool.session.get().is_none(),
            "load_prompts=false must skip connecting entirely"
        );
    }

    #[tokio::test]
    async fn http_prompts_short_circuits_when_load_prompts_false() {
        let tool = McpStreamableHttpTool::new("h", "http://127.0.0.1:0/unused").load_prompts(false);
        let prompts = tool.prompts().await.unwrap();
        assert!(prompts.is_empty());
        assert!(tool.session.get().is_none());
    }

    #[tokio::test]
    async fn websocket_prompts_short_circuits_when_load_prompts_false() {
        let tool = McpWebsocketTool::new("w", "ws://127.0.0.1:0/unused").load_prompts(false);
        let prompts = tool.prompts().await.unwrap();
        assert!(prompts.is_empty());
        assert!(tool.session.get().is_none());
    }

    #[test]
    fn tool_source_name_matches_configured_name() {
        let stdio = McpStdioTool::new("stdio-name", "cmd");
        assert_eq!(ToolSource::source_name(&stdio), "stdio-name");
        let http = McpStreamableHttpTool::new("http-name", "https://example.com/mcp");
        assert_eq!(ToolSource::source_name(&http), "http-name");
        let ws = McpWebsocketTool::new("ws-name", "wss://example.com/mcp");
        assert_eq!(ToolSource::source_name(&ws), "ws-name");
    }
}
