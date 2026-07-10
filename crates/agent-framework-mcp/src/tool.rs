//! High-level MCP tools: [`McpStdioTool`] and [`McpStreamableHttpTool`].
//!
//! Each owns a lazily-connected, shared [`McpClient`] and turns the server's
//! tool catalog into ready-to-use [`ToolDefinition`]s whose executors call
//! back into that shared session — hand them straight to
//! `ChatAgent::builder().tools(..)`.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::OnceCell;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::{ApprovalMode, Tool, ToolDefinition};

use crate::client::McpClient;
use crate::protocol::{normalize_mcp_name, ToolDescriptor};
use crate::transport::{McpStdioTransport, McpStreamableHttpTransport, McpWebsocketTransport};

/// The `clientInfo.name` this crate sends during `initialize`.
const CLIENT_NAME: &str = "agent-framework-rs";
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
fn build_tool_definitions(
    client: Arc<McpClient>,
    tools: &[ToolDescriptor],
    allowed_tools: Option<&HashSet<String>>,
    approval_mode: &McpApprovalMode,
) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| (t, normalize_mcp_name(&t.name)))
        .filter(|(_, local_name)| {
            allowed_tools
                .map(|allowed| allowed.contains(local_name.as_str()))
                .unwrap_or(true)
        })
        .map(|(descriptor, local_name)| {
            let executor: Arc<dyn Tool> = Arc::new(McpToolExecutor {
                local_name: local_name.clone(),
                description: descriptor.description.clone().unwrap_or_default(),
                parameters: descriptor.input_schema.clone(),
                remote_name: descriptor.name.clone(),
                client: client.clone(),
            });
            let mut definition = ToolDefinition::from_tool(executor);
            definition.approval_mode = approval_mode.resolve(&local_name);
            definition
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

    /// Connect to the server and perform the `initialize` handshake.
    ///
    /// Idempotent and safe to call concurrently: the first caller performs
    /// the handshake, later/concurrent callers await and reuse its result.
    pub async fn connect(&self) -> Result<()> {
        self.session
            .get_or_try_init(|| async {
                let transport = McpStdioTransport::spawn(
                    &self.command,
                    &self.args,
                    self.env.as_ref(),
                    self.cwd.as_deref(),
                )
                .await?;
                let client = McpClient::new(Arc::new(transport));
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
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
                let client = McpClient::new(Arc::new(transport));
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
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

    /// Connect to the server and perform the `initialize` handshake.
    ///
    /// Idempotent and safe to call concurrently: the first caller performs
    /// the handshake, later/concurrent callers await and reuse its result.
    pub async fn connect(&self) -> Result<()> {
        self.session
            .get_or_try_init(|| async {
                let transport = McpWebsocketTransport::connect(&self.url, &self.headers).await?;
                let client = McpClient::new(Arc::new(transport));
                client.initialize(CLIENT_NAME, CLIENT_VERSION).await?;
                Ok::<_, Error>(Arc::new(client))
            })
            .await?;
        Ok(())
    }

    /// Connect (if not already connected) and return one [`ToolDefinition`]
    /// per server tool that passes the [`Self::allowed_tools`] filter, each
    /// wired to call back through the shared session.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::ToolDescriptor;
    use serde_json::json;

    fn descriptor(name: &str) -> ToolDescriptor {
        ToolDescriptor {
            name: name.to_string(),
            description: Some(format!("{name} tool")),
            input_schema: json!({"type": "object", "properties": {}}),
            output_schema: None,
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
}
