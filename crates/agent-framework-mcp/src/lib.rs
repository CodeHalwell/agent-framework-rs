//! # agent-framework-mcp
//!
//! A Model Context Protocol (MCP) client for `agent-framework-rs`. Connects to
//! MCP servers, lists their tools, and turns them into
//! [`ToolDefinition`](agent_framework_core::tools::ToolDefinition)s that plug
//! straight into a `ChatAgent`.
//!
//! This is the Rust equivalent of `agent_framework._mcp` (`MCPStdioTool`,
//! `MCPStreamableHTTPTool`) in the Python reference implementation.
//!
//! ## Transports
//!
//! - [`McpStdioTool`] / [`McpStdioTransport`] — spawns the server as a child
//!   process and speaks newline-delimited JSON-RPC over its stdin/stdout.
//! - [`McpStreamableHttpTool`] / [`McpStreamableHttpTransport`] — POSTs
//!   JSON-RPC messages to an HTTP endpoint, accepting either a single
//!   `application/json` response or a `text/event-stream` response scanned
//!   for the matching reply.
//! - [`McpWebsocketTool`] / [`McpWebsocketTransport`] — connects over a
//!   WebSocket (`ws://` or `wss://`) using the `"mcp"` subprotocol, framing
//!   each JSON-RPC message as one text frame.
//!
//! ## Protocol coverage
//!
//! Speaks MCP protocol version `2025-06-18` during `initialize`, and accepts
//! (without rejecting) an older version the server negotiates down to, such
//! as `2025-03-26` or `2024-11-05`. Implements the `initialize` /
//! `notifications/initialized` handshake, `ping`, `tools/list` (with cursor
//! pagination), and `tools/call`.
//!
//! ## Not implemented (future work)
//!
//! - **Prompts** (`prompts/list` / `prompts/get`). Only tools are exposed as
//!   agent functions.
//! - **Sampling / roots callbacks.** This client does not act on
//!   server-initiated requests (e.g. a server asking the client to run a
//!   completion); such requests are logged and ignored rather than answered.
//! - **Standalone GET-based SSE listening** for the streamable HTTP
//!   transport (server-initiated messages outside of a request/response
//!   cycle).
//! - **Automatic reconnect** on a broken pipe/connection; failures are
//!   surfaced as clear errors instead.
//!
//! ## Example
//!
//! ```no_run
//! use agent_framework_core::prelude::*;
//! use agent_framework_mcp::McpStdioTool;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let mcp = McpStdioTool::new("filesystem", "npx")
//!     .args(["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
//!     .description("Local filesystem access");
//!
//! // Connects (if needed) and lists the server's tools as ToolDefinitions.
//! let tools = mcp.tool_definitions().await?;
//!
//! let agent = ChatAgent::builder(client)
//!     .name("assistant")
//!     .instructions("You can read files when needed.")
//!     .tools(tools)
//!     .build();
//!
//! let response = agent.run_once("List the files in /tmp").await?;
//! println!("{}", response.text());
//!
//! mcp.close().await?;
//! # Ok(())
//! # }
//! ```
//!
//! A [`McpWebsocketTool`] is built the same way, given a `ws://`/`wss://` URL:
//!
//! ```no_run
//! use agent_framework_mcp::McpWebsocketTool;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let mcp = McpWebsocketTool::new("realtime-service", "wss://service.example.com/mcp")
//!     .headers([("Authorization", "Bearer token")])
//!     .description("Real-time service operations");
//!
//! let tools = mcp.tool_definitions().await?;
//! # let _ = tools;
//! mcp.close().await?;
//! # Ok(())
//! # }
//! ```

mod client;
mod protocol;
mod tool;
mod transport;

pub use client::McpClient;
pub use protocol::{
    CallToolResult, ContentBlock, Implementation, InitializeResult, ListToolsResult, RpcError,
    ToolDescriptor, COMPATIBLE_PROTOCOL_VERSIONS, PROTOCOL_VERSION,
};
pub use tool::{McpApprovalMode, McpStdioTool, McpStreamableHttpTool, McpWebsocketTool};
pub use transport::{
    McpStdioTransport, McpStreamableHttpTransport, McpTransport, McpWebsocketTransport,
};
