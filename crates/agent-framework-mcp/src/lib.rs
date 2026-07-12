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
//! pagination), `tools/call`, and `notifications/tools/list_changed` /
//! `notifications/prompts/list_changed` (see "Dynamic tools" below).
//!
//! ## Dynamic tools ([`ToolSource`](agent_framework_core::tools::ToolSource))
//!
//! [`McpStdioTool`], [`McpStreamableHttpTool`], and [`McpWebsocketTool`] all
//! implement `ToolSource`, so `ChatAgent::builder().tool_source(Arc::new(mcp))`
//! resolves the server's tool list fresh on every agent run instead of
//! freezing it at build time (the alternative, `.tool_definitions().await`
//! once up front). Resolution connects lazily on first use and thereafter
//! serves [`McpClient::list_tools_cached`]'s cached result, which
//! self-invalidates when the server sends
//! `notifications/tools/list_changed` (likewise `.prompts()` and
//! [`McpClient::list_prompts_cached`] for `notifications/prompts/list_changed`).
//! Set `.load_tools(false)` / `.load_prompts(false)` (default `true`) to skip
//! loading either category entirely, without a round trip. A connection or
//! listing failure during resolution propagates out of the agent run rather
//! than being swallowed.
//!
//! ## Prompts, sampling, and roots
//!
//! - **Prompts** (`prompts/list` / `prompts/get`): [`McpClient::list_prompts`]
//!   / [`McpClient::get_prompt`], and on each tool wrapper, `.prompts()` /
//!   `.get_prompt(name, arguments)` (mapping MCP `PromptMessage`s into core
//!   `Message`s, mirroring Python's `MCPTool.get_prompt`).
//!   `list_prompts`/`.prompts()` short-circuit to an empty list — without
//!   issuing any request — when the server didn't declare the `prompts`
//!   capability during `initialize`.
//! - **Sampling** (server-initiated `sampling/createMessage`): register a
//!   [`SamplingHandler`] via `.sampling_handler(..)` on [`McpClient`] or any
//!   tool wrapper; [`chat_client_sampling_handler`] adapts any `ChatClient`
//!   into one. The `sampling` capability is declared during `initialize`
//!   only when a handler is registered, matching the `mcp` Python SDK
//!   (which derives `ClientCapabilities` from whichever callbacks were
//!   supplied at `ClientSession` construction). All three transports route
//!   a server-initiated request to the registered handler and write the
//!   JSON-RPC response back themselves; `ping` is always answered with an
//!   empty result, and an unhandled/unknown method gets a JSON-RPC "method
//!   not found" error response rather than silence.
//! - **Roots** (server-initiated `roots/list`): register a static list via
//!   `.roots(vec![Root::new("file:///...")])` on [`McpClient`] or any tool
//!   wrapper; the `roots` capability is declared during `initialize` only
//!   when set. Static-only — there is no `notifications/roots/list_changed`
//!   support, so `listChanged` is always advertised as `false`. Note: the
//!   upstream Python `agent_framework` package never wires this up at all
//!   (even though the underlying `mcp` Python SDK it depends on supports
//!   it) — this is a case where the Rust port exceeds Python parity; see
//!   `PARITY.md`.
//!
//! ## Not implemented (future work)
//!
//! - **Standalone GET-based SSE listening** for the streamable HTTP
//!   transport (a persistent stream the server opens unprompted, outside of
//!   any request/response cycle). A server-initiated request embedded in
//!   the SSE response to an *active* `call()` **is** routed to the
//!   registered handler; only that separate, connection-initiated-by-the-
//!   server stream is unsupported.
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
mod sampling;
mod tool;
mod transport;

pub use client::McpClient;
pub use protocol::{
    CallToolResult, ContentBlock, GetPromptResult, Implementation, InitializeResult,
    ListPromptsResult, ListToolsResult, PromptArgument, PromptDescriptor, PromptMessage, RpcError,
    ToolDescriptor, COMPATIBLE_PROTOCOL_VERSIONS, PROTOCOL_VERSION,
};
pub use sampling::{
    chat_client_sampling_handler, BoxedServerRequestHandler, CreateMessageParams,
    CreateMessageResult, Root, SamplingHandler, SamplingMessage,
};
pub use tool::{McpApprovalMode, McpStdioTool, McpStreamableHttpTool, McpWebsocketTool};
pub use transport::{
    McpStdioTransport, McpStreamableHttpTransport, McpTransport, McpWebsocketTransport,
};
