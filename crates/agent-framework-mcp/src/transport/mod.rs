//! Pluggable JSON-RPC transports for MCP.
//!
//! [`McpTransport`] hides how a JSON-RPC message actually reaches the server
//! (a child process's stdio, an HTTP POST, …) behind two operations: a
//! correlated request/response `call`, and a fire-and-forget `notify`.
//! [`crate::McpClient`] is written entirely against this trait.

pub mod http;
pub mod stdio;
pub mod websocket;

use async_trait::async_trait;
use serde_json::Value;

use agent_framework_core::error::Result;

/// A JSON-RPC 2.0 transport for MCP messages.
///
/// Implementations own request-id bookkeeping and response correlation, and
/// any background I/O required to notice server-initiated messages (they are
/// logged and otherwise ignored — sampling/roots are not supported).
#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and return its `result` value.
    ///
    /// Resolves to `Err` if the server returns a JSON-RPC error response, or
    /// if the underlying transport fails before a response is received.
    async fn call(&self, method: &str, params: Value) -> Result<Value>;

    /// Send a JSON-RPC notification. No response is expected or awaited.
    async fn notify(&self, method: &str, params: Value) -> Result<()>;

    /// Best-effort, idempotent shutdown of the transport.
    async fn close(&self) -> Result<()>;
}

pub use http::McpStreamableHttpTransport;
pub use stdio::McpStdioTransport;
pub use websocket::McpWebsocketTransport;
