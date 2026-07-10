//! [`McpClient`]: JSON-RPC/MCP methods layered over any [`McpTransport`].

use std::sync::Arc;

use serde_json::{json, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{
    CallToolResult, Implementation, InitializeResult, ListToolsResult, ToolDescriptor,
    COMPATIBLE_PROTOCOL_VERSIONS, PROTOCOL_VERSION,
};
use crate::transport::McpTransport;

/// Safety cap on `tools/list` pagination, so a server that never stops
/// returning a `nextCursor` cannot spin the client forever.
const MAX_LIST_PAGES: usize = 10_000;

/// A connected MCP session: `initialize`, `ping`, `tools/list`, `tools/call`,
/// layered over any [`McpTransport`].
///
/// `McpClient` is transport-agnostic: construct it with an `Arc<dyn McpTransport>`
/// (see [`crate::McpStdioTransport`] / [`crate::McpStreamableHttpTransport`]), or
/// use [`crate::McpStdioTool`] / [`crate::McpStreamableHttpTool`], which each own
/// one for you and hand out ready-to-use [`agent_framework_core::tools::ToolDefinition`]s.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    initialize_result: RwLock<Option<InitializeResult>>,
}

impl McpClient {
    /// Wrap a transport in a fresh, uninitialized session.
    pub fn new(transport: Arc<dyn McpTransport>) -> Self {
        Self {
            transport,
            initialize_result: RwLock::new(None),
        }
    }

    /// Perform the MCP `initialize` request followed by the
    /// `notifications/initialized` notification.
    ///
    /// Idempotent: once a handshake has succeeded, later calls return the
    /// cached result without re-sending anything. The client always *sends*
    /// [`PROTOCOL_VERSION`], but accepts whatever `protocolVersion` the
    /// server responds with — including an older one — logging a warning if
    /// it isn't one this client specifically recognizes.
    pub async fn initialize(
        &self,
        client_name: &str,
        client_version: &str,
    ) -> Result<InitializeResult> {
        if let Some(existing) = self.initialize_result.read().await.clone() {
            return Ok(existing);
        }

        let params = json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": Implementation {
                name: client_name.to_string(),
                version: client_version.to_string(),
            },
        });
        let raw = self.transport.call("initialize", params).await?;
        let result: InitializeResult = serde_json::from_value(raw)
            .map_err(|e| Error::service(format!("invalid MCP initialize response: {e}")))?;

        if !COMPATIBLE_PROTOCOL_VERSIONS.contains(&result.protocol_version.as_str()) {
            tracing::warn!(
                server_protocol_version = %result.protocol_version,
                client_protocol_version = PROTOCOL_VERSION,
                "MCP server negotiated an unrecognized protocol version; proceeding anyway"
            );
        }

        self.transport
            .notify("notifications/initialized", json!({}))
            .await?;

        *self.initialize_result.write().await = Some(result.clone());
        Ok(result)
    }

    /// `ping` — verify the server is still responsive.
    pub async fn ping(&self) -> Result<()> {
        self.require_initialized().await?;
        self.transport.call("ping", json!({})).await?;
        Ok(())
    }

    /// List every tool the server exposes, transparently following
    /// `nextCursor` pagination until the server stops returning one.
    pub async fn list_tools(&self) -> Result<Vec<ToolDescriptor>> {
        self.require_initialized().await?;
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let raw = self.transport.call("tools/list", params).await?;
            let page: ListToolsResult = serde_json::from_value(raw)
                .map_err(|e| Error::service(format!("invalid MCP tools/list response: {e}")))?;
            tools.extend(page.tools);
            match page.next_cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => return Ok(tools),
            }
        }
        Err(Error::service(
            "MCP tools/list pagination exceeded the maximum page count",
        ))
    }

    /// `tools/call` — invoke a tool and return the raw result, including
    /// `is_error`, without converting an error result into an `Err`.
    pub async fn call_tool(&self, name: &str, arguments: Value) -> Result<CallToolResult> {
        self.require_initialized().await?;
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        let params = json!({ "name": name, "arguments": arguments });
        let raw = self.transport.call("tools/call", params).await?;
        Ok(CallToolResult::from_value(&raw))
    }

    /// `tools/call`, mapped to a single JSON value suitable for handing back
    /// to a model: a lone text block becomes a string (or the value it parses
    /// as, if it's valid JSON); `isError: true` becomes `Err(Error::Tool(..))`.
    pub async fn call_tool_value(&self, name: &str, arguments: Value) -> Result<Value> {
        let result = self.call_tool(name, arguments).await?;
        if result.is_error {
            return Err(Error::tool(result.error_message()));
        }
        Ok(result.to_value())
    }

    /// The server's `serverInfo`, once [`Self::initialize`] has completed.
    pub async fn server_info(&self) -> Option<Implementation> {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(|r| r.server_info.clone())
    }

    /// The protocol version negotiated with the server, once
    /// [`Self::initialize`] has completed.
    pub async fn protocol_version(&self) -> Option<String> {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(|r| r.protocol_version.clone())
    }

    /// Whether [`Self::initialize`] has completed successfully.
    pub async fn is_initialized(&self) -> bool {
        self.initialize_result.read().await.is_some()
    }

    /// Gracefully close the underlying transport (best effort, idempotent):
    /// kills the child process for stdio, or `DELETE`s the session for
    /// streamable HTTP.
    pub async fn close(&self) -> Result<()> {
        self.transport.close().await
    }

    async fn require_initialized(&self) -> Result<()> {
        if self.is_initialized().await {
            Ok(())
        } else {
            Err(Error::service(
                "MCP client is not initialized; call initialize() first",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    /// A canned, in-memory transport for exercising [`McpClient`] without any I/O.
    struct MockTransport {
        responses: Mutex<HashMap<String, VecDeque<Value>>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
            }
        }

        fn push(&self, method: &str, value: Value) {
            self.responses
                .lock()
                .unwrap()
                .entry(method.to_string())
                .or_default()
                .push_back(value);
        }
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn call(&self, method: &str, _params: Value) -> Result<Value> {
            let mut guard = self.responses.lock().unwrap();
            let queue = guard
                .get_mut(method)
                .unwrap_or_else(|| panic!("no canned responses queued for method '{method}'"));
            queue
                .pop_front()
                .ok_or_else(|| Error::service(format!("no more canned responses for {method}")))
        }

        async fn notify(&self, _method: &str, _params: Value) -> Result<()> {
            Ok(())
        }

        async fn close(&self) -> Result<()> {
            Ok(())
        }
    }

    fn init_result() -> Value {
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "serverInfo": {"name": "mock-server", "version": "0.0.1"},
        })
    }

    #[tokio::test]
    async fn initialize_caches_result_and_is_idempotent() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock);

        let first = client.initialize("c", "1").await.unwrap();
        assert_eq!(first.server_info.name, "mock-server");

        // Second call must not attempt to pull another canned response (there
        // isn't one) — it should just return the cached result.
        let second = client.initialize("c", "1").await.unwrap();
        assert_eq!(second.protocol_version, "2025-06-18");
    }

    #[tokio::test]
    async fn methods_require_initialize_first() {
        let mock = Arc::new(MockTransport::new());
        let client = McpClient::new(mock);
        let err = client.list_tools().await.unwrap_err();
        assert!(matches!(err, Error::Service(_)));
    }

    #[tokio::test]
    async fn list_tools_follows_cursor_pagination() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/list",
            json!({
                "tools": [{"name": "a", "inputSchema": {"type":"object","properties":{}}}],
                "nextCursor": "page2",
            }),
        );
        mock.push(
            "tools/list",
            json!({
                "tools": [{"name": "b", "inputSchema": {"type":"object","properties":{}}}],
            }),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(
            tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[tokio::test]
    async fn call_tool_value_maps_single_text_block_to_string() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "hello world"}], "isError": false}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let value = client
            .call_tool_value("echo", json!({"text": "hello world"}))
            .await
            .unwrap();
        assert_eq!(value, json!("hello world"));
    }

    #[tokio::test]
    async fn call_tool_value_parses_json_text_block() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "42"}], "isError": false}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let value = client
            .call_tool_value("add", json!({"a": 40, "b": 2}))
            .await
            .unwrap();
        assert_eq!(value, json!(42));
    }

    #[tokio::test]
    async fn call_tool_value_maps_is_error_to_tool_error() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push(
            "tools/call",
            json!({"content": [{"type": "text", "text": "boom"}], "isError": true}),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        let err = client
            .call_tool_value("broken", json!({}))
            .await
            .unwrap_err();
        match err {
            Error::Tool(msg) => assert_eq!(msg, "boom"),
            other => panic!("expected Error::Tool, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn ping_and_server_info_after_initialize() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        mock.push("ping", json!({}));
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        client.ping().await.unwrap();
        let info = client.server_info().await.unwrap();
        assert_eq!(info.name, "mock-server");
        assert_eq!(client.protocol_version().await.unwrap(), "2025-06-18");
    }
}
