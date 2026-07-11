//! [`McpClient`]: JSON-RPC/MCP methods layered over any [`McpTransport`].

use std::sync::{Arc, RwLock as StdRwLock};

use serde_json::{json, Value};
use tokio::sync::RwLock;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{
    CallToolResult, GetPromptResult, Implementation, InitializeResult, ListPromptsResult,
    ListToolsResult, PromptDescriptor, ToolDescriptor, COMPATIBLE_PROTOCOL_VERSIONS,
    PROTOCOL_VERSION,
};
use crate::sampling::{dispatch_server_request, Root, SamplingHandler, ServerRequestHandlers};
use crate::transport::McpTransport;

/// Safety cap on `tools/list`/`prompts/list` pagination, so a server that
/// never stops returning a `nextCursor` cannot spin the client forever.
const MAX_LIST_PAGES: usize = 10_000;

/// A connected MCP session: `initialize`, `ping`, `tools/list`, `tools/call`,
/// `prompts/list`, `prompts/get`, layered over any [`McpTransport`], plus
/// server-initiated `sampling/createMessage` / `roots/list` support.
///
/// `McpClient` is transport-agnostic: construct it with an `Arc<dyn McpTransport>`
/// (see [`crate::McpStdioTransport`] / [`crate::McpStreamableHttpTransport`]), or
/// use [`crate::McpStdioTool`] / [`crate::McpStreamableHttpTool`], which each own
/// one for you and hand out ready-to-use [`agent_framework_core::tools::ToolDefinition`]s.
pub struct McpClient {
    transport: Arc<dyn McpTransport>,
    initialize_result: RwLock<Option<InitializeResult>>,
    handlers: Arc<StdRwLock<ServerRequestHandlers>>,
}

impl McpClient {
    /// Wrap a transport in a fresh, uninitialized session.
    ///
    /// Installs the server-request dispatcher into `transport` immediately
    /// (see [`McpTransport::set_server_request_handler`]), so
    /// [`Self::sampling_handler`] / [`Self::roots`] take effect for any
    /// server-initiated request the transport sees from this point on, even
    /// ones that arrive before [`Self::initialize`] completes.
    pub fn new(transport: Arc<dyn McpTransport>) -> Self {
        let handlers: Arc<StdRwLock<ServerRequestHandlers>> = Arc::default();
        let dispatch_handlers = handlers.clone();
        transport.set_server_request_handler(Arc::new(move |method: String, params: Value| {
            let handlers = dispatch_handlers.clone();
            Box::pin(async move { dispatch_server_request(&handlers, &method, params).await })
        }));
        Self {
            transport,
            initialize_result: RwLock::new(None),
            handlers,
        }
    }

    /// Register the handler for server-initiated `sampling/createMessage`
    /// requests. Declares the `sampling` capability on the next
    /// [`Self::initialize`] call — set this up before initializing, matching
    /// the `mcp` Python SDK, which derives the capability from whichever
    /// callback was passed to the session at construction time.
    pub fn sampling_handler(self, handler: SamplingHandler) -> Self {
        self.handlers.write().unwrap().sampling = Some(handler);
        self
    }

    /// Register a static list of filesystem roots, answered when the server
    /// sends `roots/list`. Declares the `roots` capability on the next
    /// [`Self::initialize`] call, the same way [`Self::sampling_handler`] does.
    pub fn roots(self, roots: Vec<Root>) -> Self {
        self.handlers.write().unwrap().roots = Some(roots);
        self
    }

    /// The `capabilities` object sent during `initialize`: `sampling`/`roots`
    /// are included only when a handler is registered via
    /// [`Self::sampling_handler`] / [`Self::roots`] — matching the `mcp`
    /// Python SDK, which derives `ClientCapabilities` from whichever
    /// callbacks were supplied. Unlike that SDK (which always sets
    /// `roots.listChanged: true` whenever any roots callback is present,
    /// regardless of whether it ever actually sends that notification),
    /// this always advertises `listChanged: false`: this crate's root list
    /// is static, so that's the honest answer.
    fn client_capabilities(&self) -> Value {
        let handlers = self.handlers.read().unwrap();
        let mut capabilities = serde_json::Map::new();
        if handlers.sampling.is_some() {
            capabilities.insert("sampling".to_string(), json!({}));
        }
        if handlers.roots.is_some() {
            capabilities.insert("roots".to_string(), json!({ "listChanged": false }));
        }
        Value::Object(capabilities)
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
            "capabilities": self.client_capabilities(),
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

    /// List every prompt the server exposes, transparently following
    /// `nextCursor` pagination until the server stops returning one.
    ///
    /// Short-circuits to an empty list, without issuing any request, if the
    /// server didn't declare the `prompts` capability during `initialize` —
    /// the Rust equivalent of the Python reference's try/except around
    /// `session.list_prompts()` (which logs and treats a failure the same
    /// way), checking the negotiated capability up front instead of
    /// discarding an expected error.
    pub async fn list_prompts(&self) -> Result<Vec<PromptDescriptor>> {
        self.require_initialized().await?;
        if !self.supports_prompts().await {
            return Ok(Vec::new());
        }
        let mut prompts = Vec::new();
        let mut cursor: Option<String> = None;
        for _ in 0..MAX_LIST_PAGES {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let raw = self.transport.call("prompts/list", params).await?;
            let page: ListPromptsResult = serde_json::from_value(raw)
                .map_err(|e| Error::service(format!("invalid MCP prompts/list response: {e}")))?;
            prompts.extend(page.prompts);
            match page.next_cursor {
                Some(next) if !next.is_empty() => cursor = Some(next),
                _ => return Ok(prompts),
            }
        }
        Err(Error::service(
            "MCP prompts/list pagination exceeded the maximum page count",
        ))
    }

    /// `prompts/get` — fetch a rendered prompt's messages.
    ///
    /// `arguments` should be a JSON object of string values per the MCP
    /// spec (`GetPromptRequest.params.arguments?: {[key: string]: string}`);
    /// `Value::Null` is sent as `{}`, mirroring [`Self::call_tool`].
    ///
    /// Unlike [`Self::list_prompts`], this does not pre-check the `prompts`
    /// capability: a caller invoking a specific prompt by name presumably
    /// already knows it exists, and a server that doesn't support prompts at
    /// all will simply answer with a JSON-RPC error, surfaced here as
    /// [`Error::Service`] — mirroring how the Python reference's
    /// `get_prompt` lets the underlying request fail naturally.
    pub async fn get_prompt(&self, name: &str, arguments: Value) -> Result<GetPromptResult> {
        self.require_initialized().await?;
        let arguments = if arguments.is_null() {
            json!({})
        } else {
            arguments
        };
        let params = json!({ "name": name, "arguments": arguments });
        let raw = self.transport.call("prompts/get", params).await?;
        serde_json::from_value(raw)
            .map_err(|e| Error::service(format!("invalid MCP prompts/get response: {e}")))
    }

    /// Whether the server declared the `prompts` capability during
    /// [`Self::initialize`]. `false` before `initialize` has completed.
    pub async fn supports_prompts(&self) -> bool {
        self.initialize_result
            .read()
            .await
            .as_ref()
            .map(InitializeResult::supports_prompts)
            .unwrap_or(false)
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
    use crate::sampling::CreateMessageResult;
    use async_trait::async_trait;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Mutex;

    /// A canned, in-memory transport for exercising [`McpClient`] without any I/O.
    struct MockTransport {
        responses: Mutex<HashMap<String, VecDeque<Value>>>,
        /// Every `(method, params)` pair passed to `call`, in order — lets
        /// tests assert on what a higher-level `McpClient` method actually
        /// sent (e.g. the `capabilities` object during `initialize`).
        sent: Mutex<Vec<(String, Value)>>,
    }

    impl MockTransport {
        fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                sent: Mutex::new(Vec::new()),
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

        /// The params of the most recent `call(method, ..)`, if any.
        fn last_params(&self, method: &str) -> Option<Value> {
            self.sent
                .lock()
                .unwrap()
                .iter()
                .rev()
                .find(|(m, _)| m == method)
                .map(|(_, p)| p.clone())
        }
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn call(&self, method: &str, params: Value) -> Result<Value> {
            self.sent.lock().unwrap().push((method.to_string(), params));
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

    // -- Capability declaration ------------------------------------------

    #[tokio::test]
    async fn initialize_omits_sampling_and_roots_capabilities_by_default() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"], json!({}));
    }

    #[tokio::test]
    async fn initialize_declares_sampling_capability_only_when_handler_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { Ok(CreateMessageResult::text("assistant", "hi", "m")) }));
        let client = McpClient::new(mock.clone()).sampling_handler(handler);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"], json!({"sampling": {}}));
    }

    #[tokio::test]
    async fn initialize_declares_roots_capability_only_when_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone()).roots(vec![Root::new("file:///tmp")]);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(
            sent["capabilities"],
            json!({"roots": {"listChanged": false}})
        );
    }

    #[tokio::test]
    async fn initialize_declares_both_capabilities_when_both_set() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let handler: SamplingHandler =
            Arc::new(|_| Box::pin(async { Ok(CreateMessageResult::text("assistant", "hi", "m")) }));
        let client = McpClient::new(mock.clone())
            .sampling_handler(handler)
            .roots(vec![Root::new("file:///tmp")]);
        client.initialize("c", "1").await.unwrap();
        let sent = mock.last_params("initialize").unwrap();
        assert_eq!(sent["capabilities"]["sampling"], json!({}));
        assert_eq!(sent["capabilities"]["roots"]["listChanged"], json!(false));
    }

    // -- Prompts ----------------------------------------------------------

    fn init_result_with_prompts() -> Value {
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {"prompts": {}},
            "serverInfo": {"name": "mock-server", "version": "0.0.1"},
        })
    }

    #[tokio::test]
    async fn list_prompts_short_circuits_without_prompts_capability() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result());
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        assert!(!client.supports_prompts().await);
        let prompts = client.list_prompts().await.unwrap();
        assert!(prompts.is_empty());
        // No `prompts/list` request should ever have been sent.
        assert!(mock.last_params("prompts/list").is_none());
    }

    #[tokio::test]
    async fn list_prompts_follows_cursor_pagination_when_supported() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push(
            "prompts/list",
            json!({
                "prompts": [{"name": "greet"}],
                "nextCursor": "page2",
            }),
        );
        mock.push(
            "prompts/list",
            json!({
                "prompts": [{"name": "farewell"}],
            }),
        );
        let client = McpClient::new(mock);
        client.initialize("c", "1").await.unwrap();
        assert!(client.supports_prompts().await);
        let prompts = client.list_prompts().await.unwrap();
        assert_eq!(
            prompts.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
            vec!["greet", "farewell"]
        );
    }

    #[tokio::test]
    async fn get_prompt_maps_messages() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push(
            "prompts/get",
            json!({
                "description": "A greeting",
                "messages": [
                    {"role": "user", "content": {"type": "text", "text": "Say hi to Ada"}},
                ],
            }),
        );
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        let result = client
            .get_prompt("greet", json!({"name": "Ada"}))
            .await
            .unwrap();
        assert_eq!(result.messages.len(), 1);
        assert_eq!(result.messages[0].role, "user");
        let sent = mock.last_params("prompts/get").unwrap();
        assert_eq!(sent["name"], "greet");
        assert_eq!(sent["arguments"]["name"], "Ada");
    }

    #[tokio::test]
    async fn get_prompt_null_arguments_sent_as_empty_object() {
        let mock = Arc::new(MockTransport::new());
        mock.push("initialize", init_result_with_prompts());
        mock.push("prompts/get", json!({"messages": []}));
        let client = McpClient::new(mock.clone());
        client.initialize("c", "1").await.unwrap();
        client.get_prompt("greet", Value::Null).await.unwrap();
        let sent = mock.last_params("prompts/get").unwrap();
        assert_eq!(sent["arguments"], json!({}));
    }
}
