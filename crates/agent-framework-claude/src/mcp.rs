//! The in-process MCP server that exposes the agent's function tools to the
//! CLI (the SDK's `create_sdk_mcp_server`). The CLI only takes custom tools
//! over MCP, so they are declared as an `sdk` server in `--mcp-config` and
//! the CLI tunnels the server's JSON-RPC through `mcp_message` control
//! requests.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use agent_framework_core::types::{FunctionArguments, FunctionCallContent};
use serde_json::{json, Map, Value};

use crate::options::FunctionApprovalHandler;

/// The server's name (upstream's `TOOLS_MCP_SERVER_NAME`).
pub const TOOLS_MCP_SERVER_NAME: &str = "_agent_framework_tools";

/// The MCP protocol version answered when the CLI does not ask for one.
const DEFAULT_PROTOCOL_VERSION: &str = "2024-11-05";

/// The CLI-side name of one of the agent's tools
/// (`mcp__<server>__<tool>`), as allowed with `--allowedTools`.
pub fn qualified_tool_name(tool: &str) -> String {
    format!("mcp__{TOOLS_MCP_SERVER_NAME}__{tool}")
}

/// Serves the agent's executable function tools.
pub(crate) struct ToolServer {
    tools: Vec<ToolDefinition>,
    approval: Option<Arc<dyn FunctionApprovalHandler>>,
}

impl ToolServer {
    pub fn new(
        tools: Vec<ToolDefinition>,
        approval: Option<Arc<dyn FunctionApprovalHandler>>,
    ) -> Self {
        Self { tools, approval }
    }

    pub fn name(&self) -> &str {
        TOOLS_MCP_SERVER_NAME
    }

    /// The tools' MCP descriptors. The full parameter schema is passed
    /// through, with `type: object` and `properties` always set.
    pub fn list(&self) -> Vec<Value> {
        self.tools
            .iter()
            .map(|t| {
                let mut schema = match &t.parameters {
                    Value::Object(m) => m.clone(),
                    _ => Map::new(),
                };
                let properties = schema
                    .get("properties")
                    .filter(|p| p.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                schema.insert("type".into(), json!("object"));
                schema.insert("properties".into(), properties);
                json!({"name": t.name, "description": t.description, "inputSchema": schema})
            })
            .collect()
    }

    /// Answer one JSON-RPC message; `None` for notifications.
    pub async fn handle(&self, message: &Value) -> Option<Value> {
        let id = message.get("id").cloned();
        let method = message
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = id?;
        let result = match method {
            "initialize" => json!({
                "protocolVersion": message
                    .pointer("/params/protocolVersion")
                    .and_then(Value::as_str)
                    .unwrap_or(DEFAULT_PROTOCOL_VERSION),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": TOOLS_MCP_SERVER_NAME, "version": "1.0.0"},
            }),
            "ping" => json!({}),
            "tools/list" => json!({"tools": self.list()}),
            "tools/call" => {
                let name = message
                    .pointer("/params/name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let args = message
                    .pointer("/params/arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match self.call(name, args).await {
                    Some(r) => r,
                    None => {
                        return Some(json!({
                            "jsonrpc": "2.0", "id": id,
                            "error": {"code": -32602, "message": format!("Tool '{name}' not found")},
                        }))
                    }
                }
            }
            other => {
                return Some(json!({
                    "jsonrpc": "2.0", "id": id,
                    "error": {"code": -32601, "message": format!("Method '{other}' not found")},
                }))
            }
        };
        Some(json!({"jsonrpc": "2.0", "id": id, "result": result}))
    }

    /// Run one tool; `None` when it is unknown. Failures and denials are
    /// returned to the model as text, as upstream.
    pub async fn call(&self, name: &str, args: Value) -> Option<Value> {
        let tool = self.tools.iter().find(|t| t.name == name)?;
        let text = |t: String| json!({"content": [{"type": "text", "text": t}]});
        if tool.approval_mode == ApprovalMode::AlwaysRequire {
            let approved = match &self.approval {
                Some(handler) => {
                    let arguments = args.as_object().map(|m| {
                        FunctionArguments::Object(m.clone().into_iter().collect::<HashMap<_, _>>())
                    });
                    let call = FunctionCallContent::new(
                        format!("af-claude-approval::{name}"),
                        name,
                        arguments,
                    );
                    handler.approve(call).await
                }
                None => false,
            };
            if !approved {
                tracing::warn!(
                    target: "agent_framework::claude",
                    tool = name,
                    "denying a tool that requires approval ({})",
                    if self.approval.is_some() { "callback denied" } else { "no callback configured" }
                );
                return Some(text(if self.approval.is_some() {
                    format!(
                        "Tool '{name}' requires human approval (approval_mode='always_require') \
                         and the request was denied."
                    )
                } else {
                    format!(
                        "Tool '{name}' requires human approval (approval_mode='always_require') \
                         but no on_function_approval callback is configured on the agent; the \
                         request was denied."
                    )
                }));
            }
        }
        let executor = tool.executor.as_ref()?;
        Some(match executor.invoke(args).await {
            Ok(Value::String(s)) => text(s),
            Ok(Value::Null) => text(String::new()),
            Ok(other) => text(other.to_string()),
            Err(e) => text(format!("Error: {e}")),
        })
    }
}
