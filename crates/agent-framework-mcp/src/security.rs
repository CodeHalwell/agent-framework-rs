//! FIDES for MCP tools: restriction-only labels from server annotations, and
//! IFC labels a server attaches to its results (`_meta.ifc`), wired into
//! [`agent_framework_core::security`] (feature `fides`).
//!
//! Port of upstream's `SecureMCPToolProxy`, `apply_mcp_security_labels` and
//! the `_meta.ifc` result stamping in `agent_framework.security`. They live
//! here rather than in core because core cannot depend on this crate.
//!
//! Every tool an MCP server lists is declared in a [`ToolSecurityRegistry`]
//! (normally the one of your [`SecureAgentConfig`](agent_framework_core::security::SecureAgentConfig)):
//!
//! - **Annotations are hints, not authority.** Only `openWorldHint: true`
//!   changes anything (results become untrusted); every remote tool keeps a
//!   `PUBLIC` confidentiality cap (unless `mark_write_tools_as_sinks` is
//!   off) and rejects an untrusted context. A local
//!   [`McpSecurityPolicy::annotation_override`] (by *remote* tool name)
//!   replaces the derived labels.
//! - **Result labels.** A server that labels its results
//!   (`_meta.ifc = {"integrity": "trusted"|"untrusted", "confidentiality":
//!   "public"|"private"}`) can only make the local label stricter: the two
//!   are combined. With [`McpSecurityPolicy::trust_server_ifc`] a complete,
//!   valid server label is authoritative for that result instead. Missing,
//!   partial or malformed metadata leaves the local label alone.
//!
//! Tools are re-declared each time they are listed, so tools a server adds
//! later (`notifications/tools/list_changed`) are labeled before they can be
//! called.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_core::security::SecureAgentConfig;
//! use agent_framework_mcp::security::SecureMcpToolProxy;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let security = Arc::new(SecureAgentConfig::builder().build()?);
//! let github = SecureMcpToolProxy::from_url("https://mcp.example.com/mcp")
//!     .name("github")
//!     .headers([("Authorization", "Bearer <token>")])
//!     .registry(security.registry().clone());
//! let agent = security
//!     .apply(Agent::builder(client))
//!     .tools(github.tool_definitions().await?)
//!     .build();
//! # let _ = agent;
//! # Ok(())
//! # }
//! ```
//!
//! # Divergences
//!
//! - Labels live in the registry, keyed by the tool's local (normalized)
//!   name, rather than in each function's `additional_properties`.
//! - A Rust tool result is one JSON value, so a combined server label is
//!   embedded as its `security_label` member (an object result) or by
//!   wrapping it as `{"result": ..., "security_label": ...}`; a trusted
//!   server label is handed to the tracking middleware directly
//!   ([`set_authoritative_result_label`]).
//! - `apply_mcp_security_labels` on an already-built tool cannot rewire its
//!   executors here; use [`SecureMcpToolProxy`] (or
//!   [`secure_tool_definitions`]), which builds the labeled executors.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::security::{
    combine_labels, map_mcp_annotations_to_labels, set_authoritative_result_label,
    ConfidentialityLabel, ContentLabel, IntegrityLabel, ToolSecurity, ToolSecurityRegistry,
};
use agent_framework_core::tools::{Tool, ToolDefinition, ToolSource};
use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::client::McpClient;
use crate::tool::{build_tool_definitions_with, ExecutorParts};
use crate::{McpStdioTool, McpStreamableHttpTool, McpWebsocketTool};

/// The local policy applied to one MCP connection's tools.
#[derive(Debug, Clone)]
pub struct McpSecurityPolicy {
    default_integrity: IntegrityLabel,
    overrides: HashMap<String, (IntegrityLabel, Option<ConfidentialityLabel>)>,
    mark_write_tools_as_sinks: bool,
    trust_server_ifc: bool,
}

impl Default for McpSecurityPolicy {
    /// Untrusted by default, write tools capped at `PUBLIC`, server IFC
    /// labels restriction-only.
    fn default() -> Self {
        Self {
            default_integrity: IntegrityLabel::Untrusted,
            overrides: HashMap::new(),
            mark_write_tools_as_sinks: true,
            trust_server_ifc: false,
        }
    }
}

impl McpSecurityPolicy {
    /// The default policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// The integrity of a tool's results when its annotations do not
    /// require a stricter one (default `Untrusted`).
    pub fn default_integrity(mut self, integrity: IntegrityLabel) -> Self {
        self.default_integrity = integrity;
        self
    }

    /// Replace the annotation-derived labels of the tool the server calls
    /// `remote_name`. Applies to this connection only: the mapping is not
    /// bound to a server identity, so reuse it elsewhere only after
    /// authorizing it for that server's tools.
    pub fn annotation_override(
        mut self,
        remote_name: impl Into<String>,
        integrity: IntegrityLabel,
        max_confidentiality: Option<ConfidentialityLabel>,
    ) -> Self {
        self.overrides
            .insert(remote_name.into(), (integrity, max_confidentiality));
        self
    }

    /// Whether to apply the `PUBLIC` confidentiality cap as each tool's
    /// `max_allowed_confidentiality` (default `true`).
    pub fn mark_write_tools_as_sinks(mut self, mark: bool) -> Self {
        self.mark_write_tools_as_sinks = mark;
        self
    }

    /// Whether a complete, valid server `_meta.ifc` label is authoritative
    /// for its result (default `false`: it can only restrict). Annotations
    /// stay non-authoritative either way.
    pub fn trust_server_ifc(mut self, trust: bool) -> Self {
        self.trust_server_ifc = trust;
        self
    }

    /// The declarations for one tool, merged onto any existing ones (only
    /// `source_integrity`, `max_allowed_confidentiality` and
    /// `accepts_untrusted` are set, as upstream).
    fn declaration(
        &self,
        remote_name: &str,
        annotations: Option<&Value>,
        existing: Option<ToolSecurity>,
    ) -> ToolSecurity {
        let (integrity, max_confidentiality, accepts_untrusted) =
            match self.overrides.get(remote_name) {
                Some((integrity, max)) => (*integrity, *max, false),
                None => map_mcp_annotations_to_labels(annotations, self.default_integrity),
            };
        let mut security = existing.unwrap_or_default();
        security.source_integrity = Some(integrity);
        security.accepts_untrusted = accepts_untrusted;
        security.max_allowed_confidentiality = if self.mark_write_tools_as_sinks {
            max_confidentiality
        } else {
            None
        };
        security
    }
}

/// Parse an MCP result's `_meta` into a label: `_meta.ifc` with both an
/// `integrity` (`trusted` / `untrusted`) and a `confidentiality` (`public` /
/// `private`). Anything missing, partial or unknown is `None`.
pub fn label_from_mcp_meta(meta: Option<&Value>) -> Option<ContentLabel> {
    let ifc = meta?.get("ifc")?.as_object()?;
    let integrity = IntegrityLabel::parse(ifc.get("integrity")?.as_str()?)?;
    let confidentiality = ConfidentialityLabel::parse(ifc.get("confidentiality")?.as_str()?)?;
    if !matches!(
        confidentiality,
        ConfidentialityLabel::Public | ConfidentialityLabel::Private
    ) {
        return None;
    }
    Some(ContentLabel::new(integrity, confidentiality))
}

/// Apply a server label to one result: without one, the result is left as
/// is (the tracking middleware labels it from the registry); a trusted one
/// is handed to the tracking middleware as authoritative; otherwise it is
/// combined with `local` (so it can only restrict) and embedded as the
/// result's `security_label`.
pub fn stamp_mcp_result(
    value: Value,
    local: &ContentLabel,
    server: Option<ContentLabel>,
    trust_server_ifc: bool,
) -> Value {
    let Some(server) = server else {
        return value;
    };
    if trust_server_ifc {
        set_authoritative_result_label(server);
        return value;
    }
    let mut label = combine_labels([local, &server]);
    match value {
        Value::Object(mut map) => {
            if let Some(Ok(existing)) = map.get("security_label").map(ContentLabel::from_value) {
                label = combine_labels([&label, &existing]);
            }
            map.insert("security_label".into(), label.to_value());
            Value::Object(map)
        }
        other => {
            let mut map = Map::new();
            map.insert("result".into(), other);
            map.insert("security_label".into(), label.to_value());
            Value::Object(map)
        }
    }
}

/// The text of an MCP error result. An error's text reaches the model
/// without passing through result labeling, so when the server labels the
/// error untrusted or confidential (trusted as is under `trust_server_ifc`,
/// otherwise combined with `local` so it can only restrict) the details are
/// withheld rather than returned unlabeled.
pub fn mcp_error_text(
    tool: &str,
    message: String,
    local: &ContentLabel,
    server: Option<&ContentLabel>,
    trust_server_ifc: bool,
) -> String {
    let Some(server) = server else {
        return message;
    };
    let effective = if trust_server_ifc {
        server.clone()
    } else {
        combine_labels([local, server])
    };
    if effective.is_trusted() && effective.is_public() {
        return message;
    }
    format!(
        "tool '{tool}' failed; details withheld ({})",
        if effective.is_trusted() {
            "confidential source"
        } else {
            "untrusted source"
        }
    )
}

/// Calls a remote tool and stamps its result's IFC label.
struct SecureMcpExecutor {
    local_name: String,
    description: String,
    parameters: Value,
    remote_name: String,
    client: Arc<McpClient>,
    registry: ToolSecurityRegistry,
    default_integrity: IntegrityLabel,
    trust_server_ifc: bool,
}

impl SecureMcpExecutor {
    /// Call the remote tool: the labeled result, and the host payload.
    async fn call(&self, arguments: Value) -> Result<(Value, Map<String, Value>)> {
        let raw = self
            .client
            .call_tool_raw(&self.remote_name, arguments)
            .await?;
        let result = crate::protocol::CallToolResult::from_value(&raw);
        let server = label_from_mcp_meta(result.meta.as_ref());
        if result.is_error {
            return Err(Error::tool(mcp_error_text(
                &self.local_name,
                result.error_message(),
                &self.local_label(),
                server.as_ref(),
                self.trust_server_ifc,
            )));
        }
        let mut properties = Map::new();
        if raw.to_string().len() <= crate::tool::MAX_HOST_PAYLOAD_SIZE_BYTES {
            properties.insert(
                agent_framework_core::types::MCP_TOOL_RESULT_HOST_PAYLOAD_KEY.into(),
                raw,
            );
        } else {
            tracing::warn!(
                "Omitting MCP Host payload because its encoded size exceeds {} bytes.",
                crate::tool::MAX_HOST_PAYLOAD_SIZE_BYTES
            );
        }
        let value = stamp_mcp_result(
            result.to_value(),
            &self.local_label(),
            server,
            self.trust_server_ifc,
        );
        Ok((value, properties))
    }

    /// The current local output policy (read at call time, as upstream).
    fn local_label(&self) -> ContentLabel {
        let declared = self.registry.get(&self.local_name).unwrap_or_default();
        ContentLabel::new(
            declared.source_integrity.unwrap_or(self.default_integrity),
            declared
                .confidentiality
                .unwrap_or(ConfidentialityLabel::Public),
        )
    }
}

#[async_trait]
impl Tool for SecureMcpExecutor {
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
        self.call(arguments).await.map(|(value, _)| value)
    }

    /// As [`McpToolExecutor`](crate::tool), the raw `tools/call` result is
    /// kept for the host (container-file citations and the like) under
    /// [`MCP_TOOL_RESULT_HOST_PAYLOAD_KEY`](agent_framework_core::types::MCP_TOOL_RESULT_HOST_PAYLOAD_KEY),
    /// beside the security-labeled, model-visible result; it never reaches
    /// the model.
    async fn invoke_with_properties(
        &self,
        arguments: Value,
        _ctx: &agent_framework_core::middleware::FunctionInvocationContext,
    ) -> Result<(Value, Map<String, Value>)> {
        self.call(arguments).await
    }
}

/// Declare `tools` in `registry` under `policy` and build their labeled
/// executors.
fn secure_definitions(
    client: Arc<McpClient>,
    tools: &[crate::ToolDescriptor],
    listing: (
        Option<&std::collections::HashSet<String>>,
        &crate::McpApprovalMode,
    ),
    registry: &ToolSecurityRegistry,
    policy: &McpSecurityPolicy,
) -> Vec<ToolDefinition> {
    build_tool_definitions_with(
        client,
        tools,
        listing.0,
        listing.1,
        &|parts: ExecutorParts<'_>| {
            let remote_name = parts.descriptor.name.clone();
            let declaration = policy.declaration(
                &remote_name,
                parts.descriptor.annotations.as_ref(),
                registry.get(&parts.local_name),
            );
            tracing::info!(
                tool = %remote_name,
                integrity = ?declaration.source_integrity,
                max_confidentiality = ?declaration.max_allowed_confidentiality,
                "MCP auto-label"
            );
            registry.declare(parts.local_name.clone(), declaration);
            Arc::new(SecureMcpExecutor {
                local_name: parts.local_name,
                description: parts.descriptor.description.clone().unwrap_or_default(),
                parameters: parts.descriptor.input_schema.clone(),
                remote_name,
                client: parts.client,
                registry: registry.clone(),
                default_integrity: policy.default_integrity,
                trust_server_ifc: policy.trust_server_ifc,
            })
        },
    )
}

/// One of this crate's MCP tools.
pub enum McpToolKind {
    /// A stdio server.
    Stdio(McpStdioTool),
    /// A streamable-HTTP server.
    StreamableHttp(McpStreamableHttpTool),
    /// A WebSocket server.
    Websocket(McpWebsocketTool),
}

impl From<McpStdioTool> for McpToolKind {
    fn from(tool: McpStdioTool) -> Self {
        Self::Stdio(tool)
    }
}

impl From<McpStreamableHttpTool> for McpToolKind {
    fn from(tool: McpStreamableHttpTool) -> Self {
        Self::StreamableHttp(tool)
    }
}

impl From<McpWebsocketTool> for McpToolKind {
    fn from(tool: McpWebsocketTool) -> Self {
        Self::Websocket(tool)
    }
}

impl McpToolKind {
    fn name(&self) -> &str {
        match self {
            Self::Stdio(t) => t.name(),
            Self::StreamableHttp(t) => t.name(),
            Self::Websocket(t) => t.name(),
        }
    }

    async fn client(&self) -> Result<Arc<McpClient>> {
        match self {
            Self::Stdio(t) => t.connected_client().await,
            Self::StreamableHttp(t) => t.connected_client().await,
            Self::Websocket(t) => t.connected_client().await,
        }
    }

    fn listing_policy(
        &self,
    ) -> (
        Option<&std::collections::HashSet<String>>,
        &crate::McpApprovalMode,
        bool,
    ) {
        match self {
            Self::Stdio(t) => t.listing_policy(),
            Self::StreamableHttp(t) => t.listing_policy(),
            Self::Websocket(t) => t.listing_policy(),
        }
    }

    async fn close(&self) -> Result<()> {
        match self {
            Self::Stdio(t) => t.close().await,
            Self::StreamableHttp(t) => t.close().await,
            Self::Websocket(t) => t.close().await,
        }
    }
}

/// An MCP tool whose tools are labeled for FIDES (upstream
/// `SecureMCPToolProxy`): every listed tool is declared in the registry
/// under the proxy's [`McpSecurityPolicy`], and each call's result carries
/// the server's IFC label when it sends one.
///
/// Use it where you would use the wrapped tool: as a [`ToolSource`] (tools
/// resolved per run, so later-discovered tools are labeled too) or through
/// [`Self::tool_definitions`]. Calls run locally, so the label-tracking and
/// policy middleware see every one, unlike a provider-hosted MCP tool.
pub struct SecureMcpToolProxy {
    tool: McpToolKind,
    policy: McpSecurityPolicy,
    registry: ToolSecurityRegistry,
}

/// Builds a [`SecureMcpToolProxy`] over a streamable-HTTP URL.
pub struct SecureMcpUrlBuilder {
    url: String,
    name: Option<String>,
    description: Option<String>,
    headers: Vec<(String, String)>,
}

impl SecureMcpUrlBuilder {
    /// The tool's name (default `"mcp"`).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The tool's description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Headers (e.g. auth) sent to the server.
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

    /// Finish with the default policy and a new registry (see
    /// [`SecureMcpToolProxy::registry`]).
    pub fn registry(self, registry: ToolSecurityRegistry) -> SecureMcpToolProxy {
        self.build().registry(registry)
    }

    /// Finish with the default policy and a new registry.
    pub fn build(self) -> SecureMcpToolProxy {
        let mut tool =
            McpStreamableHttpTool::new(self.name.unwrap_or_else(|| "mcp".into()), self.url)
                .headers(self.headers);
        if let Some(description) = self.description {
            tool = tool.description(description);
        }
        SecureMcpToolProxy::new(tool)
    }
}

impl SecureMcpToolProxy {
    /// Wrap an MCP tool, with the default policy and a new registry.
    pub fn new(tool: impl Into<McpToolKind>) -> Self {
        Self {
            tool: tool.into(),
            policy: McpSecurityPolicy::default(),
            registry: ToolSecurityRegistry::new(),
        }
    }

    /// A proxy over a streamable-HTTP MCP server at `url`.
    pub fn from_url(url: impl Into<String>) -> SecureMcpUrlBuilder {
        SecureMcpUrlBuilder {
            url: url.into(),
            name: None,
            description: None,
            headers: Vec::new(),
        }
    }

    /// The registry the tools are declared in. Pass your
    /// [`SecureAgentConfig`](agent_framework_core::security::SecureAgentConfig)'s
    /// (`config.registry().clone()`) so its middleware reads them.
    pub fn registry(mut self, registry: ToolSecurityRegistry) -> Self {
        self.registry = registry;
        self
    }

    /// The local policy.
    pub fn policy(mut self, policy: McpSecurityPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The registry in use.
    pub fn security_registry(&self) -> &ToolSecurityRegistry {
        &self.registry
    }

    /// The wrapped tool.
    pub fn mcp_tool(&self) -> &McpToolKind {
        &self.tool
    }

    /// The wrapped tool's name.
    pub fn name(&self) -> &str {
        self.tool.name()
    }

    /// Connect the wrapped tool (idempotent).
    pub async fn connect(&self) -> Result<()> {
        self.tool.client().await.map(|_| ())
    }

    /// Close the wrapped tool's connection.
    pub async fn close(&self) -> Result<()> {
        self.tool.close().await
    }

    /// Connect, list the server's tools (live), declare them and return
    /// their labeled definitions.
    pub async fn tool_definitions(&self) -> Result<Vec<ToolDefinition>> {
        let client = self.tool.client().await?;
        let tools = client.list_tools().await?;
        let (allowed, approval, _) = self.tool.listing_policy();
        Ok(secure_definitions(
            client,
            &tools,
            (allowed, approval),
            &self.registry,
            &self.policy,
        ))
    }
}

#[async_trait]
impl ToolSource for SecureMcpToolProxy {
    /// The wrapped tool's cached listing (refreshed on
    /// `notifications/tools/list_changed`), declared and labeled each time.
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>> {
        let (allowed, approval, load_tools) = self.tool.listing_policy();
        if !load_tools {
            return Ok(Vec::new());
        }
        let client = self.tool.client().await?;
        let tools = client.list_tools_cached().await?;
        Ok(secure_definitions(
            client,
            &tools,
            (allowed, approval),
            &self.registry,
            &self.policy,
        ))
    }

    fn source_name(&self) -> &str {
        self.tool.name()
    }
}

/// One-shot form of [`SecureMcpToolProxy::tool_definitions`] (upstream
/// `apply_mcp_security_labels`): list `tool`'s tools, declare them in
/// `registry` under `policy`, and return their labeled definitions.
pub async fn secure_tool_definitions(
    tool: impl Into<McpToolKind>,
    registry: &ToolSecurityRegistry,
    policy: McpSecurityPolicy,
) -> Result<Vec<ToolDefinition>> {
    SecureMcpToolProxy::new(tool)
        .registry(registry.clone())
        .policy(policy)
        .tool_definitions()
        .await
}
