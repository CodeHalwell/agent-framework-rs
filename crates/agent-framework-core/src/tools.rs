//! Tools: executable functions and hosted-tool markers.
//!
//! Rust equivalent of `agent_framework._tools`. An [`FunctionTool`] is a locally
//! executable tool; hosted [`ToolKind`] variants are markers handed to the service.
//! Both are represented uniformly to a chat client as a [`ToolDefinition`].
//!
//! Prefer [`FunctionTool::typed`] over [`FunctionTool::new`] when the arguments can
//! be expressed as a `#[derive(Deserialize, JsonSchema)]` struct: it derives the
//! parameters JSON Schema via `schemars` instead of requiring a hand-written
//! [`serde_json::Value`].

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// A boxed, owned future returned by a tool invocation.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// An executable tool the framework can invoke locally.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The tool name exposed to the model.
    fn name(&self) -> &str;

    /// A human/model-readable description.
    fn description(&self) -> &str;

    /// The JSON Schema describing the tool's parameters.
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with the given JSON arguments.
    async fn invoke(&self, arguments: Value) -> Result<Value>;

    /// Execute the tool with access to the surrounding
    /// [`FunctionInvocationContext`](crate::middleware::FunctionInvocationContext)
    /// (the agent session, middleware metadata, …).
    ///
    /// The function-invocation loop calls **this** method; the default
    /// implementation ignores the context and delegates to [`Tool::invoke`],
    /// so ordinary tools need not care. Override it for tools that read the
    /// invocation context — the Rust analogue of an upstream `FunctionTool`
    /// whose function declares a `ctx: FunctionInvocationContext` parameter
    /// (e.g. the `Agent::as_tool` wrapper, which forwards `ctx.session` to
    /// the sub-agent when `propagate_session` is enabled).
    async fn invoke_in_context(
        &self,
        arguments: Value,
        _ctx: &crate::middleware::FunctionInvocationContext,
    ) -> Result<Value> {
        self.invoke(arguments).await
    }
}

/// A dynamic source of tools, resolved fresh on every agent run instead of
/// being frozen into the agent's tool list at build time.
///
/// The motivating case is an MCP server: without this trait, wiring one into
/// a [`crate::agent::Agent`] means calling `mcp.tool_definitions().await`
/// once, up front, and handing the (now-frozen) result to
/// [`crate::agent::AgentBuilder::tools`] — the agent never notices a
/// server-side tool-catalog change (`notifications/tools/list_changed`)
/// afterward. Registering a `ToolSource` via
/// [`crate::agent::AgentBuilder::tool_source`] instead defers resolution
/// to every [`crate::agent::SupportsAgentRun::run`] / [`crate::agent::SupportsAgentRun::run_with_options`]
/// / [`crate::agent::SupportsAgentRun::run_stream`] call (see `Agent::prepare_request`),
/// so a source that caches internally — invalidating that cache when the
/// server signals a change — can serve an up-to-date catalog on every run
/// without a live round trip each time.
///
/// See `agent-framework-mcp`'s `McpStdioTool` / `McpStreamableHttpTool` /
/// `McpWebsocketTool`, which all implement this trait.
#[async_trait]
pub trait ToolSource: Send + Sync {
    /// Resolve this source's current tools.
    ///
    /// Called once per agent run. Implementations that connect to a remote
    /// server should connect lazily (on first call) and are encouraged to
    /// cache their result until something invalidates it, rather than
    /// performing a live round trip on every call.
    ///
    /// An `Err` returned here propagates out of the whole run rather than
    /// being swallowed — this mirrors the upstream Python reference, whose
    /// `Agent.run`/`run_stream` do not catch a failure raised while
    /// connecting to an `MCPTool` at run time (`_agents.py:855-865,
    /// 970-980`: `await self._async_exit_stack.enter_async_context(tool)`
    /// is not wrapped in a `try`/`except`).
    async fn resolve_tools(&self) -> Result<Vec<ToolDefinition>>;

    /// A short, human-readable name for this source, used in diagnostics
    /// (e.g. a `tracing::warn!` when one of its tools collides by name with
    /// a tool that already exists).
    fn source_name(&self) -> &str;
}

/// The category of a tool as advertised to the service.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolKind {
    /// A callable function (executed locally, unless declaration-only).
    Function,
    /// Service-side code interpreter.
    HostedCodeInterpreter,
    /// Service-side image generation.
    HostedImageGeneration,
    /// Service-side web search.
    HostedWebSearch,
    /// Service-side file search over hosted vector stores.
    HostedFileSearch { max_results: Option<u32> },
    /// Service-side MCP tool.
    HostedMcp {
        url: String,
        allowed_tools: Option<Vec<String>>,
    },
}

/// Whether a call to a tool must be approved by a human before it runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalMode {
    /// Never require approval (default).
    #[default]
    NeverRequire,
    /// Always require approval before executing.
    AlwaysRequire,
}

/// The approval gate configured on a *hosted* MCP connector -- i.e. how the
/// service itself decides whether a call to one of its MCP tools needs human
/// sign-off before it runs. Set via [`ToolDefinition::mcp_approval_mode`].
///
/// Distinct from [`ApprovalMode`], which gates *local* function-tool
/// execution in this framework's own invocation loop; a hosted MCP tool call
/// happens entirely on the service side, so `ApprovalMode` does not apply to
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum McpApprovalMode {
    /// Every tool call on the MCP server requires approval.
    Always,
    /// No tool call on the MCP server requires approval.
    Never,
    /// Approval is required or waived per tool name.
    PerTool {
        /// Tool names that always require approval.
        always: Vec<String>,
        /// Tool names that never require approval.
        never: Vec<String>,
    },
}

impl McpApprovalMode {
    /// The `parameters["approval_mode"]` wire value: the strings
    /// `"always_require"`/`"never_require"` for [`McpApprovalMode::Always`]/
    /// [`McpApprovalMode::Never`], or (for [`McpApprovalMode::PerTool`]) an
    /// object with `"always"`/`"never"` tool-name-array keys, each included
    /// only when non-empty.
    fn into_value(self) -> Value {
        match self {
            McpApprovalMode::Always => Value::String("always_require".to_string()),
            McpApprovalMode::Never => Value::String("never_require".to_string()),
            McpApprovalMode::PerTool { always, never } => {
                let mut map = Map::new();
                if !always.is_empty() {
                    map.insert("always".to_string(), serde_json::json!(always));
                }
                if !never.is_empty() {
                    map.insert("never".to_string(), serde_json::json!(never));
                }
                Value::Object(map)
            }
        }
    }
}

/// A uniform, cloneable descriptor of a tool passed via [`ChatOptions::tools`].
///
/// For function tools it carries an executor (`Arc<dyn Tool>`); hosted tools and
/// declaration-only tools carry `None`.
///
/// [`ChatOptions::tools`]: crate::types::ChatOptions::tools
#[derive(Clone)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    /// JSON Schema for the parameters (empty object for tools with no params).
    pub parameters: Value,
    pub kind: ToolKind,
    pub approval_mode: ApprovalMode,
    /// The local executor, if this is an invokable function tool.
    pub executor: Option<Arc<dyn Tool>>,
}

impl std::fmt::Debug for ToolDefinition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolDefinition")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("kind", &self.kind)
            .field("approval_mode", &self.approval_mode)
            .field("executable", &self.executor.is_some())
            .finish()
    }
}

impl ToolDefinition {
    /// Whether this tool has a local implementation to execute.
    pub fn is_executable(&self) -> bool {
        self.executor.is_some() && self.kind == ToolKind::Function
    }

    /// Whether a human must approve a call to this tool before it executes.
    pub fn requires_approval(&self) -> bool {
        self.approval_mode == ApprovalMode::AlwaysRequire
    }

    /// Builder: set the human-in-the-loop approval mode (default
    /// [`ApprovalMode::NeverRequire`]). When set to
    /// [`ApprovalMode::AlwaysRequire`], the function-invocation loop returns a
    /// [`FunctionApprovalRequestContent`] instead of executing the call.
    ///
    /// [`FunctionApprovalRequestContent`]: crate::types::FunctionApprovalRequestContent
    pub fn with_approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Builder: require human approval before every call to this tool.
    pub fn require_approval(self) -> Self {
        self.with_approval_mode(ApprovalMode::AlwaysRequire)
    }

    /// Builder: set the tool's description.
    ///
    /// Works on any [`ToolDefinition`], but is primarily useful right after
    /// a hosted constructor ([`hosted_web_search`], [`hosted_file_search`],
    /// [`hosted_code_interpreter`], [`hosted_mcp`]), none of which take a
    /// description argument. For a [`hosted_mcp`] tool specifically, the
    /// OpenAI Responses API forwards a non-empty description as the hosted
    /// MCP server's `server_description`.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Builder: the web-search tool's approximate user location.
    ///
    /// Read by the OpenAI Chat Completions and Responses APIs (as
    /// `web_search_options.user_location.approximate` /
    /// `web_search.user_location`) and by Anthropic's web-search tool
    /// (`user_location`). Ignored by Azure AI Foundry's Bing-backed web
    /// search. Writes `parameters["user_location"]`; the value's shape is
    /// provider-specific (e.g. `{"city": "Seattle", "country": "US"}`). Use
    /// immediately after [`hosted_web_search`].
    pub fn user_location(self, location: Value) -> Self {
        self.set_param("user_location", location)
    }

    /// Builder: cap the number of searches a hosted web-search tool may
    /// perform while answering a single request.
    ///
    /// Read by Anthropic only (`max_uses`); OpenAI and Azure AI Foundry
    /// ignore it. Writes `parameters["max_uses"]`. Use immediately after
    /// [`hosted_web_search`].
    pub fn max_uses(self, max_uses: u32) -> Self {
        self.set_param("max_uses", serde_json::json!(max_uses))
    }

    /// Builder: an Azure AI Foundry Bing Grounding connection id.
    ///
    /// Read by Azure AI Foundry only, to build a `bing_grounding` tool.
    /// Mutually exclusive with [`ToolDefinition::custom_connection`]: a
    /// fully-specified custom pair takes precedence over this plain id, and
    /// a *partial* custom pair (only one of the two custom fields) still
    /// disqualifies this plain id -- Azure AI Foundry then rejects the tool
    /// outright for having no usable connection. Writes
    /// `parameters["connection_id"]`. Use immediately after
    /// [`hosted_web_search`].
    pub fn connection_id(self, connection_id: impl Into<String>) -> Self {
        self.set_param("connection_id", Value::String(connection_id.into()))
    }

    /// Builder: an Azure AI Foundry Bing Custom Search connection: a
    /// connection id plus the custom-search instance name.
    ///
    /// Read by Azure AI Foundry only, to build a `bing_custom_search` tool;
    /// takes precedence over a plain [`ToolDefinition::connection_id`] when
    /// both are set. Writes `parameters["custom_connection_id"]` and
    /// `parameters["instance_name"]`. Use immediately after
    /// [`hosted_web_search`].
    pub fn custom_connection(
        self,
        connection_id: impl Into<String>,
        instance_name: impl Into<String>,
    ) -> Self {
        self.set_param("custom_connection_id", Value::String(connection_id.into()))
            .set_param("instance_name", Value::String(instance_name.into()))
    }

    /// Builder: the vector store ids a hosted file-search tool should
    /// search.
    ///
    /// Read by the OpenAI Responses API and Azure AI Foundry. Ignored by
    /// Anthropic, which has no file-search tool (unsupported by the
    /// Anthropic Messages API). Writes `parameters["vector_store_ids"]`.
    /// Use immediately after [`hosted_file_search`].
    pub fn vector_store_ids(self, ids: Vec<String>) -> Self {
        self.set_param("vector_store_ids", serde_json::json!(ids))
    }

    /// Builder: cap the number of results a hosted file-search tool returns.
    ///
    /// Read by the OpenAI Responses API only, and only as a fallback: pass
    /// `max_results` directly to [`hosted_file_search`] where possible,
    /// which takes precedence over this parameter when both are set (and is
    /// the only option Azure AI Foundry honors, since it does not read this
    /// key). Writes `parameters["max_results"]`.
    pub fn max_results(self, max_results: u32) -> Self {
        self.set_param("max_results", serde_json::json!(max_results))
    }

    /// Builder: file ids attached to a hosted code-interpreter tool's
    /// container.
    ///
    /// Read by the OpenAI Responses API, which folds them into a default
    /// `{"type": "auto"}` container unless [`ToolDefinition::container`]
    /// supplies an explicit override (which then wins outright and this key
    /// is ignored). Writes `parameters["file_ids"]`. Use immediately after
    /// [`hosted_code_interpreter`].
    pub fn file_ids(self, file_ids: Vec<String>) -> Self {
        self.set_param("file_ids", serde_json::json!(file_ids))
    }

    /// Builder: an explicit container object for a hosted code-interpreter
    /// tool, overriding the default `{"type": "auto"}` container (and any
    /// [`ToolDefinition::file_ids`]).
    ///
    /// Read by the OpenAI Responses API only. Writes
    /// `parameters["container"]`. Use immediately after
    /// [`hosted_code_interpreter`].
    pub fn container(self, container: Value) -> Self {
        self.set_param("container", container)
    }

    /// Builder: HTTP headers sent with requests to a hosted MCP server.
    ///
    /// Read by the OpenAI Responses API (forwarded verbatim as `headers`),
    /// Anthropic (only the lower-case `"authorization"` entry, mapped to
    /// `authorization_token`), and Azure AI Foundry (forwarded verbatim,
    /// when non-empty). Writes `parameters["headers"]`. Use immediately
    /// after [`hosted_mcp`].
    pub fn headers(self, headers: HashMap<String, String>) -> Self {
        self.set_param("headers", serde_json::json!(headers))
    }

    /// Builder: the hosted MCP server's own approval gate for its tool
    /// calls -- see [`McpApprovalMode`].
    ///
    /// Read by the OpenAI Responses API and Azure AI Foundry as
    /// `parameters["approval_mode"]`; not read by the Anthropic converter in
    /// this port (Anthropic's MCP connector has no per-tool approval concept
    /// here). Use immediately after [`hosted_mcp`].
    pub fn mcp_approval_mode(self, mode: McpApprovalMode) -> Self {
        self.set_param("approval_mode", mode.into_value())
    }

    /// Insert `value` at `key` in [`ToolDefinition::parameters`], coercing
    /// `parameters` to an empty object first if it is not already one (the
    /// hosted constructors always start it as one via [`empty_schema`], so
    /// this is purely defensive).
    fn set_param(mut self, key: &str, value: Value) -> Self {
        if !self.parameters.is_object() {
            self.parameters = Value::Object(Map::new());
        }
        if let Value::Object(map) = &mut self.parameters {
            map.insert(key.to_string(), value);
        }
        self
    }

    /// The OpenAI-style function spec: `{"type":"function","function":{...}}`.
    pub fn to_openai_spec(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        })
    }

    /// Build a tool definition from any [`Tool`] implementation.
    pub fn from_tool(tool: Arc<dyn Tool>) -> Self {
        Self {
            name: tool.name().to_string(),
            description: tool.description().to_string(),
            parameters: tool.parameters_schema(),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: Some(tool),
        }
    }
}

impl<T: Tool + 'static> From<Arc<T>> for ToolDefinition {
    fn from(tool: Arc<T>) -> Self {
        ToolDefinition::from_tool(tool)
    }
}

type ToolClosure = Arc<dyn Fn(Value) -> BoxFuture<Result<Value>> + Send + Sync>;

/// A concrete, locally executable tool built from a closure.
///
/// This is the Rust analogue of upstream's `FunctionTool` / the `@tool`
/// decorator (formerly `AIFunction` / `@ai_function`).
#[derive(Clone)]
pub struct FunctionTool {
    name: String,
    description: String,
    parameters: Value,
    approval_mode: ApprovalMode,
    func: ToolClosure,
    max_invocations: Option<usize>,
    max_invocation_exceptions: Option<usize>,
    // `Arc` so every `Clone` of an `FunctionTool` shares one pair of counters
    // with its source rather than silently resetting the limits; mirrors
    // Python's `invocation_count`/`invocation_exception_count` being mutable
    // state on the (singular) `AIFunction` instance itself.
    invocation_count: Arc<AtomicUsize>,
    invocation_exception_count: Arc<AtomicUsize>,
}

impl FunctionTool {
    /// Create a function tool from a hand-written JSON Schema.
    ///
    /// * `parameters` is the JSON Schema for the arguments object.
    /// * `func` receives the parsed JSON arguments and returns a JSON result.
    ///
    /// Prefer [`FunctionTool::typed`] when the arguments can be expressed as a
    /// `#[derive(Deserialize, JsonSchema)]` struct.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
        func: F,
    ) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            approval_mode: ApprovalMode::NeverRequire,
            func: Arc::new(move |args| Box::pin(func(args))),
            max_invocations: None,
            max_invocation_exceptions: None,
            invocation_count: Arc::new(AtomicUsize::new(0)),
            invocation_exception_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Create a function tool whose parameters schema and argument
    /// deserialization are derived from a Rust type, instead of a
    /// hand-written [`serde_json::Value`] schema.
    ///
    /// `Args` must implement [`schemars::JsonSchema`] (to derive the
    /// parameters schema) and [`serde::de::DeserializeOwned`] (to parse the
    /// model-supplied arguments); `Ret` need only implement
    /// [`serde::Serialize`] -- return `serde_json::Value` directly (as in
    /// the example below), or any other serializable type.
    ///
    /// # Parameters schema
    ///
    /// The schema is generated once, at construction, via `schemars`'
    /// `SchemaGenerator` (the machinery behind its `schema_for!` macro, which
    /// cannot itself target a type parameter), then lightly post-processed
    /// for OpenAI-style function parameters: the top-level `$schema` and
    /// `title` keys are stripped. For a "simple" struct (only
    /// primitive/string/number/bool/`Vec`/`Option` fields) this leaves
    /// exactly `{"type": "object", "properties": {...}, "required": [...]}`
    /// -- a field is listed in `required` unless it is an `Option<_>` or
    /// carries `#[serde(default)]`. Nested structs and enums keep
    /// `schemars`' own representation: a top-level `definitions` map with
    /// `$ref`s into it (schemars 0.8's convention for referenceable types).
    /// This is *not* inlined -- every provider converter in this workspace
    /// forwards [`ToolDefinition::parameters`] to the wire unmodified, so a
    /// `$ref`/`definitions` pair round-trips exactly like any other
    /// JSON-Schema keyword this crate doesn't otherwise interpret.
    ///
    /// # Argument errors
    ///
    /// If the model-supplied JSON arguments don't deserialize into `Args`
    /// (e.g. a required field is missing or mistyped), [`Tool::invoke`]
    /// returns `Err(`[`Error::Tool`]`)` rather than panicking or silently
    /// substituting a default -- the same `Result`-propagation shape used
    /// for every other tool-execution failure (a closure error from
    /// [`FunctionTool::new`], an [`FunctionTool::max_invocations`] limit, ...),
    /// which the function-invocation loop turns into an error
    /// [`crate::types::FunctionResultContent`] exactly as it would for any
    /// of those.
    ///
    /// # Example
    ///
    /// ```
    /// use agent_framework_core::tools::FunctionTool;
    ///
    /// #[derive(serde::Deserialize, schemars::JsonSchema)]
    /// struct WeatherArgs {
    ///     city: String,
    ///     #[serde(default)]
    ///     units: Option<String>,
    /// }
    ///
    /// let _tool = FunctionTool::typed(
    ///     "get_weather",
    ///     "Get the weather.",
    ///     |args: WeatherArgs| async move {
    ///         Ok(serde_json::json!({ "city": args.city, "temp": 21 }))
    ///     },
    /// );
    /// ```
    pub fn typed<Args, Ret, F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        f: F,
    ) -> Self
    where
        Args: DeserializeOwned + schemars::JsonSchema + Send + 'static,
        Ret: Serialize,
        F: Fn(Args) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Ret>> + Send + 'static,
    {
        let name = name.into();
        let parameters = typed_parameters_schema::<Args>();
        let err_name = name.clone();
        let f = Arc::new(f);
        let func: ToolClosure = Arc::new(move |value: Value| {
            let f = Arc::clone(&f);
            let err_name = err_name.clone();
            Box::pin(async move {
                let args: Args = serde_json::from_value(value).map_err(|e| {
                    Error::tool(format!("invalid arguments for tool '{err_name}': {e}"))
                })?;
                let ret = f(args).await?;
                serde_json::to_value(ret).map_err(|e| {
                    Error::tool(format!(
                        "failed to serialize result of tool '{err_name}': {e}"
                    ))
                })
            })
        });
        Self {
            name,
            description: description.into(),
            parameters,
            approval_mode: ApprovalMode::NeverRequire,
            func,
            max_invocations: None,
            max_invocation_exceptions: None,
            invocation_count: Arc::new(AtomicUsize::new(0)),
            invocation_exception_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Builder: set the human-in-the-loop approval mode (default
    /// [`ApprovalMode::NeverRequire`]). Carried through to the
    /// [`ToolDefinition`] produced by [`FunctionTool::into_definition`].
    pub fn with_approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Builder: cap the number of times this function may be invoked.
    ///
    /// Once [`FunctionTool::invocation_count`] reaches `max`, further calls to
    /// [`Tool::invoke`] return `Err(`[`Error::Tool`]`)` instead of running
    /// the function again -- mirrors Python's
    /// `AIFunction(max_invocations=...)` (`_tools.py:599-600, 687-690`).
    /// `None` (the default) means no limit.
    ///
    /// Unlike Python, which raises `ValueError` at construction for a value
    /// less than 1, a value of `0` is accepted here: it simply means the
    /// limit is already reached, so every invocation errors immediately
    /// (the same terminal state Python's validation exists to prevent
    /// constructing in the first place).
    ///
    /// The counter is shared by every `Clone` of this `FunctionTool` (see the
    /// note on [`FunctionTool`]'s fields), not reset per clone.
    pub fn max_invocations(mut self, max: usize) -> Self {
        self.max_invocations = Some(max);
        self
    }

    /// Builder: cap the number of invocation failures this function
    /// tolerates.
    ///
    /// Every [`Tool::invoke`] call that returns `Err` -- whether from
    /// argument deserialization (see [`FunctionTool::typed`]), the wrapped
    /// closure itself, or result serialization -- increments
    /// [`FunctionTool::invocation_exception_count`]. Once that count reaches
    /// `max`, further calls return `Err(`[`Error::Tool`]`)` immediately
    /// without re-attempting the function. `None` (the default) means no
    /// limit. Mirrors Python's `AIFunction(max_invocation_exceptions=...)`
    /// (`_tools.py:601-602, 691-698`); see [`FunctionTool::max_invocations`]
    /// for how the `0` case differs from Python's constructor-time
    /// validation.
    pub fn max_invocation_exceptions(mut self, max: usize) -> Self {
        self.max_invocation_exceptions = Some(max);
        self
    }

    /// The number of times [`Tool::invoke`] has run the wrapped function
    /// (i.e. got past any [`FunctionTool::max_invocations`]/
    /// [`FunctionTool::max_invocation_exceptions`] gate). Mirrors Python's
    /// public `invocation_count` attribute.
    pub fn invocation_count(&self) -> usize {
        self.invocation_count.load(Ordering::SeqCst)
    }

    /// The number of those invocations that returned `Err`. Mirrors
    /// Python's public `invocation_exception_count` attribute.
    pub fn invocation_exception_count(&self) -> usize {
        self.invocation_exception_count.load(Ordering::SeqCst)
    }

    /// Convert into a [`ToolDefinition`] for use in chat options.
    pub fn into_definition(self) -> ToolDefinition {
        let approval_mode = self.approval_mode;
        ToolDefinition::from_tool(Arc::new(self)).with_approval_mode(approval_mode)
    }
}

#[async_trait]
impl Tool for FunctionTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }

    /// Run the wrapped function, first enforcing
    /// [`FunctionTool::max_invocations`] and
    /// [`FunctionTool::max_invocation_exceptions`] (mirrors Python's
    /// `AIFunction.__call__`, `_tools.py:683-707`): a limit that has already
    /// been reached errors *before* the function runs and *before*
    /// [`FunctionTool::invocation_count`] is bumped again, so calling an
    /// already-exhausted function any number of further times does not
    /// drift its counters.
    ///
    /// The invocation slot is *reserved atomically* (`fetch_update`), because
    /// the function-invocation loop executes a model's parallel calls to the
    /// same tool concurrently — a plain check-then-increment would let two
    /// racing calls both slip under `max_invocations`.
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        let invocation_limit_error = || {
            Error::tool(format!(
                "Function '{}' has reached its maximum invocation limit, \
                 you can no longer use this tool.",
                self.name
            ))
        };
        // Fast-path check first so the error precedence between the two
        // limits matches Python's sequential check order.
        if let Some(max) = self.max_invocations {
            if self.invocation_count.load(Ordering::SeqCst) >= max {
                return Err(invocation_limit_error());
            }
        }
        if let Some(max) = self.max_invocation_exceptions {
            if self.invocation_exception_count.load(Ordering::SeqCst) >= max {
                return Err(Error::tool(format!(
                    "Function '{}' has reached its maximum exception limit, \
                     you tried to use this tool too many times and it kept failing.",
                    self.name
                )));
            }
        }
        match self.max_invocations {
            Some(max) => {
                // Reserve or bail: a concurrent call may have consumed the
                // last slot since the fast-path check above.
                if self
                    .invocation_count
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |c| {
                        (c < max).then(|| c + 1)
                    })
                    .is_err()
                {
                    return Err(invocation_limit_error());
                }
            }
            None => {
                self.invocation_count.fetch_add(1, Ordering::SeqCst);
            }
        }
        let result = (self.func)(arguments).await;
        if result.is_err() {
            self.invocation_exception_count
                .fetch_add(1, Ordering::SeqCst);
        }
        result
    }
}

/// Construct a hosted code-interpreter tool marker.
pub fn hosted_code_interpreter() -> ToolDefinition {
    ToolDefinition {
        name: "code_interpreter".into(),
        description: String::new(),
        parameters: empty_schema(),
        kind: ToolKind::HostedCodeInterpreter,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// Construct a hosted image-generation tool marker.
///
/// Supported by services that expose server-side image generation as a tool
/// (e.g. the OpenAI Responses API's `image_generation` tool). Its results
/// surface as [`Content::ImageGenerationToolResult`](crate::types::Content).
pub fn hosted_image_generation() -> ToolDefinition {
    ToolDefinition {
        name: "image_generation".into(),
        description: String::new(),
        parameters: empty_schema(),
        kind: ToolKind::HostedImageGeneration,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// Construct a hosted web-search tool marker.
pub fn hosted_web_search() -> ToolDefinition {
    ToolDefinition {
        name: "web_search".into(),
        description: String::new(),
        parameters: empty_schema(),
        kind: ToolKind::HostedWebSearch,
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// Construct a hosted file-search tool marker.
pub fn hosted_file_search(max_results: Option<u32>) -> ToolDefinition {
    ToolDefinition {
        name: "file_search".into(),
        description: String::new(),
        parameters: empty_schema(),
        kind: ToolKind::HostedFileSearch { max_results },
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// Construct a hosted MCP tool marker.
pub fn hosted_mcp(
    name: impl Into<String>,
    url: impl Into<String>,
    allowed_tools: Option<Vec<String>>,
) -> ToolDefinition {
    ToolDefinition {
        name: name.into(),
        description: String::new(),
        parameters: empty_schema(),
        kind: ToolKind::HostedMcp {
            url: url.into(),
            allowed_tools,
        },
        approval_mode: ApprovalMode::NeverRequire,
        executor: None,
    }
}

/// An empty JSON-Schema object (no parameters).
pub fn empty_schema() -> Value {
    serde_json::json!({ "type": "object", "properties": {} })
}

/// Derive an OpenAI-style parameters JSON Schema for `Args` via `schemars`,
/// stripping the top-level `$schema`/`title` keys. See [`FunctionTool::typed`]
/// for exactly what this does and does not normalize.
fn typed_parameters_schema<Args: schemars::JsonSchema>() -> Value {
    let root = schemars::gen::SchemaGenerator::default().into_root_schema_for::<Args>();
    let mut value = serde_json::to_value(root).unwrap_or_else(|_| empty_schema());
    if let Value::Object(map) = &mut value {
        map.remove("$schema");
        map.remove("title");
    }
    value
}

/// Configuration for the automatic function-invocation loop.
///
/// Mirrors `FunctionInvocationConfiguration`.
#[derive(Debug, Clone)]
pub struct FunctionInvocationConfig {
    pub enabled: bool,
    pub max_iterations: usize,
    pub max_consecutive_errors_per_request: usize,
    pub terminate_on_unknown_calls: bool,
    pub include_detailed_errors: bool,
}

impl Default for FunctionInvocationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_iterations: 40,
            max_consecutive_errors_per_request: 3,
            terminate_on_unknown_calls: false,
            include_detailed_errors: false,
        }
    }
}

impl FunctionInvocationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_iterations < 1 {
            return Err(Error::Configuration("max_iterations must be >= 1".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // region: typed() schema derivation

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    struct WeatherArgs {
        city: String,
        #[serde(default)]
        units: Option<String>,
    }

    #[test]
    fn typed_schema_required_and_optional_fields_exact_json() {
        let schema = typed_parameters_schema::<WeatherArgs>();
        assert_eq!(
            schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "units": { "type": ["string", "null"], "default": null },
                },
                "required": ["city"],
            })
        );
    }

    #[test]
    fn typed_schema_strips_schema_and_title_keys() {
        let schema = typed_parameters_schema::<WeatherArgs>();
        let obj = schema.as_object().expect("object schema");
        assert!(!obj.contains_key("$schema"));
        assert!(!obj.contains_key("title"));
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    enum Priority {
        Low,
        High,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct Address {
        city: String,
        zip: Option<String>,
    }

    #[derive(serde::Deserialize, schemars::JsonSchema)]
    #[allow(dead_code)]
    struct TaskArgs {
        title: String,
        address: Address,
        priority: Priority,
    }

    #[test]
    fn typed_schema_nested_struct_and_enum_keep_schemars_ref_definitions() {
        // Nested/enum fields are not inlined: they keep schemars' own
        // `$ref`/`definitions` representation, per `FunctionTool::typed`'s
        // documented contract (every provider converter forwards
        // `parameters` to the wire unmodified, so this round-trips fine).
        let schema = typed_parameters_schema::<TaskArgs>();
        assert_eq!(
            schema,
            serde_json::json!({
                "type": "object",
                "definitions": {
                    "Address": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "zip": { "type": ["string", "null"] },
                        },
                        "required": ["city"],
                    },
                    "Priority": {
                        "type": "string",
                        "enum": ["Low", "High"],
                    },
                },
                "properties": {
                    "title": { "type": "string" },
                    "address": { "$ref": "#/definitions/Address" },
                    "priority": { "$ref": "#/definitions/Priority" },
                },
                "required": ["address", "priority", "title"],
            })
        );
    }

    // endregion

    // region: typed() argument deserialization + result serialization

    #[tokio::test]
    async fn typed_invoke_deserializes_valid_arguments_and_serializes_result() {
        let tool = FunctionTool::typed(
            "get_weather",
            "Get the weather.",
            |args: WeatherArgs| async move {
                Ok(serde_json::json!({ "city": args.city, "units": args.units }))
            },
        );
        let result = tool
            .invoke(serde_json::json!({ "city": "Seattle" }))
            .await
            .unwrap();
        assert_eq!(
            result,
            serde_json::json!({ "city": "Seattle", "units": null })
        );
    }

    #[tokio::test]
    async fn typed_invoke_missing_required_field_errors_like_a_tool_error() {
        let tool = FunctionTool::typed(
            "get_weather",
            "Get the weather.",
            |_args: WeatherArgs| async move { Ok(serde_json::Value::Null) },
        );
        let err = tool.invoke(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, Error::Tool(_)));
        assert!(err.to_string().contains("get_weather"));
    }

    #[tokio::test]
    async fn typed_invoke_wrong_field_type_errors_like_a_tool_error() {
        let tool = FunctionTool::typed(
            "get_weather",
            "Get the weather.",
            |_args: WeatherArgs| async move { Ok(serde_json::Value::Null) },
        );
        let err = tool
            .invoke(serde_json::json!({ "city": 5 }))
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Tool(_)));
    }

    #[tokio::test]
    async fn typed_invoke_error_shape_matches_a_new_style_closure_error() {
        // `new` and `typed` funnel every failure through the same
        // `Result<Value>`-returning `Tool::invoke`; a bad-argument failure
        // from `typed` and a closure failure from `new` are indistinguishable
        // in shape to the caller (both `Err(Error::Tool(_))`), which is what
        // lets `execute_tool_call` in `client.rs` treat them identically.
        let untyped = FunctionTool::new("f", "d", empty_schema(), |_args| async move {
            Err(Error::tool("boom"))
        });
        let untyped_err = untyped.invoke(serde_json::json!({})).await.unwrap_err();

        let typed = FunctionTool::typed("f", "d", |_args: WeatherArgs| async move {
            Ok(serde_json::Value::Null)
        });
        let typed_err = typed.invoke(serde_json::json!({})).await.unwrap_err();

        assert!(matches!(untyped_err, Error::Tool(_)));
        assert!(matches!(typed_err, Error::Tool(_)));
    }

    #[derive(serde::Serialize)]
    struct WeatherResult {
        city: String,
        temp_c: i32,
    }

    #[tokio::test]
    async fn typed_invoke_serializes_a_generic_serializable_return_type() {
        // `Ret` need not be `serde_json::Value`; any `Serialize` type works.
        let tool = FunctionTool::typed(
            "get_weather",
            "Get the weather.",
            |args: WeatherArgs| async move {
                Ok(WeatherResult {
                    city: args.city,
                    temp_c: 21,
                })
            },
        );
        let result = tool
            .invoke(serde_json::json!({ "city": "Portland" }))
            .await
            .unwrap();
        assert_eq!(
            result,
            serde_json::json!({ "city": "Portland", "temp_c": 21 })
        );
    }

    // endregion

    // region: invocation limits

    #[tokio::test]
    async fn max_invocations_blocks_calls_past_the_limit() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_clone = Arc::clone(&calls);
        let tool = FunctionTool::new("f", "d", empty_schema(), move |_args| {
            let calls = Arc::clone(&calls_clone);
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(serde_json::Value::Null)
            }
        })
        .max_invocations(2);

        tool.invoke(serde_json::json!({})).await.unwrap();
        tool.invoke(serde_json::json!({})).await.unwrap();
        let err = tool.invoke(serde_json::json!({})).await.unwrap_err();

        // The third call was blocked before the closure ran.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(tool.invocation_count(), 2);
        assert!(matches!(err, Error::Tool(_)));
        assert_eq!(
            err.to_string(),
            "tool error: Function 'f' has reached its maximum invocation limit, \
             you can no longer use this tool."
        );
    }

    #[tokio::test]
    async fn max_invocations_holds_under_concurrent_calls() {
        // Two parallel calls race for a single invocation slot: exactly one
        // may execute. The closure parks on a Notify so both tasks are
        // genuinely in-flight together — with a check-then-add limit both
        // would slip through; the atomic reservation admits only one.
        let gate = Arc::new(tokio::sync::Notify::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let (gate_c, calls_c) = (Arc::clone(&gate), Arc::clone(&calls));
        let tool = Arc::new(
            FunctionTool::new("f", "d", empty_schema(), move |_args| {
                let (gate, calls) = (Arc::clone(&gate_c), Arc::clone(&calls_c));
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    gate.notified().await;
                    Ok(serde_json::Value::Null)
                }
            })
            .max_invocations(1),
        );

        let (t1, t2) = (Arc::clone(&tool), Arc::clone(&tool));
        let h1 = tokio::spawn(async move { t1.invoke(serde_json::json!({})).await });
        let h2 = tokio::spawn(async move { t2.invoke(serde_json::json!({})).await });
        // Let both tasks reach the limit check before releasing the gate.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;
        gate.notify_waiters();
        gate.notify_waiters();
        let (r1, r2) = (h1.await.unwrap(), h2.await.unwrap());

        let successes = [&r1, &r2].iter().filter(|r| r.is_ok()).count();
        assert_eq!(successes, 1, "exactly one call may claim the single slot");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the closure ran once");
        assert_eq!(tool.invocation_count(), 1);
        let err = [r1, r2].into_iter().find_map(|r| r.err()).unwrap();
        assert!(err.to_string().contains("maximum invocation limit"));
    }

    #[tokio::test]
    async fn max_invocation_exceptions_blocks_calls_past_the_limit() {
        let tool = FunctionTool::new("f", "d", empty_schema(), |_args| async move {
            Err(Error::tool("boom"))
        })
        .max_invocation_exceptions(2);

        let first = tool.invoke(serde_json::json!({})).await.unwrap_err();
        let second = tool.invoke(serde_json::json!({})).await.unwrap_err();
        assert_eq!(first.to_string(), "tool error: boom");
        assert_eq!(second.to_string(), "tool error: boom");
        assert_eq!(tool.invocation_exception_count(), 2);

        // The third call is blocked by the exception cap rather than
        // re-running (and failing) the closure again.
        let third = tool.invoke(serde_json::json!({})).await.unwrap_err();
        assert_eq!(
            third.to_string(),
            "tool error: Function 'f' has reached its maximum exception limit, \
             you tried to use this tool too many times and it kept failing."
        );
        assert_eq!(tool.invocation_exception_count(), 2);
    }

    #[tokio::test]
    async fn invocation_counters_are_shared_across_clones() {
        let tool = FunctionTool::new("f", "d", empty_schema(), |_args| async move {
            Ok(serde_json::Value::Null)
        })
        .max_invocations(1);
        let clone = tool.clone();

        clone.invoke(serde_json::json!({})).await.unwrap();
        let err = tool.invoke(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, Error::Tool(_)));
        assert_eq!(tool.invocation_count(), 1);
    }

    // endregion

    // region: hosted-tool builder setters (parameter-key contract)

    #[test]
    fn description_setter_sets_tool_definition_description() {
        let tool = hosted_mcp("docs", "https://mcp.example.com", None).description("My MCP");
        assert_eq!(tool.description, "My MCP");
    }

    #[test]
    fn user_location_setter_sets_parameter_key() {
        let loc = serde_json::json!({ "city": "Seattle", "country": "US" });
        let tool = hosted_web_search().user_location(loc.clone());
        assert_eq!(tool.parameters["user_location"], loc);
    }

    #[test]
    fn max_uses_setter_sets_parameter_key() {
        let tool = hosted_web_search().max_uses(5);
        assert_eq!(tool.parameters["max_uses"], serde_json::json!(5));
    }

    #[test]
    fn connection_id_setter_sets_parameter_key() {
        let tool = hosted_web_search().connection_id("conn-1");
        assert_eq!(
            tool.parameters["connection_id"],
            serde_json::json!("conn-1")
        );
    }

    #[test]
    fn custom_connection_setter_sets_both_parameter_keys() {
        let tool = hosted_web_search().custom_connection("custom-conn", "my-instance");
        assert_eq!(
            tool.parameters["custom_connection_id"],
            serde_json::json!("custom-conn")
        );
        assert_eq!(
            tool.parameters["instance_name"],
            serde_json::json!("my-instance")
        );
    }

    #[test]
    fn vector_store_ids_setter_sets_parameter_key() {
        let tool = hosted_file_search(None).vector_store_ids(vec!["vs_1".into(), "vs_2".into()]);
        assert_eq!(
            tool.parameters["vector_store_ids"],
            serde_json::json!(["vs_1", "vs_2"])
        );
    }

    #[test]
    fn max_results_setter_sets_parameter_key() {
        let tool = hosted_file_search(None).max_results(7);
        assert_eq!(tool.parameters["max_results"], serde_json::json!(7));
    }

    #[test]
    fn file_ids_setter_sets_parameter_key() {
        let tool = hosted_code_interpreter().file_ids(vec!["file-1".into()]);
        assert_eq!(tool.parameters["file_ids"], serde_json::json!(["file-1"]));
    }

    #[test]
    fn container_setter_sets_parameter_key() {
        let container = serde_json::json!({ "type": "secure", "id": "c1" });
        let tool = hosted_code_interpreter().container(container.clone());
        assert_eq!(tool.parameters["container"], container);
    }

    #[test]
    fn headers_setter_sets_parameter_key() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer x".to_string());
        let tool = hosted_mcp("docs", "https://mcp.example.com", None).headers(headers);
        assert_eq!(
            tool.parameters["headers"],
            serde_json::json!({ "authorization": "Bearer x" })
        );
    }

    #[test]
    fn mcp_approval_mode_always_sets_string_parameter() {
        let tool = hosted_mcp("docs", "https://mcp.example.com", None)
            .mcp_approval_mode(McpApprovalMode::Always);
        assert_eq!(
            tool.parameters["approval_mode"],
            serde_json::json!("always_require")
        );
    }

    #[test]
    fn mcp_approval_mode_never_sets_string_parameter() {
        let tool = hosted_mcp("docs", "https://mcp.example.com", None)
            .mcp_approval_mode(McpApprovalMode::Never);
        assert_eq!(
            tool.parameters["approval_mode"],
            serde_json::json!("never_require")
        );
    }

    #[test]
    fn mcp_approval_mode_per_tool_sets_object_parameter_with_both_sides() {
        let tool = hosted_mcp("docs", "https://mcp.example.com", None).mcp_approval_mode(
            McpApprovalMode::PerTool {
                always: vec!["delete".to_string()],
                never: vec!["read".to_string()],
            },
        );
        assert_eq!(
            tool.parameters["approval_mode"],
            serde_json::json!({ "always": ["delete"], "never": ["read"] })
        );
    }

    #[test]
    fn mcp_approval_mode_per_tool_omits_the_empty_side() {
        // Regression guard for the Azure AI Foundry converter, which treats
        // *presence* of the "always" key as the whole `require_approval`
        // decision (`if let Some(always) = ... else if let Some(never) =
        // ...`): an unconditionally-emitted empty `"always": []` would
        // silently defeat a never-only `PerTool` config there.
        let never_only = hosted_mcp("docs", "https://mcp.example.com", None).mcp_approval_mode(
            McpApprovalMode::PerTool {
                always: vec![],
                never: vec!["read".to_string()],
            },
        );
        assert_eq!(
            never_only.parameters["approval_mode"],
            serde_json::json!({ "never": ["read"] })
        );

        let always_only = hosted_mcp("docs", "https://mcp.example.com", None).mcp_approval_mode(
            McpApprovalMode::PerTool {
                always: vec!["delete".to_string()],
                never: vec![],
            },
        );
        assert_eq!(
            always_only.parameters["approval_mode"],
            serde_json::json!({ "always": ["delete"] })
        );
    }

    #[test]
    fn setters_chain_together_on_a_single_hosted_mcp_tool() {
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), "Bearer x".to_string());
        let tool = hosted_mcp("docs", "https://mcp.example.com", None)
            .description("Docs server")
            .headers(headers.clone())
            .mcp_approval_mode(McpApprovalMode::Always);
        assert_eq!(tool.description, "Docs server");
        assert_eq!(tool.parameters["headers"], serde_json::json!(headers));
        assert_eq!(
            tool.parameters["approval_mode"],
            serde_json::json!("always_require")
        );
    }

    // endregion
}
