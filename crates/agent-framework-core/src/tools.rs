//! Tools: executable functions and hosted-tool markers.
//!
//! Rust equivalent of `agent_framework._tools`. An [`AiFunction`] is a locally
//! executable tool; [`HostedTool`] variants are markers handed to the service.
//! Both are represented uniformly to a chat client as a [`ToolDefinition`].

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

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
}

/// The category of a tool as advertised to the service.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolKind {
    /// A callable function (executed locally, unless declaration-only).
    Function,
    /// Service-side code interpreter.
    HostedCodeInterpreter,
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
/// This is the Rust analogue of `AIFunction` / the `@ai_function` decorator.
#[derive(Clone)]
pub struct AiFunction {
    name: String,
    description: String,
    parameters: Value,
    approval_mode: ApprovalMode,
    func: ToolClosure,
}

impl AiFunction {
    /// Create a function tool.
    ///
    /// * `parameters` is the JSON Schema for the arguments object.
    /// * `func` receives the parsed JSON arguments and returns a JSON result.
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
        }
    }

    /// Builder: set the human-in-the-loop approval mode (default
    /// [`ApprovalMode::NeverRequire`]). Carried through to the
    /// [`ToolDefinition`] produced by [`AiFunction::into_definition`].
    pub fn with_approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Convert into a [`ToolDefinition`] for use in chat options.
    pub fn into_definition(self) -> ToolDefinition {
        let approval_mode = self.approval_mode;
        ToolDefinition::from_tool(Arc::new(self)).with_approval_mode(approval_mode)
    }
}

#[async_trait]
impl Tool for AiFunction {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn parameters_schema(&self) -> Value {
        self.parameters.clone()
    }
    async fn invoke(&self, arguments: Value) -> Result<Value> {
        (self.func)(arguments).await
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
