//! Agent options and the callbacks the CLI consults during a run.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use agent_framework_core::error::Result;
use agent_framework_core::types::FunctionCallContent;
use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::{Map, Value};

/// Environment variable prefix for the [`ClaudeAgentOptions`] settings read
/// by [`ClaudeAgentOptions::from_env`] (`CLAUDE_AGENT_CLI_PATH`,
/// `CLAUDE_AGENT_MODEL`, `CLAUDE_AGENT_CWD`, `CLAUDE_AGENT_PERMISSION_MODE`,
/// `CLAUDE_AGENT_MAX_TURNS`, `CLAUDE_AGENT_MAX_BUDGET_USD`).
pub const ENV_PREFIX: &str = "CLAUDE_AGENT_";

/// Extended thinking configuration (`--thinking` /
/// `--max-thinking-tokens`).
#[derive(Debug, Clone, PartialEq)]
pub enum ThinkingConfig {
    /// The model decides (`--thinking adaptive`).
    Adaptive {
        /// `--thinking-display`, e.g. `"summarized"`.
        display: Option<String>,
    },
    /// A fixed budget (`--max-thinking-tokens`).
    Enabled {
        /// The token budget.
        budget_tokens: u64,
        /// `--thinking-display`.
        display: Option<String>,
    },
    /// No thinking (`--thinking disabled`).
    Disabled,
}

/// The CLI's answer to a tool-permission request.
#[derive(Debug, Clone, PartialEq)]
pub enum PermissionResult {
    /// Allow the call, optionally with rewritten input or permission
    /// updates (`updatedPermissions`, CLI-shaped JSON).
    Allow {
        /// Replacement input; the original is used when `None`.
        updated_input: Option<Value>,
        /// Permission updates to apply.
        updated_permissions: Option<Vec<Value>>,
    },
    /// Deny the call with a message shown to the model.
    Deny {
        /// Why.
        message: String,
        /// Whether to interrupt the whole turn.
        interrupt: bool,
    },
}

impl PermissionResult {
    /// Allow unchanged.
    pub fn allow() -> Self {
        Self::Allow {
            updated_input: None,
            updated_permissions: None,
        }
    }

    /// Deny with `message`.
    pub fn deny(message: impl Into<String>) -> Self {
        Self::Deny {
            message: message.into(),
            interrupt: false,
        }
    }
}

/// A tool-permission request from the CLI (`can_use_tool`).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolPermissionRequest {
    /// The tool the model wants to call (built-in, MCP or this agent's).
    pub tool_name: String,
    /// The call's input.
    pub input: Value,
    /// The tool-use id, when the CLI sends one.
    pub tool_use_id: Option<String>,
    /// Permission updates the CLI suggests (CLI-shaped JSON).
    pub suggestions: Vec<Value>,
    /// The path that triggered the request, if any.
    pub blocked_path: Option<String>,
    /// Why the CLI is asking.
    pub decision_reason: Option<String>,
    /// The sub-agent asking, if any.
    pub agent_id: Option<String>,
    /// The request as received (`title`, `display_name`, `description`, ...).
    pub raw: Value,
}

/// Decides tool permissions the CLI defers to the host (the SDK's
/// `can_use_tool`). Setting one makes the CLI route permission prompts here
/// (`--permission-prompt-tool stdio`). An error answers the CLI with a
/// control error, which it treats as a denial.
#[async_trait]
pub trait ToolPermissionHandler: Send + Sync {
    /// Decide one request.
    async fn can_use_tool(&self, request: ToolPermissionRequest) -> Result<PermissionResult>;
}

#[async_trait]
impl<F> ToolPermissionHandler for F
where
    F: Fn(ToolPermissionRequest) -> BoxFuture<'static, Result<PermissionResult>> + Send + Sync,
{
    async fn can_use_tool(&self, request: ToolPermissionRequest) -> Result<PermissionResult> {
        self(request).await
    }
}

/// Approves calls to this agent's tools declared with
/// [`ApprovalMode::AlwaysRequire`](agent_framework_core::tools::ApprovalMode)
/// (upstream's `on_function_approval`).
///
/// The CLI runs its own tool loop, so the framework cannot round-trip an
/// approval request through the caller; this callback is the enforcement
/// point instead. It receives the pending call (with a synthetic
/// `af-claude-approval::<tool>` call id) and returns whether to run it.
/// Without a handler such calls are denied.
#[async_trait]
pub trait FunctionApprovalHandler: Send + Sync {
    /// Whether to run `call`.
    async fn approve(&self, call: FunctionCallContent) -> bool;
}

#[async_trait]
impl<F> FunctionApprovalHandler for F
where
    F: Fn(FunctionCallContent) -> BoxFuture<'static, bool> + Send + Sync,
{
    async fn approve(&self, call: FunctionCallContent) -> bool {
        self(call).await
    }
}

/// A hook callback: receives the hook input and the tool-use id, returns the
/// CLI-shaped hook output (`{"continue": ..., "decision": ...,
/// "hookSpecificOutput": ...}`; use the CLI's names, e.g. `continue` and
/// `async`).
pub type HookCallback =
    Arc<dyn Fn(Value, Option<String>) -> BoxFuture<'static, Result<Value>> + Send + Sync>;

/// Hooks for one event whose tool name matches `matcher`.
#[derive(Clone, Default)]
pub struct HookMatcher {
    /// A tool-name pattern (`"Bash"`, `"Write|Edit"`); `None` matches all.
    pub matcher: Option<String>,
    /// The callbacks, run by the CLI in order.
    pub hooks: Vec<HookCallback>,
    /// Timeout in seconds.
    pub timeout: Option<f64>,
}

impl std::fmt::Debug for HookMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookMatcher")
            .field("matcher", &self.matcher)
            .field("hooks", &self.hooks.len())
            .field("timeout", &self.timeout)
            .finish()
    }
}

/// Options for a [`ClaudeAgent`](crate::ClaudeAgent): how the CLI is
/// launched and configured. Mirrors upstream's `ClaudeAgentOptions`.
#[derive(Clone, Default)]
pub struct ClaudeAgentOptions {
    /// The system prompt (`--system-prompt`; empty when unset, as the SDK).
    pub system_prompt: Option<String>,
    /// The CLI executable. Default: `claude` on `PATH`, then the usual
    /// install locations.
    pub cli_path: Option<PathBuf>,
    /// The CLI's working directory.
    pub cwd: Option<PathBuf>,
    /// Extra environment variables for the CLI.
    pub env: HashMap<String, String>,
    /// A settings file path or JSON (`--settings`).
    pub settings: Option<String>,
    /// The model (`--model`), e.g. `"sonnet"`.
    pub model: Option<String>,
    /// The fallback model (`--fallback-model`).
    pub fallback_model: Option<String>,
    /// Tools Claude may use without asking (`--allowedTools`).
    pub allowed_tools: Vec<String>,
    /// Tools Claude may not use (`--disallowedTools`).
    pub disallowed_tools: Vec<String>,
    /// MCP servers the CLI connects to itself, CLI-shaped
    /// (`{"type": "stdio", "command": ..., "args": [...]}`).
    pub mcp_servers: Map<String, Value>,
    /// `default`, `acceptEdits`, `plan` or `bypassPermissions`.
    pub permission_mode: Option<String>,
    /// Maximum agentic turns (`--max-turns`).
    pub max_turns: Option<u32>,
    /// Spending limit (`--max-budget-usd`).
    pub max_budget_usd: Option<f64>,
    /// Extra directories in context (`--add-dir`).
    pub add_dirs: Vec<PathBuf>,
    /// Sandbox settings, merged into `--settings` as `sandbox`.
    pub sandbox: Option<Value>,
    /// Custom sub-agent definitions (sent with `initialize`).
    pub agents: Option<Map<String, Value>>,
    /// A JSON schema for structured output (`--json-schema`); the result's
    /// `structured_output` becomes the response's `value`.
    pub output_schema: Option<Value>,
    /// Enable file checkpointing for rewind.
    pub enable_file_checkpointing: bool,
    /// Beta features (`--betas`).
    pub betas: Vec<String>,
    /// Local plugin directories (`--plugin-dir`).
    pub plugin_dirs: Vec<PathBuf>,
    /// Which settings files to load (`--setting-sources`).
    pub setting_sources: Option<Vec<String>>,
    /// Extended thinking.
    pub thinking: Option<ThinkingConfig>,
    /// `low`, `medium`, `high`, `xhigh` or `max` (`--effort`).
    pub effort: Option<String>,
    /// Skills to enable (sent with `initialize`); `None` enables all.
    pub skills: Option<Vec<String>>,
    /// A fixed CLI session id (a UUID) for a new conversation.
    pub session_id: Option<String>,
    /// API-side task budget in tokens (`--task-budget`).
    pub task_budget: Option<u64>,
    /// Emit hook lifecycle events (`--include-hook-events`).
    pub include_hook_events: bool,
    /// Use only [`Self::mcp_servers`] (`--strict-mcp-config`).
    pub strict_mcp_config: bool,
    /// Continue the most recent conversation (`--continue`).
    pub continue_conversation: bool,
    /// Fork a resumed session to a new id (`--fork-session`).
    pub fork_session: bool,
    /// Further CLI flags: `(name, value)` becomes `--name value`, or
    /// `--name` when the value is `None`.
    pub extra_args: Vec<(String, Option<String>)>,
    /// Hooks by event name (`PreToolUse`, `PostToolUse`, ...).
    pub hooks: HashMap<String, Vec<HookMatcher>>,
    /// Tool-permission callback (`can_use_tool`).
    pub can_use_tool: Option<Arc<dyn ToolPermissionHandler>>,
    /// Approval callback for this agent's tools that require approval.
    pub on_function_approval: Option<Arc<dyn FunctionApprovalHandler>>,
}

impl std::fmt::Debug for ClaudeAgentOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeAgentOptions")
            .field("cli_path", &self.cli_path)
            .field("model", &self.model)
            .field("cwd", &self.cwd)
            .field("permission_mode", &self.permission_mode)
            .field("max_turns", &self.max_turns)
            .field("allowed_tools", &self.allowed_tools)
            .field("mcp_servers", &self.mcp_servers.keys().collect::<Vec<_>>())
            .field("hooks", &self.hooks.keys().collect::<Vec<_>>())
            .field("can_use_tool", &self.can_use_tool.is_some())
            .field("on_function_approval", &self.on_function_approval.is_some())
            .finish_non_exhaustive()
    }
}

impl ClaudeAgentOptions {
    /// Empty options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Fill unset settings from `CLAUDE_AGENT_*` environment variables
    /// (explicit values win, as upstream's `load_settings`).
    pub fn from_env(mut self) -> Self {
        let var = |name: &str| {
            std::env::var(format!("{ENV_PREFIX}{name}"))
                .ok()
                .filter(|v| !v.trim().is_empty())
        };
        if self.cli_path.is_none() {
            self.cli_path = var("CLI_PATH").map(PathBuf::from);
        }
        if self.model.is_none() {
            self.model = var("MODEL");
        }
        if self.cwd.is_none() {
            self.cwd = var("CWD").map(PathBuf::from);
        }
        if self.permission_mode.is_none() {
            self.permission_mode = var("PERMISSION_MODE");
        }
        if self.max_turns.is_none() {
            self.max_turns = var("MAX_TURNS").and_then(|v| v.parse().ok());
        }
        if self.max_budget_usd.is_none() {
            self.max_budget_usd = var("MAX_BUDGET_USD").and_then(|v| v.parse().ok());
        }
        self
    }
}
