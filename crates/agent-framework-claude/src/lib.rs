//! The Claude Code CLI as an agent: Rust counterpart of upstream's
//! `agent_framework_claude` (`ClaudeAgent`).
//!
//! Upstream builds on the Python Claude Agent SDK, which drives the `claude`
//! CLI as a subprocess. There is no Rust SDK, so this crate drives the CLI
//! directly over the same stream-json protocol the SDK speaks: the CLI runs
//! Claude's own agent loop (file edits, shell, web, MCP servers, sub-agents)
//! and this agent turns its partial messages into [`AgentResponseUpdate`]s.
//!
//! ```no_run
//! use agent_framework_claude::{ClaudeAgent, ClaudeAgentOptions};
//! use agent_framework_core::agent::SupportsAgentRun;
//! use agent_framework_core::types::Message;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let agent = ClaudeAgent::builder()
//!     .instructions("You are a careful code reviewer.")
//!     .builtin_tool("Read")
//!     .builtin_tool("Grep")
//!     .options(ClaudeAgentOptions {
//!         model: Some("sonnet".into()),
//!         permission_mode: Some("plan".into()),
//!         ..Default::default()
//!     })
//!     .build()?;
//! let mut session = agent.create_session();
//! let first = agent
//!     .run(vec![Message::user("Summarize src/lib.rs")], Some(&mut session))
//!     .await?;
//! // The session now carries the CLI's session id; the next run resumes it.
//! let next = agent
//!     .run(vec![Message::user("Any bugs?")], Some(&mut session))
//!     .await?;
//! # let _ = (first, next);
//! # Ok(())
//! # }
//! ```
//!
//! # How a run works
//!
//! Each run starts one CLI process (`--output-format stream-json
//! --input-format stream-json --verbose --include-partial-messages`, plus
//! flags from [`ClaudeAgentOptions`]), resuming the session's CLI
//! conversation (`--resume=<id>`) when the session has one. It sends the
//! `initialize` control request (hook registrations, sub-agents, skills),
//! applies per-run overrides (`set_model` from
//! [`ChatOptions::model`](agent_framework_core::types::ChatOptions),
//! `set_permission_mode` from its `additional_properties["permission_mode"]`),
//! writes the prompt as one `user` message and reads until the `result`:
//!
//! - `text_delta` / `thinking_delta` partial messages become text and
//!   reasoning updates;
//! - an `assistant` message carrying an `error` fails the run;
//! - the `result` becomes a final update with usage, finish reason (Claude's
//!   stop reasons mapped onto the framework's), the CLI session id as
//!   `conversation_id`, `total_cost_usd`, and any structured output (the
//!   response's `value` for a non-streaming run). An error result fails the
//!   run.
//!
//! Then stdin is closed and the process exits. Sessions are isolated by
//! construction: nothing is shared between runs but the CLI's own session
//! store.
//!
//! Several messages are passed as a JSON-lines transcript under a fixed
//! preamble ([`format_prompt`]); a single user message is passed as is.
//!
//! # Tools
//!
//! - Built-in CLI tools by name ([`ClaudeAgentBuilder::builtin_tool`]) are
//!   passed as `--tools`.
//! - Function tools ([`ClaudeAgentBuilder::tool`], and executable
//!   [`AgentRunOptions::additional_tools`](agent_framework_core::agent::AgentRunOptions))
//!   are served from an in-process MCP server ([`TOOLS_MCP_SERVER_NAME`])
//!   that the CLI reaches through `mcp_message` control requests, and are
//!   allowed automatically. A tool with
//!   [`ApprovalMode::AlwaysRequire`](agent_framework_core::tools::ApprovalMode)
//!   runs only when [`ClaudeAgentOptions::on_function_approval`] approves
//!   it; without a handler it is denied.
//! - MCP servers go in [`ClaudeAgentOptions::mcp_servers`]; the CLI connects
//!   to them itself. A hosted MCP tool is refused at build time.
//! - [`ClaudeAgentOptions::can_use_tool`] answers the CLI's permission
//!   prompts; [`ClaudeAgentOptions::hooks`] registers hook callbacks.
//!
//! # Divergences
//!
//! - No injected `ClaudeSDKClient`: every run owns its process.
//! - The agent runs no context providers or middleware (as the other
//!   remote agents here); history lives in the CLI's session.
//! - The CLI version check and bundled CLI of the Python SDK are not
//!   ported; the CLI is found on `PATH` or the usual install locations, or
//!   set with [`ClaudeAgentOptions::cli_path`] / `CLAUDE_AGENT_CLI_PATH`.
//! - A streamed run's session id is reported on the final update's
//!   `conversation_id`; the stream does not write it into the session it
//!   consumes.
//! - SDK features upstream does not use (session stores and mirroring,
//!   interrupts, rewind, MCP status/toggle, task tracking) are not ported.

mod agent;
mod mcp;
mod options;
mod transport;

pub use agent::{
    format_prompt, ClaudeAgent, ClaudeAgentBuilder, AGENT_PROVIDER_NAME, STRUCTURED_OUTPUT_KEY,
};
pub use mcp::{qualified_tool_name, TOOLS_MCP_SERVER_NAME};
pub use options::{
    ClaudeAgentOptions, FunctionApprovalHandler, HookCallback, HookMatcher, PermissionResult,
    ThinkingConfig, ToolPermissionHandler, ToolPermissionRequest, ENV_PREFIX,
};

#[cfg(doc)]
use agent_framework_core::types::AgentResponseUpdate;
