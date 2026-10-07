//! A shell tool for agents: run model-written commands locally or in a
//! container.
//!
//! Port of upstream's `agent_framework_tools.shell` (Python) and
//! `Microsoft.Agents.AI.Tools.Shell` (.NET).
//!
//! | Item | Role |
//! | --- | --- |
//! | [`LocalShellTool`] | Runs commands on the host. Approval required by default. |
//! | [`DockerShellTool`] | Runs commands in a container with locked-down defaults. |
//! | [`ShellExecutor`] | The backend trait both implement; plug in your own. |
//! | [`ShellPolicy`] | Allow/deny pre-filter (not a security boundary). |
//! | [`ShellEnvironmentProvider`] | Tells the model which shell, OS and CLIs it has. |
//! | [`ShellSession`] | The persistent-shell protocol both tools use. |
//!
//! # Security model
//!
//! A shell tool hands the model arbitrary code execution. The defences, in
//! order of strength:
//!
//! 1. **Sandbox tier.** [`DockerShellTool`] isolates commands from the host
//!    (no network, read-only root, dropped capabilities, unprivileged user,
//!    memory and process caps). [`LocalShellTool`] has no isolation at all.
//! 2. **Approval in the loop.** The function from `as_function` requires a
//!    human to approve every call by default. On [`LocalShellTool`]
//!    switching that off needs an explicit `acknowledge_unsafe(true)`.
//! 3. **[`ShellPolicy`]**: a UX pre-filter for operator-specific patterns.
//!    A model can trivially bypass it; it ships with no patterns so it does
//!    not suggest otherwise.
//!
//! Commands are shell syntax, so they are passed to a shell (`bash -c`,
//! `pwsh -Command`, or a persistent shell's stdin), as upstream does. The
//! tool's own plumbing (the shell binary, `docker` invocations) is always
//! exec'd with an argv, never through a host shell. A timeout terminates the
//! command's whole process group, not just the shell.
//!
//! # Hosted shell content
//!
//! Upstream marks the tool's function with `kind="shell"` so the OpenAI
//! Responses client can declare it as a native local shell and translate
//! `shell_call` / `shell_call_output` items. This port's tool is an ordinary
//! function tool, which works with every chat client; the structured result
//! converts to the hosted-shell content types with
//! [`ShellResult::to_command_output`] and [`ShellResult::to_tool_result`].

mod docker;
mod environment;
mod executor;
mod policy;
mod process;
mod resolve;
mod session;
mod tool;
mod truncate;
mod types;

pub use docker::{
    is_docker_available, DockerShellTool, DockerShellToolBuilder, DEFAULT_CONTAINER_USER,
    DEFAULT_IMAGE, DEFAULT_MEMORY, DEFAULT_NETWORK, DEFAULT_PIDS_LIMIT, DEFAULT_WORKDIR,
};
pub use environment::{
    default_instructions_formatter, ShellEnvironmentProvider, ShellEnvironmentProviderOptions,
    ShellEnvironmentSnapshot, ShellFamily,
};
pub use executor::ShellExecutor;
pub use policy::{ShellDecision, ShellPolicy, ShellRequest};
pub use resolve::{is_powershell, resolve_shell, ShellSpec, SHELL_ENV_OVERRIDE};
pub use session::ShellSession;
pub use tool::{
    LocalShellTool, LocalShellToolBuilder, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT,
    DEFAULT_TOOL_NAME,
};
pub use truncate::{truncate_head_tail, truncate_text_head_tail};
pub use types::{ShellError, ShellMode, ShellResult};
