//! [`LocalShellTool`]: run model-written commands on the host.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::tools::{ApprovalMode, FunctionTool, ToolDefinition};
use async_trait::async_trait;
use serde_json::{json, Value};

use super::environment::ShellFamily;
use super::executor::ShellExecutor;
use super::policy::{ShellDecision, ShellPolicy, ShellRequest};
use super::process::run_stateless;
use super::resolve::{is_powershell, resolve_shell, ShellSpec};
use super::session::ShellSession;
use super::types::{ShellError, ShellMode, ShellResult};

/// Default per-command timeout.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);
/// Default byte limit per output stream.
pub const DEFAULT_MAX_OUTPUT_BYTES: usize = 64 * 1024;
/// Default name of the function the model calls.
pub const DEFAULT_TOOL_NAME: &str = "run_shell";

const PERSISTENT_DESCRIPTION: &str =
    "Execute a single shell command on the local machine and return its \
stdout, stderr, and exit code. Commands run in a persistent session so \
`cd` and environment variables from previous calls are preserved. \
Approval is required by default.";

const STATELESS_DESCRIPTION: &str =
    "Execute a single shell command on the local machine and return its \
stdout, stderr, and exit code. Each command runs in a fresh subprocess, \
so `cd` and environment variables do not persist between calls. \
Approval is required by default.";

pub(crate) type CommandHook = Arc<dyn Fn(&str) + Send + Sync>;

/// POSIX single-quote `value`: no expansion happens inside single quotes,
/// and an embedded `'` becomes `'\''`.
pub(crate) fn quote_posix(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

/// PowerShell single-quote `value`: literal, with `'` doubled.
pub(crate) fn quote_powershell(value: &str) -> String {
    format!("'{}'", value.replace('\'', "''"))
}

/// Apply `policy`, then fire the audit hook. Shared with the Docker tool.
pub(crate) fn admit(
    policy: &ShellPolicy,
    hook: Option<&CommandHook>,
    command: &str,
    workdir: Option<PathBuf>,
) -> Result<(), ShellError> {
    let request = ShellRequest {
        command: command.to_string(),
        workdir,
    };
    if let ShellDecision::Deny(reason) = policy.evaluate(&request) {
        return Err(ShellError::Rejected(reason));
    }
    if let Some(hook) = hook {
        // An audit hook must not take the tool down with it.
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| hook(command))).is_err() {
            tracing::error!("shell on_command hook panicked");
        }
    }
    Ok(())
}

/// The `run_shell` function definition shared by the local and Docker tools.
pub(crate) fn shell_function<E>(
    executor: E,
    name: &str,
    description: &str,
    approval_mode: ApprovalMode,
) -> ToolDefinition
where
    E: ShellExecutor + Clone + 'static,
{
    let parameters = json!({
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute."
            }
        },
        "required": ["command"]
    });
    let tool_name = name.to_string();
    FunctionTool::new(name, description, parameters, move |args: Value| {
        let executor = executor.clone();
        let tool_name = tool_name.clone();
        async move {
            let command = args
                .get("command")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    agent_framework_core::error::Error::tool(format!(
                        "invalid arguments for tool '{tool_name}': missing string field 'command'"
                    ))
                })?
                .to_string();
            match executor.run(&command, None).await {
                Ok(result) => Ok(Value::String(result.format_for_model())),
                // A policy rejection is an answer for the model, not a
                // failure: it sees the reason and can try something else.
                Err(err @ ShellError::Rejected(_)) => Ok(Value::String(err.to_string())),
                Err(err) => Err(err.into()),
            }
        }
    })
    .with_approval_mode(approval_mode)
    .into_definition()
}

/// Builder for [`LocalShellTool`]. Every setting has upstream's default.
#[derive(Clone)]
pub struct LocalShellToolBuilder {
    mode: ShellMode,
    shell: Option<ShellSpec>,
    workdir: Option<PathBuf>,
    confine_workdir: bool,
    env: Vec<(String, String)>,
    clean_env: bool,
    policy: ShellPolicy,
    timeout: Option<Duration>,
    max_output_bytes: usize,
    approval_mode: ApprovalMode,
    acknowledge_unsafe: bool,
    on_command: Option<CommandHook>,
}

impl Default for LocalShellToolBuilder {
    fn default() -> Self {
        Self {
            mode: ShellMode::Persistent,
            shell: None,
            workdir: None,
            confine_workdir: true,
            env: Vec::new(),
            clean_env: false,
            policy: ShellPolicy::new(),
            timeout: Some(DEFAULT_TIMEOUT),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            approval_mode: ApprovalMode::AlwaysRequire,
            acknowledge_unsafe: false,
            on_command: None,
        }
    }
}

impl std::fmt::Debug for LocalShellToolBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalShellToolBuilder")
            .field("mode", &self.mode)
            .field("shell", &self.shell)
            .field("workdir", &self.workdir)
            .field("confine_workdir", &self.confine_workdir)
            .field(
                "env_keys",
                &self.env.iter().map(|(k, _)| k).collect::<Vec<_>>(),
            )
            .field("clean_env", &self.clean_env)
            .field("policy", &self.policy)
            .field("timeout", &self.timeout)
            .field("max_output_bytes", &self.max_output_bytes)
            .field("approval_mode", &self.approval_mode)
            .finish_non_exhaustive()
    }
}

impl LocalShellToolBuilder {
    /// [`ShellMode::Persistent`] (default) or [`ShellMode::Stateless`].
    pub fn mode(mut self, mode: ShellMode) -> Self {
        self.mode = mode;
        self
    }

    /// The shell to run: a command line (`"bash --norc"`, split with POSIX
    /// shell-word rules) or an argv. Defaults to the
    /// `AGENT_FRAMEWORK_SHELL` environment variable, then the platform
    /// default (`pwsh`/`powershell` on Windows, `bash` or `sh` elsewhere).
    pub fn shell(mut self, shell: impl Into<ShellSpec>) -> Self {
        self.shell = Some(shell.into());
        self
    }

    /// The working directory for commands. Defaults to the process's own.
    pub fn workdir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.workdir = Some(dir.into());
        self
    }

    /// In persistent mode, prefix every command with a `cd` back into the
    /// working directory so one call's `cd` does not leak into the next.
    /// Default `true`.
    ///
    /// This is a re-anchor, **not** confinement: `cd /tmp && rm -rf .` in a
    /// single call still runs in `/tmp`. Neither this nor [`ShellPolicy`]
    /// restricts file access; only an isolated executor does.
    pub fn confine_workdir(mut self, confine: bool) -> Self {
        self.confine_workdir = confine;
        self
    }

    /// Add an environment variable for every command.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// Add several environment variables for every command.
    pub fn envs<I, K, V>(mut self, vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        self.env
            .extend(vars.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
    }

    /// Do not inherit this process's environment: commands see only the
    /// variables given with [`env`](Self::env). Default `false`.
    ///
    /// Use it to keep the host's secrets (API keys, tokens) out of reach of
    /// model-written commands. With no `PATH`, commands must be named by
    /// absolute path unless the shell supplies its own default.
    pub fn clean_env(mut self, clean: bool) -> Self {
        self.clean_env = clean;
        self
    }

    /// The policy applied before approval. Defaults to an empty policy that
    /// allows every non-empty command. A UX pre-filter, not a security
    /// boundary: see [`ShellPolicy`].
    pub fn policy(mut self, policy: ShellPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The per-command timeout; `None` disables it. Default 30 seconds.
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// The byte limit per output stream before head/tail truncation.
    /// Default 64 KiB. Must be positive.
    pub fn max_output_bytes(mut self, max: usize) -> Self {
        self.max_output_bytes = max;
        self
    }

    /// The approval mode of the function returned by
    /// [`LocalShellTool::as_function`]. Default
    /// [`ApprovalMode::AlwaysRequire`].
    ///
    /// **Approval is this tool's security boundary.** Setting
    /// [`ApprovalMode::NeverRequire`] makes [`build`](Self::build) fail unless
    /// [`acknowledge_unsafe`](Self::acknowledge_unsafe) is also set.
    pub fn approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Confirm that running without approval is intended: every command the
    /// model writes then runs on the host with this process's privileges,
    /// and [`ShellPolicy`] will not stop a determined model. For untrusted
    /// input prefer [`DockerShellTool`](super::DockerShellTool).
    pub fn acknowledge_unsafe(mut self, acknowledge: bool) -> Self {
        self.acknowledge_unsafe = acknowledge;
        self
    }

    /// An audit hook called with every command that passes the policy, for
    /// logging or telemetry. A panic in the hook is caught and logged.
    pub fn on_command<F>(mut self, hook: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_command = Some(Arc::new(hook));
        self
    }

    /// Build the tool, resolving the shell now.
    pub fn build(self) -> Result<LocalShellTool, ShellError> {
        if self.approval_mode == ApprovalMode::NeverRequire && !self.acknowledge_unsafe {
            return Err(ShellError::Config(
                "Setting approval mode to NeverRequire disables the only built-in security \
                 boundary of LocalShellTool. If you understand the risk (arbitrary commands \
                 run on the host with the agent's privileges; ShellPolicy is a UX pre-filter, \
                 not a defense), call acknowledge_unsafe(true). For untrusted input prefer a \
                 sandboxed executor such as DockerShellTool."
                    .into(),
            ));
        }
        if self.max_output_bytes == 0 {
            return Err(ShellError::Config(
                "max_output_bytes must be positive".into(),
            ));
        }
        let env = if self.clean_env {
            Some(self.env.into_iter().collect::<HashMap<_, _>>())
        } else if self.env.is_empty() {
            None
        } else {
            let mut merged: HashMap<String, String> = std::env::vars().collect();
            merged.extend(self.env);
            Some(merged)
        };
        let interactive_argv = resolve_shell(self.shell.as_ref(), true)?;
        let stateless_argv = resolve_shell(self.shell.as_ref(), false)?;
        let session = (self.mode == ShellMode::Persistent).then(|| {
            ShellSession::new(
                interactive_argv.clone(),
                self.workdir.clone(),
                env.clone(),
                self.max_output_bytes,
            )
        });
        Ok(LocalShellTool {
            inner: Arc::new(Inner {
                mode: self.mode,
                workdir: self.workdir,
                confine_workdir: self.confine_workdir,
                env,
                policy: self.policy,
                timeout: self.timeout,
                max_output_bytes: self.max_output_bytes,
                approval_mode: self.approval_mode,
                on_command: self.on_command,
                interactive_argv,
                stateless_argv,
                session,
            }),
        })
    }
}

struct Inner {
    mode: ShellMode,
    workdir: Option<PathBuf>,
    confine_workdir: bool,
    env: Option<HashMap<String, String>>,
    policy: ShellPolicy,
    timeout: Option<Duration>,
    max_output_bytes: usize,
    approval_mode: ApprovalMode,
    on_command: Option<CommandHook>,
    interactive_argv: Vec<String>,
    stateless_argv: Vec<String>,
    session: Option<ShellSession>,
}

/// A shell tool that runs model-written commands on the local machine.
///
/// Port of upstream's `LocalShellTool` (.NET `LocalShellExecutor`). Commands
/// go through the configured shell (`bash -c`, `pwsh -Command`, or a
/// persistent shell reading stdin), exactly as upstream: the command is shell
/// syntax by design, so it is **not** split into an argv.
///
/// # Security
///
/// * There is **no process-level isolation**: commands run with this
///   process's user, filesystem, network and environment.
/// * **Approval is the boundary.** The function from
///   [`as_function`](Self::as_function) requires human approval by default,
///   and turning that off needs an explicit
///   [`acknowledge_unsafe`](LocalShellToolBuilder::acknowledge_unsafe).
/// * [`ShellPolicy`] is a UX pre-filter that a model can trivially bypass.
/// * [`confine_workdir`](LocalShellToolBuilder::confine_workdir) re-anchors
///   each command; it does not confine file access.
/// * By default commands inherit this process's environment, secrets
///   included; use [`clean_env`](LocalShellToolBuilder::clean_env) to stop
///   that.
/// * For untrusted input use [`DockerShellTool`](super::DockerShellTool), and
///   keep approval on.
///
/// # Single-session ownership
///
/// A persistent-mode tool belongs to one conversation, i.e. one user: its
/// shell keeps state across calls and serialises them. Clones share that
/// shell. Create one tool per session and [`close`](ShellExecutor::close) it
/// when the session ends; use [`ShellMode::Stateless`] if one instance must
/// be shared.
///
/// ```no_run
/// use agent_framework_core::prelude::*;
/// use agent_framework_tools::shell::LocalShellTool;
///
/// # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
/// let shell = LocalShellTool::builder().workdir("/srv/project").build()?;
/// let agent = Agent::builder(client)
///     .instructions("You can run shell commands.")
///     .tool(shell.as_function())
///     .build();
/// // Each `run_shell` call now comes back as an approval request first.
/// # let _ = agent;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct LocalShellTool {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for LocalShellTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalShellTool")
            .field("mode", &self.inner.mode)
            .field("argv", &self.argv())
            .field("workdir", &self.inner.workdir)
            .field("approval_mode", &self.inner.approval_mode)
            .finish_non_exhaustive()
    }
}

impl LocalShellTool {
    /// A builder with upstream's defaults: persistent mode, approval
    /// required, 30 second timeout, 64 KiB output limit, empty policy.
    pub fn builder() -> LocalShellToolBuilder {
        LocalShellToolBuilder::default()
    }

    /// The tool's mode.
    pub fn mode(&self) -> ShellMode {
        self.inner.mode
    }

    /// The approval mode [`as_function`](Self::as_function) applies.
    pub fn approval_mode(&self) -> ApprovalMode {
        self.inner.approval_mode
    }

    /// The configured per-command timeout.
    pub fn timeout(&self) -> Option<Duration> {
        self.inner.timeout
    }

    /// The configured output limit per stream, in bytes.
    pub fn max_output_bytes(&self) -> usize {
        self.inner.max_output_bytes
    }

    /// The environment commands run with, or `None` when it is inherited
    /// unchanged.
    pub fn environment(&self) -> Option<&HashMap<String, String>> {
        self.inner.env.as_ref()
    }

    /// The shell argv: the long-lived shell in persistent mode, or the prefix
    /// the command is appended to in stateless mode.
    pub fn argv(&self) -> &[String] {
        match self.inner.mode {
            ShellMode::Persistent => &self.inner.interactive_argv,
            ShellMode::Stateless => &self.inner.stateless_argv,
        }
    }

    /// The function to hand an agent, named `run_shell` with a
    /// mode-appropriate description and this tool's approval mode.
    pub fn as_function(&self) -> ToolDefinition {
        self.as_function_with(DEFAULT_TOOL_NAME, None)
    }

    /// [`as_function`](Self::as_function) with a custom name and description.
    ///
    /// **Avoid name collisions.** Approval decisions can be matched by tool
    /// name, so naming this tool after another tool that is auto-approved may
    /// let shell calls through without a human. Pick a name no other tool
    /// uses.
    pub fn as_function_with(&self, name: &str, description: Option<&str>) -> ToolDefinition {
        let default = match self.inner.mode {
            ShellMode::Persistent => PERSISTENT_DESCRIPTION,
            ShellMode::Stateless => STATELESS_DESCRIPTION,
        };
        shell_function(
            self.clone(),
            name,
            description.unwrap_or(default),
            self.inner.approval_mode,
        )
    }

    /// Prefix a `cd` back into the working directory, when confinement is on.
    fn reanchor(&self, command: &str) -> String {
        let Some(dir) = self
            .inner
            .workdir
            .as_ref()
            .filter(|_| self.inner.confine_workdir)
        else {
            return command.to_string();
        };
        let dir = dir.to_string_lossy();
        if is_powershell(&self.inner.interactive_argv) {
            format!(
                "Set-Location -LiteralPath {}\n{command}",
                quote_powershell(&dir)
            )
        } else {
            format!("cd -- {}\n{command}", quote_posix(&dir))
        }
    }
}

#[async_trait]
impl ShellExecutor for LocalShellTool {
    /// Spawn the persistent shell now. A no-op in stateless mode.
    async fn start(&self) -> Result<(), ShellError> {
        match &self.inner.session {
            Some(session) => session.start().await,
            None => Ok(()),
        }
    }

    /// Stop the persistent shell. A later `run` starts a new one.
    async fn close(&self) -> Result<(), ShellError> {
        if let Some(session) = &self.inner.session {
            session.close().await;
        }
        Ok(())
    }

    /// Apply the policy and audit hook, then run the command. Approval is
    /// not applied here: the agent's function-invocation loop does that for
    /// the function from [`as_function`](LocalShellTool::as_function).
    async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        admit(
            &self.inner.policy,
            self.inner.on_command.as_ref(),
            command,
            self.inner.workdir.clone(),
        )?;
        let timeout = timeout.or(self.inner.timeout);
        match &self.inner.session {
            Some(session) => session.run(&self.reanchor(command), timeout).await,
            None => {
                run_stateless(
                    &self.inner.stateless_argv,
                    command,
                    self.inner.workdir.as_ref(),
                    self.inner.env.as_ref(),
                    timeout,
                    self.inner.max_output_bytes,
                )
                .await
            }
        }
    }

    fn shell_family(&self) -> Option<ShellFamily> {
        Some(if is_powershell(&self.inner.interactive_argv) {
            ShellFamily::PowerShell
        } else {
            ShellFamily::Posix
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_posix_blocks_dollar_expansion() {
        assert_eq!(quote_posix("$(rm -rf /)"), "'$(rm -rf /)'");
    }

    #[test]
    fn quote_posix_escapes_embedded_single_quote() {
        assert_eq!(quote_posix("it's fine"), "'it'\\''s fine'");
    }

    #[test]
    fn quote_powershell_blocks_dollar_expansion() {
        assert_eq!(quote_powershell("$malicious"), "'$malicious'");
    }

    #[test]
    fn quote_powershell_doubles_embedded_single_quote() {
        assert_eq!(quote_powershell("a'b"), "'a''b'");
    }

    #[test]
    fn reanchors_powershell_paths() {
        let tool = LocalShellTool::builder()
            .shell("pwsh")
            .workdir("C:\\repo")
            .build()
            .unwrap();
        assert!(tool
            .reanchor("Get-ChildItem")
            .starts_with("Set-Location -LiteralPath 'C:\\repo'\nGet-ChildItem"));
    }

    #[test]
    fn reanchors_posix_paths_with_quoting() {
        let tool = LocalShellTool::builder()
            .shell("/bin/sh")
            .workdir("/srv/it's here")
            .build()
            .unwrap();
        assert_eq!(tool.reanchor("ls"), "cd -- '/srv/it'\\''s here'\nls");
        let unconfined = LocalShellTool::builder()
            .shell("/bin/sh")
            .workdir("/srv")
            .confine_workdir(false)
            .build()
            .unwrap();
        assert_eq!(unconfined.reanchor("ls"), "ls");
    }
}
