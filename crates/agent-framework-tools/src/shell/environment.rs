//! [`ShellEnvironmentProvider`]: tell the model what shell it is driving.
//!
//! Probes the shell once (family and version, working directory, which CLI
//! tools are installed) and adds an instructions block to every run, so the
//! model writes PowerShell in a PowerShell session and bash in a bash one.
//! The probe goes through any [`ShellExecutor`], so it works for both the
//! local and the Docker tool.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::memory::{ContextProvider, SessionContext};
use async_trait::async_trait;
use tokio::sync::Mutex;

use super::executor::ShellExecutor;
use super::types::{ShellError, ShellResult};

/// The shell families the provider knows how to describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShellFamily {
    /// bash, sh, zsh and other POSIX shells.
    Posix,
    /// Windows PowerShell or PowerShell 7+.
    PowerShell,
}

impl ShellFamily {
    /// The family this platform's default shell belongs to.
    pub fn detect() -> Self {
        if cfg!(windows) {
            ShellFamily::PowerShell
        } else {
            ShellFamily::Posix
        }
    }
}

/// What the probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShellEnvironmentSnapshot {
    /// Detected or configured shell family.
    pub family: ShellFamily,
    /// A short description of the host OS (`linux x86_64`).
    pub os_description: String,
    /// The shell's reported version, or `None` when it did not report one.
    pub shell_version: Option<String>,
    /// The working directory the shell reported, or `""` when probing failed.
    pub working_directory: String,
    /// Each probed CLI tool and the first line of its `--version` output;
    /// `None` when it is missing or did not answer in time.
    pub tool_versions: BTreeMap<String, Option<String>>,
}

type Formatter = Arc<dyn Fn(&ShellEnvironmentSnapshot) -> String + Send + Sync>;

/// Configuration for [`ShellEnvironmentProvider`].
#[derive(Clone)]
pub struct ShellEnvironmentProviderOptions {
    /// CLI tools whose `--version` is probed. Default: `git`, `node`,
    /// `python`, `docker`. Names that are not plain identifiers
    /// (`[A-Za-z0-9._-]+`) are skipped, because the name is spliced into a
    /// shell command.
    pub probe_tools: Vec<String>,
    /// Override the detected family, e.g. bash on Windows or pwsh on Linux.
    pub override_family: Option<ShellFamily>,
    /// Timeout for each probe command. Default 5 seconds.
    pub probe_timeout: Duration,
    /// Renders the instructions block. Default
    /// [`default_instructions_formatter`].
    pub instructions_formatter: Option<Formatter>,
}

impl Default for ShellEnvironmentProviderOptions {
    fn default() -> Self {
        Self {
            probe_tools: ["git", "node", "python", "docker"]
                .into_iter()
                .map(String::from)
                .collect(),
            override_family: None,
            probe_timeout: Duration::from_secs(5),
            instructions_formatter: None,
        }
    }
}

impl fmt::Debug for ShellEnvironmentProviderOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellEnvironmentProviderOptions")
            .field("probe_tools", &self.probe_tools)
            .field("override_family", &self.override_family)
            .field("probe_timeout", &self.probe_timeout)
            .field(
                "instructions_formatter",
                &self.instructions_formatter.is_some(),
            )
            .finish()
    }
}

impl ShellEnvironmentProviderOptions {
    /// Builder: set the custom instructions formatter.
    pub fn with_instructions_formatter<F>(mut self, formatter: F) -> Self
    where
        F: Fn(&ShellEnvironmentSnapshot) -> String + Send + Sync + 'static,
    {
        self.instructions_formatter = Some(Arc::new(formatter));
        self
    }
}

/// A [`ContextProvider`] that adds a shell-environment block to the
/// instructions of every run. It registers no tools.
///
/// The probe runs once, on the first run, and its snapshot is cached; call
/// [`refresh`](Self::refresh) after something it depends on changed. A
/// probe command that fails with an expected error (policy rejection, spawn
/// failure, timeout) is recorded as a missing value. [`ShellError::Other`]
/// propagates, so bugs in a custom executor are not swallowed, and a failed
/// probe leaves nothing cached so the next run tries again.
pub struct ShellEnvironmentProvider {
    executor: Arc<dyn ShellExecutor>,
    options: ShellEnvironmentProviderOptions,
    snapshot: RwLock<Option<ShellEnvironmentSnapshot>>,
    probe_lock: Mutex<()>,
}

impl fmt::Debug for ShellEnvironmentProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShellEnvironmentProvider")
            .field("options", &self.options)
            .field("snapshot", &self.current_snapshot())
            .finish_non_exhaustive()
    }
}

fn is_plain_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

fn first_non_empty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .map(str::to_string)
}

impl ShellEnvironmentProvider {
    /// A provider probing through `executor` with `options` (or the
    /// defaults).
    pub fn new(
        executor: Arc<dyn ShellExecutor>,
        options: Option<ShellEnvironmentProviderOptions>,
    ) -> Self {
        Self {
            executor,
            options: options.unwrap_or_default(),
            snapshot: RwLock::new(None),
            probe_lock: Mutex::new(()),
        }
    }

    /// The cached snapshot, or `None` before the first probe.
    pub fn current_snapshot(&self) -> Option<ShellEnvironmentSnapshot> {
        self.snapshot
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// Probe again and replace the cached snapshot.
    pub async fn refresh(&self) -> std::result::Result<ShellEnvironmentSnapshot, ShellError> {
        let _guard = self.probe_lock.lock().await;
        let snapshot = self.probe().await?;
        *self.snapshot.write().unwrap_or_else(|e| e.into_inner()) = Some(snapshot.clone());
        Ok(snapshot)
    }

    async fn get_or_probe(&self) -> std::result::Result<ShellEnvironmentSnapshot, ShellError> {
        if let Some(snapshot) = self.current_snapshot() {
            return Ok(snapshot);
        }
        // Concurrent first callers share one probe.
        let _guard = self.probe_lock.lock().await;
        if let Some(snapshot) = self.current_snapshot() {
            return Ok(snapshot);
        }
        let snapshot = self.probe().await?;
        *self.snapshot.write().unwrap_or_else(|e| e.into_inner()) = Some(snapshot.clone());
        Ok(snapshot)
    }

    async fn probe(&self) -> std::result::Result<ShellEnvironmentSnapshot, ShellError> {
        let family = self
            .options
            .override_family
            .unwrap_or_else(ShellFamily::detect);
        self.executor.start().await?;
        let (shell_version, working_directory) = self.probe_shell_and_cwd(family).await?;

        let mut tool_versions = BTreeMap::new();
        let mut seen: Vec<String> = Vec::new();
        for tool in &self.options.probe_tools {
            let lower = tool.to_ascii_lowercase();
            if seen.contains(&lower) {
                continue;
            }
            seen.push(lower);
            let version = self.probe_tool_version(tool).await?;
            tool_versions.insert(tool.clone(), version);
        }

        Ok(ShellEnvironmentSnapshot {
            family,
            os_description: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            shell_version,
            working_directory,
            tool_versions,
        })
    }

    async fn probe_shell_and_cwd(
        &self,
        family: ShellFamily,
    ) -> std::result::Result<(Option<String>, String), ShellError> {
        let command = match family {
            ShellFamily::PowerShell => {
                "Write-Output (\"VERSION=\" + $PSVersionTable.PSVersion.ToString()); Write-Output (\"CWD=\" + (Get-Location).Path)"
            }
            ShellFamily::Posix => {
                "echo \"VERSION=${BASH_VERSION:-${ZSH_VERSION:-unknown}}\"; echo \"CWD=$(pwd)\""
            }
        };
        let Some(result) = self.run_probe(command).await? else {
            return Ok((None, String::new()));
        };
        let mut version = None;
        let mut cwd = String::new();
        for line in result.stdout.lines().map(str::trim) {
            if let Some(value) = line.strip_prefix("VERSION=") {
                let value = value.trim();
                version = (!value.is_empty() && value != "unknown").then(|| value.to_string());
            } else if let Some(value) = line.strip_prefix("CWD=") {
                cwd = value.trim().to_string();
            }
        }
        Ok((version, cwd))
    }

    async fn probe_tool_version(
        &self,
        tool: &str,
    ) -> std::result::Result<Option<String>, ShellError> {
        // The name is spliced into a command: refuse anything that could
        // carry shell syntax.
        if !is_plain_tool_name(tool) {
            return Ok(None);
        }
        let Some(result) = self.run_probe(&format!("{tool} --version")).await? else {
            return Ok(None);
        };
        if result.exit_code != 0 {
            return Ok(None);
        }
        // Some CLIs print their version on stderr.
        Ok(first_non_empty_line(&result.stdout).or_else(|| first_non_empty_line(&result.stderr)))
    }

    async fn run_probe(
        &self,
        command: &str,
    ) -> std::result::Result<Option<ShellResult>, ShellError> {
        match self
            .executor
            .run(command, Some(self.options.probe_timeout))
            .await
        {
            Ok(result) if result.timed_out => Ok(None),
            Ok(result) => Ok(Some(result)),
            Err(ShellError::Other(err)) => Err(ShellError::Other(err)),
            Err(_) => Ok(None),
        }
    }
}

#[async_trait]
impl ContextProvider for ShellEnvironmentProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let snapshot = self
            .get_or_probe()
            .await
            .map_err(|e| Error::AgentExecution(format!("shell environment probe failed: {e}")))?;
        let block = match &self.options.instructions_formatter {
            Some(formatter) => formatter(&snapshot),
            None => default_instructions_formatter(&snapshot),
        };
        ctx.add_instructions(block);
        Ok(())
    }
}

/// The default instructions block. Public so a custom formatter can wrap it.
pub fn default_instructions_formatter(snapshot: &ShellEnvironmentSnapshot) -> String {
    let mut lines: Vec<String> = vec!["## Shell environment".into()];
    let version = snapshot
        .shell_version
        .as_deref()
        .map(|v| format!(" {v}"))
        .unwrap_or_default();
    match snapshot.family {
        ShellFamily::PowerShell => {
            lines.push(format!(
                "You are operating a PowerShell{version} session on {}.",
                snapshot.os_description
            ));
            lines.push("Use PowerShell idioms, NOT bash:".into());
            lines.push(
                "- Set environment variables with `$env:NAME = 'value'` (NOT `NAME=value`).".into(),
            );
            lines.push(
                "- Change directory with `Set-Location` or `cd`. Paths use `\\` separators.".into(),
            );
            lines.push("- Reference environment variables as `$env:NAME` (NOT `$NAME`).".into());
            lines.push(
                "- The system temp directory is `[System.IO.Path]::GetTempPath()` (NOT `/tmp`)."
                    .into(),
            );
            lines.push("- Pipe to `Out-Null` to suppress output (NOT `> /dev/null`).".into());
        }
        ShellFamily::Posix => {
            lines.push(format!(
                "You are operating a POSIX shell{version} session on {}.",
                snapshot.os_description
            ));
            lines.push("Use POSIX shell idioms (bash/sh).".into());
            lines.push(
                "- Set environment variables for the next command with `export NAME=value`.".into(),
            );
            lines.push("- Reference environment variables as `$NAME` or `${NAME}`.".into());
            lines.push("- Paths use `/` separators.".into());
        }
    }
    if !snapshot.working_directory.is_empty() {
        lines.push(format!("Working directory: {}", snapshot.working_directory));
    }
    let installed: Vec<String> = snapshot
        .tool_versions
        .iter()
        .filter_map(|(name, v)| v.as_ref().map(|v| format!("{name} ({v})")))
        .collect();
    let missing: Vec<&str> = snapshot
        .tool_versions
        .iter()
        .filter(|(_, v)| v.is_none())
        .map(|(name, _)| name.as_str())
        .collect();
    if !installed.is_empty() {
        lines.push(format!("Available CLIs: {}", installed.join(", ")));
    }
    if !missing.is_empty() {
        lines.push(format!("Not installed: {}", missing.join(", ")));
    }
    lines.join("\n")
}
