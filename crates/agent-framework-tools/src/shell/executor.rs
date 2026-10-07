//! The [`ShellExecutor`] trait: the swappable backend that runs commands.

use std::time::Duration;

use async_trait::async_trait;

use super::types::{ShellError, ShellResult};

/// A backend that runs shell commands.
///
/// [`LocalShellTool`](super::LocalShellTool) runs them on the host with no
/// process-level isolation; approval-in-the-loop is its boundary.
/// [`DockerShellTool`](super::DockerShellTool) runs them in a container, which
/// is the intended boundary when the runtime is trusted and the default
/// isolation flags are kept. Implement this trait to plug in another backend
/// (a microVM, a remote host, a WASI runtime).
///
/// # Single-session ownership
///
/// An executor, and the tool wrapping it, serves one conversation, i.e. one
/// user. In persistent mode it owns a long-lived shell whose state (working
/// directory, exported variables, background jobs, files) is visible to every
/// later command, and one pipe serialises every call. Build one per session,
/// close it when the session ends, and do not share a persistent-mode
/// instance across users, tenants or concurrent conversations. If a shared
/// instance is unavoidable, use stateless mode.
#[async_trait]
pub trait ShellExecutor: Send + Sync {
    /// Initialise the backend eagerly. A no-op when already started.
    async fn start(&self) -> Result<(), ShellError>;

    /// Release every backend resource. Idempotent.
    async fn close(&self) -> Result<(), ShellError>;

    /// Run `command` and return its result.
    ///
    /// `timeout` overrides the executor's configured default for this call;
    /// `None` uses that default. Implementations must enforce the timeout
    /// themselves and must not leak processes when it fires or when the
    /// returned future is dropped.
    async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError>;
}
