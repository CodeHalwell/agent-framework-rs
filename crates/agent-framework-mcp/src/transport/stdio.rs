//! Stdio transport: spawns an MCP server as a child process and speaks
//! newline-delimited JSON-RPC over its stdin/stdout.
//!
//! Note: MCP's stdio framing is one JSON object per line — unlike LSP, there
//! are no `Content-Length` headers.

use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Stdio;
use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;

use agent_framework_core::error::{Error, Result};

use crate::protocol::{self, IdGenerator, IncomingMessage, RpcError};
use crate::transport::McpTransport;

type PendingMap = StdMutex<HashMap<i64, oneshot::Sender<std::result::Result<Value, RpcError>>>>;

/// An MCP transport backed by a child process's stdin/stdout.
///
/// The child is spawned with `kill_on_drop`, and is additionally killed
/// explicitly by [`McpStdioTransport::close`] and by `Drop`, so the process
/// never outlives this transport.
pub struct McpStdioTransport {
    inner: std::sync::Arc<StdioInner>,
}

struct StdioInner {
    stdin: AsyncMutex<ChildStdin>,
    child: StdMutex<Child>,
    pending: PendingMap,
    next_id: IdGenerator,
    reader_task: StdMutex<Option<JoinHandle<()>>>,
    stderr_task: StdMutex<Option<JoinHandle<()>>>,
}

impl McpStdioTransport {
    /// Spawn `command` with `args`, wiring up stdio pipes and background
    /// tasks that route responses back to their callers and drain
    /// notifications/stderr to `tracing`.
    ///
    /// `env`, if provided, adds/overrides variables on top of the inherited
    /// parent environment. `cwd`, if provided, sets the child's working
    /// directory.
    pub async fn spawn(
        command: impl AsRef<OsStr>,
        args: &[String],
        env: Option<&HashMap<String, String>>,
        cwd: Option<&Path>,
    ) -> Result<Self> {
        let command_ref = command.as_ref();
        let mut cmd = Command::new(command_ref);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(env) = env {
            cmd.envs(env);
        }
        if let Some(cwd) = cwd {
            cmd.current_dir(cwd);
        }

        let mut child = cmd.spawn().map_err(|e| {
            Error::service(format!(
                "failed to start MCP server '{}': {e}",
                command_ref.to_string_lossy()
            ))
        })?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stdin pipe"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stdout pipe"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| Error::service("MCP child process has no stderr pipe"))?;

        let pending: PendingMap = StdMutex::new(HashMap::new());
        let inner = std::sync::Arc::new(StdioInner {
            stdin: AsyncMutex::new(stdin),
            child: StdMutex::new(child),
            pending,
            next_id: IdGenerator::new(),
            reader_task: StdMutex::new(None),
            stderr_task: StdMutex::new(None),
        });

        let reader_task = spawn_reader(stdout, inner.clone());
        let stderr_task = spawn_stderr_drain(stderr);
        *inner.reader_task.lock().unwrap() = Some(reader_task);
        *inner.stderr_task.lock().unwrap() = Some(stderr_task);

        Ok(Self { inner })
    }

    async fn write_line(&self, message: &Value) -> Result<()> {
        let mut line = serde_json::to_string(message)
            .map_err(|e| Error::service(format!("failed to encode MCP message: {e}")))?;
        line.push('\n');
        let mut stdin = self.inner.stdin.lock().await;
        stdin
            .write_all(line.as_bytes())
            .await
            .map_err(|e| Error::service(format!("failed to write to MCP server stdin: {e}")))?;
        stdin
            .flush()
            .await
            .map_err(|e| Error::service(format!("failed to flush MCP server stdin: {e}")))?;
        Ok(())
    }
}

#[async_trait]
impl McpTransport for McpStdioTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.next();
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().unwrap().insert(id, tx);

        let request = protocol::build_request(id, method, params);
        if let Err(e) = self.write_line(&request).await {
            self.inner.pending.lock().unwrap().remove(&id);
            return Err(e);
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(rpc_err)) => Err(Error::service(rpc_err.to_string())),
            Err(_) => Err(Error::service(
                "MCP server closed the connection before responding",
            )),
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let notification = protocol::build_notification(method, params);
        self.write_line(&notification).await
    }

    async fn close(&self) -> Result<()> {
        if let Ok(mut child) = self.inner.child.lock() {
            let _ = child.start_kill();
        }
        Ok(())
    }
}

impl Drop for StdioInner {
    fn drop(&mut self) {
        if let Ok(mut child) = self.child.lock() {
            let _ = child.start_kill();
        }
        if let Ok(mut task) = self.reader_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        if let Ok(mut task) = self.stderr_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        // Fail any requests still waiting on a response so callers don't hang.
        if let Ok(mut pending) = self.pending.lock() {
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err(RpcError {
                    code: -1,
                    message: "MCP stdio transport dropped".to_string(),
                    data: None,
                }));
            }
        }
    }
}

/// Spawn the background task that reads newline-delimited JSON-RPC messages
/// from the server's stdout and routes them: responses go to their waiting
/// caller by id, notifications and server-initiated requests are logged and
/// otherwise ignored (no sampling/roots support).
fn spawn_reader(
    stdout: tokio::process::ChildStdout,
    inner: std::sync::Arc<StdioInner>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stdout).lines();
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => route_incoming(&inner, value),
                        Err(e) => {
                            tracing::warn!(error = %e, line = %trimmed, "MCP: non-JSON line from server stdout");
                        }
                    }
                }
                Ok(None) => {
                    tracing::debug!("MCP server stdout closed");
                    break;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "MCP: error reading server stdout");
                    break;
                }
            }
        }
        // The server is gone; unblock anyone still waiting on a response.
        let mut pending = inner.pending.lock().unwrap();
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(RpcError {
                code: -1,
                message: "MCP server connection closed".to_string(),
                data: None,
            }));
        }
    })
}

fn route_incoming(inner: &std::sync::Arc<StdioInner>, value: Value) {
    match protocol::parse_incoming(value) {
        IncomingMessage::Response { id, result } => {
            if let Some(tx) = inner.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(result);
            } else {
                tracing::debug!(id, "MCP: response for unknown/already-resolved request id");
            }
        }
        IncomingMessage::Notification { method, params } => {
            tracing::debug!(method = %method, params = %params, "MCP server notification");
        }
        IncomingMessage::ServerRequest { id, method, params } => {
            tracing::warn!(
                id = %id,
                method = %method,
                params = %params,
                "MCP server sent a server-initiated request; sampling/roots are not supported, ignoring"
            );
        }
        IncomingMessage::Malformed(v) => {
            tracing::warn!(raw = %v, "MCP: unrecognized JSON-RPC message shape");
        }
    }
}

/// Spawn the background task that drains the server's stderr to `tracing`.
fn spawn_stderr_drain(stderr: tokio::process::ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if !line.trim().is_empty() {
                tracing::debug!(target: "mcp::server_stderr", "{line}");
            }
        }
    })
}
