//! The CLI subprocess and its stream-json control protocol.
//!
//! One process per run: the CLI is started with `--input-format
//! stream-json --output-format stream-json`, initialized with a
//! `control_request` (`initialize`), sent one `user` message, and read until
//! its `result`. While it runs it may send its own control requests
//! (`can_use_tool`, `hook_callback`, `mcp_message`), which are answered on
//! stdin. Closing stdin after the result ends the process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use serde_json::{json, Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, oneshot, Mutex};

use crate::mcp::ToolServer;
use crate::options::{
    ClaudeAgentOptions, HookCallback, PermissionResult, ThinkingConfig, ToolPermissionRequest,
};

/// How long `initialize` (and other host-sent control requests) may take.
const CONTROL_TIMEOUT: Duration = Duration::from_secs(60);
/// How long the CLI may take to exit once stdin is closed.
const EXIT_TIMEOUT: Duration = Duration::from_secs(5);
/// How much of the CLI's stderr is kept for error messages.
const STDERR_TAIL: usize = 4096;

/// Everything a run's command line depends on besides the options.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct LaunchSpec {
    /// The CLI session to resume (`--resume=<id>`).
    pub resume: Option<String>,
    /// Built-in tool names (`--tools`).
    pub builtin_tools: Vec<String>,
    /// The in-process tool server's name and tools' qualified names.
    pub tool_server: Option<(String, Vec<String>)>,
}

/// The CLI arguments (after the executable) for `options` and `spec`,
/// following the Python SDK's `SubprocessCLITransport._build_command`.
pub(crate) fn build_args(options: &ClaudeAgentOptions, spec: &LaunchSpec) -> Vec<String> {
    let mut cmd: Vec<String> = vec![
        "--output-format".into(),
        "stream-json".into(),
        "--verbose".into(),
    ];
    let mut push = |a: &str, b: String| {
        cmd.push(a.to_string());
        cmd.push(b);
    };
    push(
        "--system-prompt",
        options.system_prompt.clone().unwrap_or_default(),
    );
    if !spec.builtin_tools.is_empty() {
        push("--tools", spec.builtin_tools.join(","));
    }
    let mut allowed = options.allowed_tools.clone();
    if let Some((_, names)) = &spec.tool_server {
        allowed.extend(names.iter().cloned());
    }
    if !allowed.is_empty() {
        push("--allowedTools", allowed.join(","));
    }
    if let Some(turns) = options.max_turns.filter(|t| *t > 0) {
        push("--max-turns", turns.to_string());
    }
    if let Some(budget) = options.max_budget_usd {
        push("--max-budget-usd", budget.to_string());
    }
    if !options.disallowed_tools.is_empty() {
        push("--disallowedTools", options.disallowed_tools.join(","));
    }
    if let Some(budget) = options.task_budget {
        push("--task-budget", budget.to_string());
    }
    if let Some(model) = &options.model {
        push("--model", model.clone());
    }
    if let Some(model) = &options.fallback_model {
        push("--fallback-model", model.clone());
    }
    if !options.betas.is_empty() {
        push("--betas", options.betas.join(","));
    }
    if options.can_use_tool.is_some() {
        push("--permission-prompt-tool", "stdio".into());
    }
    if let Some(mode) = &options.permission_mode {
        push("--permission-mode", mode.clone());
    }
    if options.continue_conversation {
        cmd.push("--continue".into());
    }
    if let Some(resume) = &spec.resume {
        cmd.push(format!("--resume={resume}"));
    }
    if let Some(id) = &options.session_id {
        cmd.push(format!("--session-id={id}"));
    }
    if let Some(settings) = settings_value(options) {
        cmd.push("--settings".into());
        cmd.push(settings);
    }
    for dir in &options.add_dirs {
        cmd.push("--add-dir".into());
        cmd.push(dir.display().to_string());
    }
    let mut servers = options.mcp_servers.clone();
    if let Some((name, _)) = &spec.tool_server {
        servers.insert(name.clone(), json!({"type": "sdk", "name": name}));
    }
    if !servers.is_empty() {
        cmd.push("--mcp-config".into());
        cmd.push(json!({ "mcpServers": servers }).to_string());
    }
    // Always on: text and thinking stream from partial messages.
    cmd.push("--include-partial-messages".into());
    if options.include_hook_events {
        cmd.push("--include-hook-events".into());
    }
    if options.strict_mcp_config {
        cmd.push("--strict-mcp-config".into());
    }
    if options.fork_session {
        cmd.push("--fork-session".into());
    }
    if let Some(sources) = &options.setting_sources {
        cmd.push(format!("--setting-sources={}", sources.join(",")));
    }
    for dir in &options.plugin_dirs {
        cmd.push("--plugin-dir".into());
        cmd.push(dir.display().to_string());
    }
    for (flag, value) in &options.extra_args {
        match value {
            None => cmd.push(format!("--{flag}")),
            Some(v) if v.starts_with('-') => cmd.push(format!("--{flag}={v}")),
            Some(v) => {
                cmd.push(format!("--{flag}"));
                cmd.push(v.clone());
            }
        }
    }
    match &options.thinking {
        Some(ThinkingConfig::Adaptive { display }) => {
            cmd.extend(["--thinking".into(), "adaptive".into()]);
            if let Some(d) = display {
                cmd.extend(["--thinking-display".into(), d.clone()]);
            }
        }
        Some(ThinkingConfig::Enabled {
            budget_tokens,
            display,
        }) => {
            cmd.extend(["--max-thinking-tokens".into(), budget_tokens.to_string()]);
            if let Some(d) = display {
                cmd.extend(["--thinking-display".into(), d.clone()]);
            }
        }
        Some(ThinkingConfig::Disabled) => cmd.extend(["--thinking".into(), "disabled".into()]),
        None => {}
    }
    if let Some(effort) = &options.effort {
        cmd.extend(["--effort".into(), effort.clone()]);
    }
    if let Some(schema) = &options.output_schema {
        cmd.extend(["--json-schema".into(), schema.to_string()]);
    }
    cmd.extend(["--input-format".into(), "stream-json".into()]);
    cmd
}

/// `--settings`: the settings as given, or (with a sandbox) a JSON object
/// merging the settings (JSON or a file) with `sandbox`.
fn settings_value(options: &ClaudeAgentOptions) -> Option<String> {
    match (&options.settings, &options.sandbox) {
        (None, None) => None,
        (Some(settings), None) => Some(settings.clone()),
        (settings, Some(sandbox)) => {
            let mut obj: Map<String, Value> = match settings.as_deref().map(str::trim) {
                Some(s) if s.starts_with('{') && s.ends_with('}') => {
                    serde_json::from_str(s).unwrap_or_default()
                }
                Some(path) => std::fs::read_to_string(path)
                    .ok()
                    .and_then(|t| serde_json::from_str(&t).ok())
                    .unwrap_or_default(),
                None => Map::new(),
            };
            obj.insert("sandbox".into(), sandbox.clone());
            Some(Value::Object(obj).to_string())
        }
    }
}

/// Find the CLI: `claude` on `PATH`, then the usual install locations.
pub(crate) fn find_cli() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path) {
            let candidate = dir.join("claude");
            if candidate.is_file() {
                return Ok(candidate);
            }
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let mut locations: Vec<PathBuf> = Vec::new();
    if let Some(home) = &home {
        locations.push(home.join(".npm-global/bin/claude"));
    }
    locations.push(PathBuf::from("/usr/local/bin/claude"));
    if let Some(home) = &home {
        for rel in [
            ".local/bin/claude",
            "node_modules/.bin/claude",
            ".yarn/bin/claude",
            ".claude/local/claude",
        ] {
            locations.push(home.join(rel));
        }
    }
    locations.into_iter().find(|p| p.is_file()).ok_or_else(|| {
        Error::AgentInitialization(
            "Claude Code CLI not found: install it (npm install -g @anthropic-ai/claude-code) \
                 or set cli_path / CLAUDE_AGENT_CLI_PATH"
                .into(),
        )
    })
}

/// What the host answers the CLI's control requests with.
#[derive(Clone, Default)]
pub(crate) struct Handlers {
    pub can_use_tool: Option<Arc<dyn crate::options::ToolPermissionHandler>>,
    pub hooks: HashMap<String, HookCallback>,
    pub tool_server: Option<Arc<ToolServer>>,
}

type Pending = Arc<StdMutex<HashMap<String, oneshot::Sender<Result<Value>>>>>;
type Writer = Arc<Mutex<Option<ChildStdin>>>;

/// A running CLI process.
pub(crate) struct Connection {
    child: Child,
    writer: Writer,
    pending: Pending,
    counter: AtomicU64,
    messages: mpsc::UnboundedReceiver<Value>,
    stderr: Arc<StdMutex<String>>,
}

async fn write_line(writer: &Writer, value: &Value) -> Result<()> {
    let mut guard = writer.lock().await;
    let stdin = guard
        .as_mut()
        .ok_or_else(|| Error::AgentExecution("the Claude CLI's input is closed".into()))?;
    let mut line = value.to_string();
    line.push('\n');
    stdin
        .write_all(line.as_bytes())
        .await
        .map_err(|e| Error::AgentExecution(format!("writing to the Claude CLI failed: {e}")))?;
    stdin
        .flush()
        .await
        .map_err(|e| Error::AgentExecution(format!("writing to the Claude CLI failed: {e}")))
}

impl Connection {
    /// Start the CLI.
    pub async fn spawn(
        cli: &Path,
        args: &[String],
        options: &ClaudeAgentOptions,
        handlers: Handlers,
    ) -> Result<Self> {
        let mut command = Command::new(cli);
        command
            .args(args)
            .env_remove("CLAUDECODE")
            .env("CLAUDE_CODE_ENTRYPOINT", "sdk-rs")
            .envs(&options.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if options.enable_file_checkpointing {
            command.env("CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING", "true");
        }
        if let Some(cwd) = &options.cwd {
            command.current_dir(cwd);
        }
        let mut child = command.spawn().map_err(|e| {
            Error::AgentInitialization(format!(
                "failed to start the Claude CLI at {}: {e}",
                cli.display()
            ))
        })?;
        let stdin = child.stdin.take();
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::AgentInitialization("the Claude CLI has no stdout".into()))?;
        let stderr_pipe = child.stderr.take();
        let writer: Writer = Arc::new(Mutex::new(stdin));
        let pending: Pending = Arc::new(StdMutex::new(HashMap::new()));
        let stderr = Arc::new(StdMutex::new(String::new()));
        if let Some(mut pipe) = stderr_pipe {
            let tail = stderr.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 1024];
                while let Ok(n) = pipe.read(&mut buf).await {
                    if n == 0 {
                        break;
                    }
                    let chunk = String::from_utf8_lossy(&buf[..n]);
                    tracing::debug!(target: "agent_framework::claude", stderr = %chunk);
                    let mut t = tail.lock().unwrap_or_else(|p| p.into_inner());
                    t.push_str(&chunk);
                    if t.len() > STDERR_TAIL {
                        let mut cut = t.len() - STDERR_TAIL;
                        while !t.is_char_boundary(cut) {
                            cut += 1;
                        }
                        t.drain(..cut);
                    }
                }
            });
        }
        let (tx, messages) = mpsc::unbounded_channel();
        let reader_pending = pending.clone();
        let reader_writer = writer.clone();
        let handlers = Arc::new(handlers);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let Ok(message) = serde_json::from_str::<Value>(line) else {
                    tracing::debug!(target: "agent_framework::claude", "skipping non-JSON CLI output");
                    continue;
                };
                match message.get("type").and_then(Value::as_str) {
                    Some("control_response") => {
                        let response = message.get("response").cloned().unwrap_or(Value::Null);
                        let id = response
                            .get("request_id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let sender = reader_pending
                            .lock()
                            .unwrap_or_else(|p| p.into_inner())
                            .remove(&id);
                        if let Some(sender) = sender {
                            let result = if response.get("subtype").and_then(Value::as_str)
                                == Some("error")
                            {
                                Err(Error::AgentExecution(format!(
                                    "Claude CLI control request failed: {}",
                                    response
                                        .get("error")
                                        .and_then(Value::as_str)
                                        .unwrap_or("unknown error")
                                )))
                            } else {
                                Ok(response.get("response").cloned().unwrap_or(json!({})))
                            };
                            let _ = sender.send(result);
                        }
                    }
                    Some("control_request") => {
                        let writer = reader_writer.clone();
                        let handlers = handlers.clone();
                        tokio::spawn(async move {
                            let reply = answer_control_request(&handlers, &message).await;
                            let _ = write_line(&writer, &reply).await;
                        });
                    }
                    Some("control_cancel_request") => {}
                    _ => {
                        if tx.send(message).is_err() {
                            break;
                        }
                    }
                }
            }
            // Fail anything still waiting for a control response.
            let waiting: Vec<_> = reader_pending
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .drain()
                .collect();
            for (_, sender) in waiting {
                let _ = sender.send(Err(Error::AgentExecution(
                    "the Claude CLI exited before answering a control request".into(),
                )));
            }
        });
        Ok(Self {
            child,
            writer,
            pending,
            counter: AtomicU64::new(0),
            messages,
            stderr,
        })
    }

    /// Send a control request and wait for its response.
    pub async fn control(&self, request: Value) -> Result<Value> {
        let n = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        let id = format!(
            "req_{n}_{}",
            &uuid::Uuid::new_v4().simple().to_string()[..8]
        );
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .insert(id.clone(), tx);
        let subtype = request
            .get("subtype")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        write_line(
            &self.writer,
            &json!({"type": "control_request", "request_id": id, "request": request}),
        )
        .await?;
        match tokio::time::timeout(CONTROL_TIMEOUT, rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(self.exited_error()),
            Err(_) => {
                self.pending
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .remove(&id);
                Err(Error::AgentExecution(format!(
                    "Claude CLI control request timed out: {subtype}"
                )))
            }
        }
    }

    /// Send the user's prompt.
    pub async fn send_prompt(&self, prompt: &str) -> Result<()> {
        write_line(
            &self.writer,
            &json!({
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": null,
                "session_id": "default",
            }),
        )
        .await
    }

    /// The next message from the CLI; `None` once it has exited.
    pub async fn next_message(&mut self) -> Option<Value> {
        self.messages.recv().await
    }

    /// An error for a CLI that went away, with its stderr tail.
    pub fn exited_error(&self) -> Error {
        let tail = self
            .stderr
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .trim()
            .to_string();
        if tail.is_empty() {
            Error::AgentExecution("the Claude CLI exited before producing a result".into())
        } else {
            Error::AgentExecution(format!(
                "the Claude CLI exited before producing a result: {tail}"
            ))
        }
    }

    /// Close stdin and let the CLI exit (killing it if it lingers).
    pub async fn close(mut self) {
        self.writer.lock().await.take();
        if tokio::time::timeout(EXIT_TIMEOUT, self.child.wait())
            .await
            .is_err()
        {
            let _ = self.child.kill().await;
        }
    }
}

/// Answer one control request from the CLI.
async fn answer_control_request(handlers: &Handlers, message: &Value) -> Value {
    let request_id = message.get("request_id").cloned().unwrap_or(Value::Null);
    let request = message.get("request").cloned().unwrap_or(Value::Null);
    let result = handle(handlers, &request).await;
    match result {
        Ok(response) => json!({
            "type": "control_response",
            "response": {"subtype": "success", "request_id": request_id, "response": response},
        }),
        Err(e) => json!({
            "type": "control_response",
            "response": {"subtype": "error", "request_id": request_id, "error": e.to_string()},
        }),
    }
}

async fn handle(handlers: &Handlers, request: &Value) -> Result<Value> {
    let str_field = |k: &str| request.get(k).and_then(Value::as_str).map(str::to_string);
    match request.get("subtype").and_then(Value::as_str) {
        Some("can_use_tool") => {
            let handler = handlers.can_use_tool.as_ref().ok_or_else(|| {
                Error::AgentExecution("canUseTool callback is not provided".into())
            })?;
            let input = request.get("input").cloned().unwrap_or(json!({}));
            let permission = ToolPermissionRequest {
                tool_name: str_field("tool_name").unwrap_or_default(),
                input: input.clone(),
                tool_use_id: str_field("tool_use_id"),
                suggestions: request
                    .get("permission_suggestions")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                blocked_path: str_field("blocked_path"),
                decision_reason: str_field("decision_reason"),
                agent_id: str_field("agent_id"),
                raw: request.clone(),
            };
            Ok(match handler.can_use_tool(permission).await? {
                PermissionResult::Allow {
                    updated_input,
                    updated_permissions,
                } => {
                    let mut out = json!({
                        "behavior": "allow",
                        "updatedInput": updated_input.unwrap_or(input),
                    });
                    if let Some(p) = updated_permissions {
                        out["updatedPermissions"] = Value::Array(p);
                    }
                    out
                }
                PermissionResult::Deny { message, interrupt } => {
                    let mut out = json!({"behavior": "deny", "message": message});
                    if interrupt {
                        out["interrupt"] = Value::Bool(true);
                    }
                    out
                }
            })
        }
        Some("hook_callback") => {
            let id = str_field("callback_id").unwrap_or_default();
            let callback = handlers.hooks.get(&id).ok_or_else(|| {
                Error::AgentExecution(format!("No hook callback found for ID: {id}"))
            })?;
            callback(
                request.get("input").cloned().unwrap_or(Value::Null),
                str_field("tool_use_id"),
            )
            .await
        }
        Some("mcp_message") => {
            let server = str_field("server_name").unwrap_or_default();
            let message = request.get("message").cloned().unwrap_or(Value::Null);
            if server.is_empty() || message.is_null() {
                return Err(Error::AgentExecution(
                    "Missing server_name or message for MCP request".into(),
                ));
            }
            let response = match &handlers.tool_server {
                Some(tools) if tools.name() == server => tools.handle(&message).await,
                _ => Some(json!({
                    "jsonrpc": "2.0",
                    "id": message.get("id").cloned().unwrap_or(Value::Null),
                    "error": {"code": -32601, "message": format!("Server '{server}' not found")},
                })),
            };
            Ok(json!({"mcp_response": response.unwrap_or(json!({"jsonrpc": "2.0", "result": {}}))}))
        }
        other => Err(Error::AgentExecution(format!(
            "Unsupported control request subtype: {}",
            other.unwrap_or("<none>")
        ))),
    }
}

/// The `initialize` request: hook registrations (callback ids → `hooks`),
/// sub-agents and skills.
pub(crate) fn initialize_request(
    options: &ClaudeAgentOptions,
    hooks: &mut HashMap<String, HookCallback>,
) -> Value {
    let mut config = Map::new();
    let mut next = 0usize;
    let mut events: Vec<_> = options.hooks.iter().collect();
    events.sort_by(|a, b| a.0.cmp(b.0));
    for (event, matchers) in events {
        if matchers.is_empty() {
            continue;
        }
        let mut entries = Vec::new();
        for matcher in matchers {
            let mut ids = Vec::new();
            for callback in &matcher.hooks {
                let id = format!("hook_{next}");
                next += 1;
                hooks.insert(id.clone(), callback.clone());
                ids.push(Value::String(id));
            }
            let mut entry = json!({"matcher": matcher.matcher, "hookCallbackIds": ids});
            if let Some(t) = matcher.timeout {
                entry["timeout"] = json!(t);
            }
            entries.push(entry);
        }
        config.insert(event.clone(), Value::Array(entries));
    }
    let mut request = json!({
        "subtype": "initialize",
        "hooks": if config.is_empty() { Value::Null } else { Value::Object(config) },
    });
    if let Some(agents) = &options.agents {
        request["agents"] = Value::Object(agents.clone());
    }
    if let Some(skills) = &options.skills {
        request["skills"] = json!(skills);
    }
    request
}
