//! [`ClaudeAgent`]: the Claude Code CLI as a [`SupportsAgentRun`] agent.

use std::collections::HashMap;
use std::sync::Arc;

use agent_framework_core::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, Content, FinishReason, Message, Role, TextReasoningContent,
    UsageContent, UsageDetails,
};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::mcp::{qualified_tool_name, ToolServer, TOOLS_MCP_SERVER_NAME};
use crate::options::ClaudeAgentOptions;
use crate::transport::{
    build_args, find_cli, initialize_request, Connection, Handlers, LaunchSpec,
};

/// The provider name recorded for telemetry (upstream `AGENT_PROVIDER_NAME`).
pub const AGENT_PROVIDER_NAME: &str = "anthropic.claude";

/// The `additional_properties` key carrying the result's structured output
/// on the final streamed update (a non-streaming run puts it in
/// [`AgentResponse::value`]).
pub const STRUCTURED_OUTPUT_KEY: &str = "structured_output";

/// Claude stop reasons → framework finish reasons (others pass through).
fn finish_reason(stop: &str) -> FinishReason {
    FinishReason::new(match stop {
        "end_turn" | "stop_sequence" => FinishReason::STOP,
        "max_tokens" => FinishReason::LENGTH,
        "tool_use" => FinishReason::TOOL_CALLS,
        "refusal" => FinishReason::CONTENT_FILTER,
        other => other,
    })
}

/// An agent backed by the Claude Code CLI (Python `ClaudeAgent`, over the
/// Claude Agent SDK). Each run starts a CLI process, resuming the session's
/// CLI conversation when it has one, and ends it once the run's result
/// arrives, so concurrent runs and distinct sessions never share a process.
#[derive(Clone)]
pub struct ClaudeAgent {
    inner: Arc<Inner>,
}

struct Inner {
    id: String,
    name: Option<String>,
    description: Option<String>,
    options: ClaudeAgentOptions,
    builtin_tools: Vec<String>,
    tools: Vec<ToolDefinition>,
}

impl std::fmt::Debug for ClaudeAgent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClaudeAgent")
            .field("id", &self.inner.id)
            .field("name", &self.inner.name)
            .field("builtin_tools", &self.inner.builtin_tools)
            .field(
                "tools",
                &self.inner.tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
            )
            .field("options", &self.inner.options)
            .finish()
    }
}

/// Builds a [`ClaudeAgent`].
#[derive(Debug, Default)]
pub struct ClaudeAgentBuilder {
    id: Option<String>,
    name: Option<String>,
    description: Option<String>,
    instructions: Option<String>,
    options: ClaudeAgentOptions,
    builtin_tools: Vec<String>,
    tools: Vec<ToolDefinition>,
}

impl ClaudeAgentBuilder {
    /// The agent id (default: a random UUID).
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// The agent name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The agent description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The system prompt (overrides [`ClaudeAgentOptions::system_prompt`]).
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// The CLI options. Unset settings are filled from `CLAUDE_AGENT_*`
    /// variables at [`Self::build`].
    pub fn options(mut self, options: ClaudeAgentOptions) -> Self {
        self.options = options;
        self
    }

    /// Enable a built-in CLI tool by name (`"Read"`, `"Write"`, `"Bash"`,
    /// `"Glob"`, ...); passed as `--tools`.
    pub fn builtin_tool(mut self, name: impl Into<String>) -> Self {
        self.builtin_tools.push(name.into());
        self
    }

    /// Add a function tool. It is served to the CLI from an in-process MCP
    /// server and allowed automatically.
    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Build the agent. Fails for a hosted MCP tool (configure the server
    /// natively in [`ClaudeAgentOptions::mcp_servers`] instead, since the
    /// CLI connects to MCP servers itself).
    pub fn build(self) -> Result<ClaudeAgent> {
        let mut options = self.options.from_env();
        if self.instructions.is_some() {
            options.system_prompt = self.instructions;
        }
        let mut tools = Vec::new();
        for tool in self.tools {
            if let ToolKind::HostedMcp { .. } = tool.kind {
                return Err(Error::Configuration(format!(
                    "MCP server '{name}' cannot be passed to ClaudeAgent as a tool: the Claude \
                     Agent SDK connects to MCP servers itself. Configure it in \
                     ClaudeAgentOptions::mcp_servers, e.g. {{\"{name}\": {{\"type\": \"stdio\", \
                     \"command\": \"python\", \"args\": [\"server.py\"]}}}}",
                    name = tool.name
                )));
            }
            if tool.is_executable() {
                tools.push(tool);
            } else {
                tracing::debug!(target: "agent_framework::claude", tool = %tool.name, "skipping a tool with no local executor");
            }
        }
        Ok(ClaudeAgent {
            inner: Arc::new(Inner {
                id: self.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                name: self.name,
                description: self.description,
                options,
                builtin_tools: self.builtin_tools,
                tools,
            }),
        })
    }
}

impl ClaudeAgent {
    /// A builder.
    pub fn builder() -> ClaudeAgentBuilder {
        ClaudeAgentBuilder::default()
    }

    /// An agent with `options` and nothing else.
    pub fn new(options: ClaudeAgentOptions) -> Result<Self> {
        Self::builder().options(options).build()
    }

    /// The resolved options (after `CLAUDE_AGENT_*` defaults).
    pub fn options(&self) -> &ClaudeAgentOptions {
        &self.inner.options
    }

    /// The CLI arguments a run would use, resuming `resume` (for
    /// inspection and tests).
    pub fn cli_args(&self, resume: Option<&str>) -> Vec<String> {
        build_args(&self.inner.options, &self.spec(resume, &[]))
    }

    fn spec(&self, resume: Option<&str>, extra: &[ToolDefinition]) -> LaunchSpec {
        let names: Vec<String> = self
            .inner
            .tools
            .iter()
            .chain(extra)
            .map(|t| qualified_tool_name(&t.name))
            .collect();
        LaunchSpec {
            resume: resume.map(str::to_string),
            builtin_tools: self.inner.builtin_tools.clone(),
            tool_server: (!names.is_empty()).then(|| (TOOLS_MCP_SERVER_NAME.to_string(), names)),
        }
    }

    /// Start a run: the CLI's updates arrive on the returned channel; the
    /// final one carries the CLI session id as `conversation_id`.
    fn start(
        &self,
        messages: &[Message],
        resume: Option<String>,
        options: AgentRunOptions,
    ) -> mpsc::UnboundedReceiver<Result<AgentResponseUpdate>> {
        let (tx, rx) = mpsc::unbounded_channel();
        let agent = self.clone();
        let prompt = format_prompt(messages);
        tokio::spawn(async move {
            // The run lives as long as its receiver: when the caller drops the
            // stream (or the run future), the drive future is dropped, which
            // drops the connection and kills the CLI, even while it is silent
            // or executing a tool.
            let closed = tx.clone();
            tokio::select! {
                result = agent.drive(prompt, resume, options, &tx) => {
                    if let Err(e) = result {
                        let _ = tx.send(Err(e));
                    }
                }
                () = closed.closed() => {}
            }
        });
        rx
    }

    async fn drive(
        &self,
        prompt: String,
        resume: Option<String>,
        run: AgentRunOptions,
        tx: &mpsc::UnboundedSender<Result<AgentResponseUpdate>>,
    ) -> Result<()> {
        let options = &self.inner.options;
        let extra: Vec<ToolDefinition> = run
            .additional_tools
            .iter()
            .filter(|t| t.is_executable())
            .cloned()
            .collect();
        let spec = self.spec(resume.as_deref(), &extra);
        let args = build_args(options, &spec);
        let cli = match &options.cli_path {
            Some(path) => path.clone(),
            None => find_cli()?,
        };
        let mut hooks = HashMap::new();
        let initialize = initialize_request(options, &mut hooks);
        let tools: Vec<ToolDefinition> = self.inner.tools.iter().cloned().chain(extra).collect();
        let handlers = Handlers {
            can_use_tool: options.can_use_tool.clone(),
            hooks,
            tool_server: (!tools.is_empty())
                .then(|| Arc::new(ToolServer::new(tools, options.on_function_approval.clone()))),
        };
        let mut connection = Connection::spawn(&cli, &args, options, handlers).await?;
        let outcome = async {
            connection.control(initialize).await?;
            // Per-run overrides the CLI can change after connecting.
            if let Some(chat) = &run.chat_options {
                if let Some(model) = &chat.model {
                    connection
                        .control(json!({"subtype": "set_model", "model": model}))
                        .await?;
                }
                if let Some(mode) = chat.additional_properties.get("permission_mode") {
                    connection
                        .control(json!({"subtype": "set_permission_mode", "mode": mode}))
                        .await?;
                }
            }
            connection.send_prompt(&prompt).await?;
            loop {
                let Some(message) = connection.next_message().await else {
                    return Err(connection.exited_error());
                };
                match translate(&message)? {
                    Step::Update(update) => {
                        if tx.send(Ok(update)).is_err() {
                            return Ok(());
                        }
                    }
                    Step::Done(update) => {
                        let _ = tx.send(Ok(update));
                        return Ok(());
                    }
                    Step::Skip => {}
                }
            }
        }
        .await;
        connection.close().await;
        outcome
    }
}

/// What one CLI message contributes to the run.
enum Step {
    Update(AgentResponseUpdate),
    Done(AgentResponseUpdate),
    Skip,
}

fn delta_update(content: Content, raw: &Value) -> AgentResponseUpdate {
    AgentResponseUpdate {
        contents: vec![content],
        role: Some(Role::assistant()),
        raw_representation: Some(raw.clone()),
        ..Default::default()
    }
}

/// Translate one CLI message (upstream's `_stream_run` loop).
fn translate(message: &Value) -> Result<Step> {
    match message.get("type").and_then(Value::as_str) {
        Some("stream_event") => {
            let event = message.get("event").unwrap_or(&Value::Null);
            if event.get("type").and_then(Value::as_str) != Some("content_block_delta") {
                return Ok(Step::Skip);
            }
            let delta = event.get("delta").unwrap_or(&Value::Null);
            let field = |k: &str| {
                delta
                    .get(k)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            };
            Ok(match delta.get("type").and_then(Value::as_str) {
                Some("text_delta") => field("text")
                    .map(|t| Step::Update(delta_update(Content::text(t), message)))
                    .unwrap_or(Step::Skip),
                Some("thinking_delta") => field("thinking")
                    .map(|t| {
                        Step::Update(delta_update(
                            Content::TextReasoning(TextReasoningContent {
                                text: t,
                                ..Default::default()
                            }),
                            message,
                        ))
                    })
                    .unwrap_or(Step::Skip),
                _ => Step::Skip,
            })
        }
        // Content already streamed as partial messages; only errors matter.
        Some("assistant") => match message.get("error").and_then(Value::as_str) {
            None => Ok(Step::Skip),
            Some(kind) => {
                let mut text = match kind {
                    "authentication_failed" => "Authentication failed with Claude API".to_string(),
                    "billing_error" => "Billing error with Claude API".to_string(),
                    "rate_limit" => "Rate limit exceeded for Claude API".to_string(),
                    "invalid_request" => "Invalid request to Claude API".to_string(),
                    "server_error" => "Claude API server error".to_string(),
                    "unknown" => "Unknown error from Claude API".to_string(),
                    other => format!("Claude API error: {other}"),
                };
                let detail = message
                    .pointer("/message/content")
                    .and_then(Value::as_array)
                    .and_then(|blocks| {
                        blocks.iter().find_map(|b| {
                            (b.get("type").and_then(Value::as_str) == Some("text"))
                                .then(|| b.get("text").and_then(Value::as_str))
                                .flatten()
                        })
                    });
                if let Some(detail) = detail {
                    text = format!("{text}: {detail}");
                }
                Err(Error::AgentExecution(text))
            }
        },
        Some("result") => {
            if message.get("is_error").and_then(Value::as_bool) == Some(true) {
                let detail = message
                    .get("result")
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .or_else(|| {
                        message.get("errors").and_then(Value::as_array).map(|e| {
                            e.iter()
                                .filter_map(Value::as_str)
                                .collect::<Vec<_>>()
                                .join("; ")
                        })
                    })
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "Unknown error from Claude API".into());
                return Err(Error::AgentExecution(format!("Claude API error: {detail}")));
            }
            let usage = message.get("usage").unwrap_or(&Value::Null);
            let count = |k: &str| usage.get(k).and_then(Value::as_u64);
            let (input, output) = (count("input_tokens"), count("output_tokens"));
            let details = UsageDetails {
                input_token_count: input,
                output_token_count: output,
                total_token_count: input.zip(output).map(|(i, o)| i + o),
                cache_creation_input_token_count: count("cache_creation_input_tokens"),
                cache_read_input_token_count: count("cache_read_input_tokens"),
                ..Default::default()
            };
            let mut update = AgentResponseUpdate {
                finish_reason: message
                    .get("stop_reason")
                    .and_then(Value::as_str)
                    .map(finish_reason),
                conversation_id: message
                    .get("session_id")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                raw_representation: Some(message.clone()),
                ..Default::default()
            };
            if details != UsageDetails::default() {
                update
                    .contents
                    .push(Content::Usage(UsageContent { details }));
            }
            if let Some(cost) = message.get("total_cost_usd").filter(|c| !c.is_null()) {
                update
                    .additional_properties
                    .insert("total_cost_usd".into(), cost.clone());
            }
            if let Some(output) = message.get("structured_output").filter(|v| !v.is_null()) {
                update
                    .additional_properties
                    .insert(STRUCTURED_OUTPUT_KEY.into(), output.clone());
            }
            Ok(Step::Done(update))
        }
        _ => Ok(Step::Skip),
    }
}

/// Python `json.dumps` of a string (ASCII-only, `\uXXXX` escapes).
fn py_json(s: &str) -> String {
    let encoded = serde_json::to_string(s).unwrap_or_else(|_| "\"\"".into());
    let mut out = String::with_capacity(encoded.len());
    for c in encoded.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut units = [0u16; 2];
            for unit in c.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// The prompt for a run: a lone user message's text, otherwise the messages
/// as one JSON record per line under a fixed preamble (upstream's
/// `_format_prompt`, byte for byte).
pub fn format_prompt(messages: &[Message]) -> String {
    match messages {
        [] => String::new(),
        [only] if only.role == Role::user() => only.text(),
        _ => {
            let mut out = String::from(
                "The following messages were supplied to this agent in conversation order.\n\
                 Each JSON record contains the original speaker's role and message content.\n\
                 Use these messages as context. If the final message is a user request, \
                 respond to it while following your instructions.\n",
            );
            let records: Vec<String> = messages
                .iter()
                .map(|m| {
                    format!(
                        "{{\"role\": {}, \"content\": {}}}",
                        py_json(m.role.as_str()),
                        py_json(&m.text())
                    )
                })
                .collect();
            out.push_str(&records.join("\n"));
            out
        }
    }
}

#[async_trait]
impl SupportsAgentRun for ClaudeAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        self.run_with_options(messages, session, AgentRunOptions::default())
            .await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        let resume = session
            .as_ref()
            .and_then(|s| s.service_session_id().map(str::to_string));
        let mut rx = self.start(&messages, resume, options);
        let mut updates = Vec::new();
        while let Some(update) = rx.recv().await {
            updates.push(update?);
        }
        let mut response = AgentResponse::from_updates(updates);
        response.value = response.additional_properties.remove(STRUCTURED_OUTPUT_KEY);
        if let Some(name) = &self.inner.name {
            for m in &mut response.messages {
                m.author_name.get_or_insert_with(|| name.clone());
            }
        }
        if let (Some(session), Some(id)) = (session, response.conversation_id.as_deref()) {
            session.set_service_session_id(id);
        }
        Ok(response)
    }

    /// Streams text and reasoning deltas as they arrive. The final update
    /// carries the usage, finish reason, the CLI session id (as
    /// `conversation_id`, to resume with) and any structured output (under
    /// [`STRUCTURED_OUTPUT_KEY`] in `additional_properties`).
    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let resume = session
            .as_ref()
            .and_then(|s| s.service_session_id().map(str::to_string));
        let mut rx = self.start(&messages, resume, options.unwrap_or_default());
        let author = self.inner.name.clone();
        Ok(futures::stream::poll_fn(move |cx| rx.poll_recv(cx))
            .map(move |item| {
                item.map(|mut update| {
                    if update.author_name.is_none() && !update.contents.is_empty() {
                        update.author_name = author.clone();
                    }
                    update
                })
            })
            .boxed())
    }

    fn id(&self) -> &str {
        &self.inner.id
    }

    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    fn description(&self) -> Option<&str> {
        self.inner.description.as_deref()
    }
}
