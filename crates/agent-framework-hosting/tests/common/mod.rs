//! Shared test helpers: a mock agent, a tiny workflow, and HTTP/SSE utilities
//! for driving routers via `tower::ServiceExt::oneshot` (no sockets).
#![allow(dead_code)]

use std::sync::Arc;

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use futures::StreamExt;
use serde_json::Value;
use tower::ServiceExt;

use agent_framework_core::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use agent_framework_core::error::Result;
use agent_framework_core::session::AgentSession;
use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, Content, FinishReason, FunctionArguments,
    FunctionCallContent, FunctionResultContent, Message, Role, UsageDetails,
};
use agent_framework_core::workflow::{FunctionExecutor, Workflow, WorkflowBuilder};

/// A scripted agent that streams multiple text deltas, to exercise the live SSE
/// paths: `run` returns the concatenation as one message; `run_stream` yields
/// one [`AgentResponseUpdate`] per delta (real incremental streaming).
pub struct StreamingAgent {
    id: String,
    deltas: Vec<String>,
}

impl StreamingAgent {
    pub fn new(id: impl Into<String>, deltas: Vec<&str>) -> Self {
        Self {
            id: id.into(),
            deltas: deltas.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }
}

#[async_trait]
impl SupportsAgentRun for StreamingAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: vec![Message::assistant(self.deltas.concat())],
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let updates: Vec<Result<AgentResponseUpdate>> = self
            .deltas
            .iter()
            .map(|d| {
                Ok(AgentResponseUpdate {
                    contents: vec![Content::text(d)],
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// An agent whose `run_stream` yields an effectively unbounded sequence of
/// text deltas, tracking how many it produced and flipping a `cancelled` flag
/// when the stream is dropped. Lets a test assert that a disconnected client
/// stops the underlying run (cancellation) and that a slow/absent consumer
/// bounds production (backpressure) instead of running away.
pub struct CancelTrackingAgent {
    id: String,
    produced: Arc<std::sync::atomic::AtomicUsize>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl CancelTrackingAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            produced: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            cancelled: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Count of deltas the agent's stream has yielded so far.
    pub fn produced(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        self.produced.clone()
    }

    /// Set to `true` once the streaming task drops the agent stream.
    pub fn cancelled(&self) -> Arc<std::sync::atomic::AtomicBool> {
        self.cancelled.clone()
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }
}

#[async_trait]
impl SupportsAgentRun for CancelTrackingAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: vec![Message::assistant("done")],
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        use std::sync::atomic::Ordering;

        /// Flips `cancelled` when the stream (and thus this guard) is dropped.
        struct CancelGuard(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for CancelGuard {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let guard = CancelGuard(self.cancelled.clone());
        let produced = self.produced.clone();
        // A large-but-finite stream: far more than any channel capacity, so a
        // working cancellation/backpressure path stops it early, but it never
        // truly spins forever if something regresses.
        let stream = futures::stream::unfold(
            (0usize, guard, produced),
            |(n, guard, produced)| async move {
                if n >= 1_000_000 {
                    return None;
                }
                // Yield so the producer task can be preempted (and cancelled)
                // between deltas.
                tokio::task::yield_now().await;
                produced.fetch_add(1, Ordering::SeqCst);
                let update = Ok(AgentResponseUpdate {
                    contents: vec![Content::text(format!("tok{n}"))],
                    role: Some(Role::assistant()),
                    ..Default::default()
                });
                Some((update, (n + 1, guard, produced)))
            },
        );
        Ok(stream.boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// A scripted agent: echoes the concatenated input text behind a fixed prefix,
/// so tests can verify both routing and input parsing. Optionally reports usage.
pub struct MockAgent {
    id: String,
    name: Option<String>,
    prefix: String,
    usage: Option<UsageDetails>,
    finish_reason: Option<String>,
}

impl MockAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: None,
            prefix: "echo: ".to_string(),
            usage: None,
            finish_reason: None,
        }
    }

    /// Make the agent report a specific finish reason, including one outside
    /// OpenAI's vocabulary.
    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn with_usage(mut self, input: u64, output: u64) -> Self {
        self.usage = Some(UsageDetails {
            input_token_count: Some(input),
            output_token_count: Some(output),
            total_token_count: Some(input + output),
            ..Default::default()
        });
        self
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }
}

#[async_trait]
impl SupportsAgentRun for MockAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        let input = messages
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>()
            .join(" ");
        let reply = format!("{}{}", self.prefix, input.trim());
        Ok(AgentResponse {
            messages: vec![Message::assistant(reply)],
            usage_details: self.usage.clone(),
            finish_reason: self
                .finish_reason
                .as_deref()
                .map(agent_framework_core::types::FinishReason::new),
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// An agent whose turn declares function calls for the *caller* to execute —
/// what a real agent produces when its tools are client-side.
///
/// `run` returns them all on one assistant message. `run_stream` replays a
/// script of `(call_id, name, argument_fragment)` updates, so a test can
/// exercise a call whose arguments arrive across several updates, which is
/// the shape that makes streaming indices and item ids matter.
pub struct ToolCallingAgent {
    id: String,
    text: String,
    calls: Vec<(String, String, String)>,
    stream_script: Vec<StreamStep>,
    finish_reason: Option<String>,
}

/// One scripted streaming update: either text, or an argument fragment for a
/// call named by `call_id`.
pub enum StreamStep {
    Text(String),
    Call {
        call_id: String,
        name: String,
        arguments: String,
    },
}

impl ToolCallingAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: String::new(),
            calls: Vec::new(),
            stream_script: Vec::new(),
            finish_reason: None,
        }
    }

    /// Assistant text accompanying the calls. Left empty, the turn is calls
    /// only — the case where OpenAI sends `content: null`.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = text.into();
        self
    }

    /// Declare a call on the buffered turn. `arguments` is a raw JSON string,
    /// as a provider streams it.
    pub fn with_call(
        mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        self.calls
            .push((call_id.into(), name.into(), arguments.into()));
        self
    }

    pub fn with_finish_reason(mut self, reason: impl Into<String>) -> Self {
        self.finish_reason = Some(reason.into());
        self
    }

    /// Script the streaming path explicitly. Without one, `run_stream` emits
    /// the text and then each call as a single whole update.
    pub fn streaming(mut self, script: Vec<StreamStep>) -> Self {
        self.stream_script = script;
        self
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }

    fn reason(&self) -> Option<FinishReason> {
        self.finish_reason.as_deref().map(FinishReason::new)
    }

    fn call_contents(&self) -> Vec<Content> {
        self.calls
            .iter()
            .map(|(call_id, name, arguments)| {
                Content::FunctionCall(FunctionCallContent::new(
                    call_id,
                    name,
                    Some(FunctionArguments::Raw(arguments.clone())),
                ))
            })
            .collect()
    }
}

#[async_trait]
impl SupportsAgentRun for ToolCallingAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        let mut contents = Vec::new();
        if !self.text.is_empty() {
            contents.push(Content::text(&self.text));
        }
        contents.extend(self.call_contents());
        Ok(AgentResponse {
            messages: vec![Message {
                role: Role::assistant(),
                contents,
                author_name: None,
                message_id: None,
                additional_properties: Default::default(),
            }],
            finish_reason: self.reason(),
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        let mut script: Vec<StreamStep> = Vec::new();
        if self.stream_script.is_empty() {
            if !self.text.is_empty() {
                script.push(StreamStep::Text(self.text.clone()));
            }
            for (call_id, name, arguments) in &self.calls {
                script.push(StreamStep::Call {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }
        } else {
            for step in &self.stream_script {
                script.push(match step {
                    StreamStep::Text(t) => StreamStep::Text(t.clone()),
                    StreamStep::Call {
                        call_id,
                        name,
                        arguments,
                    } => StreamStep::Call {
                        call_id: call_id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                });
            }
        }

        let last = script.len().saturating_sub(1);
        let reason = self.reason();
        let updates: Vec<Result<AgentResponseUpdate>> = script
            .into_iter()
            .enumerate()
            .map(|(i, step)| {
                let contents = match step {
                    StreamStep::Text(text) => vec![Content::text(text)],
                    StreamStep::Call {
                        call_id,
                        name,
                        arguments,
                    } => vec![Content::FunctionCall(FunctionCallContent::new(
                        call_id,
                        name,
                        Some(FunctionArguments::Raw(arguments)),
                    ))],
                };
                Ok(AgentResponseUpdate {
                    contents,
                    role: Some(Role::assistant()),
                    // Only the last update carries the reason, as
                    // `response_to_updates` does.
                    finish_reason: (i == last).then(|| reason.clone()).flatten(),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// An agent that records the messages it was handed, so a test can assert
/// what a request *reached the agent as* rather than only what came back.
pub struct RecordingAgent {
    id: String,
    seen: Arc<std::sync::Mutex<Vec<Message>>>,
}

impl RecordingAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            seen: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Handle to the recorded messages, readable after the run.
    pub fn seen(&self) -> Arc<std::sync::Mutex<Vec<Message>>> {
        self.seen.clone()
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }
}

#[async_trait]
impl SupportsAgentRun for RecordingAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        *self.seen.lock().unwrap() = messages;
        Ok(AgentResponse {
            messages: vec![Message::assistant("ok")],
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        *self.seen.lock().unwrap() = messages;
        Ok(futures::stream::iter(vec![Ok(AgentResponseUpdate {
            contents: vec![Content::text("ok")],
            role: Some(Role::assistant()),
            ..Default::default()
        })])
        .boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// An agent whose turn carries a function call that has **already been
/// answered** — the shape core leaves behind after a local tool ran, and
/// the shape a provider returns for a hosted tool it executed itself.
pub struct ResolvedCallAgent {
    id: String,
    text: String,
    /// `(call_id, name, arguments, result)`.
    resolved: Vec<(String, String, String, Value)>,
    /// A call left genuinely outstanding, if any.
    outstanding: Option<(String, String, String)>,
}

impl ResolvedCallAgent {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            text: "done".to_string(),
            resolved: Vec::new(),
            outstanding: None,
        }
    }

    pub fn with_resolved(
        mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
        result: Value,
    ) -> Self {
        self.resolved
            .push((call_id.into(), name.into(), arguments.into(), result));
        self
    }

    pub fn with_outstanding(
        mut self,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        self.outstanding = Some((call_id.into(), name.into(), arguments.into()));
        self
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }

    /// The call/result pairs, plus any outstanding call, as core leaves
    /// them: the pair split across two messages, which is how a tool round
    /// trip folds back into a conversation.
    fn messages(&self) -> Vec<Message> {
        let mut calls = Vec::new();
        let mut results = Vec::new();
        for (call_id, name, arguments, result) in &self.resolved {
            calls.push(Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Raw(arguments.clone())),
            )));
            results.push(Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(result.clone()),
            )));
        }
        if let Some((call_id, name, arguments)) = &self.outstanding {
            calls.push(Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Raw(arguments.clone())),
            )));
        }
        vec![
            message_with(Role::assistant(), calls),
            message_with(Role::tool(), results),
            Message::assistant(&self.text),
        ]
    }
}

fn message_with(role: Role, contents: Vec<Content>) -> Message {
    Message {
        role,
        contents,
        author_name: None,
        message_id: None,
        additional_properties: Default::default(),
    }
}

#[async_trait]
impl SupportsAgentRun for ResolvedCallAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: self.messages(),
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        // One update per message, so the result arrives in the same update
        // as nothing else — the ordering a streaming filter has to survive.
        let updates: Vec<Result<AgentResponseUpdate>> = self
            .messages()
            .into_iter()
            .map(|m| {
                Ok(AgentResponseUpdate {
                    contents: m.contents,
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// An agent that announces a call with **no arguments yet** and then
/// streams its argument fragments — the shape this repository's own
/// Responses parser produces from `response.output_item.added` followed by
/// `response.function_call_arguments.delta`.
pub struct AnnounceThenStreamAgent {
    id: String,
    call_id: String,
    name: String,
    fragments: Vec<String>,
}

impl AnnounceThenStreamAgent {
    pub fn new(
        id: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        fragments: Vec<&str>,
    ) -> Self {
        Self {
            id: id.into(),
            call_id: call_id.into(),
            name: name.into(),
            fragments: fragments.into_iter().map(str::to_string).collect(),
        }
    }

    pub fn arc(self) -> Arc<dyn SupportsAgentRun> {
        Arc::new(self)
    }
}

#[async_trait]
impl SupportsAgentRun for AnnounceThenStreamAgent {
    async fn run(
        &self,
        _messages: Vec<Message>,
        _session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        Ok(AgentResponse {
            messages: vec![message_with(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    &self.call_id,
                    &self.name,
                    Some(FunctionArguments::Raw(self.fragments.concat())),
                ))],
            )],
            ..Default::default()
        })
    }

    async fn run_stream(
        &self,
        _messages: Vec<Message>,
        _session: Option<AgentSession>,
        _options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        // The announcement carries no arguments at all — `None`, not `""`.
        let mut updates: Vec<Result<AgentResponseUpdate>> = vec![Ok(AgentResponseUpdate {
            contents: vec![Content::FunctionCall(FunctionCallContent::new(
                &self.call_id,
                &self.name,
                None,
            ))],
            role: Some(Role::assistant()),
            ..Default::default()
        })];
        updates.extend(self.fragments.iter().map(|f| {
            Ok(AgentResponseUpdate {
                contents: vec![Content::FunctionCall(FunctionCallContent::new(
                    &self.call_id,
                    "",
                    Some(FunctionArguments::Raw(f.clone())),
                ))],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        }));
        Ok(futures::stream::iter(updates).boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// A single-executor workflow that yields `"workflow: {input}"` as its output.
pub fn echo_workflow() -> Workflow {
    let echo = FunctionExecutor::new("echo", |msg: Value, ctx| async move {
        let text = msg.as_str().unwrap_or_default().to_string();
        ctx.yield_output(Value::String(format!("workflow: {text}")))
            .await?;
        Ok(())
    });
    WorkflowBuilder::new()
        .add_executor(Arc::new(echo))
        .set_start("echo")
        .name("Echo Workflow")
        .description("Echoes its input")
        .build()
        .expect("workflow builds")
}

// ---------------------------------------------------------------------------
// HTTP helpers
// ---------------------------------------------------------------------------

/// Run one request against `app` and return `(status, body_bytes)`.
pub async fn send(app: Router, request: Request<Body>) -> (StatusCode, Vec<u8>) {
    let response = app.oneshot(request).await.expect("router responds");
    let status = response.status();
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("body collects");
    (status, bytes.to_vec())
}

/// `GET uri`, parsing the JSON body.
pub async fn get_json(app: Router, uri: &str) -> (StatusCode, Value) {
    let request = Request::builder()
        .uri(uri)
        .body(Body::empty())
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, parse_json(&bytes))
}

/// `POST uri` with a JSON body, parsing the JSON response.
pub async fn post_json(app: Router, uri: &str, body: &Value) -> (StatusCode, Value) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, parse_json(&bytes))
}

/// `POST uri` with a raw string body, returning the raw text response (for SSE
/// and malformed-payload tests).
pub async fn post_raw(app: Router, uri: &str, body: String) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .expect("request builds");
    let (status, bytes) = send(app, request).await;
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

fn parse_json(bytes: &[u8]) -> Value {
    serde_json::from_slice(bytes)
        .unwrap_or_else(|e| panic!("invalid JSON: {e}: {}", String::from_utf8_lossy(bytes)))
}

/// Extract the `data:` payloads from an SSE body, in order.
pub fn parse_sse(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(str::to_string)
        .collect()
}

/// Parse the SSE `data:` payloads, dropping the terminal `[DONE]`, into JSON.
pub fn parse_sse_json(text: &str) -> Vec<Value> {
    parse_sse(text)
        .into_iter()
        .filter(|d| d != "[DONE]")
        .map(|d| serde_json::from_str(&d).expect("SSE data is JSON"))
        .collect()
}
