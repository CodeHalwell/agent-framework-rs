//! Expose a built [`Workflow`] as an [`SupportsAgentRun`]. Rust analogue of
//! `_workflows/_agent.py` (`WorkflowAgent`).
//!
//! [`WorkflowAgent::run`] feeds the input messages to the workflow (as the
//! JSON-serialized conversation), runs it, and maps the workflow's
//! [`Output`](WorkflowEvent::Output) events into the response messages:
//! `Vec<Message>` / `Message` / string payloads become messages, other
//! payloads are JSON-stringified. Any pending request-info events are surfaced
//! as `user_input_requests` on the response (mirroring Python, which maps
//! `RequestInfoEvent` to a `request_info` function-approval request).
//!
//! Both `run` and `run_stream_with_thread` fire the session's context
//! providers' `after_run` hook with the input and response messages
//! (mirroring [`Agent`](crate::agent::Agent) and, upstream, Python's
//! `WorkflowAgent._notify_thread_of_new_messages` calls) — a write-back only,
//! not a read-back: `before_run` is never invoked, so prior session history is
//! not fed into the workflow's input, matching Python's behavior exactly.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use crate::agent::{AgentRunOptions, AgentRunStream, SupportsAgentRun};
use crate::error::Result;
use crate::history::ensure_history_provider;
use crate::session::AgentSession;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::{
    AgentResponse, AgentResponseUpdate, Content, FunctionApprovalRequestContent, FunctionArguments,
    FunctionCallContent, Message, Role,
};
use crate::workflow::{Workflow, WorkflowEvent};

/// The synthetic function name used to surface a pending request as a
/// user-input request (matches Python's `REQUEST_INFO_FUNCTION_NAME`).
const REQUEST_INFO_FUNCTION_NAME: &str = "request_info";

/// An [`SupportsAgentRun`] that wraps a [`Workflow`] and exposes it through the agent
/// interface. Rust analogue of `WorkflowAgent`.
#[derive(Clone)]
pub struct WorkflowAgent {
    workflow: Workflow,
    id: String,
    name: Option<String>,
    description: Option<String>,
}

impl WorkflowAgent {
    /// Wrap `workflow` as an agent named `name`.
    pub fn new(workflow: Workflow, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            workflow,
            id: format!("workflow_agent_{}", uuid::Uuid::new_v4().simple()),
            name: Some(name),
            description: None,
        }
    }

    /// Set an explicit id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The wrapped workflow.
    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Stream the workflow's agent activity as [`AgentResponseUpdate`]s
    /// (without thread write-back).
    ///
    /// Maps the engine's `run_stream` events: each
    /// [`AgentRunUpdate`](WorkflowEvent::AgentRunUpdate) forwards a streamed
    /// agent update and each [`RequestInfo`](WorkflowEvent::RequestInfo) becomes
    /// a user-input request update. Other events are ignored. Private helper
    /// behind [`WorkflowAgent::run_stream_with_thread`] and the object-safe
    /// [`SupportsAgentRun::run_stream`] trait method.
    fn stream_events(&self, messages: Vec<Message>) -> AgentRunStream {
        let input = serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(Vec::new()));
        let name = self.name.clone();
        let stream = self.workflow.run_stream(input);
        let mapped = stream.filter_map(move |event| {
            let name = name.clone();
            async move { convert_event(&event, name.as_deref()).map(Ok) }
        });
        Box::pin(mapped)
    }

    /// Stream the workflow's agent activity and, once the stream completes,
    /// fire `session`'s context providers' `after_run` hook the way
    /// [`Agent::run_stream`](crate::agent::Agent::run_stream) does:
    /// `messages` and the response messages reconstructed from the emitted
    /// updates are passed through. An infallible inherent convenience; the
    /// object-safe [`SupportsAgentRun::run_stream`] trait method wraps this in
    /// `Ok(..)`.
    ///
    /// Mirrors Python's `WorkflowAgent.run_stream`, which likewise only
    /// notifies the thread of new messages after the stream is exhausted — it
    /// does not feed the session's prior history back into the workflow's own
    /// input (see [`SupportsAgentRun::run`] on this type for the same convention on the
    /// non-streaming path).
    ///
    /// Because a history provider's storage is shared via `Arc`, the
    /// write-back is observable on the original session through any clone of
    /// it once the returned stream is fully consumed (same pattern as
    /// `Agent`'s).
    pub fn run_stream_with_thread(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
    ) -> AgentRunStream {
        let mut session = session.unwrap_or_else(|| self.create_session());
        ensure_history_provider(&mut session);
        let inner = self.stream_events(messages.clone());
        Box::pin(forward_and_persist(inner, session, messages))
    }

    /// Wrap this workflow-agent as a [`ToolDefinition`] callable by another
    /// agent (mirrors [`Agent::as_tool`](crate::agent::Agent::as_tool)).
    /// The tool takes a single `task` string and returns the response text.
    pub fn as_tool(&self) -> ToolDefinition {
        let agent = Arc::new(self.clone());
        let tool_name = self.name.clone().unwrap_or_else(|| self.id.clone());
        let description = self.description.clone().unwrap_or_default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "task": { "type": "string", "description": format!("Task for {tool_name}") } },
            "required": ["task"],
        });
        FunctionTool::new(tool_name, description, schema, move |args: Value| {
            let agent = agent.clone();
            async move {
                let task = args
                    .get("task")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let response = agent.run(vec![Message::user(task)], None).await?;
                Ok(Value::String(response.text()))
            }
        })
        .into_definition()
    }

    /// Map the workflow's outputs into response messages.
    fn outputs_to_messages(&self, outputs: Vec<Value>) -> Vec<Message> {
        let mut messages = Vec::new();
        for out in outputs {
            if let Ok(msgs) = serde_json::from_value::<Vec<Message>>(out.clone()) {
                messages.extend(msgs);
                continue;
            }
            if let Ok(msg) = serde_json::from_value::<Message>(out.clone()) {
                messages.push(msg);
                continue;
            }
            match out {
                Value::String(s) => messages.push(Message::assistant(s)),
                other => messages.push(Message::assistant(other.to_string())),
            }
        }
        messages
    }
}

/// Convert a pending request into a user-input (function-approval) message.
fn request_message(request_id: &str, data: &Value, name: Option<&str>) -> Message {
    let mut args: HashMap<String, Value> = HashMap::new();
    args.insert(
        "request_id".to_string(),
        Value::String(request_id.to_string()),
    );
    args.insert("data".to_string(), data.clone());
    let call = FunctionCallContent::new(
        request_id,
        REQUEST_INFO_FUNCTION_NAME,
        Some(FunctionArguments::Object(args)),
    );
    let approval = FunctionApprovalRequestContent {
        id: request_id.to_string(),
        function_call: call.clone(),
    };
    let mut msg = Message::with_contents(
        Role::assistant(),
        vec![
            Content::FunctionCall(call),
            Content::FunctionApprovalRequest(approval),
        ],
    );
    msg.author_name = name.map(str::to_string);
    msg
}

/// Convert a live workflow event into an agent update (used by `run_stream`).
fn convert_event(event: &WorkflowEvent, name: Option<&str>) -> Option<AgentResponseUpdate> {
    match event {
        WorkflowEvent::AgentRunUpdate { update, .. } => {
            // The orchestration layer emits a serialized `AgentResponseUpdate`
            // per streamed update (see `run_agent_and_emit`); forward it,
            // attributing the workflow-agent's name when the update carries none.
            let mut u: AgentResponseUpdate = serde_json::from_value(update.clone()).ok()?;
            if u.author_name.is_none() {
                u.author_name = name.map(str::to_string);
            }
            Some(u)
        }
        WorkflowEvent::RequestInfo {
            request_id,
            request_data,
            ..
        } => {
            let msg = request_message(request_id, request_data, name);
            Some(AgentResponseUpdate {
                contents: msg.contents,
                role: Some(msg.role),
                author_name: msg.author_name,
                ..Default::default()
            })
        }
        _ => None,
    }
}

/// Forward `inner`'s updates unchanged, and once it completes, fire
/// `session`'s context providers' `after_run` hook with both `input` and the
/// reconstructed response messages. Used by
/// [`WorkflowAgent::run_stream_with_thread`]; mirrors
/// [`Agent`](crate::agent::Agent)'s analogous internal stream
/// forwarder.
fn forward_and_persist(
    inner: AgentRunStream,
    session: AgentSession,
    input: Vec<Message>,
) -> impl futures::Stream<Item = Result<AgentResponseUpdate>> + Send {
    let finish: Option<(AgentSession, Vec<Message>)> = Some((session, input));
    futures::stream::unfold(
        (inner, Vec::<AgentResponseUpdate>::new(), false, finish),
        move |(mut inner, mut collected, done, mut finish)| async move {
            if done {
                return None;
            }
            match inner.next().await {
                Some(Ok(update)) => {
                    collected.push(update.clone());
                    Some((Ok(update), (inner, collected, false, finish)))
                }
                Some(Err(e)) => Some((Err(e), (inner, collected, true, finish))),
                None => {
                    if let Some((session, input)) = finish.take() {
                        let response = AgentResponse::from_updates(collected.clone());
                        for cp in session.context_providers {
                            if let Err(e) = cp.after_run(&input, &response.messages, None).await {
                                return Some((Err(e), (inner, collected, true, None)));
                            }
                        }
                    }
                    None
                }
            }
        },
    )
}

#[async_trait]
impl SupportsAgentRun for WorkflowAgent {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut AgentSession>,
    ) -> Result<AgentResponse> {
        let mut owned_session;
        let session: &mut AgentSession = match session {
            Some(s) => s,
            None => {
                owned_session = self.create_session();
                &mut owned_session
            }
        };
        ensure_history_provider(session);

        let input = serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(Vec::new()));
        let run = self.workflow.run(input).await?;

        let mut response_messages = self.outputs_to_messages(run.outputs());

        // Surface any outstanding requests as user-input requests.
        for pending in run.pending_requests() {
            response_messages.push(request_message(
                &pending.request_id,
                &pending.request_data,
                self.name.as_deref(),
            ));
        }

        // Fire the session's context providers' `after_run` hook with both
        // the input and the response messages, exactly like `Agent::run`.
        // Mirrors Python's `WorkflowAgent.run`, which calls
        // `_notify_thread_of_new_messages` with the same two message sets
        // after collecting the run's updates; like Python, `before_run` is
        // never invoked here, so prior session history is not fed back into
        // the workflow's own input — this is a write-back, not a
        // read-and-write.
        for cp in session.context_providers.clone() {
            cp.after_run(&messages, &response_messages, None).await?;
        }

        Ok(AgentResponse {
            messages: response_messages,
            ..Default::default()
        })
    }

    /// Real streaming override: maps the workflow's live events to agent
    /// updates as they happen, with the session's context providers notified
    /// on completion. Per-run [`AgentRunOptions`] are not supported (the
    /// wrapped workflow's executors carry their own options); non-empty
    /// options are ignored with a warning.
    async fn run_stream(
        &self,
        messages: Vec<Message>,
        session: Option<AgentSession>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        if let Some(opts) = &options {
            if !opts.is_empty() {
                tracing::warn!(
                    agent = %self.id,
                    "WorkflowAgent does not support per-run options; ignoring them"
                );
            }
        }
        Ok(self.run_stream_with_thread(messages, session))
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn create_session(&self) -> AgentSession {
        // Eagerly attach a fresh `InMemoryHistoryProvider`, rather than
        // relying solely on `ensure_history_provider` being called later.
        // `run_stream_with_thread` takes `Option<AgentSession>` *by value*,
        // like `Agent::run_stream` does, so the only way a caller observes
        // the post-stream write-back through a clone taken beforehand is if
        // the provider (and therefore its `Arc`) already exists at clone
        // time. Mirrors `Agent::create_session`.
        let mut session = AgentSession::new();
        ensure_history_provider(&mut session);
        session
    }
}

/// Extension trait adding [`Workflow::as_agent`](WorkflowAgentExt::as_agent) so
/// a built workflow can be exposed as an [`SupportsAgentRun`] fluently.
pub trait WorkflowAgentExt {
    /// Wrap this workflow as a [`WorkflowAgent`] named `name`.
    fn as_agent(&self, name: impl Into<String>) -> WorkflowAgent;
}

impl WorkflowAgentExt for Workflow {
    fn as_agent(&self, name: impl Into<String>) -> WorkflowAgent {
        WorkflowAgent::new(self.clone(), name)
    }
}
