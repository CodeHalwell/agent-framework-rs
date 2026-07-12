//! [`A2AAgent`]: wraps an [`A2AClient`] as a local [`Agent`].

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use agent_framework_core::agent::{Agent, AgentRunOptions, AgentRunStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::threads::AgentThread;
use agent_framework_core::types::{
    AgentRunResponse, AgentRunResponseUpdate, ChatMessage, Content, DataContent, IntoMessages,
    Role, UriContent,
};

use crate::client::{A2AClient, A2AEventStream};
use crate::types::{
    AgentCard, Artifact, FileData, FilePart, FileWithBytes, FileWithUri, Message as A2AMessage,
    MessageRole, MessageSendParams, MessageStreamEvent, Part, SendMessageResult, Task, TaskState,
    TextPart,
};

/// Agent2Agent (A2A) protocol client agent.
///
/// Wraps an [`A2AClient`] so a remote, A2A-compliant agent can be used
/// anywhere the framework expects a local [`Agent`]: [`ChatMessage`]s in,
/// [`AgentRunResponse`] out, with `contextId`/`taskId` continuity tracked per
/// [`AgentThread`]. See the crate docs for the exact mapping and how this
/// diverges from the Python reference implementation.
#[derive(Debug, Clone)]
pub struct A2AAgent {
    id: String,
    name: Option<String>,
    description: Option<String>,
    client: Arc<A2AClient>,
}

impl A2AAgent {
    /// Point at a remote agent's JSON-RPC endpoint directly. The real
    /// [`AgentCard`] is discovered lazily — on the first [`Agent::run`] call,
    /// or explicitly via [`Self::initialize`] — falling back to using `url`
    /// itself as the JSON-RPC endpoint if discovery isn't available (many
    /// minimal A2A servers don't expose a `.well-known` document).
    pub fn from_url(name: impl Into<String>, url: impl Into<String>) -> Self {
        Self::from_client(name, A2AClient::from_url(url))
    }

    /// Use an already-known [`AgentCard`] (e.g. fetched out-of-band, or
    /// embedded in configuration). No discovery call is ever made.
    pub fn from_card(name: impl Into<String>, card: AgentCard) -> Self {
        Self::from_client(name, A2AClient::from_card(card))
    }

    /// Wrap a caller-configured [`A2AClient`] (custom headers, auth,
    /// timeout, …) as an agent.
    pub fn from_client(name: impl Into<String>, client: A2AClient) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: Some(name.into()),
            description: None,
            client: Arc::new(client),
        }
    }

    /// Override the agent id (defaults to a random UUID).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the agent description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// The underlying client, for direct `A2AClient` calls (`get_task`,
    /// `cancel_task`, `send_message_stream`, …) alongside the agent's `run`.
    pub fn client(&self) -> &Arc<A2AClient> {
        &self.client
    }

    /// Ergonomic run without an explicit thread (mirrors
    /// `ChatAgent::run_once`): the conversation starts fresh every call,
    /// since no [`AgentThread`] is carried across calls to persist
    /// `contextId`/`taskId`.
    pub async fn run_once(&self, messages: impl IntoMessages) -> Result<AgentRunResponse> {
        self.run(messages.into_messages(), None).await
    }

    /// Resolve the remote [`AgentCard`], propagating discovery failures.
    ///
    /// [`Agent::run`] performs the best-effort equivalent of this
    /// automatically on first use; call this explicitly when you want to
    /// know discovery actually succeeded (e.g. to read `capabilities` /
    /// `skills` before the first run) rather than have it silently fall back
    /// to using the constructor URL as the JSON-RPC endpoint.
    pub async fn initialize(&self) -> Result<AgentCard> {
        self.client.get_agent_card().await
    }

    /// Best-effort card discovery used internally by [`Agent::run`]: unlike
    /// [`Self::initialize`], never fails the run just because `.well-known`
    /// discovery isn't available.
    async fn ensure_initialized(&self) {
        let _ = self.client.get_agent_card().await;
    }
}

/// `contextId`/`taskId` continuity for one [`AgentThread`], packed into
/// [`AgentThread::service_thread_id`] as JSON (that field is a single
/// string, and A2A conversation continuity needs both ids).
///
/// The Python reference does not track this at all: `A2AAgent.run_stream`
/// accepts a `thread` parameter but never reads or writes it, so every call
/// sends a context-less message and relies entirely on whatever session
/// affinity the remote agent infers on its own. Tracking it here lets a
/// caller that reuses the same [`AgentThread`] across calls have a real
/// multi-turn A2A conversation, including resuming a task that is paused in
/// [`TaskState::InputRequired`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct ThreadState {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    context_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    task_id: Option<String>,
}

impl ThreadState {
    /// Decode from `AgentThread::service_thread_id()`; anything absent,
    /// unparseable, or from an unrelated thread just yields an empty state,
    /// i.e. "start a fresh A2A conversation".
    fn decode(raw: Option<&str>) -> Self {
        raw.and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default()
    }

    /// `None` when there is nothing worth persisting.
    fn encode(&self) -> Option<String> {
        if self.context_id.is_none() && self.task_id.is_none() {
            None
        } else {
            serde_json::to_string(self).ok()
        }
    }
}

#[async_trait]
impl Agent for A2AAgent {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        // Mirrors the Python reference: only the newest message is sent —
        // earlier turns are already known to the remote agent via its
        // `contextId`/`taskId`, which this port tracks below.
        let last = messages.last().ok_or_else(|| {
            Error::AgentExecution("A2AAgent::run requires at least one message".into())
        })?;

        self.ensure_initialized().await;

        let mut owned_thread;
        let thread: &mut AgentThread = match thread {
            Some(t) => t,
            None => {
                owned_thread = self.get_new_thread();
                &mut owned_thread
            }
        };

        let state = ThreadState::decode(thread.service_thread_id());
        let outgoing = chat_message_to_a2a_message(
            last,
            state.context_id.as_deref(),
            state.task_id.as_deref(),
        )?;

        let result = self
            .client
            .send_message(MessageSendParams::new(outgoing))
            .await?;

        let (out_messages, response_id, new_state) = match result {
            SendMessageResult::Message(m) => {
                let response_id = m.message_id.clone();
                let new_state = ThreadState {
                    context_id: m.context_id.clone().or_else(|| state.context_id.clone()),
                    task_id: m.task_id.clone(),
                };
                (
                    vec![a2a_message_to_chat_message(&m)?],
                    response_id,
                    new_state,
                )
            }
            SendMessageResult::Task(t) => {
                let response_id = t.id.clone();
                let new_state = ThreadState {
                    context_id: Some(t.context_id.clone()),
                    task_id: Some(t.id.clone()),
                };
                (task_to_chat_messages(&t)?, response_id, new_state)
            }
        };

        // Best effort: this only fails if `thread` already owns a local
        // message store (mutually exclusive with a service thread id on
        // `AgentThread`) — e.g. a thread borrowed from a `ChatAgent`. In that
        // case the run still succeeds; it just won't get contextId/taskId
        // continuity on that particular thread.
        if let Some(encoded) = new_state.encode() {
            let _ = thread.set_service_thread_id(encoded);
        }

        let mut response = AgentRunResponse {
            messages: out_messages,
            response_id: Some(response_id),
            ..Default::default()
        };
        if let Some(name) = &self.name {
            for m in &mut response.messages {
                if m.author_name.is_none() {
                    m.author_name = Some(name.clone());
                }
            }
        }
        Ok(response)
    }

    /// Real streaming override: sends the newest message via the client's
    /// `message/stream` SSE endpoint and maps each
    /// [`MessageStreamEvent`] to an [`AgentRunResponseUpdate`] as it arrives,
    /// accumulating `contextId`/`taskId` and writing the resulting
    /// continuity state back onto the (owned) thread once the stream ends —
    /// mirroring [`Agent::run`]'s bookkeeping on this type. Per-run
    /// [`AgentRunOptions`] have no A2A representation; non-empty options are
    /// ignored with a warning.
    ///
    /// Note: unlike [`ChatAgent`](agent_framework_core::agent::ChatAgent),
    /// A2A continuity state lives in a plain (non-shared) thread field, so the
    /// end-of-stream write-back is not observable through a thread clone taken
    /// before streaming — carry the returned continuity forward via
    /// [`Agent::run`] when you need multi-turn A2A conversations.
    async fn run_stream(
        &self,
        messages: Vec<ChatMessage>,
        thread: Option<AgentThread>,
        options: Option<AgentRunOptions>,
    ) -> Result<AgentRunStream> {
        if let Some(opts) = &options {
            if !opts.is_empty() {
                tracing::warn!(
                    agent = %self.id,
                    "A2AAgent does not support per-run options; ignoring them"
                );
            }
        }

        let last = messages
            .last()
            .ok_or_else(|| {
                Error::AgentExecution("A2AAgent::run_stream requires at least one message".into())
            })?
            .clone();

        self.ensure_initialized().await;

        let thread = thread.unwrap_or_else(|| self.get_new_thread());
        let state = ThreadState::decode(thread.service_thread_id());
        let outgoing = chat_message_to_a2a_message(
            &last,
            state.context_id.as_deref(),
            state.task_id.as_deref(),
        )?;

        let inner = self
            .client
            .send_message_stream(MessageSendParams::new(outgoing))
            .await?;

        // `state` (decoded above) is threaded through the forwarder, which
        // accumulates `contextId`/`taskId` from the streamed events and writes
        // the result back onto the thread once the stream ends.
        let stream = a2a_forward_stream(inner, thread, state, self.name.clone());
        Ok(stream.boxed())
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Map one A2A `message/stream` event to zero or more agent updates, threading
/// `contextId`/`taskId` continuity through `state`. Pure (no I/O) so it can be
/// unit-tested directly, the same way the crate tests the non-streaming
/// conversions.
fn stream_event_to_updates(
    event: &MessageStreamEvent,
    state: &mut ThreadState,
    name: Option<&str>,
) -> Vec<Result<AgentRunResponseUpdate>> {
    // `response_id` follows the same rule as the non-streaming `run()`
    // (`SendMessageResult` mapping above): a bare message exposes its
    // message id, task-bearing events expose the task id — so aggregating
    // the stream yields the same `AgentRunResponse.response_id` and
    // streaming clients can cancel/poll the A2A task.
    let (updates, response_id) = match event {
        MessageStreamEvent::Message(m) => {
            if m.context_id.is_some() {
                state.context_id = m.context_id.clone();
            }
            // Replace unconditionally, exactly like the non-streaming
            // `SendMessageResult::Message` mapping: a standalone message
            // with no task id means there is no active task, and keeping a
            // previous (completed) task's id would make the next outgoing
            // message continue the wrong task. (`context_id` above keeps the
            // old value when absent — also matching run().)
            state.task_id = m.task_id.clone();
            (
                wrap_message(a2a_message_to_chat_message(m), name),
                Some(m.message_id.clone()),
            )
        }
        MessageStreamEvent::Task(t) => {
            state.context_id = Some(t.context_id.clone());
            state.task_id = Some(t.id.clone());
            let updates = match task_to_chat_messages(t) {
                Ok(msgs) => msgs
                    .into_iter()
                    .map(|cm| Ok(message_to_update(cm, name)))
                    .collect(),
                Err(e) => vec![Err(e)],
            };
            (updates, Some(t.id.clone()))
        }
        MessageStreamEvent::StatusUpdate(e) => {
            state.context_id = Some(e.context_id.clone());
            state.task_id = Some(e.task_id.clone());
            let updates = match &e.status.message {
                Some(msg) => wrap_message(a2a_message_to_chat_message(msg), name),
                None => Vec::new(),
            };
            (updates, Some(e.task_id.clone()))
        }
        MessageStreamEvent::ArtifactUpdate(e) => {
            state.context_id = Some(e.context_id.clone());
            state.task_id = Some(e.task_id.clone());
            (
                wrap_message(artifact_to_chat_message(&e.artifact), name),
                Some(e.task_id.clone()),
            )
        }
    };
    updates
        .into_iter()
        .map(|u| {
            u.map(|mut update| {
                update.response_id = response_id.clone();
                update
            })
        })
        .collect()
}

fn wrap_message(
    result: Result<ChatMessage>,
    name: Option<&str>,
) -> Vec<Result<AgentRunResponseUpdate>> {
    match result {
        Ok(cm) => vec![Ok(message_to_update(cm, name))],
        Err(e) => vec![Err(e)],
    }
}

fn message_to_update(cm: ChatMessage, name: Option<&str>) -> AgentRunResponseUpdate {
    AgentRunResponseUpdate {
        contents: cm.contents,
        role: Some(cm.role),
        author_name: cm.author_name.or_else(|| name.map(str::to_string)),
        message_id: cm.message_id,
        ..Default::default()
    }
}

/// State carried while forwarding an A2A event stream as agent updates.
struct A2AForward {
    inner: A2AEventStream,
    queue: VecDeque<Result<AgentRunResponseUpdate>>,
    state: ThreadState,
    thread: Option<AgentThread>,
    name: Option<String>,
    done: bool,
}

/// Forward an A2A `message/stream` event stream as agent updates, writing the
/// accumulated `contextId`/`taskId` continuity back onto `thread` once the
/// stream is exhausted.
fn a2a_forward_stream(
    inner: A2AEventStream,
    thread: AgentThread,
    state: ThreadState,
    name: Option<String>,
) -> impl futures::Stream<Item = Result<AgentRunResponseUpdate>> + Send {
    futures::stream::unfold(
        A2AForward {
            inner,
            queue: VecDeque::new(),
            state,
            thread: Some(thread),
            name,
            done: false,
        },
        |mut st| async move {
            loop {
                if let Some(item) = st.queue.pop_front() {
                    return Some((item, st));
                }
                if st.done {
                    // Stream exhausted: persist the accumulated continuity.
                    if let Some(mut thread) = st.thread.take() {
                        if let Some(encoded) = st.state.encode() {
                            let _ = thread.set_service_thread_id(encoded);
                        }
                    }
                    return None;
                }
                match st.inner.next().await {
                    Some(Ok(event)) => {
                        let updates =
                            stream_event_to_updates(&event, &mut st.state, st.name.as_deref());
                        st.queue.extend(updates);
                    }
                    Some(Err(e)) => {
                        st.done = true;
                        st.queue.push_back(Err(e));
                    }
                    None => st.done = true,
                }
            }
        },
    )
}

// ---------------------------------------------------------------------
// ChatMessage <-> A2A Message/Part conversion
// ---------------------------------------------------------------------

/// Convert a framework [`ChatMessage`] into an A2A [`A2AMessage`], carrying
/// forward `contextId`/`taskId` for conversation continuity.
///
/// Mirrors the Python reference's `_chat_message_to_a2a_message`: text,
/// error, URI, data, and hosted-file content map to A2A parts; any other
/// content kind (function calls/results, usage, …) is rejected, since A2A has
/// no representation for them. The message role is always sent as `user`,
/// matching Python — framework messages passed to `run` are treated as user
/// input regardless of their original [`Role`].
fn chat_message_to_a2a_message(
    message: &ChatMessage,
    context_id: Option<&str>,
    task_id: Option<&str>,
) -> Result<A2AMessage> {
    if message.contents.is_empty() {
        return Err(Error::Content(
            "ChatMessage.contents is empty; cannot convert to an A2A message".into(),
        ));
    }

    let mut parts = Vec::with_capacity(message.contents.len());
    for content in &message.contents {
        parts.push(content_to_part(content)?);
    }

    let metadata = if message.additional_properties.is_empty() {
        None
    } else {
        Some(serde_json::to_value(&message.additional_properties)?)
    };

    Ok(A2AMessage {
        kind: "message".to_string(),
        role: MessageRole::User,
        parts,
        message_id: message
            .message_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        task_id: task_id.map(str::to_string),
        context_id: context_id.map(str::to_string),
        metadata,
    })
}

fn content_to_part(content: &Content) -> Result<Part> {
    match content {
        Content::Text(t) => Ok(Part::Text(TextPart {
            text: t.text.clone(),
            metadata: None,
        })),
        Content::Error(e) => Ok(Part::Text(TextPart {
            text: e
                .message
                .clone()
                .unwrap_or_else(|| "An error occurred.".to_string()),
            metadata: None,
        })),
        Content::Uri(u) => Ok(Part::File(FilePart {
            file: FileData::Uri(FileWithUri {
                uri: u.uri.clone(),
                mime_type: Some(u.media_type.clone()),
                name: None,
            }),
            metadata: None,
        })),
        Content::Data(d) => Ok(Part::File(FilePart {
            file: FileData::Bytes(FileWithBytes {
                bytes: data_uri_payload(&d.uri)?.to_string(),
                mime_type: d.media_type.clone(),
                name: None,
            }),
            metadata: None,
        })),
        Content::HostedFile(h) => Ok(Part::File(FilePart {
            file: FileData::Uri(FileWithUri {
                uri: h.file_id.clone(),
                mime_type: None,
                name: None,
            }),
            metadata: None,
        })),
        other => Err(Error::Content(format!(
            "A2A does not support content type '{}' in an outgoing message",
            content_type_name(other)
        ))),
    }
}

/// The `type` tag of a [`Content`] item, for error messages. An exhaustive
/// match (no wildcard) so a new [`Content`] variant fails to compile here
/// rather than being silently miscategorized.
fn content_type_name(content: &Content) -> &'static str {
    match content {
        Content::Text(_) => "text",
        Content::TextReasoning(_) => "text_reasoning",
        Content::Data(_) => "data",
        Content::Uri(_) => "uri",
        Content::Error(_) => "error",
        Content::FunctionCall(_) => "function_call",
        Content::FunctionResult(_) => "function_result",
        Content::Usage(_) => "usage",
        Content::HostedFile(_) => "hosted_file",
        Content::HostedVectorStore(_) => "hosted_vector_store",
        Content::FunctionApprovalRequest(_) => "function_approval_request",
        Content::FunctionApprovalResponse(_) => "function_approval_response",
        Content::Unknown => "unknown",
    }
}

/// Extract the base64 payload from a `data:` URI (e.g.
/// `data:image/png;base64,AAAA` -> `AAAA`), matching the Python reference's
/// `_get_uri_data`. [`DataContent::uri`] is always base64 (see
/// [`DataContent::from_bytes`]), so no decode/re-encode round trip is needed
/// in either conversion direction — just this substring extraction, and its
/// mirror image in [`part_to_content`].
fn data_uri_payload(uri: &str) -> Result<&str> {
    uri.split_once(";base64,")
        .map(|(_, payload)| payload)
        .ok_or_else(|| Error::Content(format!("expected a base64 data URI, got: {uri}")))
}

/// Convert A2A [`Part`]s into framework [`Content`] items.
///
/// Mirrors the Python reference's `_a2a_parts_to_contents`: text parts map to
/// [`Content::Text`]; file parts map to [`Content::Uri`] or [`Content::Data`]
/// depending on whether the file carries a URI or inline bytes; data parts
/// (arbitrary structured JSON) are serialized to text, same as Python.
fn a2a_parts_to_contents(parts: &[Part]) -> Result<Vec<Content>> {
    parts.iter().map(part_to_content).collect()
}

fn part_to_content(part: &Part) -> Result<Content> {
    match part {
        Part::Text(t) => Ok(Content::text(t.text.clone())),
        Part::File(f) => match &f.file {
            FileData::Uri(u) => Ok(Content::Uri(UriContent {
                uri: u.uri.clone(),
                media_type: u.mime_type.clone().unwrap_or_default(),
            })),
            FileData::Bytes(b) => Ok(Content::Data(DataContent {
                uri: format!(
                    "data:{};base64,{}",
                    b.mime_type.as_deref().unwrap_or("application/octet-stream"),
                    b.bytes
                ),
                media_type: b.mime_type.clone(),
            })),
        },
        Part::Data(d) => {
            let text = serde_json::to_string(&d.data)
                .map_err(|e| Error::Content(format!("failed to serialize A2A data part: {e}")))?;
            Ok(Content::text(text))
        }
    }
}

fn a2a_role_to_role(role: MessageRole) -> Role {
    match role {
        MessageRole::Agent => Role::assistant(),
        MessageRole::User => Role::user(),
    }
}

/// Extract a JSON object's entries as `additional_properties`; anything else
/// (absent, or not an object) just yields an empty map.
fn metadata_to_additional_properties(metadata: &Option<Value>) -> HashMap<String, Value> {
    match metadata {
        Some(Value::Object(map)) => map.clone().into_iter().collect(),
        _ => HashMap::new(),
    }
}

fn a2a_message_to_chat_message(message: &A2AMessage) -> Result<ChatMessage> {
    Ok(ChatMessage {
        role: a2a_role_to_role(message.role),
        contents: a2a_parts_to_contents(&message.parts)?,
        author_name: None,
        message_id: Some(message.message_id.clone()),
        additional_properties: metadata_to_additional_properties(&message.metadata),
    })
}

fn artifact_to_chat_message(artifact: &Artifact) -> Result<ChatMessage> {
    Ok(ChatMessage {
        role: Role::assistant(),
        contents: a2a_parts_to_contents(&artifact.parts)?,
        author_name: None,
        message_id: Some(artifact.artifact_id.clone()),
        additional_properties: metadata_to_additional_properties(&artifact.metadata),
    })
}

/// Map a returned [`Task`] to zero or more [`ChatMessage`]s.
///
/// Priority:
/// 1. If the task is paused in [`TaskState::InputRequired`] and carries a
///    `status.message`, that message *is* the response — this surfaces what
///    the remote agent is asking for. This branch has no equivalent in the
///    Python reference (see the crate docs "Divergences" section): Python
///    only ever looks at `artifacts`/`history`, so it would silently produce
///    no messages here unless the server happens to also put the question in
///    `history`.
/// 2. Otherwise, mirrors Python's `_task_to_chat_messages`: prefer
///    `artifacts` (one [`ChatMessage`] per artifact), falling back to the
///    last `history` entry, else no messages at all (the task is still
///    `working`/`submitted` with nothing to show yet — poll
///    [`A2AClient::get_task`](crate::client::A2AClient::get_task) for
///    updates).
fn task_to_chat_messages(task: &Task) -> Result<Vec<ChatMessage>> {
    if task.status.state == TaskState::InputRequired {
        if let Some(msg) = &task.status.message {
            return Ok(vec![a2a_message_to_chat_message(msg)?]);
        }
    }
    if let Some(artifacts) = &task.artifacts {
        if !artifacts.is_empty() {
            return artifacts.iter().map(artifact_to_chat_message).collect();
        }
    }
    if let Some(history) = &task.history {
        if let Some(last) = history.last() {
            return Ok(vec![a2a_message_to_chat_message(last)?]);
        }
    }
    Ok(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DataPart;
    use agent_framework_core::types::{
        ErrorContent, FunctionCallContent, HostedFileContent, TextContent,
    };

    fn text_message(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }

    // -- ChatMessage -> A2A Message -----------------------------------

    #[test]
    fn chat_message_to_a2a_message_maps_text() {
        let msg = text_message("hello there");
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        assert_eq!(a2a.role, MessageRole::User);
        assert_eq!(a2a.parts.len(), 1);
        match &a2a.parts[0] {
            Part::Text(t) => assert_eq!(t.text, "hello there"),
            other => panic!("expected TextPart, got {other:?}"),
        }
    }

    #[test]
    fn chat_message_to_a2a_message_maps_error_content() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::user(),
            vec![Content::Error(ErrorContent {
                message: Some("Test error message".into()),
                error_code: None,
                details: None,
            })],
        );
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        match &a2a.parts[0] {
            Part::Text(t) => assert_eq!(t.text, "Test error message"),
            other => panic!("expected TextPart, got {other:?}"),
        }
    }

    #[test]
    fn chat_message_to_a2a_message_maps_uri_content() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::user(),
            vec![Content::Uri(UriContent {
                uri: "http://example.com/file.pdf".into(),
                media_type: "application/pdf".into(),
            })],
        );
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        match &a2a.parts[0] {
            Part::File(f) => match &f.file {
                FileData::Uri(u) => {
                    assert_eq!(u.uri, "http://example.com/file.pdf");
                    assert_eq!(u.mime_type.as_deref(), Some("application/pdf"));
                }
                FileData::Bytes(_) => panic!("expected a URI file"),
            },
            other => panic!("expected FilePart, got {other:?}"),
        }
    }

    #[test]
    fn chat_message_to_a2a_message_maps_data_content() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::user(),
            vec![Content::Data(DataContent::from_bytes(
                b"hello",
                "text/plain",
            ))],
        );
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        match &a2a.parts[0] {
            Part::File(f) => match &f.file {
                FileData::Bytes(b) => {
                    assert_eq!(b.mime_type.as_deref(), Some("text/plain"));
                    // "hello" base64-encoded.
                    assert_eq!(b.bytes, "aGVsbG8=");
                }
                FileData::Uri(_) => panic!("expected a bytes file"),
            },
            other => panic!("expected FilePart, got {other:?}"),
        }
    }

    #[test]
    fn chat_message_to_a2a_message_maps_hosted_file() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::user(),
            vec![Content::HostedFile(HostedFileContent {
                file_id: "hosted://storage/document.pdf".into(),
            })],
        );
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        match &a2a.parts[0] {
            Part::File(f) => match &f.file {
                FileData::Uri(u) => {
                    assert_eq!(u.uri, "hosted://storage/document.pdf");
                    assert!(u.mime_type.is_none());
                }
                FileData::Bytes(_) => panic!("expected a URI file"),
            },
            other => panic!("expected FilePart, got {other:?}"),
        }
    }

    #[test]
    fn chat_message_to_a2a_message_rejects_function_call_content() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call-1",
                "get_weather",
                None,
            ))],
        );
        let err = chat_message_to_a2a_message(&msg, None, None).unwrap_err();
        assert!(matches!(err, Error::Content(_)));
    }

    #[test]
    fn chat_message_to_a2a_message_empty_contents_errors() {
        let msg = ChatMessage::with_contents(agent_framework_core::types::Role::user(), Vec::new());
        let err = chat_message_to_a2a_message(&msg, None, None).unwrap_err();
        assert!(matches!(err, Error::Content(_)));
    }

    #[test]
    fn chat_message_to_a2a_message_carries_context_and_task_id() {
        let msg = text_message("continue please");
        let a2a = chat_message_to_a2a_message(&msg, Some("ctx-1"), Some("task-1")).unwrap();
        assert_eq!(a2a.context_id.as_deref(), Some("ctx-1"));
        assert_eq!(a2a.task_id.as_deref(), Some("task-1"));
    }

    #[test]
    fn chat_message_to_a2a_message_with_multiple_contents() {
        let msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::user(),
            vec![
                Content::Text(TextContent::new("Here's the analysis:")),
                Content::Data(DataContent::from_bytes(
                    b"binary data",
                    "application/octet-stream",
                )),
                Content::Uri(UriContent {
                    uri: "https://example.com/image.png".into(),
                    media_type: "image/png".into(),
                }),
            ],
        );
        let a2a = chat_message_to_a2a_message(&msg, None, None).unwrap();
        assert_eq!(a2a.parts.len(), 3);
        assert!(matches!(a2a.parts[0], Part::Text(_)));
        assert!(matches!(a2a.parts[1], Part::File(_)));
        assert!(matches!(a2a.parts[2], Part::File(_)));
    }

    // -- A2A Part -> Content --------------------------------------------

    #[test]
    fn a2a_parts_to_contents_maps_text_uri_bytes_and_data() {
        let parts = vec![
            Part::Text(TextPart {
                text: "hi".into(),
                metadata: None,
            }),
            Part::File(FilePart {
                file: FileData::Uri(FileWithUri {
                    uri: "https://x/y.png".into(),
                    mime_type: Some("image/png".into()),
                    name: None,
                }),
                metadata: None,
            }),
            Part::File(FilePart {
                file: FileData::Bytes(FileWithBytes {
                    bytes: "aGVsbG8=".into(),
                    mime_type: Some("text/plain".into()),
                    name: None,
                }),
                metadata: None,
            }),
            Part::Data(DataPart {
                data: serde_json::json!({"key": "value", "number": 42}),
                metadata: None,
            }),
        ];
        let contents = a2a_parts_to_contents(&parts).unwrap();
        assert_eq!(contents.len(), 4);
        assert!(matches!(&contents[0], Content::Text(t) if t.text == "hi"));
        assert!(matches!(&contents[1], Content::Uri(u) if u.uri == "https://x/y.png"));
        match &contents[2] {
            Content::Data(d) => assert_eq!(d.uri, "data:text/plain;base64,aGVsbG8="),
            other => panic!("expected Content::Data, got {other:?}"),
        }
        match &contents[3] {
            Content::Text(t) => assert_eq!(t.text, r#"{"key":"value","number":42}"#),
            other => panic!("expected Content::Text, got {other:?}"),
        }
    }

    #[test]
    fn a2a_message_to_chat_message_maps_agent_role_and_metadata() {
        let a2a = A2AMessage {
            kind: "message".to_string(),
            role: MessageRole::Agent,
            parts: vec![Part::Text(TextPart {
                text: "hi".into(),
                metadata: None,
            })],
            message_id: "m1".into(),
            task_id: None,
            context_id: None,
            metadata: Some(serde_json::json!({"source": "test"})),
        };
        let cm = a2a_message_to_chat_message(&a2a).unwrap();
        assert_eq!(cm.role, agent_framework_core::types::Role::assistant());
        assert_eq!(cm.message_id.as_deref(), Some("m1"));
        assert_eq!(
            cm.additional_properties.get("source").unwrap(),
            &serde_json::json!("test")
        );
    }

    // -- Task -> ChatMessages ---------------------------------------------

    fn completed_task(artifacts: Vec<Artifact>) -> Task {
        Task {
            id: "task-1".into(),
            context_id: "ctx-1".into(),
            status: crate::types::TaskStatus {
                state: TaskState::Completed,
                message: None,
                timestamp: None,
            },
            artifacts: if artifacts.is_empty() {
                None
            } else {
                Some(artifacts)
            },
            history: None,
            metadata: None,
        }
    }

    fn artifact(id: &str, text: &str) -> Artifact {
        Artifact {
            artifact_id: id.into(),
            name: None,
            description: None,
            parts: vec![Part::Text(TextPart {
                text: text.into(),
                metadata: None,
            })],
            metadata: None,
        }
    }

    #[test]
    fn task_to_chat_messages_prefers_artifacts() {
        let task = completed_task(vec![
            artifact("art-1", "first"),
            artifact("art-2", "second"),
        ]);
        let messages = task_to_chat_messages(&task).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text(), "first");
        assert_eq!(messages[0].message_id.as_deref(), Some("art-1"));
        assert_eq!(messages[1].text(), "second");
        for m in &messages {
            assert_eq!(m.role, agent_framework_core::types::Role::assistant());
        }
    }

    #[test]
    fn task_to_chat_messages_falls_back_to_history_last() {
        let mut task = completed_task(vec![]);
        task.history = Some(vec![
            A2AMessage {
                kind: "message".to_string(),
                role: MessageRole::User,
                parts: vec![Part::Text(TextPart {
                    text: "question".into(),
                    metadata: None,
                })],
                message_id: "h1".into(),
                task_id: None,
                context_id: None,
                metadata: None,
            },
            A2AMessage {
                kind: "message".to_string(),
                role: MessageRole::Agent,
                parts: vec![Part::Text(TextPart {
                    text: "answer".into(),
                    metadata: None,
                })],
                message_id: "h2".into(),
                task_id: None,
                context_id: None,
                metadata: None,
            },
        ]);
        let messages = task_to_chat_messages(&task).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "answer");
        assert_eq!(
            messages[0].role,
            agent_framework_core::types::Role::assistant()
        );
    }

    #[test]
    fn task_to_chat_messages_input_required_uses_status_message() {
        let mut task = completed_task(vec![]);
        task.status = crate::types::TaskStatus {
            state: TaskState::InputRequired,
            message: Some(A2AMessage {
                kind: "message".to_string(),
                role: MessageRole::Agent,
                parts: vec![Part::Text(TextPart {
                    text: "What city?".into(),
                    metadata: None,
                })],
                message_id: "q1".into(),
                task_id: None,
                context_id: None,
                metadata: None,
            }),
            timestamp: None,
        };
        let messages = task_to_chat_messages(&task).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "What city?");
    }

    #[test]
    fn task_to_chat_messages_input_required_without_status_message_falls_back() {
        let mut task = completed_task(vec![artifact("art-1", "partial")]);
        task.status = crate::types::TaskStatus {
            state: TaskState::InputRequired,
            message: None,
            timestamp: None,
        };
        let messages = task_to_chat_messages(&task).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "partial");
    }

    #[test]
    fn task_to_chat_messages_empty_when_nothing_available() {
        let task = completed_task(vec![]);
        let messages = task_to_chat_messages(&task).unwrap();
        assert!(messages.is_empty());
    }

    // -- ThreadState --------------------------------------------------

    // -- Streaming event -> update mapping (Agent::run_stream override) -----

    #[test]
    fn stream_event_message_maps_to_update_and_tracks_ids() {
        let mut state = ThreadState::default();
        let ev = MessageStreamEvent::Message(A2AMessage {
            kind: "message".into(),
            role: MessageRole::Agent,
            parts: vec![Part::Text(TextPart {
                text: "hi".into(),
                metadata: None,
            })],
            message_id: "m1".into(),
            task_id: Some("t1".into()),
            context_id: Some("c1".into()),
            metadata: None,
        });
        let updates = stream_event_to_updates(&ev, &mut state, Some("weather"));
        assert_eq!(updates.len(), 1);
        let u = updates.into_iter().next().unwrap().unwrap();
        assert_eq!(u.text(), "hi");
        assert_eq!(u.author_name.as_deref(), Some("weather"));
        // Same rule as non-streaming run(): a bare message exposes its
        // message id as the response id.
        assert_eq!(u.response_id.as_deref(), Some("m1"));
        // Continuity tracked from the event.
        assert_eq!(state.context_id.as_deref(), Some("c1"));
        assert_eq!(state.task_id.as_deref(), Some("t1"));
    }

    #[test]
    fn stream_event_bare_message_clears_a_stale_task_id() {
        // Matches run()'s SendMessageResult::Message mapping: a standalone
        // message without a task id means no active task — a previous
        // (completed) task's id must not leak into the next outgoing message.
        let mut state = ThreadState {
            context_id: Some("c1".into()),
            task_id: Some("finished-task".into()),
        };
        let ev = MessageStreamEvent::Message(A2AMessage {
            kind: "message".into(),
            role: MessageRole::Agent,
            parts: vec![Part::Text(TextPart {
                text: "standalone".into(),
                metadata: None,
            })],
            message_id: "m2".into(),
            task_id: None,
            context_id: None,
            metadata: None,
        });
        let updates = stream_event_to_updates(&ev, &mut state, None);
        assert_eq!(updates.len(), 1);
        assert!(state.task_id.is_none());
        // Context continuity keeps the previous id when absent (run() parity).
        assert_eq!(state.context_id.as_deref(), Some("c1"));
    }

    #[test]
    fn stream_event_status_update_surfaces_message_and_tracks_ids() {
        let mut state = ThreadState::default();
        let ev = MessageStreamEvent::StatusUpdate(crate::types::TaskStatusUpdateEvent {
            task_id: "t2".into(),
            context_id: "c2".into(),
            status: crate::types::TaskStatus {
                state: TaskState::InputRequired,
                message: Some(A2AMessage {
                    kind: "message".into(),
                    role: MessageRole::Agent,
                    parts: vec![Part::Text(TextPart {
                        text: "What city?".into(),
                        metadata: None,
                    })],
                    message_id: "q1".into(),
                    task_id: None,
                    context_id: None,
                    metadata: None,
                }),
                timestamp: None,
            },
            is_final: false,
            metadata: None,
        });
        let updates = stream_event_to_updates(&ev, &mut state, None);
        assert_eq!(updates.len(), 1);
        let u = updates.into_iter().next().unwrap().unwrap();
        assert_eq!(u.text(), "What city?");
        // Task-bearing events expose the task id as the response id, so a
        // streaming client can cancel/poll the task after aggregation.
        assert_eq!(u.response_id.as_deref(), Some("t2"));
        assert_eq!(state.context_id.as_deref(), Some("c2"));
        assert_eq!(state.task_id.as_deref(), Some("t2"));
    }

    #[test]
    fn stream_event_status_update_without_message_yields_no_update_but_tracks_ids() {
        let mut state = ThreadState::default();
        let ev = MessageStreamEvent::StatusUpdate(crate::types::TaskStatusUpdateEvent {
            task_id: "t3".into(),
            context_id: "c3".into(),
            status: crate::types::TaskStatus {
                state: TaskState::Working,
                message: None,
                timestamp: None,
            },
            is_final: false,
            metadata: None,
        });
        let updates = stream_event_to_updates(&ev, &mut state, None);
        assert!(updates.is_empty(), "no status message → no update");
        assert_eq!(state.task_id.as_deref(), Some("t3"));
    }

    #[test]
    fn stream_event_task_updates_carry_the_task_id_as_response_id() {
        let mut state = ThreadState::default();
        let ev = MessageStreamEvent::Task(crate::types::Task {
            id: "task-42".into(),
            context_id: "c9".into(),
            status: crate::types::TaskStatus {
                state: TaskState::Completed,
                message: None,
                timestamp: None,
            },
            history: Some(vec![A2AMessage {
                kind: "message".into(),
                role: MessageRole::Agent,
                parts: vec![Part::Text(TextPart {
                    text: "done".into(),
                    metadata: None,
                })],
                message_id: "m9".into(),
                task_id: Some("task-42".into()),
                context_id: Some("c9".into()),
                metadata: None,
            }]),
            artifacts: None,
            metadata: None,
        });
        let updates = stream_event_to_updates(&ev, &mut state, None);
        assert!(!updates.is_empty());
        for u in updates {
            let u = u.unwrap();
            // Mirrors run(): a Task result's response_id is the task id.
            assert_eq!(u.response_id.as_deref(), Some("task-42"));
        }
        assert_eq!(state.task_id.as_deref(), Some("task-42"));
        assert_eq!(state.context_id.as_deref(), Some("c9"));
    }

    #[test]
    fn thread_state_encode_decode_round_trip() {
        let state = ThreadState {
            context_id: Some("ctx-1".into()),
            task_id: Some("task-1".into()),
        };
        let encoded = state.encode().unwrap();
        let decoded = ThreadState::decode(Some(&encoded));
        assert_eq!(decoded, state);
    }

    #[test]
    fn thread_state_encode_is_none_when_empty() {
        assert!(ThreadState::default().encode().is_none());
    }

    #[test]
    fn thread_state_decode_defaults_on_missing_or_garbage() {
        assert_eq!(ThreadState::decode(None), ThreadState::default());
        assert_eq!(
            ThreadState::decode(Some("not json")),
            ThreadState::default()
        );
        assert_eq!(
            ThreadState::decode(Some("some-unrelated-conversation-id")),
            ThreadState::default()
        );
    }

    // -- Agent trait plumbing -------------------------------------------

    #[test]
    fn from_url_sets_name_and_random_id() {
        let agent = A2AAgent::from_url("weather", "https://weather.example.com/rpc");
        assert_eq!(agent.name(), Some("weather"));
        assert!(!agent.id().is_empty());
    }

    #[test]
    fn with_id_and_with_description_override_defaults() {
        let agent = A2AAgent::from_url("weather", "https://weather.example.com/rpc")
            .with_id("fixed-id")
            .with_description("Gives forecasts");
        assert_eq!(agent.id(), "fixed-id");
        assert_eq!(agent.description(), Some("Gives forecasts"));
    }

    #[test]
    fn display_name_falls_back_to_id_when_unset() {
        // A2AAgent's constructors always set a name, but the trait default
        // for `display_name` should still prefer it over `id`.
        let agent = A2AAgent::from_url("weather", "https://weather.example.com/rpc");
        assert_eq!(agent.display_name(), "weather");
    }
}
