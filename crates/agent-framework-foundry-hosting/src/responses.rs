//! `ResponsesHostServer`: an agent or a workflow on the agentserver's
//! Responses routes (upstream `ResponsesHostServer` over
//! `ResponsesAgentServerHost`).
//!
//! | Route | |
//! | --- | --- |
//! | `POST {prefix}/responses` | create (sync, `stream`, `background`) |
//! | `GET {prefix}/responses/{id}` | a stored response |
//! | `DELETE {prefix}/responses/{id}` | delete it |
//! | `POST {prefix}/responses/{id}/cancel` | cancel a background response |
//! | `GET {prefix}/responses/{id}/input_items` | the request's input items |
//!
//! History follows upstream's default `history_source="agent_server"`: the
//! host keeps each stored response's transcript and sends the prior
//! transcript (from `previous_response_id` or `conversation`) followed by
//! the new input to the model. `store: false` persists nothing.
//!
//! Streams are `event: <type>` / `data: <json>` SSE frames with a
//! `sequence_number`: `response.created`, `response.in_progress`, the
//! message's `output_item.added` / `content_part.added` /
//! `output_text.delta`… / `output_text.done` / `content_part.done` /
//! `output_item.done`, one `output_item.added`/`done` pair per declared
//! function call, and `response.completed` (or `response.failed`).

use std::collections::HashMap;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::error::Result;
use agent_framework_core::types::{AgentResponse, ChatOptions, Content, Message};
use agent_framework_core::workflow::Workflow;
use agent_framework_hosting::responses::openai_error;
use agent_framework_hosting::{responses_from_run, responses_to_run, ResponsesRequest};
use async_trait::async_trait;
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::FoundryHostConfig;
use crate::host::{with_host_routes, HostOptions};
use crate::scope::{ids, FoundryRequestContext};

/// A turn's request, as the hosted agent or workflow sees it (upstream
/// `HostedResponseRequest`).
#[derive(Debug, Clone)]
pub struct HostedResponseRequest {
    /// The new response's id.
    pub response_id: String,
    /// The raw `POST /responses` body.
    pub body: Value,
    /// This request's input as messages.
    pub messages: Vec<Message>,
    /// The prior transcript (from `previous_response_id` or
    /// `conversation`).
    pub history: Vec<Message>,
    /// `previous_response_id`.
    pub previous_response_id: Option<String>,
    /// The conversation id.
    pub conversation_id: Option<String>,
    /// The platform request context.
    pub context: FoundryRequestContext,
}

/// A stored response.
#[derive(Debug, Clone)]
pub struct StoredResponse {
    /// The response object.
    pub response: Value,
    /// The request's input items.
    pub input_items: Vec<Value>,
    /// The transcript up to and including this response's output.
    pub transcript: Vec<Message>,
}

/// Where responses (and conversation transcripts) are kept.
#[async_trait]
pub trait ResponseStore: Send + Sync {
    /// The response stored under `id`.
    async fn get(&self, id: &str) -> Result<Option<StoredResponse>>;
    /// Store (or replace) a response.
    async fn put(&self, id: &str, response: StoredResponse) -> Result<()>;
    /// Delete a response; whether it existed.
    async fn delete(&self, id: &str) -> Result<bool>;
    /// A conversation's transcript.
    async fn conversation(&self, id: &str) -> Result<Option<Vec<Message>>>;
    /// Replace a conversation's transcript.
    async fn set_conversation(&self, id: &str, transcript: Vec<Message>) -> Result<()>;
}

/// A process-local [`ResponseStore`].
#[derive(Debug, Clone, Default)]
pub struct InMemoryResponseStore {
    responses: Arc<Mutex<HashMap<String, StoredResponse>>>,
    conversations: Arc<Mutex<HashMap<String, Vec<Message>>>>,
}

#[async_trait]
impl ResponseStore for InMemoryResponseStore {
    async fn get(&self, id: &str) -> Result<Option<StoredResponse>> {
        Ok(self.responses.lock().unwrap().get(id).cloned())
    }
    async fn put(&self, id: &str, response: StoredResponse) -> Result<()> {
        self.responses
            .lock()
            .unwrap()
            .insert(id.to_string(), response);
        Ok(())
    }
    async fn delete(&self, id: &str) -> Result<bool> {
        Ok(self.responses.lock().unwrap().remove(id).is_some())
    }
    async fn conversation(&self, id: &str) -> Result<Option<Vec<Message>>> {
        Ok(self.conversations.lock().unwrap().get(id).cloned())
    }
    async fn set_conversation(&self, id: &str, transcript: Vec<Message>) -> Result<()> {
        self.conversations
            .lock()
            .unwrap()
            .insert(id.to_string(), transcript);
        Ok(())
    }
}

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Creates the agent for one request.
pub type AgentFactory = Arc<dyn Fn() -> BoxFuture<Result<Arc<dyn SupportsAgentRun>>> + Send + Sync>;
/// Creates a fresh workflow for one request.
pub type WorkflowFactory = Arc<dyn Fn(&HostedResponseRequest) -> Result<Workflow> + Send + Sync>;
/// Maps a request to a workflow's input.
pub type WorkflowInputParser = Arc<dyn Fn(&HostedResponseRequest) -> Result<Value> + Send + Sync>;

#[derive(Clone)]
enum Target {
    Agent(Arc<dyn SupportsAgentRun>),
    AgentFactory(AgentFactory),
    Workflow {
        factory: WorkflowFactory,
        parse: Option<WorkflowInputParser>,
    },
}

/// Serves an agent or workflow on the Responses routes. See the
/// [module docs](self).
#[derive(Clone)]
pub struct ResponsesHostServer {
    target: Target,
    config: FoundryHostConfig,
    store: Arc<dyn ResponseStore>,
    prefix: String,
    host: HostOptions,
}

impl std::fmt::Debug for ResponsesHostServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResponsesHostServer")
            .field("config", &self.config)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl ResponsesHostServer {
    fn new(target: Target) -> Self {
        Self {
            target,
            config: FoundryHostConfig::local(),
            store: Arc::new(InMemoryResponseStore::default()),
            prefix: String::new(),
            host: HostOptions::default(),
        }
    }

    /// Serve one agent instance for every request. The agent must keep no
    /// per-conversation state outside its session; otherwise use
    /// [`Self::agent_factory`].
    pub fn agent(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self::new(Target::Agent(agent))
    }

    /// Create an agent per request.
    pub fn agent_factory<F, Fut>(factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Arc<dyn SupportsAgentRun>>> + Send + 'static,
    {
        Self::new(Target::AgentFactory(Arc::new(move || Box::pin(factory()))))
    }

    /// Serve a workflow, built fresh per request. Without
    /// [`Self::with_workflow_input`] the workflow's input is the turn's
    /// messages (prior transcript followed by the new input) as JSON.
    pub fn workflow(
        factory: impl Fn(&HostedResponseRequest) -> Result<Workflow> + Send + Sync + 'static,
    ) -> Self {
        Self::new(Target::Workflow {
            factory: Arc::new(factory),
            parse: None,
        })
    }

    /// Map each request to the workflow's input (upstream `parse_response`).
    pub fn with_workflow_input(
        mut self,
        parse: impl Fn(&HostedResponseRequest) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        if let Target::Workflow { parse: slot, .. } = &mut self.target {
            *slot = Some(Arc::new(parse));
        }
        self
    }

    /// The host configuration (default [`FoundryHostConfig::local`]).
    pub fn with_config(mut self, config: FoundryHostConfig) -> Self {
        self.config = config;
        self
    }

    /// Where responses are stored (default in memory).
    pub fn with_response_store(mut self, store: Arc<dyn ResponseStore>) -> Self {
        self.store = store;
        self
    }

    /// A route prefix (e.g. `/v1`).
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into().trim_end_matches('/').to_string();
        self
    }

    /// Lifecycle hooks and server-version segments for the shared routes.
    pub fn with_host_options(mut self, options: HostOptions) -> Self {
        self.host = options;
        self
    }

    /// The router: the Responses routes plus the shared host routes.
    pub fn into_router(self) -> Router {
        let p = self.prefix.clone();
        let host = self.host.clone().server_version_segment("responses");
        let state = Arc::new(Inner {
            server: self,
            running: Mutex::new(HashMap::new()),
        });
        let router = Router::new()
            .route(&format!("{p}/responses"), post(create))
            .route(
                &format!("{p}/responses/{{id}}"),
                get(get_response).delete(delete_response),
            )
            .route(&format!("{p}/responses/{{id}}/cancel"), post(cancel))
            .route(
                &format!("{p}/responses/{{id}}/input_items"),
                get(input_items),
            )
            .with_state(state);
        with_host_routes(router, host)
    }

    /// Serve on the configured port.
    pub async fn serve(self) -> Result<()> {
        let config = self.config.clone();
        crate::host::serve(self.into_router(), &config).await
    }
}

struct Inner {
    server: ResponsesHostServer,
    running: Mutex<HashMap<String, tokio::task::AbortHandle>>,
}

/// The `POST /responses` body fields the host reads.
#[derive(Debug, Clone, Deserialize, Default)]
struct CreateResponse {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    input: Value,
    #[serde(default)]
    instructions: Option<String>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    store: Option<bool>,
    #[serde(default)]
    background: bool,
    #[serde(default)]
    previous_response_id: Option<String>,
    #[serde(default)]
    conversation: Option<Value>,
    #[serde(default)]
    metadata: Option<Map<String, Value>>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_output_tokens: Option<u32>,
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(openai_error(message, "invalid_request_error", None)),
    )
        .into_response()
}

fn not_found(id: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(openai_error(
            format!("Response with id '{id}' not found."),
            "invalid_request_error",
            Some("not_found"),
        )),
    )
        .into_response()
}

fn server_error(message: impl Into<String>) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(openai_error(message, "server_error", None)),
    )
        .into_response()
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

/// Everything a turn needs, resolved before it runs.
#[derive(Clone)]
struct Prepared {
    request: HostedResponseRequest,
    create: CreateResponse,
    store: bool,
    model: String,
    input_items: Vec<Value>,
    created_at: u64,
}

impl Prepared {
    /// The response object's common fields.
    fn decorate(&self, mut response: Value, status: &str) -> Value {
        let id = &self.request.response_id;
        response["id"] = json!(id);
        response["object"] = json!("response");
        response["created_at"] = json!(self.created_at);
        response["model"] = json!(self.model);
        response["status"] = json!(status);
        response["background"] = json!(self.create.background);
        response["store"] = json!(self.store);
        response["previous_response_id"] = json!(self.request.previous_response_id);
        response["conversation"] = match &self.request.conversation_id {
            Some(c) => json!({"id": c}),
            None => Value::Null,
        };
        response["instructions"] = json!(self.create.instructions);
        response["metadata"] = json!(self.create.metadata.clone().unwrap_or_default());
        if response.get("output").is_none() {
            response["output"] = json!([]);
        }
        response
    }

    fn skeleton(&self, status: &str) -> Value {
        self.decorate(json!({}), status)
    }
}

/// Normalize the request input into input items with ids.
fn input_items_of(input: &Value, response_id: &str) -> Vec<Value> {
    let items: Vec<Value> = match input {
        Value::String(text) => vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        })],
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    };
    items
        .into_iter()
        .map(|mut item| {
            if item.is_object() && item.get("id").is_none() {
                let prefix = match item.get("type").and_then(Value::as_str) {
                    Some("function_call_output") => "fco",
                    _ => "msg",
                };
                item["id"] = json!(ids::new_id(prefix, Some(response_id)));
            }
            item
        })
        .collect()
}

fn conversation_id(value: &Option<Value>) -> std::result::Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(Value::Object(o)) => match o.get("id").and_then(Value::as_str) {
            Some(id) if !id.is_empty() => Ok(Some(id.to_string())),
            _ => Err("conversation.id must be a non-empty string.".into()),
        },
        _ => Err("conversation must be a string or an object with an id.".into()),
    }
}

async fn create(
    State(inner): State<Arc<Inner>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let body: Value = match serde_json::from_slice(&body) {
        Ok(v @ Value::Object(_)) => v,
        _ => return bad_request("The request body must be a JSON object."),
    };
    let create: CreateResponse = match serde_json::from_value(body.clone()) {
        Ok(c) => c,
        Err(e) => return bad_request(format!("Invalid request: {e}")),
    };
    let store = create.store.unwrap_or(true);
    if create.background && !store {
        return bad_request("background responses require store=true.");
    }
    let conversation = match conversation_id(&create.conversation) {
        Ok(c) => c,
        Err(e) => return bad_request(e),
    };
    if conversation.is_some() && create.previous_response_id.is_some() {
        return bad_request("previous_response_id and conversation cannot both be set.");
    }
    let server = &inner.server;
    let history = if let Some(previous) = &create.previous_response_id {
        match server.store.get(previous).await {
            Ok(Some(stored)) => stored.transcript,
            Ok(None) => {
                return bad_request(format!("Previous response with id '{previous}' not found."))
            }
            Err(e) => return server_error(e.to_string()),
        }
    } else if let Some(conversation) = &conversation {
        match server.store.conversation(conversation).await {
            Ok(t) => t.unwrap_or_default(),
            Err(e) => return server_error(e.to_string()),
        }
    } else {
        Vec::new()
    };
    let response_id = ids::response_id(create.previous_response_id.as_deref());
    let messages = responses_to_run(&ResponsesRequest {
        input: create.input.clone(),
        ..Default::default()
    });
    let prepared = Prepared {
        request: HostedResponseRequest {
            response_id: response_id.clone(),
            body,
            messages,
            history,
            previous_response_id: create.previous_response_id.clone(),
            conversation_id: conversation,
            context: FoundryRequestContext::from_headers(&headers, None),
        },
        input_items: input_items_of(&create.input, &response_id),
        model: create
            .model
            .clone()
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| {
                if server.config.agent_name.is_empty() {
                    "agent".into()
                } else {
                    server.config.agent_name.clone()
                }
            }),
        store,
        created_at: now_secs(),
        create,
    };

    if prepared.create.background {
        return start_background(inner, prepared).await;
    }
    if prepared.create.stream {
        let (tx, rx) = mpsc::channel::<Event>(256);
        let task_inner = inner.clone();
        let task = tokio::spawn(async move {
            let mut events = Events::new(tx);
            run_turn(&task_inner, &prepared, Some(&mut events)).await;
        });
        return sse(&inner.server.config, rx, Some(task.abort_handle()));
    }
    let (status, response) = run_turn(&inner, &prepared, None).await;
    if status == "failed" {
        let message = response
            .pointer("/error/message")
            .and_then(Value::as_str)
            .unwrap_or("The agent run failed.")
            .to_string();
        return server_error(message);
    }
    Json(response).into_response()
}

/// Aborts a request's task when its stream is dropped (the client went
/// away), unless the turn is a background one.
struct AbortOnDrop(Option<tokio::task::AbortHandle>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            handle.abort();
        }
    }
}

fn sse(
    config: &FoundryHostConfig,
    rx: mpsc::Receiver<Event>,
    abort: Option<tokio::task::AbortHandle>,
) -> Response {
    let guard = AbortOnDrop(abort);
    let stream = ReceiverStream::new(rx).map(move |event| {
        let _ = &guard;
        Ok::<_, Infallible>(event)
    });
    let sse = Sse::new(stream);
    if config.sse_keepalive_interval > 0 {
        sse.keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(config.sse_keepalive_interval))
                .text("keep-alive"),
        )
        .into_response()
    } else {
        sse.into_response()
    }
}

async fn start_background(inner: Arc<Inner>, prepared: Prepared) -> Response {
    let id = prepared.request.response_id.clone();
    let queued = prepared.skeleton("queued");
    let stored = StoredResponse {
        response: queued.clone(),
        input_items: prepared.input_items.clone(),
        transcript: Vec::new(),
    };
    if let Err(e) = inner.server.store.put(&id, stored).await {
        return server_error(e.to_string());
    }
    let stream = prepared.create.stream;
    let (tx, rx) = mpsc::channel::<Event>(256);
    let task_inner = inner.clone();
    let task_id = id.clone();
    let task = tokio::spawn(async move {
        let mut events = Events::new(tx);
        run_turn(&task_inner, &prepared, stream.then_some(&mut events)).await;
        task_inner.running.lock().unwrap().remove(&task_id);
    });
    inner
        .running
        .lock()
        .unwrap()
        .insert(id, task.abort_handle());
    if stream {
        // The run continues if the client disconnects.
        sse(&inner.server.config, rx, None)
    } else {
        Json(queued).into_response()
    }
}

/// Sequenced SSE events for one response.
struct Events {
    tx: mpsc::Sender<Event>,
    sequence: u64,
}

impl Events {
    fn new(tx: mpsc::Sender<Event>) -> Self {
        Self { tx, sequence: 0 }
    }

    async fn emit(&mut self, kind: &str, mut payload: Value) {
        payload["type"] = json!(kind);
        payload["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        let event = Event::default().event(kind).data(payload.to_string());
        // A disconnected client does not stop a background run.
        let _ = self.tx.send(event).await;
    }
}

/// The streamed assistant message, opened on the first text delta.
struct OpenMessage {
    id: String,
    text: String,
}

/// Run one turn, emitting stream events when `events` is set; store the
/// outcome when `store` is on. Returns the final status and response.
async fn run_turn(
    inner: &Arc<Inner>,
    prepared: &Prepared,
    mut events: Option<&mut Events>,
) -> (String, Value) {
    let server = &inner.server;
    if let Some(events) = events.as_deref_mut() {
        let skeleton = prepared.skeleton("in_progress");
        events
            .emit(
                "response.created",
                json!({"response": prepared.skeleton("queued")}),
            )
            .await;
        events
            .emit("response.in_progress", json!({"response": skeleton}))
            .await;
    } else if prepared.create.background {
        let _ = update_status(server, prepared, "in_progress").await;
    }
    let mut open: Option<OpenMessage> = None;
    let result = execute(server, prepared, events.as_deref_mut(), &mut open).await;
    let (status, response, transcript) = match result {
        Ok(agent_response) => {
            let response = final_response(
                prepared,
                &agent_response,
                open.as_ref().map(|m| m.id.as_str()),
            );
            let status = response["status"]
                .as_str()
                .unwrap_or("completed")
                .to_string();
            let mut transcript = prepared.request.history.clone();
            transcript.extend(prepared.request.messages.iter().cloned());
            transcript.extend(agent_response.messages.iter().cloned());
            (status, response, transcript)
        }
        Err(e) => {
            tracing::error!(response_id = %prepared.request.response_id, error = %e, "hosted response failed");
            let mut response = prepared.skeleton("failed");
            response["error"] = json!({"code": "server_error", "message": "The agent run failed."});
            ("failed".to_string(), response, Vec::new())
        }
    };
    if let Some(events) = events {
        if let Some(message) = open.take() {
            close_message(events, &message).await;
        }
        if status != "failed" {
            for (index, item) in response["output"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                if item["type"] == "function_call" {
                    events
                        .emit(
                            "response.output_item.added",
                            json!({"output_index": index, "item": item}),
                        )
                        .await;
                    events
                        .emit(
                            "response.function_call_arguments.done",
                            json!({
                                "output_index": index,
                                "item_id": item["id"],
                                "arguments": item["arguments"],
                            }),
                        )
                        .await;
                    events
                        .emit(
                            "response.output_item.done",
                            json!({"output_index": index, "item": item}),
                        )
                        .await;
                }
            }
        }
        let kind = match status.as_str() {
            "failed" => "response.failed",
            "incomplete" => "response.incomplete",
            _ => "response.completed",
        };
        events.emit(kind, json!({"response": response})).await;
    }
    if prepared.store {
        let stored = StoredResponse {
            response: response.clone(),
            input_items: prepared.input_items.clone(),
            transcript: transcript.clone(),
        };
        if let Err(e) = server
            .store
            .put(&prepared.request.response_id, stored)
            .await
        {
            tracing::error!(error = %e, "failed to store a hosted response");
        }
        if let (Some(conversation), false) = (&prepared.request.conversation_id, status == "failed")
        {
            if let Err(e) = server
                .store
                .set_conversation(conversation, transcript)
                .await
            {
                tracing::error!(error = %e, "failed to store a conversation transcript");
            }
        }
    }
    (status, response)
}

async fn update_status(
    server: &ResponsesHostServer,
    prepared: &Prepared,
    status: &str,
) -> Result<()> {
    let id = &prepared.request.response_id;
    if let Some(mut stored) = server.store.get(id).await? {
        stored.response["status"] = json!(status);
        server.store.put(id, stored).await?;
    }
    Ok(())
}

async fn close_message(events: &mut Events, message: &OpenMessage) {
    let part = json!({"type": "output_text", "text": message.text, "annotations": []});
    events
        .emit(
            "response.output_text.done",
            json!({
                "item_id": message.id, "output_index": 0, "content_index": 0, "text": message.text,
            }),
        )
        .await;
    events
        .emit(
            "response.content_part.done",
            json!({
                "item_id": message.id, "output_index": 0, "content_index": 0, "part": part,
            }),
        )
        .await;
    events
        .emit(
            "response.output_item.done",
            json!({
                "output_index": 0,
                "item": {
                    "id": message.id, "type": "message", "status": "completed", "role": "assistant",
                    "content": [part],
                },
            }),
        )
        .await;
}

async fn text_delta(
    events: &mut Events,
    open: &mut Option<OpenMessage>,
    response_id: &str,
    delta: &str,
) {
    if delta.is_empty() {
        return;
    }
    if open.is_none() {
        let id = ids::message_id(Some(response_id));
        events
            .emit("response.output_item.added", json!({
                "output_index": 0,
                "item": {"id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": []},
            }))
            .await;
        events
            .emit(
                "response.content_part.added",
                json!({
                    "item_id": id, "output_index": 0, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }),
            )
            .await;
        *open = Some(OpenMessage {
            id,
            text: String::new(),
        });
    }
    if let Some(message) = open.as_mut() {
        message.text.push_str(delta);
        events
            .emit(
                "response.output_text.delta",
                json!({
                    "item_id": message.id, "output_index": 0, "content_index": 0, "delta": delta,
                }),
            )
            .await;
    }
}

fn run_options(create: &CreateResponse) -> AgentRunOptions {
    let mut options = ChatOptions::new();
    options.temperature = create.temperature;
    options.top_p = create.top_p;
    options.max_tokens = create.max_output_tokens;
    options.instructions = create.instructions.clone();
    AgentRunOptions::new().with_chat_options(options)
}

async fn execute(
    server: &ResponsesHostServer,
    prepared: &Prepared,
    events: Option<&mut Events>,
    open: &mut Option<OpenMessage>,
) -> Result<AgentResponse> {
    let request = &prepared.request;
    let agent = match &server.target {
        Target::Agent(agent) => agent.clone(),
        Target::AgentFactory(factory) => factory().await?,
        Target::Workflow { factory, parse } => {
            let workflow = factory(request)?;
            let input = match parse {
                Some(parse) => parse(request)?,
                None => {
                    let mut all = request.history.clone();
                    all.extend(request.messages.iter().cloned());
                    serde_json::to_value(all)?
                }
            };
            let run = workflow.run(input).await?;
            let text = run
                .outputs()
                .iter()
                .map(output_text)
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if let Some(events) = events {
                text_delta(events, open, &request.response_id, &text).await;
            }
            return Ok(AgentResponse {
                messages: vec![Message::assistant(text)],
                ..Default::default()
            });
        }
    };
    let mut messages = request.history.clone();
    messages.extend(request.messages.iter().cloned());
    let options = run_options(&prepared.create);
    let session = agent.create_session();
    match events {
        None => {
            let mut session = session;
            agent
                .run_with_options(messages, Some(&mut session), options)
                .await
        }
        Some(events) => {
            let mut stream = agent
                .run_stream(messages, Some(session), Some(options))
                .await?;
            let mut updates = Vec::new();
            while let Some(update) = stream.next().await {
                let update = update?;
                let delta: String = update
                    .contents
                    .iter()
                    .filter_map(|c| match c {
                        Content::Text(t) => Some(t.text.as_str()),
                        _ => None,
                    })
                    .collect();
                text_delta(events, open, &request.response_id, &delta).await;
                updates.push(update);
            }
            Ok(AgentResponse::from_updates(updates))
        }
    }
}

/// A workflow output as text.
fn output_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => {
            if let Ok(messages) = serde_json::from_value::<Vec<Message>>(other.clone()) {
                return messages
                    .iter()
                    .filter(|m| m.role.to_string() == "assistant")
                    .map(Message::text)
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            if let Ok(message) = serde_json::from_value::<Message>(other.clone()) {
                return message.text();
            }
            other.to_string()
        }
    }
}

/// The final response object, with agentserver ids; the streamed message
/// keeps the id it was streamed under.
fn final_response(
    prepared: &Prepared,
    response: &AgentResponse,
    streamed_message: Option<&str>,
) -> Value {
    let id = &prepared.request.response_id;
    let object = responses_from_run(response, id, &prepared.model);
    let status = object.status;
    let mut value = serde_json::to_value(&object).unwrap_or_else(|_| json!({}));
    if let Some(items) = value.get_mut("output").and_then(Value::as_array_mut) {
        let mut first_message = true;
        for item in items.iter_mut() {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => {
                    item["id"] = json!(match (first_message, streamed_message) {
                        (true, Some(streamed)) => streamed.to_string(),
                        _ => ids::message_id(Some(id)),
                    });
                    first_message = false;
                }
                Some("function_call") => item["id"] = json!(ids::function_call_id(Some(id))),
                _ => {}
            }
        }
    }
    prepared.decorate(value, status)
}

async fn get_response(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    match inner.server.store.get(&id).await {
        Ok(Some(stored)) => Json(stored.response).into_response(),
        Ok(None) => not_found(&id),
        Err(e) => server_error(e.to_string()),
    }
}

async fn delete_response(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    if let Some(handle) = inner.running.lock().unwrap().remove(&id) {
        handle.abort();
    }
    match inner.server.store.delete(&id).await {
        Ok(true) => Json(json!({"id": id, "object": "response", "deleted": true})).into_response(),
        Ok(false) => not_found(&id),
        Err(e) => server_error(e.to_string()),
    }
}

async fn cancel(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    let stored = match inner.server.store.get(&id).await {
        Ok(Some(stored)) => stored,
        Ok(None) => return not_found(&id),
        Err(e) => return server_error(e.to_string()),
    };
    if stored.response["background"] != json!(true) {
        return bad_request("Only background responses can be cancelled.");
    }
    let handle = inner.running.lock().unwrap().remove(&id);
    let mut stored = stored;
    if let Some(handle) = handle {
        handle.abort();
        stored.response["status"] = json!("cancelled");
        if let Err(e) = inner.server.store.put(&id, stored.clone()).await {
            return server_error(e.to_string());
        }
    } else if stored.response["status"] != json!("cancelled") {
        return bad_request("Cannot cancel a response that has already finished.");
    }
    Json(stored.response).into_response()
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default)]
    order: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

async fn input_items(
    State(inner): State<Arc<Inner>>,
    Path(id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Response {
    let stored = match inner.server.store.get(&id).await {
        Ok(Some(stored)) => stored,
        Ok(None) => return not_found(&id),
        Err(e) => return server_error(e.to_string()),
    };
    let mut items = stored.input_items;
    if query.order.as_deref() != Some("asc") {
        items.reverse();
    }
    let limit = query.limit.unwrap_or(20).clamp(1, 100);
    let has_more = items.len() > limit;
    items.truncate(limit);
    Json(json!({
        "object": "list",
        "first_id": items.first().and_then(|i| i.get("id")).cloned(),
        "last_id": items.last().and_then(|i| i.get("id")).cloned(),
        "has_more": has_more,
        "data": items,
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_carry_and_reuse_partition_keys() {
        let first = ids::response_id(None);
        assert!(first.starts_with("caresp_"));
        assert_eq!(first.len(), "caresp_".len() + 50);
        let key = ids::partition_key(&first).unwrap();
        assert!(key.ends_with("00"));
        let next = ids::message_id(Some(&first));
        assert_eq!(ids::partition_key(&next).unwrap(), key);
    }

    #[test]
    fn string_input_becomes_a_user_message_item() {
        let items = input_items_of(&json!("hi"), "caresp_x");
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[0]["content"][0]["text"], "hi");
        assert!(items[0]["id"].as_str().unwrap().starts_with("msg_"));
    }
}
