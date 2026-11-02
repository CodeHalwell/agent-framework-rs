//! `ResponsesHostServer`: an agent or a workflow on the agentserver's
//! Responses routes (upstream `ResponsesHostServer` over
//! `ResponsesAgentServerHost`).
//!
//! | Route | |
//! | --- | --- |
//! | `POST {prefix}/responses` | create (sync, `stream`, `background`) |
//! | `GET {prefix}/responses/{id}` | a stored response; `?stream=true[&starting_after=N]` replays a background response's events (live while it runs) |
//! | `DELETE {prefix}/responses/{id}` | delete it |
//! | `POST {prefix}/responses/{id}/cancel` | cancel a background response |
//! | `GET {prefix}/responses/{id}/input_items` | the request's input items |
//!
//! History follows upstream's default `history_source="agent_server"`: the
//! host keeps each stored response's transcript and sends the prior
//! transcript (from `previous_response_id` or `conversation`) followed by
//! the new input to the model. `store: false` persists nothing.
//!
//! [`HistorySource`] selects who supplies prior messages (the default
//! replays the stored transcript; `Agent` / `Service` resume a host-kept
//! agent session). Turns on one conversation are serialized.
//!
//! Streams are `event: <type>` / `data: <json>` SSE frames with a
//! `sequence_number`: `response.created`, `response.in_progress`, each
//! output item's events in order (`output_item.added`, its content or
//! argument deltas, `output_item.done`; message text, refusals, reasoning
//! summaries, function and MCP calls, tool outputs, images, approval and
//! OAuth consent requests), and `response.completed` (or `incomplete` /
//! `failed`), emitted only after the response is persisted. Sync and
//! streamed responses carry the same `output` and `usage`.
//!
//! Workflows paused on `request_info` (with a
//! [checkpoint store](ResponsesHostServer::with_checkpoint_store)) end with
//! `request_info` function calls; answering them with
//! `function_call_output` items resumes the workflow from its checkpoint.

use std::collections::HashMap;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::error::Result;
use agent_framework_core::types::Content;
use agent_framework_core::types::{AgentResponse, ChatOptions, Message};
use agent_framework_core::workflow::{CheckpointStorage, Workflow};
use agent_framework_hosting::responses::openai_error;
use agent_framework_hosting::{responses_to_run, ResponsesRequest};
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
use crate::output::{OutputError, OutputTracker};
use crate::scope::{ids, FoundryRequestContext};
use crate::stores::{
    FunctionApprovalStore, FunctionApprovalStoreProvider, InMemoryFunctionApprovalStore,
};

/// A turn's request, as the hosted agent or workflow sees it (upstream
/// `HostedResponseRequest`).
#[derive(Clone)]
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
    /// The server's checkpoint store ([`ResponsesHostServer::with_checkpoint_store`]).
    /// A workflow factory passes it to
    /// [`WorkflowBuilder::with_checkpointing`](agent_framework_core::workflow::WorkflowBuilder::with_checkpointing)
    /// so a `request_info` pause can be resumed on a later turn.
    pub checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
}

impl std::fmt::Debug for HostedResponseRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostedResponseRequest")
            .field("response_id", &self.response_id)
            .field("previous_response_id", &self.previous_response_id)
            .field("conversation_id", &self.conversation_id)
            .field("checkpoint_storage", &self.checkpoint_storage.is_some())
            .finish_non_exhaustive()
    }
}

/// Who supplies prior messages to the model (upstream `history_source`),
/// independently of the caller's `store` flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistorySource {
    /// The host replays the stored outer transcript before this turn's
    /// input, and runs the agent with `store: false` so a storing client
    /// does not keep a second copy.
    #[default]
    AgentServer,
    /// Only this turn's input is sent; the agent's own session (history
    /// providers, or a service conversation it chose) supplies the rest.
    /// The session is kept by the host between stored turns.
    Agent,
    /// Only this turn's input is sent; on a stored turn the client runs with
    /// `store: true` and the service conversation it issues is kept in the
    /// host's session for the next turn. Agents only.
    Service,
}

/// Rewrites the caller's model options before an agent runs (upstream
/// `prepare_options`).
pub type OptionsHook =
    Arc<dyn Fn(&HostedResponseRequest, ChatOptions) -> Result<ChatOptions> + Send + Sync>;

/// A workflow paused on `request_info` requests at the end of a stored
/// response: the checkpoint to resume and the wire call ids standing for
/// the workflow's request ids.
#[derive(Debug, Clone, PartialEq)]
pub struct WorkflowPause {
    /// The checkpoint holding the pending requests.
    pub checkpoint_id: String,
    /// `(wire id, workflow request id)` pairs, in request order. The wire id
    /// is the `call_id` a `function_call_output` / `computer_call_output`
    /// answers, or the `approval_request_id` of an `mcp_approval_response`.
    pub requests: Vec<(String, String)>,
    /// The pending requests' data, in the same order (to build typed replies:
    /// approval responses, computer results).
    pub request_data: Vec<Value>,
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
    /// The workflow pause this response ended on, if any.
    pub pause: Option<WorkflowPause>,
    /// The agent session's state bag after this turn (`AgentSession::to_dict`
    /// without history), restored into a `HistorySource::AgentServer`
    /// continuation so session-scoped state (e.g. FIDES labels and hidden
    /// variables) carries over while the transcript is replayed.
    pub session_state: Option<Value>,
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
    /// A conversation's agent-session state (see
    /// [`StoredResponse::session_state`]). The default keeps none, so a
    /// store that does not override it loses session state across turns.
    async fn conversation_state(&self, _id: &str) -> Result<Option<Value>> {
        Ok(None)
    }
    /// Replace a conversation's agent-session state.
    async fn set_conversation_state(&self, _id: &str, _state: Value) -> Result<()> {
        Ok(())
    }
    /// Replace a conversation's transcript and session state together (the
    /// state is cleared when `state` is `None`). The host writes a turn's
    /// conversation through this one call, so a store should apply it
    /// atomically: a transcript paired with another turn's state would carry
    /// the wrong session-scoped state (e.g. FIDES labels) into the next turn.
    /// The default calls [`Self::set_conversation`] then
    /// [`Self::set_conversation_state`], which is not atomic; the host
    /// restores the previous transcript and state if it fails.
    async fn update_conversation(
        &self,
        id: &str,
        transcript: Vec<Message>,
        state: Option<Value>,
    ) -> Result<()> {
        self.set_conversation(id, transcript).await?;
        if let Some(state) = state {
            self.set_conversation_state(id, state).await?;
        }
        Ok(())
    }
}

/// A conversation's transcript and session state.
type ConversationRecord = (Vec<Message>, Option<Value>);

/// A process-local [`ResponseStore`].
#[derive(Debug, Clone, Default)]
pub struct InMemoryResponseStore {
    responses: Arc<Mutex<HashMap<String, StoredResponse>>>,
    conversations: Arc<Mutex<HashMap<String, ConversationRecord>>>,
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
        Ok(self
            .conversations
            .lock()
            .unwrap()
            .get(id)
            .map(|(transcript, _)| transcript.clone()))
    }
    async fn set_conversation(&self, id: &str, transcript: Vec<Message>) -> Result<()> {
        self.conversations
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_default()
            .0 = transcript;
        Ok(())
    }
    async fn conversation_state(&self, id: &str) -> Result<Option<Value>> {
        Ok(self
            .conversations
            .lock()
            .unwrap()
            .get(id)
            .and_then(|(_, state)| state.clone()))
    }
    async fn set_conversation_state(&self, id: &str, state: Value) -> Result<()> {
        self.conversations
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_default()
            .1 = Some(state);
        Ok(())
    }
    async fn update_conversation(
        &self,
        id: &str,
        transcript: Vec<Message>,
        state: Option<Value>,
    ) -> Result<()> {
        self.conversations
            .lock()
            .unwrap()
            .insert(id.to_string(), (transcript, state));
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
    consent_origins: Option<Vec<String>>,
    checkpoints: Option<Arc<dyn CheckpointStorage>>,
    history: HistorySource,
    prepare_options: Option<OptionsHook>,
    approval_provider: Option<Arc<dyn FunctionApprovalStoreProvider>>,
    local_approvals: Arc<InMemoryFunctionApprovalStore>,
    /// The agent sessions of the `Agent` / `Service` history sources: in a
    /// session store (with history, so they survive a restart) when the
    /// `session-store` feature is on, else kept live in process.
    #[cfg(feature = "session-store")]
    sessions: Arc<dyn crate::sessions::SessionStoreProvider>,
    #[cfg(not(feature = "session-store"))]
    sessions: Arc<Mutex<HashMap<String, SessionSnapshot>>>,
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
            consent_origins: None,
            checkpoints: None,
            history: HistorySource::default(),
            prepare_options: None,
            approval_provider: None,
            local_approvals: Arc::default(),
            #[cfg(feature = "session-store")]
            sessions: Arc::new(crate::sessions::InMemorySessionStoreProvider::default()),
            #[cfg(not(feature = "session-store"))]
            sessions: Arc::default(),
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

    /// Who supplies prior messages to the model (default
    /// [`HistorySource::AgentServer`]). With `Agent` or `Service`, the
    /// host keeps each stored turn's agent session (in process), keyed by
    /// the response id and the conversation, and resumes it on the next
    /// turn.
    pub fn with_history_source(mut self, source: HistorySource) -> Self {
        self.history = source;
        self
    }

    /// Rewrite the caller's model options before an agent runs (upstream
    /// `prepare_options`). Not applied to workflows.
    pub fn with_prepare_options(
        mut self,
        hook: impl Fn(&HostedResponseRequest, ChatOptions) -> Result<ChatOptions>
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.prepare_options = Some(Arc::new(hook));
        self
    }

    /// Where function-approval requests are kept between turns (e.g.
    /// [`FoundryStorage`](crate::FoundryStorage)); without one they are kept
    /// in process. A tool that requires approval surfaces as an
    /// `mcp_approval_request`; a later turn's `mcp_approval_response`
    /// (`approval_request_id`, `approve`) is resolved through this store into
    /// the agent's function-approval response, so the call runs or is
    /// rejected.
    pub fn with_function_approval_store(
        mut self,
        provider: Arc<dyn FunctionApprovalStoreProvider>,
    ) -> Self {
        self.approval_provider = Some(provider);
        self
    }

    /// Where the agent sessions of the `Agent` and `Service` history sources
    /// are kept (feature `session-store`): e.g. a
    /// [`FoundrySessionStoreProvider`](crate::sessions::FoundrySessionStoreProvider),
    /// so `previous_response_id` and `conversation` continuations, history
    /// included, survive a restart. The default is a process-local store.
    #[cfg(feature = "session-store")]
    pub fn with_session_store_provider(
        mut self,
        provider: Arc<dyn crate::sessions::SessionStoreProvider>,
    ) -> Self {
        self.sessions = provider;
        self
    }

    /// The session handle one turn loads from and saves to.
    fn host_sessions(&self, context: &FoundryRequestContext) -> Result<HostSessions> {
        #[cfg(feature = "session-store")]
        {
            Ok(HostSessions(
                self.sessions.session_store(&self.config, context)?,
                self.config.agent_name.clone(),
            ))
        }
        #[cfg(not(feature = "session-store"))]
        {
            let _ = context;
            Ok(HostSessions(self.sessions.clone()))
        }
    }

    fn approval_store(
        &self,
        context: &FoundryRequestContext,
    ) -> Result<Arc<dyn FunctionApprovalStore>> {
        match &self.approval_provider {
            Some(provider) => provider.approval_store(&self.config, context),
            None => Ok(self.local_approvals.clone()),
        }
    }

    /// Where workflow checkpoints are kept: it is handed to the workflow
    /// factory ([`HostedResponseRequest::checkpoint_storage`]) and, when a
    /// run pauses on `request_info`, its pending requests are returned as
    /// `request_info` function calls; answering them with
    /// `function_call_output` items (and `previous_response_id`) resumes the
    /// workflow from its checkpoint (upstream's checkpoint-backed workflow
    /// continuation).
    pub fn with_checkpoint_store(mut self, storage: Arc<dyn CheckpointStorage>) -> Self {
        self.checkpoints = Some(storage);
        self
    }

    /// Restrict OAuth consent links surfaced as `oauth_consent_request`
    /// items to these origins (`https://host[:port]`). Without it any safe
    /// absolute HTTPS link is allowed.
    pub fn with_allowed_oauth_consent_origins(
        mut self,
        origins: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.consent_origins = Some(origins.into_iter().map(Into::into).collect());
        self
    }

    /// Lifecycle hooks and server-version segments for the shared routes.
    pub fn with_host_options(mut self, options: HostOptions) -> Self {
        self.host = options;
        self
    }

    /// The router: the Responses routes plus the shared host routes.
    pub fn into_router(self) -> Router {
        let host = self
            .host
            .clone()
            .server_version_segment("responses")
            .agent_identity(
                self.config.agent_name.clone(),
                self.config.agent_version.clone(),
                self.config.session_id.clone(),
            );
        let (router, _) = self.routes(host);
        router
    }

    /// The router and its shared state.
    fn routes(self, host: HostOptions) -> (Router, Arc<Inner>) {
        let p = self.prefix.clone();
        let state = Arc::new(Inner {
            server: self,
            running: Mutex::new(HashMap::new()),
            replays: Mutex::new(HashMap::new()),
            conversations: Mutex::new(HashMap::new()),
            finals: Mutex::new(HashMap::new()),
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
            .with_state(state.clone());
        (with_host_routes(router, host), state)
    }

    /// Serve on the configured port.
    pub async fn serve(self) -> Result<()> {
        let config = self.config.clone();
        crate::host::serve(self.into_router(), &config).await
    }
}

/// Close a background response's replay (its streams end once they have
/// sent every recorded event) and evict it after [`REPLAY_TTL`].
fn end_replay(inner: &Arc<Inner>, id: &str, replay: &Arc<Replay>) {
    replay.finish();
    let evict = inner.clone();
    let id = id.to_string();
    let replay = replay.clone();
    tokio::spawn(async move {
        tokio::time::sleep(REPLAY_TTL).await;
        let mut replays = evict.replays.lock().unwrap();
        // Only this run's buffer (a later run cannot reuse the id, but be exact).
        if replays.get(&id).is_some_and(|r| Arc::ptr_eq(r, &replay)) {
            replays.remove(&id);
        }
    });
}

struct Inner {
    server: ResponsesHostServer,
    running: Mutex<HashMap<String, tokio::task::AbortHandle>>,
    replays: Mutex<HashMap<String, Arc<Replay>>>,
    conversations: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Per response with a live [`TerminalHandle`]: serializes its terminal
    /// writes with cancel / delete.
    finals: Mutex<HashMap<String, Arc<Terminal>>>,
}

/// One response's terminal-write coordination. A turn's persistence runs in
/// its own task (so a disconnect cannot cut it short), which also means it
/// outlives a cancel or delete aborting the turn. Both take `state`, so their
/// writes never interleave, and a cancel or delete first raises `stop`, which
/// the persistence checks before each write: whatever it has not yet
/// written, it skips. A cancel or delete therefore wins unless the turn's
/// writes all committed first.
#[derive(Default)]
struct Terminal {
    stop: std::sync::atomic::AtomicBool,
    state: tokio::sync::Mutex<TerminalState>,
}

#[derive(Default)]
struct TerminalState {
    /// Cancelled or deleted: a persistence that starts later writes nothing.
    finalized: bool,
    /// A persistence stopped part-way: its response record may read as
    /// finished, but the turn did not commit, so it is still cancellable.
    stopped: bool,
}

impl Terminal {
    fn stopping(&self) -> bool {
        self.stop.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// A holder of one response's [`Terminal`]: the turn (from before it can be
/// cancelled until its persistence ends, shared by every clone of its
/// [`Prepared`]) and each cancel or delete while it runs. Holders of one
/// response share its entry in [`Inner::finals`]; the last to drop removes
/// it, so the map holds only responses with a live holder and nothing is
/// kept after the fact (a turn's later writes go through the entry it
/// already holds).
struct TerminalHandle {
    inner: Arc<Inner>,
    id: String,
    terminal: Option<Arc<Terminal>>,
}

impl std::ops::Deref for TerminalHandle {
    type Target = Terminal;

    fn deref(&self) -> &Terminal {
        self.terminal.as_ref().expect("taken only on drop")
    }
}

impl Drop for TerminalHandle {
    fn drop(&mut self) {
        let mut terminals = self.inner.finals.lock().unwrap();
        // Released under the map lock, so concurrent holders serialize and
        // exactly the last one sees only the map's reference left.
        drop(self.terminal.take());
        if terminals
            .get(&self.id)
            .is_some_and(|t| Arc::strong_count(t) == 1)
        {
            terminals.remove(&self.id);
        }
    }
}

impl Inner {
    /// A holder of `id`'s terminal coordination (see [`TerminalHandle`]).
    fn terminal(self: &Arc<Self>, id: &str) -> TerminalHandle {
        let terminal = self
            .finals
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_default()
            .clone();
        TerminalHandle {
            inner: self.clone(),
            id: id.to_string(),
            terminal: Some(terminal),
        }
    }

    /// The lock serializing one conversation's turns, from loading its
    /// transcript through persisting the new one, so concurrent turns
    /// cannot drop each other.
    fn conversation_lock(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.conversations.lock().unwrap();
        locks.retain(|_, l| Arc::strong_count(l) > 1);
        locks.entry(id.to_string()).or_default().clone()
    }
}

/// How long a background response's events stay replayable after it ends
/// (the agentserver's replay TTL).
const REPLAY_TTL: Duration = Duration::from_secs(600);

/// A background response's stream events, kept for
/// `GET /responses/{id}?stream=true` (live while it runs, then for
/// [`REPLAY_TTL`]).
#[derive(Default)]
struct Replay {
    events: Mutex<Vec<(String, Value)>>,
    done: std::sync::atomic::AtomicBool,
    changed: tokio::sync::Notify,
}

impl Replay {
    fn push(&self, kind: &str, payload: &Value) {
        self.events
            .lock()
            .unwrap()
            .push((kind.to_string(), payload.clone()));
        self.changed.notify_waiters();
    }

    fn finish(&self) {
        self.done.store(true, std::sync::atomic::Ordering::SeqCst);
        self.changed.notify_waiters();
    }
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
    resume: Option<Resume>,
    /// Held for the turn's whole life (it travels with the turn into
    /// streaming and background tasks).
    _conversation_turn: Option<Arc<tokio::sync::OwnedMutexGuard<()>>>,
    /// The predecessor's session state (`HistorySource::AgentServer`).
    session_state: Option<Value>,
    /// This turn's session state after the run, for persistence.
    session_out: Arc<Mutex<Option<Value>>>,
    /// Held from before the turn can be cancelled or deleted until its
    /// persistence ends.
    terminal: Arc<TerminalHandle>,
    /// The agent, when resolved before the turn runs (see
    /// [`snapshot_inherited_session`]).
    agent: Option<Arc<dyn SupportsAgentRun>>,
}

/// Replies resuming a paused workflow.
#[derive(Clone)]
struct Resume {
    checkpoint_id: String,
    replies: HashMap<String, Value>,
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
    let conversation_turn = match &conversation {
        Some(id) => Some(Arc::new(inner.conversation_lock(id).lock_owned().await)),
        None => None,
    };
    let mut pause = None;
    let mut session_state = None;
    let history = if let Some(previous) = &create.previous_response_id {
        match server.store.get(previous).await {
            Ok(Some(stored)) => {
                // An unfinished predecessor has no transcript yet:
                // continuing from it would silently drop its turn.
                let status = stored.response["status"].as_str().unwrap_or_default();
                if matches!(status, "queued" | "in_progress") {
                    return bad_request(format!(
                        "Previous response with id '{previous}' is still {status}; \
                         wait for it to finish before continuing from it."
                    ));
                }
                pause = stored.pause;
                session_state = stored.session_state;
                stored.transcript
            }
            Ok(None) => {
                return bad_request(format!("Previous response with id '{previous}' not found."))
            }
            Err(e) => return server_error(e.to_string()),
        }
    } else if let Some(conversation) = &conversation {
        match server.store.conversation_state(conversation).await {
            Ok(state) => session_state = state,
            Err(e) => return server_error(e.to_string()),
        }
        match server.store.conversation(conversation).await {
            Ok(t) => t.unwrap_or_default(),
            Err(e) => return server_error(e.to_string()),
        }
    } else {
        Vec::new()
    };
    if !store
        && server.history == HistorySource::Service
        && (create.previous_response_id.is_some() || conversation.is_some())
    {
        return bad_request(
            "store=false cannot resume service-managed history; start a new one-shot request.",
        );
    }
    let resume = match (&server.target, pause) {
        (Target::Workflow { .. }, Some(pause)) => match workflow_replies(&pause, &create.input) {
            Ok(replies) => Some(Resume {
                checkpoint_id: pause.checkpoint_id,
                replies,
            }),
            Err(e) => return bad_request(e),
        },
        _ => None,
    };
    let response_id = ids::response_id(create.previous_response_id.as_deref());
    let context = FoundryRequestContext::from_headers(&headers, None);
    // A workflow resume's input is its replies, consumed above.
    let messages = if resume.is_some() {
        Vec::new()
    } else {
        match turn_messages(server, &context, &create.input).await {
            Ok(messages) => messages,
            Err(e) => return bad_request(e),
        }
    };
    let prepared = Prepared {
        request: HostedResponseRequest {
            response_id: response_id.clone(),
            body,
            messages,
            history,
            previous_response_id: create.previous_response_id.clone(),
            conversation_id: conversation,
            context,
            checkpoint_storage: server.checkpoints.clone(),
        },
        resume,
        _conversation_turn: conversation_turn,
        session_state,
        session_out: Arc::default(),
        terminal: Arc::new(inner.terminal(&response_id)),
        agent: None,
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
    let mut prepared = prepared;
    if let Err(e) = snapshot_inherited_session(server, &mut prepared).await {
        return match e {
            TurnError::Public(message) => bad_request(message),
            TurnError::Run(e) => {
                tracing::error!(error = %e, "failed to record a turn's inherited session");
                server_error("The response could not be started.".to_string())
            }
        };
    }

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
        // A cancelled background response keeps this: its history and input
        // (as upstream replays a finished predecessor's history and input).
        transcript: prepared
            .request
            .history
            .iter()
            .chain(&prepared.request.messages)
            .cloned()
            .collect(),
        pause: None,
        // The inherited session state travels with the inherited transcript:
        // a cancelled record must not replay history without the security
        // state (FIDES labels, hidden variables) that history carried.
        session_state: prepared.session_state.clone(),
    };
    if let Err(e) = inner.server.store.put(&id, stored).await {
        return server_error(e.to_string());
    }
    let stream = prepared.create.stream;
    let (tx, rx) = mpsc::channel::<Event>(256);
    let replay = Arc::new(Replay::default());
    inner
        .replays
        .lock()
        .unwrap()
        .insert(id.clone(), replay.clone());
    let task_inner = inner.clone();
    let task_id = id.clone();
    // Registered under the lock the task's own cleanup takes, so a turn that
    // finishes instantly cannot remove its entry before it is inserted (which
    // would leave a finished task registered as running).
    let mut running = inner.running.lock().unwrap();
    let task = tokio::spawn(async move {
        let mut events = Events::new(tx);
        events.replay = Some(replay.clone());
        // Events are always produced (and recorded for replay); with no
        // streaming client the channel's receiver is simply gone.
        run_turn(&task_inner, &prepared, Some(&mut events)).await;
        task_inner.running.lock().unwrap().remove(&task_id);
        end_replay(&task_inner, &task_id, &replay);
    });
    running.insert(id, task.abort_handle());
    drop(running);
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
    replay: Option<Arc<Replay>>,
}

impl Events {
    fn new(tx: mpsc::Sender<Event>) -> Self {
        Self {
            tx,
            sequence: 0,
            replay: None,
        }
    }

    async fn emit(&mut self, kind: &str, mut payload: Value) {
        payload["type"] = json!(kind);
        payload["sequence_number"] = json!(self.sequence);
        self.sequence += 1;
        if let Some(replay) = &self.replay {
            replay.push(kind, &payload);
        }
        let event = Event::default().event(kind).data(payload.to_string());
        // A disconnected client does not stop a background run.
        let _ = self.tx.send(event).await;
    }

    async fn emit_all(&mut self, events: Vec<(String, Value)>) {
        for (kind, payload) in events {
            self.emit(&kind, payload).await;
        }
    }
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
    let mut tracker = OutputTracker::new(
        &prepared.request.response_id,
        server.consent_origins.clone(),
    );
    let result = execute(server, prepared, events.as_deref_mut(), &mut tracker).await;
    let mut pause = None;
    let mut approvals = Vec::new();
    let (status, response, transcript) = match result {
        Ok((agent_response, paused)) => {
            pause = paused;
            tracker.close_all();
            if let Some(events) = events.as_deref_mut() {
                events.emit_all(tracker.drain()).await;
            }
            approvals = tracker.take_approvals();
            let response = final_response(prepared, tracker);
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
            let message = match e {
                TurnError::Public(m) => m,
                TurnError::Run(_) => "The agent run failed.".to_string(),
            };
            response["error"] = json!({"code": "server_error", "message": message});
            let mut transcript = prepared.request.history.clone();
            transcript.extend(prepared.request.messages.iter().cloned());
            ("failed".to_string(), response, transcript)
        }
    };
    // Persist before the terminal event: a client that closes the stream
    // once it sees `response.completed` aborts this task, which must not
    // cancel the write it was told succeeded.
    let (status, response) = if prepared.store {
        // Spawned so a disconnect (which drops this task) cannot cut the
        // writes short and leave the conversation half-updated.
        let mut join_failed = prepared.skeleton("failed");
        let inner = inner.clone();
        let prepared = prepared.clone();
        tokio::spawn(async move {
            // Under the response's terminal lock: a cancel or delete either
            // waits for these writes or has already finalized the response,
            // in which case nothing is written.
            let terminal = prepared.terminal.clone();
            let mut state = terminal.state.lock().await;
            if state.finalized || terminal.stopping() {
                return ("cancelled".to_string(), response);
            }
            match persist(
                &inner.server,
                &prepared,
                &status,
                &response,
                transcript,
                pause,
                approvals,
                &terminal,
            )
            .await
            {
                Ok(true) => (status, response),
                Ok(false) => {
                    state.stopped = true;
                    ("cancelled".to_string(), response)
                }
                Err(_) if terminal.stopping() => {
                    state.stopped = true;
                    ("cancelled".to_string(), response)
                }
                Err(e) => {
                    // A turn whose outcome could not be stored did not
                    // succeed: report it as failed, and record that (best
                    // effort) so a background response is not left in
                    // progress.
                    tracing::error!(error = %e, "failed to persist a hosted response");
                    let mut failed = prepared.skeleton("failed");
                    failed["error"] = json!({
                        "code": "server_error",
                        "message": "The response could not be stored.",
                    });
                    let record = StoredResponse {
                        response: failed.clone(),
                        input_items: prepared.input_items.clone(),
                        transcript: prepared
                            .request
                            .history
                            .iter()
                            .chain(&prepared.request.messages)
                            .cloned()
                            .collect(),
                        pause: None,
                        session_state: prepared.session_state.clone(),
                    };
                    if let Err(e) = inner
                        .server
                        .store
                        .put(&prepared.request.response_id, record)
                        .await
                    {
                        tracing::error!(error = %e, "failed to record a persistence failure");
                    }
                    ("failed".to_string(), failed)
                }
            }
        })
        .await
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "the persistence task failed");
            join_failed["error"] = json!({
                "code": "server_error",
                "message": "The response could not be stored.",
            });
            ("failed".to_string(), join_failed)
        })
    } else {
        (status, response)
    };
    if let Some(events) = events {
        let kind = match status.as_str() {
            "failed" => "response.failed",
            "incomplete" => "response.incomplete",
            _ => "response.completed",
        };
        events.emit(kind, json!({"response": response})).await;
    }
    (status, response)
}

/// Store the response, then (unless it failed) its conversation's
/// transcript. Stops (returning `false`) before any write once `terminal`
/// is told to stop.
#[allow(clippy::too_many_arguments)] // one turn's outcome, consumed once
async fn persist(
    server: &ResponsesHostServer,
    prepared: &Prepared,
    status: &str,
    response: &Value,
    transcript: Vec<Message>,
    pause: Option<WorkflowPause>,
    approvals: Vec<(String, Content)>,
    terminal: &Terminal,
) -> Result<bool> {
    if !approvals.is_empty() {
        let store = server.approval_store(&prepared.request.context)?;
        for (id, request) in &approvals {
            store.save_approval_request(id, request).await?;
        }
    }
    // A turn that produced no new snapshot (a failed or interrupted run, a
    // workflow) keeps the state it inherited, next to the inherited
    // transcript: continuing from it must not start from a fresh session.
    let session_state = prepared
        .session_out
        .lock()
        .unwrap()
        .take()
        .or_else(|| prepared.session_state.clone());
    let stored = StoredResponse {
        response: response.clone(),
        input_items: prepared.input_items.clone(),
        transcript: transcript.clone(),
        pause,
        session_state: session_state.clone(),
    };
    if terminal.stopping() {
        return Ok(false);
    }
    server
        .store
        .put(&prepared.request.response_id, stored)
        .await?;
    if let (Some(conversation), false) = (&prepared.request.conversation_id, status == "failed") {
        // A cancel or delete that arrived during the response write keeps
        // the conversation as it was.
        if terminal.stopping() {
            return Ok(false);
        }
        // Transcript and state move together. The previous pair is read
        // first (under the conversation's turn lock) so a failed update can
        // put it back rather than leave a new transcript with an old state.
        let previous_transcript = server.store.conversation(conversation).await?;
        let previous_state = server.store.conversation_state(conversation).await?;
        if let Err(e) = server
            .store
            .update_conversation(conversation, transcript, session_state)
            .await
        {
            if let Err(restore) = server
                .store
                .update_conversation(
                    conversation,
                    previous_transcript.unwrap_or_default(),
                    previous_state,
                )
                .await
            {
                tracing::error!(error = %restore, "failed to restore a conversation after a failed update");
            }
            return Err(e);
        }
    }
    Ok(true)
}

/// Why a turn failed: a run error (its text stays private) or a projection
/// failure whose message is safe to return.
#[derive(Debug)]
enum TurnError {
    Run(agent_framework_core::error::Error),
    Public(String),
}

impl std::fmt::Display for TurnError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Run(e) => e.fmt(f),
            Self::Public(m) => f.write_str(m),
        }
    }
}

impl From<agent_framework_core::error::Error> for TurnError {
    fn from(e: agent_framework_core::error::Error) -> Self {
        Self::Run(e)
    }
}

impl From<OutputError> for TurnError {
    fn from(e: OutputError) -> Self {
        Self::Public(e.0)
    }
}

impl From<serde_json::Error> for TurnError {
    fn from(e: serde_json::Error) -> Self {
        Self::Run(e.into())
    }
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

fn run_options(create: &CreateResponse) -> ChatOptions {
    let mut options = ChatOptions::new();
    options.temperature = create.temperature;
    options.top_p = create.top_p;
    options.max_tokens = create.max_output_tokens;
    options.instructions = create.instructions.clone();
    options
}

/// Fold a response (or each streamed update) into the tracker, emitting
/// the stream events as they are produced.
async fn execute(
    server: &ResponsesHostServer,
    prepared: &Prepared,
    events: Option<&mut Events>,
    tracker: &mut OutputTracker,
) -> std::result::Result<(AgentResponse, Option<WorkflowPause>), TurnError> {
    let request = &prepared.request;
    let agent = match &server.target {
        _ if prepared.agent.is_some() => prepared.agent.clone().expect("checked"),
        Target::Agent(agent) => agent.clone(),
        Target::AgentFactory(factory) => factory().await?,
        Target::Workflow { factory, parse } => {
            if server.history == HistorySource::Service {
                return Err(TurnError::Public(
                    "HistorySource::Service is only supported for agents.".into(),
                ));
            }
            let workflow = factory(request)?;
            let storage = server.checkpoints.clone();
            let run = match (&prepared.resume, &storage) {
                (Some(resume), Some(storage)) => {
                    let mut run = workflow
                        .run_from_checkpoint(&resume.checkpoint_id, storage.clone())
                        .await?;
                    run.send_responses(resume.replies.clone()).await?;
                    run
                }
                _ => {
                    let input = match parse {
                        Some(parse) => parse(request)?,
                        None => {
                            let mut all = request.history.clone();
                            all.extend(request.messages.iter().cloned());
                            serde_json::to_value(all)?
                        }
                    };
                    workflow.run(input).await?
                }
            };
            let response = workflow_response(&run.outputs());
            tracker.handle_response(&response)?;
            let pending = run.pending_requests();
            let pause = match (&storage, pending.is_empty()) {
                (Some(storage), false) => {
                    let checkpoint = pause_checkpoint(storage.as_ref(), workflow.id(), &pending)
                        .await?
                        .ok_or_else(|| {
                            TurnError::Public(
                                "The workflow paused without a checkpoint holding its requests; \
                                 build it with the request's checkpoint storage."
                                    .into(),
                            )
                        })?;
                    let mut requests = Vec::new();
                    let mut request_data = Vec::new();
                    for request in &pending {
                        let wire_id =
                            match serde_json::from_value::<Content>(request.request_data.clone()) {
                                // An approval surfaces as `mcp_approval_request`;
                                // its reply is an `mcp_approval_response`.
                                Ok(approval @ Content::FunctionApprovalRequest(_)) => {
                                    tracker.handle(&approval, None, false)?;
                                    tracker.last_approval_id().ok_or_else(|| {
                                        TurnError::Public(
                                            "The approval request was not projected.".into(),
                                        )
                                    })?
                                }
                                other => {
                                    let wire_id =
                                        ids::new_id("call", Some(&prepared.request.response_id));
                                    tracker.handle(
                                        &request_info_call(
                                            &wire_id,
                                            &request.request_data,
                                            other.ok(),
                                        ),
                                        None,
                                        false,
                                    )?;
                                    wire_id
                                }
                            };
                        requests.push((wire_id, request.request_id.clone()));
                        request_data.push(request.request_data.clone());
                    }
                    Some(WorkflowPause {
                        checkpoint_id: checkpoint,
                        requests,
                        request_data,
                    })
                }
                (None, false) => {
                    tracing::warn!(
                        "the workflow paused on request_info but the server has no checkpoint store; \
                         its requests cannot be answered on a later turn"
                    );
                    None
                }
                _ => None,
            };
            if let Some(events) = events {
                events.emit_all(tracker.drain()).await;
            }
            return Ok((response, pause));
        }
    };
    let host_history = server.history == HistorySource::AgentServer;
    let mut messages = if host_history {
        request.history.clone()
    } else {
        Vec::new()
    };
    messages.extend(request.messages.iter().cloned());
    let mut options = run_options(&prepared.create);
    options.store = match server.history {
        HistorySource::AgentServer => Some(false),
        HistorySource::Service => Some(prepared.store),
        HistorySource::Agent => (!prepared.store).then_some(false),
    };
    if let Some(hook) = &server.prepare_options {
        options = hook(request, options)?;
    }
    let options = AgentRunOptions::new().with_chat_options(options);
    let sessions = if host_history {
        None
    } else {
        Some(server.host_sessions(&request.context)?)
    };
    let (mut session, local_history) = match &sessions {
        // The transcript is replayed; only the state bag carries over.
        None => match &prepared.session_state {
            Some(state) => (
                agent_framework_core::session::AgentSession::from_dict(state)?,
                None,
            ),
            None => (agent.create_session(), None),
        },
        Some(sessions) => {
            let (mut session, prior) =
                load_session(server, sessions, request, agent.as_ref()).await?;
            let history = attach_local_history(&mut session, prior);
            (session, history)
        }
    };
    let response = match events {
        None => {
            let response = agent
                .run_with_options(messages, Some(&mut session), options)
                .await?;
            if let Some(id) = response.conversation_id.as_deref() {
                session.try_adopt_service_session_id(id);
            }
            tracker.handle_response(&response)?;
            response
        }
        Some(events) => {
            // The stream owns a clone; the state bag is shared, the service
            // conversation id is adopted from the updates.
            let mut stream = agent
                .run_stream(messages, Some(session.clone()), Some(options))
                .await?;
            let mut updates = Vec::new();
            while let Some(update) = stream.next().await {
                let update = update?;
                if let Some(id) = update.conversation_id.as_deref() {
                    session.try_adopt_service_session_id(id);
                }
                tracker.handle_update(&update)?;
                events.emit_all(tracker.drain()).await;
                updates.push(update);
            }
            AgentResponse::from_updates(updates)
        }
    };
    if host_history {
        // Detached, without history: the next turn replays the transcript.
        let mut state = session.to_dict();
        if let Some(map) = state.as_object_mut() {
            map.remove("service_session_id");
        }
        *prepared.session_out.lock().unwrap() = Some(state);
    }
    if !host_history && prepared.store {
        // Part of persistence: a session that cannot be saved fails the
        // turn rather than leaving its continuation unresumable.
        if let Some(sessions) = &sessions {
            sessions
                .save(
                    agent.as_ref(),
                    &response_session_key(&request.response_id),
                    &session,
                    local_history.as_ref(),
                )
                .await?;
            if let Some(conversation) = &request.conversation_id {
                sessions
                    .save(
                        agent.as_ref(),
                        &conversation_session_key(conversation),
                        &session,
                        local_history.as_ref(),
                    )
                    .await?;
            }
        }
    }
    Ok((response, None))
}

/// The agent session to continue: the conversation's, else the previous
/// response's, else a new one.
fn response_session_key(response_id: &str) -> String {
    format!("response-{response_id}")
}

fn conversation_session_key(conversation_id: &str) -> String {
    format!("conversation-{conversation_id}")
}

/// One turn's view of the host-kept agent sessions.
#[cfg(feature = "session-store")]
struct HostSessions(
    Arc<dyn agent_framework_core::session_store::AgentSessionStore>,
    /// The configured agent name, the stable namespace sessions are stored
    /// under (see [`Namespaced`]).
    String,
);

/// An agent presented to the session store under a stable, host-level id,
/// so sessions resume across agent instances: an `agent_factory` may build
/// each request's agent with a fresh id, and session stores key by agent id.
#[cfg(feature = "session-store")]
struct Namespaced<'a> {
    inner: &'a dyn SupportsAgentRun,
    id: String,
}

#[cfg(feature = "session-store")]
impl<'a> Namespaced<'a> {
    fn new(inner: &'a dyn SupportsAgentRun, configured: &str) -> Self {
        let name = if configured.is_empty() {
            inner.name().unwrap_or("agent")
        } else {
            configured
        };
        Self {
            inner,
            id: format!("foundry-responses/{name}"),
        }
    }
}

#[cfg(feature = "session-store")]
#[async_trait]
impl SupportsAgentRun for Namespaced<'_> {
    async fn run(
        &self,
        messages: Vec<Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
    ) -> Result<AgentResponse> {
        self.inner.run(messages, session).await
    }

    async fn run_with_options(
        &self,
        messages: Vec<Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
        options: AgentRunOptions,
    ) -> Result<AgentResponse> {
        self.inner
            .run_with_options(messages, session, options)
            .await
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.inner.name()
    }

    fn create_session(&self) -> agent_framework_core::session::AgentSession {
        self.inner.create_session()
    }
}
#[cfg(not(feature = "session-store"))]
struct HostSessions(Arc<Mutex<HashMap<String, SessionSnapshot>>>);

/// A detached copy of a session (its `to_dict` state and the messages of the
/// history provider the host attached), so each response keeps its own
/// continuation state: live clones would share the state bag and history.
#[cfg(not(feature = "session-store"))]
#[derive(Clone)]
struct SessionSnapshot {
    state: Value,
    history: Option<Vec<Message>>,
}

/// Without the `session-store` feature the host owns the local history: it
/// attaches an in-memory history provider (seeded from the snapshot) to a
/// session the service does not manage, and keeps its handle to snapshot it.
#[cfg(not(feature = "session-store"))]
fn attach_local_history(
    session: &mut agent_framework_core::session::AgentSession,
    prior: Option<Vec<Message>>,
) -> Option<agent_framework_core::history::InMemoryHistoryProvider> {
    if session.service_session_id().is_some() {
        return None;
    }
    let provider = agent_framework_core::history::InMemoryHistoryProvider::with_messages(
        prior.unwrap_or_default(),
    );
    session
        .context_providers
        .insert(0, Arc::new(provider.clone()));
    Some(provider)
}

#[cfg(feature = "session-store")]
fn attach_local_history(
    _session: &mut agent_framework_core::session::AgentSession,
    _prior: Option<Vec<Message>>,
) -> Option<agent_framework_core::history::InMemoryHistoryProvider> {
    None
}

/// For a stored conversation turn kept in host sessions, record the
/// conversation's session as it is now under the turn's own response key,
/// before the turn can run, fail or be cancelled (even while queued). The
/// conversation's turn lock is held, so this is exactly the session the turn
/// inherits; a completed turn replaces it with its own, and a cancelled or
/// failed one keeps it, so continuing from it never picks up the
/// conversation's later turns. Resolves the agent (kept for the turn) to
/// address its sessions.
async fn snapshot_inherited_session(
    server: &ResponsesHostServer,
    prepared: &mut Prepared,
) -> std::result::Result<(), TurnError> {
    let Some(conversation) = &prepared.request.conversation_id else {
        return Ok(());
    };
    if !prepared.store || server.history == HistorySource::AgentServer {
        return Ok(());
    }
    let agent = match &server.target {
        Target::Agent(agent) => agent.clone(),
        Target::AgentFactory(factory) => factory().await?,
        Target::Workflow { .. } => return Ok(()),
    };
    let sessions = server.host_sessions(&prepared.request.context)?;
    sessions
        .copy_or_fresh(
            agent.as_ref(),
            &conversation_session_key(conversation),
            &response_session_key(&prepared.request.response_id),
        )
        .await?;
    prepared.agent = Some(agent);
    Ok(())
}

/// What [`HostSessions::load`] returns: the session and, without the
/// `session-store` feature, its detached history.
type LoadedSession = (
    agent_framework_core::session::AgentSession,
    Option<Vec<Message>>,
);

impl HostSessions {
    async fn load(&self, agent: &dyn SupportsAgentRun, key: &str) -> Result<Option<LoadedSession>> {
        #[cfg(feature = "session-store")]
        {
            let key = agent_framework_core::session_store::AgentSessionStoreKey::new(key)?;
            let agent = Namespaced::new(agent, &self.1);
            Ok(self.0.get_session(&agent, &key).await?.map(|s| (s, None)))
        }
        #[cfg(not(feature = "session-store"))]
        {
            let _ = agent;
            let snapshot = self.0.lock().unwrap().get(key).cloned();
            snapshot
                .map(|snapshot| {
                    Ok((
                        agent_framework_core::session::AgentSession::from_dict(&snapshot.state)?,
                        snapshot.history,
                    ))
                })
                .transpose()
        }
    }

    /// Store `from`'s session under `to` as it is now, or a new session
    /// when `from` has none.
    async fn copy_or_fresh(
        &self,
        agent: &dyn SupportsAgentRun,
        from: &str,
        to: &str,
    ) -> Result<()> {
        #[cfg(feature = "session-store")]
        {
            let from = agent_framework_core::session_store::AgentSessionStoreKey::new(from)?;
            let to = agent_framework_core::session_store::AgentSessionStoreKey::new(to)?;
            let namespaced = Namespaced::new(agent, &self.1);
            let session = match self.0.get_session(&namespaced, &from).await? {
                Some(session) => session,
                None => agent.create_session(),
            };
            self.0.save_session(&namespaced, &to, &session).await
        }
        #[cfg(not(feature = "session-store"))]
        {
            let mut sessions = self.0.lock().unwrap();
            let snapshot = match sessions.get(from) {
                Some(snapshot) => snapshot.clone(),
                None => SessionSnapshot {
                    state: agent.create_session().to_dict(),
                    history: None,
                },
            };
            sessions.insert(to.to_string(), snapshot);
            Ok(())
        }
    }

    async fn save(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &str,
        session: &agent_framework_core::session::AgentSession,
        history: Option<&agent_framework_core::history::InMemoryHistoryProvider>,
    ) -> Result<()> {
        #[cfg(feature = "session-store")]
        {
            let _ = history;
            let key = agent_framework_core::session_store::AgentSessionStoreKey::new(key)?;
            let agent = Namespaced::new(agent, &self.1);
            self.0.save_session(&agent, &key, session).await
        }
        #[cfg(not(feature = "session-store"))]
        {
            let _ = agent;
            let snapshot = SessionSnapshot {
                state: session.to_dict(),
                history: history.map(|h| h.list_messages()),
            };
            self.0.lock().unwrap().insert(key.to_string(), snapshot);
            Ok(())
        }
    }
}

/// How many cancelled or failed predecessors [`load_session`] looks through.
const MAX_INHERITED_HOPS: usize = 64;

async fn load_session(
    server: &ResponsesHostServer,
    sessions: &HostSessions,
    request: &HostedResponseRequest,
    agent: &dyn SupportsAgentRun,
) -> std::result::Result<LoadedSession, TurnError> {
    let key = match (&request.conversation_id, &request.previous_response_id) {
        (Some(conversation), _) => Some(conversation_session_key(conversation)),
        (None, Some(previous)) => Some(response_session_key(previous)),
        _ => None,
    };
    let stored = match &key {
        Some(key) => sessions.load(agent, key).await?,
        None => None,
    };
    match (
        stored,
        &request.previous_response_id,
        &request.conversation_id,
    ) {
        (Some(loaded), _, _) => Ok(loaded),
        (None, Some(previous), None) => inherited_session(server, sessions, previous, agent)
            .await?
            .ok_or_else(|| {
                TurnError::Public(format!(
                    "Cannot find an existing agent session for previous_response_id={previous}."
                ))
            }),
        _ => Ok((agent.create_session(), None)),
    }
}

/// The session a cancelled or failed response inherited, since only a
/// completed turn saves one under its response (the counterpart of the
/// `session_state` an `AgentServer` record keeps): its predecessor's, else
/// a new one. (A conversation turn records its inherited session under its
/// own key before running; see [`snapshot_inherited_session`].) `None` when `previous` is not such a
/// response (or the chain is too long or broken).
async fn inherited_session(
    server: &ResponsesHostServer,
    sessions: &HostSessions,
    previous: &str,
    agent: &dyn SupportsAgentRun,
) -> std::result::Result<Option<LoadedSession>, TurnError> {
    let mut current = previous.to_string();
    for _ in 0..MAX_INHERITED_HOPS {
        let Some(stored) = server.store.get(&current).await? else {
            return Ok(None);
        };
        let response = &stored.response;
        if !matches!(response["status"].as_str(), Some("cancelled" | "failed")) {
            return Ok(None);
        }
        if response["conversation"]["id"].is_string() {
            // A conversation turn records the session it inherited under its
            // own key before it runs; the conversation's current session has
            // moved on, so without that record there is nothing to resume.
            return Ok(None);
        }
        let Some(next) = response["previous_response_id"].as_str() else {
            return Ok(Some((agent.create_session(), None)));
        };
        if let Some(loaded) = sessions.load(agent, &response_session_key(next)).await? {
            return Ok(Some(loaded));
        }
        current = next.to_string();
    }
    Ok(None)
}

/// This turn's input as messages, with each `mcp_approval_response` item
/// resolved through the approval store into the original request's
/// function-approval response (upstream `_item_to_message`).
async fn turn_messages(
    server: &ResponsesHostServer,
    context: &FoundryRequestContext,
    input: &Value,
) -> std::result::Result<Vec<Message>, String> {
    let items = match input {
        Value::Array(items) => items.clone(),
        other => {
            return Ok(responses_to_run(&ResponsesRequest {
                input: other.clone(),
                ..Default::default()
            }))
        }
    };
    let mut messages = Vec::new();
    let mut pending: Vec<Value> = Vec::new();
    let flush = |pending: &mut Vec<Value>, messages: &mut Vec<Message>| {
        if !pending.is_empty() {
            messages.extend(responses_to_run(&ResponsesRequest {
                input: Value::Array(std::mem::take(pending)),
                ..Default::default()
            }));
        }
    };
    for item in items {
        #[cfg(feature = "experimental-computer-use")]
        if let Some(message) = crate::computer::item_to_message(&item)? {
            flush(&mut pending, &mut messages);
            messages.push(message);
            continue;
        }
        if item.get("type").and_then(Value::as_str) != Some("mcp_approval_response") {
            pending.push(item);
            continue;
        }
        flush(&mut pending, &mut messages);
        let (Some(id), Some(approve)) = (
            item.get("approval_request_id").and_then(Value::as_str),
            item.get("approve").and_then(Value::as_bool),
        ) else {
            return Err(
                "An mcp_approval_response requires an approval_request_id and a boolean approve."
                    .into(),
            );
        };
        let store = server.approval_store(context).map_err(|e| e.to_string())?;
        let request = store
            .load_approval_request(id)
            .await
            .map_err(|_| format!("Approval request with ID '{id}' does not exist."))?;
        let Content::FunctionApprovalRequest(request) = request else {
            return Err(format!(
                "Approval request with ID '{id}' is not a function approval request."
            ));
        };
        messages.push(Message::with_contents(
            agent_framework_core::types::Role::user(),
            vec![Content::FunctionApprovalResponse(
                request.create_response(approve),
            )],
        ));
    }
    flush(&mut pending, &mut messages);
    Ok(messages)
}

/// The newest checkpoint of `workflow_id` holding every pending request.
async fn pause_checkpoint(
    storage: &dyn CheckpointStorage,
    workflow_id: &str,
    pending: &[agent_framework_core::workflow::PendingRequest],
) -> Result<Option<String>> {
    let checkpoints = storage.list(Some(workflow_id)).await?;
    Ok(checkpoints
        .into_iter()
        .filter(|cp| {
            pending.iter().all(|p| {
                cp.pending_requests
                    .iter()
                    .any(|q| q.request_id == p.request_id)
            })
        })
        .max_by_key(|cp| cp.timestamp_millis)
        .map(|cp| cp.checkpoint_id))
}

/// A pending request as the call a client answers (upstream `_project`): a
/// framework function call keeps its name and arguments under the wire id, a
/// computer call keeps its actions under the wire id; anything else is
/// `request_info({request_id, request})`.
fn request_info_call(wire_id: &str, data: &Value, content: Option<Content>) -> Content {
    use agent_framework_core::types::{FunctionArguments, FunctionCallContent};
    match content {
        Some(Content::FunctionCall(mut call)) => {
            call.call_id = wire_id.to_string();
            return Content::FunctionCall(call);
        }
        #[cfg(feature = "experimental-computer-use")]
        Some(Content::ComputerToolCall(mut call)) => {
            call.call_id = wire_id.to_string();
            return Content::ComputerToolCall(call);
        }
        _ => {}
    }
    let mut arguments = HashMap::new();
    arguments.insert("request_id".to_string(), json!(wire_id));
    arguments.insert("request".to_string(), data.clone());
    Content::FunctionCall(FunctionCallContent::new(
        wire_id,
        "request_info",
        Some(FunctionArguments::Object(arguments)),
    ))
}

/// Map the turn's replies onto the paused workflow's requests (upstream
/// `_load_replies`): `function_call_output` answers a call, an
/// `mcp_approval_response` answers an approval with the original request's
/// function-approval response, and a `computer_call_output` answers a
/// computer call with its typed result. The whole pending batch must be
/// answered, exactly once, and not mixed with new input. Function outputs
/// that parse as JSON are passed as JSON; otherwise as the string.
fn workflow_replies(
    pause: &WorkflowPause,
    input: &Value,
) -> std::result::Result<HashMap<String, Value>, String> {
    let items: Vec<Value> = match input {
        Value::Array(items) => items.clone(),
        Value::Null => Vec::new(),
        other => vec![other.clone()],
    };
    if items.is_empty() {
        return Err("Workflow replies require an exact pending stored checkpoint.".into());
    }
    let pending = |wire: &str| {
        pause.requests.iter().position(|(w, _)| w == wire).map(|i| {
            let data = pause.request_data.get(i).cloned().unwrap_or(Value::Null);
            (
                pause.requests[i].1.clone(),
                serde_json::from_value::<Content>(data).ok(),
            )
        })
    };
    let mut replies = HashMap::new();
    for item in items {
        let (request_id, reply) = match item.get("type").and_then(Value::as_str) {
            Some("mcp_approval_response") => {
                let (Some(wire), Some(approve)) = (
                    item.get("approval_request_id").and_then(Value::as_str),
                    item.get("approve").and_then(Value::as_bool),
                ) else {
                    return Err(
                        "An approval reply requires its exact ID and a boolean decision.".into(),
                    );
                };
                let Some((request_id, Some(Content::FunctionApprovalRequest(request)))) =
                    pending(wire)
                else {
                    return Err(
                        "The approval is unavailable in this exact pending checkpoint.".into(),
                    );
                };
                let response = Content::FunctionApprovalResponse(request.create_response(approve));
                (
                    request_id,
                    serde_json::to_value(response).map_err(|e| e.to_string())?,
                )
            }
            Some(kind @ ("function_call_output" | "computer_call_output")) => {
                let Some(call_id) = item.get("call_id").and_then(Value::as_str) else {
                    return Err("A workflow reply requires its pending call ID.".into());
                };
                let Some((request_id, content)) = pending(call_id) else {
                    return Err(
                        "A reply must match its wire ID in the exact pending workflow checkpoint."
                            .into(),
                    );
                };
                if matches!(content, Some(Content::FunctionApprovalRequest(_))) {
                    return Err("An approval cannot be replaced by a function result.".into());
                }
                if kind == "computer_call_output" {
                    (request_id, computer_reply(&item, content)?)
                } else {
                    let output = match item.get("output") {
                        Some(Value::String(text)) => serde_json::from_str(text)
                            .unwrap_or_else(|_| Value::String(text.clone())),
                        Some(other) => other.clone(),
                        None => Value::Null,
                    };
                    (request_id, output)
                }
            }
            _ => return Err("Pending workflow replies cannot be mixed with new input.".into()),
        };
        if replies.insert(request_id, reply).is_some() {
            return Err("Duplicate workflow replies are not allowed.".into());
        }
    }
    if replies.len() != pause.requests.len() {
        return Err("Answer exactly the complete pending workflow request batch.".into());
    }
    Ok(replies)
}

/// A `computer_call_output` answering a paused workflow's computer call: the
/// typed result, correlated with the original call.
#[cfg(feature = "experimental-computer-use")]
fn computer_reply(item: &Value, pending: Option<Content>) -> std::result::Result<Value, String> {
    let Some(Content::ComputerToolCall(call)) = pending else {
        return Err("The computer reply has no matching pending computer request.".into());
    };
    let message = crate::computer::item_to_message(item)?.ok_or_else(|| {
        "A computer reply requires a typed screenshot and safety acknowledgement.".to_string()
    })?;
    let [Content::ComputerToolResult(mut result)] = <[Content; 1]>::try_from(message.contents)
        .map_err(|_| {
            "A computer reply requires a typed screenshot and safety acknowledgement.".to_string()
        })?
    else {
        return Err(
            "A computer reply requires a typed screenshot and safety acknowledgement.".into(),
        );
    };
    result.call_id = call.call_id;
    serde_json::to_value(Content::ComputerToolResult(result)).map_err(|e| e.to_string())
}

#[cfg(not(feature = "experimental-computer-use"))]
fn computer_reply(_: &Value, _: Option<Content>) -> std::result::Result<Value, String> {
    Err("Computer replies need the `experimental-computer-use` feature.".into())
}

/// A workflow's outputs as an agent response: agent responses and messages
/// keep their contents; anything else becomes text.
fn workflow_response(outputs: &[Value]) -> AgentResponse {
    let mut messages = Vec::new();
    for value in outputs {
        if let Ok(response) = serde_json::from_value::<AgentResponse>(value.clone()) {
            if !response.messages.is_empty() {
                messages.extend(response.messages);
                continue;
            }
        }
        if let Ok(list) = serde_json::from_value::<Vec<Message>>(value.clone()) {
            if !list.is_empty() {
                messages.extend(
                    list.into_iter()
                        .filter(|m| m.role.to_string() == "assistant"),
                );
                continue;
            }
        }
        if let Ok(message) = serde_json::from_value::<Message>(value.clone()) {
            messages.push(message);
            continue;
        }
        let text = output_text(value);
        if !text.is_empty() {
            messages.push(Message::assistant(text));
        }
    }
    AgentResponse {
        messages,
        ..Default::default()
    }
}

/// A workflow output as text.
fn output_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

/// The final response object from the tracker's items.
fn final_response(prepared: &Prepared, tracker: OutputTracker) -> Value {
    let (output, usage, incomplete) = tracker.finish();
    let output_text: String = output
        .iter()
        .filter(|i| i["type"] == "message")
        .flat_map(|i| i["content"].as_array().cloned().unwrap_or_default())
        .filter(|p| p["type"] == "output_text")
        .filter_map(|p| p["text"].as_str().map(str::to_string))
        .collect();
    let status = if incomplete.is_some() {
        "incomplete"
    } else {
        "completed"
    };
    let mut value = json!({
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": true,
        "tool_choice": "auto",
        "tools": [],
        "error": null,
        "incomplete_details": incomplete.map(|reason| json!({"reason": reason})),
    });
    if let Some(usage) = usage {
        value["usage"] = usage;
    }
    prepared.decorate(value, status)
}

#[derive(Debug, Deserialize, Default)]
struct GetQuery {
    #[serde(default)]
    stream: Option<String>,
    #[serde(default)]
    starting_after: Option<String>,
}

async fn get_response(
    State(inner): State<Arc<Inner>>,
    Path(id): Path<String>,
    Query(query): Query<GetQuery>,
) -> Response {
    let stored = match inner.server.store.get(&id).await {
        Ok(Some(stored)) => stored,
        Ok(None) => return not_found(&id),
        Err(e) => return server_error(e.to_string()),
    };
    if !matches!(query.stream.as_deref(), Some("true") | Some("1")) {
        return Json(stored.response).into_response();
    }
    let after: i64 = match query.starting_after.as_deref().map(str::parse) {
        None => -1,
        Some(Ok(n)) => n,
        Some(Err(_)) => return bad_request("starting_after must be an integer"),
    };
    if stored.response["background"] != json!(true) {
        return bad_request(
            "This response cannot be streamed because it was not created with background=true.",
        );
    }
    let Some(replay) = inner.replays.lock().unwrap().get(&id).cloned() else {
        return bad_request(
            "This response cannot be streamed because it was not created with stream=true \
             or the stream TTL has expired.",
        );
    };
    let (tx, rx) = mpsc::channel::<Event>(256);
    tokio::spawn(async move {
        let mut next = 0usize;
        loop {
            let changed = replay.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            let batch: Vec<(String, Value)> = replay.events.lock().unwrap()[next..].to_vec();
            next += batch.len();
            for (kind, payload) in batch {
                if payload["sequence_number"].as_i64().unwrap_or(-1) <= after {
                    continue;
                }
                let event = Event::default().event(kind).data(payload.to_string());
                if tx.send(event).await.is_err() {
                    return;
                }
            }
            let done = replay.done.load(std::sync::atomic::Ordering::SeqCst);
            if done && next >= replay.events.lock().unwrap().len() {
                return;
            }
            changed.await;
        }
    });
    sse(&inner.server.config, rx, None)
}

async fn delete_response(State(inner): State<Arc<Inner>>, Path(id): Path<String>) -> Response {
    let handle = inner.running.lock().unwrap().remove(&id);
    let was_running = handle.is_some();
    let terminal = inner.terminal(&id);
    if let Some(handle) = handle {
        terminal
            .stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        handle.abort();
    }
    // A deleted response is not replayable: end any open replay streams
    // (the aborted task never will) and drop its buffer.
    let replay = inner.replays.lock().unwrap().remove(&id);
    if let Some(replay) = replay {
        replay.finish();
    }
    // An in-flight persistence (it outlives the abort) has been told to
    // stop: wait for it, then finalize so one that has not yet started
    // writes nothing. The response stays deleted.
    let mut state = terminal.state.lock().await;
    let deleted = inner.server.store.delete(&id).await;
    if was_running || matches!(deleted, Ok(true)) {
        state.finalized = true;
    }
    match deleted {
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
        // The aborted turn's persistence outlives the abort: tell it to stop,
        // then wait for it under the response's terminal lock (or finalize
        // first, so it writes nothing). The turn may have committed its
        // terminal status meanwhile: only an unfinished response (or one
        // whose persistence stopped part-way) becomes cancelled.
        let terminal = inner.terminal(&id);
        terminal
            .stop
            .store(true, std::sync::atomic::Ordering::SeqCst);
        handle.abort();
        let mut state = terminal.state.lock().await;
        stored = match inner.server.store.get(&id).await {
            Ok(Some(current)) => current,
            Ok(None) => return not_found(&id),
            Err(e) => return server_error(e.to_string()),
        };
        let unfinished = state.stopped
            || matches!(
                stored.response["status"].as_str(),
                Some("queued" | "in_progress")
            );
        if !unfinished {
            if stored.response["status"] == json!("cancelled") {
                return Json(stored.response).into_response();
            }
            return bad_request("Cannot cancel a response that has already finished.");
        }
        stored.response["status"] = json!("cancelled");
        // The aborted task never closes its replay: record the cancellation,
        // end the replay's streams, and schedule its eviction.
        let replay = inner.replays.lock().unwrap().get(&id).cloned();
        if let Some(replay) = replay {
            if !replay.done.load(std::sync::atomic::Ordering::SeqCst) {
                let sequence = replay.events.lock().unwrap().len();
                replay.push(
                    "response.cancelled",
                    &json!({
                        "type": "response.cancelled",
                        "sequence_number": sequence,
                        "response": stored.response,
                    }),
                );
                end_replay(&inner, &id, &replay);
            }
        }
        if let Err(e) = inner.server.store.put(&id, stored.clone()).await {
            return server_error(e.to_string());
        }
        state.finalized = true;
        state.stopped = false;
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

    /// Replies at once, except to `hang`, which never finishes.
    struct Hang;

    #[async_trait]
    impl SupportsAgentRun for Hang {
        async fn run(
            &self,
            messages: Vec<Message>,
            _session: Option<&mut agent_framework_core::session::AgentSession>,
        ) -> Result<AgentResponse> {
            if messages.last().map(|m| m.text()).as_deref() == Some("hang") {
                std::future::pending::<()>().await;
            }
            Ok(AgentResponse {
                messages: vec![Message::assistant("done")],
                ..Default::default()
            })
        }

        fn id(&self) -> &str {
            "hang"
        }
    }

    #[tokio::test]
    async fn terminal_coordination_is_reclaimed() {
        use tower::ServiceExt;
        let (router, inner) =
            ResponsesHostServer::agent(Arc::new(Hang)).routes(HostOptions::default());
        let call = |method: &str, path: String, body: Option<Value>| {
            let request = axum::http::Request::builder()
                .method(method)
                .uri(path)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    body.map(|b| b.to_string()).unwrap_or_default(),
                ))
                .unwrap();
            let router = router.clone();
            async move {
                let response = router.oneshot(request).await.unwrap();
                let status = response.status();
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                (
                    status,
                    serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null),
                )
            }
        };
        for i in 0..20 {
            let input = if i % 2 == 0 { "hang" } else { "hi" };
            let (_, queued) = call(
                "POST",
                "/responses".into(),
                Some(json!({"input": input, "background": true})),
            )
            .await;
            let id = queued["id"].as_str().unwrap().to_string();
            if i % 4 < 2 {
                // Cancel a hanging turn, or (once it finished) fail to.
                call("POST", format!("/responses/{id}/cancel"), Some(json!({}))).await;
            }
            let (status, _) = call("DELETE", format!("/responses/{id}"), None).await;
            assert_eq!(status, axum::http::StatusCode::OK);
            // Deleting again, or an unknown id, keeps nothing either.
            call("DELETE", format!("/responses/{id}"), None).await;
            call("POST", format!("/responses/{id}/cancel"), Some(json!({}))).await;
        }
        let (_, done) = call("POST", "/responses".into(), Some(json!({"input": "hi"}))).await;
        assert_eq!(done["status"], "completed");
        // Aborted tasks drop their holders as the runtime reclaims them.
        for _ in 0..100 {
            if inner.finals.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(inner.finals.lock().unwrap().len(), 0, "no tombstones");
    }

    #[test]
    fn string_input_becomes_a_user_message_item() {
        let items = input_items_of(&json!("hi"), "caresp_x");
        assert_eq!(items[0]["role"], "user");
        assert_eq!(items[0]["content"][0]["text"], "hi");
        assert!(items[0]["id"].as_str().unwrap().starts_with("msg_"));
    }
}
