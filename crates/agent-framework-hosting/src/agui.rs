//! AG-UI protocol hosting (CopilotKit's SupportsAgentRun-User Interaction protocol).
//!
//! Serves one or more agents over the AG-UI protocol, mirroring the Python
//! `agent_framework_ag_ui` package. AG-UI is Server-Sent Events with camelCase
//! JSON events discriminated by a SCREAMING_SNAKE `type` field.
//!
//! # Route
//! `POST {path}` (default `/`) accepts a [`RunAgentInput`]-shaped body
//! (`threadId`, `runId`, `messages[]`, `tools[]`, `state?`, `context?`,
//! `forwardedProps?`) and responds `text/event-stream`.
//!
//! # Event sequence
//! Emitted with the exact `type` strings the Python bridge
//! (`AgentFrameworkEventBridge`) and the upstream `ag-ui-protocol` SDK use:
//!
//! - `RUN_STARTED` `{threadId, runId}`
//! - per assistant text: `TEXT_MESSAGE_START` `{messageId, role}` →
//!   `TEXT_MESSAGE_CONTENT` `{messageId, delta}`* → `TEXT_MESSAGE_END`
//!   `{messageId}` (one text message per run, closed at the end — mirrors the
//!   bridge, which never resets `current_message_id`)
//! - per tool call: `TOOL_CALL_START` `{toolCallId, toolCallName,
//!   parentMessageId?}` → `TOOL_CALL_ARGS` `{toolCallId, delta}` →
//!   `TOOL_CALL_END` `{toolCallId}` (emitted on the tool's result, or at
//!   finalize for an unexecuted / frontend tool call)
//! - per tool result: `TOOL_CALL_END` `{toolCallId}` then `TOOL_CALL_RESULT`
//!   `{messageId, toolCallId, content, role}`
//! - per approval request: `TOOL_CALL_END` `{toolCallId}` then `CUSTOM`
//!   `{name: "function_approval_request", value}`
//! - `RUN_FINISHED` `{threadId, runId}` — or, on agent failure, `RUN_ERROR`
//!   `{message, code?}` in its place.
//!
//! Wire fidelity: `type` values are the SDK's SCREAMING_SNAKE strings; fields
//! are camelCase; `None`/absent fields are omitted (the SDK encodes events with
//! `by_alias=True, exclude_none=True`); frames are `data: {json}\n\n` with no
//! `[DONE]` sentinel (AG-UI, unlike OpenAI streaming, has none).
//!
//! # The frontend-tool pattern
//! In Python, client-declared `tools` are registered on the agent as
//! *declaration-only* `AIFunction`s: when the model calls one it is **not**
//! executed server-side, so no result is produced, and the bridge surfaces it
//! as `TOOL_CALL_START` / `TOOL_CALL_ARGS` / `TOOL_CALL_END` **without** a
//! `TOOL_CALL_RESULT` — the browser executes it. We mirror the *event* half of
//! this faithfully: any tool call the agent emits without a matching result is
//! closed with a `TOOL_CALL_END` at finalize and never gets a `TOOL_CALL_RESULT`.
//!
//! # Divergences from the Python reference
//! - **Live streaming.** The object-safe [`SupportsAgentRun`] trait exposes `run_stream`,
//!   so this router drives it and frames each [`AgentResponseUpdate`] into
//!   AG-UI events as it arrives — one `TEXT_MESSAGE_CONTENT` per text delta,
//!   `TOOL_CALL_*` per call fragment. `type` ordering and payloads match the
//!   bridge. (Agents whose `run_stream` falls back to the buffered default —
//!   e.g. a plain `Arc<dyn SupportsAgentRun>` that only implements `run` — still emit one
//!   `TEXT_MESSAGE_CONTENT` per message, as before.)
//!
//!   [`AgentResponseUpdate`]: agent_framework_core::types::AgentResponseUpdate
//! - **Client `tools` are injected as declaration-only tools.** Client-declared
//!   `tools` in the `RunAgentInput` are mapped to declaration-only
//!   [`ToolDefinition`]s (no
//!   executor) and passed to the agent via per-run
//!   [`AgentRunOptions`]. When the
//!   model calls one, the function-invocation loop returns the call unexecuted
//!   (see [`crate::agui`]'s frontend-tool note), and it is framed as
//!   `TOOL_CALL_*` **without** a `TOOL_CALL_RESULT` — the browser runs it. An
//!   agent that does not support per-run options (anything other than a
//!   `Agent`) logs a warning and ignores the injected tools.
//! - **State events need a snapshot store.** Without one the router is
//!   stateless: inbound `state` is accepted and ignored, and no
//!   `STATE_SNAPSHOT` / `MESSAGES_SNAPSHOT` is emitted. With one
//!   ([`AgUiRouter::snapshot_store`], see [`snapshot`]) the router persists
//!   each thread, emits `STATE_SNAPSHOT` at the start of a run and
//!   `MESSAGES_SNAPSHOT` at the end, replays a thread to a client that
//!   sends no messages, and resumes paused approvals. `STATE_DELTA` and
//!   predictive state (`predict_state_config` / `state_schema`) have no
//!   equivalent on the core `SupportsAgentRun` trait and are not emitted.
//!   `CUSTOM` is emitted only for approval requests.
//!
//! # Thread snapshots
//! Opt in with [`AgUiRouter::snapshot_store`] and an authorization mode —
//! [`AgUiRouter::authorization_scope`] for an authenticated app,
//! [`AgUiRouter::single_user`] for one principal. A store without a mode
//! does not build: a thread id identifies a thread but does not authorize
//! access to it (upstream #9413). Per request, when a store is configured:
//!
//! - **Hydration.** A request with no `messages` and no `resume` replays the
//!   stored thread without running the agent: `RUN_STARTED`,
//!   `STATE_SNAPSHOT`, `MESSAGES_SNAPSHOT`, then `RUN_FINISHED` carrying any
//!   pending interrupts as its `outcome`.
//! - **A run.** The stored transcript is the server's record. Incoming
//!   messages are matched against it — by id, or by role and content for a
//!   client that replays history without ids — and only what is new
//!   extends it; of that, only user turns and results for tool calls the
//!   server issued are kept, so a client cannot plant `system` or
//!   `assistant` turns in a thread's history. Request `state` overlays
//!   stored state. The agent runs against the thread's restored session,
//!   into which client state is merged *minus* anything shaped like
//!   tool-approval state and any key the server wrote (upstream #9368,
//!   tightened — see [`snapshot`]). A tool call nothing has answered stays
//!   in the record but is not sent to the provider. Pending approvals are
//!   recorded as interrupts on `RUN_FINISHED`, and the thread is saved
//!   before the closing events go out.
//! - **A resume.** `resume` entries must answer exactly the interrupts stored
//!   for this scope and thread (upstream #9365), or the run ends with
//!   `RUN_ERROR` and an `APPROVAL_RESUME_*` code; a thread with interrupts
//!   pending refuses any other turn (`APPROVAL_RESUME_REQUIRED`) until they
//!   are answered or cancelled, as upstream does. The approved call is
//!   rebuilt from the stored interrupt, never from the request. The
//!   interrupts are spent — saved as answered — *before* the agent runs, so
//!   an approved call runs at most once even if the run then fails or the
//!   client disconnects; if the store cannot record that, nothing runs.
//!
//! Turns on one thread run one at a time within the process (a second
//! waits for the first's save), so a double-click cannot resume one
//! approval twice and concurrent turns do not overwrite each other. Replicas
//! sharing a store need the store to serialize writes as well.
//!
//! A store failure while saving a finished run is logged and never replaces
//! events already streamed; the previous snapshot stays authoritative. A run
//! that fails, or whose client disconnects, is not saved.
//!
//! # Client
//! With the `agui-client` feature, `client::AgUiChatClient` (re-exported
//! here) is the other end of this protocol: a
//! [`ChatClient`](agent_framework_core::client::ChatClient) that posts a
//! [`RunAgentInput`] to any AG-UI server and turns the event stream back into
//! chat updates. It shares this module's wire model ([`RunAgentInput`],
//! [`event_type`]) with the router.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::session::AgentSession;
use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
use agent_framework_core::types::{
    Content, FunctionApprovalRequestContent, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, Role, TextContent,
};

use crate::registry::IntoAgentRegistration;
use crate::sse::sse_events_stream;
use crate::util;

#[cfg(feature = "agui-client")]
#[cfg_attr(docsrs, doc(cfg(feature = "agui-client")))]
pub mod client;
pub mod snapshot;

#[cfg(feature = "agui-client")]
#[cfg_attr(docsrs, doc(cfg(feature = "agui-client")))]
pub use client::{
    messages_to_agui, state_carrier, AgUiChatClient, AgUiEventConverter, STATE_CARRIER_KEY,
};
use snapshot::AuthorizationMode;
pub use snapshot::{
    AgUiThreadSnapshot, AgUiThreadSnapshotStore, AuthorizationScopeResolver,
    InMemoryAgUiThreadSnapshotStore,
};

/// The AG-UI event `type` discriminators, spelled as the protocol spells them.
///
/// The router emits these and the `agui-client` feature's `AgUiChatClient`
/// parses them, so both ends share one set of names.
pub mod event_type {
    /// A run began: `{threadId, runId}`.
    pub const RUN_STARTED: &str = "RUN_STARTED";
    /// A run ended normally: `{threadId, runId, result?, outcome?}`.
    pub const RUN_FINISHED: &str = "RUN_FINISHED";
    /// A run failed: `{message, code?}`, in place of `RUN_FINISHED`.
    pub const RUN_ERROR: &str = "RUN_ERROR";
    /// An assistant text message opens: `{messageId, role}`.
    pub const TEXT_MESSAGE_START: &str = "TEXT_MESSAGE_START";
    /// A text delta: `{messageId, delta}`.
    pub const TEXT_MESSAGE_CONTENT: &str = "TEXT_MESSAGE_CONTENT";
    /// An assistant text message closes: `{messageId}`.
    pub const TEXT_MESSAGE_END: &str = "TEXT_MESSAGE_END";
    /// A self-contained text delta: `{messageId?, delta?}`.
    pub const TEXT_MESSAGE_CHUNK: &str = "TEXT_MESSAGE_CHUNK";
    /// A tool call opens: `{toolCallId, toolCallName, parentMessageId?}`.
    pub const TOOL_CALL_START: &str = "TOOL_CALL_START";
    /// A tool-call argument delta: `{toolCallId, delta}`.
    pub const TOOL_CALL_ARGS: &str = "TOOL_CALL_ARGS";
    /// A tool call closes: `{toolCallId}`.
    pub const TOOL_CALL_END: &str = "TOOL_CALL_END";
    /// A self-contained tool-call fragment, shorthand for start/args/end:
    /// `{toolCallId?, toolCallName?, parentMessageId?, delta?}`.
    pub const TOOL_CALL_CHUNK: &str = "TOOL_CALL_CHUNK";
    /// A server-side tool result: `{messageId, toolCallId, content, role}`.
    pub const TOOL_CALL_RESULT: &str = "TOOL_CALL_RESULT";
    /// A reasoning delta: `{messageId, delta}`.
    pub const REASONING_MESSAGE_CONTENT: &str = "REASONING_MESSAGE_CONTENT";
    /// A self-contained reasoning delta: `{messageId?, delta?}`.
    pub const REASONING_MESSAGE_CHUNK: &str = "REASONING_MESSAGE_CHUNK";
    /// An opaque encrypted reasoning value for a message or tool call:
    /// `{subtype: "message" | "tool-call", entityId, encryptedValue}`.
    pub const REASONING_ENCRYPTED_VALUE: &str = "REASONING_ENCRYPTED_VALUE";
    /// The whole shared state: `{snapshot}`.
    pub const STATE_SNAPSHOT: &str = "STATE_SNAPSHOT";
    /// A JSON Patch (RFC 6902) against the shared state: `{delta}`.
    pub const STATE_DELTA: &str = "STATE_DELTA";
    /// The whole message list: `{messages}`.
    pub const MESSAGES_SNAPSHOT: &str = "MESSAGES_SNAPSHOT";
    /// An application-defined event: `{name, value}`.
    pub const CUSTOM: &str = "CUSTOM";
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// One agent bound to a path.
struct AgUiRoute {
    path: String,
    name: String,
    agent: Arc<dyn SupportsAgentRun>,
}

/// Per-route handler state.
struct AgUiState {
    name: String,
    agent: Arc<dyn SupportsAgentRun>,
    authorization: Option<RouteAuthorization>,
    snapshots: Option<Arc<dyn AgUiThreadSnapshotStore>>,
    /// Shared by every route, as the store is.
    thread_locks: Arc<ThreadLocks>,
}

/// A route's resolved authorization mode.
enum RouteAuthorization {
    Resolver(AuthorizationScopeResolver),
    /// The opaque scope [`AgUiRouter::single_user`] generated for this route.
    Fixed(String),
}

/// Serves agents over the AG-UI protocol.
///
/// A single agent is bound with [`AgUiRouter::for_agent`] (default path `/`);
/// [`AgUiRouter::path`] overrides that path and [`AgUiRouter::agent`] adds more
/// agents at further paths.
///
/// ```no_run
/// # use std::sync::Arc;
/// # use agent_framework_core::agent::Agent;
/// # use agent_framework_hosting::agui::AgUiRouter;
/// # fn demo(chat: Agent, planner: Agent) -> axum::Router {
/// AgUiRouter::for_agent("chat", chat)
///     .path("/agent")
///     .agent("planner", planner, "/planner")
///     .into_router()
/// # }
/// ```
pub struct AgUiRouter {
    routes: Vec<AgUiRoute>,
    snapshots: Option<Arc<dyn AgUiThreadSnapshotStore>>,
    authorization: Option<AuthorizationMode>,
    /// More than one authorization mode was configured.
    authorization_conflict: bool,
}

impl AgUiRouter {
    /// Serve `agent` at the default path `/`, identified by `name`.
    ///
    /// Accepts a [`Agent`](agent_framework_core::agent::Agent), a
    /// [`WorkflowAgent`](agent_framework_core::workflow::WorkflowAgent), or an
    /// `Arc<dyn SupportsAgentRun>` (see [`IntoAgentRegistration`]).
    pub fn for_agent(name: impl Into<String>, agent: impl IntoAgentRegistration) -> Self {
        Self {
            routes: vec![AgUiRoute {
                path: "/".to_string(),
                name: name.into(),
                agent: agent.into_agent_registration().agent,
            }],
            snapshots: None,
            authorization: None,
            authorization_conflict: false,
        }
    }

    /// Override the path of the primary agent (the one from
    /// [`AgUiRouter::for_agent`]).
    pub fn path(mut self, path: impl Into<String>) -> Self {
        if let Some(first) = self.routes.first_mut() {
            first.path = path.into();
        }
        self
    }

    /// Add another agent at `path`. Paths must be unique across the router.
    pub fn agent(
        mut self,
        name: impl Into<String>,
        agent: impl IntoAgentRegistration,
        path: impl Into<String>,
    ) -> Self {
        self.routes.push(AgUiRoute {
            path: path.into(),
            name: name.into(),
            agent: agent.into_agent_registration().agent,
        });
        self
    }

    /// Persist each thread in `store` (see [`snapshot`]). Requires an
    /// authorization mode — [`Self::authorization_scope`] or
    /// [`Self::single_user`] — or the router does not build.
    ///
    /// The stored transcript is sent in full on every run, so the router
    /// clears the thread session's service conversation id. It cannot clear
    /// an agent's *default* `ChatOptions::conversation_id`, which the agent
    /// falls back to: an agent built with one would receive every turn twice
    /// (the transcript and the service conversation), and — on any hosting
    /// route, snapshots or not — would share that one service conversation
    /// across every caller. Do not serve such an agent here.
    ///
    /// Every route shares the store, keyed by authorization scope and thread
    /// id. Clients of two agents on one router that reuse a thread id
    /// therefore share that thread; give each agent its own router and store
    /// when that matters.
    pub fn snapshot_store(mut self, store: Arc<dyn AgUiThreadSnapshotStore>) -> Self {
        self.snapshots = Some(store);
        self
    }

    /// Resolve each request's authorization scope with `resolver`, which
    /// sees the request head (headers, and the extensions an authentication
    /// layer populated). Its scope keys every thread snapshot; `Err(status)`
    /// rejects the request before any state is read.
    ///
    /// Also usable without a snapshot store, as a request gate.
    pub fn authorization_scope<F>(mut self, resolver: F) -> Self
    where
        F: Fn(&axum::http::request::Parts) -> Result<String, axum::http::StatusCode>
            + Send
            + Sync
            + 'static,
    {
        self.set_authorization(AuthorizationMode::Resolver(Arc::new(resolver)));
        self
    }

    /// Declare that every caller of this router is one authorization
    /// principal. Each route gets one opaque scope, stable for the life of
    /// the router and different from any other router's.
    ///
    /// The scope does not survive a restart, so a durable store loses its
    /// threads on one; return a fixed scope from
    /// [`Self::authorization_scope`] instead when that matters. Never use
    /// this for callers with distinct authorization.
    pub fn single_user(mut self) -> Self {
        self.set_authorization(AuthorizationMode::SingleUser);
        self
    }

    fn set_authorization(&mut self, mode: AuthorizationMode) {
        if self.authorization.is_some() {
            self.authorization_conflict = true;
        }
        self.authorization = Some(mode);
    }

    /// Build the axum router: one `POST {path}` per registered agent. Composable
    /// and nestable into a larger app.
    ///
    /// # Panics
    ///
    /// When the configuration is invalid — see [`Self::try_into_router`].
    pub fn into_router(self) -> Router {
        self.try_into_router()
            .unwrap_or_else(|e| panic!("invalid AG-UI router configuration: {e}"))
    }

    /// Build the axum router, or report why the configuration is invalid: a
    /// snapshot store without an authorization mode, or more than one
    /// authorization mode.
    pub fn try_into_router(self) -> agent_framework_core::error::Result<Router> {
        use agent_framework_core::error::Error;
        if self.authorization_conflict {
            return Err(Error::Configuration(
                "configure exactly one AG-UI authorization mode: authorization_scope or \
                 single_user"
                    .into(),
            ));
        }
        if self.snapshots.is_some() && self.authorization.is_none() {
            return Err(Error::Configuration(
                "an AG-UI snapshot store needs an explicit authorization mode: a thread id \
                 identifies a thread but does not authorize access to it. Use \
                 authorization_scope for authenticated callers, or single_user"
                    .into(),
            ));
        }
        let mut router = Router::new();
        let thread_locks = Arc::new(ThreadLocks::default());
        for route in self.routes {
            let authorization = self.authorization.as_ref().map(|mode| match mode {
                AuthorizationMode::Resolver(r) => RouteAuthorization::Resolver(r.clone()),
                AuthorizationMode::SingleUser => RouteAuthorization::Fixed(format!(
                    "{}{}",
                    snapshot::SINGLE_USER_SCOPE_PREFIX,
                    uuid::Uuid::new_v4().simple()
                )),
            });
            let state = Arc::new(AgUiState {
                name: route.name,
                agent: route.agent,
                authorization,
                snapshots: self.snapshots.clone(),
                thread_locks: thread_locks.clone(),
            });
            let sub = Router::new()
                .route(&route.path, post(run_agent))
                .with_state(state);
            router = router.merge(sub);
        }
        Ok(router)
    }
}

// ---------------------------------------------------------------------------
// Handler
// ---------------------------------------------------------------------------

async fn run_agent(
    State(state): State<Arc<AgUiState>>,
    parts: axum::http::request::Parts,
    body: String,
) -> Response {
    // Authorization first: an unauthenticated caller gets the resolver's
    // status, not a parse error about a body it should not have sent.
    let scope = match &state.authorization {
        None => None,
        Some(RouteAuthorization::Fixed(scope)) => Some(scope.clone()),
        Some(RouteAuthorization::Resolver(resolve)) => match resolve(&parts) {
            Ok(scope) if !scope.is_empty() => Some(scope),
            Ok(_) => {
                tracing::error!("AG-UI authorization_scope resolver returned an empty scope");
                return error_response(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    "An internal error has occurred.".into(),
                );
            }
            Err(status) => return error_response(status, "Unauthorized.".into()),
        },
    };

    // Parse the body ourselves so malformed input becomes a clean 400 with an
    // AG-UI-shaped error body rather than an axum extractor rejection.
    let input: RunAgentInput = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => return bad_request(format!("Invalid RunAgentInput: {e}")),
    };

    if let (Some(store), Some(scope)) = (state.snapshots.clone(), scope) {
        return run_with_snapshots(state, store, scope, input).await;
    }

    // Thread/run ids are echoed back; generate them when the client omits them
    // (mirrors the Python orchestrator's `input_data.get(...) or uuid4()`).
    let thread_id = input
        .thread_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    tracing::debug!(agent = %state.name, thread_id = %thread_id, run_id = %run_id, "AG-UI run");

    let messages = input_to_messages(&input.messages);

    // Client-declared tools become declaration-only per-run tools; when the
    // model calls one it is returned unexecuted so the browser runs it.
    let additional_tools = parse_client_tools(&input.tools);
    let run_options = if additional_tools.is_empty() {
        None
    } else {
        Some(AgentRunOptions {
            additional_tools,
            ..Default::default()
        })
    };

    // Live streaming: drive `run_stream` and frame each update into AG-UI
    // events as it arrives (one `TEXT_MESSAGE_CONTENT` per text delta, etc.).
    // Bounded channel: a slow client applies backpressure and a disconnected
    // client cancels the run (see `crate::sse`).
    let agent = state.agent.clone();
    let (tx, rx) = crate::sse::bounded_sse_channel();
    let disconnect = tx.clone();
    tokio::spawn(async move {
        let produce = async move {
            if tx.send(run_started(&thread_id, &run_id)).await.is_err() {
                return;
            }
            match agent.run_stream(messages, None, run_options).await {
                Ok(mut stream) => {
                    let mut framing = AgUiFraming::new();
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(update) => {
                                let mut events = Vec::new();
                                framing.push_contents(&update.contents, &mut events);
                                for ev in events {
                                    if tx.send(ev).await.is_err() {
                                        return;
                                    }
                                }
                            }
                            // SupportsAgentRun failure mid-stream → RUN_ERROR in
                            // place of RUN_FINISHED (still a 200 SSE stream;
                            // error in-band).
                            Err(e) => {
                                let _ = tx
                                    .send(json!({ "type": event_type::RUN_ERROR, "message": e.to_string() }))
                                    .await;
                                return;
                            }
                        }
                    }
                    let mut events = Vec::new();
                    framing.finalize(&mut events);
                    events.push(run_finished(&thread_id, &run_id));
                    for ev in events {
                        if tx.send(ev).await.is_err() {
                            return;
                        }
                    }
                }
                // Failure before the stream even opens.
                Err(e) => {
                    let _ = tx
                        .send(json!({ "type": event_type::RUN_ERROR, "message": e.to_string() }))
                        .await;
                }
            }
        };
        tokio::select! {
            _ = disconnect.closed() => {}
            _ = produce => {}
        }
    });

    sse_events_stream(rx)
}

/// Per-thread locks, so turns on one thread run one at a time.
///
/// Every turn is read → run → save. Two at once on the same thread — a
/// double-click, two tabs — would both read the same snapshot: both would
/// resume the same approval and run its call twice, and the later save
/// would discard the earlier turn. The lock covers one process; replicas
/// sharing a store need the store to serialize writes as well.
#[derive(Default)]
struct ThreadLocks {
    locks: std::sync::Mutex<HashMap<ThreadKey, Weak<tokio::sync::Mutex<()>>>>,
}

/// `(scope, thread_id)`.
type ThreadKey = (String, String);

impl ThreadLocks {
    async fn lock(&self, scope: &str, thread_id: &str) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.locks.lock().expect("thread locks poisoned");
            // Entries die with their last guard; sweep them as the map grows.
            if locks.len() > 64 {
                locks.retain(|_, weak| weak.strong_count() > 0);
            }
            let key = (scope.to_string(), thread_id.to_string());
            match locks.get(&key).and_then(Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(tokio::sync::Mutex::new(()));
                    locks.insert(key, Arc::downgrade(&lock));
                    lock
                }
            }
        };
        lock.lock_owned().await
    }
}

/// Send `events` and close the stream.
fn send_all(events: Vec<Value>) -> Response {
    let (tx, rx) = crate::sse::bounded_sse_channel();
    tokio::spawn(async move {
        for ev in events {
            if tx.send(ev).await.is_err() {
                return;
            }
        }
    });
    sse_events_stream(rx)
}

fn internal_error() -> Response {
    error_response(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "An internal error has occurred.".into(),
    )
}

/// The snapshot-backed run: hydrate, or run and save. See the module docs'
/// *Thread snapshots*.
async fn run_with_snapshots(
    state: Arc<AgUiState>,
    store: Arc<dyn AgUiThreadSnapshotStore>,
    scope: String,
    input: RunAgentInput,
) -> Response {
    let thread_id = match input.thread_id.clone() {
        Some(id) if id.is_empty() => return bad_request("threadId must not be empty".into()),
        Some(id) => Some(id),
        None => None,
    };
    let run_id = input
        .run_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Hydration: replay the stored thread without running the agent.
    if input.messages.is_empty() && input.resume.is_none() {
        let Some(thread_id) = thread_id else {
            let thread_id = uuid::Uuid::new_v4().to_string();
            return send_all(vec![
                run_started(&thread_id, &run_id),
                run_finished(&thread_id, &run_id),
            ]);
        };
        // Under the thread lock: a resume spends its interrupts in one save
        // and records the run in a later one, and a hydration between the
        // two would show neither the pending approval nor its outcome.
        let _guard = state.thread_locks.lock(&scope, &thread_id).await;
        let stored = match store.get(&scope, &thread_id).await {
            Ok(stored) => stored.unwrap_or_default(),
            Err(e) => {
                tracing::error!(error = %e, "AG-UI snapshot store read failed");
                return internal_error();
            }
        };
        let mut events = vec![run_started(&thread_id, &run_id)];
        if let Some(state) = &stored.state {
            events.push(json!({ "type": event_type::STATE_SNAPSHOT, "snapshot": state }));
        }
        if !stored.messages.is_empty() {
            events.push(
                json!({ "type": event_type::MESSAGES_SNAPSHOT, "messages": stored.messages }),
            );
        }
        events.push(run_finished_with(
            &thread_id,
            &run_id,
            stored.interrupt.as_deref().unwrap_or_default(),
        ));
        return send_all(events);
    }

    // A generated thread id names a thread nobody can have stored yet.
    let thread_id = thread_id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    tracing::debug!(agent = %state.name, thread_id = %thread_id, run_id = %run_id, "AG-UI run (snapshots)");

    // Held from the read to the save; see `ThreadLocks`.
    let guard = state.thread_locks.lock(&scope, &thread_id).await;
    let mut stored = match store.get(&scope, &thread_id).await {
        Ok(stored) => stored.unwrap_or_default(),
        Err(e) => {
            tracing::error!(error = %e, "AG-UI snapshot store read failed");
            return internal_error();
        }
    };
    let pending = stored.interrupt.take().unwrap_or_default();

    // A resume answers exactly the interrupts stored for this thread, and a
    // thread with interrupts pending takes nothing else: running a new turn
    // past them would either discard them or leave a call the conversation
    // has moved on from waiting to run out of context. Upstream's contract;
    // a client that wants to move on cancels them with a `cancelled` resume.
    let resolved = match &input.resume {
        None if !pending.is_empty() => {
            return send_all(vec![
                run_started(&thread_id, &run_id),
                json!({
                    "type": event_type::RUN_ERROR,
                    "message": "Pending interrupts require a resume entry for every interruptId.",
                    "code": "APPROVAL_RESUME_REQUIRED",
                }),
            ]);
        }
        None => Vec::new(),
        Some(resume) => match snapshot::resolve_resume(resume, &pending) {
            Ok(resolved) => resolved,
            Err(e) => {
                return send_all(vec![
                    run_started(&thread_id, &run_id),
                    json!({ "type": event_type::RUN_ERROR, "message": e.message, "code": e.code }),
                ]);
            }
        },
    };
    if !resolved.is_empty() {
        // Spend the interrupts before anything runs: a run that fails or
        // whose client disconnects after the tool executed must not leave
        // the approval resumable a second time. If the store will not take
        // it, nothing runs — at most once, never twice.
        if let Err(e) = store.save(&scope, &thread_id, stored.clone()).await {
            tracing::error!(error = %e, "AG-UI snapshot store could not spend the resumed interrupts");
            return send_all(vec![
                run_started(&thread_id, &run_id),
                json!({
                    "type": event_type::RUN_ERROR,
                    "message": "The approval could not be recorded; nothing was run.",
                    "code": "SNAPSHOT_STORE_UNAVAILABLE",
                }),
            ]);
        }
    }

    let mut transcript = snapshot::reconcile_messages(&stored.messages, &input.messages);
    let mut messages = input_to_messages(&transcript);
    prune_unanswered_calls(&mut messages);
    if !resolved.is_empty() {
        // The approval request, then its answer: what the function-invocation
        // loop needs to run (or refuse) the call. The call joins the
        // transcript now that it is no longer pending.
        messages.push(Message::with_contents(
            Role::assistant(),
            resolved
                .iter()
                .map(|r| Content::FunctionApprovalRequest(r.request.clone()))
                .collect(),
        ));
        messages.push(Message::with_contents(
            Role::user(),
            resolved
                .iter()
                .map(|r| Content::FunctionApprovalResponse(r.response.clone()))
                .collect(),
        ));
        let calls: Vec<Value> = resolved
            .iter()
            .map(|r| {
                let fc = &r.response.function_call;
                tool_call_message_entry(
                    &fc.call_id,
                    &fc.name,
                    &arguments_delta(fc).unwrap_or_default(),
                    fc.protected_data.as_deref(),
                )
            })
            .collect();
        transcript.push(json!({ "id": util::msg_id(), "role": "assistant", "toolCalls": calls }));
    }

    let client_keys: HashSet<String> = stored.client_state_keys.iter().cloned().collect();
    let effective_state = snapshot::effective_state(stored.state.as_ref(), input.state.as_ref());

    // The thread's session, restored without its service conversation id:
    // the transcript is sent in full every run, so continuing a service-side
    // conversation as well would hand the model every turn twice.
    let session = match &stored.session_state {
        Some(saved) => AgentSession::from_dict(saved).unwrap_or_else(|e| {
            tracing::error!(error = %e, "AG-UI snapshot session state did not restore; starting a fresh session");
            AgentSession::new()
        }),
        None => AgentSession::new(),
    };
    let mut accepted_keys = Vec::new();
    if let Some(client_state) = &input.state {
        let (accepted, ignored) =
            snapshot::merge_client_state(&session, client_state, &client_keys);
        accepted_keys = accepted;
        if !ignored.is_empty() {
            tracing::warn!(
                keys = ?ignored,
                "ignoring client shared state the server owns, or that carries tool approval authority"
            );
        }
    }
    let session_handle = session.clone();

    let additional_tools = parse_client_tools(&input.tools);
    let run_options = (!additional_tools.is_empty()).then(|| AgentRunOptions {
        additional_tools,
        ..Default::default()
    });

    let agent = state.agent.clone();
    let (tx, rx) = crate::sse::bounded_sse_channel();
    let disconnect = tx.clone();
    tokio::spawn(async move {
        let produce = async move {
            let _guard = guard;
            if tx.send(run_started(&thread_id, &run_id)).await.is_err() {
                return;
            }
            if let Some(state) = &effective_state {
                let ev = json!({ "type": event_type::STATE_SNAPSHOT, "snapshot": state });
                if tx.send(ev).await.is_err() {
                    return;
                }
            }
            let mut stream = match agent.run_stream(messages, Some(session), run_options).await {
                Ok(stream) => stream,
                Err(e) => {
                    let _ = tx
                        .send(json!({ "type": event_type::RUN_ERROR, "message": e.to_string() }))
                        .await;
                    return;
                }
            };
            let mut framing = AgUiFraming::new();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(update) => {
                        let mut events = Vec::new();
                        framing.push_contents(&update.contents, &mut events);
                        for ev in events {
                            if tx.send(ev).await.is_err() {
                                return;
                            }
                        }
                    }
                    // A failed run is not saved: the previous snapshot stays
                    // authoritative (minus any interrupts this run spent).
                    Err(e) => {
                        let _ = tx
                            .send(
                                json!({ "type": event_type::RUN_ERROR, "message": e.to_string() }),
                            )
                            .await;
                        return;
                    }
                }
            }
            let mut events = Vec::new();
            framing.finalize(&mut events);

            transcript.extend(framing.transcript());
            let interrupts: Vec<Value> = framing
                .approvals
                .iter()
                .map(snapshot::approval_interrupt)
                .collect();
            let client_state_keys = snapshot::owned_client_keys(
                client_keys.into_iter().chain(accepted_keys),
                effective_state.as_ref(),
                &session_handle,
            );
            let mut session_state = session_handle.to_dict();
            session_state["service_session_id"] = Value::Null;
            // Saved before the closing events, so a client that hydrates on
            // `RUN_FINISHED` sees this run. A failure is logged, never
            // streamed: the events it would replace are the run's own.
            let saved = AgUiThreadSnapshot {
                messages: transcript.clone(),
                state: effective_state,
                interrupt: (!interrupts.is_empty()).then(|| interrupts.clone()),
                session_state: Some(session_state),
                client_state_keys,
            };
            if let Err(e) = store.save(&scope, &thread_id, saved).await {
                tracing::error!(error = %e, "AG-UI snapshot save failed; keeping the previous snapshot");
            }

            events.push(json!({ "type": event_type::MESSAGES_SNAPSHOT, "messages": transcript }));
            events.push(run_finished_with(&thread_id, &run_id, &interrupts));
            for ev in events {
                if tx.send(ev).await.is_err() {
                    return;
                }
            }
        };
        tokio::select! {
            _ = disconnect.closed() => {}
            _ = produce => {}
        }
    });

    sse_events_stream(rx)
}

/// Map AG-UI client-declared tools (`{name, description, parameters}`) into
/// **declaration-only** [`ToolDefinition`]s (no executor). Passed to the agent
/// as per-run `additional_tools`: when the model calls one, the
/// function-invocation loop returns the call to the caller unexecuted, and the
/// AG-UI framing surfaces it as `TOOL_CALL_*` without a `TOOL_CALL_RESULT` —
/// the browser executes it (the frontend-tool pattern).
fn parse_client_tools(tools: &[Value]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .filter_map(|t| {
            let obj = t.as_object()?;
            let name = obj.get("name").and_then(Value::as_str)?.to_string();
            if name.is_empty() {
                return None;
            }
            let description = obj
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let parameters = obj
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));
            Some(ToolDefinition {
                name,
                description,
                parameters,
                kind: ToolKind::Function,
                approval_mode: ApprovalMode::NeverRequire,
                executor: None,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Event framing
// ---------------------------------------------------------------------------

fn run_started(thread_id: &str, run_id: &str) -> Value {
    json!({ "type": event_type::RUN_STARTED, "threadId": thread_id, "runId": run_id })
}

fn run_finished(thread_id: &str, run_id: &str) -> Value {
    json!({ "type": event_type::RUN_FINISHED, "threadId": thread_id, "runId": run_id })
}

/// `RUN_FINISHED`, carrying `interrupts` as its `outcome` when there are any.
fn run_finished_with(thread_id: &str, run_id: &str, interrupts: &[Value]) -> Value {
    let mut ev = run_finished(thread_id, run_id);
    if let Some(outcome) = snapshot::interrupt_outcome(interrupts) {
        ev["outcome"] = outcome;
    }
    ev
}

/// Incremental AG-UI event framing, driven one streamed update's contents at a
/// time (via [`AgUiFraming::push_contents`]) and closed with
/// [`AgUiFraming::finalize`]. This reproduces the bridge's stateful framing
/// across the whole run: a single text message spans the run (lazily opened on
/// the first text delta, never reset), and tool calls are tracked so unresulted
/// ones (frontend tools) are closed at the end.
struct AgUiFraming {
    /// The run's single text message id, opened on the first text delta.
    message_id: Option<String>,
    /// Tool calls opened (in order) and those already closed.
    started_tools: Vec<String>,
    /// Whether each `started_tools` occurrence has been closed. Kept per
    /// occurrence, not per id: a provider can reuse a call id within one
    /// run, and each start needs its own end.
    ended_tools: Vec<bool>,
    /// What the run said, in order, for the thread snapshot.
    segments: Vec<Segment>,
    /// The approvals the run paused on.
    approvals: Vec<FunctionApprovalRequestContent>,
    /// A replay token carried by reasoning content, waiting for the call it
    /// belongs to. Gemini's fallback placement puts the signature on the
    /// thought part just before a call; the Gemini converter backfills an
    /// unsigned call from it, and recording it on the call keeps that pairing
    /// once the reasoning itself is gone from the transcript.
    pending_signature: Option<String>,
}

/// A tool call as the thread snapshot records it.
struct RecordedCall {
    id: String,
    name: String,
    /// The argument fragments, accumulated.
    arguments: String,
    /// The provider's replay token (Gemini's `thoughtSignature`). A
    /// completed call is replayed on every later turn, and Gemini rejects
    /// one that comes back without it.
    protected_data: Option<String>,
}

/// One stretch of a run's output, as the thread snapshot records it.
enum Segment {
    Text(String),
    /// Consecutive tool calls.
    Calls(Vec<RecordedCall>),
    /// A tool result, already an AG-UI message.
    Result(Value),
}

impl AgUiFraming {
    fn new() -> Self {
        Self {
            message_id: None,
            started_tools: Vec::new(),
            ended_tools: Vec::new(),
            segments: Vec::new(),
            approvals: Vec::new(),
            pending_signature: None,
        }
    }

    /// The run's output as AG-UI messages, for the thread snapshot, in the
    /// order it happened: "let me check" → call → result → "it's sunny"
    /// stays four messages rather than collapsing the text after the tool
    /// work. The first text stretch keeps the streamed message id. A call
    /// the run paused on for approval is left out — it is the interrupt, and
    /// a resume records it once it is answered.
    fn transcript(&self) -> Vec<Value> {
        // Each pending approval is for the latest declaration of its call id:
        // a provider can reuse an id within one run, and dropping an earlier,
        // completed declaration would orphan its result.
        let mut unmatched: Vec<&str> = self
            .approvals
            .iter()
            .map(|a| a.function_call.call_id.as_str())
            .collect();
        let mut skip: HashSet<(usize, usize)> = HashSet::new();
        for (si, segment) in self.segments.iter().enumerate().rev() {
            if let Segment::Calls(calls) = segment {
                for (ci, call) in calls.iter().enumerate().rev() {
                    if let Some(pos) = unmatched.iter().position(|p| *p == call.id.as_str()) {
                        unmatched.swap_remove(pos);
                        skip.insert((si, ci));
                    }
                }
            }
        }
        let mut first_text = true;
        let mut out = Vec::new();
        for (si, segment) in self.segments.iter().enumerate() {
            match segment {
                Segment::Text(text) => {
                    let id = match (&self.message_id, first_text) {
                        (Some(mid), true) => mid.clone(),
                        _ => util::msg_id(),
                    };
                    first_text = false;
                    out.push(json!({ "id": id, "role": "assistant", "content": text }));
                }
                Segment::Calls(calls) => {
                    let calls: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .filter(|(ci, _)| !skip.contains(&(si, *ci)))
                        .map(|(_, call)| {
                            tool_call_message_entry(
                                &call.id,
                                &call.name,
                                &call.arguments,
                                call.protected_data.as_deref(),
                            )
                        })
                        .collect();
                    if !calls.is_empty() {
                        out.push(
                            json!({ "id": util::msg_id(), "role": "assistant", "toolCalls": calls }),
                        );
                    }
                }
                Segment::Result(message) => out.push(message.clone()),
            }
        }
        out
    }

    /// Close `tool_call_id` if this run opened it and has not closed it. A
    /// result or approval for a call opened in an earlier request — a
    /// resumed approval — gets no `TOOL_CALL_END`, which with no matching
    /// start is a protocol violation clients reject (upstream does the same).
    /// The earliest open occurrence is the one closed.
    fn end_tool(&mut self, tool_call_id: &str, out: &mut Vec<Value>) {
        if tool_call_id.is_empty() {
            return;
        }
        let open = self
            .started_tools
            .iter()
            .zip(&self.ended_tools)
            .position(|(t, ended)| t == tool_call_id && !ended);
        if let Some(i) = open {
            self.ended_tools[i] = true;
            out.push(json!({ "type": event_type::TOOL_CALL_END, "toolCallId": tool_call_id }));
        }
    }

    /// Frame one streamed update's `contents`, pushing AG-UI events onto `out`.
    fn push_contents(&mut self, contents: &[Content], out: &mut Vec<Value>) {
        for content in contents {
            // Content the Gemini converter sends as a wire part breaks a
            // carrier's pairing with the next call, so a stale signature never
            // reaches an unrelated one: text (even empty), inline data, URIs
            // and tool results. A call consumes the carrier itself; approval
            // requests and responses emit no part and keep it.
            if matches!(
                content,
                Content::Text(_) | Content::Data(_) | Content::Uri(_) | Content::FunctionResult(_)
            ) {
                self.pending_signature = None;
            }
            match content {
                Content::TextReasoning(t) => {
                    // Only a signed part replaces the carrier: an unsigned
                    // one that merely follows must not erase it (the Gemini
                    // converter's rule).
                    if let Some(sig) = t.protected_data.as_deref().filter(|s| !s.is_empty()) {
                        self.pending_signature = Some(sig.to_string());
                    }
                }
                Content::Text(TextContent { text, .. }) if !text.is_empty() => {
                    let mid = match &self.message_id {
                        Some(mid) => mid.clone(),
                        None => {
                            let mid = util::msg_id();
                            out.push(json!({
                                "type": event_type::TEXT_MESSAGE_START,
                                "messageId": mid,
                                "role": "assistant",
                            }));
                            self.message_id = Some(mid.clone());
                            mid
                        }
                    };
                    out.push(json!({
                        "type": event_type::TEXT_MESSAGE_CONTENT,
                        "messageId": mid,
                        "delta": text,
                    }));
                    match self.segments.last_mut() {
                        Some(Segment::Text(buf)) => buf.push_str(text),
                        _ => self.segments.push(Segment::Text(text.clone())),
                    }
                }
                Content::FunctionCall(fc) => {
                    let tool_call_id = coalesce_call_id(fc, self.started_tools.last());
                    if !fc.name.is_empty() {
                        let mut start = json!({
                            "type": event_type::TOOL_CALL_START,
                            "toolCallId": tool_call_id,
                            "toolCallName": fc.name,
                        });
                        // parentMessageId is included only when a text message
                        // is open (omitted-when-None, per the SDK).
                        if let Some(mid) = &self.message_id {
                            start["parentMessageId"] = json!(mid);
                        }
                        out.push(start);
                        self.started_tools.push(tool_call_id.clone());
                        self.ended_tools.push(false);
                        let entry = RecordedCall {
                            id: tool_call_id.clone(),
                            name: fc.name.clone(),
                            arguments: String::new(),
                            // A call consumes the held carrier whether or not
                            // it uses it, as the Gemini converter does — a
                            // call with its own token must not leave the
                            // carrier for the next, unrelated call. The
                            // call's own (non-empty) token wins.
                            protected_data: {
                                let carrier = self.pending_signature.take();
                                fc.protected_data
                                    .clone()
                                    .filter(|s| !s.is_empty())
                                    .or(carrier)
                            },
                        };
                        match self.segments.last_mut() {
                            Some(Segment::Calls(calls)) => calls.push(entry),
                            _ => self.segments.push(Segment::Calls(vec![entry])),
                        }
                    }
                    let delta = arguments_delta(fc).filter(|d| !d.is_empty());
                    if delta.is_some() || fc.protected_data.as_ref().is_some_and(|t| !t.is_empty())
                    {
                        let entry = self.segments.iter_mut().rev().find_map(|s| match s {
                            Segment::Calls(calls) => {
                                calls.iter_mut().rev().find(|c| c.id == tool_call_id)
                            }
                            _ => None,
                        });
                        if let Some(entry) = entry {
                            if let Some(delta) = &delta {
                                entry.arguments.push_str(delta);
                            }
                            // The token can arrive on any fragment.
                            // Only a non-empty token replaces the recorded
                            // one: an empty fragment must not erase a
                            // signature backfilled from the carrier.
                            if let Some(token) =
                                fc.protected_data.as_ref().filter(|t| !t.is_empty())
                            {
                                entry.protected_data = Some(token.clone());
                            }
                        }
                    }
                    if let Some(delta) = delta {
                        out.push(json!({
                            "type": event_type::TOOL_CALL_ARGS,
                            "toolCallId": tool_call_id,
                            "delta": delta,
                        }));
                    }
                }
                Content::FunctionResult(fr) => {
                    self.end_tool(&fr.call_id, out);
                    let message_id = util::msg_id();
                    let content = result_content(fr);
                    self.segments.push(Segment::Result(json!({
                        "id": message_id,
                        "role": "tool",
                        "toolCallId": fr.call_id,
                        "content": content,
                    })));
                    out.push(json!({
                        "type": event_type::TOOL_CALL_RESULT,
                        "messageId": message_id,
                        "toolCallId": fr.call_id,
                        "content": content,
                        "role": "tool",
                    }));
                }
                Content::FunctionApprovalRequest(ar) => {
                    self.end_tool(&ar.function_call.call_id, out);
                    out.push(approval_custom_event(ar));
                    self.approvals.push(ar.clone());
                }
                // Reasoning, data, uri, error, usage, hosted-file/-vector-store,
                // approval *responses* → no AG-UI event (documented subset).
                _ => {}
            }
        }
    }

    /// Emit the run's closing events: close any tool call that never received a
    /// result (frontend-tool pattern), in open order, then close the run's text
    /// message if one was opened. Mirrors the orchestrator's
    /// `pending_without_end` sweep.
    fn finalize(&mut self, out: &mut Vec<Value>) {
        for (tool_call_id, ended) in self.started_tools.iter().zip(&self.ended_tools) {
            if !ended {
                out.push(json!({ "type": event_type::TOOL_CALL_END, "toolCallId": tool_call_id }));
            }
        }
        if let Some(mid) = &self.message_id {
            out.push(json!({ "type": event_type::TEXT_MESSAGE_END, "messageId": mid }));
        }
    }
}

/// Drop tool calls nothing answers from what the agent is given. A frontend
/// tool call the user never ran stays in the stored record, but sent to a
/// provider it is an unanswered call, which OpenAI rejects with a 400 —
/// every later turn on the thread would fail.
///
/// Results are paired with calls in order — a result answers the nearest
/// preceding unanswered call with its id, the rule the function-invocation
/// loop uses — so an earlier result cannot vouch for a later call that
/// reuses its id.
fn prune_unanswered_calls(messages: &mut Vec<Message>) {
    let mut outstanding: Vec<(usize, usize, &str)> = Vec::new();
    for (mi, message) in messages.iter().enumerate() {
        for (ci, content) in message.contents.iter().enumerate() {
            match content {
                Content::FunctionCall(fc) => outstanding.push((mi, ci, fc.call_id.as_str())),
                Content::FunctionResult(r) => {
                    if let Some(i) = outstanding.iter().rposition(|(_, _, id)| *id == r.call_id) {
                        outstanding.remove(i);
                    }
                }
                _ => {}
            }
        }
    }
    let unanswered: HashSet<(usize, usize)> = outstanding
        .into_iter()
        .map(|(mi, ci, _)| (mi, ci))
        .collect();
    for (mi, message) in messages.iter_mut().enumerate() {
        let mut ci = 0;
        message.contents.retain(|_| {
            let keep = !unanswered.contains(&(mi, ci));
            ci += 1;
            keep
        });
    }
    messages.retain(|m| !m.contents.is_empty() || m.role != Role::assistant());
}

/// One entry of an AG-UI assistant message's `toolCalls`. A provider replay
/// token rides as `encryptedValue`, the protocol's field for it, which
/// [`tool_call_to_content`] reads back.
fn tool_call_message_entry(
    id: &str,
    name: &str,
    arguments: &str,
    protected_data: Option<&str>,
) -> Value {
    let mut entry = json!({
        "id": id,
        "type": "function",
        "function": { "name": name, "arguments": arguments },
    });
    if let Some(token) = protected_data {
        entry["encryptedValue"] = Value::String(token.to_string());
    }
    entry
}

/// The AG-UI `CUSTOM` event carrying a function-approval request (mirrors the
/// bridge's `function_approval_request` custom event).
fn approval_custom_event(ar: &FunctionApprovalRequestContent) -> Value {
    let arguments = ar
        .function_call
        .parse_arguments()
        .map(|m| Value::Object(m.into_iter().collect()))
        .unwrap_or(Value::Null);
    json!({
        "type": event_type::CUSTOM,
        "name": "function_approval_request",
        "value": {
            "id": ar.id,
            "function_call": {
                "call_id": ar.function_call.call_id,
                "name": ar.function_call.name,
                "arguments": arguments,
            },
        },
    })
}

/// Resolve the tool-call id for a call fragment: its own `call_id`, else the
/// currently open tool call, else a fresh id (mirrors `_coalesce_tool_call_id`).
fn coalesce_call_id(fc: &FunctionCallContent, current: Option<&String>) -> String {
    if !fc.call_id.is_empty() {
        fc.call_id.clone()
    } else if let Some(cur) = current {
        cur.clone()
    } else {
        util::short_hex()
    }
}

/// The `TOOL_CALL_ARGS` delta string for a call: a raw argument string as-is, a
/// parsed object JSON-serialized (mirrors the bridge's `delta_str`).
fn arguments_delta(fc: &FunctionCallContent) -> Option<String> {
    match &fc.arguments {
        None => None,
        Some(FunctionArguments::Raw(s)) => Some(s.clone()),
        Some(FunctionArguments::Object(map)) => {
            Some(serde_json::to_string(map).unwrap_or_default())
        }
    }
}

/// The `TOOL_CALL_RESULT` content string: a dict/array JSON-serialized, any
/// other value stringified, `None`/exception → the exception text or empty
/// (mirrors the bridge's `result_content`).
fn result_content(fr: &FunctionResultContent) -> String {
    match &fr.result {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Null) | None => fr.exception.clone().unwrap_or_default(),
        Some(other) => other.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Input model & message mapping
// ---------------------------------------------------------------------------

/// The AG-UI `RunAgentInput` (tolerant subset).
///
/// Fields are camelCase per the protocol; snake_case aliases are also accepted
/// for the routing ids (the Python server reads `input_data["thread_id"]` /
/// `["run_id"]`). Unknown fields are ignored.
///
/// It also serializes, as the request body the `agui-client` feature's
/// `AgUiChatClient` posts: camelCase, unset optional fields left out, the
/// list fields always present.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
pub struct RunAgentInput {
    /// Conversation thread id (echoed on `RUN_STARTED`/`RUN_FINISHED`).
    #[serde(alias = "thread_id", skip_serializing_if = "Option::is_none")]
    pub thread_id: Option<String>,
    /// Current run id (echoed on `RUN_STARTED`/`RUN_FINISHED`).
    #[serde(alias = "run_id", skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    /// Conversation history (AG-UI `Message` objects).
    pub messages: Vec<Value>,
    /// Client-declared tools, injected as declaration-only per-run tools so
    /// their calls round-trip back to the browser (see module docs).
    pub tools: Vec<Value>,
    /// Shared state. Ignored without a snapshot store; with one, overlaid
    /// on the thread's stored state (see module docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<Value>,
    /// Contextual objects (accepted and ignored).
    pub context: Vec<Value>,
    /// Extra forwarded properties (accepted and ignored).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub forwarded_props: Option<Value>,
    /// Interrupt descriptors the client can resume from (accepted and
    /// ignored by the router).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub available_interrupts: Option<Value>,
    /// Resume entries that continue a paused run. Ignored without a
    /// snapshot store; with one, they answer the thread's stored interrupts
    /// (see module docs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resume: Option<Value>,
}

/// Map AG-UI `messages[]` to core [`Message`]s.
///
/// Mirrors `agui_messages_to_agent_framework`: `role:"tool"` →
/// [`FunctionResultContent`]; an assistant with `toolCalls` →
/// [`FunctionCallContent`]s (plus any text); everything else → a text message
/// under the mapped role.
fn input_to_messages(messages: &[Value]) -> Vec<Message> {
    messages.iter().filter_map(map_message).collect()
}

fn map_message(msg: &Value) -> Option<Message> {
    let obj = msg.as_object()?;
    let role = obj.get("role").and_then(Value::as_str).unwrap_or("user");

    // Tool result message.
    if role == "tool" {
        let call_id = obj
            .get("toolCallId")
            .or_else(|| obj.get("tool_call_id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let result = obj
            .get("content")
            .or_else(|| obj.get("result"))
            .cloned()
            .unwrap_or(Value::String(String::new()));
        return Some(Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(result),
            ))],
        ));
    }

    // Assistant message carrying tool calls.
    let tool_calls = obj.get("toolCalls").or_else(|| obj.get("tool_calls"));
    if let Some(Value::Array(calls)) = tool_calls {
        let mut contents: Vec<Content> = Vec::new();
        if let Some(text) = obj.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                contents.push(Content::Text(TextContent::new(text)));
            }
        }
        for tc in calls {
            if let Some(fc) = tool_call_to_content(tc) {
                contents.push(Content::FunctionCall(fc));
            }
        }
        return Some(Message::with_contents(Role::assistant(), contents));
    }

    // Plain text message under its role.
    let text = content_text(obj.get("content"));
    Some(Message::new(role_from(role), text))
}

/// Convert one AG-UI `ToolCall` (`{id, type:"function", function:{name,
/// arguments}}`) into a [`FunctionCallContent`].
fn tool_call_to_content(tc: &Value) -> Option<FunctionCallContent> {
    let obj = tc.as_object()?;
    if obj.get("type").and_then(Value::as_str) != Some("function") {
        return None;
    }
    let func = obj.get("function").and_then(Value::as_object);
    let call_id = obj.get("id").and_then(Value::as_str).unwrap_or_default();
    let name = func
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = func.and_then(|f| f.get("arguments")).map(|a| match a {
        Value::String(s) => FunctionArguments::Raw(s.clone()),
        other => FunctionArguments::Raw(other.to_string()),
    });
    let mut call = FunctionCallContent::new(call_id, name, arguments);
    call.protected_data = obj
        .get("encryptedValue")
        .and_then(Value::as_str)
        .map(str::to_string);
    Some(call)
}

/// Extract display text from an AG-UI `content` value (a string, or an array of
/// content parts each carrying `text`).
fn content_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
            .collect::<Vec<_>>()
            .join(""),
        Some(Value::Null) | None => String::new(),
        Some(other) => other.to_string(),
    }
}

/// Map an AG-UI role to a core [`Role`]. Unknown roles (e.g. `developer`) fall
/// back to `user`, matching the Python adapter's `.get(role, Role.USER)`.
fn role_from(role: &str) -> Role {
    match role {
        "assistant" => Role::assistant(),
        "system" => Role::system(),
        "tool" => Role::tool(),
        "user" => Role::user(),
        _ => Role::user(),
    }
}

fn bad_request(message: String) -> Response {
    error_response(axum::http::StatusCode::BAD_REQUEST, message)
}

fn error_response(status: axum::http::StatusCode, message: String) -> Response {
    (
        status,
        Json(Value::Object({
            let mut m = Map::new();
            m.insert("error".to_string(), json!({ "message": message }));
            m
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str) -> Content {
        Content::FunctionCall(FunctionCallContent::new(
            id,
            "f",
            Some(FunctionArguments::Raw("{}".into())),
        ))
    }

    #[test]
    fn a_reused_call_id_gets_an_end_for_each_start() {
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        let result =
            |id: &str| Content::FunctionResult(FunctionResultContent::new(id, Some(json!("done"))));
        framing.push_contents(
            &[call("c1"), result("c1"), call("c1"), result("c1")],
            &mut out,
        );
        framing.finalize(&mut out);
        let count = |ty: &str| out.iter().filter(|e| e["type"] == ty).count();
        assert_eq!(count("TOOL_CALL_START"), 2);
        assert_eq!(count("TOOL_CALL_END"), 2, "{out:?}");
    }

    #[test]
    fn pruning_pairs_results_with_calls_in_order() {
        // `c1` completed, then a later frontend call reused `c1` and was
        // never answered: only the later declaration is pruned.
        let mut messages = vec![
            Message::with_contents(Role::assistant(), vec![call("c1")]),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent::new(
                    "c1",
                    Some(json!("done")),
                ))],
            ),
            Message::with_contents(Role::assistant(), vec![call("c1")]),
            Message::user("never mind"),
        ];
        prune_unanswered_calls(&mut messages);
        assert_eq!(messages.len(), 3, "{messages:?}");
        assert!(matches!(messages[0].contents[0], Content::FunctionCall(_)));
        assert_eq!(messages[2].role, Role::user());
    }

    #[test]
    fn a_stored_call_keeps_its_replay_token() {
        // Gemini rejects a replayed call without its `thoughtSignature`; a
        // snapshot thread replays every completed call on every turn.
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        let mut signed = FunctionCallContent::new("c1", "f", None);
        signed.protected_data = Some("c2ln".into());
        let mut args =
            FunctionCallContent::new("c1", "", Some(FunctionArguments::Raw("{}".into())));
        args.protected_data = None;
        framing.push_contents(
            &[
                Content::FunctionCall(signed),
                Content::FunctionCall(args),
                Content::FunctionResult(FunctionResultContent::new("c1", Some(json!("done")))),
            ],
            &mut out,
        );
        let transcript = framing.transcript();
        let entry = &transcript[0]["toolCalls"][0];
        assert_eq!(entry["encryptedValue"], "c2ln");
        // And it comes back on the next turn.
        let call = tool_call_to_content(entry).unwrap();
        assert_eq!(call.protected_data.as_deref(), Some("c2ln"));
        assert_eq!(call.arguments, Some(FunctionArguments::Raw("{}".into())));
    }

    #[test]
    fn a_signature_on_preceding_reasoning_is_recorded_on_the_call() {
        // Gemini's fallback placement: the signature rides the thought part
        // just before the call, which itself carries none.
        let reasoning = |sig: Option<&str>| {
            Content::TextReasoning(agent_framework_core::types::TextReasoningContent {
                text: "thinking".into(),
                protected_data: sig.map(str::to_string),
                ..Default::default()
            })
        };
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        framing.push_contents(
            &[
                reasoning(Some("carrier")),
                // An unsigned thought that follows must not erase it.
                reasoning(None),
                call("c1"),
                Content::FunctionResult(FunctionResultContent::new("c1", Some(json!("done")))),
                reasoning(Some("stale")),
                Content::text("visible text in between"),
                call("c2"),
                Content::FunctionResult(FunctionResultContent::new("c2", Some(json!("done")))),
            ],
            &mut out,
        );
        let transcript = framing.transcript();
        let calls: Vec<&Value> = transcript
            .iter()
            .filter_map(|m| m.get("toolCalls"))
            .flat_map(|c| c.as_array().unwrap())
            .collect();
        assert_eq!(calls[0]["encryptedValue"], "carrier");
        assert!(
            calls[1].get("encryptedValue").is_none(),
            "a signature crossed visible text to an unrelated call: {calls:?}"
        );
    }

    #[test]
    fn a_signed_call_consumes_the_carrier_it_does_not_use() {
        // A call with its own token still uses up the held carrier, or the
        // next unsigned call would be stamped with a signature belonging to
        // a different call.
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        let mut own =
            FunctionCallContent::new("c1", "f", Some(FunctionArguments::Raw("{}".into())));
        own.protected_data = Some("own".into());
        framing.push_contents(
            &[
                Content::TextReasoning(agent_framework_core::types::TextReasoningContent {
                    text: "thinking".into(),
                    protected_data: Some("carrier".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(own),
                call("c2"),
            ],
            &mut out,
        );
        let transcript = framing.transcript();
        let calls = transcript[0]["toolCalls"].as_array().unwrap();
        assert_eq!(calls[0]["encryptedValue"], "own");
        assert!(calls[1].get("encryptedValue").is_none(), "{calls:?}");
    }

    #[test]
    fn a_carrier_survives_an_empty_token_and_breaks_on_data() {
        let carrier = || {
            Content::TextReasoning(agent_framework_core::types::TextReasoningContent {
                text: "thinking".into(),
                protected_data: Some("carrier".into()),
                ..Default::default()
            })
        };
        let mut empty_token =
            FunctionCallContent::new("c1", "f", Some(FunctionArguments::Raw("{}".into())));
        empty_token.protected_data = Some(String::new());
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        framing.push_contents(
            &[
                // An empty token on the call (or a later fragment) must not
                // erase the backfilled signature.
                carrier(),
                Content::FunctionCall(empty_token),
                // Inline data is a wire part: it breaks the next pairing.
                carrier(),
                Content::Data(agent_framework_core::types::DataContent::from_bytes(
                    b"x",
                    "image/png",
                )),
                call("c2"),
            ],
            &mut out,
        );
        let transcript = framing.transcript();
        let calls: Vec<&Value> = transcript
            .iter()
            .filter_map(|m| m.get("toolCalls"))
            .flat_map(|c| c.as_array().unwrap())
            .collect();
        assert_eq!(calls[0]["encryptedValue"], "carrier");
        assert!(calls[1].get("encryptedValue").is_none(), "{calls:?}");
    }

    #[test]
    fn a_reused_call_id_keeps_the_completed_declaration() {
        // A provider reused `c1`: the first call completed, the second is
        // paused for approval. Only the paused declaration is left out of
        // the transcript; the completed one keeps its result company.
        let mut framing = AgUiFraming::new();
        let mut out = Vec::new();
        framing.push_contents(
            &[
                call("c1"),
                Content::FunctionResult(FunctionResultContent::new("c1", Some(json!("done")))),
                call("c1"),
                Content::FunctionApprovalRequest(FunctionApprovalRequestContent {
                    id: "req_1".into(),
                    function_call: FunctionCallContent::new("c1", "f", None),
                }),
            ],
            &mut out,
        );
        let transcript = framing.transcript();
        let declared: Vec<&Value> = transcript
            .iter()
            .filter_map(|m| m.get("toolCalls"))
            .flat_map(|c| c.as_array().unwrap())
            .collect();
        assert_eq!(declared.len(), 1, "{transcript:?}");
        assert_eq!(transcript[0]["toolCalls"][0]["id"], "c1");
        assert_eq!(transcript[1]["role"], "tool");
    }
}
