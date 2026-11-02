//! `InvocationsHostServer`: an agent or workflow on the agentserver's
//! Invocations protocol, with durable per-session agent state (upstream
//! `InvocationsHostServer` over `InvocationAgentServerHost`).
//!
//! `POST /invocations[?agent_session_id=…]` runs one turn. The default
//! request body is `{"message": "<text>", "stream": false, "options": {}}`;
//! an agent replies `{"response": "<text>"}`, or streams `event: delta`
//! (`{"text"}`) frames and a final `event: done` (`{"session_id"}`), with
//! `event: error` (`{"message", "code", "status"}`) on failure. Every
//! response carries `x-agent-invocation-id` and `x-agent-session-id`.
//!
//! The agent's session is loaded from the session store before the turn
//! and saved after it (also after a failure), one turn at a time per
//! session. Locally the session is the request's `agent_session_id` (or the
//! platform's, or a new one); hosted it is the platform session scoped to
//! the platform user, so one user can never load another's session.
//!
//! A workflow needs a parser mapping each request to a [`WorkflowTurn`]; it
//! replies `{"output": [frames]}` (or streams them, then `done`), where a
//! frame is `{"type": "output", "data", "executor_id"}` or `{"type":
//! "request_info", "data", "request_id", "source_executor_id"}`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session_store::{AgentSessionStore, AgentSessionStoreKey};
use agent_framework_core::types::{ChatOptions, Message};
use agent_framework_core::workflow::{Workflow, WorkflowEvent};
use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Map, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::config::FoundryHostConfig;
use crate::host::{error_response, with_host_routes, HostOptions};
use crate::responses::AgentFactory;
use crate::scope::{headers, FoundryRequestContext, FoundryRequestScope};
use crate::sessions::{InMemorySessionStoreProvider, SessionStoreProvider};

/// Options an invocation may not set: the agent owns them.
const AGENT_CONTROLLED: &[&str] = &[
    "additional_function_arguments",
    "client_kwargs",
    "compaction_strategy",
    "function_invocation_kwargs",
    "instructions",
    "middleware",
    "session",
    "tokenizer",
    "tools",
];

/// One `POST /invocations` as a parser sees it.
#[derive(Debug, Clone)]
pub struct InvocationRequest {
    /// The invocation id.
    pub invocation_id: String,
    /// The resolved session id.
    pub session_id: String,
    /// The JSON body (`Null` when the body is not JSON).
    pub body: Value,
    /// The raw body.
    pub raw_body: axum::body::Bytes,
    /// The platform request context.
    pub context: FoundryRequestContext,
}

/// An agent turn (upstream `InvocationRun`).
#[derive(Debug, Clone, Default)]
pub struct InvocationRun {
    /// The turn's input.
    pub messages: Vec<Message>,
    /// Caller run options (`temperature`, `top_p`, `max_tokens`, `model`,
    /// …; others reach the client as additional properties).
    pub options: Map<String, Value>,
    /// Whether to stream.
    pub stream: bool,
}

/// A workflow turn (upstream `WorkflowTurn`).
#[derive(Debug, Clone, Default)]
pub struct WorkflowTurn {
    /// The workflow's input.
    pub input: Value,
    /// Whether to stream frames.
    pub stream: bool,
}

/// Maps a request to an agent turn.
pub type InvocationParser = Arc<dyn Fn(&InvocationRequest) -> Result<InvocationRun> + Send + Sync>;
/// Maps a request to a workflow turn.
pub type WorkflowTurnParser = Arc<dyn Fn(&InvocationRequest) -> Result<WorkflowTurn> + Send + Sync>;
/// Creates a fresh workflow per request.
pub type InvocationWorkflowFactory =
    Arc<dyn Fn(&InvocationRequest) -> Result<Workflow> + Send + Sync>;

#[derive(Clone)]
enum Target {
    Agent(Arc<dyn SupportsAgentRun>),
    AgentFactory(AgentFactory),
    Workflow {
        factory: InvocationWorkflowFactory,
        parse: WorkflowTurnParser,
    },
}

/// Serves an agent or workflow on the Invocations protocol. See the
/// [module docs](self).
#[derive(Clone)]
pub struct InvocationsHostServer {
    target: Target,
    parse: Option<InvocationParser>,
    config: FoundryHostConfig,
    sessions: Arc<dyn SessionStoreProvider>,
    legacy_wire_format: bool,
    host: HostOptions,
    openapi: Option<Value>,
    asyncapi_json: Option<Value>,
    asyncapi_yaml: Option<String>,
}

impl std::fmt::Debug for InvocationsHostServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvocationsHostServer")
            .field("config", &self.config)
            .field("legacy_wire_format", &self.legacy_wire_format)
            .finish_non_exhaustive()
    }
}

impl InvocationsHostServer {
    fn new(target: Target) -> Self {
        Self {
            target,
            parse: None,
            config: FoundryHostConfig::local(),
            sessions: Arc::new(InMemorySessionStoreProvider::default()),
            legacy_wire_format: false,
            host: HostOptions::default(),
            openapi: None,
            asyncapi_json: None,
            asyncapi_yaml: None,
        }
    }

    /// Serve one agent instance (its per-conversation state must live in
    /// its session).
    pub fn agent(agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self::new(Target::Agent(agent))
    }

    /// Create an agent per request.
    pub fn agent_factory<F, Fut>(factory: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Arc<dyn SupportsAgentRun>>> + Send + 'static,
    {
        Self::new(Target::AgentFactory(Arc::new(move || Box::pin(factory()))))
    }

    /// Serve a workflow built fresh per request; `parse` maps each request
    /// to its turn (required, as upstream).
    pub fn workflow(
        factory: impl Fn(&InvocationRequest) -> Result<Workflow> + Send + Sync + 'static,
        parse: impl Fn(&InvocationRequest) -> Result<WorkflowTurn> + Send + Sync + 'static,
    ) -> Self {
        Self::new(Target::Workflow {
            factory: Arc::new(factory),
            parse: Arc::new(parse),
        })
    }

    /// Parse agent requests yourself (default: `message` / `stream` /
    /// `options`).
    pub fn with_parse_request(
        mut self,
        parse: impl Fn(&InvocationRequest) -> Result<InvocationRun> + Send + Sync + 'static,
    ) -> Self {
        self.parse = Some(Arc::new(parse));
        self
    }

    /// The host configuration (default [`FoundryHostConfig::local`]).
    pub fn with_config(mut self, config: FoundryHostConfig) -> Self {
        self.config = config;
        self
    }

    /// Where agent sessions persist (default: in memory; use a
    /// [`FoundrySessionStoreProvider`](crate::sessions::FoundrySessionStoreProvider)
    /// with root `invocation_sessions` for Foundry storage).
    pub fn with_session_store_provider(mut self, provider: Arc<dyn SessionStoreProvider>) -> Self {
        self.sessions = provider;
        self
    }

    /// The deprecated plain-text reply and raw streamed chunks.
    pub fn with_legacy_wire_format(mut self, legacy: bool) -> Self {
        self.legacy_wire_format = legacy;
        self
    }

    /// An OpenAPI document served at `GET /invocations/docs/openapi.json`
    /// (404 `not_found` without one), as the agentserver does.
    pub fn with_openapi_spec(mut self, spec: Value) -> Self {
        self.openapi = Some(spec);
        self
    }

    /// An AsyncAPI document served at `GET /invocations/docs/asyncapi.json`.
    pub fn with_asyncapi_spec_json(mut self, spec: Value) -> Self {
        self.asyncapi_json = Some(spec);
        self
    }

    /// An AsyncAPI document (YAML text) served at
    /// `GET /invocations/docs/asyncapi.yaml`.
    pub fn with_asyncapi_spec_yaml(mut self, spec: impl Into<String>) -> Self {
        self.asyncapi_yaml = Some(spec.into());
        self
    }

    /// Lifecycle hooks and server-version segments for the shared routes.
    pub fn with_host_options(mut self, options: HostOptions) -> Self {
        self.host = options;
        self
    }

    /// The router: the Invocations routes plus the shared host routes.
    pub fn into_router(self) -> Router {
        let host = self
            .host
            .clone()
            .server_version_segment("invocations")
            .agent_identity(
                self.config.agent_name.clone(),
                self.config.agent_version.clone(),
                self.config.session_id.clone(),
            );
        let (openapi, asyncapi_json, asyncapi_yaml) = (
            self.openapi.clone(),
            self.asyncapi_json.clone(),
            self.asyncapi_yaml.clone(),
        );
        let state = Arc::new(Inner {
            server: self,
            locks: Mutex::new(HashMap::new()),
        });
        let router = Router::new()
            .route("/invocations", post(invoke))
            .route("/invocations/{id}", axum::routing::get(unknown_invocation))
            .route("/invocations/{id}/cancel", post(unknown_invocation))
            .with_state(state)
            .route(
                "/invocations/docs/openapi.json",
                axum::routing::get(move || async move { spec_json(openapi, "OpenAPI") }),
            )
            .route(
                "/invocations/docs/asyncapi.json",
                axum::routing::get(
                    move || async move { spec_json(asyncapi_json, "AsyncAPI (JSON)") },
                ),
            )
            .route(
                "/invocations/docs/asyncapi.yaml",
                axum::routing::get(move || async move {
                    match asyncapi_yaml {
                        Some(yaml) => (
                            [(
                                axum::http::header::CONTENT_TYPE,
                                "application/yaml; charset=utf-8",
                            )],
                            yaml,
                        )
                            .into_response(),
                        None => error_response(
                            StatusCode::NOT_FOUND,
                            "not_found",
                            "No AsyncAPI (YAML) spec registered",
                        ),
                    }
                }),
            );
        with_host_routes(router, host)
    }

    /// Serve on the configured port.
    pub async fn serve(self) -> Result<()> {
        let config = self.config.clone();
        crate::host::serve(self.into_router(), &config).await
    }
}

struct Inner {
    server: InvocationsHostServer,
    locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
}

impl Inner {
    fn lock_for(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut locks = self.locks.lock().unwrap();
        locks.retain(|_, l| Arc::strong_count(l) > 1);
        locks.entry(key.to_string()).or_default().clone()
    }
}

async fn unknown_invocation(Path(id): Path<String>) -> Response {
    error_response(
        StatusCode::NOT_FOUND,
        "not_found",
        format!("Invocation '{id}' was not found."),
    )
}

/// An id usable in headers and keys: `[A-Za-z0-9-_.:]{1,256}`, else a new
/// UUID (upstream `_sanitize_id`).
fn sanitize_id(value: Option<&str>) -> String {
    match value {
        Some(v)
            if !v.is_empty()
                && v.len() <= 256
                && v.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"-_.:".contains(&b)) =>
        {
            v.to_string()
        }
        _ => uuid::Uuid::new_v4().to_string(),
    }
}

fn error_json(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({"error": message.into()}))).into_response()
}

fn with_ids(mut response: Response, invocation_id: &str, session_id: &str) -> Response {
    let h = response.headers_mut();
    if let Ok(v) = HeaderValue::from_str(invocation_id) {
        h.insert(HeaderName::from_static(headers::INVOCATION_ID), v);
    }
    if let Ok(v) = HeaderValue::from_str(session_id) {
        h.insert(HeaderName::from_static(headers::SESSION_ID), v);
    }
    response
}

/// The trusted storage scope and the session key for this request.
fn resolve_session(
    config: &FoundryHostConfig,
    context: &FoundryRequestContext,
    routed: &[String],
) -> Result<(Option<FoundryRequestScope>, String)> {
    if !config.is_hosted {
        let session = context.session_id.clone().unwrap_or_default();
        return Ok((None, session));
    }
    let user = context.user_id.clone().ok_or_else(|| {
        Error::Configuration(
            "The hosted environment is missing user_id in the request context.".into(),
        )
    })?;
    if routed.len() > 1 {
        return Err(Error::Configuration(
            "Hosted Invocations requires exactly one agent_session_id query parameter.".into(),
        ));
    }
    let routed = routed.first();
    let scope = if !config.session_id.is_empty() {
        if routed.is_some_and(|r| *r != config.session_id) {
            return Err(Error::Configuration(
                "The request agent_session_id does not match the platform session ID.".into(),
            ));
        }
        FoundryRequestScope::from_context(config, context, None)?
    } else {
        let Some(routed) = routed.filter(|r| Some(*r) == context.session_id.as_ref()) else {
            return Err(Error::Configuration(
                "Hosted Invocations without FOUNDRY_AGENT_SESSION_ID require an explicit routed \
                 agent_session_id query parameter matching the request context."
                    .into(),
            ));
        };
        if context.call_id.is_none() {
            return Err(Error::Configuration(
                "Foundry hosted requests require a trusted user ID and call ID.".into(),
            ));
        }
        FoundryRequestScope {
            session_id: routed.clone(),
            user_id: Some(user.clone()),
            call_id: context.call_id.clone(),
            is_hosted: true,
        }
    };
    // The key frames session and user so neither can spoof the other.
    let key = serde_json::to_string(&json!([scope.session_id, user])).unwrap_or_default();
    Ok((Some(scope), key))
}

fn default_parse(request: &InvocationRequest) -> Result<InvocationRun> {
    let Value::Object(body) = &request.body else {
        return Err(Error::Configuration(
            "The invocation must be a JSON object.".into(),
        ));
    };
    let message = match body.get("message") {
        Some(Value::String(m)) => m.clone(),
        _ => return Err(Error::Configuration("message must be a string.".into())),
    };
    let stream = match body.get("stream") {
        None => false,
        Some(Value::Bool(b)) => *b,
        _ => return Err(Error::Configuration("stream must be a boolean.".into())),
    };
    let options = match body.get("options") {
        None => Map::new(),
        Some(Value::Object(o)) => o.clone(),
        _ => return Err(Error::Configuration("options must be an object.".into())),
    };
    Ok(InvocationRun {
        messages: vec![Message::user(message)],
        options,
        stream,
    })
}

fn run_options(options: &Map<String, Value>) -> Result<AgentRunOptions> {
    let reserved: Vec<&str> = AGENT_CONTROLLED
        .iter()
        .copied()
        .filter(|k| options.contains_key(*k))
        .collect();
    if !reserved.is_empty() {
        return Err(Error::Configuration(format!(
            "Invocations options cannot set agent-controlled fields: {}.",
            reserved.join(", ")
        )));
    }
    let mut chat = ChatOptions::new();
    let bad = |k: &str| Error::Configuration(format!("options.{k} has the wrong type."));
    for (key, value) in options {
        match key.as_str() {
            "temperature" => {
                chat.temperature = Some(value.as_f64().ok_or_else(|| bad(key))? as f32)
            }
            "top_p" => chat.top_p = Some(value.as_f64().ok_or_else(|| bad(key))? as f32),
            "max_tokens" => {
                chat.max_tokens = Some(
                    value
                        .as_u64()
                        .and_then(|v| u32::try_from(v).ok())
                        .ok_or_else(|| bad(key))?,
                )
            }
            "model" | "model_id" => {
                chat.model = Some(value.as_str().ok_or_else(|| bad(key))?.into())
            }
            "seed" => chat.seed = Some(value.as_i64().ok_or_else(|| bad(key))?),
            _ => {
                chat.additional_properties
                    .insert(key.clone(), value.clone());
            }
        }
    }
    Ok(AgentRunOptions::new().with_chat_options(chat))
}

async fn invoke(
    State(inner): State<Arc<Inner>>,
    headers: HeaderMap,
    Query(query): Query<Vec<(String, String)>>,
    body: axum::body::Bytes,
) -> Response {
    let server = &inner.server;
    let invocation_id = sanitize_id(
        headers
            .get(headers::INVOCATION_ID)
            .and_then(|v| v.to_str().ok()),
    );
    let routed: Vec<String> = query
        .iter()
        .filter(|(k, _)| k == "agent_session_id")
        .map(|(_, v)| v.clone())
        .collect();
    let requested = routed.first().cloned().or_else(|| {
        (!server.config.session_id.is_empty()).then(|| server.config.session_id.clone())
    });
    let session_id = sanitize_id(requested.as_deref());
    let context = FoundryRequestContext::from_headers(&headers, Some(session_id.clone()));
    let (scope, partition) = match resolve_session(&server.config, &context, &routed) {
        Ok(resolved) => resolved,
        Err(e) => {
            tracing::error!(error = %e, "failed to resolve the invocation session");
            return with_ids(
                error_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
                &invocation_id,
                &session_id,
            );
        }
    };
    let visible_session = scope
        .as_ref()
        .map(|s| s.session_id.clone())
        .unwrap_or_else(|| session_id.clone());
    let request = InvocationRequest {
        invocation_id: invocation_id.clone(),
        session_id: visible_session.clone(),
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
        raw_body: body,
        context,
    };
    let response = match &server.target {
        Target::Workflow { factory, parse } => {
            run_workflow(
                inner.clone(),
                factory.clone(),
                parse(&request),
                &request,
                &partition,
            )
            .await
        }
        _ => run_agent(inner.clone(), &request, partition).await,
    };
    with_ids(response, &invocation_id, &visible_session)
}

async fn resolve_agent(target: &Target) -> Result<Arc<dyn SupportsAgentRun>> {
    match target {
        Target::Agent(agent) => Ok(agent.clone()),
        Target::AgentFactory(factory) => factory().await,
        Target::Workflow { .. } => Err(Error::Configuration("No agent is configured.".into())),
    }
}

/// (message, status, code) for a failed agent turn.
fn agent_failure(e: &Error) -> (String, StatusCode, &'static str) {
    if e.to_string()
        .contains("Another request advanced this agent session")
    {
        return (
            "Another request advanced this agent session; reload before retrying.".into(),
            StatusCode::CONFLICT,
            "session_conflict",
        );
    }
    (
        "Agent invocation failed.".into(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "invocation_failed",
    )
}

fn frame(event: &str, data: Value) -> Event {
    Event::default().event(event).data(data.to_string())
}

fn sse(config: &FoundryHostConfig, rx: mpsc::Receiver<Event>) -> Response {
    let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);
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

async fn run_agent(inner: Arc<Inner>, request: &InvocationRequest, partition: String) -> Response {
    let server = &inner.server;
    let parsed = match &server.parse {
        Some(parse) => parse(request),
        None => default_parse(request),
    };
    let (run, options) = match parsed.and_then(|run| run_options(&run.options).map(|o| (run, o))) {
        Ok(ok) => ok,
        Err(e) => return error_json(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let store = match server
        .sessions
        .session_store(&server.config, &request.context)
    {
        Ok(store) => store,
        Err(e) => {
            tracing::error!(error = %e, "failed to open the invocation session store");
            return error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to prepare invocation request.",
            );
        }
    };
    let key = match AgentSessionStoreKey::new(partition.clone()) {
        Ok(key) => key,
        Err(e) => return error_json(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let session_id = request.session_id.clone();
    if !run.stream {
        let lock = inner.lock_for(&partition);
        let _turn = lock.lock().await;
        let result = agent_turn(&server.target, &store, &key, run, options, None).await;
        return match result {
            Ok(text) if server.legacy_wire_format => text.into_response(),
            Ok(text) => Json(json!({"response": text})).into_response(),
            Err(e) => {
                tracing::error!(error = %e, "invocation failed");
                let (message, status, _) = agent_failure(&e);
                error_json(status, message)
            }
        };
    }
    let (tx, rx) = mpsc::channel::<Event>(256);
    let legacy = server.legacy_wire_format;
    let task_inner = inner.clone();
    tokio::spawn(async move {
        let lock = task_inner.lock_for(&partition);
        let _turn = lock.lock().await;
        let result = agent_turn(
            &task_inner.server.target,
            &store,
            &key,
            run,
            options,
            Some((tx.clone(), legacy)),
        )
        .await;
        match result {
            Ok(_) if !legacy => {
                let _ = tx
                    .send(frame("done", json!({"session_id": session_id})))
                    .await;
            }
            Ok(_) => {}
            Err(e) => {
                tracing::error!(error = %e, "invocation stream failed");
                let (message, status, code) = agent_failure(&e);
                if !legacy {
                    let _ = tx
                        .send(frame(
                            "error",
                            json!({"message": message, "code": code, "status": status.as_u16()}),
                        ))
                        .await;
                }
            }
        }
    });
    sse(&server.config, rx)
}

/// Load the session, run the turn (streaming deltas when `stream` is set),
/// and save the session whether or not the run succeeded.
async fn agent_turn(
    target: &Target,
    store: &Arc<dyn AgentSessionStore>,
    key: &AgentSessionStoreKey,
    run: InvocationRun,
    options: AgentRunOptions,
    stream: Option<(mpsc::Sender<Event>, bool)>,
) -> Result<String> {
    let agent = resolve_agent(target).await?;
    let mut session = store.get_or_create_session(agent.as_ref(), key).await?;
    let result = match &stream {
        None => agent
            .run_with_options(run.messages, Some(&mut session), options)
            .await
            .map(|r| {
                // An agent that does not update the session itself still
                // hands back the service conversation it continued.
                if let Some(id) = r.conversation_id.as_deref() {
                    session.try_adopt_service_session_id(id);
                }
                r.text()
            }),
        Some((tx, legacy)) => {
            async {
                let mut updates = agent
                    .run_stream(run.messages, Some(session.clone()), Some(options))
                    .await?;
                let mut text = String::new();
                while let Some(update) = updates.next().await {
                    let update = update?;
                    // The streamed run owns a clone of the session whose
                    // service conversation id is not shared back, so adopt
                    // it from the updates (text-less ones included) before
                    // the session is saved.
                    if let Some(id) = update.conversation_id.as_deref() {
                        session.try_adopt_service_session_id(id);
                    }
                    let delta = update.text();
                    if delta.is_empty() {
                        continue;
                    }
                    text.push_str(&delta);
                    let event = if *legacy {
                        Event::default().data(delta)
                    } else {
                        frame("delta", json!({"text": delta}))
                    };
                    let _ = tx.send(event).await;
                }
                Ok(text)
            }
            .await
        }
    };
    let saved = store.save_session(agent.as_ref(), key, &session).await;
    match (result, saved) {
        (Ok(text), Ok(())) => Ok(text),
        (Ok(_), Err(e)) => Err(e),
        (Err(run), Ok(())) => Err(run),
        (Err(run), Err(save)) => Err(Error::AgentExecution(format!(
            "Invocation failed: {run}; session persistence also failed: {save}"
        ))),
    }
}

fn spec_json(spec: Option<Value>, kind: &str) -> Response {
    match spec {
        Some(spec) => Json(spec).into_response(),
        None => error_response(
            StatusCode::NOT_FOUND,
            "not_found",
            format!("No {kind} spec registered"),
        ),
    }
}

fn workflow_frame(event: &WorkflowEvent) -> Option<Value> {
    match event {
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => Some(json!({"type": "output", "data": data, "executor_id": source_executor_id})),
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            request_data,
        } => Some(json!({
            "type": "request_info",
            "data": request_data,
            "request_id": request_id,
            "source_executor_id": source_executor_id,
        })),
        _ => None,
    }
}

async fn run_workflow(
    inner: Arc<Inner>,
    factory: InvocationWorkflowFactory,
    turn: Result<WorkflowTurn>,
    request: &InvocationRequest,
    partition: &str,
) -> Response {
    let turn = match turn {
        Ok(turn) => turn,
        Err(e) => return error_json(StatusCode::BAD_REQUEST, e.to_string()),
    };
    let workflow = match factory(request) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(error = %e, "failed to build the workflow");
            return error_json(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Workflow invocation failed.",
            );
        }
    };
    let lock = inner.lock_for(partition);
    if !turn.stream {
        let _turn = lock.lock().await;
        return match workflow.run(turn.input).await {
            Ok(run) => {
                let frames: Vec<Value> = run.events().iter().filter_map(workflow_frame).collect();
                Json(json!({"output": frames})).into_response()
            }
            Err(e) => {
                tracing::error!(error = %e, "workflow invocation failed");
                error_json(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Workflow invocation failed.",
                )
            }
        };
    }
    let (tx, rx) = mpsc::channel::<Event>(256);
    let session_id = request.session_id.clone();
    tokio::spawn(async move {
        let _turn = lock.lock().await;
        let mut events = workflow.run_stream(turn.input);
        let mut failed = None;
        while let Some(event) = events.next().await {
            if let WorkflowEvent::Failed { error } = &event {
                failed = Some(error.clone());
            }
            if let Some(f) = workflow_frame(&event) {
                let kind = f["type"].as_str().unwrap_or("output").to_string();
                let _ = tx.send(frame(&kind, f)).await;
            }
        }
        let last = match failed {
            None => frame("done", json!({"session_id": session_id})),
            Some(error) => {
                tracing::error!(%error, "workflow invocation stream failed");
                frame(
                    "error",
                    json!({"message": "Workflow invocation failed.", "code": "workflow_invocation_failed", "status": 500}),
                )
            }
        };
        let _ = tx.send(last).await;
    });
    sse(&inner.server.config, rx)
}
