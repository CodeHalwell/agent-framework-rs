//! The container host (upstream agentserver `AgentServerHost`): the routes
//! every protocol shares (`GET /readiness`, the snapshot lifecycle hooks),
//! request-id and server-version headers, error bodies, and serving.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use axum::extract::Request;
use axum::http::{HeaderName, HeaderValue, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::config::FoundryHostConfig;
use crate::scope::headers;

/// The body `GET /readiness` returns by default.
pub const HEALTHY_BODY: &str = r#"{"status":"healthy"}"#;

/// The server-version segment this crate reports in `x-platform-server`.
pub fn server_version() -> String {
    format!("agent-framework-rs/{}", env!("CARGO_PKG_VERSION"))
}

/// An async lifecycle hook (`before-snapshot` / `after-restore`).
pub type LifecycleHook =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

/// The session a restored process now serves (the agentserver's
/// `AgentSessionContext`), from the `after-restore` request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentSessionContext {
    /// The session served by the process.
    pub session_id: String,
    /// This materialization of the process.
    pub restore_id: String,
    /// Environment values that differ from the captured process; already
    /// applied to the process environment when the hook runs.
    pub session_env_overrides: BTreeMap<String, String>,
}

/// An `after-restore` hook receiving the session context.
pub type AfterRestoreHook = Arc<
    dyn Fn(AgentSessionContext) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync,
>;

const SESSION_ID_ENV: &str = "FOUNDRY_AGENT_SESSION_ID";

/// What restores have done to this process.
#[derive(Default)]
struct LifecycleState {
    assigned_session_id: Option<String>,
    completed_restore_ids: HashSet<String>,
    /// The values the overridden variables had before any restore.
    captured: HashMap<String, Option<String>>,
    /// Variables the last restore set.
    applied: HashSet<String>,
}

/// The agentserver error body: `{"error": {"code", "message"}}`.
pub fn error_response(status: StatusCode, code: &str, message: impl Into<String>) -> Response {
    let mut response = (
        status,
        Json(json!({"error": {"code": code, "message": message.into()}})),
    )
        .into_response();
    let source = if status.is_client_error() {
        "user"
    } else {
        "upstream"
    };
    response.headers_mut().insert(
        HeaderName::from_static(headers::ERROR_SOURCE),
        HeaderValue::from_static(source),
    );
    response
}

/// Options for the shared routes.
#[derive(Clone, Default)]
pub struct HostOptions {
    before_snapshot: Option<LifecycleHook>,
    after_restore: Option<LifecycleHook>,
    after_restore_context: Option<AfterRestoreHook>,
    server_version_segments: Vec<String>,
    agent_name: String,
    agent_version: String,
    session_id: String,
}

impl std::fmt::Debug for HostOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostOptions")
            .field("before_snapshot", &self.before_snapshot.is_some())
            .field("after_restore", &self.after_restore.is_some())
            .field("server_version_segments", &self.server_version_segments)
            .finish()
    }
}

impl HostOptions {
    /// Run `hook` when the platform is about to snapshot the container.
    pub fn before_snapshot(mut self, hook: LifecycleHook) -> Self {
        self.before_snapshot = Some(hook);
        self
    }

    /// Run `hook` after the platform restores the container.
    pub fn after_restore(mut self, hook: LifecycleHook) -> Self {
        self.after_restore = Some(hook);
        self
    }

    /// Run `hook` with the restored session's context after the platform
    /// restores the container (after its environment overrides are
    /// applied).
    pub fn after_restore_with_context(mut self, hook: AfterRestoreHook) -> Self {
        self.after_restore_context = Some(hook);
        self
    }

    /// The agent identity recorded on every request span (the servers set it
    /// from their [`FoundryHostConfig`]).
    pub fn agent_identity(
        mut self,
        name: impl Into<String>,
        version: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        self.agent_name = name.into();
        self.agent_version = version.into();
        self.session_id = session_id.into();
        self
    }

    /// Add a protocol segment to `x-platform-server`.
    pub fn server_version_segment(mut self, segment: impl Into<String>) -> Self {
        self.server_version_segments.push(segment.into());
        self
    }
}

async fn lifecycle(
    hook: Option<LifecycleHook>,
    name: &'static str,
    body: axum::body::Bytes,
) -> Response {
    let valid =
        body.is_empty() || serde_json::from_slice::<Value>(&body).is_ok_and(|v| v.is_object());
    if !valid {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!("invalid_{name}_request"),
            format!("The {} request is invalid.", name.replace('_', "-")),
        );
    }
    if let Some(hook) = hook {
        if let Err(e) = hook().await {
            tracing::error!(error = %e, "the {name} lifecycle hook failed");
            return error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("{name}_failed"),
                format!("The {} hook failed.", name.replace('_', "-")),
            );
        }
    }
    Json(json!({"status": "ok"})).into_response()
}

fn parse_session_context(body: &[u8]) -> std::result::Result<AgentSessionContext, ()> {
    let payload: Value = serde_json::from_slice(body).map_err(|_| ())?;
    let context = payload
        .get("session_context")
        .and_then(Value::as_object)
        .ok_or(())?;
    let session_id = context
        .get("session_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && !s.contains('\0'))
        .ok_or(())?;
    let restore_id = context
        .get("restore_id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or(())?;
    let mut overrides = BTreeMap::new();
    match context.get("session_env_overrides") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            for (key, value) in map {
                let value = value.as_str().ok_or(())?;
                if key.is_empty() || key.contains('=') || key.contains('\0') || value.contains('\0')
                {
                    return Err(());
                }
                overrides.insert(key.clone(), value.to_string());
            }
        }
        Some(_) => return Err(()),
    }
    if overrides
        .get(SESSION_ID_ENV)
        .is_some_and(|id| id != session_id)
    {
        return Err(());
    }
    Ok(AgentSessionContext {
        session_id: session_id.to_string(),
        restore_id: restore_id.to_string(),
        session_env_overrides: overrides,
    })
}

fn set_env(values: &HashMap<String, Option<String>>) {
    for (key, value) in values {
        match value {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }
}

/// `after-restore`: bind the process to the session (a second session is
/// `409 session_mismatch`; a repeated restore id is a no-op), apply its
/// environment overrides (`FOUNDRY_AGENT_SESSION_ID` defaults to the
/// session), and run the hooks; a failing hook restores the previous
/// environment.
async fn after_restore(
    options: HostOptions,
    state: Arc<tokio::sync::Mutex<LifecycleState>>,
    body: axum::body::Bytes,
) -> Response {
    let Ok(context) = parse_session_context(&body) else {
        return error_response(
            StatusCode::BAD_REQUEST,
            "invalid_after_restore_request",
            "The after-restore request is invalid.",
        );
    };
    let mut state = state.lock().await;
    if state
        .assigned_session_id
        .as_ref()
        .is_some_and(|id| *id != context.session_id)
    {
        return error_response(
            StatusCode::CONFLICT,
            "session_mismatch",
            "The restored process is already assigned to another session.",
        );
    }
    if state.completed_restore_ids.contains(&context.restore_id) {
        return Json(json!({"status": "ok"})).into_response();
    }
    let mut effective: HashMap<String, String> = context
        .session_env_overrides
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    effective
        .entry(SESSION_ID_ENV.to_string())
        .or_insert_with(|| context.session_id.clone());
    let keys: HashSet<String> = state
        .applied
        .iter()
        .cloned()
        .chain(effective.keys().cloned())
        .collect();
    let previous: HashMap<String, Option<String>> = keys
        .iter()
        .map(|k| (k.clone(), std::env::var(k).ok()))
        .collect();
    for (key, value) in &previous {
        state
            .captured
            .entry(key.clone())
            .or_insert_with(|| value.clone());
    }
    let next: HashMap<String, Option<String>> = keys
        .iter()
        .map(|k| {
            let value = effective
                .get(k)
                .cloned()
                .or_else(|| state.captured.get(k).cloned().flatten());
            (k.clone(), value)
        })
        .collect();
    set_env(&next);
    let mut failed = false;
    if let Some(hook) = &options.after_restore {
        failed |= hook().await.is_err();
    }
    if !failed {
        if let Some(hook) = &options.after_restore_context {
            failed |= hook(context.clone()).await.is_err();
        }
    }
    if failed {
        set_env(&previous);
        tracing::error!("the after_restore lifecycle hook failed");
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "after_restore_failed",
            "The after-restore hook failed.",
        );
    }
    state
        .assigned_session_id
        .get_or_insert_with(|| context.session_id.clone());
    state.completed_restore_ids.insert(context.restore_id);
    state.applied = effective.into_keys().collect();
    Json(json!({"status": "ok"})).into_response()
}

/// The trace and parent span ids of a W3C `traceparent`
/// (`00-<32 hex>-<16 hex>-<2 hex>`), when it is valid.
pub fn parse_traceparent(value: &str) -> Option<(String, String)> {
    let parts: Vec<&str> = value.trim().split('-').collect();
    let hex = |s: &str, n: usize| {
        s.len() == n
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    };
    match parts.as_slice() {
        [version, trace, parent, flags, ..]
            if hex(version, 2)
                && *version != "ff"
                && hex(trace, 32)
                && hex(parent, 16)
                && hex(flags, 2)
                && trace.bytes().any(|b| b != b'0')
                && parent.bytes().any(|b| b != b'0') =>
        {
            Some((trace.to_string(), parent.to_string()))
        }
        _ => None,
    }
}

/// The span every request runs in: request id, method, path, the caller's
/// W3C trace (`trace_id` / `parent_span_id` from `traceparent`), and the
/// agent identity. With the `otel` feature it is made a child of the
/// caller's span.
fn request_span(request: &Request, request_id: &str, options: &HostOptions) -> tracing::Span {
    let traceparent = request
        .headers()
        .get("traceparent")
        .and_then(|v| v.to_str().ok())
        .and_then(parse_traceparent);
    let (trace_id, parent_id) = traceparent.unzip();
    let span = tracing::info_span!(
        "foundry.request",
        otel.kind = "server",
        http.request.method = %request.method(),
        url.path = %request.uri().path(),
        request_id = %request_id,
        trace_id = trace_id.as_deref().unwrap_or(""),
        parent_span_id = parent_id.as_deref().unwrap_or(""),
        gen_ai.agent.name = %options.agent_name,
        gen_ai.agent.version = %options.agent_version,
        foundry.session.id = %options.session_id,
    );
    #[cfg(feature = "otel")]
    {
        use opentelemetry::propagation::TextMapPropagator;
        use tracing_opentelemetry::OpenTelemetrySpanExt;
        struct Headers<'a>(&'a axum::http::HeaderMap);
        impl opentelemetry::propagation::Extractor for Headers<'_> {
            fn get(&self, key: &str) -> Option<&str> {
                self.0.get(key).and_then(|v| v.to_str().ok())
            }
            fn keys(&self) -> Vec<&str> {
                self.0.keys().map(|k| k.as_str()).collect()
            }
        }
        // The caller's trace, and its W3C baggage (which carries the ids
        // [`FoundryEnrichmentSpanProcessor`] stamps).
        let parent = opentelemetry_sdk::propagation::TraceContextPropagator::new()
            .extract(&Headers(request.headers()));
        let parent = opentelemetry_sdk::propagation::BaggagePropagator::new()
            .extract_with_context(&parent, &Headers(request.headers()));
        let _ = span.set_parent(parent);
    }
    span
}

/// Adds Foundry identity attributes to every span (the agentserver's
/// `_FoundryEnrichmentSpanProcessor`; feature `otel`):
///
/// - `microsoft.foundry` = `True`, `microsoft.foundry.project.id`, and the
///   agent identity (`gen_ai.agent.name` / `version` / `id`) from the host
///   configuration;
/// - `microsoft.session.id` from the `azure.ai.agentserver.session_id`
///   baggage entry, else the configured session id;
/// - `gen_ai.conversation.id` and
///   `azure.ai.agentserver.invocations.invocation_id` from the
///   `azure.ai.agentserver.conversation_id` / `invocation_id` baggage entries
///   (request spans join the caller's W3C baggage).
///
/// [`configure_observability`] installs it. Divergence: the agent identity
/// is stamped when a span starts (upstream stamps it as the span ends, so a
/// framework cannot overwrite it); a span that sets its own
/// `gen_ai.agent.*` afterwards keeps its value.
#[cfg(feature = "otel")]
#[derive(Debug, Clone, Default)]
pub struct FoundryEnrichmentSpanProcessor {
    agent_name: String,
    agent_version: String,
    agent_id: String,
    project_id: String,
    session_id: String,
}

#[cfg(feature = "otel")]
impl FoundryEnrichmentSpanProcessor {
    /// Baggage key of the session id.
    pub const SESSION_BAGGAGE: &'static str = "azure.ai.agentserver.session_id";
    /// Baggage key of the conversation id.
    pub const CONVERSATION_BAGGAGE: &'static str = "azure.ai.agentserver.conversation_id";
    /// Baggage key of the invocation id.
    pub const INVOCATION_BAGGAGE: &'static str = "azure.ai.agentserver.invocation_id";

    /// The identity of the agent `config` describes.
    pub fn from_config(config: &FoundryHostConfig) -> Self {
        Self {
            agent_name: config.agent_name.clone(),
            agent_version: config.agent_version.clone(),
            agent_id: config.agent_id.clone(),
            project_id: config.project_id.clone(),
            session_id: config.session_id.clone(),
        }
    }
}

#[cfg(feature = "otel")]
impl opentelemetry_sdk::trace::SpanProcessor for FoundryEnrichmentSpanProcessor {
    fn on_start(&self, span: &mut opentelemetry_sdk::trace::Span, cx: &opentelemetry::Context) {
        use opentelemetry::baggage::BaggageExt as _;
        use opentelemetry::trace::Span as _;
        use opentelemetry::KeyValue;
        let mut set = |key: &'static str, value: &str| {
            if !value.is_empty() {
                span.set_attribute(KeyValue::new(key, value.to_string()));
            }
        };
        set("microsoft.foundry", "True");
        set("microsoft.foundry.project.id", &self.project_id);
        set("gen_ai.agent.name", &self.agent_name);
        set("gen_ai.agent.version", &self.agent_version);
        set("gen_ai.agent.id", &self.agent_id);
        let baggage = cx.baggage();
        let entry = |key: &str| baggage.get(key).map(|v| v.as_str().to_string());
        let session = entry(Self::SESSION_BAGGAGE).unwrap_or_else(|| self.session_id.clone());
        set("microsoft.session.id", &session);
        set(
            "gen_ai.conversation.id",
            &entry(Self::CONVERSATION_BAGGAGE).unwrap_or_default(),
        );
        set(
            "azure.ai.agentserver.invocations.invocation_id",
            &entry(Self::INVOCATION_BAGGAGE).unwrap_or_default(),
        );
    }

    fn on_end(&self, _span: opentelemetry_sdk::trace::SpanData) {}

    fn force_flush(&self) -> opentelemetry_sdk::error::OTelSdkResult {
        Ok(())
    }

    fn shutdown_with_timeout(
        &self,
        _timeout: std::time::Duration,
    ) -> opentelemetry_sdk::error::OTelSdkResult {
        Ok(())
    }
}

/// Export telemetry as the agentserver does: to Application Insights when
/// `APPLICATIONINSIGHTS_CONNECTION_STRING` is configured, else to the OTLP
/// endpoint when one is, else nowhere (`Ok(None)`). The pipeline is
/// installed as the global `tracing` subscriber; keep it alive and shut it
/// down on exit. Every span carries the Foundry identity attributes
/// ([`FoundryEnrichmentSpanProcessor`]).
#[cfg(feature = "otel")]
pub fn configure_observability(
    config: &FoundryHostConfig,
) -> Result<Option<agent_framework_core::observability::export::OtelPipeline>> {
    use agent_framework_core::observability::export::OtelExport;
    let service = if config.agent_name.is_empty() {
        "agent-framework-foundry-hosting".to_string()
    } else {
        config.agent_name.clone()
    };
    let export = if !config.appinsights_connection_string.is_empty() {
        OtelExport::new(service).with_azure_monitor(config.appinsights_connection_string.clone())
    } else if !config.otlp_endpoint.is_empty() {
        OtelExport::new(service).with_endpoint(config.otlp_endpoint.clone())
    } else {
        return Ok(None);
    };
    let pipeline = export
        .with_span_processor(FoundryEnrichmentSpanProcessor::from_config(config))
        .build()?;
    pipeline.install()?;
    Ok(Some(pipeline))
}

/// Wrap a protocol router with the shared routes and headers: `GET
/// /readiness`, `POST /_agent/before-snapshot` and `/_agent/after-restore`
/// (and their `/_before_snapshot` / `/_after_restore` aliases), an
/// `x-request-id` on every response (the caller's, or a new one), and
/// `x-platform-server`.
pub fn with_host_routes(router: Router, options: HostOptions) -> Router {
    let mut version = server_version();
    for segment in &options.server_version_segments {
        version.push(' ');
        version.push_str(segment);
    }
    let version =
        HeaderValue::from_str(&version).unwrap_or(HeaderValue::from_static("agent-framework-rs"));
    let before = options.before_snapshot.clone();
    let before_alias = before.clone();
    let span_options = options.clone();
    let lifecycle_state = Arc::new(tokio::sync::Mutex::new(LifecycleState::default()));
    let (after, after_state) = (options.clone(), lifecycle_state.clone());
    let (after_alias, after_alias_state) = (options.clone(), lifecycle_state);
    router
        .route(
            "/readiness",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    HEALTHY_BODY,
                )
            }),
        )
        .route(
            "/_agent/before-snapshot",
            post(move |body| lifecycle(before.clone(), "before_snapshot", body)),
        )
        .route(
            "/_before_snapshot",
            post(move |body| lifecycle(before_alias.clone(), "before_snapshot", body)),
        )
        .route(
            "/_agent/after-restore",
            post(move |body| after_restore(after.clone(), after_state.clone(), body)),
        )
        .route(
            "/_after_restore",
            post(move |body| after_restore(after_alias.clone(), after_alias_state.clone(), body)),
        )
        .layer(axum::middleware::from_fn(
            move |request: Request, next: Next| {
                let version = version.clone();
                let span_options = span_options.clone();
                async move {
                    let request_id = request
                        .headers()
                        .get(headers::REQUEST_ID)
                        .cloned()
                        .unwrap_or_else(|| {
                            HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                                .unwrap_or(HeaderValue::from_static("unknown"))
                        });
                    let span = request_span(
                        &request,
                        request_id.to_str().unwrap_or_default(),
                        &span_options,
                    );
                    let mut response = {
                        use tracing::Instrument;
                        next.run(request).instrument(span).await
                    };
                    let h = response.headers_mut();
                    h.insert(HeaderName::from_static(headers::REQUEST_ID), request_id);
                    h.insert(HeaderName::from_static(headers::SERVER_VERSION), version);
                    response
                }
            },
        ))
}

/// Serve `router` on `0.0.0.0:<config.port>` until the process ends.
pub async fn serve(router: Router, config: &FoundryHostConfig) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", config.port))
        .await
        .map_err(|e| Error::Configuration(format!("failed to bind port {}: {e}", config.port)))?;
    axum::serve(listener, router)
        .await
        .map_err(|e| Error::Configuration(format!("server error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn traceparent_is_validated() {
        let ids = parse_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01");
        assert_eq!(
            ids,
            Some((
                "4bf92f3577b34da6a3ce929d0e0e4736".into(),
                "00f067aa0ba902b7".into()
            ))
        );
        for bad in [
            "",
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
            "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
            "00-4BF92F3577B34DA6A3CE929D0E0E4736-00f067aa0ba902b7-01",
        ] {
            assert_eq!(parse_traceparent(bad), None, "{bad}");
        }
    }
}
