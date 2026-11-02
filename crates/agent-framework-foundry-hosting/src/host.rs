//! The container host (upstream agentserver `AgentServerHost`): the routes
//! every protocol shares (`GET /readiness`, the snapshot lifecycle hooks),
//! request-id and server-version headers, error bodies, and serving.

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
    server_version_segments: Vec<String>,
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
    let after = options.after_restore.clone();
    let before_alias = before.clone();
    let after_alias = after.clone();
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
            post(move |body| lifecycle(after.clone(), "after_restore", body)),
        )
        .route(
            "/_after_restore",
            post(move |body| lifecycle(after_alias.clone(), "after_restore", body)),
        )
        .layer(axum::middleware::from_fn(
            move |request: Request, next: Next| {
                let version = version.clone();
                async move {
                    let request_id = request
                        .headers()
                        .get(headers::REQUEST_ID)
                        .cloned()
                        .unwrap_or_else(|| {
                            HeaderValue::from_str(&uuid::Uuid::new_v4().to_string())
                                .unwrap_or(HeaderValue::from_static("unknown"))
                        });
                    let mut response = next.run(request).await;
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
