//! Security middleware for the DevUI-style router: an anti-DNS-rebinding
//! `Host`-header allowlist and optional bearer-token auth.
//!
//! Both are **opt-in**, wired through [`crate::AgentHost::with_allowed_hosts`]
//! and [`crate::AgentHost::with_bearer_token`]; [`crate::AgentHost::into_router`]
//! applies a layer only when the corresponding config was set, so the default
//! (unconfigured) router is byte-for-byte unchanged. See
//! [`crate::AgentHost::into_secure_router`] for a secure-by-default entry point
//! that fills in the host allowlist automatically.
//!
//! Mirrors the upstream Python DevUI server's `host_header` middleware and
//! `auth_enabled` bearer check (UPSTREAM_DRIFT.md §14).

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{header, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::responses::openai_error;

/// The default `Host` allowlist: loopback only (by name and literal), any
/// port. Used both by [`AllowedHosts::default_localhost`] and by
/// [`crate::AgentHost::into_secure_router`].
pub fn default_hosts() -> Vec<String> {
    vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "[::1]".to_string(),
        "::1".to_string(),
    ]
}

/// A `Host`-header allowlist, compared with any port stripped.
#[derive(Clone)]
pub struct AllowedHosts(Arc<Vec<String>>);

impl AllowedHosts {
    /// Build an allowlist from exact host names (no port; ports are stripped
    /// from the incoming `Host` header before comparison, so any port matches).
    pub fn new(hosts: Vec<String>) -> Self {
        Self(Arc::new(hosts))
    }

    /// The default allowlist: loopback only, any port.
    pub fn default_localhost() -> Self {
        Self::new(default_hosts())
    }

    fn allows(&self, host_header: &str) -> bool {
        let host = strip_port(host_header);
        self.0.iter().any(|allowed| allowed == host)
    }
}

/// Strip a trailing `:port` from a `Host` header value. IPv6 literals
/// (`[::1]:8080`) keep their brackets; bare IPv6 (`::1`, no brackets, no port)
/// passes through unchanged.
fn strip_port(host_header: &str) -> &str {
    if let Some(rest) = host_header.strip_prefix('[') {
        // "[::1]" or "[::1]:8080" — keep through the closing bracket. `rest`
        // starts one byte into `host_header`, so shift the found index by 1.
        if let Some(end) = rest.find(']') {
            return &host_header[..=end + 1];
        }
        return host_header;
    }
    match host_header.rfind(':') {
        // A bare IPv6 literal (e.g. "::1") has multiple colons and no port;
        // only strip when there's exactly one colon.
        Some(idx) if host_header.matches(':').count() == 1 => &host_header[..idx],
        _ => host_header,
    }
}

/// Anti-DNS-rebinding guard: reject (403) requests whose `Host` header is not
/// in the allowlist. Requests with **no** `Host` header are allowed through
/// (e.g. HTTP/1.0 clients, or in-process `tower::ServiceExt::oneshot` calls
/// that never set one).
pub async fn host_guard(
    State(allowed): State<AllowedHosts>,
    request: Request,
    next: Next,
) -> Response {
    if let Some(host) = request.headers().get(header::HOST) {
        let host_str = match host.to_str() {
            Ok(h) => h,
            Err(_) => return forbidden("Invalid Host header"),
        };
        if !allowed.allows(host_str) {
            return forbidden(&format!("Host '{host_str}' is not allowed"));
        }
    }
    next.run(request).await
}

fn forbidden(message: &str) -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(openai_error(
            message,
            "invalid_request_error",
            Some("host_not_allowed"),
        )),
    )
        .into_response()
}

/// A configured bearer token, compared exactly against the `Authorization`
/// header.
#[derive(Clone)]
pub struct BearerToken(Arc<String>);

impl BearerToken {
    pub fn new(token: impl Into<String>) -> Self {
        Self(Arc::new(token.into()))
    }
}

/// Bearer-token auth: reject (401) requests missing `Authorization: Bearer
/// <token>` or presenting the wrong token.
pub async fn bearer_auth(
    State(expected): State<BearerToken>,
    request: Request,
    next: Next,
) -> Response {
    let provided = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    match provided {
        Some(token) if token == expected.0.as_str() => next.run(request).await,
        _ => unauthorized(),
    }
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(openai_error(
            "Missing or invalid bearer token",
            "invalid_request_error",
            Some("unauthorized"),
        )),
    )
        .into_response()
}

/// A reusable hosting-security layer for a **composed** router.
///
/// [`crate::AgentHost::into_router`] applies its bearer/host middleware only to
/// the DevUI routes it builds — routers `merge`d or `nest`ed onto it afterwards
/// (an [`OpenAiRouter`](crate::openai_compat::OpenAiRouter), an
/// [`A2ARouter`](crate::a2a::A2ARouter), an
/// [`AgUiRouter`](crate::agui::AgUiRouter)) are **not** covered. When you serve
/// more than one protocol surface, build the whole app first, then wrap the
/// final router with this so every execution endpoint is guarded uniformly:
///
/// ```no_run
/// # use agent_framework_core::agent::Agent;
/// use agent_framework_hosting::{
///     a2a::A2ARouter, openai_compat::OpenAiRouter, security::HostingSecurity, AgentHost,
/// };
///
/// # async fn demo(assistant: Agent) -> std::io::Result<()> {
/// let app = AgentHost::new()
///     .agent("assistant", assistant.clone())
///     .into_router()
///     .merge(OpenAiRouter::for_agent("assistant", assistant.clone()).into_router())
///     .nest(
///         "/a2a",
///         A2ARouter::for_agent("assistant", assistant, "http://localhost:8080/a2a").into_router(),
///     );
///
/// // Apply auth + host allowlist to the ENTIRE composed app, after all merges.
/// let app = HostingSecurity::new()
///     .with_bearer_token("secret")
///     .with_default_localhost_hosts()
///     .apply(app);
///
/// let listener = tokio::net::TcpListener::bind(("127.0.0.1", 8080)).await?;
/// axum::serve(listener, app).await
/// # }
/// ```
#[derive(Clone, Default)]
pub struct HostingSecurity {
    bearer_token: Option<String>,
    allowed_hosts: Option<Vec<String>>,
}

impl HostingSecurity {
    /// An unconfigured layer (applying it is a no-op until you set a guard).
    pub fn new() -> Self {
        Self::default()
    }

    /// Require `Authorization: Bearer <token>` on every request.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(token.into());
        self
    }

    /// Reject requests whose `Host` header is not in `hosts` (anti-DNS-
    /// rebinding; ports ignored).
    pub fn with_allowed_hosts(mut self, hosts: Vec<String>) -> Self {
        self.allowed_hosts = Some(hosts);
        self
    }

    /// Apply the default loopback `Host` allowlist (`localhost`, `127.0.0.1`,
    /// `[::1]`, any port).
    pub fn with_default_localhost_hosts(mut self) -> Self {
        self.allowed_hosts = Some(default_hosts());
        self
    }

    /// Whether any guard is configured (applying an empty layer is a no-op).
    pub fn is_configured(&self) -> bool {
        self.bearer_token.is_some() || self.allowed_hosts.is_some()
    }

    /// Wrap `router` with the configured guards. Layers apply outermost-last-
    /// added-first, so the host guard is added after (and thus runs before) the
    /// bearer check: a rebinding attempt is rejected before token comparison.
    pub fn apply(&self, mut router: axum::Router) -> axum::Router {
        if let Some(token) = &self.bearer_token {
            router = router.layer(axum::middleware::from_fn_with_state(
                BearerToken::new(token.clone()),
                bearer_auth,
            ));
        }
        if let Some(hosts) = &self.allowed_hosts {
            router = router.layer(axum::middleware::from_fn_with_state(
                AllowedHosts::new(hosts.clone()),
                host_guard,
            ));
        }
        router
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_port_handles_ipv4_ipv6_and_bare_hosts() {
        assert_eq!(strip_port("localhost"), "localhost");
        assert_eq!(strip_port("localhost:8080"), "localhost");
        assert_eq!(strip_port("127.0.0.1"), "127.0.0.1");
        assert_eq!(strip_port("127.0.0.1:3000"), "127.0.0.1");
        assert_eq!(strip_port("[::1]"), "[::1]");
        assert_eq!(strip_port("[::1]:8080"), "[::1]");
        assert_eq!(strip_port("::1"), "::1");
    }

    #[test]
    fn default_localhost_allows_common_loopback_forms() {
        let allowed = AllowedHosts::default_localhost();
        assert!(allowed.allows("localhost"));
        assert!(allowed.allows("localhost:1234"));
        assert!(allowed.allows("127.0.0.1"));
        assert!(allowed.allows("127.0.0.1:9999"));
        assert!(allowed.allows("[::1]:8080"));
        assert!(!allowed.allows("evil.com"));
        assert!(!allowed.allows("evil.com:80"));
    }
}
