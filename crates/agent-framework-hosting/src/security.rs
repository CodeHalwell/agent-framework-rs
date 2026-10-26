//! Security middleware for the DevUI-style router: a local-access guard (an
//! anti-DNS-rebinding `Host`-header allowlist plus a cross-site check on
//! state-changing requests) and optional bearer-token auth.
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

/// The local-access guard: an anti-DNS-rebinding `Host` allowlist plus a
/// cross-site check on state-changing requests.
///
/// Reject (403) requests whose `Host` header is not in the allowlist.
/// Requests with **no** `Host` header are allowed through (e.g. HTTP/1.0
/// clients, or in-process `tower::ServiceExt::oneshot` calls that never set
/// one).
///
/// Then, for a state-changing method, reject a request the browser marks as
/// cross-site (upstream #9575). The allowlist alone does not cover this: a
/// page on any origin can make the user's browser `POST` to
/// `http://localhost:8080/v1/responses`, and the browser sends `Host:
/// localhost`, so the allowlist passes and a hostile page gets to drive the
/// agents this router serves. `Sec-Fetch-Site` and `Origin` are the headers
/// that distinguish that from the local DevUI page's own fetches, and a
/// browser will not let script forge either.
///
/// Applied together because both say the same thing — this router is serving
/// a local caller — and both are configured by the same
/// [`crate::AgentHost::with_allowed_hosts`] /
/// [`crate::AgentHost::into_secure_router`] choice. A caller who wants the
/// router reachable cross-origin leaves the allowlist unset, which is
/// upstream's `AllowRemoteAccess`.
///
/// Not ported: upstream also requires the *peer address* to be loopback.
/// That needs `ConnectInfo<SocketAddr>`, which only exists when the caller
/// serves the router with `into_make_service_with_connect_info`, so a
/// middleware here cannot require it without changing what `into_router`
/// returns. The `Host` allowlist is this port's stand-in, as it has been
/// since the guard was written.
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

    if is_state_changing(request.method()) && !browser_request_is_allowed(&request) {
        return forbidden("Cross-site request rejected");
    }

    next.run(request).await
}

/// Whether a method can change server state. `GET`/`HEAD`/`OPTIONS`/`TRACE`
/// cannot, so a cross-site *read* is left to the browser's own CORS rules
/// (this router sets no permissive CORS headers, so script cannot read the
/// reply anyway).
fn is_state_changing(method: &axum::http::Method) -> bool {
    !matches!(
        *method,
        axum::http::Method::GET
            | axum::http::Method::HEAD
            | axum::http::Method::OPTIONS
            | axum::http::Method::TRACE
    )
}

/// Whether a state-changing request is *not* a cross-site browser request.
///
/// Two independent signals, mirroring upstream:
///
/// * `Sec-Fetch-Site: cross-site` is a flat no. Every current browser sends
///   this header and script cannot set it.
/// * An `Origin` must then match the request's own `Host` — scheme, host and
///   port. A request with no `Origin` at all is allowed: that is a non-browser
///   client (`curl`, an SDK, a test), which is not the threat here, and the
///   `Host` allowlist above already stands between it and a rebinding attack.
fn browser_request_is_allowed(request: &Request) -> bool {
    let headers = request.headers();

    for value in headers.get_all(SEC_FETCH_SITE) {
        if value
            .to_str()
            .is_ok_and(|v| v.trim().eq_ignore_ascii_case("cross-site"))
        {
            return false;
        }
    }

    let mut origins = headers.get_all(header::ORIGIN).into_iter();
    let Some(origin) = origins.next() else {
        return true;
    };
    if origins.next().is_some() {
        // More than one `Origin` is malformed; no browser sends it.
        return false;
    }
    let Ok(origin) = origin.to_str() else {
        return false;
    };

    // The `Host` header is what the request was addressed to; compare the
    // origin against it. No `Host` means there is nothing to compare, and the
    // allowlist above already let it through for the same reason.
    let Some(host) = headers.get(header::HOST).and_then(|h| h.to_str().ok()) else {
        return true;
    };

    origin_matches_host(origin, host)
}

/// `Sec-Fetch-Site`, which `axum::http::header` has no constant for.
const SEC_FETCH_SITE: &str = "sec-fetch-site";

/// Whether `origin` (a serialized origin: `scheme "://" host [":" port]`)
/// names the same host and port as `host` (a `Host` header value).
///
/// Anything that is not a bare origin — a path, query or fragment, userinfo,
/// an opaque `"null"`, a scheme other than `http`/`https`, an unparseable
/// value — fails. Ports are compared after filling in the scheme's default,
/// so `http://localhost` and `localhost:80` match.
///
/// The **scheme** is not compared, unlike upstream: a server-side request
/// carries no trustworthy scheme (`Host` has none, `request.uri()` has none
/// on an HTTP/1.1 origin-form request, and `X-Forwarded-Proto` is a hop's
/// claim). It costs nothing here — only one process can be listening on a
/// given host and port, so a page served from the same host and port over the
/// other scheme does not exist to attack from.
fn origin_matches_host(origin: &str, host: &str) -> bool {
    let (scheme, rest) = match origin.split_once("://") {
        Some((s, r)) => (s, r),
        None => return false,
    };
    let default_port = match scheme.to_ascii_lowercase().as_str() {
        "https" => 443,
        "http" => 80,
        _ => return false,
    };
    // A bare origin has no path, query, fragment or userinfo.
    if rest.is_empty()
        || rest.contains('/')
        || rest.contains('?')
        || rest.contains('#')
        || rest.contains('@')
    {
        return false;
    }

    let (origin_host, origin_port) = split_host_port(rest);
    let (target_host, target_port) = split_host_port(host);

    let origin_port = match origin_port {
        Some(p) => match p.parse::<u16>() {
            Ok(p) => p,
            Err(_) => return false,
        },
        None => default_port,
    };
    let target_port = match target_port {
        // The target's scheme is not in the `Host` header. A request that
        // carried an explicit port must match it exactly; one that did not is
        // on its scheme's default port, and the origin's scheme decides which
        // that is — the same assumption upstream makes.
        Some(p) => match p.parse::<u16>() {
            Ok(p) => p,
            Err(_) => return false,
        },
        None => default_port,
    };

    origin_port == target_port && hosts_match(origin_host, target_host)
}

/// Split a `host[:port]` authority, keeping an IPv6 literal's brackets with
/// the host. Returns `(host, port)`.
fn split_host_port(authority: &str) -> (&str, Option<&str>) {
    let host = strip_port(authority);
    let port = authority[host.len()..].strip_prefix(':');
    (host, port)
}

/// Compare two host names: IPv6 brackets stripped, ASCII case-insensitive.
/// Unlike upstream this does not normalise IP literals (`127.0.0.1` vs
/// `127.000.000.001`), matching the exact-string comparison the `Host`
/// allowlist above already uses.
fn hosts_match(a: &str, b: &str) -> bool {
    fn unbracket(h: &str) -> &str {
        h.strip_prefix('[')
            .and_then(|r| r.strip_suffix(']'))
            .unwrap_or(h)
    }
    unbracket(a).eq_ignore_ascii_case(unbracket(b))
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
///
/// The scheme is matched case-insensitively and the credential trimmed, per
/// RFC 7235 and upstream, and the token itself is compared in constant time
/// (see [`fixed_time_eq`]).
pub async fn bearer_auth(
    State(expected): State<BearerToken>,
    request: Request,
    next: Next,
) -> Response {
    let provided = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(bearer_credential);

    match provided {
        Some(token) if fixed_time_eq(token.as_bytes(), expected.0.as_bytes()) => {
            next.run(request).await
        }
        _ => unauthorized(),
    }
}

/// The credential from an `Authorization: Bearer <token>` header value, or
/// `None` if the header does not use the bearer scheme.
fn bearer_credential(value: &str) -> Option<&str> {
    let (scheme, credential) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("Bearer")
        .then(|| credential.trim())
}

/// Compare two byte strings without an early exit on the first differing
/// byte, so the time taken does not reveal how much of a guess was right.
///
/// Equivalent to .NET's `CryptographicOperations.FixedTimeEquals`, which
/// upstream's DevUI token check uses, and like it this leaks the *length* of
/// the configured token (the length check returns immediately) — not a secret
/// worth the complexity of hiding.
fn fixed_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    // `black_box` keeps the accumulate-then-test shape from being rewritten
    // into a short-circuiting comparison.
    std::hint::black_box(diff) == 0
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
