//! Tests for the opt-in DevUI security middleware: the anti-DNS-rebinding
//! `Host`-header allowlist, the cross-site check on state-changing requests
//! and bearer-token auth (UPSTREAM_DRIFT.md §14).

mod common;

use agent_framework_hosting::AgentHost;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use tower::ServiceExt;

use common::MockAgent;

fn host() -> AgentHost {
    AgentHost::new().agent("assistant", MockAgent::new("assistant-1").arc())
}

async fn get_with_host(app: Router, uri: &str, host_header: Option<&str>) -> StatusCode {
    let mut builder = Request::builder().uri(uri);
    if let Some(h) = host_header {
        builder = builder.header(header::HOST, h);
    }
    let request = builder.body(Body::empty()).expect("request builds");
    app.oneshot(request)
        .await
        .expect("router responds")
        .status()
}

async fn get_with_bearer(app: Router, uri: &str, token: Option<&str>) -> StatusCode {
    let mut builder = Request::builder().uri(uri);
    if let Some(t) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let request = builder.body(Body::empty()).expect("request builds");
    app.oneshot(request)
        .await
        .expect("router responds")
        .status()
}

// ---------------------------------------------------------------------------
// Host-header guard (anti-DNS-rebinding)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn host_guard_rejects_disallowed_host() {
    let app = host()
        .with_allowed_hosts(vec!["localhost".to_string()])
        .into_router();
    let status = get_with_host(app, "/health", Some("evil.com")).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn host_guard_allows_allowlisted_host() {
    let app = host()
        .with_allowed_hosts(vec!["localhost".to_string()])
        .into_router();
    let status = get_with_host(app, "/health", Some("localhost")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn host_guard_allows_allowlisted_host_any_port() {
    let app = host()
        .with_allowed_hosts(vec!["localhost".to_string()])
        .into_router();
    let status = get_with_host(app, "/health", Some("localhost:54321")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn host_guard_allows_missing_host_header() {
    let app = host()
        .with_allowed_hosts(vec!["localhost".to_string()])
        .into_router();
    let status = get_with_host(app, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn into_router_default_has_no_host_guard() {
    // Opt-in: with no `with_allowed_hosts` call, into_router() is unchanged
    // and lets any Host header through.
    let app = host().into_router();
    let status = get_with_host(app, "/health", Some("evil.com")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn secure_router_rejects_disallowed_host_by_default() {
    let app = host().into_secure_router();
    let status = get_with_host(app, "/health", Some("evil.com")).await;
    assert_eq!(status, StatusCode::FORBIDDEN);
}

#[tokio::test]
async fn secure_router_allows_default_loopback_hosts() {
    let app = host().into_secure_router();
    assert_eq!(
        get_with_host(app.clone(), "/health", Some("localhost")).await,
        StatusCode::OK
    );
    assert_eq!(
        get_with_host(app.clone(), "/health", Some("127.0.0.1:9000")).await,
        StatusCode::OK
    );
    assert_eq!(get_with_host(app, "/health", None).await, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Bearer-token auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn bearer_auth_rejects_missing_token() {
    let app = host().with_bearer_token("secret").into_router();
    let status = get_with_bearer(app, "/health", None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn bearer_auth_rejects_wrong_token() {
    let app = host().with_bearer_token("secret").into_router();
    let status = get_with_bearer(app, "/health", Some("wrong")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn bearer_auth_allows_correct_token() {
    let app = host().with_bearer_token("secret").into_router();
    let status = get_with_bearer(app, "/health", Some("secret")).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn into_router_default_has_no_bearer_auth() {
    // Opt-in: with no `with_bearer_token` call, into_router() is unchanged
    // and requires no Authorization header.
    let app = host().into_router();
    let status = get_with_bearer(app, "/health", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn both_layers_compose() {
    let app = host()
        .with_bearer_token("secret")
        .with_allowed_hosts(vec!["localhost".to_string()])
        .into_router();

    // Bad host still rejected even with a correct token.
    let mut builder = Request::builder().uri("/health");
    builder = builder.header(header::HOST, "evil.com");
    builder = builder.header(header::AUTHORIZATION, "Bearer secret");
    let request = builder.body(Body::empty()).expect("request builds");
    let status = app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds")
        .status();
    assert_eq!(status, StatusCode::FORBIDDEN);

    // Good host, missing token.
    let mut builder = Request::builder().uri("/health");
    builder = builder.header(header::HOST, "localhost");
    let request = builder.body(Body::empty()).expect("request builds");
    let status = app
        .clone()
        .oneshot(request)
        .await
        .expect("router responds")
        .status();
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    // Good host, good token.
    let mut builder = Request::builder().uri("/health");
    builder = builder.header(header::HOST, "localhost");
    builder = builder.header(header::AUTHORIZATION, "Bearer secret");
    let request = builder.body(Body::empty()).expect("request builds");
    let status = app
        .oneshot(request)
        .await
        .expect("router responds")
        .status();
    assert_eq!(status, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Composed multi-surface security (HostingSecurity applied to the final app)
// ---------------------------------------------------------------------------

use agent_framework_hosting::a2a::A2ARouter;
use agent_framework_hosting::agui::AgUiRouter;
use agent_framework_hosting::openai_compat::OpenAiRouter;
use agent_framework_hosting::HostingSecurity;

/// Build an app that merges/nests all four hosting surfaces, guarded by a
/// single outer [`HostingSecurity`] bearer layer.
fn composed_secured_app() -> Router {
    let agent = MockAgent::new("assistant-1").arc();
    let app = AgentHost::new()
        .agent("assistant", agent.clone())
        .into_router()
        .merge(OpenAiRouter::for_agent("assistant", agent.clone()).into_router())
        .merge(
            AgUiRouter::for_agent("assistant", agent.clone())
                .path("/agui")
                .into_router(),
        )
        .nest(
            "/a2a",
            A2ARouter::for_agent("assistant", agent, "http://localhost:8080/a2a").into_router(),
        );
    HostingSecurity::new()
        .with_bearer_token("secret")
        .apply(app)
}

async fn post_status(app: Router, uri: &str, token: Option<&str>, body: &str) -> StatusCode {
    let mut builder = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(t) = token {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {t}"));
    }
    let request = builder
        .body(Body::from(body.to_string()))
        .expect("request builds");
    app.oneshot(request)
        .await
        .expect("router responds")
        .status()
}

/// Every execution endpoint across all four surfaces must reject a request with
/// no bearer token once the composed app is wrapped with `HostingSecurity` —
/// not just the DevUI routes `AgentHost` builds directly.
#[tokio::test]
async fn composed_security_rejects_missing_credentials_on_every_surface() {
    let endpoints = [
        ("/v1/responses", r#"{"model":"assistant","input":"hi"}"#),
        (
            "/v1/chat/completions",
            r#"{"model":"assistant","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/agui", r#"{"messages":[]}"#),
        (
            "/a2a/",
            r#"{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"kind":"message","role":"user","messageId":"m1","parts":[{"kind":"text","text":"hi"}]}}}"#,
        ),
    ];

    for (uri, body) in endpoints {
        let status = post_status(composed_secured_app(), uri, None, body).await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "endpoint {uri} must reject a missing bearer token"
        );
    }
}

/// With the correct token, the same endpoints pass the auth layer (they do not
/// return 401 — the handler runs).
#[tokio::test]
async fn composed_security_admits_valid_credentials_on_every_surface() {
    let endpoints = [
        ("/v1/responses", r#"{"model":"assistant","input":"hi"}"#),
        (
            "/v1/chat/completions",
            r#"{"model":"assistant","messages":[{"role":"user","content":"hi"}]}"#,
        ),
        ("/agui", r#"{"messages":[]}"#),
        (
            "/a2a/",
            r#"{"jsonrpc":"2.0","id":1,"method":"message/send","params":{"message":{"kind":"message","role":"user","messageId":"m1","parts":[{"kind":"text","text":"hi"}]}}}"#,
        ),
    ];

    for (uri, body) in endpoints {
        let status = post_status(composed_secured_app(), uri, Some("secret"), body).await;
        assert_ne!(
            status,
            StatusCode::UNAUTHORIZED,
            "endpoint {uri} must admit a valid bearer token"
        );
    }
}

// ---------------------------------------------------------------------------
// Cross-site state-changing requests (upstream #9575)
// ---------------------------------------------------------------------------

/// `POST /v1/responses` with the given browser headers, through a router that
/// has the loopback allowlist configured.
async fn post_as_browser(
    host_header: &str,
    sec_fetch_site: Option<&str>,
    origin: Option<&str>,
) -> StatusCode {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header(header::HOST, host_header)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(site) = sec_fetch_site {
        builder = builder.header("sec-fetch-site", site);
    }
    if let Some(o) = origin {
        builder = builder.header(header::ORIGIN, o);
    }
    let request = builder
        .body(Body::from(
            serde_json::json!({ "model": "assistant", "input": "hi" }).to_string(),
        ))
        .expect("request builds");
    host()
        .into_secure_router()
        .oneshot(request)
        .await
        .expect("router responds")
        .status()
}

/// A loopback `Host` allowlist does not stop a hostile page: the browser
/// sends `Host: localhost` for `fetch("http://localhost:8080/v1/responses")`
/// from *any* origin. `Sec-Fetch-Site` is what distinguishes the two.
#[tokio::test]
async fn a_cross_site_post_is_rejected() {
    assert_eq!(
        post_as_browser(
            "localhost:8080",
            Some("cross-site"),
            Some("https://evil.example")
        )
        .await,
        StatusCode::FORBIDDEN
    );
}

#[tokio::test]
async fn cross_site_is_matched_case_insensitively() {
    assert_eq!(
        post_as_browser("localhost:8080", Some("Cross-Site"), None).await,
        StatusCode::FORBIDDEN
    );
}

/// The DevUI page's own fetches are same-origin and must still work.
#[tokio::test]
async fn a_same_origin_post_is_allowed() {
    assert_eq!(
        post_as_browser(
            "localhost:8080",
            Some("same-origin"),
            Some("http://localhost:8080")
        )
        .await,
        StatusCode::OK
    );
}

/// An `Origin` that does not match what the request was addressed to is
/// refused even without `Sec-Fetch-Site` — an older browser, or a proxy that
/// dropped the header. Host and port are compared; the scheme is not, since a
/// server-side request carries no trustworthy one (see `origin_matches_host`).
#[tokio::test]
async fn a_mismatched_origin_is_rejected_without_sec_fetch_site() {
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("http://localhost:9999")).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("ws://localhost:8080")).await,
        StatusCode::FORBIDDEN,
        "only http and https origins are recognised"
    );
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("http://localhost:8080/evil")).await,
        StatusCode::FORBIDDEN,
        "a serialized origin has no path"
    );
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("http://evil.example:8080")).await,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("null")).await,
        StatusCode::FORBIDDEN,
        "an opaque origin names no host to match"
    );
}

/// A matching `Origin` passes, including when one side leaves the scheme's
/// default port implicit.
#[tokio::test]
async fn a_matching_origin_is_allowed() {
    assert_eq!(
        post_as_browser("localhost:8080", None, Some("http://localhost:8080")).await,
        StatusCode::OK
    );
    assert_eq!(
        post_as_browser("localhost", None, Some("http://localhost")).await,
        StatusCode::OK
    );
    assert_eq!(
        post_as_browser("localhost", None, Some("http://localhost:80")).await,
        StatusCode::OK,
        "an explicit default port is the same origin"
    );
    assert_eq!(
        post_as_browser("127.0.0.1:8080", None, Some("http://127.0.0.1:8080")).await,
        StatusCode::OK
    );
}

/// A non-browser client sends no `Origin` and is not the threat here; the
/// `Host` allowlist is what stands between it and a rebinding attack.
#[tokio::test]
async fn a_post_with_no_browser_headers_is_allowed() {
    assert_eq!(
        post_as_browser("localhost:8080", None, None).await,
        StatusCode::OK
    );
}

/// Only state-changing methods are checked: a cross-site `GET` cannot change
/// anything and script cannot read the reply without CORS headers, which this
/// router does not set.
#[tokio::test]
async fn a_cross_site_get_is_still_allowed() {
    let request = Request::builder()
        .uri("/v1/entities")
        .header(header::HOST, "localhost:8080")
        .header("sec-fetch-site", "cross-site")
        .header(header::ORIGIN, "https://evil.example")
        .body(Body::empty())
        .expect("request builds");
    let status = host()
        .into_secure_router()
        .oneshot(request)
        .await
        .expect("router responds")
        .status();
    assert_eq!(status, StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Bearer scheme parsing (upstream #9575)
// ---------------------------------------------------------------------------

async fn get_with_raw_authorization(value: &str) -> StatusCode {
    let request = Request::builder()
        .uri("/v1/entities")
        .header(header::AUTHORIZATION, value)
        .body(Body::empty())
        .expect("request builds");
    host()
        .with_bearer_token("secret")
        .into_router()
        .oneshot(request)
        .await
        .expect("router responds")
        .status()
}

/// RFC 7235 makes the auth scheme case-insensitive, and upstream matches it
/// that way; the credential is trimmed.
#[tokio::test]
async fn the_bearer_scheme_is_case_insensitive_and_the_token_is_trimmed() {
    assert_eq!(
        get_with_raw_authorization("bearer secret").await,
        StatusCode::OK
    );
    assert_eq!(
        get_with_raw_authorization("BEARER secret").await,
        StatusCode::OK
    );
    assert_eq!(
        get_with_raw_authorization("Bearer  secret ").await,
        StatusCode::OK
    );
}

#[tokio::test]
async fn another_scheme_or_a_wrong_token_is_still_refused() {
    assert_eq!(
        get_with_raw_authorization("Basic secret").await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        get_with_raw_authorization("Bearer wrong").await,
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        get_with_raw_authorization("Bearer secretx").await,
        StatusCode::UNAUTHORIZED,
        "a token with the right prefix is not the right token"
    );
}
