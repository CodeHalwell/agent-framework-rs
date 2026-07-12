//! Tests for the opt-in DevUI security middleware: the anti-DNS-rebinding
//! `Host`-header allowlist and bearer-token auth (UPSTREAM_DRIFT.md §14).

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
