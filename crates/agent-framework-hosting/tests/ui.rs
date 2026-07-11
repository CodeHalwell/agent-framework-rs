//! Embedded DevUI page tests: served at `/` and `/ui` through the real
//! `AgentHost` router, `text/html`, contains the entities-fetch marker, and
//! ships no external URLs.

mod common;

use agent_framework_hosting::AgentHost;
use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use axum::Router;
use tower::ServiceExt;

use common::MockAgent;

/// `GET uri`, returning `(status, content_type, body_text)`.
async fn get(app: Router, uri: &str) -> (StatusCode, String, String) {
    let res = app
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .expect("router responds");
    let status = res.status();
    let content_type = res
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = axum::body::to_bytes(res.into_body(), usize::MAX)
        .await
        .expect("body collects");
    (
        status,
        content_type,
        String::from_utf8_lossy(&bytes).into_owned(),
    )
}

fn host() -> AgentHost {
    AgentHost::new().agent("assistant", MockAgent::new("a1").named("Assistant").arc())
}

#[tokio::test]
async fn root_serves_html_debug_page() {
    let (status, content_type, body) = get(host().into_router(), "/").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        content_type.contains("text/html"),
        "content-type: {content_type}"
    );

    // It is an HTML document that fetches the entity list.
    assert!(body.contains("<!DOCTYPE html>") || body.contains("<!doctype html>"));
    assert!(body.contains("<title>"));
    assert!(
        body.contains("/v1/entities"),
        "entities-fetch marker present"
    );
    assert!(body.contains("/v1/responses"), "responses marker present");
}

#[tokio::test]
async fn ui_path_also_serves_the_page() {
    let (status, content_type, body) = get(host().into_router(), "/ui").await;
    assert_eq!(status, StatusCode::OK);
    assert!(content_type.contains("text/html"));
    assert!(body.contains("/v1/entities"));
}

#[tokio::test]
async fn page_has_no_external_urls() {
    let (_, _, body) = get(host().into_router(), "/").await;

    // Any absolute http(s) URL must point at localhost (a doc/example only).
    for marker in ["http://", "https://"] {
        let mut idx = 0;
        while let Some(pos) = body[idx..].find(marker) {
            let start = idx + pos + marker.len();
            let rest = &body[start..];
            assert!(
                rest.starts_with("localhost") || rest.starts_with("127.0.0.1"),
                "external URL found near: {}",
                &body[idx + pos..(idx + pos + 48).min(body.len())]
            );
            idx = start;
        }
    }

    // No external resource references at all.
    assert!(
        !body.contains("src=\"//"),
        "no protocol-relative script/img"
    );
    assert!(!body.contains("href=\"//"), "no protocol-relative link");
    assert!(!body.contains("cdn"), "no CDN references");
}

#[tokio::test]
async fn page_documents_its_limits() {
    let (_, _, body) = get(host().into_router(), "/").await;

    // It advertises itself as the debug UI, not the upstream React DevUI.
    assert!(body.contains("React DevUI"));
    // It renders a resume-not-supported notice for pending request_info events.
    assert!(body.contains("request_info"));
    assert!(body.to_lowercase().contains("not supported"));
}
