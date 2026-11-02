//! `FoundryToolbox` against an in-process MCP server double.

use std::sync::{Arc, Mutex};

use agent_framework_azure::StaticTokenCredential;
use agent_framework_foundry_hosting::{FoundryRequestContext, FoundryToolbox};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

type Seen = Arc<Mutex<Vec<HeaderMap>>>;

async fn mcp(State(seen): State<Seen>, headers: HeaderMap, Json(body): Json<Value>) -> Response {
    seen.lock().unwrap().push(headers);
    let Some(id) = body.get("id").cloned() else {
        return StatusCode::ACCEPTED.into_response();
    };
    let reply = match body["method"].as_str() {
        Some("initialize") => json!({"jsonrpc": "2.0", "id": id, "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "toolbox", "version": "1"},
        }}),
        Some("tools/list") => json!({"jsonrpc": "2.0", "id": id, "result": {"tools": [
            {"name": "search", "description": "Search.", "inputSchema": {"type": "object"}},
        ]}}),
        _ => {
            json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}})
        }
    };
    Json(reply).into_response()
}

#[tokio::test]
async fn toolbox_requests_carry_auth_features_and_call_id() {
    let seen: Seen = Arc::default();
    let app = Router::new()
        .route(
            "/toolboxes/tb/mcp",
            post(mcp).delete(|| async { StatusCode::OK }),
        )
        .with_state(seen.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    // Without a URL, TOOLBOX_ENDPOINT, or project endpoint + TOOLBOX_NAME
    // there is nothing to connect to.
    let missing = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .build_with(|name| (name == "FOUNDRY_PROJECT_ENDPOINT").then(|| format!("http://{addr}")))
        .await;
    assert!(missing.is_err());

    let tool = FoundryToolbox::new(Arc::new(StaticTokenCredential::new("tok")))
        .url(format!("http://{addr}/toolboxes/tb/mcp"))
        .request_context(FoundryRequestContext {
            call_id: Some("call-9".into()),
            ..Default::default()
        })
        .build_with(|name| (name == "FOUNDRY_AGENT_TOOLSET_FEATURES").then(|| "Extra=1".into()))
        .await
        .unwrap();
    let definitions = tool.tool_definitions().await.unwrap();
    assert_eq!(definitions.len(), 1);
    assert!(definitions[0].name.contains("search"));
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty());
    for headers in seen.iter() {
        assert_eq!(headers["authorization"], "Bearer tok");
        assert_eq!(headers["foundry-features"], "Toolboxes=V1Preview,Extra=1");
        assert_eq!(headers["x-agent-foundry-call-id"], "call-9");
    }
}
