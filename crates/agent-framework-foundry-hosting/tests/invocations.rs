#![cfg(feature = "session-store")]
//! The Invocations host server, driven in-process.

mod common;

use std::sync::Arc;

use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};
use agent_framework_foundry_hosting::{FoundryHostConfig, InvocationsHostServer, WorkflowTurn};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use common::*;
use serde_json::json;

#[tokio::test]
async fn sessions_persist_across_invocations() {
    let echo = Echo::default();
    let router = InvocationsHostServer::agent(agent(echo.clone())).into_router();
    let (status, headers, body) = send(
        &router,
        post(
            "/invocations?agent_session_id=s1",
            json!({"message": "one"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(headers["x-agent-session-id"], "s1");
    assert!(headers.contains_key("x-agent-invocation-id"));
    let reply: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(reply["response"], "1 messages, last: one");

    let (_, reply) = json(
        &router,
        post(
            "/invocations?agent_session_id=s1",
            json!({"message": "two"}),
        ),
    )
    .await;
    assert_eq!(
        reply["response"], "3 messages, last: two",
        "history came from the session"
    );
    let (_, reply) = json(
        &router,
        post(
            "/invocations?agent_session_id=s2",
            json!({"message": "fresh"}),
        ),
    )
    .await;
    assert_eq!(reply["response"], "1 messages, last: fresh");

    let mut request = post("/invocations?agent_session_id=s1", json!({"message": "x"}));
    request
        .headers_mut()
        .insert("x-agent-invocation-id", "inv-1".parse().unwrap());
    let (_, headers, _) = send(&router, request).await;
    assert_eq!(headers["x-agent-invocation-id"], "inv-1");
}

#[tokio::test]
async fn streaming_frames_and_bad_requests() {
    let router = InvocationsHostServer::agent(agent(Echo::default())).into_router();
    let (status, _, body) = send(
        &router,
        post(
            "/invocations?agent_session_id=s",
            json!({"message": "hi", "stream": true}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&body);
    let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(kinds, ["delta", "delta", "done"]);
    let text: String = events[..2]
        .iter()
        .map(|(_, d)| d["text"].as_str().unwrap())
        .collect();
    assert_eq!(text, "1 messages, last: hi");
    assert_eq!(events[2].1["session_id"], "s");

    for bad in [
        json!({"message": 1}),
        json!({"message": "x", "stream": "yes"}),
        json!({"message": "x", "options": {"tools": []}}),
        json!([1]),
    ] {
        let (status, body) = json(&router, post("/invocations", bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["error"].is_string());
    }
    let (status, body) = json(&router, get("/invocations/abc")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body["error"]["code"], "not_found");
}

#[tokio::test]
async fn hosted_requests_are_scoped_to_the_platform_user() {
    let config = FoundryHostConfig {
        is_hosted: true,
        session_id: "sandbox".into(),
        ..FoundryHostConfig::local()
    };
    let router = InvocationsHostServer::agent(agent(Echo::default()))
        .with_config(config)
        .into_router();
    let request = |user: Option<&str>, query: &str, message: &str| {
        let mut builder = Request::post(format!("/invocations{query}"))
            .header("content-type", "application/json")
            .header("x-agent-foundry-call-id", "call");
        if let Some(user) = user {
            builder = builder.header("x-agent-user-id", user);
        }
        builder
            .body(Body::from(json!({"message": message}).to_string()))
            .unwrap()
    };
    let (status, _) = json(&router, request(None, "", "x")).await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "a user id is required"
    );
    let (status, _) = json(&router, request(Some("u1"), "?agent_session_id=other", "x")).await;
    assert_eq!(
        status,
        StatusCode::INTERNAL_SERVER_ERROR,
        "a foreign session is refused"
    );

    let (status, headers, _) = send(&router, request(Some("u1"), "", "a")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers["x-agent-session-id"], "sandbox");
    let (_, reply) = json(&router, request(Some("u1"), "", "b")).await;
    assert_eq!(reply["response"], "3 messages, last: b");
    let (_, reply) = json(&router, request(Some("u2"), "", "c")).await;
    assert_eq!(
        reply["response"], "1 messages, last: c",
        "another user's session is separate"
    );
}

#[tokio::test]
async fn workflows_reply_with_frames() {
    let router = InvocationsHostServer::workflow(
        |_| {
            WorkflowBuilder::new()
                .add_executor(Arc::new(FunctionExecutor::new(
                    "double",
                    |msg, ctx| async move {
                        ctx.yield_output(json!(msg.as_i64().unwrap_or(0) * 2)).await
                    },
                )))
                .set_start("double")
                .build()
        },
        |request| {
            Ok(WorkflowTurn {
                input: request.body["n"].clone(),
                stream: request.body["stream"] == json!(true),
            })
        },
    )
    .into_router();
    let (status, body) = json(&router, post("/invocations", json!({"n": 21}))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["output"],
        json!([{"type": "output", "data": 42, "executor_id": "double"}])
    );
    let (_, _, body) = send(
        &router,
        post("/invocations", json!({"n": 2, "stream": true})),
    )
    .await;
    let events = sse_events(&body);
    assert_eq!(events[0].0, "output");
    assert_eq!(events[0].1["data"], 4);
    assert_eq!(events.last().unwrap().0, "done");
}

#[tokio::test]
async fn config_reads_the_platform_environment() {
    let vars = [
        ("FOUNDRY_AGENT_NAME", "bot"),
        ("FOUNDRY_AGENT_VERSION", "3"),
        ("FOUNDRY_HOSTING_ENVIRONMENT", "prod"),
        ("FOUNDRY_AGENT_SESSION_ID", "s"),
        ("PORT", "9000"),
    ];
    let lookup = |name: &str| {
        vars.iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.to_string())
    };
    let config = FoundryHostConfig::from_lookup(lookup).unwrap();
    assert_eq!(config.agent_id, "bot:3");
    assert!(config.is_hosted);
    assert_eq!(config.port, 9000);
    assert_eq!(config.sse_keepalive_interval, 15);
    assert!(FoundryHostConfig::from_lookup(|n| (n == "PORT").then(|| "x".to_string())).is_err());
    assert!(FoundryHostConfig::from_lookup(|n| match n {
        "FOUNDRY_HOSTING_ENVIRONMENT" => Some("1".into()),
        "FOUNDRY_AGENT_SESSION_GUID" => Some("NOT-A-GUID".into()),
        _ => None,
    })
    .is_err());
    let local = FoundryHostConfig::from_lookup(|_| None).unwrap();
    assert_eq!(
        (local.port, local.is_hosted, local.agent_id.as_str()),
        (8088, false, "")
    );
}
