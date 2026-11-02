//! The Responses host server over a real agent, driven in-process.

mod common;

use std::sync::Arc;

use agent_framework_core::workflow::{FunctionExecutor, WorkflowBuilder};
use agent_framework_foundry_hosting::{FoundryHostConfig, ResponsesHostServer};
use axum::http::StatusCode;
use common::*;
use serde_json::json;

#[tokio::test]
async fn health_and_platform_headers() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    let (status, headers, body) = send(&router, get("/readiness")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body, r#"{"status":"healthy"}"#);
    assert!(headers.contains_key("x-request-id"));
    assert!(headers["x-platform-server"]
        .to_str()
        .unwrap()
        .contains("responses"));
    let mut request = get("/readiness");
    request
        .headers_mut()
        .insert("x-request-id", "abc".parse().unwrap());
    let (_, headers, _) = send(&router, request).await;
    assert_eq!(headers["x-request-id"], "abc");
    let (status, body) = json(&router, post("/_agent/before-snapshot", json!({}))).await;
    assert_eq!((status, body), (StatusCode::OK, json!({"status": "ok"})));
}

#[tokio::test]
async fn a_stored_turn_continues_from_previous_response_id() {
    let echo = Echo::default();
    let router = ResponsesHostServer::agent(agent(echo.clone())).into_router();
    let (status, first) = json(&router, post("/responses", json!({"input": "hello"}))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(first["status"], "completed");
    assert!(first["id"].as_str().unwrap().starts_with("caresp_"));
    assert_eq!(first["output"][0]["type"], "message");
    assert!(first["output"][0]["id"]
        .as_str()
        .unwrap()
        .starts_with("msg_"));
    assert_eq!(
        first["output"][0]["content"][0]["text"],
        "1 messages, last: hello"
    );

    let id = first["id"].as_str().unwrap();
    let (_, second) = json(
        &router,
        post(
            "/responses",
            json!({"input": "again", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(second["previous_response_id"], id);
    // The prior transcript (user + assistant) precedes the new input.
    assert_eq!(
        second["output"][0]["content"][0]["text"],
        "3 messages, last: again"
    );

    let (status, fetched) = json(&router, get(&format!("/responses/{id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["id"], id);
    let (_, items) = json(&router, get(&format!("/responses/{id}/input_items"))).await;
    assert_eq!(items["object"], "list");
    assert_eq!(items["data"][0]["content"][0]["text"], "hello");

    let (status, deleted) = json(
        &router,
        axum::http::Request::delete(format!("/responses/{id}"))
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(
        (status, deleted["deleted"].clone()),
        (StatusCode::OK, json!(true))
    );
    let (status, _) = json(&router, get(&format!("/responses/{id}"))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn conversations_and_unstored_turns() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    for (input, expected) in [("a", "1 messages, last: a"), ("b", "3 messages, last: b")] {
        let (_, r) = json(
            &router,
            post(
                "/responses",
                json!({"input": input, "conversation": {"id": "conv_1"}}),
            ),
        )
        .await;
        assert_eq!(r["conversation"]["id"], "conv_1");
        assert_eq!(r["output"][0]["content"][0]["text"], expected);
    }
    let (_, r) = json(
        &router,
        post("/responses", json!({"input": "x", "store": false})),
    )
    .await;
    let (status, _) = json(
        &router,
        get(&format!("/responses/{}", r["id"].as_str().unwrap())),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "store=false persists nothing"
    );

    for bad in [
        json!({"input": "x", "background": true, "store": false}),
        json!({"input": "x", "previous_response_id": "caresp_missing"}),
        json!({"input": "x", "previous_response_id": "a", "conversation": "b"}),
    ] {
        let (status, body) = json(&router, post("/responses", bad)).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(body["error"]["message"].is_string());
    }
}

#[tokio::test]
async fn streaming_emits_the_response_event_sequence() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    let (status, headers, body) = send(
        &router,
        post("/responses", json!({"input": "hi", "stream": true})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers["content-type"]
        .to_str()
        .unwrap()
        .starts_with("text/event-stream"));
    let events = sse_events(&body);
    let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        kinds,
        [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.completed",
        ]
    );
    for (i, (_, data)) in events.iter().enumerate() {
        assert_eq!(data["sequence_number"], i);
    }
    let deltas: String = events
        .iter()
        .filter(|(k, _)| k == "response.output_text.delta")
        .map(|(_, d)| d["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, "1 messages, last: hi");
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["status"], "completed");
    assert_eq!(
        completed["output"][0]["id"], events[2].1["item"]["id"],
        "the final item keeps its streamed id"
    );
}

#[tokio::test]
async fn background_responses_complete_and_can_be_polled() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "bg", "background": true})),
    )
    .await;
    assert_eq!(queued["status"], "queued");
    assert_eq!(queued["background"], true);
    let id = queued["id"].as_str().unwrap().to_string();
    let mut last = json!(null);
    for _ in 0..100 {
        let (_, polled) = json(&router, get(&format!("/responses/{id}"))).await;
        if polled["status"] == "completed" {
            last = polled;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        last["output"][0]["content"][0]["text"],
        "1 messages, last: bg"
    );
    let (status, _) = json(&router, post(&format!("/responses/{id}/cancel"), json!({}))).await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a finished response cannot be cancelled"
    );
}

/// Never answers, so a background run stays in flight.
struct Hang;

#[async_trait::async_trait]
impl agent_framework_core::client::ChatClient for Hang {
    async fn get_response(
        &self,
        _: Vec<agent_framework_core::prelude::Message>,
        _: agent_framework_core::prelude::ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::prelude::ChatResponse> {
        std::future::pending().await
    }
    async fn get_streaming_response(
        &self,
        _: Vec<agent_framework_core::prelude::Message>,
        _: agent_framework_core::prelude::ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_background_response_can_be_cancelled() {
    let hang = Arc::new(agent_framework_core::prelude::Agent::builder(Hang).build());
    let router = ResponsesHostServer::agent(hang).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "x", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap();
    let (status, cancelled) =
        json(&router, post(&format!("/responses/{id}/cancel"), json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{cancelled}");
    assert_eq!(cancelled["status"], "cancelled");
    let (_, polled) = json(&router, get(&format!("/responses/{id}"))).await;
    assert_eq!(polled["status"], "cancelled");
}

#[tokio::test]
async fn workflows_and_prefixes() {
    let router = ResponsesHostServer::workflow(|request| {
        assert!(request.response_id.starts_with("caresp_"));
        WorkflowBuilder::new()
            .add_executor(Arc::new(FunctionExecutor::new(
                "upper",
                |msg, ctx| async move {
                    let text = msg.as_str().unwrap_or_default().to_uppercase();
                    ctx.yield_output(json!(text)).await
                },
            )))
            .set_start("upper")
            .build()
    })
    .with_workflow_input(|request| Ok(json!(request.messages[0].text())))
    .with_prefix("/v1")
    .with_config(FoundryHostConfig {
        agent_name: "shouter".into(),
        ..FoundryHostConfig::local()
    })
    .into_router();
    let (status, r) = json(&router, post("/v1/responses", json!({"input": "quiet"}))).await;
    assert_eq!(status, StatusCode::OK, "{r}");
    assert_eq!(r["model"], "shouter");
    assert_eq!(r["output"][0]["content"][0]["text"], "QUIET");
}
