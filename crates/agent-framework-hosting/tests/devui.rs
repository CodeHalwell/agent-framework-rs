//! DevUI-style API tests: entity discovery and `/v1/responses` execution
//! (agent and workflow, JSON and SSE), plus error paths.

mod common;

use agent_framework_core::agent::ChatAgent;
use agent_framework_hosting::AgentHost;
use axum::http::StatusCode;
use serde_json::json;

use common::{
    echo_workflow, get_json, parse_sse, parse_sse_json, post_json, post_raw, MockAgent,
    StreamingAgent,
};

fn host() -> AgentHost {
    AgentHost::new()
        .agent(
            "assistant",
            MockAgent::new("assistant-1").named("Assistant").arc(),
        )
        .workflow("echo", echo_workflow())
}

#[tokio::test]
async fn health_reports_entity_count() {
    let (status, body) = get_json(host().into_router(), "/health").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "healthy");
    assert_eq!(body["entities_count"], 2);
    assert_eq!(body["framework"], "agent_framework");
}

#[tokio::test]
async fn entities_list_has_agent_and_workflow() {
    let (status, body) = get_json(host().into_router(), "/v1/entities").await;
    assert_eq!(status, StatusCode::OK);
    let entities = body["entities"].as_array().unwrap();
    assert_eq!(entities.len(), 2);

    let agent = entities.iter().find(|e| e["id"] == "assistant").unwrap();
    assert_eq!(agent["type"], "agent");
    assert_eq!(agent["name"], "Assistant");
    assert_eq!(agent["framework"], "agent_framework");
    assert_eq!(agent["source"], "in_memory");

    let workflow = entities.iter().find(|e| e["id"] == "echo").unwrap();
    assert_eq!(workflow["type"], "workflow");
    assert_eq!(workflow["name"], "Echo Workflow");
    assert_eq!(workflow["description"], "Echoes its input");
}

#[tokio::test]
async fn entity_info_agent_and_workflow() {
    let app = host().into_router();
    let (status, agent) = get_json(app.clone(), "/v1/entities/assistant/info").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(agent["id"], "assistant");
    assert_eq!(agent["type"], "agent");

    let (status, workflow) = get_json(app, "/v1/entities/echo/info").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(workflow["type"], "workflow");
    assert_eq!(workflow["start_executor_id"], "echo");
    assert_eq!(workflow["input_schema"], json!({ "type": "string" }));
}

#[tokio::test]
async fn entity_info_unknown_is_404() {
    let (status, body) = get_json(host().into_router(), "/v1/entities/nope/info").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["error"]["message"].as_str().unwrap().contains("nope"));
}

#[tokio::test]
async fn responses_agent_non_stream() {
    let body = json!({
        "input": "hello world",
        "metadata": { "entity_id": "assistant" },
    });
    let (status, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["object"], "response");
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["output_text"], "echo: hello world");
    // Aggregated output message.
    let text = &resp["output"][0]["content"][0]["text"];
    assert_eq!(text, "echo: hello world");
}

#[tokio::test]
async fn responses_entity_id_from_model_field() {
    // A plain OpenAI client that only sets `model` should still route.
    let body = json!({ "input": "hi", "model": "assistant" });
    let (status, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["output_text"], "echo: hi");
}

#[tokio::test]
async fn responses_agent_stream_event_sequence() {
    let body = json!({
        "input": "stream me",
        "stream": true,
        "metadata": { "entity_id": "assistant" },
    });
    let (status, text) = post_raw(host().into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let raw = parse_sse(&text);
    assert_eq!(raw.last().unwrap(), "[DONE]", "stream ends with [DONE]");

    let events = parse_sse_json(&text);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types.first(), Some(&"response.created"));
    assert!(types.contains(&"response.in_progress"));
    assert!(types.contains(&"response.output_item.added"));
    assert!(types.contains(&"response.content_part.added"));
    assert!(types.contains(&"response.output_text.delta"));
    assert_eq!(types.last(), Some(&"response.completed"));

    // The delta carries the reply text.
    let delta = events
        .iter()
        .find(|e| e["type"] == "response.output_text.delta")
        .unwrap();
    assert_eq!(delta["delta"], "echo: stream me");

    // Sequence numbers are strictly increasing.
    let seqs: Vec<u64> = events
        .iter()
        .map(|e| e["sequence_number"].as_u64().unwrap())
        .collect();
    assert!(seqs.windows(2).all(|w| w[0] < w[1]));

    // The completed response aggregates the text.
    let completed = events.last().unwrap();
    assert_eq!(completed["response"]["output_text"], "echo: stream me");
}

#[tokio::test]
async fn responses_uses_real_chat_agent() {
    // A real ChatAgent (not just the mock) flows end-to-end via a mock client.
    use agent_framework_core::client::{ChatClient, ChatStream};
    use agent_framework_core::types::{ChatOptions, ChatResponse};
    use async_trait::async_trait;

    struct FixedClient;
    #[async_trait]
    impl ChatClient for FixedClient {
        async fn get_response(
            &self,
            _m: Vec<agent_framework_core::types::Message>,
            _o: ChatOptions,
        ) -> agent_framework_core::error::Result<ChatResponse> {
            Ok(ChatResponse::from_text("real agent reply"))
        }
        async fn get_streaming_response(
            &self,
            _m: Vec<agent_framework_core::types::Message>,
            _o: ChatOptions,
        ) -> agent_framework_core::error::Result<ChatStream> {
            unreachable!("hosting uses run(), not run_stream()")
        }
    }

    let agent = ChatAgent::builder(FixedClient)
        .name("real")
        .description("a real chat agent")
        .build();
    let host = AgentHost::new().agent("real", agent);

    let body = json!({ "input": "hi", "metadata": { "entity_id": "real" } });
    let (status, resp) = post_json(host.into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["output_text"], "real agent reply");
}

#[tokio::test]
async fn responses_workflow_non_stream_returns_outputs() {
    let body = json!({
        "input": "data",
        "metadata": { "entity_id": "echo" },
    });
    let (status, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(resp["status"], "completed");
    assert_eq!(resp["outputs"][0], "workflow: data");
    assert_eq!(resp["output"][0]["content"][0]["text"], "workflow: data");
}

#[tokio::test]
async fn responses_workflow_stream_maps_events() {
    let body = json!({
        "input": "data",
        "stream": true,
        "metadata": { "entity_id": "echo" },
    });
    let (status, text) = post_raw(host().into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(types.first(), Some(&"response.created"));
    assert_eq!(types.last(), Some(&"response.completed"));

    // Executor lifecycle mapped to output items.
    let invoked = events
        .iter()
        .find(|e| {
            e["type"] == "response.output_item.added" && e["item"]["type"] == "executor_action"
        })
        .expect("executor_action added");
    assert_eq!(invoked["item"]["executor_id"], "echo");
    assert_eq!(invoked["item"]["status"], "in_progress");

    let done = events
        .iter()
        .find(|e| e["type"] == "response.output_item.done")
        .expect("executor_action done");
    assert_eq!(done["item"]["status"], "completed");

    // Workflow output mapped to a message item.
    let output_msg = events.iter().any(|e| {
        e["type"] == "response.output_item.added"
            && e["item"]["type"] == "message"
            && e["item"]["content"][0]["text"] == "workflow: data"
    });
    assert!(output_msg, "workflow output mapped to a message item");

    // A workflow_event.completed debug event is present (status/superstep/etc).
    assert!(events
        .iter()
        .any(|e| e["type"] == "response.workflow_event.completed"));

    assert_eq!(
        events.last().unwrap()["response"]["outputs"][0],
        "workflow: data"
    );
}

#[tokio::test]
async fn responses_missing_entity_id_is_400() {
    let body = json!({ "input": "hi" });
    let (status, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(resp["error"]["code"], "missing_entity_id");
}

#[tokio::test]
async fn responses_unknown_entity_is_404() {
    let body = json!({ "input": "hi", "metadata": { "entity_id": "ghost" } });
    let (status, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(resp["error"]["message"].as_str().unwrap().contains("ghost"));
}

#[tokio::test]
async fn responses_agent_stream_emits_incremental_deltas() {
    // A multi-delta streaming agent yields one `response.output_text.delta` per
    // update, and the terminal `response.completed` aggregates them.
    let host = AgentHost::new().agent(
        "streamer",
        StreamingAgent::new("s1", vec!["Hel", "lo ", "world"]).arc(),
    );

    let body = json!({
        "input": "go",
        "stream": true,
        "metadata": { "entity_id": "streamer" },
    });
    let (status, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let deltas: Vec<&str> = events
        .iter()
        .filter(|e| e["type"] == "response.output_text.delta")
        .map(|e| e["delta"].as_str().unwrap())
        .collect();
    assert_eq!(deltas, vec!["Hel", "lo ", "world"], "one delta per update");

    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert_eq!(completed["response"]["output_text"], "Hello world");
}
