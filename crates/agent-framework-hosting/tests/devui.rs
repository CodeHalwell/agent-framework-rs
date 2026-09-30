//! DevUI-style API tests: entity discovery and `/v1/responses` execution
//! (agent and workflow, JSON and SSE), plus error paths.

mod common;

use agent_framework_core::agent::Agent;
use agent_framework_hosting::AgentHost;
use axum::http::StatusCode;
use serde_json::json;

use common::{
    echo_workflow, get_json, parse_sse, parse_sse_json, post_json, post_raw, MockAgent, StreamStep,
    StreamingAgent, ToolCallingAgent,
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
    // A real Agent (not just the mock) flows end-to-end via a mock client.
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

    let agent = Agent::builder(FixedClient)
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

/// The terminal event must follow the response's `status`, not the optional
/// `incomplete_details`. Those two came apart once an unfamiliar provider
/// reason started producing an `incomplete` response with no
/// schema-nameable detail — and keying on the detail announced
/// `response.completed` around exactly the payload this path exists to flag.
#[tokio::test]
async fn an_unfamiliar_finish_reason_streams_response_incomplete() {
    let host = AgentHost::new().agent(
        "assistant",
        MockAgent::new("assistant-1")
            .with_finish_reason("model_context_window_exceeded")
            .arc(),
    );
    let body = json!({
        "input": "stream me",
        "stream": true,
        "metadata": { "entity_id": "assistant" },
    });
    let (status, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(
        types.last(),
        Some(&"response.incomplete"),
        "a cut-off turn must not close with response.completed"
    );

    let terminal = events.last().unwrap();
    assert_eq!(terminal["response"]["status"], "incomplete");
    // No schema-nameable detail for this reason, but the reason survives.
    assert!(terminal["response"]["incomplete_details"].is_null());
    assert_eq!(
        terminal["response"]["x_finish_reason"],
        "model_context_window_exceeded"
    );
}

/// The two OpenAI does name still fill `incomplete_details` and still close
/// with `response.incomplete`.
#[tokio::test]
async fn a_content_filtered_turn_streams_response_incomplete_with_detail() {
    let host = AgentHost::new().agent(
        "assistant",
        MockAgent::new("assistant-1")
            .with_finish_reason("content_filter")
            .arc(),
    );
    let body = json!({
        "input": "stream me",
        "stream": true,
        "metadata": { "entity_id": "assistant" },
    });
    let (status, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let terminal = events.last().unwrap();
    assert_eq!(terminal["type"], "response.incomplete");
    assert_eq!(
        terminal["response"]["incomplete_details"]["reason"],
        "content_filter"
    );
}

/// The Responses output array is heterogeneous: an assistant message and a
/// function call are *sibling items*, not a message with calls attached.
/// Core leaves a `FunctionCallContent` intact for the caller to execute, so
/// serializing only the text would tell the client to run a call it was
/// never shown.
#[tokio::test]
async fn declared_calls_become_function_call_output_items() {
    let host = AgentHost::new().agent(
        "assistant",
        ToolCallingAgent::new("a1")
            .with_text("checking now")
            .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
            .with_call("call_2", "get_time", r#"{"tz":"UTC"}"#)
            .arc(),
    );
    let body = json!({ "input": "go", "metadata": { "entity_id": "assistant" } });
    let (status, resp) = post_json(host.into_router(), "/v1/responses", &body).await;
    assert_eq!(status, StatusCode::OK);

    let output = resp["output"].as_array().expect("output array");
    assert_eq!(output.len(), 3, "the message, then one item per call");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "checking now");

    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["call_id"], "call_1");
    assert_eq!(output[1]["name"], "get_weather");
    assert_eq!(
        output[1]["arguments"], r#"{"city":"Oslo"}"#,
        "arguments are a JSON *string*, as the wire format has it"
    );
    assert_eq!(output[1]["status"], "completed");
    assert_ne!(
        output[1]["id"], output[1]["call_id"],
        "the item id names this output item; call_id names the call to echo back"
    );

    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(output[2]["call_id"], "call_2");
    assert_eq!(output[2]["name"], "get_time");

    // `output_text` still carries only the text — a call is not narration.
    assert_eq!(resp["output_text"], "checking now");
}

/// A turn that is only a call gets no message item on the buffered path.
/// OpenAI omits it, and an empty assistant message would read to a client as
/// a blank answer rather than as work to do.
#[tokio::test]
async fn a_call_only_turn_has_no_message_item() {
    let host = AgentHost::new().agent(
        "assistant",
        ToolCallingAgent::new("a1")
            .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
            .arc(),
    );
    let body = json!({ "input": "go", "metadata": { "entity_id": "assistant" } });
    let (_, resp) = post_json(host.into_router(), "/v1/responses", &body).await;
    let output = resp["output"].as_array().expect("output array");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "function_call");
}

/// An ordinary turn is unchanged: exactly one message item, no call items.
#[tokio::test]
async fn an_ordinary_turn_still_serializes_one_message_item() {
    let body = json!({ "input": "hi", "metadata": { "entity_id": "assistant" } });
    let (_, resp) = post_json(host().into_router(), "/v1/responses", &body).await;
    let output = resp["output"].as_array().expect("output array");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"][0]["text"], "echo: hi");
}

/// Streaming announces each call once as a sibling output item, then feeds
/// it argument fragments. A provider may stream one call across several
/// updates, and re-announcing the item each time would read to a client as
/// several distinct calls.
#[tokio::test]
async fn streamed_calls_are_announced_once_then_fed_argument_deltas() {
    let host = AgentHost::new().agent(
        "assistant",
        ToolCallingAgent::new("a1")
            .streaming(vec![
                StreamStep::Text("checking".into()),
                StreamStep::Call {
                    call_id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: r#"{"city":"#.into(),
                },
                StreamStep::Call {
                    call_id: "call_2".into(),
                    name: "get_time".into(),
                    arguments: r#"{"tz":"UTC"}"#.into(),
                },
                StreamStep::Call {
                    call_id: "call_1".into(),
                    name: "get_weather".into(),
                    arguments: r#""Oslo"}"#.into(),
                },
            ])
            .arc(),
    );
    let body = json!({
        "input": "go",
        "stream": true,
        "metadata": { "entity_id": "assistant" },
    });
    let (status, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_json(&text);

    // Each call is announced exactly once, as its own output item.
    let added: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| {
            e["type"] == "response.output_item.added" && e["item"]["type"] == "function_call"
        })
        .collect();
    assert_eq!(
        added.len(),
        2,
        "one announcement per call, not per fragment"
    );
    assert_eq!(added[0]["item"]["call_id"], "call_1");
    assert_eq!(added[0]["item"]["name"], "get_weather");
    assert_eq!(
        added[0]["item"]["arguments"], "",
        "the announcement carries no arguments; the deltas do"
    );
    assert_eq!(added[0]["item"]["status"], "in_progress");
    assert_eq!(added[1]["item"]["call_id"], "call_2");

    // The message holds output_index 0 (the preamble announced it), so the
    // calls start at 1 and keep those indices.
    assert_eq!(added[0]["output_index"], 1);
    assert_eq!(added[1]["output_index"], 2);

    let arg_deltas: Vec<&serde_json::Value> = events
        .iter()
        .filter(|e| e["type"] == "response.function_call_arguments.delta")
        .collect();
    assert_eq!(arg_deltas.len(), 3, "one delta per scripted fragment");
    assert_eq!(arg_deltas[0]["delta"], r#"{"city":"#);
    assert_eq!(arg_deltas[1]["delta"], r#"{"tz":"UTC"}"#);
    assert_eq!(arg_deltas[2]["delta"], r#""Oslo"}"#);

    // A later fragment routes back to the item that was announced for it.
    let call_1_item = added[0]["item"]["id"].as_str().unwrap();
    assert_eq!(arg_deltas[0]["item_id"], call_1_item);
    assert_eq!(
        arg_deltas[2]["item_id"], call_1_item,
        "the second fragment of call_1 belongs to call_1's item"
    );
    assert_eq!(arg_deltas[2]["output_index"], 1);
    assert_ne!(arg_deltas[1]["item_id"], call_1_item);

    // Sequence numbers stay monotonic across the interleaved event kinds.
    let seqs: Vec<u64> = events
        .iter()
        .filter_map(|e| e["sequence_number"].as_u64())
        .collect();
    assert!(
        seqs.windows(2).all(|w| w[0] < w[1]),
        "sequence numbers must strictly increase: {seqs:?}"
    );

    // The terminal payload carries the same items, under the same ids the
    // client already saw — otherwise it cannot correlate them.
    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    let output = completed["response"]["output"]
        .as_array()
        .expect("output array");
    assert_eq!(output.len(), 3, "the message, then both calls");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["id"], call_1_item);
    assert_eq!(output[1]["call_id"], "call_1");
    assert_eq!(
        output[1]["arguments"], r#"{"city":"Oslo"}"#,
        "the aggregate carries the reassembled arguments, not a fragment"
    );
    assert_eq!(output[2]["call_id"], "call_2");
    assert_eq!(output[2]["id"], added[1]["item"]["id"]);
}

/// A streamed turn that is only a call still carries the message item the
/// preamble announced at `output_index` 0 — dropping it would shift every
/// call one index away from the event that announced it.
#[tokio::test]
async fn a_streamed_call_only_turn_keeps_the_announced_message_item() {
    let host = AgentHost::new().agent(
        "assistant",
        ToolCallingAgent::new("a1")
            .with_call("call_1", "get_weather", r#"{"city":"Oslo"}"#)
            .arc(),
    );
    let body = json!({
        "input": "go",
        "stream": true,
        "metadata": { "entity_id": "assistant" },
    });
    let (_, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    let events = parse_sse_json(&text);

    let announced = events
        .iter()
        .find(|e| e["type"] == "response.output_item.added" && e["item"]["type"] == "message")
        .expect("the preamble announced a message item");
    assert_eq!(announced["output_index"], 0);

    let completed = events.last().unwrap();
    let output = completed["response"]["output"]
        .as_array()
        .expect("output array");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(
        output[0]["id"], announced["item"]["id"],
        "and the terminal payload completes that same item"
    );
    assert_eq!(output[1]["type"], "function_call");
    assert_eq!(output[1]["call_id"], "call_1");
}

/// An ordinary streamed turn emits no call events at all — the new event
/// kinds are additive, not a change to the existing sequence.
#[tokio::test]
async fn an_ordinary_stream_emits_no_call_events() {
    let host = AgentHost::new().agent(
        "streamer",
        StreamingAgent::new("s1", vec!["Hel", "lo"]).arc(),
    );
    let body = json!({
        "input": "go",
        "stream": true,
        "metadata": { "entity_id": "streamer" },
    });
    let (_, text) = post_raw(host.into_router(), "/v1/responses", body.to_string()).await;
    let events = parse_sse_json(&text);
    assert!(
        !events
            .iter()
            .any(|e| e["type"] == "response.function_call_arguments.delta"),
        "no calls declared, so no argument deltas"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["type"] == "response.output_item.added")
            .count(),
        1,
        "only the preamble's message item"
    );
}
