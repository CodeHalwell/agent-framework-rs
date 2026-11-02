//! The Responses host server over a real agent, driven in-process.

mod common;

use std::sync::Arc;

use agent_framework_core::client::ChatClient;
use agent_framework_core::prelude::*;
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

/// Calls `lookup` once, then answers with text, reporting usage and a
/// reasoning step; streams each step's contents.
#[derive(Clone, Default)]
struct ToolUser {
    calls: Arc<std::sync::atomic::AtomicUsize>,
}

impl ToolUser {
    fn step(&self, messages: &[agent_framework_core::types::Message]) -> Vec<Content> {
        use agent_framework_core::types::{
            FunctionCallContent, TextReasoningContent, UsageContent, UsageDetails,
        };
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let has_result = messages.iter().any(|m| {
            m.contents
                .iter()
                .any(|c| matches!(c, Content::FunctionResult(_)))
        });
        let usage = Content::Usage(UsageContent {
            details: UsageDetails {
                input_token_count: Some(5),
                output_token_count: Some(2),
                ..Default::default()
            },
        });
        if has_result {
            vec![Content::text("found it"), usage]
        } else {
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "need a lookup".into(),
                    ..Default::default()
                }),
                Content::FunctionCall(FunctionCallContent::new(
                    "call_1",
                    "lookup",
                    Some(agent_framework_core::types::FunctionArguments::Raw(
                        r#"{"q":"x"}"#.into(),
                    )),
                )),
                usage,
            ]
        }
    }
}

#[async_trait::async_trait]
impl ChatClient for ToolUser {
    async fn get_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        let contents = self.step(&messages);
        Ok(ChatResponse {
            messages: vec![agent_framework_core::types::Message::with_contents(
                Role::assistant(),
                contents,
            )],
            ..Default::default()
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        use futures::StreamExt;
        let updates: Vec<_> = self
            .step(&messages)
            .into_iter()
            .map(|c| {
                Ok(ChatResponseUpdate {
                    contents: vec![c],
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

fn tool_agent() -> Arc<dyn SupportsAgentRun> {
    let tool = agent_framework_core::tools::FunctionTool::new(
        "lookup",
        "Look something up",
        json!({"type": "object", "properties": {"q": {"type": "string"}}}),
        |args: serde_json::Value| async move { Ok(json!({"found": args["q"]})) },
    )
    .into_definition();
    Arc::new(
        Agent::builder(ToolUser::default())
            .name("tools")
            .tool(tool)
            .build(),
    )
}

#[tokio::test]
async fn rich_output_items_are_emitted_in_order() {
    let router = ResponsesHostServer::agent(tool_agent()).into_router();
    let (status, response) = json(&router, post("/responses", json!({"input": "go"}))).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    let types: Vec<&str> = response["output"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["type"].as_str().unwrap())
        .collect();
    assert_eq!(
        types,
        [
            "reasoning",
            "function_call",
            "function_call_output",
            "message"
        ]
    );
    assert_eq!(response["output"][0]["summary"][0]["text"], "need a lookup");
    assert_eq!(response["output"][1]["arguments"], r#"{"q":"x"}"#);
    assert_eq!(response["output"][2]["call_id"], "call_1");
    assert_eq!(response["output"][2]["output"], r#"{"found":"x"}"#);
    assert_eq!(response["output_text"], "found it");
    assert_eq!(response["usage"]["input_tokens"], 10);
    assert_eq!(response["usage"]["total_tokens"], 14);

    // Streamed: the same items, with the function call streamed in place.
    let (_, _, body) = send(
        &router,
        post("/responses", json!({"input": "go", "stream": true})),
    )
    .await;
    let events = sse_events(&body);
    let kinds: Vec<&str> = events.iter().map(|(k, _)| k.as_str()).collect();
    let position = |k: &str| kinds.iter().position(|x| *x == k).unwrap();
    assert!(
        position("response.reasoning_summary_text.delta")
            < position("response.function_call_arguments.delta")
    );
    assert!(
        position("response.function_call_arguments.done") < position("response.output_text.delta")
    );
    let completed = &events.last().unwrap().1["response"];
    assert_eq!(completed["output"].as_array().unwrap().len(), 4);
    assert_eq!(completed["usage"]["output_tokens"], 4);
    for (i, (_, data)) in events.iter().enumerate() {
        assert_eq!(data["sequence_number"], i);
    }
}

#[tokio::test]
async fn a_workflow_pause_is_resumed_from_its_checkpoint_on_the_next_turn() {
    use agent_framework_core::workflow::{
        CheckpointStorage, InMemoryCheckpointStorage, RequestInfoExecutor, RequestResponse,
    };
    let storage: Arc<dyn CheckpointStorage> = Arc::new(InMemoryCheckpointStorage::new());
    let router = ResponsesHostServer::workflow(|request| {
        let asker = FunctionExecutor::new("asker", |msg, ctx| async move {
            if let Some(reply) = RequestResponse::from_message(&msg) {
                ctx.yield_output(json!(format!(
                    "hello {}",
                    reply.data["name"].as_str().unwrap_or("?")
                )))
                .await?;
            } else {
                ctx.send_message(json!({"question": "name?"})).await?;
            }
            Ok(())
        });
        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(asker))
            .add_executor(Arc::new(RequestInfoExecutor::new("ask")))
            .set_start("asker")
            .add_edge("asker", "ask");
        if let Some(storage) = request.checkpoint_storage.clone() {
            builder = builder.with_checkpointing(storage);
        }
        builder.build()
    })
    .with_checkpoint_store(storage)
    .into_router();

    let (status, first) = json(&router, post("/responses", json!({"input": "start"}))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let call = &first["output"][0];
    assert_eq!(call["type"], "function_call");
    assert_eq!(call["name"], "request_info");
    let call_id = call["call_id"].as_str().unwrap().to_string();
    let arguments: serde_json::Value =
        serde_json::from_str(call["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments["request"], json!({"question": "name?"}));
    let id = first["id"].as_str().unwrap();

    // Replies must answer the pending batch exactly.
    for bad in [
        json!("plain text"),
        json!([{"type": "function_call_output", "call_id": "call_nope", "output": "{}"}]),
    ] {
        let (status, body) = json(
            &router,
            post(
                "/responses",
                json!({"input": bad, "previous_response_id": id}),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    }

    let (status, second) = json(
        &router,
        post(
            "/responses",
            json!({
                "previous_response_id": id,
                "input": [{"type": "function_call_output", "call_id": call_id, "output": "{\"name\":\"Ada\"}"}],
            }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(second["output_text"], "hello Ada");
}

/// `(message count, conversation id, store, temperature)` per model call.
type Calls = Arc<std::sync::Mutex<Vec<(usize, Option<String>, Option<bool>, Option<f32>)>>>;

/// Records what each model call received; answers inside service
/// conversation `conv_1` when asked to store.
#[derive(Clone, Default)]
struct Recorder {
    calls: Calls,
}

#[async_trait::async_trait]
impl ChatClient for Recorder {
    async fn get_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        self.calls.lock().unwrap().push((
            messages.len(),
            options.conversation_id.clone(),
            options.store,
            options.temperature,
        ));
        let mut response = ChatResponse::from_text(format!("{} messages", messages.len()));
        if options.store == Some(true) {
            response.conversation_id = Some("conv_1".into());
        }
        Ok(response)
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        use futures::StreamExt;
        let response = self.get_response(messages, options).await?;
        let updates = vec![
            Ok(ChatResponseUpdate {
                contents: vec![Content::text(response.text())],
                role: Some(Role::assistant()),
                ..Default::default()
            }),
            Ok(ChatResponseUpdate {
                conversation_id: response.conversation_id.clone(),
                ..Default::default()
            }),
        ];
        Ok(futures::stream::iter(updates).boxed())
    }
}

async fn two_turns(router: &axum::Router, stream: bool) {
    let (status, _, body) = send(
        router,
        post("/responses", json!({"input": "one", "stream": stream})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let first: serde_json::Value = if stream {
        sse_events(&body).last().unwrap().1["response"].clone()
    } else {
        serde_json::from_str(&body).unwrap()
    };
    let id = first["id"].as_str().unwrap();
    let (status, _, body) = send(
        router,
        post(
            "/responses",
            json!({"input": "two", "previous_response_id": id, "stream": stream}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
}

#[tokio::test]
async fn history_sources_decide_what_the_model_receives() {
    use agent_framework_foundry_hosting::HistorySource;
    let build = |source: HistorySource, recorder: Recorder| {
        let agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(recorder).name("r").build());
        ResponsesHostServer::agent(agent)
            .with_history_source(source)
            .into_router()
    };

    // agent_server: the host replays the transcript and the client must not store.
    let recorder = Recorder::default();
    two_turns(&build(HistorySource::AgentServer, recorder.clone()), false).await;
    let calls = recorder.calls.lock().unwrap().clone();
    assert_eq!(calls[1].0, 3, "prior user + assistant + new input");
    assert!(calls.iter().all(|c| c.2 == Some(false) && c.1.is_none()));

    // service: only the new input, continuing the service conversation.
    for stream in [false, true] {
        let recorder = Recorder::default();
        two_turns(&build(HistorySource::Service, recorder.clone()), stream).await;
        let calls = recorder.calls.lock().unwrap().clone();
        let last = calls.last().unwrap();
        assert_eq!(last.0, 1, "stream={stream}: only this turn's input");
        assert_eq!(last.1.as_deref(), Some("conv_1"), "stream={stream}");
        assert_eq!(last.2, Some(true));
    }

    // A service continuation cannot run unstored; an unknown session fails.
    let router = build(HistorySource::Service, Recorder::default());
    let (status, _) = json(
        &router,
        post(
            "/responses",
            json!({"input": "x", "store": false, "previous_response_id": "caresp_x"}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn prepare_options_rewrites_the_callers_options() {
    let recorder = Recorder::default();
    let agent: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(recorder.clone()).name("r").build());
    let router = ResponsesHostServer::agent(agent)
        .with_prepare_options(|_request, mut options| {
            options.temperature = Some(0.25);
            Ok(options)
        })
        .into_router();
    let (status, _) = json(
        &router,
        post("/responses", json!({"input": "x", "temperature": 0.9})),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(recorder.calls.lock().unwrap()[0].3, Some(0.25));
}

/// Delays every write, to expose persistence racing the terminal event.
struct SlowStore(agent_framework_foundry_hosting::InMemoryResponseStore);

#[async_trait::async_trait]
impl agent_framework_foundry_hosting::ResponseStore for SlowStore {
    async fn get(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<agent_framework_foundry_hosting::StoredResponse>>
    {
        self.0.get(id).await
    }
    async fn put(
        &self,
        id: &str,
        response: agent_framework_foundry_hosting::StoredResponse,
    ) -> agent_framework_core::error::Result<()> {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        self.0.put(id, response).await
    }
    async fn delete(&self, id: &str) -> agent_framework_core::error::Result<bool> {
        self.0.delete(id).await
    }
    async fn conversation(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<Vec<agent_framework_core::types::Message>>>
    {
        self.0.conversation(id).await
    }
    async fn set_conversation(
        &self,
        id: &str,
        transcript: Vec<agent_framework_core::types::Message>,
    ) -> agent_framework_core::error::Result<()> {
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        self.0.set_conversation(id, transcript).await
    }
}

#[tokio::test]
async fn persistence_completes_before_the_terminal_event() {
    use http_body_util::BodyExt;
    use tower::ServiceExt;
    let store = Arc::new(SlowStore(Default::default()));
    let echo = Echo::default();
    let router = ResponsesHostServer::agent(agent(echo))
        .with_response_store(store)
        .into_router();
    let response = router
        .clone()
        .oneshot(post(
            "/responses",
            json!({"input": "hi", "stream": true, "conversation": "c1"}),
        ))
        .await
        .unwrap();
    let mut body = response.into_body();
    let mut text = String::new();
    let mut id = None;
    // Read until the terminal event, then drop the stream at once.
    while let Some(frame) = body.frame().await {
        if let Ok(data) = frame.unwrap().into_data() {
            text.push_str(&String::from_utf8_lossy(&data));
        }
        if let Some((_, data)) = sse_events(&text)
            .into_iter()
            .find(|(k, _)| k == "response.completed")
        {
            id = data["response"]["id"].as_str().map(str::to_string);
            break;
        }
    }
    drop(body);
    let id = id.expect("a terminal event");
    let (status, fetched) = json(&router, get(&format!("/responses/{id}"))).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "the response was stored before completion was reported"
    );
    assert_eq!(fetched["status"], "completed");
    let (status, next) = json(
        &router,
        post(
            "/responses",
            json!({"input": "again", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{next}");
    assert_eq!(next["output_text"], "3 messages, last: again");
    let (_, in_conversation) = json(
        &router,
        post("/responses", json!({"input": "more", "conversation": "c1"})),
    )
    .await;
    assert_eq!(in_conversation["output_text"], "3 messages, last: more");
}

#[tokio::test]
async fn background_streams_can_be_replayed() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "bg", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    let (status, _, body) = send(&router, get(&format!("/responses/{id}?stream=true"))).await;
    assert_eq!(status, StatusCode::OK);
    let events = sse_events(&body);
    assert_eq!(events.first().unwrap().0, "response.created");
    assert_eq!(events.last().unwrap().0, "response.completed");
    let (_, _, body) = send(
        &router,
        get(&format!("/responses/{id}?stream=true&starting_after=2")),
    )
    .await;
    let resumed = sse_events(&body);
    assert_eq!(resumed[0].1["sequence_number"], 3);
    assert_eq!(resumed.len(), events.len() - 3);

    // Only background responses can be replayed.
    let (_, plain) = json(&router, post("/responses", json!({"input": "x"}))).await;
    let (status, _) = json(
        &router,
        get(&format!(
            "/responses/{}?stream=true",
            plain["id"].as_str().unwrap()
        )),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) = json(
        &router,
        get(&format!("/responses/{id}?stream=true&starting_after=x")),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn the_file_store_survives_a_restart() {
    use agent_framework_foundry_hosting::FileResponseStore;
    let dir = std::env::temp_dir().join(format!("af-fstore-{}", uuid_like()));
    let build = || {
        ResponsesHostServer::agent(agent(Echo::default()))
            .with_response_store(Arc::new(FileResponseStore::new(&dir)))
            .into_router()
    };
    let (_, first) = json(
        &build(),
        post("/responses", json!({"input": "hello", "conversation": "c"})),
    )
    .await;
    let id = first["id"].as_str().unwrap().to_string();
    // A new server over the same directory.
    let router = build();
    let (status, fetched) = json(&router, get(&format!("/responses/{id}"))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(fetched["output_text"], first["output_text"]);
    let (_, next) = json(
        &router,
        post(
            "/responses",
            json!({"input": "again", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(next["output_text"], "3 messages, last: again");
    let (_, conv) = json(
        &router,
        post("/responses", json!({"input": "more", "conversation": "c"})),
    )
    .await;
    assert_eq!(conv["output_text"], "3 messages, last: more");
    let _ = std::fs::remove_dir_all(&dir);
}

fn uuid_like() -> String {
    format!(
        "{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

#[tokio::test]
async fn concurrent_turns_on_one_conversation_are_both_kept() {
    // A slow store widens the window between loading and persisting.
    let router = ResponsesHostServer::agent(agent(Echo::default()))
        .with_response_store(Arc::new(SlowStore(Default::default())))
        .into_router();
    let a = json(
        &router,
        post(
            "/responses",
            json!({"input": "first", "conversation": "shared"}),
        ),
    );
    let b = json(
        &router,
        post(
            "/responses",
            json!({"input": "second", "conversation": "shared"}),
        ),
    );
    let ((sa, _), (sb, _)) = tokio::join!(a, b);
    assert_eq!((sa, sb), (StatusCode::OK, StatusCode::OK));
    let (_, last) = json(
        &router,
        post(
            "/responses",
            json!({"input": "third", "conversation": "shared"}),
        ),
    )
    .await;
    // Both earlier turns (2 messages each) precede the new input.
    assert_eq!(last["output_text"], "5 messages, last: third");
}

/// Never answers.
struct Hanging;

#[async_trait::async_trait]
impl ChatClient for Hanging {
    async fn get_response(
        &self,
        _: Vec<agent_framework_core::types::Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        std::future::pending().await
    }
    async fn get_streaming_response(
        &self,
        _: Vec<agent_framework_core::types::Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        std::future::pending().await
    }
}

#[tokio::test]
async fn a_cancelled_background_stream_replay_ends() {
    let agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(Hanging).name("h").build());
    let router = ResponsesHostServer::agent(agent).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "x", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let (status, cancelled) =
        json(&router, post(&format!("/responses/{id}/cancel"), json!({}))).await;
    assert_eq!(status, StatusCode::OK, "{cancelled}");
    assert_eq!(cancelled["status"], "cancelled");
    let replay = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        send(&router, get(&format!("/responses/{id}?stream=true"))),
    )
    .await
    .expect("the replay ends after the cancellation");
    let events = sse_events(&replay.2);
    let last = events.last().unwrap();
    assert_eq!(last.0, "response.cancelled");
    assert_eq!(last.1["response"]["status"], "cancelled");
    for (i, (_, data)) in events.iter().enumerate() {
        assert_eq!(data["sequence_number"], i);
    }
}

#[tokio::test]
async fn deleting_a_running_background_response_ends_its_replay() {
    let agent: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(Hanging).name("h").build());
    let router = ResponsesHostServer::agent(agent).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "x", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let replay = tokio::spawn({
        let router = router.clone();
        let id = id.clone();
        async move { send(&router, get(&format!("/responses/{id}?stream=true"))).await }
    });
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let (status, _) = json(
        &router,
        axum::http::Request::delete(format!("/responses/{id}"))
            .body(axum::body::Body::empty())
            .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _, body) = tokio::time::timeout(std::time::Duration::from_secs(5), replay)
        .await
        .expect("the open replay ends when its response is deleted")
        .unwrap();
    assert_eq!(status, StatusCode::OK);
    assert_eq!(sse_events(&body).first().unwrap().0, "response.created");
    let (status, _) = json(&router, get(&format!("/responses/{id}?stream=true"))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Refuses to store a completed response or any conversation.
struct FailingStore(agent_framework_foundry_hosting::InMemoryResponseStore);

#[async_trait::async_trait]
impl agent_framework_foundry_hosting::ResponseStore for FailingStore {
    async fn get(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<agent_framework_foundry_hosting::StoredResponse>>
    {
        self.0.get(id).await
    }
    async fn put(
        &self,
        id: &str,
        response: agent_framework_foundry_hosting::StoredResponse,
    ) -> agent_framework_core::error::Result<()> {
        if response.response["status"] == "completed" {
            return Err(agent_framework_core::error::Error::Other(
                "disk full".into(),
            ));
        }
        self.0.put(id, response).await
    }
    async fn delete(&self, id: &str) -> agent_framework_core::error::Result<bool> {
        self.0.delete(id).await
    }
    async fn conversation(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<Vec<agent_framework_core::types::Message>>>
    {
        self.0.conversation(id).await
    }
    async fn set_conversation(
        &self,
        _: &str,
        _: Vec<agent_framework_core::types::Message>,
    ) -> agent_framework_core::error::Result<()> {
        Err(agent_framework_core::error::Error::Other(
            "disk full".into(),
        ))
    }
}

#[tokio::test]
async fn persistence_failures_fail_the_response() {
    let router = ResponsesHostServer::agent(agent(Echo::default()))
        .with_response_store(Arc::new(FailingStore(Default::default())))
        .into_router();

    let (status, body) = json(&router, post("/responses", json!({"input": "x"}))).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{body}");
    assert!(body["error"]["message"].is_string());

    let (_, _, body) = send(
        &router,
        post("/responses", json!({"input": "x", "stream": true})),
    )
    .await;
    let events = sse_events(&body);
    let (kind, data) = events.last().unwrap();
    assert_eq!(kind, "response.failed");
    assert_eq!(data["response"]["status"], "failed");
    assert!(data["response"]["error"]["message"].is_string());
    assert!(!events.iter().any(|(k, _)| k == "response.completed"));

    // A background response whose conversation write fails ends failed.
    let (_, queued) = json(
        &router,
        post(
            "/responses",
            json!({"input": "x", "background": true, "conversation": "c"}),
        ),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    let mut last = json!(null);
    for _ in 0..100 {
        let (_, polled) = json(&router, get(&format!("/responses/{id}"))).await;
        if polled["status"] == "failed" || polled["status"] == "completed" {
            last = polled;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(last["status"], "failed", "{last}");
}

#[tokio::test]
async fn an_unfinished_predecessor_cannot_be_continued() {
    let agent_: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(Hanging).name("h").build());
    let router = ResponsesHostServer::agent(agent_).into_router();
    let (_, queued) = json(
        &router,
        post("/responses", json!({"input": "first", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    let (status, body) = json(
        &router,
        post(
            "/responses",
            json!({"input": "next", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body["error"]["message"].as_str().unwrap().contains("still"));
}

#[tokio::test]
async fn a_cancelled_predecessor_keeps_its_input() {
    // The predecessor hangs; the continuation runs on a responsive agent
    // sharing the store.
    let store: Arc<dyn agent_framework_foundry_hosting::ResponseStore> =
        Arc::new(agent_framework_foundry_hosting::InMemoryResponseStore::default());
    let hanging: Arc<dyn SupportsAgentRun> = Arc::new(Agent::builder(Hanging).name("h").build());
    let first = ResponsesHostServer::agent(hanging)
        .with_response_store(store.clone())
        .into_router();
    let (_, queued) = json(
        &first,
        post("/responses", json!({"input": "first", "background": true})),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    json(&first, post(&format!("/responses/{id}/cancel"), json!({}))).await;
    let second = ResponsesHostServer::agent(agent(Echo::default()))
        .with_response_store(store)
        .into_router();
    let (status, body) = json(
        &second,
        post(
            "/responses",
            json!({"input": "next", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["output_text"], "2 messages, last: next");
}

/// Calls `danger` until it sees a result, then reports it.
#[derive(Clone, Default)]
struct WantsDanger;

#[async_trait::async_trait]
impl ChatClient for WantsDanger {
    async fn get_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        _: ChatOptions,
    ) -> agent_framework_core::error::Result<ChatResponse> {
        use agent_framework_core::types::FunctionCallContent;
        let result = messages
            .iter()
            .flat_map(|m| &m.contents)
            .find_map(|c| match c {
                Content::FunctionResult(r) => Some(r.result.clone().unwrap_or_default()),
                _ => None,
            });
        let contents = match result {
            Some(result) => vec![Content::text(format!("done: {result}"))],
            None => vec![Content::FunctionCall(FunctionCallContent::new(
                "c1",
                "danger",
                Some(agent_framework_core::types::FunctionArguments::Raw(
                    "{}".into(),
                )),
            ))],
        };
        Ok(ChatResponse {
            messages: vec![agent_framework_core::types::Message::with_contents(
                Role::assistant(),
                contents,
            )],
            ..Default::default()
        })
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<agent_framework_core::types::Message>,
        options: ChatOptions,
    ) -> agent_framework_core::error::Result<agent_framework_core::client::ChatStream> {
        use futures::StreamExt;
        let response = self.get_response(messages, options).await?;
        let updates: Vec<_> = response.messages[0]
            .contents
            .iter()
            .cloned()
            .map(|c| {
                Ok(ChatResponseUpdate {
                    contents: vec![c],
                    role: Some(Role::assistant()),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

#[tokio::test]
async fn approval_requests_round_trip_through_mcp_approval_responses() {
    let runs = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = runs.clone();
    let tool = agent_framework_core::tools::FunctionTool::new(
        "danger",
        "Needs approval",
        json!({"type": "object"}),
        move |_args: serde_json::Value| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(json!("ran"))
            }
        },
    )
    .into_definition()
    .require_approval();
    let agent: Arc<dyn SupportsAgentRun> = Arc::new(
        Agent::builder(WantsDanger)
            .name("careful")
            .tool(tool)
            .build(),
    );
    let router = ResponsesHostServer::agent(agent).into_router();

    for approve in [true, false] {
        let before = runs.load(std::sync::atomic::Ordering::SeqCst);
        let (status, first) = json(&router, post("/responses", json!({"input": "do it"}))).await;
        assert_eq!(status, StatusCode::OK, "{first}");
        let request = first["output"]
            .as_array()
            .unwrap()
            .iter()
            .find(|i| i["type"] == "mcp_approval_request")
            .unwrap_or_else(|| panic!("an approval request: {first}"))
            .clone();
        assert_eq!(request["name"], "danger");
        assert_eq!(
            runs.load(std::sync::atomic::Ordering::SeqCst),
            before,
            "not run yet"
        );

        let (status, second) = json(
            &router,
            post(
                "/responses",
                json!({
                    "previous_response_id": first["id"],
                    "input": [{
                        "type": "mcp_approval_response",
                        "approval_request_id": request["id"],
                        "approve": approve,
                    }],
                }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{second}");
        let ran = runs.load(std::sync::atomic::Ordering::SeqCst) - before;
        assert_eq!(ran, usize::from(approve), "approve={approve}: {second}");
        assert!(second["output_text"]
            .as_str()
            .unwrap()
            .starts_with("done: "));
    }

    let (status, body) = json(
        &router,
        post(
            "/responses",
            json!({"input": [{"type": "mcp_approval_response", "approval_request_id": "mcpr_nope", "approve": true}]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
}

#[tokio::test]
async fn a_configured_approval_store_holds_the_requests() {
    use agent_framework_foundry_hosting::{
        FoundryHostConfig, FoundryRequestContext, FunctionApprovalStore,
        FunctionApprovalStoreProvider, InMemoryFunctionApprovalStore,
    };
    struct Provider(Arc<InMemoryFunctionApprovalStore>);
    impl FunctionApprovalStoreProvider for Provider {
        fn approval_store(
            &self,
            _: &FoundryHostConfig,
            _: &FoundryRequestContext,
        ) -> agent_framework_core::error::Result<Arc<dyn FunctionApprovalStore>> {
            Ok(self.0.clone())
        }
    }
    let shared = Arc::new(InMemoryFunctionApprovalStore::default());
    let tool = agent_framework_core::tools::FunctionTool::new(
        "danger",
        "",
        json!({"type": "object"}),
        |_: serde_json::Value| async { Ok(json!("ran")) },
    )
    .into_definition()
    .require_approval();
    let agent: Arc<dyn SupportsAgentRun> =
        Arc::new(Agent::builder(WantsDanger).name("c").tool(tool).build());
    let router = ResponsesHostServer::agent(agent)
        .with_function_approval_store(Arc::new(Provider(shared.clone())))
        .into_router();
    let (_, first) = json(&router, post("/responses", json!({"input": "do it"}))).await;
    let id = first["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "mcp_approval_request")
        .unwrap()["id"]
        .as_str()
        .unwrap()
        .to_string();
    let saved = shared.load_approval_request(&id).await.unwrap();
    assert!(matches!(saved, Content::FunctionApprovalRequest(_)));
}

#[tokio::test]
async fn the_agent_history_source_keeps_the_agents_history() {
    use agent_framework_foundry_hosting::HistorySource;
    let echo = Echo::default();
    let router = ResponsesHostServer::agent(agent(echo.clone()))
        .with_history_source(HistorySource::Agent)
        .into_router();
    let (status, first) = json(&router, post("/responses", json!({"input": "one"}))).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, second) = json(
        &router,
        post(
            "/responses",
            json!({"input": "two", "previous_response_id": first["id"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let seen = echo.seen.lock().unwrap();
    assert_eq!(
        seen.last().unwrap().len(),
        3,
        "prior user + assistant + new input"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelling_a_finished_background_turn_keeps_it_completed() {
    let router = ResponsesHostServer::agent(agent(Echo::default())).into_router();
    for _ in 0..50 {
        let (_, queued) = json(
            &router,
            post("/responses", json!({"input": "bg", "background": true})),
        )
        .await;
        let id = queued["id"].as_str().unwrap().to_string();
        let mut done = false;
        for _ in 0..200 {
            let (_, polled) = json(&router, get(&format!("/responses/{id}"))).await;
            if polled["status"] == "completed" {
                done = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        assert!(done);
        let (status, _) = json(&router, post(&format!("/responses/{id}/cancel"), json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (_, after) = json(&router, get(&format!("/responses/{id}"))).await;
        assert_eq!(
            after["status"], "completed",
            "a finished turn stays completed"
        );
    }
}

#[tokio::test]
async fn sibling_continuations_do_not_share_history() {
    use agent_framework_foundry_hosting::HistorySource;
    let echo = Echo::default();
    let router = ResponsesHostServer::agent(agent(echo.clone()))
        .with_history_source(HistorySource::Agent)
        .into_router();
    let (_, a) = json(&router, post("/responses", json!({"input": "a"}))).await;
    let (status, _) = json(
        &router,
        post(
            "/responses",
            json!({"input": "b", "previous_response_id": a["id"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = json(
        &router,
        post(
            "/responses",
            json!({"input": "c", "previous_response_id": a["id"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let seen = echo.seen.lock().unwrap();
    let c: Vec<String> = seen.last().unwrap().iter().map(|m| m.text()).collect();
    assert_eq!(c.len(), 3, "a, its reply, and c: {c:?}");
    assert!(!c.iter().any(|t| t == "b"), "C must not see B: {c:?}");
}

/// Counts its turns in session state and replies with the count.
struct Counter;

#[async_trait::async_trait]
impl SupportsAgentRun for Counter {
    async fn run(
        &self,
        _messages: Vec<agent_framework_core::prelude::Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
    ) -> agent_framework_core::error::Result<agent_framework_core::types::AgentResponse> {
        let session = session.expect("a session");
        let n = session
            .state
            .get("turns")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;
        session.state.insert("turns", json!(n));
        Ok(agent_framework_core::types::AgentResponse {
            messages: vec![agent_framework_core::prelude::Message::assistant(format!(
                "turn {n}"
            ))],
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        "counter"
    }
}

#[tokio::test]
async fn agent_server_continuations_keep_session_state() {
    let router = ResponsesHostServer::agent(Arc::new(Counter)).into_router();
    let text = |r: &serde_json::Value| {
        r["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    };
    let (_, first) = json(&router, post("/responses", json!({"input": "a"}))).await;
    assert_eq!(text(&first), "turn 1");
    let (_, second) = json(
        &router,
        post(
            "/responses",
            json!({"input": "b", "previous_response_id": first["id"]}),
        ),
    )
    .await;
    assert_eq!(text(&second), "turn 2", "{second}");
    // Conversations carry it too.
    let (_, c1) = json(
        &router,
        post(
            "/responses",
            json!({"input": "a", "conversation": "conv_s"}),
        ),
    )
    .await;
    assert_eq!(text(&c1), "turn 1");
    let (_, c2) = json(
        &router,
        post(
            "/responses",
            json!({"input": "b", "conversation": "conv_s"}),
        ),
    )
    .await;
    assert_eq!(text(&c2), "turn 2", "{c2}");
}

#[cfg(feature = "session-store")]
#[tokio::test]
async fn factory_agents_with_fresh_ids_resume_their_sessions() {
    use agent_framework_foundry_hosting::HistorySource;
    let echo = Echo::default();
    let factory_echo = echo.clone();
    let router = ResponsesHostServer::agent_factory(move || {
        let echo = factory_echo.clone();
        async move {
            // No explicit id: every instance gets a fresh one.
            let agent: Arc<dyn SupportsAgentRun> =
                Arc::new(Agent::builder(echo).name("echo").build());
            Ok(agent)
        }
    })
    .with_history_source(HistorySource::Agent)
    .into_router();
    let (_, first) = json(&router, post("/responses", json!({"input": "one"}))).await;
    let (status, second) = json(
        &router,
        post(
            "/responses",
            json!({"input": "two", "previous_response_id": first["id"]}),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let seen = echo.seen.lock().unwrap();
    assert_eq!(
        seen.last().unwrap().len(),
        3,
        "prior user + assistant + new input"
    );
}

/// Counts turns in session state; never answers the input "hang".
struct HangingCounter;

#[async_trait::async_trait]
impl SupportsAgentRun for HangingCounter {
    async fn run(
        &self,
        messages: Vec<agent_framework_core::prelude::Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
    ) -> agent_framework_core::error::Result<agent_framework_core::types::AgentResponse> {
        if messages.last().map(|m| m.text()) == Some("hang".to_string()) {
            std::future::pending::<()>().await;
        }
        Counter.run(messages, session).await
    }

    fn id(&self) -> &str {
        "hanging-counter"
    }
}

#[tokio::test]
async fn a_cancelled_continuation_keeps_its_inherited_session_state() {
    let router = ResponsesHostServer::agent(Arc::new(HangingCounter)).into_router();
    let text = |r: &serde_json::Value| {
        r["output"][0]["content"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    };
    let (_, first) = json(&router, post("/responses", json!({"input": "a"}))).await;
    assert_eq!(text(&first), "turn 1");
    let (_, queued) = json(
        &router,
        post(
            "/responses",
            json!({"input": "hang", "background": true, "previous_response_id": first["id"]}),
        ),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    let (status, _) = json(&router, post(&format!("/responses/{id}/cancel"), json!({}))).await;
    assert_eq!(status, StatusCode::OK);
    // Continuing from the cancelled record replays the inherited history with
    // the session state that history carried.
    let (_, next) = json(
        &router,
        post(
            "/responses",
            json!({"input": "c", "previous_response_id": id}),
        ),
    )
    .await;
    assert_eq!(text(&next), "turn 2", "{next}");
}

/// Counts turns in session state and replies with the count and how many
/// messages it was given, so a transcript paired with another turn's state
/// shows up as an inconsistent reply.
struct Tally;

#[async_trait::async_trait]
impl SupportsAgentRun for Tally {
    async fn run(
        &self,
        messages: Vec<agent_framework_core::prelude::Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
    ) -> agent_framework_core::error::Result<agent_framework_core::types::AgentResponse> {
        let session = session.expect("a session");
        let n = session
            .state
            .get("turns")
            .and_then(|v| v.as_i64())
            .unwrap_or(0)
            + 1;
        session.state.insert("turns", json!(n));
        Ok(agent_framework_core::types::AgentResponse {
            messages: vec![agent_framework_core::prelude::Message::assistant(format!(
                "turn {n}, {} messages",
                messages.len()
            ))],
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        "tally"
    }
}

/// An in-memory store whose conversation state write can fail once or be
/// slow, and that uses the trait's default (non-atomic) conversation update.
#[derive(Default)]
struct Flaky {
    inner: agent_framework_foundry_hosting::InMemoryResponseStore,
    fail_state_once: std::sync::atomic::AtomicBool,
    slow_state: std::sync::atomic::AtomicBool,
}

#[async_trait::async_trait]
impl agent_framework_foundry_hosting::ResponseStore for Flaky {
    async fn get(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<agent_framework_foundry_hosting::StoredResponse>>
    {
        self.inner.get(id).await
    }
    async fn put(
        &self,
        id: &str,
        response: agent_framework_foundry_hosting::StoredResponse,
    ) -> agent_framework_core::error::Result<()> {
        self.inner.put(id, response).await
    }
    async fn delete(&self, id: &str) -> agent_framework_core::error::Result<bool> {
        self.inner.delete(id).await
    }
    async fn conversation(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<Vec<agent_framework_core::prelude::Message>>>
    {
        self.inner.conversation(id).await
    }
    async fn set_conversation(
        &self,
        id: &str,
        transcript: Vec<agent_framework_core::prelude::Message>,
    ) -> agent_framework_core::error::Result<()> {
        self.inner.set_conversation(id, transcript).await
    }
    async fn conversation_state(
        &self,
        id: &str,
    ) -> agent_framework_core::error::Result<Option<serde_json::Value>> {
        self.inner.conversation_state(id).await
    }
    async fn set_conversation_state(
        &self,
        id: &str,
        state: serde_json::Value,
    ) -> agent_framework_core::error::Result<()> {
        use std::sync::atomic::Ordering;
        if self.slow_state.load(Ordering::SeqCst) {
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
        }
        if self.fail_state_once.swap(false, Ordering::SeqCst) {
            return Err(agent_framework_core::error::Error::Other(
                "disk full".into(),
            ));
        }
        self.inner.set_conversation_state(id, state).await
    }
}

fn tally_text(r: &serde_json::Value) -> String {
    r["output"][0]["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

#[tokio::test]
async fn a_failed_conversation_write_keeps_transcript_and_state_together() {
    use std::sync::atomic::Ordering;
    let store = Arc::new(Flaky::default());
    let router = ResponsesHostServer::agent(Arc::new(Tally))
        .with_response_store(store.clone())
        .into_router();
    let turn = |input: &'static str| {
        post(
            "/responses",
            json!({"input": input, "conversation": "conv_f"}),
        )
    };
    let (_, first) = json(&router, turn("a")).await;
    assert_eq!(tally_text(&first), "turn 1, 1 messages");
    store.fail_state_once.store(true, Ordering::SeqCst);
    let (status, _) = json(&router, turn("b")).await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let (_, third) = json(&router, turn("c")).await;
    assert_eq!(
        tally_text(&third),
        "turn 2, 3 messages",
        "the failed turn left the previous transcript and state together"
    );
}

#[tokio::test]
async fn a_disconnect_cannot_split_a_conversation_update() {
    use std::sync::atomic::Ordering;
    let store = Arc::new(Flaky::default());
    let router = ResponsesHostServer::agent(Arc::new(Tally))
        .with_response_store(store.clone())
        .into_router();
    let turn = |input: &'static str| {
        post(
            "/responses",
            json!({"input": input, "conversation": "conv_d"}),
        )
    };
    let (_, first) = json(&router, turn("a")).await;
    assert_eq!(tally_text(&first), "turn 1, 1 messages");
    store.slow_state.store(true, Ordering::SeqCst);
    // The client gives up mid-write.
    let _ = tokio::time::timeout(
        std::time::Duration::from_millis(100),
        json(&router, turn("b")),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(600)).await;
    store.slow_state.store(false, Ordering::SeqCst);
    let (_, third) = json(&router, turn("c")).await;
    assert_eq!(
        tally_text(&third),
        "turn 3, 5 messages",
        "the turn's update completed whole"
    );
}

/// [`Tally`], except that an input of `boom` fails the run.
struct Brittle;

#[async_trait::async_trait]
impl SupportsAgentRun for Brittle {
    async fn run(
        &self,
        messages: Vec<agent_framework_core::prelude::Message>,
        session: Option<&mut agent_framework_core::session::AgentSession>,
    ) -> agent_framework_core::error::Result<agent_framework_core::types::AgentResponse> {
        if messages.last().map(|m| m.text()).as_deref() == Some("boom") {
            return Err(agent_framework_core::error::Error::Other("boom".into()));
        }
        Tally.run(messages, session).await
    }

    fn id(&self) -> &str {
        "brittle"
    }
}

#[tokio::test]
async fn a_failed_continuation_keeps_the_inherited_session_state() {
    let router = ResponsesHostServer::agent(Arc::new(Brittle)).into_router();
    let (_, first) = json(&router, post("/responses", json!({"input": "a"}))).await;
    assert_eq!(tally_text(&first), "turn 1, 1 messages");
    // In the background, so the failed response's id is known.
    let (_, queued) = json(
        &router,
        post(
            "/responses",
            json!({"input": "boom", "previous_response_id": first["id"], "background": true}),
        ),
    )
    .await;
    let id = queued["id"].as_str().unwrap().to_string();
    let mut failed = json!(null);
    for _ in 0..200 {
        let (_, polled) = json(&router, get(&format!("/responses/{id}"))).await;
        if polled["status"] == "failed" {
            failed = polled;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert_eq!(failed["status"], "failed", "{failed}");
    let (_, third) = json(
        &router,
        post(
            "/responses",
            json!({"input": "c", "previous_response_id": failed["id"]}),
        ),
    )
    .await;
    assert_eq!(
        tally_text(&third),
        "turn 2, 4 messages",
        "the failed record kept turn 1's state with its inherited transcript"
    );
}
