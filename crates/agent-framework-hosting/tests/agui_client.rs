//! `AgUiChatClient` loopback tests (feature `agui-client`).
//!
//! Two kinds of server, each on a real loopback socket:
//! - this crate's own [`AgUiRouter`], so the client is exercised end to end
//!   against the server it ships beside (text, streaming, client-side tools,
//!   server-side tools, in-band run errors);
//! - a scripted server that records each request and answers with a fixed
//!   SSE body, for the wire-level details (request shape, state, interrupts,
//!   HTTP errors, malformed frames).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::Router;
use futures::StreamExt;
use serde_json::{json, Map, Value};

use agent_framework_core::agent::Agent;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::tools::{FunctionTool, ToolDefinition};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, FunctionArguments,
    FunctionCallContent, Message, Role,
};
use agent_framework_hosting::agui::{state_carrier, AgUiChatClient, AgUiRouter};

use common::StreamingAgent;

/// Serve `router` on an ephemeral loopback port; returns its base URL.
async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{addr}")
}

/// A model that answers the last tool result if there is one, otherwise
/// calls `call_tool` when it is offered, otherwise just talks.
#[derive(Clone)]
struct ScriptedModel {
    call_tool: &'static str,
    requests: Arc<AtomicUsize>,
    offered_tools: Arc<Mutex<Vec<Vec<String>>>>,
}

impl ScriptedModel {
    fn new(call_tool: &'static str) -> Self {
        Self {
            call_tool,
            requests: Arc::new(AtomicUsize::new(0)),
            offered_tools: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

#[async_trait]
impl ChatClient for ScriptedModel {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        self.offered_tools
            .lock()
            .unwrap()
            .push(options.tools.iter().map(|t| t.name.clone()).collect());
        let last_result = messages
            .last()
            .and_then(|m| m.function_results().first().map(|r| (*r).clone()));
        if let Some(result) = last_result {
            let value = match result.result {
                Some(Value::String(s)) => s,
                Some(other) => other.to_string(),
                None => String::new(),
            };
            return Ok(ChatResponse::from_text(format!("The answer is {value}.")));
        }
        if options.tools.iter().any(|t| t.name == self.call_tool) {
            let call = FunctionCallContent::new(
                "call_1",
                self.call_tool,
                Some(FunctionArguments::Raw(r#"{"city":"Paris"}"#.into())),
            );
            return Ok(ChatResponse {
                messages: vec![Message::with_contents(
                    Role::assistant(),
                    vec![Content::FunctionCall(call)],
                )],
                ..Default::default()
            });
        }
        Ok(ChatResponse::from_text("No tools needed."))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;
        let updates: Vec<Result<ChatResponseUpdate>> = response
            .messages
            .into_iter()
            .map(|m| {
                Ok(ChatResponseUpdate {
                    contents: m.contents,
                    role: Some(m.role),
                    ..Default::default()
                })
            })
            .collect();
        Ok(futures::stream::iter(updates).boxed())
    }
}

/// A counting `city -> weather` tool.
fn weather_tool(name: &str, answer: &'static str, calls: Arc<AtomicUsize>) -> ToolDefinition {
    FunctionTool::new(
        name,
        "Get the weather for a city",
        json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        move |_args| {
            let calls = calls.clone();
            async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!(answer))
            }
        },
    )
    .into_definition()
}

// ---------------------------------------------------------------------------
// Against this crate's AgUiRouter
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_response_round_trips_text_through_the_router() {
    let server = AgUiRouter::for_agent(
        "assistant",
        StreamingAgent::new("a", vec!["Hello", " world"]).arc(),
    )
    .into_router();
    let url = serve(server).await;
    let client = AgUiChatClient::new(format!("{url}/"));

    let options = ChatOptions {
        metadata: Some(
            [("thread_id".to_string(), "thread-42".to_string())]
                .into_iter()
                .collect(),
        ),
        ..Default::default()
    };
    let response = client
        .get_response(vec![Message::user("hi")], options)
        .await
        .unwrap();

    assert_eq!(response.text(), "Hello world");
    assert_eq!(response.messages.len(), 1, "{:?}", response.messages);
    assert_eq!(response.messages[0].role, Role::assistant());
    assert_eq!(response.additional_properties["thread_id"], "thread-42");
    assert!(response.additional_properties["run_id"]
        .as_str()
        .unwrap()
        .starts_with("run_"));
    assert_eq!(
        response.finish_reason,
        Some(FinishReason::new(FinishReason::STOP))
    );
}

#[tokio::test]
async fn streaming_yields_each_delta_as_it_arrives() {
    let server = AgUiRouter::for_agent(
        "assistant",
        StreamingAgent::new("a", vec!["one ", "two ", "three"]).arc(),
    )
    .path("/agent")
    .into_router();
    let url = serve(server).await;
    let client = AgUiChatClient::new(format!("{url}/agent"));

    let mut stream = client
        .get_streaming_response(vec![Message::user("count")], ChatOptions::default())
        .await
        .unwrap();
    let mut updates = Vec::new();
    while let Some(update) = stream.next().await {
        updates.push(update.unwrap());
    }

    let deltas: Vec<String> = updates
        .iter()
        .flat_map(|u| u.contents.iter())
        .filter_map(|c| match c {
            Content::Text(t) => Some(t.text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, ["one ", "two ", "three"]);
    // RUN_STARTED's ids ride on the first real update rather than opening an
    // empty message of their own.
    assert!(updates[0].additional_properties.contains_key("thread_id"));
    assert!(updates.last().unwrap().finish_reason.is_some());
    let response = ChatResponse::from_updates(updates);
    assert_eq!(response.messages.len(), 1);
    assert_eq!(response.text(), "one two three");
}

#[tokio::test]
async fn client_tools_are_declared_to_the_server_and_run_locally() {
    // Server: a model that calls `get_weather` whenever it is offered. The
    // router injects the client's declared tools as declaration-only, so the
    // call comes back to the client unexecuted.
    let model = ScriptedModel::new("get_weather");
    let server_agent = Agent::builder(model.clone()).name("server").build();
    let url = serve(AgUiRouter::for_agent("server", server_agent).into_router()).await;

    // Client: an agent whose only model is the remote AG-UI server and which
    // owns the real tool.
    let local_calls = Arc::new(AtomicUsize::new(0));
    let client_agent = Agent::builder(AgUiChatClient::new(url))
        .name("client")
        .tool(weather_tool("get_weather", "sunny", local_calls.clone()))
        .build();

    let response = client_agent.run_once("Weather in Paris?").await.unwrap();

    assert_eq!(
        local_calls.load(Ordering::SeqCst),
        1,
        "the tool runs locally"
    );
    assert!(
        response.text().ends_with("The answer is sunny."),
        "{:?}",
        response.messages
    );
    // Two runs on the server: the call, then the answer to its result.
    assert_eq!(model.requests.load(Ordering::SeqCst), 2);
    let offered = model.offered_tools.lock().unwrap().clone();
    assert!(offered.iter().all(|tools| tools == &["get_weather"]));
}

#[tokio::test]
async fn server_tools_with_results_are_not_run_locally() {
    // Server: owns and runs `server_weather` itself, so the stream carries
    // the call *and* its TOOL_CALL_RESULT.
    let server_calls = Arc::new(AtomicUsize::new(0));
    let model = ScriptedModel::new("server_weather");
    let server_agent = Agent::builder(model.clone())
        .name("server")
        .tool(weather_tool(
            "server_weather",
            "rainy",
            server_calls.clone(),
        ))
        .build();
    let url = serve(AgUiRouter::for_agent("server", server_agent).into_router()).await;

    // Client: has a local tool of its own, so its function loop is active.
    let local_calls = Arc::new(AtomicUsize::new(0));
    let client_agent = Agent::builder(AgUiChatClient::new(url))
        .name("client")
        .tool(weather_tool("local_only", "n/a", local_calls.clone()))
        .build();

    let response = client_agent.run_once("Weather?").await.unwrap();

    assert_eq!(server_calls.load(Ordering::SeqCst), 1);
    assert_eq!(local_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        model.requests.load(Ordering::SeqCst),
        2,
        "one client request"
    );
    let calls: Vec<_> = response
        .messages
        .iter()
        .flat_map(|m| m.function_calls())
        .map(|c| c.name.clone())
        .collect();
    assert_eq!(calls, ["server_weather"]);
    let results: Vec<_> = response
        .messages
        .iter()
        .flat_map(|m| m.function_results())
        .map(|r| r.result.clone())
        .collect();
    assert_eq!(results, [Some(json!("rainy"))]);
    assert!(response.text().contains("The answer is rainy."));
}

/// An agent whose run fails, so the router answers with RUN_ERROR.
struct FailingModel;

#[async_trait]
impl ChatClient for FailingModel {
    async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        Err(Error::service("model unavailable"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::service("model unavailable"))
    }
}

#[tokio::test]
async fn run_error_is_returned_in_band() {
    let agent = Agent::builder(FailingModel).name("broken").build();
    let url = serve(AgUiRouter::for_agent("broken", agent).into_router()).await;
    let client = AgUiChatClient::new(url);

    let response = client
        .get_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .unwrap();
    let error = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .find_map(|c| match c {
            Content::Error(e) => Some(e.clone()),
            _ => None,
        })
        .expect("an error content");
    assert_eq!(error.error_code.as_deref(), Some("RUN_ERROR"));
    assert!(error.message.unwrap().contains("model unavailable"));
}

// ---------------------------------------------------------------------------
// Against a scripted server
// ---------------------------------------------------------------------------

#[derive(Clone, Default)]
struct Recorded {
    bodies: Arc<Mutex<Vec<Value>>>,
    accept: Arc<Mutex<Vec<String>>>,
}

/// A server that records each request and answers with `body` (or `status`).
async fn scripted(status: StatusCode, body: &'static str) -> (String, Recorded) {
    let recorded = Recorded::default();
    let rec = recorded.clone();
    let router = Router::new().route(
        "/",
        post(move |headers: HeaderMap, raw: String| {
            let rec = rec.clone();
            async move {
                rec.bodies
                    .lock()
                    .unwrap()
                    .push(serde_json::from_str(&raw).unwrap());
                rec.accept.lock().unwrap().push(
                    headers
                        .get("accept")
                        .map(|v| v.to_str().unwrap().to_string())
                        .unwrap_or_default(),
                );
                (
                    status,
                    [("content-type", "text/event-stream")],
                    body.to_string(),
                )
                    .into_response()
            }
        }),
    );
    (serve(router).await, recorded)
}

const OK_RUN: &str = concat!(
    "data: {\"type\":\"RUN_STARTED\",\"threadId\":\"thread_1\",\"runId\":\"run_1\"}\n\n",
    "data: {\"type\":\"TEXT_MESSAGE_CONTENT\",\"messageId\":\"msg_1\",\"delta\":\"ok\"}\n\n",
    "data: {\"type\":\"RUN_FINISHED\",\"threadId\":\"thread_1\",\"runId\":\"run_1\"}\n\n",
);

#[tokio::test]
async fn request_body_uses_the_protocol_shape() {
    let (url, recorded) = scripted(StatusCode::OK, OK_RUN).await;
    let client = AgUiChatClient::new(url);

    let mut state = Map::new();
    state.insert("user_id".into(), json!("123"));
    let messages = vec![Message::user("Hello"), state_carrier(state)];
    let mut options = ChatOptions::default()
        .with_instructions("Be brief.")
        .with_tool(weather_tool(
            "get_weather",
            "x",
            Arc::new(AtomicUsize::new(0)),
        ));
    options.additional_properties.insert(
        "available_interrupts".into(),
        json!([{"id": "req_1", "type": "request_info"}]),
    );
    options.additional_properties.insert(
        "resume".into(),
        json!({"interrupts": [{"id": "req_1", "value": "approved"}]}),
    );
    options
        .additional_properties
        .insert("forwarded_props".into(), json!({"tenant": "t"}));

    let response = client.get_response(messages, options).await.unwrap();
    assert_eq!(response.text(), "ok");

    let body = recorded.bodies.lock().unwrap()[0].clone();
    assert_eq!(
        recorded.accept.lock().unwrap()[0],
        "text/event-stream",
        "asks for SSE"
    );
    assert!(body["threadId"].as_str().unwrap().starts_with("thread_"));
    assert!(body["runId"].as_str().unwrap().starts_with("run_"));
    // Instructions lead as a system message; the state carrier is not sent.
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "Be brief.");
    assert_eq!(messages[1]["role"], "user");
    assert_eq!(messages[1]["content"], "Hello");
    assert_eq!(body["state"], json!({"user_id": "123"}));
    assert_eq!(
        body["tools"],
        json!([{
            "name": "get_weather",
            "description": "Get the weather for a city",
            "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
        }])
    );
    assert_eq!(body["context"], json!([]));
    assert_eq!(body["forwardedProps"], json!({"tenant": "t"}));
    assert_eq!(
        body["availableInterrupts"],
        json!([{"id": "req_1", "reason": "input_required"}])
    );
    assert_eq!(
        body["resume"],
        json!([{"interruptId": "req_1", "status": "resolved", "payload": "approved"}])
    );
}

#[tokio::test]
async fn http_error_status_fails_the_request() {
    let (url, _) = scripted(StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").await;
    let client = AgUiChatClient::new(url);
    let err = client
        .get_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .unwrap_err();
    assert_eq!(err.status(), Some(500), "{err}");

    let err = client
        .get_streaming_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .err()
        .expect("streaming fails too");
    assert_eq!(err.status(), Some(500));
}

#[tokio::test]
async fn malformed_frames_are_skipped() {
    let body = concat!(
        "data: {\"type\":\"RUN_STARTED\",\"threadId\":\"t\",\"runId\":\"r\"}\n\n",
        "data: not json\n\n",
        ": a comment\n\n",
        "data: {\"type\":\"TEXT_MESSAGE_CONTENT\",\"messageId\":\"m\",\"delta\":\"still here\"}\n\n",
        "data: {\"type\":\"RUN_FINISHED\",\"threadId\":\"t\",\"runId\":\"r\"}",
    );
    let (url, _) = scripted(StatusCode::OK, body).await;
    let client = AgUiChatClient::new(url);
    let response = client
        .get_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(response.text(), "still here");
    // The final frame had no trailing blank line and still counted.
    assert!(response.finish_reason.is_some());
}

#[tokio::test]
async fn empty_stream_gives_an_empty_response() {
    let (url, _) = scripted(StatusCode::OK, "").await;
    let client = AgUiChatClient::new(url);
    let response = client
        .get_response(vec![Message::user("hi")], ChatOptions::default())
        .await
        .unwrap();
    assert!(response.messages.is_empty());
}

#[tokio::test]
async fn server_tool_call_surfaces_as_a_function_call() {
    // A call to a tool the client did not declare, with no result: surfaced
    // as a FunctionCallContent (upstream unwraps its server_function_call to
    // the same thing).
    let body = concat!(
        "data: {\"type\":\"RUN_STARTED\",\"threadId\":\"thread_1\",\"runId\":\"run_1\"}\n\n",
        "data: {\"type\":\"TOOL_CALL_START\",\"toolCallId\":\"call_1\",\"toolName\":\"get_time_zone\"}\n\n",
        "data: {\"type\":\"TOOL_CALL_ARGS\",\"toolCallId\":\"call_1\",\"delta\":\"{\\\"location\\\": \\\"Seattle\\\"}\"}\n\n",
        "data: {\"type\":\"RUN_FINISHED\",\"threadId\":\"thread_1\",\"runId\":\"run_1\"}\n\n",
    );
    let (url, _) = scripted(StatusCode::OK, body).await;
    let client = AgUiChatClient::new(url);
    let response = client
        .get_response(vec![Message::user("tz?")], ChatOptions::default())
        .await
        .unwrap();
    let calls = response.function_calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_time_zone");
    assert_eq!(calls[0].parse_arguments().unwrap()["location"], "Seattle");
}
