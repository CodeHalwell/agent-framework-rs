//! AG-UI protocol hosting tests: the ordered SSE event sequence for a text run,
//! tool-call event framing (including the frontend-tool pattern), `RUN_ERROR`
//! on agent failure, and a malformed-input `400`.

mod common;

use std::sync::Arc;

use async_trait::async_trait;
use axum::http::StatusCode;
use serde_json::{json, Value};

use agent_framework_core::agent::Agent;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::threads::AgentThread;
use agent_framework_core::types::{
    AgentRunResponse, ChatMessage, Content, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Role,
};
use agent_framework_hosting::agui::AgUiRouter;

use common::{parse_sse, parse_sse_json, post_raw, MockAgent};

/// The `type` string of each SSE event, in order.
fn event_types(events: &[Value]) -> Vec<String> {
    events
        .iter()
        .map(|e| e["type"].as_str().unwrap().to_string())
        .collect()
}

// ---------------------------------------------------------------------------
// Happy path: a text run
// ---------------------------------------------------------------------------

#[tokio::test]
async fn text_run_emits_exact_event_sequence() {
    let app = AgUiRouter::for_agent("assistant", MockAgent::new("a1").arc()).into_router();

    let body = json!({
        "threadId": "t-1",
        "runId": "r-1",
        "messages": [{ "id": "m1", "role": "user", "content": "hello" }],
    });
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    // AG-UI has no `[DONE]` sentinel — the run boundary is RUN_FINISHED.
    let raw = parse_sse(&text);
    assert!(
        !raw.iter().any(|d| d == "[DONE]"),
        "no [DONE] sentinel in AG-UI"
    );

    let events = parse_sse_json(&text);
    assert_eq!(
        event_types(&events),
        vec![
            "RUN_STARTED",
            "TEXT_MESSAGE_START",
            "TEXT_MESSAGE_CONTENT",
            "TEXT_MESSAGE_END",
            "RUN_FINISHED",
        ],
    );

    // RUN_STARTED / RUN_FINISHED echo the client's thread & run ids.
    let started = &events[0];
    assert_eq!(started["threadId"], "t-1");
    assert_eq!(started["runId"], "r-1");
    let finished = events.last().unwrap();
    assert_eq!(finished["threadId"], "t-1");
    assert_eq!(finished["runId"], "r-1");

    // TEXT_MESSAGE_START carries a role; START/CONTENT/END share one messageId.
    let start = &events[1];
    assert_eq!(start["role"], "assistant");
    let mid = start["messageId"].as_str().unwrap();
    assert_eq!(events[2]["messageId"], mid);
    assert_eq!(events[3]["messageId"], mid);

    // The content delta is the agent's reply.
    assert_eq!(events[2]["delta"], "echo: hello");
}

#[tokio::test]
async fn missing_ids_are_generated_and_echoed() {
    let app = AgUiRouter::for_agent("assistant", MockAgent::new("a1").arc()).into_router();

    // No threadId / runId supplied.
    let body = json!({ "messages": [{ "role": "user", "content": "hi" }] });
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let started = &events[0];
    // Both are non-empty and consistent across the run.
    let thread_id = started["threadId"].as_str().unwrap();
    let run_id = started["runId"].as_str().unwrap();
    assert!(!thread_id.is_empty() && !run_id.is_empty());
    let finished = events.last().unwrap();
    assert_eq!(finished["threadId"], thread_id);
    assert_eq!(finished["runId"], run_id);
}

// ---------------------------------------------------------------------------
// Tool calls
// ---------------------------------------------------------------------------

/// Emits a single (unexecuted) function call — the "frontend tool" shape.
struct FrontendToolAgent;

#[async_trait]
impl Agent for FrontendToolAgent {
    async fn run(
        &self,
        _messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Raw(r#"{"city":"Paris"}"#.to_string())),
        );
        Ok(AgentRunResponse {
            messages: vec![ChatMessage::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(call)],
            )],
            ..Default::default()
        })
    }
    fn id(&self) -> &str {
        "frontend-tool-agent"
    }
}

#[tokio::test]
async fn frontend_tool_call_framing_without_result() {
    let app =
        AgUiRouter::for_agent("tools", Arc::new(FrontendToolAgent) as Arc<dyn Agent>).into_router();

    let body = json!({
        "threadId": "t", "runId": "r",
        "messages": [{ "role": "user", "content": "weather?" }],
        // A client-declared (frontend) tool; accepted but not injected.
        "tools": [{ "name": "get_weather", "description": "Get weather", "parameters": {} }],
    });
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    assert_eq!(
        event_types(&events),
        vec![
            "RUN_STARTED",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END", // synthesized at finalize (no result — frontend executes it)
            "RUN_FINISHED",
        ],
    );

    let start = &events[1];
    assert_eq!(start["toolCallId"], "call_1");
    assert_eq!(start["toolCallName"], "get_weather");
    // No text message is open, so parentMessageId is omitted.
    assert!(start.get("parentMessageId").is_none());

    assert_eq!(events[2]["toolCallId"], "call_1");
    assert_eq!(events[2]["delta"], r#"{"city":"Paris"}"#);
    assert_eq!(events[3]["toolCallId"], "call_1");

    // A frontend tool call is surfaced WITHOUT a TOOL_CALL_RESULT.
    assert!(!event_types(&events).contains(&"TOOL_CALL_RESULT".to_string()));
}

/// Emits a function call followed by its (server-executed) result.
struct ExecutedToolAgent;

#[async_trait]
impl Agent for ExecutedToolAgent {
    async fn run(
        &self,
        _messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        let call = FunctionCallContent::new(
            "call_9",
            "lookup",
            Some(FunctionArguments::Raw(r#"{"q":"x"}"#.to_string())),
        );
        let result = FunctionResultContent::new("call_9", Some(json!({ "answer": 42 })));
        Ok(AgentRunResponse {
            messages: vec![ChatMessage::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(call), Content::FunctionResult(result)],
            )],
            ..Default::default()
        })
    }
    fn id(&self) -> &str {
        "executed-tool-agent"
    }
}

#[tokio::test]
async fn executed_tool_call_emits_end_then_result() {
    let app =
        AgUiRouter::for_agent("tools", Arc::new(ExecutedToolAgent) as Arc<dyn Agent>).into_router();

    let body = json!({ "threadId": "t", "runId": "r", "messages": [] });
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    assert_eq!(
        event_types(&events),
        vec![
            "RUN_STARTED",
            "TOOL_CALL_START",
            "TOOL_CALL_ARGS",
            "TOOL_CALL_END",
            "TOOL_CALL_RESULT",
            "RUN_FINISHED",
        ],
    );

    // The result event mirrors the bridge: messageId, toolCallId, content, role.
    let result = &events[4];
    assert_eq!(result["toolCallId"], "call_9");
    assert_eq!(result["role"], "tool");
    assert!(result["messageId"].is_string());
    // dict results are JSON-serialized into `content`.
    let content: Value = serde_json::from_str(result["content"].as_str().unwrap()).unwrap();
    assert_eq!(content, json!({ "answer": 42 }));
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

/// Always fails.
struct FailingAgent;

#[async_trait]
impl Agent for FailingAgent {
    async fn run(
        &self,
        _messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        Err(Error::AgentExecution("kaboom".to_string()))
    }
    fn id(&self) -> &str {
        "failing-agent"
    }
}

#[tokio::test]
async fn agent_failure_emits_run_error() {
    let app = AgUiRouter::for_agent("boom", Arc::new(FailingAgent) as Arc<dyn Agent>).into_router();

    let body = json!({ "threadId": "t", "runId": "r", "messages": [] });
    // The stream itself is a normal 200 text/event-stream; the failure is in-band.
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    assert_eq!(event_types(&events), vec!["RUN_STARTED", "RUN_ERROR"]);
    assert_eq!(events[0]["threadId"], "t");
    assert!(events[1]["message"].as_str().unwrap().contains("kaboom"));
    // No RUN_FINISHED after a RUN_ERROR.
    assert!(!event_types(&events).contains(&"RUN_FINISHED".to_string()));
}

#[tokio::test]
async fn malformed_input_is_400() {
    let app = AgUiRouter::for_agent("assistant", MockAgent::new("a1").arc()).into_router();

    let (status, body) = post_raw(app, "/", "{ not valid json".to_string()).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body.contains("Invalid RunAgentInput"));
}

// ---------------------------------------------------------------------------
// Input mapping & multi-agent routing
// ---------------------------------------------------------------------------

/// Reflects the *structure* of the messages it receives back as text, so a test
/// can assert exactly how AG-UI input mapped to core content.
struct InspectAgent;

#[async_trait]
impl Agent for InspectAgent {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        let mut parts: Vec<String> = Vec::new();
        for m in &messages {
            for c in &m.contents {
                match c {
                    Content::Text(t) => parts.push(format!("text[{}]:{}", m.role.as_str(), t.text)),
                    Content::FunctionCall(fc) => {
                        parts.push(format!("call:{}:{}", fc.call_id, fc.name))
                    }
                    Content::FunctionResult(fr) => parts.push(format!(
                        "result:{}:{}",
                        fr.call_id,
                        fr.result.as_ref().map(Value::to_string).unwrap_or_default()
                    )),
                    _ => {}
                }
            }
        }
        Ok(AgentRunResponse {
            messages: vec![ChatMessage::assistant(parts.join("|"))],
            ..Default::default()
        })
    }
    fn id(&self) -> &str {
        "inspect-agent"
    }
}

#[tokio::test]
async fn assistant_tool_call_and_tool_result_messages_are_mapped() {
    // A prior assistant tool call + a tool result in the history should map to
    // FunctionCall / FunctionResult content and reach the agent intact.
    let app =
        AgUiRouter::for_agent("inspect", Arc::new(InspectAgent) as Arc<dyn Agent>).into_router();

    let body = json!({
        "threadId": "t", "runId": "r",
        "messages": [
            { "role": "user", "content": "search" },
            {
                "role": "assistant",
                "toolCalls": [{
                    "id": "c1", "type": "function",
                    "function": { "name": "search", "arguments": "{}" }
                }]
            },
            { "role": "tool", "toolCallId": "c1", "content": "found-it" },
        ],
    });
    let (status, text) = post_raw(app, "/", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);

    let events = parse_sse_json(&text);
    let delta = events
        .iter()
        .find(|e| e["type"] == "TEXT_MESSAGE_CONTENT")
        .expect("a content event");
    let reflected = delta["delta"].as_str().unwrap();
    // User text mapped under the user role.
    assert!(reflected.contains("text[user]:search"), "got: {reflected}");
    // Assistant toolCalls mapped to a FunctionCallContent.
    assert!(reflected.contains("call:c1:search"), "got: {reflected}");
    // Tool message mapped to a FunctionResultContent carrying the result.
    assert!(
        reflected.contains("result:c1:") && reflected.contains("found-it"),
        "got: {reflected}"
    );
}

#[tokio::test]
async fn builder_serves_multiple_agents_at_distinct_paths() {
    let app = AgUiRouter::for_agent("primary", MockAgent::new("p").prefix("P: ").arc())
        .path("/primary")
        .agent(
            "secondary",
            MockAgent::new("s").prefix("S: ").arc(),
            "/secondary",
        )
        .into_router();

    let body =
        json!({ "threadId": "t", "runId": "r", "messages": [{ "role": "user", "content": "hi" }] });

    let (status, text) = post_raw(app.clone(), "/primary", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_json(&text);
    let delta = events
        .iter()
        .find(|e| e["type"] == "TEXT_MESSAGE_CONTENT")
        .unwrap();
    assert_eq!(delta["delta"], "P: hi");

    let (status, text) = post_raw(app, "/secondary", body.to_string()).await;
    assert_eq!(status, StatusCode::OK);
    let events = parse_sse_json(&text);
    let delta = events
        .iter()
        .find(|e| e["type"] == "TEXT_MESSAGE_CONTENT")
        .unwrap();
    assert_eq!(delta["delta"], "S: hi");
}
