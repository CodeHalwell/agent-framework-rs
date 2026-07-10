//! DevUI-style HTTP API: entity discovery and OpenAI-Responses-flavored
//! execution, mirroring the Python `agent_framework_devui` server.
//!
//! # Routes
//! - `GET /health` — liveness + entity count.
//! - `GET /v1/entities` — list entities (`DiscoveryResponse`).
//! - `GET /v1/entities/{id}/info` — entity details (`EntityInfo`).
//! - `POST /v1/responses` — execute an entity. Routes on `metadata.entity_id`
//!   (DevUI's convention), then `extra_body.entity_id`, then `model`.
//!
//! # SSE event mapping (`stream: true`)
//! Agents: `response.created` → `response.in_progress` →
//! `response.output_item.added` → `response.content_part.added` →
//! `response.output_text.delta`* → `response.completed` → `data: [DONE]`.
//!
//! Workflows map each engine event to a DevUI event name:
//! `ExecutorInvoked` → `response.output_item.added` (an `executor_action`
//! item), `ExecutorCompleted`/`ExecutorFailed` → `response.output_item.done`,
//! `Output` → `response.output_item.added` (a message), `RequestInfo` →
//! `response.request_info.requested`, everything else →
//! `response.workflow_event.completed`; then `response.completed` + `[DONE]`.
//!
//! # Divergences from DevUI
//! - Runs are **stateless**: no conversation store, no checkpoint resume. DevUI
//!   resumes human-in-the-loop workflows via `metadata`/checkpointing on the
//!   same `/v1/responses` route; the reference has no
//!   `/v1/entities/{id}/runs/{run_id}/responses` endpoint, so per the work
//!   package we surface pending requests (in-stream and in the final response)
//!   but do not persist runs or support resume.
//! - Streaming is computed by running to completion and then framing the result
//!   as SSE (the core `Agent` trait exposes only `run`, not `run_stream`); event
//!   ordering and payloads still match DevUI's names.

pub mod models;

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use agent_framework_core::types::{AgentRunResponse, ChatMessage, Role, UsageDetails};
use agent_framework_core::workflow::WorkflowEvent;

use crate::registry::{AgentRecord, EntityRecord, HostState, WorkflowRecord};
use crate::sse::sse_response;
use crate::util;
use models::{
    openai_error, DiscoveryResponse, EntityInfo, HealthResponse, InputTokensDetails, OutputMessage,
    OutputTokensDetails, ResponseObject, ResponsesRequest, Usage,
};

/// Build the DevUI router for a registry.
pub(crate) fn router(state: Arc<HostState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/entities", get(list_entities))
        .route("/v1/entities/{entity_id}/info", get(entity_info))
        .route("/v1/responses", post(create_response))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<Arc<HostState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy",
        entities_count: state.list().len(),
        framework: "agent_framework",
    })
}

async fn list_entities(State(state): State<Arc<HostState>>) -> Json<DiscoveryResponse> {
    let entities = state.list().iter().map(entity_info_for).collect();
    Json(DiscoveryResponse { entities })
}

async fn entity_info(
    State(state): State<Arc<HostState>>,
    Path(entity_id): Path<String>,
) -> Response {
    match state.get(&entity_id) {
        Some(record) => Json(entity_info_for(record)).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(openai_error(
                format!("Entity {entity_id} not found"),
                "invalid_request_error",
                Some("entity_not_found"),
            )),
        )
            .into_response(),
    }
}

async fn create_response(
    State(state): State<Arc<HostState>>,
    Json(request): Json<ResponsesRequest>,
) -> Response {
    let Some(entity_id) = request.entity_id() else {
        return (
            StatusCode::BAD_REQUEST,
            Json(openai_error(
                "Missing entity_id. Provide metadata.entity_id (or extra_body.entity_id, or model).",
                "invalid_request_error",
                Some("missing_entity_id"),
            )),
        )
            .into_response();
    };

    let Some(record) = state.get(&entity_id) else {
        return (
            StatusCode::NOT_FOUND,
            Json(openai_error(
                format!("Entity not found: {entity_id}"),
                "invalid_request_error",
                Some("entity_not_found"),
            )),
        )
            .into_response();
    };

    let model = request.model.clone().unwrap_or_else(|| entity_id.clone());

    match record {
        EntityRecord::Agent(agent) => run_agent(agent, &request, model).await,
        EntityRecord::Workflow(workflow) => run_workflow(workflow, &request, model).await,
    }
}

// ---------------------------------------------------------------------------
// Entity info
// ---------------------------------------------------------------------------

fn entity_info_for(record: &EntityRecord) -> EntityInfo {
    match record {
        EntityRecord::Agent(a) => EntityInfo {
            id: a.id.clone(),
            entity_type: "agent",
            name: a.name.clone(),
            description: a.description.clone(),
            framework: "agent_framework",
            // The core `Agent` trait exposes no tool list, so this is absent.
            tools: None,
            metadata: Map::new(),
            source: "in_memory",
            instructions: a.instructions.clone(),
            // `model_id` is not accessible through the `Agent` trait.
            model_id: None,
            executors: None,
            input_schema: None,
            start_executor_id: None,
        },
        EntityRecord::Workflow(w) => EntityInfo {
            id: w.id.clone(),
            entity_type: "workflow",
            name: w.name.clone(),
            description: w.description.clone(),
            framework: "agent_framework",
            tools: None,
            metadata: Map::new(),
            source: "in_memory",
            instructions: None,
            model_id: None,
            // The full executor set is not enumerable through the public
            // `Workflow` API; the start executor is.
            executors: Some(vec![w.workflow.start_executor_id().to_string()]),
            input_schema: Some(json!({ "type": "string" })),
            start_executor_id: Some(w.workflow.start_executor_id().to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// Agent execution
// ---------------------------------------------------------------------------

async fn run_agent(agent: &AgentRecord, request: &ResponsesRequest, model: String) -> Response {
    let messages = input_to_messages(&request.input);
    let input_len = approx_input_len(&request.input);

    let response = match agent.agent.run(messages, None).await {
        Ok(r) => r,
        Err(e) => return execution_error(e.to_string()),
    };

    if request.stream {
        let events = agent_stream_events(&response, &model, input_len);
        sse_response(events)
    } else {
        Json(agent_response_object(&response, &model, input_len)).into_response()
    }
}

/// Build the aggregated (non-streaming) response for an agent run.
fn agent_response_object(resp: &AgentRunResponse, model: &str, input_len: usize) -> ResponseObject {
    let text = resp.text();
    let mid = util::msg_id();
    let usage = build_usage(&resp.usage_details, input_len, text.len());
    ResponseObject {
        id: util::resp_id(),
        object: "response",
        created_at: util::now_ts(),
        model: model.to_string(),
        status: "completed",
        output: vec![OutputMessage::assistant_text(mid, text.clone())],
        output_text: Some(text),
        usage: Some(usage),
        outputs: None,
        pending_requests: Vec::new(),
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    }
}

/// Build the SSE event sequence for a streamed agent run.
fn agent_stream_events(resp: &AgentRunResponse, model: &str, input_len: usize) -> Vec<Value> {
    let rid = util::resp_id();
    let mid = util::msg_id();
    let mut seq: u64 = 0;
    let mut next = || {
        seq += 1;
        seq
    };
    let in_progress =
        serde_json::to_value(ResponseObject::in_progress(&rid, model)).unwrap_or(Value::Null);

    let mut events = vec![
        json!({ "type": "response.created", "sequence_number": next(), "response": in_progress }),
        json!({ "type": "response.in_progress", "sequence_number": next(), "response": in_progress }),
        json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "sequence_number": next(),
            "item": { "type": "message", "id": mid, "role": "assistant", "content": [], "status": "in_progress" }
        }),
        json!({
            "type": "response.content_part.added",
            "output_index": 0,
            "content_index": 0,
            "item_id": mid,
            "sequence_number": next(),
            "part": { "type": "output_text", "text": "", "annotations": [] }
        }),
    ];

    // Emit one text delta per non-empty message.
    for message in &resp.messages {
        let delta = message.text();
        if delta.is_empty() {
            continue;
        }
        events.push(json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": mid,
            "delta": delta,
            "logprobs": [],
            "sequence_number": next(),
        }));
    }

    let text = resp.text();
    let completed = ResponseObject {
        id: rid.clone(),
        object: "response",
        created_at: util::now_ts(),
        model: model.to_string(),
        status: "completed",
        output: vec![OutputMessage::assistant_text(mid, text.clone())],
        output_text: Some(text),
        usage: Some(build_usage(
            &resp.usage_details,
            input_len,
            resp.text().len(),
        )),
        outputs: None,
        pending_requests: Vec::new(),
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    };
    events.push(json!({
        "type": "response.completed",
        "sequence_number": next(),
        "response": serde_json::to_value(completed).unwrap_or(Value::Null),
    }));
    events
}

// ---------------------------------------------------------------------------
// Workflow execution
// ---------------------------------------------------------------------------

async fn run_workflow(
    workflow: &WorkflowRecord,
    request: &ResponsesRequest,
    model: String,
) -> Response {
    let input = input_to_workflow_value(&request.input);

    let run = match workflow.workflow.run(input).await {
        Ok(r) => r,
        Err(e) => return execution_error(e.to_string()),
    };

    let outputs = run.outputs();
    let pending: Vec<Value> = run
        .pending_requests()
        .into_iter()
        .map(|p| {
            json!({
                "request_id": p.request_id,
                "source_executor_id": p.source_executor_id,
                "request_data": p.request_data,
            })
        })
        .collect();

    if request.stream {
        let events = workflow_stream_events(run.events(), &outputs, &pending, &model);
        sse_response(events)
    } else {
        Json(workflow_response_object(&outputs, pending, &model)).into_response()
    }
}

/// Build the aggregated (non-streaming) response for a workflow run.
fn workflow_response_object(outputs: &[Value], pending: Vec<Value>, model: &str) -> ResponseObject {
    let output: Vec<OutputMessage> = outputs
        .iter()
        .map(|o| OutputMessage::assistant_text(util::msg_id(), value_to_text(o)))
        .collect();
    let text = outputs
        .iter()
        .map(value_to_text)
        .collect::<Vec<_>>()
        .join("\n");
    ResponseObject {
        id: util::resp_id(),
        object: "response",
        created_at: util::now_ts(),
        model: model.to_string(),
        status: "completed",
        output,
        output_text: Some(text),
        usage: None,
        outputs: Some(outputs.to_vec()),
        pending_requests: pending,
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    }
}

/// State threaded through workflow-event mapping (mirrors DevUI's per-request
/// conversion context: sequence numbers, output index, executor item ids).
struct WfCtx {
    seq: u64,
    output_index: i64,
    item_id: String,
    exec_items: HashMap<String, (String, i64)>,
}

impl WfCtx {
    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }
}

/// Build the SSE event sequence for a streamed workflow run.
fn workflow_stream_events(
    wf_events: &[WorkflowEvent],
    outputs: &[Value],
    pending: &[Value],
    model: &str,
) -> Vec<Value> {
    let rid = util::resp_id();
    let mut ctx = WfCtx {
        seq: 0,
        output_index: -1,
        item_id: util::msg_id(),
        exec_items: HashMap::new(),
    };
    let in_progress =
        serde_json::to_value(ResponseObject::in_progress(&rid, model)).unwrap_or(Value::Null);

    let mut events = vec![
        json!({ "type": "response.created", "sequence_number": ctx.next(), "response": in_progress }),
        json!({ "type": "response.in_progress", "sequence_number": ctx.next(), "response": in_progress }),
    ];

    for ev in wf_events {
        map_workflow_event(ev, &mut ctx, &mut events);
    }

    // Completed response aggregates the workflow outputs.
    let output: Vec<OutputMessage> = outputs
        .iter()
        .map(|o| OutputMessage::assistant_text(util::msg_id(), value_to_text(o)))
        .collect();
    let text = outputs
        .iter()
        .map(value_to_text)
        .collect::<Vec<_>>()
        .join("\n");
    let completed = ResponseObject {
        id: rid.clone(),
        object: "response",
        created_at: util::now_ts(),
        model: model.to_string(),
        status: "completed",
        output,
        output_text: Some(text),
        usage: None,
        outputs: Some(outputs.to_vec()),
        pending_requests: pending.to_vec(),
        parallel_tool_calls: false,
        tool_choice: "none",
        tools: Vec::new(),
    };
    events.push(json!({
        "type": "response.completed",
        "sequence_number": ctx.next(),
        "response": serde_json::to_value(completed).unwrap_or(Value::Null),
    }));
    events
}

/// Map a single engine event onto DevUI SSE event(s), pushing onto `out`.
fn map_workflow_event(ev: &WorkflowEvent, ctx: &mut WfCtx, out: &mut Vec<Value>) {
    match ev {
        WorkflowEvent::ExecutorInvoked { executor_id } => {
            ctx.output_index += 1;
            let item_id = format!("exec_{}_{}", executor_id, util::short_hex());
            ctx.exec_items
                .insert(executor_id.clone(), (item_id.clone(), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.added",
                "output_index": ctx.output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "in_progress",
                }
            }));
        }
        WorkflowEvent::ExecutorCompleted { executor_id } => {
            let (item_id, output_index) = ctx
                .exec_items
                .get(executor_id)
                .cloned()
                .unwrap_or_else(|| (format!("exec_{executor_id}"), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "completed",
                }
            }));
        }
        WorkflowEvent::ExecutorFailed { executor_id, error } => {
            let (item_id, output_index) = ctx
                .exec_items
                .get(executor_id)
                .cloned()
                .unwrap_or_else(|| (format!("exec_{executor_id}"), ctx.output_index));
            out.push(json!({
                "type": "response.output_item.done",
                "output_index": output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "executor_action",
                    "id": item_id,
                    "executor_id": executor_id,
                    "status": "failed",
                    "error": { "message": error },
                }
            }));
        }
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => {
            ctx.output_index += 1;
            out.push(json!({
                "type": "response.output_item.added",
                "output_index": ctx.output_index,
                "sequence_number": ctx.next(),
                "item": {
                    "type": "message",
                    "id": util::msg_id(),
                    "role": "assistant",
                    "content": [ { "type": "output_text", "text": value_to_text(data), "annotations": [] } ],
                    "status": "completed",
                    "metadata": { "source_executor_id": source_executor_id },
                }
            }));
        }
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            request_data,
        } => {
            out.push(json!({
                "type": "response.request_info.requested",
                "request_id": request_id,
                "source_executor_id": source_executor_id,
                "request_data": request_data,
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
            }));
        }
        other => {
            // Started, Status, SuperStep*, AgentRun*, Custom, Failed → the
            // catch-all workflow debug event (DevUI's
            // `response.workflow_event.completed`).
            out.push(json!({
                "type": "response.workflow_event.completed",
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
                "data": workflow_event_data(other),
            }));
        }
    }
}

/// The `data` payload for a catch-all `response.workflow_event.completed`.
fn workflow_event_data(ev: &WorkflowEvent) -> Value {
    match ev {
        WorkflowEvent::Started => json!({ "event_type": "WorkflowStartedEvent" }),
        WorkflowEvent::Status(state) => json!({
            "event_type": "WorkflowStatusEvent",
            "state": serde_json::to_value(state).unwrap_or(Value::Null),
        }),
        WorkflowEvent::SuperStepStarted(n) => {
            json!({ "event_type": "SuperStepStartedEvent", "step": n })
        }
        WorkflowEvent::SuperStepCompleted(n) => {
            json!({ "event_type": "SuperStepCompletedEvent", "step": n })
        }
        WorkflowEvent::AgentRunUpdate {
            executor_id,
            update,
        } => json!({
            "event_type": "AgentRunUpdateEvent",
            "executor_id": executor_id,
            "data": update,
        }),
        WorkflowEvent::AgentRun {
            executor_id,
            response,
        } => json!({
            "event_type": "AgentRunEvent",
            "executor_id": executor_id,
            "data": response,
        }),
        WorkflowEvent::Custom(v) => json!({ "event_type": "CustomEvent", "data": v }),
        WorkflowEvent::Failed { error } => {
            json!({ "event_type": "WorkflowFailedEvent", "message": error })
        }
        // Handled by the caller; included for completeness.
        WorkflowEvent::ExecutorInvoked { executor_id } => {
            json!({ "event_type": "ExecutorInvokedEvent", "executor_id": executor_id })
        }
        WorkflowEvent::ExecutorCompleted { executor_id } => {
            json!({ "event_type": "ExecutorCompletedEvent", "executor_id": executor_id })
        }
        WorkflowEvent::ExecutorFailed { executor_id, error } => json!({
            "event_type": "ExecutorFailedEvent", "executor_id": executor_id, "error": error,
        }),
        WorkflowEvent::Output {
            data,
            source_executor_id,
        } => json!({
            "event_type": "WorkflowOutputEvent",
            "source_executor_id": source_executor_id,
            "data": data,
        }),
        WorkflowEvent::RequestInfo {
            request_id,
            source_executor_id,
            request_data,
        } => json!({
            "event_type": "RequestInfoEvent",
            "request_id": request_id,
            "source_executor_id": source_executor_id,
            "request_data": request_data,
        }),
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn execution_error(message: String) -> Response {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(openai_error(
            format!("Request execution failed: {message}"),
            "server_error",
            None,
        )),
    )
        .into_response()
}

fn build_usage(usage: &Option<UsageDetails>, input_len: usize, output_len: usize) -> Usage {
    match usage {
        Some(u) => {
            let input = u.input_token_count.unwrap_or(0);
            let output = u.output_token_count.unwrap_or(0);
            let total = u.total_token_count.unwrap_or(input + output);
            Usage {
                input_tokens: input,
                output_tokens: output,
                total_tokens: total,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
            }
        }
        None => {
            // DevUI's fallback estimate: ~4 characters per token.
            let input = (input_len / 4) as u64;
            let output = (output_len / 4) as u64;
            Usage {
                input_tokens: input,
                output_tokens: output,
                total_tokens: input + output,
                input_tokens_details: InputTokensDetails { cached_tokens: 0 },
                output_tokens_details: OutputTokensDetails {
                    reasoning_tokens: 0,
                },
            }
        }
    }
}

/// Approximate the character length of the request input (for usage estimates).
fn approx_input_len(input: &Value) -> usize {
    match input {
        Value::String(s) => s.len(),
        other => other.to_string().len(),
    }
}

/// Convert an arbitrary JSON output value to display text.
fn value_to_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Parse the OpenAI-style `input` into chat messages for an agent.
///
/// Accepts a bare string, an array of input items (OpenAI `{type:"message",
/// content:[…]}` or `{role, content}`), or falls back to a stringified value.
fn input_to_messages(input: &Value) -> Vec<ChatMessage> {
    match input {
        Value::String(s) => vec![ChatMessage::user(s.clone())],
        Value::Null => vec![ChatMessage::user(String::new())],
        Value::Array(items) => {
            let msgs: Vec<ChatMessage> = items.iter().filter_map(item_to_message).collect();
            if msgs.is_empty() {
                vec![ChatMessage::user(String::new())]
            } else {
                msgs
            }
        }
        obj @ Value::Object(_) => item_to_message(obj)
            .map(|m| vec![m])
            .unwrap_or_else(|| vec![ChatMessage::user(obj.to_string())]),
        other => vec![ChatMessage::user(other.to_string())],
    }
}

/// Convert one input item into a chat message, if it carries text.
fn item_to_message(item: &Value) -> Option<ChatMessage> {
    match item {
        Value::String(s) => Some(ChatMessage::user(s.clone())),
        Value::Object(map) => {
            let role = map
                .get("role")
                .and_then(Value::as_str)
                .map(role_from)
                .unwrap_or_else(Role::user);
            let text = map.get("content").map(content_text).unwrap_or_default();
            Some(ChatMessage::new(role, text))
        }
        _ => None,
    }
}

/// Extract text from an OpenAI `content` value (string or array of parts).
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| {
                p.get("text")
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .or_else(|| p.as_str().map(str::to_string))
            })
            .collect::<Vec<_>>()
            .join(""),
        other => other.to_string(),
    }
}

fn role_from(role: &str) -> Role {
    match role {
        "user" => Role::user(),
        "assistant" => Role::assistant(),
        "system" => Role::system(),
        "tool" => Role::tool(),
        other => Role::new(other),
    }
}

/// Reduce the OpenAI-style `input` to a value fed to `Workflow::run`.
///
/// Strings and structured objects pass through; a message array is flattened to
/// its concatenated user text so a string-typed start executor still works.
fn input_to_workflow_value(input: &Value) -> Value {
    match input {
        Value::String(s) => Value::String(s.clone()),
        Value::Null => Value::String(String::new()),
        Value::Array(_) => {
            let text = input_to_messages(input)
                .iter()
                .map(ChatMessage::text)
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
        other => other.clone(),
    }
}
