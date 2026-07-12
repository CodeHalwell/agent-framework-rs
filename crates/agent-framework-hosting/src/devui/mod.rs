//! DevUI-style HTTP API: entity discovery and OpenAI-Responses-flavored
//! execution, mirroring the Python `agent_framework_devui` server.
//!
//! # Routes
//! - `GET /` and `GET /ui` — the embedded single-file debug page (see
//!   the crate's `ui` module).
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
//! - SupportsAgentRun streaming drives the core `SupportsAgentRun::run_stream` and frames each update
//!   as SSE live (one `response.output_text.delta` per update); the terminal
//!   `response.completed` aggregates the run. Event ordering and payloads match
//!   DevUI's names. Non-streaming requests stay on `SupportsAgentRun::run`. (Workflow
//!   streaming still frames the workflow's own event stream after the run.)

pub mod models;

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use agent_framework_core::types::{AgentResponse, AgentResponseUpdate, Message};
use agent_framework_core::workflow::WorkflowEvent;

use crate::registry::{AgentRecord, EntityRecord, HostState, WorkflowRecord};
use crate::responses::{
    openai_error, responses_from_run, responses_to_run, InputTokensDetails, OutputMessage,
    OutputTokensDetails, ResponseObject, ResponsesRequest, Usage,
};
use crate::sse::{sse_response, sse_response_stream};
use crate::util;
use models::{DiscoveryResponse, EntityInfo, HealthResponse};

/// Build the DevUI router for a registry.
///
/// The stateful API routes are merged with the stateless embedded debug page
/// ([`crate::ui`]) served at `GET /` and `GET /ui`.
pub(crate) fn router(state: Arc<HostState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/entities", get(list_entities))
        .route("/v1/entities/{entity_id}/info", get(entity_info))
        .route("/v1/responses", post(create_response))
        .with_state(state)
        .merge(crate::ui::router())
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
            // The core `SupportsAgentRun` trait exposes no tool list, so this is absent.
            tools: None,
            metadata: Map::new(),
            source: "in_memory",
            instructions: a.instructions.clone(),
            // `model` is not accessible through the `SupportsAgentRun` trait.
            model: None,
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
            model: None,
            // The full executor set is not enumerable through the public
            // `Workflow` API; the start executor is.
            executors: Some(vec![w.workflow.start_executor_id().to_string()]),
            input_schema: Some(json!({ "type": "string" })),
            start_executor_id: Some(w.workflow.start_executor_id().to_string()),
        },
    }
}

// ---------------------------------------------------------------------------
// SupportsAgentRun execution
// ---------------------------------------------------------------------------

async fn run_agent(agent: &AgentRecord, request: &ResponsesRequest, model: String) -> Response {
    let messages = responses_to_run(request);
    let input_len = approx_input_len(&request.input);

    if request.stream {
        // Live streaming: drive `run_stream` and frame each update as OpenAI
        // Responses SSE events (one `response.output_text.delta` per update)
        // as they arrive, then a final `response.completed`.
        let agent = agent.agent.clone();
        let (tx, rx) = futures::channel::mpsc::unbounded::<Value>();
        tokio::spawn(async move {
            let mut framing = AgentStreamFraming::new(model, input_len);
            for ev in framing.preamble() {
                let _ = tx.unbounded_send(ev);
            }
            match agent.run_stream(messages, None, None).await {
                Ok(mut stream) => {
                    let mut had_error = false;
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(update) => {
                                for ev in framing.push_update(&update) {
                                    let _ = tx.unbounded_send(ev);
                                }
                            }
                            Err(e) => {
                                let _ = tx.unbounded_send(framing.error_event(&e.to_string()));
                                had_error = true;
                                break;
                            }
                        }
                    }
                    if !had_error {
                        let _ = tx.unbounded_send(framing.completed());
                    }
                }
                Err(e) => {
                    let _ = tx.unbounded_send(framing.error_event(&e.to_string()));
                }
            }
        });
        sse_response_stream(rx)
    } else {
        let response = match agent.agent.run(messages, None).await {
            Ok(r) => r,
            Err(e) => return execution_error(e.to_string()),
        };
        Json(agent_response_object(&response, &model, input_len)).into_response()
    }
}

/// Build the aggregated (non-streaming) response for an agent run.
///
/// Delegates the OpenAI-Responses shape to [`responses_from_run`], then fills
/// in DevUI's `~4-chars-per-token` usage estimate when the run reported none.
fn agent_response_object(resp: &AgentResponse, model: &str, input_len: usize) -> ResponseObject {
    let mut obj = responses_from_run(resp, &util::resp_id(), model);
    if obj.usage.is_none() {
        let output_len = obj.output_text.as_deref().unwrap_or_default().len();
        obj.usage = Some(usage_estimate(input_len, output_len));
    }
    obj
}

/// Incremental OpenAI-Responses SSE framing for a streamed agent run, driven
/// one [`AgentResponseUpdate`] at a time. Emits the fixed preamble, one
/// `response.output_text.delta` per non-empty update, and a final
/// `response.completed` aggregating the run (text + usage).
struct AgentStreamFraming {
    model: String,
    input_len: usize,
    rid: String,
    mid: String,
    seq: u64,
    /// Updates collected so the terminal `response.completed` can aggregate the
    /// full text and usage via [`AgentResponse::from_updates`].
    collected: Vec<AgentResponseUpdate>,
}

impl AgentStreamFraming {
    fn new(model: String, input_len: usize) -> Self {
        Self {
            model,
            input_len,
            rid: util::resp_id(),
            mid: util::msg_id(),
            seq: 0,
            collected: Vec::new(),
        }
    }

    fn next(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    /// The four fixed opening events (`response.created` … `content_part.added`).
    fn preamble(&mut self) -> Vec<Value> {
        let in_progress = serde_json::to_value(ResponseObject::in_progress(&self.rid, &self.model))
            .unwrap_or(Value::Null);
        let mid = self.mid.clone();
        vec![
            json!({ "type": "response.created", "sequence_number": self.next(), "response": in_progress.clone() }),
            json!({ "type": "response.in_progress", "sequence_number": self.next(), "response": in_progress }),
            json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "sequence_number": self.next(),
                "item": { "type": "message", "id": mid, "role": "assistant", "content": [], "status": "in_progress" }
            }),
            json!({
                "type": "response.content_part.added",
                "output_index": 0,
                "content_index": 0,
                "item_id": mid,
                "sequence_number": self.next(),
                "part": { "type": "output_text", "text": "", "annotations": [] }
            }),
        ]
    }

    /// Frame one streamed update: a `response.output_text.delta` when it carries
    /// text (otherwise nothing). The update is retained for final aggregation.
    fn push_update(&mut self, update: &AgentResponseUpdate) -> Vec<Value> {
        self.collected.push(update.clone());
        let delta = update.text();
        if delta.is_empty() {
            return Vec::new();
        }
        let mid = self.mid.clone();
        let seq = self.next();
        vec![json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "item_id": mid,
            "delta": delta,
            "logprobs": [],
            "sequence_number": seq,
        })]
    }

    /// The terminal `response.completed`, aggregating all collected updates.
    fn completed(&mut self) -> Value {
        let response = AgentResponse::from_updates(std::mem::take(&mut self.collected));
        let mut completed = responses_from_run(&response, &self.rid, &self.model);
        // The streamed response item id (`mid`) was already announced in the
        // preamble; use it here too instead of `responses_from_run`'s freshly
        // generated one, so the completed event refers to the same item.
        completed.output = vec![OutputMessage::assistant_text(
            self.mid.clone(),
            completed.output_text.clone().unwrap_or_default(),
        )];
        if completed.usage.is_none() {
            let output_len = completed.output_text.as_deref().unwrap_or_default().len();
            completed.usage = Some(usage_estimate(self.input_len, output_len));
        }
        let seq = self.next();
        json!({
            "type": "response.completed",
            "sequence_number": seq,
            "response": serde_json::to_value(completed).unwrap_or(Value::Null),
        })
    }

    /// An in-band error event for a failure that occurs after streaming began.
    fn error_event(&mut self, message: &str) -> Value {
        let seq = self.next();
        json!({
            "type": "error",
            "sequence_number": seq,
            "message": format!("Request execution failed: {message}"),
        })
    }
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
        WorkflowEvent::Intermediate {
            data,
            source_executor_id,
        } => {
            // Non-terminal progress signal, analogous to `Output` but never
            // recorded as the run's final output: surfaced as a workflow
            // debug event rather than an output message item.
            out.push(json!({
                "type": "response.workflow_event.completed",
                "item_id": ctx.item_id,
                "output_index": ctx.output_index.max(0),
                "sequence_number": ctx.next(),
                "data": {
                    "event_type": "WorkflowIntermediateEvent",
                    "source_executor_id": source_executor_id,
                    "data": data,
                },
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
        WorkflowEvent::Intermediate {
            data,
            source_executor_id,
        } => json!({
            "event_type": "WorkflowIntermediateEvent",
            "source_executor_id": source_executor_id,
            "data": data,
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

/// DevUI's fallback usage estimate (~4 characters per token) for runs that
/// report no usage details, applied on top of [`responses_from_run`]'s
/// pass-through usage mapping.
fn usage_estimate(input_len: usize, output_len: usize) -> Usage {
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

/// Reduce the OpenAI-style `input` to a value fed to `Workflow::run`.
///
/// Strings and structured objects pass through; a message array is flattened to
/// its concatenated user text so a string-typed start executor still works.
fn input_to_workflow_value(input: &Value) -> Value {
    match input {
        Value::String(s) => Value::String(s.clone()),
        Value::Null => Value::String(String::new()),
        Value::Array(_) => {
            let text = crate::responses::input_to_messages(input)
                .iter()
                .map(Message::text)
                .collect::<Vec<_>>()
                .join("\n");
            Value::String(text)
        }
        other => other.clone(),
    }
}
