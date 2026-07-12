//! A2A (SupportsAgentRun-to-SupportsAgentRun) protocol hosting.
//!
//! Serves a single agent over the A2A protocol (v0.3.x):
//! - `GET /.well-known/agent-card.json` — the [`AgentCard`] (camelCase per spec).
//! - `POST /` — JSON-RPC 2.0. Methods: `message/send`, `tasks/get`,
//!   `tasks/cancel` (`message/stream` is rejected as unsupported since the card
//!   advertises `streaming: false`).
//!
//! # Type ownership (note for coordinator)
//! The A2A wire types ([`AgentCard`], the JSON-RPC envelope, Message/Part/Task/
//! Artifact) are duplicated **locally** here on purpose: the sibling
//! `agent-framework-a2a` crate (an A2A *client*) is being built in parallel and
//! we must not depend on it. These few serde structs are intentionally
//! duplicated; dedup into a shared `agent-framework-a2a-types` crate later.
//!
//! # Divergences
//! - `skills` are derived from the agent's name/description (one skill), not
//!   from its tools: the core `SupportsAgentRun` trait exposes no tool list. Callers can
//!   override via [`A2ARouter::skill`].
//! - Every `message/send` completes synchronously into a terminal `completed`
//!   `Task`; there is no `working`/`input-required` lifecycle. Consequently
//!   `tasks/cancel` on a known task returns `TaskNotCancelableError` (-32002).
//! - `TaskStatus.timestamp` (optional in the spec) is omitted.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::types::Message;

use crate::registry::IntoAgentRegistration;

// JSON-RPC standard error codes.
const PARSE_ERROR: i64 = -32700;
const INVALID_REQUEST: i64 = -32600;
const METHOD_NOT_FOUND: i64 = -32601;
const INVALID_PARAMS: i64 = -32602;
// A2A-specific error codes.
const TASK_NOT_FOUND: i64 = -32001;
const TASK_NOT_CANCELABLE: i64 = -32002;
const UNSUPPORTED_OPERATION: i64 = -32004;

// ---------------------------------------------------------------------------
// Wire types (local duplicates — see module docs)
// ---------------------------------------------------------------------------

/// The A2A agent card served at `/.well-known/agent-card.json`.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCard {
    pub protocol_version: String,
    pub name: String,
    pub description: String,
    pub url: String,
    pub preferred_transport: String,
    pub version: String,
    pub capabilities: AgentCapabilities,
    pub default_input_modes: Vec<String>,
    pub default_output_modes: Vec<String>,
    pub skills: Vec<AgentSkill>,
}

/// Advertised agent capabilities. This host does not stream A2A responses.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentCapabilities {
    pub streaming: bool,
    pub push_notifications: bool,
    pub state_transition_history: bool,
}

/// A skill entry on the agent card.
#[derive(Debug, Clone, Serialize)]
pub struct AgentSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub examples: Vec<String>,
}

/// An A2A message (`kind: "message"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct A2AMessage {
    kind: String,
    role: String,
    message_id: String,
    parts: Vec<A2APart>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    context_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    task_id: Option<String>,
}

/// A message/artifact part. Only `kind: "text"` is produced here.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct A2APart {
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    text: Option<String>,
}

impl A2APart {
    fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "text".to_string(),
            text: Some(text.into()),
        }
    }
}

/// An A2A task (`kind: "task"`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct A2ATask {
    id: String,
    context_id: String,
    status: A2ATaskStatus,
    artifacts: Vec<A2AArtifact>,
    history: Vec<A2AMessage>,
    kind: String,
}

/// Task status. `timestamp` (optional) is omitted.
#[derive(Debug, Clone, Serialize)]
struct A2ATaskStatus {
    state: String,
}

/// A task artifact.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct A2AArtifact {
    artifact_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    parts: Vec<A2APart>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

struct A2AState {
    card: AgentCard,
    agent: Arc<dyn SupportsAgentRun>,
    tasks: Mutex<HashMap<String, A2ATask>>,
}

/// Serves one agent over the A2A protocol.
pub struct A2ARouter {
    card: AgentCard,
    agent: Arc<dyn SupportsAgentRun>,
}

impl A2ARouter {
    /// Build an A2A host for `agent`, advertised at `base_url` (the JSON-RPC
    /// endpoint URL that clients POST to).
    ///
    /// Accepts a [`Agent`](agent_framework_core::agent::Agent), a
    /// [`WorkflowAgent`](agent_framework_core::workflow::WorkflowAgent), or an
    /// `Arc<dyn SupportsAgentRun>`.
    pub fn for_agent(
        name: impl Into<String>,
        agent: impl IntoAgentRegistration,
        base_url: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let reg = agent.into_agent_registration();
        let description = reg.description.clone().unwrap_or_default();
        let skill = AgentSkill {
            id: name.clone(),
            name: name.clone(),
            description: description.clone(),
            tags: Vec::new(),
            examples: Vec::new(),
        };
        let card = AgentCard {
            protocol_version: "0.3.0".to_string(),
            name,
            description,
            url: base_url.into(),
            preferred_transport: "JSONRPC".to_string(),
            version: "1.0.0".to_string(),
            capabilities: AgentCapabilities {
                streaming: false,
                push_notifications: false,
                state_transition_history: false,
            },
            default_input_modes: vec!["text".to_string()],
            default_output_modes: vec!["text".to_string()],
            skills: vec![skill],
        };
        Self {
            card,
            agent: reg.agent,
        }
    }

    /// Override the advertised card version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.card.version = version.into();
        self
    }

    /// Replace the default skill list with an explicit one.
    pub fn skill(mut self, skill: AgentSkill) -> Self {
        self.card.skills = vec![skill];
        self
    }

    /// Add a skill to the card.
    pub fn add_skill(mut self, skill: AgentSkill) -> Self {
        self.card.skills.push(skill);
        self
    }

    /// The agent card (for inspection/testing).
    pub fn card(&self) -> &AgentCard {
        &self.card
    }

    /// Build the axum router. Composable/nestable into a larger app.
    pub fn into_router(self) -> Router {
        let state = Arc::new(A2AState {
            card: self.card,
            agent: self.agent,
            tasks: Mutex::new(HashMap::new()),
        });
        Router::new()
            .route("/.well-known/agent-card.json", get(agent_card))
            .route("/", post(json_rpc))
            .with_state(state)
    }
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn agent_card(State(state): State<Arc<A2AState>>) -> Json<AgentCard> {
    Json(state.card.clone())
}

async fn json_rpc(State(state): State<Arc<A2AState>>, body: String) -> Response {
    // Parse the envelope ourselves so malformed input becomes a proper
    // JSON-RPC error rather than an axum 4xx.
    let value: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return Json(rpc_error(
                Value::Null,
                PARSE_ERROR,
                format!("Parse error: {e}"),
            ))
            .into_response();
        }
    };

    let id = value.get("id").cloned().unwrap_or(Value::Null);

    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return Json(rpc_error(
            id,
            INVALID_REQUEST,
            "Invalid Request: missing 'method'",
        ))
        .into_response();
    };
    let params = value.get("params").cloned().unwrap_or(Value::Null);

    let result = match method {
        "message/send" => handle_message_send(&state, params).await,
        "tasks/get" => handle_tasks_get(&state, params),
        "tasks/cancel" => handle_tasks_cancel(&state, params),
        "message/stream" => Err(RpcErr::new(
            UNSUPPORTED_OPERATION,
            "Streaming is not supported by this agent",
        )),
        other => Err(RpcErr::new(
            METHOD_NOT_FOUND,
            format!("Method not found: {other}"),
        )),
    };

    match result {
        Ok(value) => Json(rpc_ok(id, value)).into_response(),
        Err(err) => Json(rpc_error(id, err.code, err.message)).into_response(),
    }
}

async fn handle_message_send(state: &A2AState, params: Value) -> Result<Value, RpcErr> {
    let message: A2AMessage = params
        .get("message")
        .cloned()
        .ok_or_else(|| RpcErr::new(INVALID_PARAMS, "Invalid params: missing 'message'"))
        .and_then(|m| {
            serde_json::from_value(m)
                .map_err(|e| RpcErr::new(INVALID_PARAMS, format!("Invalid params: {e}")))
        })?;

    let input_text = message
        .parts
        .iter()
        .filter_map(|p| p.text.as_deref())
        .collect::<Vec<_>>()
        .join("");

    let run = state
        .agent
        .run(vec![Message::user(input_text)], None)
        .await
        .map_err(|e| RpcErr::new(-32603, format!("Agent execution failed: {e}")))?;
    let response_text = run.text();

    let context_id = message
        .context_id
        .clone()
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let task_id = uuid::Uuid::new_v4().to_string();

    let response_message = A2AMessage {
        kind: "message".to_string(),
        role: "agent".to_string(),
        message_id: uuid::Uuid::new_v4().to_string(),
        parts: vec![A2APart::text(response_text.clone())],
        context_id: Some(context_id.clone()),
        task_id: Some(task_id.clone()),
    };

    let task = A2ATask {
        id: task_id.clone(),
        context_id,
        status: A2ATaskStatus {
            state: "completed".to_string(),
        },
        artifacts: vec![A2AArtifact {
            artifact_id: uuid::Uuid::new_v4().to_string(),
            name: Some("response".to_string()),
            parts: vec![A2APart::text(response_text)],
        }],
        history: vec![message, response_message],
        kind: "task".to_string(),
    };

    let value = serde_json::to_value(&task).unwrap_or(Value::Null);
    state
        .tasks
        .lock()
        .expect("tasks mutex poisoned")
        .insert(task_id, task);
    Ok(value)
}

fn handle_tasks_get(state: &A2AState, params: Value) -> Result<Value, RpcErr> {
    let id = task_id_param(&params)?;
    match state.tasks.lock().expect("tasks mutex poisoned").get(&id) {
        Some(task) => Ok(serde_json::to_value(task).unwrap_or(Value::Null)),
        None => Err(RpcErr::new(TASK_NOT_FOUND, format!("Task not found: {id}"))),
    }
}

fn handle_tasks_cancel(state: &A2AState, params: Value) -> Result<Value, RpcErr> {
    let id = task_id_param(&params)?;
    let exists = state
        .tasks
        .lock()
        .expect("tasks mutex poisoned")
        .contains_key(&id);
    if exists {
        // Tasks complete synchronously and are terminal, so they cannot be
        // canceled.
        Err(RpcErr::new(
            TASK_NOT_CANCELABLE,
            format!("Task {id} is in a terminal state and cannot be canceled"),
        ))
    } else {
        Err(RpcErr::new(TASK_NOT_FOUND, format!("Task not found: {id}")))
    }
}

/// Extract `params.id` (the `TaskIdParams` shape).
fn task_id_param(params: &Value) -> Result<String, RpcErr> {
    params
        .get("id")
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| RpcErr::new(INVALID_PARAMS, "Invalid params: missing task 'id'"))
}

// ---------------------------------------------------------------------------
// JSON-RPC envelope helpers
// ---------------------------------------------------------------------------

/// A JSON-RPC error to surface to the client.
struct RpcErr {
    code: i64,
    message: String,
}

impl RpcErr {
    fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

fn rpc_ok(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn rpc_error(id: Value, code: i64, message: impl Into<String>) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message.into() },
    })
}
