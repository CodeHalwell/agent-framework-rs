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

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
// Task store
// ---------------------------------------------------------------------------

/// Default maximum number of retained tasks before oldest-first eviction.
const DEFAULT_MAX_TASKS: usize = 1024;
/// Default time-to-live for a retained task.
const DEFAULT_TASK_TTL: Duration = Duration::from_secs(60 * 60);
/// Default cap on the total serialized size of retained tasks (16 MiB).
const DEFAULT_MAX_TOTAL_BYTES: usize = 16 * 1024 * 1024;

/// One retained task plus the bookkeeping used for TTL and size eviction.
struct StoredTask {
    task: A2ATask,
    inserted_at: Instant,
    /// Serialized byte size, tracked so the store can bound total memory.
    size: usize,
}

struct TaskStoreInner {
    tasks: HashMap<String, StoredTask>,
    /// Task ids in insertion order (oldest at the front) for FIFO eviction.
    order: VecDeque<String>,
    total_bytes: usize,
}

/// A bounded, self-pruning in-memory store for A2A tasks.
///
/// Without bounds, `message/send` would accumulate every task's full request,
/// response artifact, and history forever — an unbounded memory leak and a
/// straightforward remote-DoS vector on an exposed endpoint. This store caps
/// the retained task **count** and **total serialized size**, and expires
/// tasks after a **TTL**, pruning on every access. Eviction is oldest-first.
struct A2ATaskStore {
    inner: Mutex<TaskStoreInner>,
    max_tasks: usize,
    ttl: Option<Duration>,
    max_total_bytes: usize,
}

impl A2ATaskStore {
    fn new(max_tasks: usize, ttl: Option<Duration>, max_total_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(TaskStoreInner {
                tasks: HashMap::new(),
                order: VecDeque::new(),
                total_bytes: 0,
            }),
            max_tasks: max_tasks.max(1),
            ttl,
            max_total_bytes: max_total_bytes.max(1),
        }
    }

    /// Drop the front (oldest) entry, keeping `order`/`total_bytes` consistent.
    fn evict_front(inner: &mut TaskStoreInner) {
        while let Some(id) = inner.order.pop_front() {
            if let Some(stored) = inner.tasks.remove(&id) {
                inner.total_bytes = inner.total_bytes.saturating_sub(stored.size);
                return;
            }
            // `id` was already removed (e.g. via `remove`); skip the stale entry.
        }
    }

    /// Remove any entries older than the TTL (front-to-back; insertion order is
    /// age order). `now` is passed in so the logic is testable.
    fn prune_expired(&self, inner: &mut TaskStoreInner, now: Instant) {
        let Some(ttl) = self.ttl else { return };
        while let Some(id) = inner.order.front().cloned() {
            let expired = inner
                .tasks
                .get(&id)
                // `saturating_duration_since` avoids a panic if the clock
                // appears to move backwards (NTP steps, VM time anomalies).
                .map(|s| now.saturating_duration_since(s.inserted_at) >= ttl)
                // A stale id with no entry is always "prunable".
                .unwrap_or(true);
            if !expired {
                break;
            }
            inner.order.pop_front();
            if let Some(stored) = inner.tasks.remove(&id) {
                inner.total_bytes = inner.total_bytes.saturating_sub(stored.size);
            }
        }
    }

    /// Insert (or replace) a task, then enforce the TTL, count, and size caps.
    ///
    /// A single task whose own serialized size exceeds the total byte budget is
    /// **not retained**: retaining it would pin memory above the configured cap
    /// (the eviction loop can't drop the sole/newest entry), defeating the very
    /// bound meant to guard against a remote-DoS oversized response. Dropping it
    /// only means a later `tasks/get` won't find it — the response was already
    /// returned to the caller synchronously.
    fn insert(&self, id: String, task: A2ATask) {
        let size = serde_json::to_vec(&task).map(|v| v.len()).unwrap_or(0);
        if size > self.max_total_bytes {
            tracing::warn!(
                task_bytes = size,
                cap = self.max_total_bytes,
                "A2A task exceeds the retention size cap; not retained"
            );
            return;
        }
        let mut inner = self.inner.lock().expect("task store mutex poisoned");
        self.prune_expired(&mut inner, Instant::now());

        // Replacing an existing id: drop its old size/order entry first.
        if let Some(old) = inner.tasks.remove(&id) {
            inner.total_bytes = inner.total_bytes.saturating_sub(old.size);
            if let Some(pos) = inner.order.iter().position(|x| x == &id) {
                inner.order.remove(pos);
            }
        }

        inner.total_bytes = inner.total_bytes.saturating_add(size);
        inner.order.push_back(id.clone());
        inner.tasks.insert(
            id,
            StoredTask {
                task,
                inserted_at: Instant::now(),
                size,
            },
        );

        // Enforce caps. Oversized single tasks were already rejected above, so
        // once the store is down to the newest entry it is guaranteed to be
        // within the byte budget — both loops terminate.
        while inner.tasks.len() > self.max_tasks {
            Self::evict_front(&mut inner);
        }
        while inner.total_bytes > self.max_total_bytes && inner.order.len() > 1 {
            Self::evict_front(&mut inner);
        }
    }

    /// Fetch a task by id, pruning expired entries first (so an expired task
    /// reads as absent).
    fn get(&self, id: &str) -> Option<A2ATask> {
        let mut inner = self.inner.lock().expect("task store mutex poisoned");
        self.prune_expired(&mut inner, Instant::now());
        inner.tasks.get(id).map(|s| s.task.clone())
    }

    /// Whether a (non-expired) task with `id` is retained.
    fn contains(&self, id: &str) -> bool {
        self.get(id).is_some()
    }

    /// Current retained task count (after pruning). Test/inspection helper.
    #[cfg(test)]
    fn len(&self) -> usize {
        let mut inner = self.inner.lock().expect("task store mutex poisoned");
        self.prune_expired(&mut inner, Instant::now());
        inner.tasks.len()
    }

    /// Prune as of an explicit `now`, for deterministic TTL tests.
    #[cfg(test)]
    fn force_prune(&self, now: Instant) {
        let mut inner = self.inner.lock().expect("task store mutex poisoned");
        self.prune_expired(&mut inner, now);
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

struct A2AState {
    card: AgentCard,
    agent: Arc<dyn SupportsAgentRun>,
    tasks: A2ATaskStore,
}

/// Serves one agent over the A2A protocol.
pub struct A2ARouter {
    card: AgentCard,
    agent: Arc<dyn SupportsAgentRun>,
    max_tasks: usize,
    task_ttl: Option<Duration>,
    max_task_bytes: usize,
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
            max_tasks: DEFAULT_MAX_TASKS,
            task_ttl: Some(DEFAULT_TASK_TTL),
            max_task_bytes: DEFAULT_MAX_TOTAL_BYTES,
        }
    }

    /// Override the advertised card version.
    pub fn version(mut self, version: impl Into<String>) -> Self {
        self.card.version = version.into();
        self
    }

    /// Cap the number of retained tasks (oldest-first eviction beyond this).
    /// Defaults to 1024.
    pub fn max_tasks(mut self, max_tasks: usize) -> Self {
        self.max_tasks = max_tasks;
        self
    }

    /// Expire a retained task after `ttl`. Defaults to one hour; pass a large
    /// value (or use [`Self::no_task_ttl`]) to disable time-based expiry.
    pub fn task_ttl(mut self, ttl: Duration) -> Self {
        self.task_ttl = Some(ttl);
        self
    }

    /// Disable TTL-based task expiry (tasks are still bounded by count/size).
    pub fn no_task_ttl(mut self) -> Self {
        self.task_ttl = None;
        self
    }

    /// Cap the total serialized size (bytes) of retained tasks. Defaults to
    /// 16 MiB; oldest tasks are evicted to stay under the cap.
    pub fn max_task_bytes(mut self, bytes: usize) -> Self {
        self.max_task_bytes = bytes;
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
            tasks: A2ATaskStore::new(self.max_tasks, self.task_ttl, self.max_task_bytes),
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
    state.tasks.insert(task_id, task);
    Ok(value)
}

fn handle_tasks_get(state: &A2AState, params: Value) -> Result<Value, RpcErr> {
    let id = task_id_param(&params)?;
    match state.tasks.get(&id) {
        Some(task) => Ok(serde_json::to_value(task).unwrap_or(Value::Null)),
        None => Err(RpcErr::new(TASK_NOT_FOUND, format!("Task not found: {id}"))),
    }
}

fn handle_tasks_cancel(state: &A2AState, params: Value) -> Result<Value, RpcErr> {
    let id = task_id_param(&params)?;
    let exists = state.tasks.contains(&id);
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

#[cfg(test)]
mod task_store_tests {
    use super::*;

    fn dummy_task(id: &str) -> A2ATask {
        A2ATask {
            id: id.to_string(),
            context_id: "ctx".to_string(),
            status: A2ATaskStatus {
                state: "completed".to_string(),
            },
            artifacts: vec![A2AArtifact {
                artifact_id: "art".to_string(),
                name: Some("response".to_string()),
                parts: vec![A2APart::text("hello world")],
            }],
            history: Vec::new(),
            kind: "task".to_string(),
        }
    }

    #[test]
    fn evicts_oldest_beyond_capacity() {
        let store = A2ATaskStore::new(3, None, DEFAULT_MAX_TOTAL_BYTES);
        for i in 0..5 {
            store.insert(format!("t{i}"), dummy_task(&format!("t{i}")));
        }
        assert_eq!(store.len(), 3, "count must stay capped");
        // The two oldest were evicted; the three newest remain.
        assert!(store.get("t0").is_none());
        assert!(store.get("t1").is_none());
        assert!(store.get("t2").is_some());
        assert!(store.get("t4").is_some());
    }

    #[test]
    fn ttl_expires_old_tasks() {
        let ttl = Duration::from_secs(30);
        let store = A2ATaskStore::new(100, Some(ttl), DEFAULT_MAX_TOTAL_BYTES);
        store.insert("t0".to_string(), dummy_task("t0"));
        assert_eq!(store.len(), 1);
        // Prune as if the TTL had elapsed.
        store.force_prune(Instant::now() + ttl + Duration::from_secs(1));
        assert_eq!(store.len(), 0, "expired task must be pruned");
        assert!(store.get("t0").is_none());
    }

    #[test]
    fn total_size_cap_evicts_oldest() {
        // Budget for roughly two tasks forces eviction down toward two.
        let one = serde_json::to_vec(&dummy_task("t0")).unwrap().len();
        let store = A2ATaskStore::new(1000, None, one * 2 + 1);
        for i in 0..10 {
            store.insert(format!("t{i}"), dummy_task(&format!("t{i}")));
        }
        // The store never grows without bound: only the most recent handful of
        // tasks (bounded by the byte budget) are retained.
        assert!(
            store.len() <= 2,
            "size cap must bound retention: {}",
            store.len()
        );
        assert!(store.get("t9").is_some(), "newest task is always retained");
    }

    #[test]
    fn oversized_single_task_is_not_retained() {
        // A task larger than the entire byte budget must be rejected, not
        // exempted — otherwise a lone oversized task pins memory above the cap.
        let one = serde_json::to_vec(&dummy_task("t0")).unwrap().len();
        let store = A2ATaskStore::new(1000, None, one.saturating_sub(1));
        store.insert("t0".to_string(), dummy_task("t0"));
        assert_eq!(store.len(), 0, "oversized task must not be retained");
        assert!(store.get("t0").is_none());
    }

    #[test]
    fn replacing_same_id_does_not_double_count() {
        let store = A2ATaskStore::new(1000, None, DEFAULT_MAX_TOTAL_BYTES);
        store.insert("t0".to_string(), dummy_task("t0"));
        store.insert("t0".to_string(), dummy_task("t0"));
        assert_eq!(store.len(), 1);
    }
}
