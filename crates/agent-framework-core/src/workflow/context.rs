//! The context handed to each executor during a superstep.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::events::WorkflowEvent;
use super::shared_state::SharedState;
use crate::error::Result;

/// A request for external information recorded by an executor, before the
/// runner assigns it as a pending request and emits a `RequestInfo` event.
pub(crate) struct RequestDraft {
    pub request_id: String,
    /// The executor the eventual response should be routed back to.
    pub reply_to: String,
    pub data: Value,
}

/// The effects drained from a context after an executor runs: messages to
/// send, workflow outputs, custom events, and info requests.
pub(crate) type DrainedEffects = (
    Vec<WorkflowMessage>,
    Vec<Value>,
    Vec<WorkflowEvent>,
    Vec<RequestDraft>,
);

/// A message routed between executors within the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowMessage {
    pub data: Value,
    pub source_id: String,
    /// If set, deliver only to this target; otherwise route along all edges.
    pub target_id: Option<String>,
}

#[derive(Default)]
struct Inner {
    sent: Vec<WorkflowMessage>,
    outputs: Vec<Value>,
    events: Vec<WorkflowEvent>,
    requests: Vec<RequestDraft>,
}

/// Collects the effects an executor produces while handling a message:
/// messages to send downstream, workflow outputs, custom events, info
/// requests, and access to run-scoped [`SharedState`].
///
/// This is a cheap, cloneable handle (interior mutability) so executor
/// closures may own it and hold it across `await` points. Rust equivalent of
/// `WorkflowContext`.
#[derive(Clone)]
pub struct WorkflowContext {
    executor_id: String,
    source_ids: Arc<Vec<String>>,
    shared: SharedState,
    inner: Arc<Mutex<Inner>>,
}

impl WorkflowContext {
    pub(crate) fn new(executor_id: String, source_ids: Vec<String>, shared: SharedState) -> Self {
        Self {
            executor_id,
            source_ids: Arc::new(source_ids),
            shared,
            inner: Arc::new(Mutex::new(Inner::default())),
        }
    }

    /// Send a message to all connected downstream executors.
    pub async fn send_message(&self, data: impl Into<Value>) -> Result<()> {
        self.inner.lock().unwrap().sent.push(WorkflowMessage {
            data: data.into(),
            source_id: self.executor_id.clone(),
            target_id: None,
        });
        Ok(())
    }

    /// Send a message to a specific downstream executor.
    pub async fn send_to(
        &self,
        target_id: impl Into<String>,
        data: impl Into<Value>,
    ) -> Result<()> {
        self.inner.lock().unwrap().sent.push(WorkflowMessage {
            data: data.into(),
            source_id: self.executor_id.clone(),
            target_id: Some(target_id.into()),
        });
        Ok(())
    }

    /// Emit a workflow-level output.
    pub async fn yield_output(&self, data: impl Into<Value>) -> Result<()> {
        self.inner.lock().unwrap().outputs.push(data.into());
        Ok(())
    }

    /// Add a custom event to the event stream.
    pub fn add_event(&self, event: WorkflowEvent) {
        self.inner.lock().unwrap().events.push(event);
    }

    /// Request external information (human-in-the-loop).
    ///
    /// Records a request whose response, once supplied via the run handle, is
    /// delivered back to this executor as a message. Mirrors Python's
    /// `ctx.request_info`.
    pub async fn request_info(&self, data: impl Into<Value>) -> Result<()> {
        let reply_to = self.executor_id.clone();
        self.record_request(reply_to, data.into());
        Ok(())
    }

    /// Record a request whose response is routed back to `reply_to`.
    ///
    /// Used by [`RequestInfoExecutor`](super::RequestInfoExecutor) to route a
    /// response to the upstream requester rather than to itself. A fresh
    /// `request_id` is generated.
    pub(crate) fn record_request(&self, reply_to: String, data: Value) {
        self.record_request_with_id(uuid::Uuid::new_v4().to_string(), reply_to, data);
    }

    /// Record a request with an explicit `request_id`.
    ///
    /// Used by [`WorkflowExecutor`](super::WorkflowExecutor) to forward a
    /// sub-workflow's request under its original id so responses correlate.
    pub(crate) fn record_request_with_id(&self, request_id: String, reply_to: String, data: Value) {
        self.inner.lock().unwrap().requests.push(RequestDraft {
            request_id,
            reply_to,
            data,
        });
    }

    /// Access the run-scoped [`SharedState`] shared by all executors.
    pub fn shared_state(&self) -> SharedState {
        self.shared.clone()
    }

    /// The id of the executor this context belongs to.
    pub fn executor_id(&self) -> &str {
        &self.executor_id
    }

    /// The id(s) of the executor(s) that sent the current message.
    pub fn source_executor_ids(&self) -> &[String] {
        &self.source_ids
    }

    /// Drain the accumulated effects (used by the runner).
    pub(crate) fn take(&self) -> DrainedEffects {
        let mut inner = self.inner.lock().unwrap();
        (
            std::mem::take(&mut inner.sent),
            std::mem::take(&mut inner.outputs),
            std::mem::take(&mut inner.events),
            std::mem::take(&mut inner.requests),
        )
    }
}
