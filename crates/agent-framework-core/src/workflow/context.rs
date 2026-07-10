//! The context handed to each executor during a superstep.

use serde_json::Value;
use std::sync::{Arc, Mutex};

use super::events::WorkflowEvent;
use crate::error::Result;

/// The effects drained from a context after an executor runs: messages to
/// send, workflow outputs, custom events, and info requests.
pub(crate) type DrainedEffects = (
    Vec<WorkflowMessage>,
    Vec<Value>,
    Vec<WorkflowEvent>,
    Vec<(String, Value)>,
);

/// A message routed between executors within the graph.
#[derive(Debug, Clone)]
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
    requests: Vec<(String, Value)>,
}

/// Collects the effects an executor produces while handling a message:
/// messages to send downstream, workflow outputs, and custom events.
///
/// This is a cheap, cloneable handle (interior mutability) so executor
/// closures may own it and hold it across `await` points. Rust equivalent of
/// `WorkflowContext`.
#[derive(Clone)]
pub struct WorkflowContext {
    executor_id: String,
    source_ids: Arc<Vec<String>>,
    inner: Arc<Mutex<Inner>>,
}

impl WorkflowContext {
    pub(crate) fn new(executor_id: String, source_ids: Vec<String>) -> Self {
        Self {
            executor_id,
            source_ids: Arc::new(source_ids),
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

    /// Request external input, recording a `RequestInfo`.
    pub async fn request_info(
        &self,
        request_id: impl Into<String>,
        data: impl Into<Value>,
    ) -> Result<()> {
        self.inner
            .lock()
            .unwrap()
            .requests
            .push((request_id.into(), data.into()));
        Ok(())
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
