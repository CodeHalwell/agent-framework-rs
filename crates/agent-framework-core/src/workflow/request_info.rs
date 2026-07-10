//! Human-in-the-loop request/response primitives.
//!
//! Mirrors the Agent Framework's request-info pattern. A [`RequestInfoExecutor`]
//! node turns an incoming message into an outstanding request surfaced to the
//! caller as a [`WorkflowEvent::RequestInfo`](super::WorkflowEvent::RequestInfo)
//! event; the run pauses (`IdleWithPendingRequests`) until the caller supplies a
//! response, which is delivered back to the requesting executor as a
//! [`RequestResponse`] message.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::context::WorkflowContext;
use super::executor::Executor;
use crate::error::Result;

/// An outstanding request awaiting an external response.
///
/// Tracked in the live run state and persisted in checkpoints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingRequest {
    /// Correlates the request with its response.
    pub request_id: String,
    /// The executor that surfaced the request (shown as the event source).
    pub source_executor_id: String,
    /// The executor the response is delivered back to.
    #[serde(default)]
    pub reply_to_executor_id: String,
    /// The request payload.
    pub request_data: Value,
}

/// The message delivered back to the requesting executor when a response is
/// supplied. Carries the original request data alongside the response, mirroring
/// Python's response `Message` which retains `original_request`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestResponse {
    /// The id of the request being answered.
    pub request_id: String,
    /// The response payload supplied by the caller.
    pub data: Value,
    /// The original request payload.
    pub original_request: Value,
}

impl RequestResponse {
    /// Attempt to interpret an incoming message value as a [`RequestResponse`].
    pub fn from_message(value: &Value) -> Option<Self> {
        serde_json::from_value(value.clone()).ok()
    }
}

/// A built-in node that surfaces incoming messages as external requests.
///
/// When a message reaches this node it records a request whose response is
/// routed back to the executor that sent the message (the requester), and emits
/// a `RequestInfo` event. The run pauses once the message queue drains while the
/// request is unanswered.
///
/// Divergence from the Python engine: the reference implementation uses
/// `ctx.request_info()` + `@response_handler` on ordinary executors rather than
/// a dedicated node, and (in .NET) routes the response along the node's outgoing
/// edges. Here — per the port's design — the response is delivered back to the
/// requesting (upstream) executor.
pub struct RequestInfoExecutor {
    id: String,
}

impl RequestInfoExecutor {
    /// Create a request-info node with the given id.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

#[async_trait]
impl Executor for RequestInfoExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // Route the eventual response back to whoever sent us this request.
        let reply_to = ctx
            .source_executor_ids()
            .first()
            .cloned()
            .unwrap_or_else(|| ctx.executor_id().to_string());
        ctx.record_request(reply_to, message);
        Ok(())
    }
}
