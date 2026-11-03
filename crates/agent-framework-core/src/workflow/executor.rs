//! The executor trait and function-style executors.

use async_trait::async_trait;
use serde_json::Value;
use std::future::Future;
use std::sync::Arc;

use super::context::WorkflowContext;
use crate::error::Result;

/// A node in a workflow graph.
///
/// Rust equivalent of the Python `Executor`. Each executor receives a message
/// (as a JSON value) and a [`WorkflowContext`] handle through which it sends
/// messages, yields outputs, and emits events.
#[async_trait]
pub trait Executor: Send + Sync {
    /// A unique id for this executor within the workflow.
    fn id(&self) -> &str;

    /// Handle an incoming message.
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()>;

    /// Capture serializable state for checkpointing.
    ///
    /// Stateful executors override this to return a JSON snapshot that is stored
    /// in the checkpoint and handed back to [`Executor::restore_state`] on
    /// resume. Returning `None` (the default) means the executor is stateless.
    async fn snapshot_state(&self) -> Option<Value> {
        None
    }

    /// Restore state previously produced by [`Executor::snapshot_state`].
    ///
    /// The default implementation ignores the state. Only executors that
    /// snapshot state need to implement this.
    async fn restore_state(&self, _state: Value) -> Result<()> {
        Ok(())
    }

    /// Reset per-run state before a fresh run (upstream `IResettableExecutor`):
    /// an executor instance is shared by every run of its workflow, so one
    /// that keeps per-run state clears it here. Called on every executor at
    /// the start of a run (not on a resume from a checkpoint). The default
    /// does nothing.
    async fn reset(&self) -> Result<()> {
        Ok(())
    }

    /// The message types this executor accepts, sends and yields (upstream
    /// `DescribeProtocol`). Payloads here are JSON, so types are names the
    /// workflow's author chooses (or JSON type names). The default accepts
    /// anything and declares nothing.
    fn describe_protocol(&self) -> super::protocol::ProtocolDescriptor {
        super::protocol::ProtocolDescriptor::accepts_all()
    }

    /// For a request port (a node that hands requests to the caller), its
    /// request and response type names; `None` for other executors.
    fn request_port(&self) -> Option<super::protocol::RequestPortInfo> {
        None
    }
}

type ExecFn =
    Arc<dyn Fn(Value, WorkflowContext) -> crate::tools::BoxFuture<Result<()>> + Send + Sync>;

/// An executor built from an async closure. Rust analogue of `@executor`.
#[derive(Clone)]
pub struct FunctionExecutor {
    id: String,
    func: ExecFn,
    protocol: Option<super::protocol::ProtocolDescriptor>,
}

impl FunctionExecutor {
    /// Create a function executor from an async closure.
    ///
    /// The closure receives the message and an owned [`WorkflowContext`] handle
    /// (cheap to clone), so it may freely hold the context across `await`s.
    pub fn new<F, Fut>(id: impl Into<String>, func: F) -> Self
    where
        F: Fn(Value, WorkflowContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self {
            id: id.into(),
            func: Arc::new(move |value, ctx| Box::pin(func(value, ctx))),
            protocol: None,
        }
    }

    /// Declare what this executor accepts, sends and yields.
    pub fn with_protocol(mut self, protocol: super::protocol::ProtocolDescriptor) -> Self {
        self.protocol = Some(protocol);
        self
    }
}

#[async_trait]
impl Executor for FunctionExecutor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        (self.func)(message, ctx).await
    }
    fn describe_protocol(&self) -> super::protocol::ProtocolDescriptor {
        self.protocol
            .clone()
            .unwrap_or_else(super::protocol::ProtocolDescriptor::accepts_all)
    }
}
