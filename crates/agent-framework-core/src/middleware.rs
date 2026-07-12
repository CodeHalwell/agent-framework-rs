//! Middleware pipelines for agents, chat clients, and function invocation.
//!
//! Rust equivalent of `agent_framework._middleware`. Middleware receives an
//! owned context and a [`Next`] continuation. Call `next.run(ctx)` to continue
//! the chain, mutate the context to observe/override results, or return the
//! context directly (optionally with `terminate = true`) to short-circuit.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::Result;
use crate::tools::BoxFuture;
use crate::types::{AgentResponse, ChatMessage, ChatOptions, ChatResponse};

/// The terminal handler invoked at the end of a middleware chain.
pub type Terminal<C> = Box<dyn FnOnce(C) -> BoxFuture<Result<C>> + Send>;

/// A middleware that transforms a context of type `C`.
#[async_trait]
pub trait Middleware<C: Send + 'static>: Send + Sync {
    async fn process(&self, ctx: C, next: Next<C>) -> Result<C>;
}

/// The continuation passed to a [`Middleware`]. Calling [`Next::run`] invokes
/// the remaining middleware and, finally, the terminal handler.
pub struct Next<C: Send + 'static> {
    middlewares: Arc<Vec<Arc<dyn Middleware<C>>>>,
    index: usize,
    terminal: Option<Terminal<C>>,
}

impl<C: Send + 'static> Next<C> {
    /// Continue the chain with the given context.
    pub async fn run(mut self, ctx: C) -> Result<C> {
        if self.index < self.middlewares.len() {
            let mw = self.middlewares[self.index].clone();
            let next = Next {
                middlewares: self.middlewares.clone(),
                index: self.index + 1,
                terminal: self.terminal.take(),
            };
            mw.process(ctx, next).await
        } else if let Some(term) = self.terminal.take() {
            term(ctx).await
        } else {
            Ok(ctx)
        }
    }
}

/// A pipeline of middleware of a single category.
pub struct MiddlewarePipeline<C: Send + 'static> {
    middlewares: Arc<Vec<Arc<dyn Middleware<C>>>>,
}

impl<C: Send + 'static> Default for MiddlewarePipeline<C> {
    fn default() -> Self {
        Self {
            middlewares: Arc::new(Vec::new()),
        }
    }
}

impl<C: Send + 'static> Clone for MiddlewarePipeline<C> {
    fn clone(&self) -> Self {
        Self {
            middlewares: self.middlewares.clone(),
        }
    }
}

impl<C: Send + 'static> MiddlewarePipeline<C> {
    pub fn new(middlewares: Vec<Arc<dyn Middleware<C>>>) -> Self {
        Self {
            middlewares: Arc::new(middlewares),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Execute the pipeline, running `terminal` after all middleware.
    pub async fn execute(&self, ctx: C, terminal: Terminal<C>) -> Result<C> {
        let next = Next {
            middlewares: self.middlewares.clone(),
            index: 0,
            terminal: Some(terminal),
        };
        next.run(ctx).await
    }
}

/// Context flowing through the agent middleware pipeline.
pub struct AgentContext {
    pub messages: Vec<ChatMessage>,
    pub is_streaming: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    /// The run result; populated by the terminal handler or overridden here.
    pub result: Option<AgentResponse>,
    /// If set to true, the pipeline stops without running further middleware.
    pub terminate: bool,
}

impl AgentContext {
    pub fn new(messages: Vec<ChatMessage>, is_streaming: bool) -> Self {
        Self {
            messages,
            is_streaming,
            metadata: HashMap::new(),
            result: None,
            terminate: false,
        }
    }
}

/// Context flowing through the chat middleware pipeline.
pub struct ChatContext {
    pub messages: Vec<ChatMessage>,
    pub chat_options: ChatOptions,
    pub is_streaming: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    pub result: Option<ChatResponse>,
    pub terminate: bool,
}

impl ChatContext {
    pub fn new(messages: Vec<ChatMessage>, chat_options: ChatOptions, is_streaming: bool) -> Self {
        Self {
            messages,
            chat_options,
            is_streaming,
            metadata: HashMap::new(),
            result: None,
            terminate: false,
        }
    }
}

/// Context flowing through the function middleware pipeline.
pub struct FunctionInvocationContext {
    pub function_name: String,
    pub arguments: serde_json::Value,
    pub metadata: HashMap<String, serde_json::Value>,
    pub result: Option<serde_json::Value>,
    pub terminate: bool,
}

impl FunctionInvocationContext {
    pub fn new(function_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            function_name: function_name.into(),
            arguments,
            metadata: HashMap::new(),
            result: None,
            terminate: false,
        }
    }
}

/// Convenience type aliases for each middleware category.
pub type AgentMiddleware = dyn Middleware<AgentContext>;
/// Chat middleware operates on a [`ChatContext`].
pub type ChatMiddleware = dyn Middleware<ChatContext>;
/// Function middleware operates on a [`FunctionInvocationContext`].
pub type FunctionMiddleware = dyn Middleware<FunctionInvocationContext>;

/// Adapter to build a [`Middleware`] from an async closure.
pub struct FnMiddleware<C, F> {
    f: F,
    _marker: std::marker::PhantomData<fn(C)>,
}

impl<C, F> FnMiddleware<C, F> {
    pub fn new(f: F) -> Self {
        Self {
            f,
            _marker: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<C, F, Fut> Middleware<C> for FnMiddleware<C, F>
where
    C: Send + 'static,
    F: Fn(C, Next<C>) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<C>> + Send,
{
    async fn process(&self, ctx: C, next: Next<C>) -> Result<C> {
        (self.f)(ctx, next).await
    }
}
