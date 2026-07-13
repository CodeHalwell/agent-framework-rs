//! Middleware pipelines for agents, chat clients, and function invocation.
//!
//! Rust equivalent of `agent_framework._middleware`. Middleware receives an
//! owned context and a [`Next`] continuation. Call `next.run(ctx)` to continue
//! the chain, mutate the context to observe/override results, or return the
//! context directly (optionally with `terminate = true`) to short-circuit.

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::tools::{BoxFuture, ToolDefinition};
use crate::types::{AgentResponse, ChatOptions, ChatResponse, Message};

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
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    /// The run result; populated by the terminal handler or overridden here.
    pub result: Option<AgentResponse>,
    /// If set to true, the pipeline stops without running further middleware.
    pub terminate: bool,
}

impl AgentContext {
    pub fn new(messages: Vec<Message>, is_streaming: bool) -> Self {
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
    pub messages: Vec<Message>,
    pub chat_options: ChatOptions,
    pub is_streaming: bool,
    pub metadata: HashMap<String, serde_json::Value>,
    pub result: Option<ChatResponse>,
    pub terminate: bool,
}

impl ChatContext {
    pub fn new(messages: Vec<Message>, chat_options: ChatOptions, is_streaming: bool) -> Self {
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

/// The live, mutable tool list of an in-flight agent run (progressive tool
/// exposure).
///
/// Handed to function middleware and tools via
/// [`FunctionInvocationContext::tools`]; a clone is a view onto the *same*
/// list. Mutations take effect on the **next** iteration of the
/// function-calling loop — they never affect tool calls already requested in
/// the in-flight batch, because the loop snapshots the list once per model
/// iteration. Mirrors upstream `FunctionInvocationContext.tools` +
/// `add_tools`/`remove_tools` (`_middleware.py`).
#[derive(Clone, Default)]
pub struct LiveToolList {
    inner: Arc<std::sync::Mutex<Vec<ToolDefinition>>>,
}

impl LiveToolList {
    /// A live list seeded with the run's current tools.
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        Self {
            inner: Arc::new(std::sync::Mutex::new(tools)),
        }
    }

    /// Add tools to the run (available to the model on the next iteration).
    ///
    /// Errors if any added tool's name collides with a tool already in the
    /// list (mirrors upstream's `ValueError` on duplicate names); the whole
    /// batch is validated first, so a duplicate leaves the list unchanged.
    pub fn add_tools(&self, tools: impl IntoIterator<Item = ToolDefinition>) -> Result<()> {
        let batch: Vec<ToolDefinition> = tools.into_iter().collect();
        let mut list = self.inner.lock().unwrap();
        for tool in &batch {
            if list.iter().any(|t| t.name == tool.name)
                || batch.iter().filter(|t| t.name == tool.name).count() > 1
            {
                return Err(Error::Configuration(format!(
                    "cannot add tool '{}': a tool with that name already exists in this run",
                    tool.name
                )));
            }
        }
        list.extend(batch);
        Ok(())
    }

    /// Remove tools by name (effective on the next iteration). Names not
    /// currently present are ignored.
    pub fn remove_tools<'a>(&self, names: impl IntoIterator<Item = &'a str>) {
        let to_remove: std::collections::HashSet<&str> = names.into_iter().collect();
        self.inner
            .lock()
            .unwrap()
            .retain(|t| !to_remove.contains(t.name.as_str()));
    }

    /// Whether a tool with `name` is currently in the list.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.lock().unwrap().iter().any(|t| t.name == name)
    }

    /// A point-in-time copy of the list (what the next iteration will use).
    pub fn snapshot(&self) -> Vec<ToolDefinition> {
        self.inner.lock().unwrap().clone()
    }
}

/// Context flowing through the function middleware pipeline.
pub struct FunctionInvocationContext {
    pub function_name: String,
    pub arguments: serde_json::Value,
    /// The [`AgentSession`](crate::session::AgentSession) of the agent run
    /// this invocation belongs to, if the call originated from an agent run
    /// with a session. Middleware may read it; tools receive it via
    /// [`Tool::invoke_in_context`](crate::tools::Tool::invoke_in_context)
    /// (the hook behind `Agent::as_tool` with `propagate_session`). Mirrors
    /// upstream's `FunctionInvocationContext.session`.
    pub session: Option<crate::session::AgentSession>,
    /// The live, mutable tool list of the current agent run, or `None` when
    /// the function is invoked outside a function-calling loop (e.g. via
    /// [`Tool::invoke`](crate::tools::Tool::invoke) directly). Middleware and
    /// tools may [`add_tools`](FunctionInvocationContext::add_tools) /
    /// [`remove_tools`](FunctionInvocationContext::remove_tools); mutations
    /// take effect on the **next** model iteration, not the in-flight batch.
    pub tools: Option<LiveToolList>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub result: Option<serde_json::Value>,
    pub terminate: bool,
}

impl FunctionInvocationContext {
    pub fn new(function_name: impl Into<String>, arguments: serde_json::Value) -> Self {
        Self {
            function_name: function_name.into(),
            arguments,
            session: None,
            tools: None,
            metadata: HashMap::new(),
            result: None,
            terminate: false,
        }
    }

    /// Builder: attach the agent session this invocation belongs to.
    pub fn with_session(mut self, session: Option<crate::session::AgentSession>) -> Self {
        self.session = session;
        self
    }

    /// Builder: attach the run's live tool list.
    pub fn with_tools(mut self, tools: Option<LiveToolList>) -> Self {
        self.tools = tools;
        self
    }

    /// Add tools to the current agent run (progressive tool exposure); see
    /// [`LiveToolList::add_tools`]. Errors when this invocation is not bound
    /// to a live agent run (mirrors upstream's `RuntimeError`).
    pub fn add_tools(&self, tools: impl IntoIterator<Item = ToolDefinition>) -> Result<()> {
        self.tools
            .as_ref()
            .ok_or_else(|| {
                Error::Configuration(
                    "cannot add tools: this FunctionInvocationContext is not bound to a \
                     live agent run"
                        .into(),
                )
            })?
            .add_tools(tools)
    }

    /// Remove tools from the current agent run by name; see
    /// [`LiveToolList::remove_tools`]. Errors when this invocation is not
    /// bound to a live agent run.
    pub fn remove_tools<'a>(&self, names: impl IntoIterator<Item = &'a str>) -> Result<()> {
        self.tools
            .as_ref()
            .ok_or_else(|| {
                Error::Configuration(
                    "cannot remove tools: this FunctionInvocationContext is not bound to a \
                     live agent run"
                        .into(),
                )
            })
            .map(|t| t.remove_tools(names))
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
