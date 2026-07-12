//! Context / memory providers.
//!
//! Rust equivalent of `agent_framework._memory`. A [`ContextProvider`] injects
//! extra instructions, messages, and tools into an agent invocation without
//! persisting them to the conversation history.

use async_trait::async_trait;
use std::sync::Arc;

use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::ChatMessage;

/// Additional context supplied by a provider for a single invocation.
#[derive(Debug, Clone, Default)]
pub struct Context {
    pub instructions: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<ToolDefinition>,
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }
}

/// A source of per-invocation context (memory, RAG, etc.).
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Called before the model is invoked; returns context to inject.
    async fn invoking(&self, messages: &[ChatMessage]) -> Result<Context>;

    /// Optional hook fired when a new thread is created.
    async fn thread_created(&self, _thread_id: Option<&str>) -> Result<()> {
        Ok(())
    }

    /// Optional hook fired after an invocation completes, on *both* the
    /// success and failure paths.
    ///
    /// Mirrors Python's `ContextProvider.invoked(..., invoke_exception=...)`
    /// (`_memory.py:119-138`): providers can observe the request and the
    /// outcome. On success, `response_messages` holds the produced messages and
    /// `error` is `None`. On failure, `error` carries the run failure and
    /// `response_messages` is empty. A provider that only cares about
    /// successful turns can ignore `error`.
    async fn invoked(
        &self,
        _request_messages: &[ChatMessage],
        _response_messages: &[ChatMessage],
        _error: Option<&Error>,
    ) -> Result<()> {
        Ok(())
    }
}

/// Fan-out/fan-in over multiple [`ContextProvider`]s, merging their output.
#[derive(Default, Clone)]
pub struct AggregateContextProvider {
    providers: Vec<Arc<dyn ContextProvider>>,
}

impl AggregateContextProvider {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_providers(providers: Vec<Arc<dyn ContextProvider>>) -> Self {
        Self { providers }
    }

    /// Add a provider.
    pub fn add(&mut self, provider: Arc<dyn ContextProvider>) {
        self.providers.push(provider);
    }

    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

#[async_trait]
impl ContextProvider for AggregateContextProvider {
    async fn invoking(&self, messages: &[ChatMessage]) -> Result<Context> {
        let mut merged = Context::new();
        for provider in &self.providers {
            let ctx = provider.invoking(messages).await?;
            merged.instructions = match (merged.instructions.take(), ctx.instructions) {
                (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
                (Some(a), None) => Some(a),
                (None, b) => b,
            };
            merged.messages.extend(ctx.messages);
            merged.tools.extend(ctx.tools);
        }
        Ok(merged)
    }

    async fn thread_created(&self, thread_id: Option<&str>) -> Result<()> {
        for provider in &self.providers {
            provider.thread_created(thread_id).await?;
        }
        Ok(())
    }

    async fn invoked(
        &self,
        request: &[ChatMessage],
        response: &[ChatMessage],
        error: Option<&Error>,
    ) -> Result<()> {
        for provider in &self.providers {
            provider.invoked(request, response, error).await?;
        }
        Ok(())
    }
}
