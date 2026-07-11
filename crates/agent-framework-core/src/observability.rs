//! Lightweight OpenTelemetry GenAI-style span instrumentation, built only on
//! the [`tracing`] crate.
//!
//! This is a dependency-light port of the Python `observability.py`
//! instrumentation. It emits `tracing` spans that follow the OpenTelemetry
//! GenAI semantic conventions, so that an OTel bridge (e.g.
//! `tracing-opentelemetry`) can export them without any additional glue:
//!
//! * span names: `chat {model}`, `invoke_agent {agent}`, `execute_tool {tool}`
//!   — the human-readable name is carried in the `otel.name` field (the static
//!   `tracing` metadata name is the bare operation, since `tracing` requires a
//!   literal span name).
//! * attributes: `gen_ai.operation.name`, `gen_ai.system`,
//!   `gen_ai.request.model`, `gen_ai.response.finish_reasons`,
//!   `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `error.type`,
//!   and — only when content capture is explicitly enabled —
//!   `gen_ai.input.messages` / `gen_ai.output.messages`.
//!
//! The main entry point is [`ObservableChatClient`], a [`ChatClient`] decorator.
//! Tool execution inside [`FunctionInvokingChatClient`] and the
//! [`ChatAgent`](crate::agent::ChatAgent) run paths are instrumented directly by
//! those types using the span constructors here.
//!
//! [`FunctionInvokingChatClient`]: crate::client::FunctionInvokingChatClient

use async_trait::async_trait;
use futures::StreamExt;
use tracing::field::Empty;
use tracing::{Instrument, Span};

use crate::client::{ChatClient, ChatStream};
use crate::error::{Error, Result};
use crate::types::{ChatMessage, ChatOptions, ChatResponse};

/// OpenTelemetry GenAI semantic-convention attribute keys.
pub mod attr {
    pub const OPERATION: &str = "gen_ai.operation.name";
    /// The provider/system tag, e.g. `"openai"`. Supplied by the client.
    pub const SYSTEM: &str = "gen_ai.system";
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const RESPONSE_ID: &str = "gen_ai.response.id";
    pub const FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    pub const INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    pub const ERROR_TYPE: &str = "error.type";
    pub const TOOL_NAME: &str = "gen_ai.tool.name";
    pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
    pub const AGENT_NAME: &str = "gen_ai.agent.name";
    pub const AGENT_ID: &str = "gen_ai.agent.id";
    pub const INPUT_MESSAGES: &str = "gen_ai.input.messages";
    pub const OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
    /// The human-readable span name override consumed by OTel bridges.
    pub const OTEL_NAME: &str = "otel.name";
}

/// OpenTelemetry GenAI operation names.
pub mod op {
    pub const CHAT: &str = "chat";
    pub const INVOKE_AGENT: &str = "invoke_agent";
    pub const EXECUTE_TOOL: &str = "execute_tool";
}

/// The `error.type` value for a framework [`Error`]: its variant discriminant.
pub fn error_type(err: &Error) -> String {
    // The Display of the thiserror variants is prefixed with a stable label,
    // e.g. "tool error: …"; take the label as a compact type tag.
    match err {
        Error::AgentInitialization(_) => "agent_initialization",
        Error::AgentExecution(_) => "agent_execution",
        Error::Serialization(_) => "serialization",
        Error::Content(_) => "content",
        Error::Tool(_) => "tool",
        Error::Service(_) => "service",
        Error::ServiceStatus { .. } => "service",
        Error::Workflow(_) => "workflow",
        Error::AdditionItemMismatch(_) => "addition_item_mismatch",
        Error::Configuration(_) => "configuration",
        Error::Json(_) => "json",
        Error::Other(_) => "other",
    }
    .to_string()
}

/// Build a `chat {model}` span for a chat-completion request.
pub fn chat_span(system: &str, model: &str) -> Span {
    let span = tracing::info_span!(
        "chat",
        otel.name = Empty,
        gen_ai.operation.name = op::CHAT,
        gen_ai.system = system,
        gen_ai.request.model = model,
        gen_ai.response.id = Empty,
        gen_ai.response.finish_reasons = Empty,
        gen_ai.usage.input_tokens = Empty,
        gen_ai.usage.output_tokens = Empty,
        error.type = Empty,
        gen_ai.input.messages = Empty,
        gen_ai.output.messages = Empty,
    );
    span.record(attr::OTEL_NAME, format!("{} {}", op::CHAT, model).as_str());
    span
}

/// Build an `invoke_agent {agent}` span for an agent run.
pub fn agent_span(agent_name: &str, agent_id: &str) -> Span {
    let span = tracing::info_span!(
        "invoke_agent",
        otel.name = Empty,
        gen_ai.operation.name = op::INVOKE_AGENT,
        gen_ai.agent.name = agent_name,
        gen_ai.agent.id = agent_id,
        gen_ai.usage.input_tokens = Empty,
        gen_ai.usage.output_tokens = Empty,
        error.type = Empty,
    );
    span.record(
        attr::OTEL_NAME,
        format!("{} {}", op::INVOKE_AGENT, agent_name).as_str(),
    );
    span
}

/// Build an `execute_tool {tool}` span for a single tool invocation.
pub fn tool_span(tool_name: &str, call_id: &str) -> Span {
    let span = tracing::info_span!(
        "execute_tool",
        otel.name = Empty,
        gen_ai.operation.name = op::EXECUTE_TOOL,
        gen_ai.tool.name = tool_name,
        gen_ai.tool.call.id = call_id,
        error.type = Empty,
    );
    span.record(
        attr::OTEL_NAME,
        format!("{} {}", op::EXECUTE_TOOL, tool_name).as_str(),
    );
    span
}

/// Record the response-side attributes (finish reason, usage, id) onto `span`.
pub fn record_response(span: &Span, response: &ChatResponse, capture_content: bool) {
    if let Some(reason) = &response.finish_reason {
        span.record(attr::FINISH_REASONS, reason.as_str());
    }
    if let Some(id) = &response.response_id {
        span.record(attr::RESPONSE_ID, id.as_str());
    }
    if let Some(usage) = &response.usage_details {
        if let Some(input) = usage.input_token_count {
            span.record(attr::INPUT_TOKENS, input);
        }
        if let Some(output) = usage.output_token_count {
            span.record(attr::OUTPUT_TOKENS, output);
        }
    }
    if capture_content {
        span.record(
            attr::OUTPUT_MESSAGES,
            messages_json(&response.messages).as_str(),
        );
    }
}

/// Serialize messages to a compact JSON string for content-capture attributes.
fn messages_json(messages: &[ChatMessage]) -> String {
    serde_json::to_string(messages).unwrap_or_default()
}

/// A [`ChatClient`] decorator that emits a `chat` span per request following the
/// OpenTelemetry GenAI semantic conventions.
///
/// Message-content capture (`gen_ai.input.messages` / `gen_ai.output.messages`)
/// is **off by default**; enable it with
/// [`ObservableChatClient::with_content_capture`], mirroring Python's
/// `enable_sensitive_data` flag.
pub struct ObservableChatClient<C: ChatClient> {
    inner: C,
    system: String,
    capture_content: bool,
}

impl<C: ChatClient> ObservableChatClient<C> {
    /// Wrap `inner`, tagging spans with the given provider/system name (the
    /// `gen_ai.system` attribute), e.g. `"openai"` or `"anthropic"`.
    pub fn new(inner: C, system: impl Into<String>) -> Self {
        Self {
            inner,
            system: system.into(),
            capture_content: false,
        }
    }

    /// Enable or disable capturing message content on spans (default: off).
    pub fn with_content_capture(mut self, capture: bool) -> Self {
        self.capture_content = capture;
        self
    }

    /// A reference to the wrapped client.
    pub fn inner(&self) -> &C {
        &self.inner
    }

    fn model_for(&self, options: &ChatOptions) -> String {
        options
            .model_id
            .clone()
            .or_else(|| self.inner.model_id().map(str::to_string))
            .unwrap_or_default()
    }
}

#[async_trait]
impl<C: ChatClient> ChatClient for ObservableChatClient<C> {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let span = chat_span(&self.system, &self.model_for(&options));
        if self.capture_content {
            span.record(attr::INPUT_MESSAGES, messages_json(&messages).as_str());
        }
        let capture = self.capture_content;
        async move {
            let result = self.inner.get_response(messages, options).await;
            let span = Span::current();
            match &result {
                Ok(response) => record_response(&span, response, capture),
                Err(err) => {
                    span.record(attr::ERROR_TYPE, error_type(err).as_str());
                }
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let span = chat_span(&self.system, &self.model_for(&options));
        if self.capture_content {
            span.record(attr::INPUT_MESSAGES, messages_json(&messages).as_str());
        }
        let capture = self.capture_content;
        // Instrument the initiation future with the span (never hold an
        // `enter()` guard across an await); attribute recording happens as
        // the stream drains and completes.
        let inner = self
            .inner
            .get_streaming_response(messages, options)
            .instrument(span.clone())
            .await;

        let inner = match inner {
            Ok(s) => s,
            Err(err) => {
                span.record(attr::ERROR_TYPE, error_type(&err).as_str());
                return Err(err);
            }
        };

        // Aggregate updates to recover finish reason / usage from the final
        // chunks, recording them onto the span when the stream ends.
        let state = (inner, ChatResponse::default(), Some(span), false, capture);
        let stream = futures::stream::unfold(
            state,
            |(mut inner, mut agg, mut span, done, capture)| async move {
                if done {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(update)) => {
                        agg.absorb_update(update.clone());
                        Some((Ok(update), (inner, agg, span, false, capture)))
                    }
                    Some(Err(err)) => {
                        if let Some(span) = &span {
                            span.record(attr::ERROR_TYPE, error_type(&err).as_str());
                        }
                        Some((Err(err), (inner, agg, span.take(), true, capture)))
                    }
                    None => {
                        if let Some(span) = span.take() {
                            agg.finalize();
                            record_response(&span, &agg, capture);
                        }
                        None
                    }
                }
            },
        );
        Ok(stream.boxed())
    }

    fn model_id(&self) -> Option<&str> {
        self.inner.model_id()
    }
}
