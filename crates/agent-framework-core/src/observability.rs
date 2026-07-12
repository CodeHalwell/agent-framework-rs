//! Lightweight OpenTelemetry GenAI-style span instrumentation, built on the
//! [`tracing`] crate, plus optional GenAI metrics behind the `otel-metrics`
//! feature.
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
//! * chat-span attributes: `gen_ai.operation.name`, `gen_ai.system` /
//!   `gen_ai.provider.name` (dual-emitted — see [`attr::PROVIDER_NAME`]),
//!   `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.response.id`,
//!   `gen_ai.response.finish_reasons`, `gen_ai.usage.{input,output}_tokens`,
//!   the request parameters (`gen_ai.request.{temperature,top_p,max_tokens,
//!   seed,frequency_penalty,presence_penalty,stop_sequences}`,
//!   `gen_ai.conversation.id`), `error.type` plus the `tracing-opentelemetry`
//!   "special fields" `otel.status_code` / `otel.status_message` — and, only
//!   when content capture is explicitly enabled, `gen_ai.input.messages` /
//!   `gen_ai.output.messages`, `gen_ai.system_instructions`, and
//!   `gen_ai.tool.definitions`.
//! * tool-span attributes: `gen_ai.tool.name`, `gen_ai.tool.call.id`,
//!   `gen_ai.tool.description`, `gen_ai.tool.type`, and (content-capture-gated)
//!   `gen_ai.tool.call.arguments` / `gen_ai.tool.call.result`.
//!
//! The main entry point is [`ObservableChatClient`], a [`ChatClient`] decorator.
//! Tool execution inside [`FunctionInvokingChatClient`] and the
//! [`Agent`](crate::agent::Agent) run paths are instrumented directly by
//! those types using the span constructors here.
//!
//! ## Metrics (`otel-metrics` feature)
//!
//! With the `otel-metrics` feature enabled, [`ObservableChatClient`] also
//! records two histograms through the `opentelemetry` **API** crate only
//! (mirroring `observability.py:788-803`, bucket boundaries at `:65-96`):
//! [`metrics::TOKEN_USAGE_METRIC`] (`gen_ai.client.token.usage`, unit
//! `"tokens"`) and [`metrics::OPERATION_DURATION_METRIC`]
//! (`gen_ai.client.operation.duration`, unit `"s"`). A third histogram,
//! [`metrics::FUNCTION_INVOCATION_DURATION_METRIC`]
//! (`agent_framework.function.invocation.duration`), is defined for tool-call
//! timing; see [`metrics::record_function_invocation_duration`] for why it
//! isn't wired to a call site yet. This crate never depends on an OTel SDK:
//! without an application-installed `MeterProvider` (via
//! [`opentelemetry::global::set_meter_provider`]) the instruments are no-ops,
//! so the feature is safe to enable unconditionally.
//!
//! ## Wiring to a real OTel backend
//!
//! Neither the `tracing` spans nor the `otel-metrics` histograms are exported
//! anywhere by this crate — that stays the application's job. A minimal
//! bridge, using `tracing-opentelemetry` for spans and `opentelemetry_sdk` for
//! metrics:
//!
//! ```ignore
//! use opentelemetry_sdk::trace::SdkTracerProvider;
//! use opentelemetry_sdk::metrics::SdkMeterProvider;
//! use tracing_subscriber::layer::SubscriberExt;
//!
//! let tracer_provider = SdkTracerProvider::builder()
//!     // .with_batch_exporter(otlp_span_exporter) / .with_simple_exporter(...) …
//!     .build();
//! let meter_provider = SdkMeterProvider::builder()
//!     // .with_reader(periodic_reader_wrapping_your_metric_exporter) …
//!     .build();
//! opentelemetry::global::set_meter_provider(meter_provider); // powers `otel-metrics`
//!
//! let tracer = tracer_provider.tracer("agent_framework");
//! let subscriber = tracing_subscriber::registry()
//!     .with(tracing_opentelemetry::layer().with_tracer(tracer));
//! tracing::subscriber::set_global_default(subscriber).unwrap();
//! ```
//!
//! Without any of this, spans are still emitted to whatever plain `tracing`
//! subscriber you do have (e.g. for structured logging), and `otel-metrics`
//! instruments silently drop their measurements — zero required setup either
//! way.
//!
//! ## Environment configuration
//!
//! [`ObservabilityConfig::from_env`] reads `ENABLE_SENSITIVE_DATA` (mirrors
//! Python's `enable_sensitive_data`, `observability.py:347-394`) for use with
//! [`ObservableChatClient::from_env`]. Unlike Python, there is no
//! `ENABLE_OTEL` equivalent to read: this crate's spans are plain `tracing`
//! spans, already effectively free without a subscriber attached, so there is
//! no separate "enable observability" switch.
//!
//! [`FunctionInvokingChatClient`]: crate::client::FunctionInvokingChatClient

use async_trait::async_trait;
use futures::StreamExt;
use tracing::field::Empty;
use tracing::{Instrument, Span};

use crate::client::{ChatClient, ChatStream};
use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::{ChatOptions, ChatResponse, Message};

/// OpenTelemetry GenAI semantic-convention attribute keys.
pub mod attr {
    pub const OPERATION: &str = "gen_ai.operation.name";
    /// The provider/system tag, e.g. `"openai"`. Supplied by the client.
    pub const SYSTEM: &str = "gen_ai.system";
    /// Current semantic-convention replacement for [`SYSTEM`]; both are set
    /// from the same value so older and newer consumers each find what they
    /// expect on the span.
    pub const PROVIDER_NAME: &str = "gen_ai.provider.name";
    pub const REQUEST_MODEL: &str = "gen_ai.request.model";
    pub const RESPONSE_MODEL: &str = "gen_ai.response.model";
    pub const RESPONSE_ID: &str = "gen_ai.response.id";
    pub const FINISH_REASONS: &str = "gen_ai.response.finish_reasons";
    pub const INPUT_TOKENS: &str = "gen_ai.usage.input_tokens";
    pub const OUTPUT_TOKENS: &str = "gen_ai.usage.output_tokens";
    /// Input tokens written to a provider-managed cache.
    pub const CACHE_CREATION_INPUT_TOKENS: &str = "gen_ai.usage.cache_creation.input_tokens";
    /// Input tokens served from a provider-managed cache.
    pub const CACHE_READ_INPUT_TOKENS: &str = "gen_ai.usage.cache_read.input_tokens";
    /// Output tokens spent on reasoning.
    pub const REASONING_OUTPUT_TOKENS: &str = "gen_ai.usage.reasoning.output_tokens";
    /// Low-cardinality prompt name (e.g. for a named/templated prompt).
    pub const PROMPT_NAME: &str = "gen_ai.prompt.name";
    pub const REQUEST_TEMPERATURE: &str = "gen_ai.request.temperature";
    pub const REQUEST_TOP_P: &str = "gen_ai.request.top_p";
    pub const REQUEST_MAX_TOKENS: &str = "gen_ai.request.max_tokens";
    pub const REQUEST_SEED: &str = "gen_ai.request.seed";
    pub const REQUEST_FREQUENCY_PENALTY: &str = "gen_ai.request.frequency_penalty";
    pub const REQUEST_PRESENCE_PENALTY: &str = "gen_ai.request.presence_penalty";
    pub const REQUEST_STOP_SEQUENCES: &str = "gen_ai.request.stop_sequences";
    pub const CONVERSATION_ID: &str = "gen_ai.conversation.id";
    /// Content-capture-gated: the system/instructions prompt, JSON-encoded as
    /// `[{"type":"text","content":...}]` (mirrors
    /// `observability.py:1444-1448`).
    pub const SYSTEM_INSTRUCTIONS: &str = "gen_ai.system_instructions";
    /// Content-capture-gated: the request's tool list, JSON-encoded (mirrors
    /// `_tools_to_dict`, `observability.py:1388-1391`).
    pub const TOOL_DEFINITIONS: &str = "gen_ai.tool.definitions";
    pub const ERROR_TYPE: &str = "error.type";
    pub const TOOL_NAME: &str = "gen_ai.tool.name";
    pub const TOOL_CALL_ID: &str = "gen_ai.tool.call.id";
    pub const TOOL_DESCRIPTION: &str = "gen_ai.tool.description";
    pub const TOOL_TYPE: &str = "gen_ai.tool.type";
    /// Content-capture-gated tool-call arguments (JSON-encoded).
    pub const TOOL_CALL_ARGUMENTS: &str = "gen_ai.tool.call.arguments";
    /// Content-capture-gated tool-call result (JSON-encoded).
    pub const TOOL_CALL_RESULT: &str = "gen_ai.tool.call.result";
    pub const AGENT_NAME: &str = "gen_ai.agent.name";
    pub const AGENT_ID: &str = "gen_ai.agent.id";
    pub const INPUT_MESSAGES: &str = "gen_ai.input.messages";
    pub const OUTPUT_MESSAGES: &str = "gen_ai.output.messages";
    /// The human-readable span name override consumed by OTel bridges.
    pub const OTEL_NAME: &str = "otel.name";
    /// `tracing-opentelemetry` special field: sets the OTel span status code
    /// (`"ERROR"` or `"OK"`) when a bridge is attached.
    pub const OTEL_STATUS_CODE: &str = "otel.status_code";
    /// `tracing-opentelemetry` special field: sets the OTel span status
    /// description. Used here to carry the exception message, mirroring
    /// Python's `capture_exception` (`observability.py:1407-1411`, which
    /// calls `set_status(description=repr(exception))`).
    pub const OTEL_STATUS_MESSAGE: &str = "otel.status_message";
}

/// OpenTelemetry GenAI operation names.
pub mod op {
    pub const CHAT: &str = "chat";
    pub const INVOKE_AGENT: &str = "invoke_agent";
    pub const EXECUTE_TOOL: &str = "execute_tool";
    pub const EMBEDDINGS: &str = "embeddings";
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
        Error::ServiceInvalidAuth { .. } => "service_invalid_auth",
        Error::ServiceInvalidRequest { .. } => "service_invalid_request",
        Error::ServiceContentFilter { .. } => "service_content_filter",
        Error::Workflow(_) => "workflow",
        Error::AdditionItemMismatch(_) => "addition_item_mismatch",
        Error::Configuration(_) => "configuration",
        Error::Json(_) => "json",
        Error::Other(_) => "other",
    }
    .to_string()
}

/// Build a `chat {model}` span for a chat-completion request.
///
/// Only the fields that are cheap/always-known at call time are set here
/// (`gen_ai.system` / `gen_ai.provider.name`, `gen_ai.request.model`); the
/// rest of the request/response/error attribute set is filled in afterward
/// via [`record_request`], [`record_response`], and [`record_error`] — mirrors
/// `_get_span_attributes` (`observability.py:1345-1404`).
pub fn chat_span(system: &str, model: &str) -> Span {
    let span = tracing::info_span!(
        "chat",
        otel.name = Empty,
        gen_ai.operation.name = op::CHAT,
        gen_ai.system = system,
        gen_ai.provider.name = system,
        gen_ai.request.model = model,
        gen_ai.response.model = Empty,
        gen_ai.response.id = Empty,
        gen_ai.response.finish_reasons = Empty,
        gen_ai.usage.input_tokens = Empty,
        gen_ai.usage.output_tokens = Empty,
        gen_ai.usage.cache_creation.input_tokens = Empty,
        gen_ai.usage.cache_read.input_tokens = Empty,
        gen_ai.usage.reasoning.output_tokens = Empty,
        gen_ai.request.temperature = Empty,
        gen_ai.request.top_p = Empty,
        gen_ai.request.max_tokens = Empty,
        gen_ai.request.seed = Empty,
        gen_ai.request.frequency_penalty = Empty,
        gen_ai.request.presence_penalty = Empty,
        gen_ai.request.stop_sequences = Empty,
        gen_ai.conversation.id = Empty,
        gen_ai.system_instructions = Empty,
        gen_ai.tool.definitions = Empty,
        error.type = Empty,
        otel.status_code = Empty,
        otel.status_message = Empty,
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
        // Declared for forward-compatibility with `record_error`; the
        // current `agent.rs` error path only records `error.type` directly,
        // so these stay unset until that call site migrates.
        otel.status_code = Empty,
        otel.status_message = Empty,
    );
    span.record(
        attr::OTEL_NAME,
        format!("{} {}", op::INVOKE_AGENT, agent_name).as_str(),
    );
    span
}

/// Build an `execute_tool {tool}` span with the full OTel GenAI tool
/// attribute set: name, call id, description, a fixed `"function"` tool type
/// (the only kind executed through this in-process loop — mirrors Python's
/// `get_function_span_attributes`, `observability.py:1284-1302`), and
/// placeholders for the content-capture-gated call arguments/result (fill
/// with [`record_tool_arguments`] / [`record_tool_result`]) and the
/// error/status fields (fill with [`record_error`]).
///
/// [`tool_span`] is the source-compatible two-argument form used today by
/// `client.rs`'s `FunctionInvokingChatClient`; it delegates here with
/// `description = None`. New call sites — and that eventual migration —
/// should call this directly to get a real tool description and light up the
/// content-capture-gated attributes.
pub fn tool_span_ex(tool_name: &str, call_id: &str, description: Option<&str>) -> Span {
    let span = tracing::info_span!(
        "execute_tool",
        otel.name = Empty,
        gen_ai.operation.name = op::EXECUTE_TOOL,
        gen_ai.tool.name = tool_name,
        gen_ai.tool.call.id = call_id,
        gen_ai.tool.description = Empty,
        gen_ai.tool.type = "function",
        gen_ai.tool.call.arguments = Empty,
        gen_ai.tool.call.result = Empty,
        error.type = Empty,
        otel.status_code = Empty,
        otel.status_message = Empty,
    );
    if let Some(description) = description {
        if !description.is_empty() {
            span.record(attr::TOOL_DESCRIPTION, description);
        }
    }
    span.record(
        attr::OTEL_NAME,
        format!("{} {}", op::EXECUTE_TOOL, tool_name).as_str(),
    );
    span
}

/// Build an `execute_tool {tool}` span (source-compatible two-argument form).
///
/// Delegates to [`tool_span_ex`] with no description. Prefer `tool_span_ex`
/// for new call sites.
pub fn tool_span(tool_name: &str, call_id: &str) -> Span {
    tool_span_ex(tool_name, call_id, None)
}

/// Record tool-call arguments onto a tool span, gated by content capture
/// (mirrors the `SENSITIVE_DATA_ENABLED`-gated `gen_ai.tool.call.arguments`
/// capture in Python's `AIFunction.invoke`, `_tools.py:751-759`).
pub fn record_tool_arguments(span: &Span, arguments: &serde_json::Value, capture_content: bool) {
    if !capture_content {
        return;
    }
    span.record(attr::TOOL_CALL_ARGUMENTS, arguments.to_string().as_str());
}

/// Record a tool-call result onto a tool span, gated by content capture
/// (mirrors the `gen_ai.tool.call.result` capture in Python's
/// `AIFunction.invoke`, `_tools.py:779-787`).
pub fn record_tool_result(span: &Span, result: &serde_json::Value, capture_content: bool) {
    if !capture_content {
        return;
    }
    span.record(attr::TOOL_CALL_RESULT, result.to_string().as_str());
}

/// Record an error onto a span following the existing `error.type` pattern:
/// `error.type` (the framework [`Error`] variant tag), plus the
/// `tracing-opentelemetry` "special fields" `otel.status_code` (`"ERROR"`)
/// and `otel.status_message` (the exception's `Display` text) so that a
/// bridge sets real OTel span status. This mirrors Python's
/// `capture_exception` (`record_exception` + `set_status`,
/// `observability.py:1407-1411`) as far as bare `tracing` fields allow —
/// there is no span-events API here without also taking on an SDK
/// dependency.
pub fn record_error(span: &Span, err: &Error) {
    span.record(attr::ERROR_TYPE, error_type(err).as_str());
    span.record(attr::OTEL_STATUS_CODE, "ERROR");
    span.record(attr::OTEL_STATUS_MESSAGE, err.to_string().as_str());
}

/// Record the response-side attributes (finish reason, usage, id, model) onto
/// `span`, mirroring `_get_response_attributes` (`observability.py:1488-1512`).
pub fn record_response(span: &Span, response: &ChatResponse, capture_content: bool) {
    if let Some(reason) = &response.finish_reason {
        span.record(attr::FINISH_REASONS, reason.as_str());
    }
    if let Some(id) = &response.response_id {
        span.record(attr::RESPONSE_ID, id.as_str());
    }
    if let Some(model) = &response.model_id {
        span.record(attr::RESPONSE_MODEL, model.as_str());
    }
    if let Some(usage) = &response.usage_details {
        if let Some(input) = usage.input_token_count {
            span.record(attr::INPUT_TOKENS, input);
        }
        if let Some(output) = usage.output_token_count {
            span.record(attr::OUTPUT_TOKENS, output);
        }
        if let Some(v) = usage.cache_creation_input_token_count {
            span.record(attr::CACHE_CREATION_INPUT_TOKENS, v);
        }
        if let Some(v) = usage.cache_read_input_token_count {
            span.record(attr::CACHE_READ_INPUT_TOKENS, v);
        }
        if let Some(v) = usage.reasoning_output_token_count {
            span.record(attr::REASONING_OUTPUT_TOKENS, v);
        }
    }
    if capture_content {
        span.record(
            attr::OUTPUT_MESSAGES,
            messages_json(&response.messages).as_str(),
        );
    }
}

/// Record request-side attributes onto a `chat` span from [`ChatOptions`],
/// mirroring `_get_span_attributes` (`observability.py:1345-1404`). System
/// instructions and the serialized tool list are additionally gated by
/// `capture_content` (mirrors Python's `SENSITIVE_DATA_ENABLED` gate).
pub fn record_request(span: &Span, options: &ChatOptions, capture_content: bool) {
    if let Some(v) = options.temperature {
        span.record(attr::REQUEST_TEMPERATURE, f64::from(v));
    }
    if let Some(v) = options.top_p {
        span.record(attr::REQUEST_TOP_P, f64::from(v));
    }
    if let Some(v) = options.max_tokens {
        span.record(attr::REQUEST_MAX_TOKENS, u64::from(v));
    }
    if let Some(v) = options.seed {
        span.record(attr::REQUEST_SEED, v);
    }
    if let Some(v) = options.frequency_penalty {
        span.record(attr::REQUEST_FREQUENCY_PENALTY, f64::from(v));
    }
    if let Some(v) = options.presence_penalty {
        span.record(attr::REQUEST_PRESENCE_PENALTY, f64::from(v));
    }
    if let Some(stop) = &options.stop {
        if !stop.is_empty() {
            span.record(
                attr::REQUEST_STOP_SEQUENCES,
                serde_json::to_string(stop).unwrap_or_default().as_str(),
            );
        }
    }
    if let Some(id) = &options.conversation_id {
        span.record(attr::CONVERSATION_ID, id.as_str());
    }
    if capture_content {
        if let Some(instructions) = &options.instructions {
            if !instructions.is_empty() {
                span.record(
                    attr::SYSTEM_INSTRUCTIONS,
                    system_instructions_json(instructions).as_str(),
                );
            }
        }
        if !options.tools.is_empty() {
            span.record(
                attr::TOOL_DEFINITIONS,
                tool_definitions_json(&options.tools).as_str(),
            );
        }
    }
}

/// JSON-encode a system/instructions prompt as
/// `[{"type":"text","content":...}]`, mirroring
/// `observability.py:1444-1448`.
fn system_instructions_json(instructions: &str) -> String {
    serde_json::json!([{ "type": "text", "content": instructions }]).to_string()
}

/// JSON-encode a tool list for `gen_ai.tool.definitions`, mirroring the
/// shape produced by `_tools_to_dict` (`_tools.py:827-857`) closely enough to
/// be useful without depending on any provider-specific wire format.
fn tool_definitions_json(tools: &[ToolDefinition]) -> String {
    let list: Vec<serde_json::Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect();
    serde_json::to_string(&list).unwrap_or_default()
}

/// Serialize messages to a compact JSON string for content-capture attributes.
fn messages_json(messages: &[Message]) -> String {
    serde_json::to_string(messages).unwrap_or_default()
}

/// Per-stream bookkeeping threaded through [`ObservableChatClient`]'s
/// `get_streaming_response` `unfold` state, so the finalization arm can
/// record response attributes and, with `otel-metrics`, the completion
/// histograms. `system` / `request_model` / `start` are only needed for the
/// latter, so they (and the work to populate them) are compiled out
/// entirely when the feature is off.
struct StreamTelemetry {
    capture: bool,
    #[cfg(feature = "otel-metrics")]
    system: String,
    #[cfg(feature = "otel-metrics")]
    request_model: String,
    #[cfg(feature = "otel-metrics")]
    start: std::time::Instant,
}

/// A [`ChatClient`] decorator that emits a `chat` span per request following the
/// OpenTelemetry GenAI semantic conventions.
///
/// Message-content capture (`gen_ai.input.messages` / `gen_ai.output.messages`,
/// plus `gen_ai.system_instructions` / `gen_ai.tool.definitions`) is **off by
/// default**; enable it with [`ObservableChatClient::with_content_capture`],
/// mirroring Python's `enable_sensitive_data` flag — or construct via
/// [`ObservableChatClient::from_env`] to read that flag from
/// `ENABLE_SENSITIVE_DATA`.
pub struct ObservableChatClient<C: ChatClient> {
    inner: C,
    system: String,
    capture_content: bool,
}

impl<C: ChatClient> ObservableChatClient<C> {
    /// Wrap `inner`, tagging spans with the given provider/system name (the
    /// `gen_ai.system` / `gen_ai.provider.name` attributes), e.g. `"openai"`
    /// or `"anthropic"`.
    pub fn new(inner: C, system: impl Into<String>) -> Self {
        Self {
            inner,
            system: system.into(),
            capture_content: false,
        }
    }

    /// Wrap `inner` using [`ObservabilityConfig::from_env`] to decide whether
    /// content capture is enabled (`ENABLE_SENSITIVE_DATA`). Equivalent to:
    ///
    /// ```ignore
    /// ObservableChatClient::new(inner, system)
    ///     .with_content_capture(ObservabilityConfig::from_env().enable_sensitive_data)
    /// ```
    pub fn from_env(inner: C, system: impl Into<String>) -> Self {
        let config = ObservabilityConfig::from_env();
        Self::new(inner, system).with_content_capture(config.enable_sensitive_data)
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
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let request_model = self.model_for(&options);
        let span = chat_span(&self.system, &request_model);
        record_request(&span, &options, self.capture_content);
        if self.capture_content {
            span.record(attr::INPUT_MESSAGES, messages_json(&messages).as_str());
        }
        let capture = self.capture_content;
        #[cfg(feature = "otel-metrics")]
        let start = std::time::Instant::now();
        async move {
            let result = self.inner.get_response(messages, options).await;
            let span = Span::current();
            match &result {
                Ok(response) => {
                    record_response(&span, response, capture);
                    #[cfg(feature = "otel-metrics")]
                    {
                        let (input_tokens, output_tokens) = response
                            .usage_details
                            .as_ref()
                            .map(|u| (u.input_token_count, u.output_token_count))
                            .unwrap_or((None, None));
                        metrics::record_chat_completion(
                            &self.system,
                            &request_model,
                            response.model_id.as_deref(),
                            input_tokens,
                            output_tokens,
                            start.elapsed(),
                        );
                    }
                }
                Err(err) => {
                    record_error(&span, err);
                }
            }
            result
        }
        .instrument(span)
        .await
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let request_model = self.model_for(&options);
        let span = chat_span(&self.system, &request_model);
        record_request(&span, &options, self.capture_content);
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
                record_error(&span, &err);
                return Err(err);
            }
        };

        // Aggregate updates to recover finish reason / usage from the final
        // chunks, recording them onto the span (and, with `otel-metrics`, the
        // completion histograms) when the stream ends.
        let telemetry = StreamTelemetry {
            capture,
            #[cfg(feature = "otel-metrics")]
            system: self.system.clone(),
            #[cfg(feature = "otel-metrics")]
            request_model: request_model.clone(),
            #[cfg(feature = "otel-metrics")]
            start: std::time::Instant::now(),
        };
        let state = (inner, ChatResponse::default(), Some(span), false, telemetry);
        let stream = futures::stream::unfold(
            state,
            |(mut inner, mut agg, mut span, done, telemetry)| async move {
                if done {
                    return None;
                }
                match inner.next().await {
                    Some(Ok(update)) => {
                        agg.absorb_update(update.clone());
                        Some((Ok(update), (inner, agg, span, false, telemetry)))
                    }
                    Some(Err(err)) => {
                        if let Some(span) = &span {
                            record_error(span, &err);
                        }
                        Some((Err(err), (inner, agg, span.take(), true, telemetry)))
                    }
                    None => {
                        if let Some(span) = span.take() {
                            agg.finalize();
                            record_response(&span, &agg, telemetry.capture);
                            #[cfg(feature = "otel-metrics")]
                            {
                                let (input_tokens, output_tokens) = agg
                                    .usage_details
                                    .as_ref()
                                    .map(|u| (u.input_token_count, u.output_token_count))
                                    .unwrap_or((None, None));
                                metrics::record_chat_completion(
                                    &telemetry.system,
                                    &telemetry.request_model,
                                    agg.model_id.as_deref(),
                                    input_tokens,
                                    output_tokens,
                                    telemetry.start.elapsed(),
                                );
                            }
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

/// Observability configuration read from the process environment, mirroring
/// (a subset of) Python's `ObservabilitySettings` (`observability.py:347-394`).
///
/// This port intentionally does **not** read `ENABLE_OTEL`: Python uses that
/// flag to skip constructing spans entirely when no heavier OTel SDK is
/// configured. This crate's spans are plain [`tracing`] spans, which are
/// already effectively free when no subscriber is attached, so there is no
/// separate "enable" switch — attach a subscriber (optionally bridged to
/// OTel; see the [module docs](self)) to turn observability on.
///
/// This crate does **not** build an OTel `Resource`, exporter, or
/// `MeterProvider` — see the [module docs](self) for wiring those yourself.
#[derive(Debug, Clone, Default)]
pub struct ObservabilityConfig {
    /// Whether message / system-instructions / tool-definition / tool-call
    /// content capture is enabled. Reads `ENABLE_SENSITIVE_DATA` (mirrors
    /// Python's `enable_sensitive_data` — "Warning: Sensitive events should
    /// only be enabled on test and development environments.").
    pub enable_sensitive_data: bool,
}

impl ObservabilityConfig {
    /// Read configuration from the process environment. Unset or
    /// unrecognized values default to `false` (matching Python's default).
    /// Recognized truthy values (case-insensitive, surrounding whitespace
    /// ignored): `"1"`, `"true"`, `"yes"`, `"on"`.
    pub fn from_env() -> Self {
        Self {
            enable_sensitive_data: env_flag("ENABLE_SENSITIVE_DATA"),
        }
    }

    /// The `OTEL_SERVICE_NAME` passthrough, defaulting to `"agent_framework"`
    /// (mirrors Python's `_create_resource`, `observability.py:336-344`).
    /// This crate does not build a `Resource` from it — it's exposed so an
    /// app wiring its own OTel `Resource` (see the [module docs](self)) can
    /// stay consistent with whatever this reads. Reads the environment fresh
    /// on every call rather than caching it at [`from_env`](Self::from_env)
    /// time.
    pub fn otel_service_name(&self) -> String {
        std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "agent_framework".to_string())
    }
}

/// Parse a boolean-ish environment variable, matching Python's
/// pydantic-settings-style truthy strings. Missing or unrecognized values are
/// `false`.
fn env_flag(key: &str) -> bool {
    std::env::var(key).is_ok_and(|v| {
        matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    })
}

/// Metrics instruments behind the `otel-metrics` feature (see the crate's
/// `[features]` table). Uses the `opentelemetry` **API** crate only
/// (`"metrics"` feature, no default features): instrument creation and
/// `.record()` calls are inert until an application installs a real
/// `MeterProvider` via [`opentelemetry::global::set_meter_provider`] — the
/// API's own default global provider is a no-op — so enabling this feature
/// never *requires* an application to also wire up an OTel SDK. See the
/// [module docs](super) for how to do that when you do want real metrics out
/// the other end.
///
/// Mirrors upstream's two chat-client histograms (`observability.py:788-803`,
/// bucket boundaries at `:65-96`) and the function-invocation-duration
/// histogram (`_tools.py`'s `_default_histogram`).
#[cfg(feature = "otel-metrics")]
pub mod metrics {
    use std::sync::OnceLock;
    use std::time::Duration;

    use opentelemetry::metrics::Histogram;
    use opentelemetry::KeyValue;

    use super::{attr, op};

    /// Bucket boundaries for `gen_ai.client.token.usage`, matching upstream's
    /// `TOKEN_USAGE_BUCKET_BOUNDARIES` (`observability.py:65-80`).
    pub const TOKEN_USAGE_BUCKET_BOUNDARIES: &[f64] = &[
        1.0,
        4.0,
        16.0,
        64.0,
        256.0,
        1024.0,
        4096.0,
        16384.0,
        65536.0,
        262_144.0,
        1_048_576.0,
        4_194_304.0,
        16_777_216.0,
        67_108_864.0,
    ];

    /// Bucket boundaries for `gen_ai.client.operation.duration` and
    /// `agent_framework.function.invocation.duration`, matching upstream's
    /// `OPERATION_DURATION_BUCKET_BOUNDARIES` (`observability.py:81-96`).
    pub const OPERATION_DURATION_BUCKET_BOUNDARIES: &[f64] = &[
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92,
    ];

    /// `Meters.LLM_TOKEN_USAGE`: the token-usage histogram name.
    pub const TOKEN_USAGE_METRIC: &str = "gen_ai.client.token.usage";
    /// `Meters.LLM_OPERATION_DURATION`: the operation-duration histogram name.
    pub const OPERATION_DURATION_METRIC: &str = "gen_ai.client.operation.duration";
    /// `OtelAttr.MEASUREMENT_FUNCTION_INVOCATION_DURATION`: the
    /// function-invocation-duration histogram name.
    pub const FUNCTION_INVOCATION_DURATION_METRIC: &str =
        "agent_framework.function.invocation.duration";

    const TOKEN_TYPE: &str = "gen_ai.token.type";
    const TOKEN_TYPE_INPUT: &str = "input";
    const TOKEN_TYPE_OUTPUT: &str = "output";
    const FUNCTION_NAME: &str = "agent_framework.function.name";

    struct Metrics {
        token_usage: Histogram<u64>,
        operation_duration: Histogram<f64>,
        function_invocation_duration: Histogram<f64>,
    }

    static METRICS: OnceLock<Metrics> = OnceLock::new();

    /// The lazily-initialized instrument set, bound to whichever
    /// [`opentelemetry::global`] meter provider is installed the first time
    /// this is called in this process — matches `global::meter`'s own
    /// "bound at call time" semantics, so install your provider before
    /// running any instrumented code.
    fn instruments() -> &'static Metrics {
        METRICS.get_or_init(|| {
            let meter = opentelemetry::global::meter("agent_framework");
            Metrics {
                token_usage: meter
                    .u64_histogram(TOKEN_USAGE_METRIC)
                    .with_unit("tokens")
                    .with_description("Captures the token usage of chat clients")
                    .with_boundaries(TOKEN_USAGE_BUCKET_BOUNDARIES.to_vec())
                    .build(),
                operation_duration: meter
                    .f64_histogram(OPERATION_DURATION_METRIC)
                    .with_unit("s")
                    .with_description("Captures the duration of chat client operations")
                    .with_boundaries(OPERATION_DURATION_BUCKET_BOUNDARIES.to_vec())
                    .build(),
                function_invocation_duration: meter
                    .f64_histogram(FUNCTION_INVOCATION_DURATION_METRIC)
                    .with_unit("s")
                    .with_description("Measures the duration of a function's execution")
                    .with_boundaries(OPERATION_DURATION_BUCKET_BOUNDARIES.to_vec())
                    .build(),
            }
        })
    }

    /// Record one completed chat call's histograms, mirroring
    /// `_capture_response` (`observability.py:1525-1543`): the token-usage
    /// histogram records once per token type present on the response; the
    /// operation-duration histogram always records. Only called from the
    /// success path of [`super::ObservableChatClient`] — like upstream, a
    /// failed call records neither histogram.
    pub(super) fn record_chat_completion(
        provider: &str,
        request_model: &str,
        response_model: Option<&str>,
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
        duration: Duration,
    ) {
        let m = instruments();
        let mut base = vec![
            KeyValue::new(attr::OPERATION, op::CHAT),
            KeyValue::new(attr::PROVIDER_NAME, provider.to_string()),
            KeyValue::new(attr::REQUEST_MODEL, request_model.to_string()),
        ];
        if let Some(model) = response_model {
            base.push(KeyValue::new(attr::RESPONSE_MODEL, model.to_string()));
        }
        if let Some(input) = input_tokens {
            let mut attrs = base.clone();
            attrs.push(KeyValue::new(TOKEN_TYPE, TOKEN_TYPE_INPUT));
            m.token_usage.record(input, &attrs);
        }
        if let Some(output) = output_tokens {
            let mut attrs = base.clone();
            attrs.push(KeyValue::new(TOKEN_TYPE, TOKEN_TYPE_OUTPUT));
            m.token_usage.record(output, &attrs);
        }
        m.operation_duration.record(duration.as_secs_f64(), &base);
    }

    /// Record the function-invocation-duration histogram for one tool call.
    ///
    /// Not yet called anywhere in this crate: the timing measurement belongs
    /// around `exec.invoke(...)` in `client.rs`'s
    /// `FunctionInvokingChatClient::execute_tool_call`, which is out of scope
    /// here (see the observability task's final report for the exact
    /// follow-up). The instrument and its recording logic are complete and
    /// tested on their own so that call site only needs to wrap its
    /// invocation with a timer and call this — plus, ideally, switch
    /// `tool_span` to [`super::tool_span_ex`] and add
    /// [`super::record_tool_arguments`] / [`super::record_tool_result`] calls
    /// at the same time.
    pub fn record_function_invocation_duration(
        tool_name: &str,
        duration: Duration,
        error_type: Option<&str>,
    ) {
        let m = instruments();
        let mut attrs = vec![KeyValue::new(FUNCTION_NAME, tool_name.to_string())];
        if let Some(err) = error_type {
            attrs.push(KeyValue::new(attr::ERROR_TYPE, err.to_string()));
        }
        m.function_invocation_duration
            .record(duration.as_secs_f64(), &attrs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Attribute string values (cross-language wire contract) ----------

    #[test]
    fn cache_reasoning_and_embedding_attrs_match_upstream() {
        // These strings are the OTel GenAI attribute keys upstream emits;
        // they must match exactly for cross-tool/cross-language consistency.
        assert_eq!(
            attr::CACHE_CREATION_INPUT_TOKENS,
            "gen_ai.usage.cache_creation.input_tokens"
        );
        assert_eq!(
            attr::CACHE_READ_INPUT_TOKENS,
            "gen_ai.usage.cache_read.input_tokens"
        );
        assert_eq!(
            attr::REASONING_OUTPUT_TOKENS,
            "gen_ai.usage.reasoning.output_tokens"
        );
        assert_eq!(attr::PROMPT_NAME, "gen_ai.prompt.name");
        assert_eq!(op::EMBEDDINGS, "embeddings");
    }

    // -- ObservabilityConfig::from_env -----------------------------------

    /// Guards env var mutation: tests run on multiple threads within a
    /// crate, and env vars are process-global (same pattern as
    /// `agent-framework-copilotstudio`'s / `agent-framework-mem0`'s
    /// `ENV_MUTEX`).
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_env() {
        // SAFETY: serialized by ENV_MUTEX; no other test in this crate
        // touches these variables (confirmed via a repo-wide grep before
        // adding this).
        unsafe {
            std::env::remove_var("ENABLE_SENSITIVE_DATA");
            std::env::remove_var("OTEL_SERVICE_NAME");
        }
    }

    #[test]
    fn from_env_defaults_to_disabled_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        let config = ObservabilityConfig::from_env();
        assert!(!config.enable_sensitive_data);
        assert_eq!(config.otel_service_name(), "agent_framework");
    }

    #[test]
    fn from_env_reads_truthy_enable_sensitive_data() {
        let _guard = ENV_MUTEX.lock().unwrap();
        for value in ["1", "true", "TRUE", "  yes  ", "on"] {
            clear_env();
            // SAFETY: serialized by ENV_MUTEX.
            unsafe { std::env::set_var("ENABLE_SENSITIVE_DATA", value) };
            let config = ObservabilityConfig::from_env();
            assert!(
                config.enable_sensitive_data,
                "expected {value:?} to be truthy"
            );
        }
        clear_env();
    }

    #[test]
    fn from_env_rejects_unrecognized_values() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        // SAFETY: serialized by ENV_MUTEX.
        unsafe { std::env::set_var("ENABLE_SENSITIVE_DATA", "nope") };
        let config = ObservabilityConfig::from_env();
        clear_env();
        assert!(!config.enable_sensitive_data);
    }

    #[test]
    fn otel_service_name_reads_env_override() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        // SAFETY: serialized by ENV_MUTEX.
        unsafe { std::env::set_var("OTEL_SERVICE_NAME", "my-service") };
        let config = ObservabilityConfig::from_env();
        let name = config.otel_service_name();
        clear_env();
        assert_eq!(name, "my-service");
    }

    // -- error_type --------------------------------------------------------

    #[test]
    fn error_type_gives_the_granular_service_errors_distinct_tags() {
        // The three newer variants get their own `error.type` tags, distinct
        // from both each other and the generic "service"/"service" tags that
        // `Error::Service`/`Error::ServiceStatus` share — that granularity is
        // the whole point of adding them.
        assert_eq!(
            error_type(&Error::service_invalid_auth("x")),
            "service_invalid_auth"
        );
        assert_eq!(
            error_type(&Error::service_invalid_request("x")),
            "service_invalid_request"
        );
        assert_eq!(
            error_type(&Error::service_content_filter("x")),
            "service_content_filter"
        );
        assert_eq!(error_type(&Error::service("x")), "service");
        assert_eq!(
            error_type(&Error::service_status(500, "x", None)),
            "service"
        );
    }
}
