//! # agent-framework-anthropic
//!
//! An Anthropic (Claude) [`ChatClient`] for `agent-framework-rs`.
//!
//! Talks directly to the Anthropic Messages API (`POST /v1/messages`), the
//! same way `agent-framework-openai` talks to Chat Completions: hand-rolled
//! request/response JSON conversion plus a hand-rolled SSE parser, with no
//! dependency on Anthropic's own SDK.
//!
//! ```no_run
//! use agent_framework_anthropic::AnthropicClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = AnthropicClient::new("sk-ant-...", "claude-sonnet-4-5-20250929");
//! let agent = ChatAgent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

mod convert;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, Content, FunctionArguments,
    FunctionCallContent, Role, TextContent, TextReasoningContent, UsageContent,
};
use futures::StreamExt;
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Parse Anthropic's `retry-after` header into a delay in seconds.
///
/// Anthropic returns `retry-after` (in integer seconds) alongside `429` and
/// overloaded `529` responses; we honor that hint on [`Error::ServiceStatus`]
/// so a retry layer can wait exactly as long as the server asks. A date-form
/// or unparseable value is treated as absent.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}
/// `max_tokens` is required by the Anthropic Messages API; this is used
/// whenever neither `ChatOptions::max_tokens` nor a client-level override is
/// set.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// An Anthropic (Claude) Messages API chat client.
#[derive(Clone)]
pub struct AnthropicClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    max_tokens: u32,
    default_options: ChatOptions,
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .field("max_tokens", &self.inner.max_tokens)
            .finish_non_exhaustive()
    }
}

impl AnthropicClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                max_tokens: DEFAULT_MAX_TOKENS,
                default_options: ChatOptions::default(),
            }),
        }
    }

    /// Build a client from the `ANTHROPIC_API_KEY` (and optional
    /// `ANTHROPIC_BASE_URL`) environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::Configuration("ANTHROPIC_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("ANTHROPIC_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for proxies or private deployments).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Override the default `max_tokens` sent when `ChatOptions::max_tokens`
    /// is unset (the Anthropic API requires this field on every request).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        Arc::make_mut(&mut self.inner).max_tokens = max_tokens;
        self
    }

    /// Set default [`ChatOptions`] applied as a base under any options passed
    /// per-request (per-request options take precedence; see
    /// [`ChatOptions::merge`]).
    pub fn with_default_options(mut self, options: ChatOptions) -> Self {
        Arc::make_mut(&mut self.inner).default_options = options;
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    fn build_body(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
        let effective = self.inner.default_options.clone().merge(options.clone());
        let model = effective
            .model_id
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        let max_tokens = effective.max_tokens.unwrap_or(self.inner.max_tokens);
        convert::build_request(messages, &effective, &model, max_tokens, stream)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/v1/messages", self.inner.base_url.trim_end_matches('/'));
        let resp = self
            .inner
            .http
            .post(&url)
            .header("x-api-key", &self.inner.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service_status(
                status.as_u16(),
                format!("Anthropic API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for AnthropicClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_body(&messages, &options, false);
        let resp = self.post(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown Anthropic error")
                .to_string();
            return Err(Error::service(msg));
        }
        Ok(convert::parse_response(&value))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(parse_sse_stream(resp).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Turn an Anthropic Messages API SSE HTTP response into a stream of
/// [`ChatResponseUpdate`]s.
fn parse_sse_stream(
    resp: reqwest::Response,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    futures::stream::unfold(
        SseState {
            byte_stream,
            buffer: String::new(),
            queued: VecDeque::new(),
            tool_use_ids: HashMap::new(),
            done: false,
        },
        |mut state| async move {
            loop {
                if let Some(update) = state.queued.pop_front() {
                    return Some((Ok(update), state));
                }
                if state.done {
                    return None;
                }
                match state.byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        state.buffer.push_str(&String::from_utf8_lossy(&bytes));
                        while let Some(pos) = state.buffer.find('\n') {
                            let line = state.buffer[..pos].trim().to_string();
                            state.buffer.drain(..=pos);
                            // Anthropic SSE frames an `event: <type>` line
                            // before each `data: {...}` line, but the JSON
                            // payload also carries its own `type` field, so
                            // (like the openai crate) we only need the
                            // `data:` lines.
                            let Some(data) = line.strip_prefix("data:") else {
                                continue;
                            };
                            let data = data.trim();
                            if data.is_empty() {
                                continue;
                            }
                            let Ok(value) = serde_json::from_str::<Value>(data) else {
                                continue;
                            };
                            if value.get("type").and_then(Value::as_str) == Some("error") {
                                let msg = value
                                    .get("error")
                                    .and_then(|e| e.get("message"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("unknown Anthropic stream error")
                                    .to_string();
                                state.done = true;
                                return Some((Err(Error::service(msg)), state));
                            }
                            if let Some(update) =
                                parse_stream_event(&value, &mut state.tool_use_ids)
                            {
                                state.queued.push_back(update);
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((Err(Error::service(format!("stream error: {e}"))), state));
                    }
                    None => return None,
                }
            }
        },
    )
}

/// State carried across `unfold` iterations while parsing the SSE stream.
struct SseState {
    byte_stream: ByteStream,
    buffer: String,
    queued: VecDeque<ChatResponseUpdate>,
    /// `content_block` index -> `tool_use` call id, so `input_json_delta`
    /// fragments (which carry only the index) resolve to the right call.
    tool_use_ids: HashMap<i64, String>,
    done: bool,
}

/// Parse one decoded SSE event into an update, or `None` for event types that
/// carry no content of their own (`content_block_stop`, `message_stop`,
/// `ping`, ...).
fn parse_stream_event(
    value: &Value,
    tool_use_ids: &mut HashMap<i64, String>,
) -> Option<ChatResponseUpdate> {
    match value.get("type").and_then(Value::as_str)? {
        "message_start" => {
            let message = value.get("message")?;
            let response_id = message.get("id").and_then(Value::as_str).map(String::from);
            let model_id = message
                .get("model")
                .and_then(Value::as_str)
                .map(String::from);
            let mut contents = Vec::new();
            if let Some(usage) = message.get("usage") {
                if let Some(usage_content) = convert::parse_message_start_usage(usage) {
                    contents.push(Content::Usage(usage_content));
                }
            }
            Some(ChatResponseUpdate {
                contents,
                role: Some(Role::assistant()),
                response_id,
                model_id,
                ..Default::default()
            })
        }
        "content_block_start" => {
            let index = value.get("index").and_then(Value::as_i64).unwrap_or(0);
            let block = value.get("content_block")?;
            if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                // Text/thinking blocks always start empty; real content only
                // arrives via `content_block_delta`.
                return None;
            }
            let id = block
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            tool_use_ids.insert(index, id.clone());
            Some(ChatResponseUpdate {
                contents: vec![Content::FunctionCall(FunctionCallContent::new(
                    id, name, None,
                ))],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        }
        "content_block_delta" => {
            let index = value.get("index").and_then(Value::as_i64).unwrap_or(0);
            let delta = value.get("delta")?;
            let content = match delta.get("type").and_then(Value::as_str)? {
                "text_delta" => Content::Text(TextContent::new(
                    delta
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )),
                "thinking_delta" => Content::TextReasoning(TextReasoningContent {
                    text: delta
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    annotations: None,
                }),
                "input_json_delta" => {
                    let call_id = tool_use_ids.get(&index).cloned().unwrap_or_default();
                    let partial = delta
                        .get("partial_json")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    Content::FunctionCall(FunctionCallContent::new(
                        call_id,
                        "",
                        Some(FunctionArguments::Raw(partial.to_string())),
                    ))
                }
                _ => return None,
            };
            Some(ChatResponseUpdate {
                contents: vec![content],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        }
        "message_delta" => {
            let mut contents = Vec::new();
            if let Some(usage) = value.get("usage") {
                contents.push(Content::Usage(UsageContent {
                    details: convert::parse_usage(usage),
                }));
            }
            let finish_reason = value
                .get("delta")
                .and_then(|d| d.get("stop_reason"))
                .and_then(Value::as_str)
                .map(convert::map_stop_reason);
            Some(ChatResponseUpdate {
                contents,
                finish_reason,
                ..Default::default()
            })
        }
        // `content_block_stop`, `message_stop`, `ping`, and anything else
        // carry no content of their own.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse_frame(event: &str, data: &Value) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    async fn collect_updates(text: String) -> Vec<ChatResponseUpdate> {
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = SseState {
            byte_stream,
            buffer: String::new(),
            queued: VecDeque::new(),
            tool_use_ids: HashMap::new(),
            done: false,
        };
        let mut updates = Vec::new();
        if let Some(Ok(bytes)) = state.byte_stream.next().await {
            state.buffer.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = state.buffer.find('\n') {
                let line = state.buffer[..pos].trim().to_string();
                state.buffer.drain(..=pos);
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(data).unwrap();
                if let Some(update) = parse_stream_event(&value, &mut state.tool_use_ids) {
                    updates.push(update);
                }
            }
        }
        updates
    }

    #[tokio::test]
    async fn stream_text_only_accumulates() {
        let mut text = String::new();
        text.push_str(&sse_frame(
            "message_start",
            &serde_json::json!({
                "type": "message_start",
                "message": { "id": "msg_1", "model": "claude-x", "usage": { "input_tokens": 25, "output_tokens": 1 } }
            }),
        ));
        text.push_str(&sse_frame(
            "content_block_start",
            &serde_json::json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "text", "text": "" } }),
        ));
        text.push_str(&sse_frame(
            "content_block_delta",
            &serde_json::json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "Hel" } }),
        ));
        text.push_str(&sse_frame(
            "content_block_delta",
            &serde_json::json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "text_delta", "text": "lo!" } }),
        ));
        text.push_str(&sse_frame(
            "content_block_stop",
            &serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ));
        text.push_str(&sse_frame(
            "message_delta",
            &serde_json::json!({ "type": "message_delta", "delta": { "stop_reason": "end_turn" }, "usage": { "output_tokens": 15 } }),
        ));
        text.push_str(&sse_frame(
            "message_stop",
            &serde_json::json!({ "type": "message_stop" }),
        ));

        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.response_id.as_deref(), Some("msg_1"));
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::stop())
        );
        let usage = resp.usage_details.unwrap();
        // input_tokens comes from message_start; output_tokens from
        // message_delta only (not doubled with message_start's placeholder).
        assert_eq!(usage.input_token_count, Some(25));
        assert_eq!(usage.output_token_count, Some(15));
    }

    #[tokio::test]
    async fn stream_tool_call_accumulates_arguments() {
        let mut text = String::new();
        text.push_str(&sse_frame(
            "content_block_start",
            &serde_json::json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {} } }),
        ));
        text.push_str(&sse_frame(
            "content_block_delta",
            &serde_json::json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": "{\"city\": \"San" } }),
        ));
        text.push_str(&sse_frame(
            "content_block_delta",
            &serde_json::json!({ "type": "content_block_delta", "index": 0, "delta": { "type": "input_json_delta", "partial_json": " Francisco\"}" } }),
        ));
        text.push_str(&sse_frame(
            "content_block_stop",
            &serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ));
        text.push_str(&sse_frame(
            "message_delta",
            &serde_json::json!({ "type": "message_delta", "delta": { "stop_reason": "tool_use" }, "usage": { "output_tokens": 20 } }),
        ));

        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "toolu_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &serde_json::json!("San Francisco")
        );
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::tool_calls())
        );
    }

    #[tokio::test]
    async fn stream_error_event_is_surfaced() {
        let text = sse_frame(
            "error",
            &serde_json::json!({ "type": "error", "error": { "type": "overloaded_error", "message": "Overloaded" } }),
        );
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = SseState {
            byte_stream,
            buffer: String::new(),
            queued: VecDeque::new(),
            tool_use_ids: HashMap::new(),
            done: false,
        };
        let bytes = state.byte_stream.next().await.unwrap().unwrap();
        state.buffer.push_str(&String::from_utf8_lossy(&bytes));
        let mut saw_error = false;
        while let Some(pos) = state.buffer.find('\n') {
            let line = state.buffer[..pos].trim().to_string();
            state.buffer.drain(..=pos);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(data).unwrap();
            if value.get("type").and_then(Value::as_str) == Some("error") {
                let msg = value
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                assert_eq!(msg, "Overloaded");
                saw_error = true;
            }
        }
        assert!(saw_error, "expected the error event to be recognized");
    }

    // region: env-var constructor

    /// Guards `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` mutation: tests
    /// within a crate run on multiple threads, and env vars are
    /// process-global, so this serializes access across the two tests below.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::set_var("ANTHROPIC_API_KEY", "sk-ant-test-123");
            std::env::set_var("ANTHROPIC_BASE_URL", "https://example.test");
        }
        let client = AnthropicClient::from_env("claude-x").unwrap();
        assert_eq!(client.inner.api_key, "sk-ant-test-123");
        assert_eq!(client.inner.base_url, "https://example.test");
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("ANTHROPIC_BASE_URL");
        }
    }

    #[test]
    fn from_env_errors_when_api_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("ANTHROPIC_API_KEY");
            std::env::remove_var("ANTHROPIC_BASE_URL");
        }
        let result = AnthropicClient::from_env("claude-x");
        assert!(result.is_err());
    }

    // endregion

    #[test]
    fn default_max_tokens_is_4096() {
        let client = AnthropicClient::new("key", "claude-x");
        let body = client.build_body(&[ChatMessage::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["max_tokens"], serde_json::json!(4096));
    }

    #[test]
    fn with_max_tokens_overrides_default() {
        let client = AnthropicClient::new("key", "claude-x").with_max_tokens(8192);
        let body = client.build_body(&[ChatMessage::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["max_tokens"], serde_json::json!(8192));
    }

    #[test]
    fn per_request_max_tokens_overrides_client_default() {
        let client = AnthropicClient::new("key", "claude-x").with_max_tokens(8192);
        let options = ChatOptions::new().with_max_tokens(256);
        let body = client.build_body(&[ChatMessage::user("hi")], &options, false);
        assert_eq!(body["max_tokens"], serde_json::json!(256));
    }

    #[test]
    fn with_default_options_merged_under_per_request_options() {
        let client = AnthropicClient::new("key", "claude-x")
            .with_default_options(ChatOptions::new().with_temperature(0.2));
        let body = client.build_body(&[ChatMessage::user("hi")], &ChatOptions::new(), false);
        // `temperature` is `f32`; compare against an `f32` literal so the
        // widened-to-f64 JSON values match exactly.
        assert_eq!(body["temperature"], serde_json::json!(0.2_f32));

        // Per-request temperature overrides the client default.
        let body2 = client.build_body(
            &[ChatMessage::user("hi")],
            &ChatOptions::new().with_temperature(0.9),
            false,
        );
        assert_eq!(body2["temperature"], serde_json::json!(0.9_f32));
    }
}
