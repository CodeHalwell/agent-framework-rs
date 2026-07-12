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
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FunctionArguments, FunctionCallContent,
    Message, Role, TextContent, TextReasoningContent, UsageContent,
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

/// Classify a non-success Anthropic Messages API HTTP response into a
/// granular [`Error`].
///
/// Upstream's Anthropic connector (`agent_framework_anthropic/_chat_client.py`)
/// does not wrap `messages.create`/`beta.messages.create` in any
/// status-specific exception handling at all — SDK errors propagate
/// unchanged, so there is no Python call-site behavior to mirror status by
/// status here. This instead applies upstream's exception *hierarchy*
/// (`agent_framework.exceptions.ServiceInvalidAuthError` /
/// `ServiceInvalidRequestError`) using Anthropic's own documented
/// status <-> `error.type` convention
/// (<https://docs.anthropic.com/en/api/errors>):
///
/// * `401` / `403` -> [`Error::ServiceInvalidAuth`] (Anthropic's
///   `authentication_error` / `permission_error`)
/// * `400` -> [`Error::ServiceInvalidRequest`], but only once the body
///   confirms `error.type == "invalid_request_error"` (Anthropic's
///   documented sole `400` type); an unparseable or unexpected body
///   conservatively falls back to the generic [`Error::ServiceStatus`]
///   rather than guessing
/// * anything else — notably `408` / `429` / `5xx`, which the retry layer
///   depends on — -> [`Error::ServiceStatus`], unchanged
///
/// Anthropic has no content-filter-specific HTTP error to classify: a
/// content-policy refusal is a `200 OK` response with `stop_reason:
/// "refusal"`, mapped to `FinishReason::CONTENT_FILTER` by
/// [`convert::map_stop_reason`] (mirroring upstream's `FINISH_REASON_MAP`)
/// rather than raised as an error, so [`Error::ServiceContentFilter`] is
/// never constructed on this path — don't invent one.
fn classify_anthropic_error(
    status: u16,
    body: &str,
    message: impl Into<String>,
    retry_after: Option<f64>,
) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 if anthropic_error_type(body).as_deref() == Some("invalid_request_error") => {
            Error::service_invalid_request(message)
        }
        _ => Error::service_status(status, message, retry_after),
    }
}

/// The Anthropic error body's `error.type`, if the body parses as JSON and
/// carries one (e.g. `"invalid_request_error"`, `"authentication_error"`).
fn anthropic_error_type(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .get("error")?
        .get("type")?
        .as_str()
        .map(str::to_string)
}

/// Build the base `POST /v1/messages` request, including the `anthropic-beta`
/// header when `betas` is non-empty. Split out from [`AnthropicClient::post`]
/// so the header-attachment logic is unit-testable without an HTTP round
/// trip.
///
/// Upstream always passes a non-empty `betas` set to `beta.messages.create`
/// (at minimum [`convert::DEFAULT_BETA_FLAGS`]), so in practice the header is
/// unconditionally present; the emptiness check here just avoids sending a
/// spurious empty header if some future caller manages to clear the default
/// set entirely.
fn new_message_request(
    http: &reqwest::Client,
    url: &str,
    api_key: &str,
    betas: &[String],
) -> reqwest::RequestBuilder {
    let mut request = http
        .post(url)
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("content-type", "application/json");
    if !betas.is_empty() {
        request = request.header("anthropic-beta", betas.join(","));
    }
    request
}
/// `max_tokens` is required by the Anthropic Messages API; this is used
/// whenever neither `ChatOptions::max_tokens` nor a client-level override is
/// set. Matches upstream's `ANTHROPIC_DEFAULT_MAX_TOKENS`
/// (`agent_framework_anthropic/_chat_client.py` ~line 53).
const DEFAULT_MAX_TOKENS: u32 = 1024;

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
    /// Additional `anthropic-beta` flags unioned with
    /// [`convert::DEFAULT_BETA_FLAGS`] on every request. Mirrors upstream's
    /// `additional_beta_flags` constructor keyword argument
    /// (`AnthropicClient.__init__`, `_chat_client.py` ~126, stored as
    /// `self.additional_beta_flags` ~203).
    additional_beta_flags: Vec<String>,
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
                additional_beta_flags: Vec::new(),
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

    /// Additional `anthropic-beta` flags to send (via the `anthropic-beta`
    /// header) on every request, unioned with the always-on
    /// [`convert::DEFAULT_BETA_FLAGS`] and any per-request flags supplied
    /// through `ChatOptions::additional_properties["additional_beta_flags"]`.
    ///
    /// Mirrors upstream's `additional_beta_flags` constructor keyword
    /// argument (`_chat_client.py` ~126, ~139-140): "Default flags are:
    /// `mcp-client-2025-04-04`, `code-execution-2025-08-25`."
    pub fn with_additional_beta_flags(
        mut self,
        flags: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Arc::make_mut(&mut self.inner).additional_beta_flags =
            flags.into_iter().map(Into::into).collect();
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Build the request body and the `anthropic-beta` flags for a request.
    ///
    /// The beta-flags computation (mirroring upstream's
    /// `chat_options.additional_properties.pop("additional_beta_flags")`,
    /// `_chat_client.py` ~254-264) must run against the same *merged* +
    /// owned [`ChatOptions`] that [`convert::build_request`] then converts,
    /// and before it does: [`convert::compute_beta_flags`] removes the
    /// `additional_beta_flags` key from `additional_properties` so it is not
    /// also copied into the request body as a stray top-level field.
    fn build_body(
        &self,
        messages: &[Message],
        options: &ChatOptions,
        stream: bool,
    ) -> (Value, Vec<String>) {
        let mut effective = self.inner.default_options.clone().merge(options.clone());
        let betas = convert::compute_beta_flags(&mut effective, &self.inner.additional_beta_flags);
        let model = effective
            .model_id
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        let max_tokens = effective.max_tokens.unwrap_or(self.inner.max_tokens);
        let body = convert::build_request(messages, &effective, &model, max_tokens, stream);
        (body, betas)
    }

    async fn post(&self, body: &Value, betas: &[String]) -> Result<reqwest::Response> {
        let url = format!("{}/v1/messages", self.inner.base_url.trim_end_matches('/'));
        let request = new_message_request(&self.inner.http, &url, &self.inner.api_key, betas);
        let resp = request
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_anthropic_error(
                status.as_u16(),
                &text,
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
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let (body, betas) = self.build_body(&messages, &options, false);
        let resp = self.post(&body, &betas).await?;
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
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let (body, betas) = self.build_body(&messages, &options, true);
        let resp = self.post(&body, &betas).await?;
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
            utf8: Utf8StreamDecoder::new(),
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
                        let decoded = state.utf8.push(&bytes);
                        state.buffer.push_str(&decoded);
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
    utf8: Utf8StreamDecoder,
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
            match block.get("type").and_then(Value::as_str)? {
                "tool_use" | "mcp_tool_use" | "server_tool_use" => {
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
                "text" | "thinking" => {
                    // Text/thinking blocks always start empty; real content
                    // only arrives via `content_block_delta`.
                    None
                }
                _ => {
                    // Atomic hosted-tool result blocks (`mcp_tool_result`,
                    // `web_search_tool_result`, `web_fetch_tool_result`,
                    // `code_execution_tool_result`, and siblings) are
                    // delivered whole in a single `content_block_start`, with
                    // no follow-up deltas: mirrors upstream, which funnels
                    // `content_block_start`'s block through the very same
                    // `_parse_message_contents` used for full (non-streaming)
                    // responses (`_process_stream_event`'s
                    // `case "content_block_start":`, `_chat_client.py`
                    // ~490-495).
                    let contents = convert::parse_content_blocks(std::slice::from_ref(block));
                    if contents.is_empty() {
                        None
                    } else {
                        Some(ChatResponseUpdate {
                            contents,
                            role: Some(Role::assistant()),
                            ..Default::default()
                        })
                    }
                }
            }
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
                    ..Default::default()
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
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_use_ids: HashMap::new(),
            done: false,
        };
        let mut updates = Vec::new();
        if let Some(Ok(bytes)) = state.byte_stream.next().await {
            let decoded = state.utf8.push(&bytes);
            state.buffer.push_str(&decoded);
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
    async fn stream_hosted_tool_use_and_result_via_content_block_start() {
        // `server_tool_use` (the hosted-tool invocation) and
        // `web_search_tool_result` (its atomic result) both arrive as
        // complete `content_block_start` blocks with no follow-up deltas --
        // mirrors upstream funneling `content_block_start`'s block through
        // the same content parser used for full responses.
        let mut text = String::new();
        text.push_str(&sse_frame(
            "content_block_start",
            &serde_json::json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "rust" } } }),
        ));
        text.push_str(&sse_frame(
            "content_block_stop",
            &serde_json::json!({ "type": "content_block_stop", "index": 0 }),
        ));
        text.push_str(&sse_frame(
            "content_block_start",
            &serde_json::json!({ "type": "content_block_start", "index": 1, "content_block": { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }] } }),
        ));
        text.push_str(&sse_frame(
            "content_block_stop",
            &serde_json::json!({ "type": "content_block_stop", "index": 1 }),
        ));

        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "srvtoolu_1");
        assert_eq!(calls[0].name, "web_search");
        let has_function_result = resp
            .messages
            .iter()
            .flat_map(|m| &m.contents)
            .any(|c| matches!(c, Content::FunctionResult(_)));
        assert!(
            has_function_result,
            "expected a FunctionResult content from the web_search_tool_result block"
        );
    }

    #[tokio::test]
    async fn stream_mcp_tool_use_via_content_block_start() {
        let text = sse_frame(
            "content_block_start",
            &serde_json::json!({ "type": "content_block_start", "index": 0, "content_block": { "type": "mcp_tool_use", "id": "mcptoolu_1", "name": "search_docs", "server_name": "docs", "input": {} } }),
        );
        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "mcptoolu_1");
        assert_eq!(calls[0].name, "search_docs");
    }

    #[tokio::test]
    async fn stream_citations_delta_is_ignored_like_upstream() {
        // Upstream's `_process_stream_event`/`_parse_message_contents` has no
        // case for the `citations_delta` delta type (it only appears inside
        // `content_block_delta`), so it falls through to a debug-logged
        // no-op; citations are only ever populated from a full `text` block's
        // `citations` array (non-streaming, or a hypothetical fully-formed
        // streamed block). This asserts the streaming path tolerates the
        // event (no panic, no spurious update) rather than mirroring
        // citation *population* during streaming, which upstream doesn't do
        // either.
        let text = sse_frame(
            "content_block_delta",
            &serde_json::json!({
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "citations_delta",
                    "citation": {
                        "type": "char_location",
                        "cited_text": "example",
                        "document_index": 0,
                        "document_title": "Doc",
                        "start_char_index": 0,
                        "end_char_index": 7
                    }
                }
            }),
        );
        let updates = collect_updates(text).await;
        assert!(
            updates.is_empty(),
            "citations_delta should not produce an update, matching upstream"
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
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_use_ids: HashMap::new(),
            done: false,
        };
        let bytes = state.byte_stream.next().await.unwrap().unwrap();
        let decoded = state.utf8.push(&bytes);
        state.buffer.push_str(&decoded);
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
    fn default_max_tokens_is_1024() {
        // Matches upstream's `ANTHROPIC_DEFAULT_MAX_TOKENS` (`_chat_client.py`
        // ~line 53), not the historical Rust default of 4096.
        let client = AnthropicClient::new("key", "claude-x");
        let (body, _betas) = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["max_tokens"], serde_json::json!(1024));
    }

    #[test]
    fn with_max_tokens_overrides_default() {
        let client = AnthropicClient::new("key", "claude-x").with_max_tokens(8192);
        let (body, _betas) = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["max_tokens"], serde_json::json!(8192));
    }

    #[test]
    fn per_request_max_tokens_overrides_client_default() {
        let client = AnthropicClient::new("key", "claude-x").with_max_tokens(8192);
        let options = ChatOptions::new().with_max_tokens(256);
        let (body, _betas) = client.build_body(&[Message::user("hi")], &options, false);
        assert_eq!(body["max_tokens"], serde_json::json!(256));
    }

    #[test]
    fn with_default_options_merged_under_per_request_options() {
        let client = AnthropicClient::new("key", "claude-x")
            .with_default_options(ChatOptions::new().with_temperature(0.2));
        let (body, _betas) = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        // `temperature` is `f32`; compare against an `f32` literal so the
        // widened-to-f64 JSON values match exactly.
        assert_eq!(body["temperature"], serde_json::json!(0.2_f32));

        // Per-request temperature overrides the client default.
        let (body2, _betas2) = client.build_body(
            &[Message::user("hi")],
            &ChatOptions::new().with_temperature(0.9),
            false,
        );
        assert_eq!(body2["temperature"], serde_json::json!(0.9_f32));
    }

    // region: beta flags

    #[test]
    fn build_body_always_includes_default_beta_flags() {
        // Upstream sends `betas` on every `beta.messages.create` call, not
        // only when hosted tools/MCP servers are present -- verified against
        // `_create_run_options` (`_chat_client.py` ~254-264).
        let client = AnthropicClient::new("key", "claude-x");
        let (_body, betas) = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert!(betas.contains(&"mcp-client-2025-04-04".to_string()));
        assert!(betas.contains(&"code-execution-2025-08-25".to_string()));
        assert_eq!(betas.len(), 2);
    }

    #[test]
    fn build_body_merges_client_level_additional_beta_flags() {
        let client =
            AnthropicClient::new("key", "claude-x").with_additional_beta_flags(["my-custom-beta"]);
        let (_body, betas) = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert!(betas.contains(&"my-custom-beta".to_string()));
        assert!(betas.contains(&"mcp-client-2025-04-04".to_string()));
        assert_eq!(betas.len(), 3);
    }

    #[test]
    fn build_body_merges_per_request_additional_beta_flags_and_strips_them_from_body() {
        let client = AnthropicClient::new("key", "claude-x");
        let mut options = ChatOptions::new();
        options.additional_properties.insert(
            "additional_beta_flags".into(),
            serde_json::json!(["request-only-beta"]),
        );
        let (body, betas) = client.build_body(&[Message::user("hi")], &options, false);
        assert!(betas.contains(&"request-only-beta".to_string()));
        // Popped, like upstream's `.pop("additional_beta_flags")` -- must not
        // leak into the JSON body as a stray top-level field.
        assert!(body.get("additional_beta_flags").is_none());
    }

    #[test]
    fn new_message_request_sets_anthropic_beta_header_when_betas_present() {
        // This is the same helper `post` calls, so it exercises the actual
        // header-attachment code path (unlike hitting the network).
        let http = reqwest::Client::new();
        let betas = vec!["a".to_string(), "b".to_string()];
        let request = new_message_request(
            &http,
            "https://api.anthropic.com/v1/messages",
            "test-key",
            &betas,
        )
        .build()
        .unwrap();
        assert_eq!(request.headers().get("anthropic-beta").unwrap(), "a,b");
    }

    #[test]
    fn new_message_request_omits_anthropic_beta_header_when_betas_empty() {
        let http = reqwest::Client::new();
        let request = new_message_request(
            &http,
            "https://api.anthropic.com/v1/messages",
            "test-key",
            &[],
        )
        .build()
        .unwrap();
        assert!(request.headers().get("anthropic-beta").is_none());
    }

    // endregion

    // region: classify_anthropic_error

    #[test]
    fn classifies_401_and_403_as_invalid_auth() {
        for status in [401, 403] {
            let body = format!(
                r#"{{"type":"error","error":{{"type":"authentication_error","message":"nope {status}"}}}}"#
            );
            let err = classify_anthropic_error(status, &body, format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidAuth { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_400_invalid_request_error_as_invalid_request() {
        let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"messages: at least one message is required"}}"#;
        let err = classify_anthropic_error(400, body, "err", None);
        assert!(
            matches!(err, Error::ServiceInvalidRequest { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_400_without_confirming_body_stays_service_status() {
        // Conservative: only reclassify once the body actually confirms
        // Anthropic's documented `invalid_request_error` type; an
        // unparseable or differently-typed body falls back to the generic
        // status-carrying variant rather than guessing.
        let err = classify_anthropic_error(400, "not json", "err", None);
        assert_eq!(err.status(), Some(400), "{err:?}");

        let err = classify_anthropic_error(
            400,
            r#"{"type":"error","error":{"type":"something_else"}}"#,
            "err",
            None,
        );
        assert_eq!(err.status(), Some(400), "{err:?}");
    }

    #[test]
    fn leaves_retryable_statuses_as_service_status() {
        // 408/429/5xx (and Anthropic's overloaded 529) must stay
        // `ServiceStatus` exactly as before — the retry layer depends on it.
        for status in [408, 429, 500, 529] {
            let err = classify_anthropic_error(status, "", format!("err {status}"), Some(1.5));
            assert_eq!(err.status(), Some(status), "{err:?}");
            assert_eq!(err.retry_after(), Some(1.5), "{err:?}");
        }
    }

    #[test]
    fn never_produces_content_filter() {
        // Anthropic has no content-filter-specific HTTP error: content-policy
        // refusals surface as `stop_reason: "refusal"` on a 200 (see
        // `map_stop_reason_covers_documented_mapping`), never as a non-success
        // status, so this path must never invent a `ServiceContentFilter`.
        let bodies = [
            "",
            "not json",
            r#"{"type":"error","error":{"type":"invalid_request_error"}}"#,
            r#"{"type":"error","error":{"type":"authentication_error"}}"#,
        ];
        for status in [400, 401, 403, 404, 422, 429, 500] {
            for body in bodies {
                let err = classify_anthropic_error(status, body, "err", None);
                assert!(
                    !matches!(err, Error::ServiceContentFilter { .. }),
                    "status {status}, body {body:?}: {err:?}"
                );
            }
        }
    }

    // endregion
}
