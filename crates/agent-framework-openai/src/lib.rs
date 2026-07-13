//! # agent-framework-openai
//!
//! An OpenAI (and OpenAI-compatible) [`ChatClient`] for `agent-framework-rs`.
//!
//! Works against the OpenAI Chat Completions API and any compatible endpoint
//! (Azure OpenAI, Ollama, together.ai, local servers, …) by overriding the
//! base URL.
//!
//! ```no_run
//! use agent_framework_openai::OpenAIChatCompletionClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIChatCompletionClient::new("sk-...", "gpt-4o-mini");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

/// Conversion between framework types and the OpenAI chat-completions wire
/// format.
///
/// This module is public (but hidden from docs) so that other
/// OpenAI-wire-compatible clients in this workspace — currently
/// `agent-framework-azure` — can reuse request/response conversion instead of
/// duplicating it. It is not intended as a stable external API.
#[doc(hidden)]
pub mod convert;
pub mod embeddings;

pub mod responses;
pub use embeddings::OpenAIEmbeddingClient;
pub use responses::OpenAIChatClient;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, FinishReason, FunctionArguments,
    FunctionCallContent, Message, Role, TextContent, UsageContent,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};

pub(crate) const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Parse a `Retry-After` header into a delay in seconds.
///
/// HTTP allows either a delay in seconds or an HTTP-date; OpenAI (and other
/// rate limiters) use the integer/decimal-seconds form for `429`/`503`, which
/// is what we honor. A date-form or unparseable value is treated as absent.
/// Shared with [`responses`](crate::responses) so both OpenAI clients attach
/// the same retry hint to [`Error::ServiceStatus`].
pub(crate) fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// Classify a non-success OpenAI-wire (Chat Completions / Responses) HTTP
/// response into a granular [`Error`].
///
/// The single point of truth for status/body interpretation, used by every
/// endpoint in this crate (`OpenAIChatCompletionClient::post` and [`responses`]) and reused
/// by `agent-framework-azure` (Azure OpenAI is
/// wire-compatible for Chat Completions and Responses), so the two stay
/// identical rather than drifting.
///
/// Mirrors upstream's `openai/_chat_client.py` / `_responses_client.py`:
///
/// ```text
/// except BadRequestError as ex:
///     if ex.code == "content_filter":
///         raise OpenAIContentFilterException(...)
///     raise ServiceResponseException(...)
/// ```
///
/// for the content-filter case, and extends upstream's exception
/// *hierarchy* — `ServiceInvalidAuthError` / `ServiceInvalidRequestError`
/// already exist in `agent_framework.exceptions`, even though today's Python
/// OpenAI client folds every other status (auth failures included) into that
/// same generic `ServiceResponseException` — to also classify by HTTP status:
///
/// * `401` / `403` -> [`Error::ServiceInvalidAuth`]
/// * `400` / `404` / `422` -> [`Error::ServiceInvalidRequest`], unless the
///   body signals a content-filter refusal (`error.code` or `error.type` ==
///   `"content_filter"`, checked at the top level and inside a nested
///   `innererror` for Azure OpenAI's shape) -> [`Error::ServiceContentFilter`]
/// * anything else — notably `408` / `429` / `5xx`, which
///   [`RetryOn::Default`](agent_framework_core::client::RetryOn::Default)
///   depends on — -> [`Error::ServiceStatus`], unchanged
///
/// `body` is parsed leniently as JSON purely to look for the content-filter
/// marker; a non-JSON or differently-shaped body just skips that check and
/// falls through to the plain status-based classification, so this never
/// panics or itself errors. `message` is used verbatim as the resulting
/// error's text (callers already format a provider-specific "OpenAI API
/// error 400: ..." string; this only picks the variant).
pub fn classify_service_error(
    status: u16,
    body: &str,
    message: impl Into<String>,
    retry_after: Option<f64>,
) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 | 404 | 422 => {
            if body_signals_content_filter(body) {
                Error::service_content_filter(message)
            } else {
                Error::service_invalid_request(message)
            }
        }
        _ => Error::service_status(status, message, retry_after),
    }
}

/// Whether an OpenAI/Azure-OpenAI-shaped error body signals a content-filter
/// refusal.
///
/// Checks `error.code` — the field the OpenAI Python SDK's
/// `BadRequestError.code` reads, and what upstream compares against
/// `"content_filter"` — tolerantly falling back to `error.type` and to Azure
/// OpenAI's nested `error.innererror` (`code`/`type`), since a
/// `"content_filter"` marker has been observed in slightly different spots
/// across OpenAI-wire-compatible providers. A non-JSON or unrecognized body
/// shape is treated as "not a content filter" rather than erroring.
fn body_signals_content_filter(body: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return false;
    };
    let error = value.get("error").unwrap_or(&value);
    is_content_filter_marker(error)
        || error
            .get("innererror")
            .is_some_and(is_content_filter_marker)
}

fn is_content_filter_marker(v: &Value) -> bool {
    v.get("code").and_then(Value::as_str) == Some("content_filter")
        || v.get("type").and_then(Value::as_str) == Some("content_filter")
}

/// An OpenAI (or OpenAI-compatible) chat client.
#[derive(Clone)]
pub struct OpenAIChatCompletionClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
}

impl OpenAIChatCompletionClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                organization: None,
            }),
        }
    }

    /// Build a client from the `OPENAI_API_KEY` (and optional
    /// `OPENAI_BASE_URL`) environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("OPENAI_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for Azure OpenAI or compatible servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).organization = Some(org.into());
        self
    }

    fn build_body(&self, messages: &[Message], options: &ChatOptions, stream: bool) -> Value {
        let mut body = Map::new();
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        body.insert("model".into(), json!(model));
        body.insert(
            "messages".into(),
            json!(convert::messages_to_openai(messages)),
        );
        convert::apply_options(&mut body, options);
        let (tools, tool_choice) = convert::tools_to_openai(options);
        if let Some(tools) = tools {
            body.insert("tools".into(), tools);
        }
        if let Some(choice) = tool_choice {
            body.insert("tool_choice".into(), choice);
        }
        if stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        Value::Object(body)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!(
            "{}/chat/completions",
            self.inner.base_url.trim_end_matches('/')
        );
        let mut req = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&self.inner.api_key)
            .json(body);
        if let Some(org) = &self.inner.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_service_error(
                status.as_u16(),
                &text,
                format!("OpenAI API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }
}

#[async_trait::async_trait]
impl ChatClient for OpenAIChatCompletionClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_body(&messages, &options, false);
        let resp = self.post(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(convert::parse_response(&value))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(parse_sse_stream(resp).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Turn an SSE HTTP response into a stream of [`ChatResponseUpdate`]s.
///
/// Public (but hidden) so `agent-framework-azure` can reuse the exact same
/// chat-completions SSE parsing for Azure OpenAI's wire-compatible stream.
#[doc(hidden)]
pub fn parse_sse_stream(
    resp: reqwest::Response,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    // `tool_ids` maps a streamed tool-call `index` to its `call_id`, so that
    // continuation chunks (which carry only the index) can be resolved back to
    // the id assigned in the first chunk.
    futures::stream::unfold(
        SseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_ids: HashMap::new(),
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
                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    return drain_or_end(state);
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    // Providers signal mid-stream failures (rate
                                    // limits, content filter, billing) as an
                                    // object with an `error` field; surface it.
                                    if let Some(err) = value.get("error") {
                                        let msg = err
                                            .get("message")
                                            .and_then(Value::as_str)
                                            .unwrap_or("unknown stream error")
                                            .to_string();
                                        state.done = true;
                                        return Some((Err(Error::service(msg)), state));
                                    }
                                    if let Some(update) = parse_delta(&value, &mut state.tool_ids) {
                                        state.queued.push_back(update);
                                    }
                                }
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((Err(Error::service(format!("stream error: {e}"))), state));
                    }
                    None => return drain_or_end(state),
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
    tool_ids: HashMap<i64, String>,
    done: bool,
}

fn drain_or_end(mut state: SseState) -> Option<(Result<ChatResponseUpdate>, SseState)> {
    match state.queued.pop_front() {
        Some(update) => {
            state.done = true;
            Some((Ok(update), state))
        }
        None => None,
    }
}

/// Parse one streamed chunk (`chat.completion.chunk`) into an update, resolving
/// tool-call ids from the index map.
fn parse_delta(value: &Value, tool_ids: &mut HashMap<i64, String>) -> Option<ChatResponseUpdate> {
    let mut update = ChatResponseUpdate {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    let mut contents: Vec<Content> = Vec::new();

    if let Some(choice) = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        if let Some(delta) = choice.get("delta") {
            if let Some(r) = delta.get("role").and_then(Value::as_str) {
                update.role = Some(Role::new(r));
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    contents.push(Content::Text(TextContent::new(text)));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let index = call.get("index").and_then(Value::as_i64).unwrap_or(0);
                    let chunk_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
                    // The first chunk of a call carries its id; record it so
                    // later index-only chunks resolve to the same call_id.
                    let id = if chunk_id.is_empty() {
                        tool_ids.get(&index).cloned().unwrap_or_default()
                    } else {
                        tool_ids.insert(index, chunk_id.to_string());
                        chunk_id.to_string()
                    };
                    let func = call.get("function");
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let args = func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    contents.push(Content::FunctionCall(FunctionCallContent::new(
                        id,
                        name,
                        Some(FunctionArguments::Raw(args)),
                    )));
                }
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            update.finish_reason = Some(FinishReason::new(fr));
        }
    }

    // The final chunk (with `stream_options.include_usage`) carries top-level
    // `usage` and no choices; surface it so streamed runs accumulate token usage
    // just like non-streaming responses.
    if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
        contents.push(Content::Usage(UsageContent {
            details: convert::parse_usage(usage),
        }));
    }

    if update.role.is_none() {
        update.role = Some(Role::assistant());
    }
    update.contents = contents;
    Some(update)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- classify_service_error --------------------------------------------
    //
    // Canned status+body combinations run through the exact classification
    // `OpenAIChatCompletionClient::post` and `responses::OpenAIChatClient::post`
    // both delegate to.

    #[test]
    fn classifies_401_and_403_as_invalid_auth() {
        for status in [401, 403] {
            let err = classify_service_error(status, "", format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidAuth { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_400_404_422_as_invalid_request_by_default() {
        for status in [400, 404, 422] {
            let body = r#"{"error":{"message":"nope","type":"invalid_request_error"}}"#;
            let err = classify_service_error(status, body, format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidRequest { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_400_with_content_filter_code_as_content_filter() {
        // Plain OpenAI shape: `error.code`.
        let body = r#"{"error":{"message":"flagged","type":"invalid_request_error","code":"content_filter"}}"#;
        let err = classify_service_error(400, body, "err", None);
        assert!(matches!(err, Error::ServiceContentFilter { .. }), "{err:?}");
    }

    #[test]
    fn classifies_azure_openai_content_filter_shape_with_innererror() {
        // Azure OpenAI's shape: outer `error.code` is already "content_filter"
        // (mirrors `openai/_exceptions.py`'s `OpenAIContentFilterException`,
        // which is constructed once `ex.code == "content_filter"` is already
        // known — the nested `innererror.code` carries the more specific
        // `ResponsibleAIPolicyViolation` detail, not the marker itself), but
        // this also tolerates the marker living only in `innererror`.
        let body = r#"{"error":{"message":"The response was filtered","code":"content_filter","innererror":{"code":"ResponsibleAIPolicyViolation"}}}"#;
        let err = classify_service_error(400, body, "err", None);
        assert!(matches!(err, Error::ServiceContentFilter { .. }), "{err:?}");

        let nested_only =
            r#"{"error":{"message":"filtered","innererror":{"code":"content_filter"}}}"#;
        let err = classify_service_error(400, nested_only, "err", None);
        assert!(matches!(err, Error::ServiceContentFilter { .. }), "{err:?}");
    }

    #[test]
    fn non_json_or_unmarked_body_does_not_trigger_content_filter() {
        let err = classify_service_error(400, "not json", "err", None);
        assert!(
            matches!(err, Error::ServiceInvalidRequest { .. }),
            "{err:?}"
        );

        let err = classify_service_error(400, r#"{"error":{"code":"other"}}"#, "err", None);
        assert!(
            matches!(err, Error::ServiceInvalidRequest { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn leaves_retryable_statuses_as_service_status() {
        // 408/429/5xx must stay `ServiceStatus` (carrying status +
        // retry_after) exactly as before — the retry layer depends on it.
        for status in [408, 429, 500, 503] {
            let err = classify_service_error(status, "", format!("err {status}"), Some(2.0));
            assert_eq!(err.status(), Some(status), "{err:?}");
            assert_eq!(err.retry_after(), Some(2.0), "{err:?}");
        }
    }

    #[test]
    fn unclassified_4xx_stays_service_status() {
        // e.g. 409 Conflict, 413 Payload Too Large: not in the
        // auth/invalid-request/content-filter buckets, so this falls back to
        // the generic status-carrying variant rather than guessing.
        let err = classify_service_error(409, "", "err", None);
        assert_eq!(err.status(), Some(409), "{err:?}");
    }

    #[test]
    fn message_text_is_preserved_verbatim() {
        let err = classify_service_error(401, "", "OpenAI API error 401: unauthorized", None);
        assert_eq!(
            err.to_string(),
            "service error: OpenAI API error 401: unauthorized"
        );
    }
}
