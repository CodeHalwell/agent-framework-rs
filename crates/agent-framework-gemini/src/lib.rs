//! # agent-framework-gemini
//!
//! A Google Gemini [`ChatClient`] for
//! `agent-framework-rs`.
//!
//! Talks directly to the Gemini `generateContent` REST API
//! (`POST /v1beta/models/{model}:generateContent`, or
//! `:streamGenerateContent?alt=sse` for streaming) with hand-rolled
//! request/response JSON conversion and a hand-rolled SSE parser — no
//! dependency on Google's own SDK. Unlike the OpenAI-shaped Chat Completions
//! wire format that `agent-framework-openai` and `agent-framework-mistral`
//! use, Gemini's request/response shape is its own:
//! `{contents:[{role:"user"|"model",parts:[{text}]}],
//! generationConfig:{temperature,maxOutputTokens},
//! systemInstruction:{parts:[{text}]}}` in, `{candidates:[{content:{parts}}],
//! usageMetadata:{promptTokenCount,candidatesTokenCount,totalTokenCount}}`
//! out. See `convert` for the full mapping.
//!
//! ```no_run
//! use agent_framework_gemini::GeminiChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = GeminiChatClient::new("AIza...", "gemini-2.5-flash");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

mod convert;

use std::collections::VecDeque;
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
use futures::StreamExt;
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
const API_VERSION: &str = "v1beta";

/// Parse Gemini's `Retry-After` header (seconds) into a delay, when present.
///
/// Gemini's REST API does not document a `Retry-After` header the way
/// Anthropic does, but `reqwest`/the underlying transport may still surface
/// one via a proxy or future API revision, so this is checked defensively —
/// mirroring `agent-framework-anthropic`'s `parse_retry_after` — rather than
/// assumed absent.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// Classify a non-success Gemini API HTTP response into a granular
/// [`Error`].
///
/// Gemini errors are shaped `{"error":{"code":<status>,"message":...,
/// "status":"INVALID_ARGUMENT"}}` (Google API-common error model,
/// <https://cloud.google.com/apis/design/errors>):
///
/// * `401` / `403` -> [`Error::ServiceInvalidAuth`] (Gemini's
///   `UNAUTHENTICATED` / `PERMISSION_DENIED`)
/// * `400` -> [`Error::ServiceInvalidRequest`], but only once the body
///   confirms `error.status == "INVALID_ARGUMENT"`; an unparseable or
///   unexpected body conservatively falls back to the generic
///   [`Error::ServiceStatus`] rather than guessing
/// * anything else — notably `429` / `5xx`, which the retry layer depends
///   on — -> [`Error::ServiceStatus`], unchanged
///
/// Gemini has no content-filter-specific HTTP error either: a blocked
/// prompt is a `200 OK` with `promptFeedback.blockReason` and no
/// `candidates`, mapped to `FinishReason::CONTENT_FILTER` by
/// [`convert::parse_response`] rather than raised as an error, so
/// [`Error::ServiceContentFilter`] is never constructed on this path.
fn classify_gemini_error(
    status: u16,
    body: &str,
    message: impl Into<String>,
    retry_after: Option<f64>,
) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 if gemini_error_status(body).as_deref() == Some("INVALID_ARGUMENT") => {
            Error::service_invalid_request(message)
        }
        _ => Error::service_status(status, message, retry_after),
    }
}

/// The Gemini error body's `error.status`, if the body parses as JSON and
/// carries one (e.g. `"INVALID_ARGUMENT"`, `"PERMISSION_DENIED"`).
fn gemini_error_status(body: &str) -> Option<String> {
    let value: Value = serde_json::from_str(body).ok()?;
    value
        .get("error")?
        .get("status")?
        .as_str()
        .map(str::to_string)
}

/// A Google Gemini `generateContent` REST API chat client.
#[derive(Clone)]
pub struct GeminiChatClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    default_options: ChatOptions,
}

impl std::fmt::Debug for GeminiChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GeminiChatClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish_non_exhaustive()
    }
}

impl GeminiChatClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                default_options: ChatOptions::default(),
            }),
        }
    }

    /// Build a client from the `GEMINI_API_KEY` environment variable,
    /// falling back to `GOOGLE_API_KEY` when the former is unset (matching
    /// upstream's Gemini connector, which accepts either). Also reads an
    /// optional `GEMINI_BASE_URL` override.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("GEMINI_API_KEY")
            .or_else(|_| std::env::var("GOOGLE_API_KEY"))
            .map_err(|_| {
                Error::Configuration("neither GEMINI_API_KEY nor GOOGLE_API_KEY is set".into())
            })?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("GEMINI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for proxies or private deployments). Should
    /// not include the `/v1beta` API-version segment; that is appended per
    /// request.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
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

    /// Build the request body and resolve the effective model id for a
    /// request (per-request `ChatOptions::model` overrides the client
    /// default; the model is a URL path segment for this API, not a body
    /// field).
    fn build_body(&self, messages: &[Message], options: &ChatOptions) -> (Value, String) {
        let effective = self.inner.default_options.clone().merge(options.clone());
        let model = effective
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        let body = convert::build_request(messages, &effective);
        (body, model)
    }

    /// POST to `{base_url}/v1beta/models/{model}:{method}`, optionally with
    /// `alt=sse` for streaming, authenticating via the `x-goog-api-key`
    /// header (rather than Gemini's alternative `?key=` query-string form,
    /// which would leak the key into logs/proxies more readily).
    async fn post(
        &self,
        method: &str,
        model: &str,
        body: &Value,
        sse: bool,
    ) -> Result<reqwest::Response> {
        let mut url = format!(
            "{}/{API_VERSION}/models/{model}:{method}",
            self.inner.base_url.trim_end_matches('/')
        );
        if sse {
            url.push_str("?alt=sse");
        }
        let resp = self
            .inner
            .http
            .post(&url)
            .header("x-goog-api-key", &self.inner.api_key)
            .header("content-type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_gemini_error(
                status.as_u16(),
                &text,
                format!("Gemini API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for GeminiChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let (body, model) = self.build_body(&messages, &options);
        let resp = self.post("generateContent", &model, &body, false).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("unknown Gemini error")
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
        let (body, model) = self.build_body(&messages, &options);
        let resp = self
            .post("streamGenerateContent", &model, &body, true)
            .await?;
        Ok(parse_sse_stream(resp).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

/// Turn a Gemini `streamGenerateContent?alt=sse` HTTP response into a stream
/// of [`ChatResponseUpdate`]s.
///
/// Unlike Anthropic's SSE protocol (typed `event:`-tagged deltas that must be
/// assembled), each Gemini `data:` line is itself a complete, self-contained
/// `GenerateContentResponse` JSON object — the same shape
/// [`convert::parse_response`] parses non-streaming, just with incremental
/// `parts` per chunk — so no `event:` line, and no cross-chunk id/index
/// bookkeeping, is needed.
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
                            if let Some(err) = value.get("error") {
                                let msg = err
                                    .get("message")
                                    .and_then(Value::as_str)
                                    .unwrap_or("unknown Gemini stream error")
                                    .to_string();
                                state.done = true;
                                return Some((Err(Error::service(msg)), state));
                            }
                            if let Some(update) = convert::parse_stream_chunk(&value) {
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
    done: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sse_frame(data: &Value) -> String {
        format!("data: {data}\n\n")
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
                if let Some(update) = convert::parse_stream_chunk(&value) {
                    updates.push(update);
                }
            }
        }
        updates
    }

    #[tokio::test]
    async fn stream_text_only_accumulates() {
        let mut text = String::new();
        text.push_str(&sse_frame(&serde_json::json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hel" }] } }],
        })));
        text.push_str(&sse_frame(&serde_json::json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": "lo!" }] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 3, "totalTokenCount": 8 },
        })));

        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::stop())
        );
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(5));
        assert_eq!(usage.output_token_count, Some(3));
        assert_eq!(usage.total_token_count, Some(8));
    }

    #[tokio::test]
    async fn stream_tool_call_arrives_whole_and_upgrades_finish_reason() {
        let text = sse_frame(&serde_json::json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "functionCall": { "name": "get_weather", "args": { "city": "San Francisco" } } }
                ] },
                "finishReason": "STOP",
            }],
        }));

        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
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
        let text = sse_frame(&serde_json::json!({
            "error": { "code": 503, "message": "The model is overloaded", "status": "UNAVAILABLE" }
        }));
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = SseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
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
            if let Some(err) = value.get("error") {
                let msg = err
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                assert_eq!(msg, "The model is overloaded");
                saw_error = true;
            }
        }
        assert!(saw_error, "expected the error event to be recognized");
    }

    // region: env-var constructor

    /// Guards `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_BASE_URL`
    /// mutation: tests within a crate run on multiple threads, and env vars
    /// are process-global, so this serializes access across the env-var
    /// tests below.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_gemini_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::remove_var("GOOGLE_API_KEY");
            std::env::set_var("GEMINI_API_KEY", "gemini-test-key");
            std::env::set_var("GEMINI_BASE_URL", "https://example.test");
        }
        let client = GeminiChatClient::from_env("gemini-x").unwrap();
        assert_eq!(client.inner.api_key, "gemini-test-key");
        assert_eq!(client.inner.base_url, "https://example.test");
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GEMINI_BASE_URL");
        }
    }

    #[test]
    fn from_env_falls_back_to_google_api_key() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::set_var("GOOGLE_API_KEY", "google-test-key");
        }
        let client = GeminiChatClient::from_env("gemini-x").unwrap();
        assert_eq!(client.inner.api_key, "google-test-key");
        unsafe {
            std::env::remove_var("GOOGLE_API_KEY");
        }
    }

    #[test]
    fn from_env_errors_when_no_key_set() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("GEMINI_API_KEY");
            std::env::remove_var("GOOGLE_API_KEY");
        }
        let result = GeminiChatClient::from_env("gemini-x");
        assert!(result.is_err());
    }

    // endregion

    #[test]
    fn build_body_uses_client_default_model() {
        let client = GeminiChatClient::new("key", "gemini-x");
        let (_, model) = client.build_body(&[Message::user("hi")], &ChatOptions::new());
        assert_eq!(model, "gemini-x");
    }

    #[test]
    fn build_body_per_request_model_overrides_client_default() {
        let client = GeminiChatClient::new("key", "gemini-x");
        let options = ChatOptions::new().with_model("gemini-y");
        let (_, model) = client.build_body(&[Message::user("hi")], &options);
        assert_eq!(model, "gemini-y");
    }

    #[test]
    fn build_body_with_default_options_merged_under_per_request_options() {
        let client = GeminiChatClient::new("key", "gemini-x")
            .with_default_options(ChatOptions::new().with_temperature(0.2));
        let (body, _) = client.build_body(&[Message::user("hi")], &ChatOptions::new());
        assert_eq!(
            body["generationConfig"]["temperature"],
            serde_json::json!(0.2_f32)
        );

        let (body2, _) = client.build_body(
            &[Message::user("hi")],
            &ChatOptions::new().with_temperature(0.9),
        );
        assert_eq!(
            body2["generationConfig"]["temperature"],
            serde_json::json!(0.9_f32)
        );
    }

    // region: classify_gemini_error

    #[test]
    fn classifies_401_and_403_as_invalid_auth() {
        for status in [401, 403] {
            let body = format!(
                r#"{{"error":{{"code":{status},"message":"nope","status":"UNAUTHENTICATED"}}}}"#
            );
            let err = classify_gemini_error(status, &body, format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidAuth { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_400_invalid_argument_as_invalid_request() {
        let body = r#"{"error":{"code":400,"message":"bad field","status":"INVALID_ARGUMENT"}}"#;
        let err = classify_gemini_error(400, body, "err", None);
        assert!(
            matches!(err, Error::ServiceInvalidRequest { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn a_400_without_confirming_body_stays_service_status() {
        let err = classify_gemini_error(400, "not json", "err", None);
        assert_eq!(err.status(), Some(400), "{err:?}");

        let err = classify_gemini_error(
            400,
            r#"{"error":{"code":400,"status":"FAILED_PRECONDITION"}}"#,
            "err",
            None,
        );
        assert_eq!(err.status(), Some(400), "{err:?}");
    }

    #[test]
    fn leaves_retryable_statuses_as_service_status() {
        for status in [429, 500, 503] {
            let err = classify_gemini_error(status, "", format!("err {status}"), Some(1.5));
            assert_eq!(err.status(), Some(status), "{err:?}");
            assert_eq!(err.retry_after(), Some(1.5), "{err:?}");
        }
    }

    #[test]
    fn never_produces_content_filter() {
        // Gemini has no content-filter-specific HTTP error: a blocked prompt
        // is a 200 with `promptFeedback.blockReason` (see
        // `convert::parse_response`'s test coverage), never a non-success
        // status.
        let bodies = [
            "",
            "not json",
            r#"{"error":{"status":"INVALID_ARGUMENT"}}"#,
            r#"{"error":{"status":"UNAUTHENTICATED"}}"#,
        ];
        for status in [400, 401, 403, 404, 429, 500] {
            for body in bodies {
                let err = classify_gemini_error(status, body, "err", None);
                assert!(
                    !matches!(err, Error::ServiceContentFilter { .. }),
                    "status {status}, body {body:?}: {err:?}"
                );
            }
        }
    }

    // endregion
}
