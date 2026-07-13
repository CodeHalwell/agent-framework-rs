//! # agent-framework-foundry-local
//!
//! A [Microsoft Foundry Local](https://learn.microsoft.com/en-us/azure/ai-foundry/foundry-local/what-is-foundry-local)
//! [`ChatClient`] for `agent-framework-rs`.
//!
//! Foundry Local runs models on-device and exposes them through an
//! OpenAI-compatible REST endpoint (`POST {base_url}/chat/completions`,
//! default base URL `http://localhost:5273/v1`). That compatibility layer
//! speaks the exact same JSON shapes as OpenAI's Chat Completions API, so
//! request/response conversion is reused from
//! [`agent_framework_openai::convert`] rather than duplicated — mirroring how
//! `agent-framework-ollama` reuses it for the same reason. Like Ollama, a
//! stock local Foundry Local instance normally requires no API key, so
//! [`FoundryLocalChatClient::new`] and [`FoundryLocalChatClient::from_env`]
//! don't require one either.
//!
//! ```no_run
//! use agent_framework_foundry_local::FoundryLocalChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = FoundryLocalChatClient::new("phi-3.5-mini");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Pointing at a non-default port (Foundry Local can bind to a
//! dynamically-chosen one):
//!
//! ```no_run
//! use agent_framework_foundry_local::FoundryLocalChatClient;
//!
//! let client =
//!     FoundryLocalChatClient::new("phi-3.5-mini").with_base_url("http://localhost:5273/v1");
//! # let _ = client;
//! ```
//!
//! The actual port Foundry Local's OpenAI-compatible endpoint listens on is
//! discovered dynamically from the Foundry Local service in real
//! deployments (it varies by install and can change across restarts); this
//! crate does not perform that discovery itself. Point
//! [`FoundryLocalChatClient`] at the right base URL with
//! [`FoundryLocalChatClient::with_base_url`] or the `FOUNDRY_LOCAL_ENDPOINT_ENV`
//! (`FOUNDRY_LOCAL_ENDPOINT`) environment variable, using whatever port the
//! Foundry Local SDK/CLI reports (`5273` above is just the common default).

pub mod convert;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
use futures::StreamExt;
use serde_json::Value;

/// The default Foundry Local OpenAI-compatible base URL.
const DEFAULT_BASE_URL: &str = "http://localhost:5273/v1";

/// The environment variable read for a non-default Foundry Local endpoint.
/// If set, [`FoundryLocalChatClient::from_env`] uses it as the base URL.
const FOUNDRY_LOCAL_ENDPOINT_ENV: &str = "FOUNDRY_LOCAL_ENDPOINT";

/// The environment variable read for an optional bearer token. Not required
/// for a stock local Foundry Local instance; useful when one sits behind an
/// authenticating proxy.
const FOUNDRY_LOCAL_API_KEY_ENV: &str = "FOUNDRY_LOCAL_API_KEY";

/// Parse a `Retry-After` header into a delay in seconds. Mirrors the
/// OpenAI/Anthropic/Azure/Ollama clients (Foundry Local's compatibility layer
/// doesn't generally emit this, but a proxy in front of it might).
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// A Microsoft Foundry Local chat client (`POST {base_url}/chat/completions`).
#[derive(Clone)]
pub struct FoundryLocalChatClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    base_url: String,
    model: String,
    /// Optional bearer token. Unset by default — a stock Foundry Local
    /// instance needs none — but some deployments sit behind a proxy that
    /// requires one.
    api_key: Option<String>,
}

impl std::fmt::Debug for FoundryLocalChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryLocalChatClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .field("has_api_key", &self.inner.api_key.is_some())
            .finish_non_exhaustive()
    }
}

impl FoundryLocalChatClient {
    /// Create a client for the given default model, targeting
    /// `http://localhost:5273/v1`.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                api_key: None,
            }),
        }
    }

    /// Build a client from the environment. Foundry Local has no required
    /// credential env var (a stock instance is unauthenticated), so this
    /// never fails on missing configuration; it only reads the optional
    /// `FOUNDRY_LOCAL_ENDPOINT_ENV` (`FOUNDRY_LOCAL_ENDPOINT`) to override
    /// the default base URL and `FOUNDRY_LOCAL_API_KEY_ENV`
    /// (`FOUNDRY_LOCAL_API_KEY`) to set a bearer token, if present.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut client = Self::new(model);
        if let Ok(endpoint) = std::env::var(FOUNDRY_LOCAL_ENDPOINT_ENV) {
            if !endpoint.trim().is_empty() {
                client = client.with_base_url(endpoint);
            }
        }
        if let Ok(api_key) = std::env::var(FOUNDRY_LOCAL_API_KEY_ENV) {
            if !api_key.trim().is_empty() {
                client = client.with_api_key(api_key);
            }
        }
        Ok(client)
    }

    /// Override the base URL (for a non-default port, or a remote Foundry
    /// Local host). Must be the OpenAI-compatible root (i.e. include the
    /// `/v1` suffix), matching [`FoundryLocalChatClient::new`]'s default.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set a bearer token sent as `Authorization: Bearer <key>`. Not needed
    /// for a stock local Foundry Local instance; useful when one sits behind
    /// an authenticating proxy.
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).api_key = Some(api_key.into());
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The configured base URL.
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    fn build_body(&self, messages: &[Message], options: &ChatOptions, stream: bool) -> Value {
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        convert::build_request(messages, options, &model, stream)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!(
            "{}/chat/completions",
            self.inner.base_url.trim_end_matches('/')
        );
        let mut req = self.inner.http.post(&url).json(body);
        if let Some(key) = &self.inner.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            // Foundry Local's OpenAI-compatibility layer is wire-compatible
            // for errors too, so status/body classification is shared
            // verbatim with `agent-framework-openai` rather than duplicated.
            return Err(agent_framework_openai::classify_service_error(
                status.as_u16(),
                &text,
                format!("Foundry Local API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for FoundryLocalChatClient {
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
        Ok(agent_framework_openai::convert::parse_response(&value))
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
                                if data.is_empty() {
                                    continue;
                                }
                                if let Ok(value) = serde_json::from_str::<Value>(data) {
                                    if let Some(err) = value.get("error") {
                                        let msg = err
                                            .get("message")
                                            .and_then(Value::as_str)
                                            .unwrap_or("unknown stream error")
                                            .to_string();
                                        state.done = true;
                                        return Some((Err(Error::service(msg)), state));
                                    }
                                    if let Some(update) =
                                        convert::parse_delta(&value, &mut state.tool_ids)
                                    {
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

#[cfg(test)]
mod tests {
    use super::*;

    // region: base URL and constructor defaults

    #[test]
    fn default_base_url_is_localhost_v1() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini");
        assert_eq!(client.base_url(), "http://localhost:5273/v1");
        assert_eq!(client.model(), "phi-3.5-mini");
    }

    #[test]
    fn with_base_url_overrides_default() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini")
            .with_base_url("http://example.test:5273/v1");
        assert_eq!(client.base_url(), "http://example.test:5273/v1");
    }

    #[test]
    fn with_api_key_sets_bearer_token() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini").with_api_key("secret");
        assert_eq!(client.inner.api_key.as_deref(), Some("secret"));
    }

    #[test]
    fn no_api_key_by_default() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini");
        assert!(client.inner.api_key.is_none());
    }

    // endregion

    // region: from_env

    /// Guards env mutation: tests within a crate run on multiple threads, and
    /// env vars are process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_never_fails_without_env_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::remove_var(FOUNDRY_LOCAL_ENDPOINT_ENV);
            std::env::remove_var(FOUNDRY_LOCAL_API_KEY_ENV);
        }
        let client = FoundryLocalChatClient::from_env("phi-3.5-mini").unwrap();
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        assert_eq!(client.model(), "phi-3.5-mini");
        assert!(client.inner.api_key.is_none());
    }

    #[test]
    fn from_env_reads_endpoint_and_api_key() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var(FOUNDRY_LOCAL_ENDPOINT_ENV, "http://192.168.1.50:5273/v1");
            std::env::set_var(FOUNDRY_LOCAL_API_KEY_ENV, "secret");
        }
        let client = FoundryLocalChatClient::from_env("phi-3.5-mini").unwrap();
        assert_eq!(client.base_url(), "http://192.168.1.50:5273/v1");
        assert_eq!(client.inner.api_key.as_deref(), Some("secret"));
        unsafe {
            std::env::remove_var(FOUNDRY_LOCAL_ENDPOINT_ENV);
            std::env::remove_var(FOUNDRY_LOCAL_API_KEY_ENV);
        }
    }

    // endregion

    // region: request building

    #[test]
    fn build_body_uses_client_default_model_when_options_model_unset() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini");
        let body = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["model"], serde_json::json!("phi-3.5-mini"));
        assert_eq!(
            body["messages"],
            serde_json::json!([{ "role": "user", "content": "hi" }])
        );
    }

    #[test]
    fn build_body_prefers_per_request_model() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini");
        let options = ChatOptions {
            model: Some("qwen2.5".to_string()),
            ..ChatOptions::new()
        };
        let body = client.build_body(&[Message::user("hi")], &options, false);
        assert_eq!(body["model"], serde_json::json!("qwen2.5"));
    }

    #[test]
    fn build_body_sets_stream_flag() {
        let client = FoundryLocalChatClient::new("phi-3.5-mini");
        let body = client.build_body(&[Message::user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    // endregion

    // region: SSE stream parsing over a synthetic byte stream

    fn sse_bytes(lines: &[String]) -> bytes::Bytes {
        bytes::Bytes::from(lines.join("\n") + "\n\n")
    }

    async fn collect_via_state(text: bytes::Bytes) -> Vec<ChatResponseUpdate> {
        let stream = futures::stream::once(async move { Ok::<_, reqwest::Error>(text) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = SseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            tool_ids: HashMap::new(),
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
                if data.is_empty() || data == "[DONE]" {
                    continue;
                }
                let value: Value = serde_json::from_str(data).unwrap();
                if let Some(update) = convert::parse_delta(&value, &mut state.tool_ids) {
                    updates.push(update);
                }
            }
        }
        updates
    }

    #[tokio::test]
    async fn streaming_chunk_produces_text_update() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "phi-3.5-mini",
            "choices": [{ "delta": { "role": "assistant", "content": "Hello" }, "finish_reason": null }],
        });
        let bytes = sse_bytes(&[format!("data: {chunk}")]);
        let updates = collect_via_state(bytes).await;
        assert_eq!(updates.len(), 1);
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello");
        assert_eq!(resp.response_id.as_deref(), Some("chatcmpl-1"));
    }

    #[tokio::test]
    async fn streaming_tool_call_and_finish_reason_accumulate() {
        let call_chunk = serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "id": "call_1", "function": { "name": "get_weather", "arguments": "{}" } }] },
                "finish_reason": null,
            }],
        });
        let finish_chunk = serde_json::json!({
            "id": "chatcmpl-2",
            "choices": [{ "delta": {}, "finish_reason": "tool_calls" }],
        });
        let bytes = sse_bytes(&[
            format!("data: {call_chunk}"),
            format!("data: {finish_chunk}"),
        ]);
        let updates = collect_via_state(bytes).await;
        assert_eq!(updates.len(), 2);
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::tool_calls())
        );
    }

    #[tokio::test]
    async fn streaming_done_sentinel_ends_stream_without_extra_update() {
        let chunk = serde_json::json!({
            "id": "chatcmpl-3",
            "choices": [{ "delta": { "content": "hi" }, "finish_reason": null }],
        });
        let bytes = sse_bytes(&[format!("data: {chunk}"), "data: [DONE]".to_string()]);
        let updates = collect_via_state(bytes).await;
        // `collect_via_state` mirrors the raw line loop (no special-casing of
        // [DONE] beyond `continue`), so this asserts the sentinel line never
        // parses into a spurious update, matching the real stream (which
        // instead terminates on it).
        assert_eq!(updates.len(), 1);
    }

    // endregion
}
