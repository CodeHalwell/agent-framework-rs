//! # agent-framework-ollama
//!
//! An [Ollama](https://ollama.com) [`ChatClient`]
//! for `agent-framework-rs`.
//!
//! Talks to a local (or remote) Ollama server's OpenAI-compatible endpoint
//! (`POST {base_url}/chat/completions`, default base URL
//! `http://localhost:11434/v1`) — see
//! <https://github.com/ollama/ollama/blob/main/docs/openai.md>. That
//! compatibility layer speaks the exact same JSON shapes as OpenAI's Chat
//! Completions API, so request/response conversion is reused from
//! [`agent_framework_openai::convert`] rather than duplicated (see
//! [`convert`] for the small, Ollama-specific streaming-delta parser that
//! *is* implemented locally). Unlike OpenAI or Anthropic, Ollama's server
//! normally requires no API key at all — a plain local install answers
//! unauthenticated requests — so [`OllamaChatClient::new`] and
//! [`OllamaChatClient::from_env`] don't require one either.
//!
//! ```no_run
//! use agent_framework_ollama::OllamaChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OllamaChatClient::new("llama3.1");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Pointing at a non-default (e.g. remote or containerized) server:
//!
//! ```no_run
//! use agent_framework_ollama::OllamaChatClient;
//!
//! let client = OllamaChatClient::new("llama3.1").with_base_url("http://my-ollama-host:11434/v1");
//! # let _ = client;
//! ```

pub mod convert;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
use futures::StreamExt;
use serde_json::Value;

/// The default Ollama OpenAI-compatible base URL for a local install.
const DEFAULT_BASE_URL: &str = "http://localhost:11434/v1";

/// The environment variable Ollama's own tooling reads for the server
/// address (e.g. `127.0.0.1:11434`, optionally with a scheme). If set,
/// [`OllamaChatClient::from_env`] derives the OpenAI-compatible base URL from
/// it; there is no API-key environment variable to require, since a stock
/// Ollama server accepts unauthenticated requests.
const OLLAMA_HOST_ENV: &str = "OLLAMA_HOST";

/// Turn an `OLLAMA_HOST` value into the OpenAI-compatible base URL Ollama
/// serves under. `OLLAMA_HOST` (as read by Ollama's own CLI/server) is
/// typically a bare `host:port` with no scheme (e.g. `127.0.0.1:11434`), but
/// a full URL is tolerated too. Either way this appends the compatibility
/// layer's `/v1` suffix (stripping any the caller already included, and any
/// trailing slash, so the result never doubles up).
fn base_url_from_host(host: &str) -> String {
    let host = host.trim().trim_end_matches('/');
    let with_scheme = if host.contains("://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let with_scheme = with_scheme.trim_end_matches("/v1").trim_end_matches('/');
    format!("{with_scheme}/v1")
}

/// Parse a `Retry-After` header into a delay in seconds. Mirrors the
/// OpenAI/Anthropic/Azure clients (Ollama's compatibility layer doesn't
/// generally emit this, but a proxy in front of it might).
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// An Ollama chat client (`POST {base_url}/chat/completions`).
#[derive(Clone)]
pub struct OllamaChatClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    base_url: String,
    model: String,
    /// Optional bearer token. Unset by default — a stock Ollama server needs
    /// none — but some deployments sit behind a proxy that requires one.
    api_key: Option<String>,
}

impl std::fmt::Debug for OllamaChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaChatClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .field("has_api_key", &self.inner.api_key.is_some())
            .finish_non_exhaustive()
    }
}

impl OllamaChatClient {
    /// Create a client for the given default model, targeting
    /// `http://localhost:11434/v1`.
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

    /// Build a client from the environment. Ollama has no required
    /// credential env var (a stock server is unauthenticated), so this never
    /// fails on missing configuration; it only reads the optional
    /// `OLLAMA_HOST_ENV` (`OLLAMA_HOST`) to override the default base URL,
    /// same as Ollama's own CLI/SDKs.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut client = Self::new(model);
        if let Ok(host) = std::env::var(OLLAMA_HOST_ENV) {
            if !host.trim().is_empty() {
                client = client.with_base_url(base_url_from_host(&host));
            }
        }
        Ok(client)
    }

    /// Override the base URL (for a remote/containerized server, or a
    /// non-default port). Must be the OpenAI-compatible root (i.e. include
    /// the `/v1` suffix), matching [`OllamaChatClient::new`]'s default.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set a bearer token sent as `Authorization: Bearer <key>`. Not needed
    /// for a stock local Ollama server; useful when one sits behind an
    /// authenticating proxy.
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
            // Ollama's OpenAI-compatibility layer is wire-compatible for
            // errors too, so status/body classification is shared verbatim
            // with `agent-framework-openai` rather than duplicated.
            return Err(agent_framework_openai::classify_service_error(
                status.as_u16(),
                &text,
                format!("Ollama API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for OllamaChatClient {
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

    // region: base URL derivation

    #[test]
    fn default_base_url_is_localhost_v1() {
        let client = OllamaChatClient::new("llama3.1");
        assert_eq!(client.base_url(), "http://localhost:11434/v1");
    }

    #[test]
    fn with_base_url_overrides_default() {
        let client =
            OllamaChatClient::new("llama3.1").with_base_url("http://example.test:11434/v1");
        assert_eq!(client.base_url(), "http://example.test:11434/v1");
    }

    #[test]
    fn base_url_from_bare_host_port_adds_scheme_and_v1() {
        assert_eq!(
            base_url_from_host("127.0.0.1:11434"),
            "http://127.0.0.1:11434/v1"
        );
    }

    #[test]
    fn base_url_from_full_url_is_normalized() {
        assert_eq!(
            base_url_from_host("http://my-host:11434"),
            "http://my-host:11434/v1"
        );
        // Already-suffixed and trailing-slash forms don't double up.
        assert_eq!(
            base_url_from_host("http://my-host:11434/v1/"),
            "http://my-host:11434/v1"
        );
        assert_eq!(
            base_url_from_host("https://my-host:11434"),
            "https://my-host:11434/v1"
        );
    }

    // endregion

    // region: from_env

    /// Guards `OLLAMA_HOST` mutation: tests within a crate run on multiple
    /// threads, and env vars are process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_never_fails_without_ollama_host() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches this variable.
        unsafe {
            std::env::remove_var(OLLAMA_HOST_ENV);
        }
        let client = OllamaChatClient::from_env("llama3.1").unwrap();
        assert_eq!(client.base_url(), DEFAULT_BASE_URL);
        assert_eq!(client.model(), "llama3.1");
    }

    #[test]
    fn from_env_reads_ollama_host() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var(OLLAMA_HOST_ENV, "192.168.1.50:11434");
        }
        let client = OllamaChatClient::from_env("llama3.1").unwrap();
        assert_eq!(client.base_url(), "http://192.168.1.50:11434/v1");
        unsafe {
            std::env::remove_var(OLLAMA_HOST_ENV);
        }
    }

    // endregion

    // region: request building

    #[test]
    fn build_body_uses_client_default_model_when_options_model_unset() {
        let client = OllamaChatClient::new("llama3.1");
        let body = client.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["model"], serde_json::json!("llama3.1"));
    }

    #[test]
    fn build_body_prefers_per_request_model() {
        let client = OllamaChatClient::new("llama3.1");
        let options = ChatOptions {
            model: Some("mistral".to_string()),
            ..ChatOptions::new()
        };
        let body = client.build_body(&[Message::user("hi")], &options, false);
        assert_eq!(body["model"], serde_json::json!("mistral"));
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
            "model": "llama3.1",
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
}
