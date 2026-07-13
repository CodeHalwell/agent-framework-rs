//! # agent-framework-mistral
//!
//! A [Mistral AI](https://mistral.ai) [`ChatClient`]
//! for `agent-framework-rs`.
//!
//! Talks to the Mistral Chat Completions API (`POST /v1/chat/completions`),
//! whose wire format is OpenAI-Chat-compatible: message shape, function-tool
//! shape, tool-call/response shape, and usage shape all match
//! [`agent-framework-openai`](agent_framework_openai)'s Chat Completions
//! client, so this crate reuses that conversion code directly (the same
//! approach `agent-framework-azure` takes for Azure OpenAI) rather than
//! duplicating it. Only the parts that genuinely differ — the base URL,
//! bearer-token auth, the supported request-option set (see
//! [`convert::apply_options`]), and error classification — are implemented
//! here; see [`convert`] for the details.
//!
//! Upstream (the Python/.NET `agent-framework`) ships no dedicated Mistral
//! chat connector at all — Mistral only appears there as an
//! embeddings/text-embedding provider. This crate instead provides a full
//! Mistral **chat** client, since Mistral's hosted models are chat models
//! first and foremost and the framework's [`ChatClient`] trait is the
//! natural fit. It can gain an embeddings client of its own once
//! `agent-framework-core` grows a shared embeddings trait to implement
//! against; until then, chat is the complete scope of this crate.
//!
//! ```no_run
//! use agent_framework_mistral::MistralChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = MistralChatClient::new("...", "mistral-large-latest");
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

pub mod convert;

use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{ChatOptions, ChatResponse, Message};
use futures::StreamExt;
use serde_json::Value;

const DEFAULT_BASE_URL: &str = "https://api.mistral.ai/v1";

/// Parse a `Retry-After` header into a delay in seconds.
///
/// Mistral returns the integer/decimal-seconds form on `429`, mirroring
/// OpenAI/Anthropic; a date-form or unparseable value is treated as absent.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// A Mistral AI chat client (`POST {base_url}/chat/completions`).
#[derive(Clone)]
pub struct MistralChatClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for MistralChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralChatClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish_non_exhaustive()
    }
}

impl MistralChatClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
            }),
        }
    }

    /// Build a client from the `MISTRAL_API_KEY` (and optional
    /// `MISTRAL_BASE_URL`) environment variables.
    ///
    /// Unlike Ollama (which runs unauthenticated by default and so has no
    /// API-key environment variable at all), Mistral's hosted API always
    /// requires a bearer token, so `MISTRAL_API_KEY` is required here.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| Error::Configuration("MISTRAL_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("MISTRAL_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for proxies or private deployments).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    fn url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.inner.base_url.trim_end_matches('/')
        )
    }

    fn build_body(&self, messages: &[Message], options: &ChatOptions, stream: bool) -> Value {
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        convert::build_request(messages, options, &model, stream)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let resp = self
            .inner
            .http
            .post(self.url())
            .bearer_auth(&self.inner.api_key)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(convert::classify_mistral_error(
                status.as_u16(),
                format!("Mistral API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for MistralChatClient {
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
        // Mistral's streaming Chat Completions wire shape (SSE
        // `chat.completion.chunk` objects, `[DONE]` terminator, a trailing
        // usage-only chunk when `stream_options.include_usage` is set, and
        // mid-stream `{"error": {...}}` objects) is identical to OpenAI's, so
        // parsing is reused verbatim rather than duplicated.
        Ok(agent_framework_openai::parse_sse_stream(resp).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> MistralChatClient {
        MistralChatClient::new("test-key", "mistral-large-latest")
    }

    // region: URL building

    #[test]
    fn url_uses_default_base_url() {
        let c = client();
        assert_eq!(c.url(), "https://api.mistral.ai/v1/chat/completions");
    }

    #[test]
    fn with_base_url_overrides_default_and_trims_trailing_slash() {
        let c = client().with_base_url("https://proxy.example.com/v1/");
        assert_eq!(c.url(), "https://proxy.example.com/v1/chat/completions");
    }

    // endregion

    // region: request body building

    #[test]
    fn build_body_defaults_to_client_model() {
        let c = client();
        let body = c.build_body(&[Message::user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["model"], serde_json::json!("mistral-large-latest"));
    }

    #[test]
    fn build_body_per_request_model_overrides_client_default() {
        let c = client();
        let options = ChatOptions::new().with_model("mistral-small-latest");
        let body = c.build_body(&[Message::user("hi")], &options, false);
        assert_eq!(body["model"], serde_json::json!("mistral-small-latest"));
    }

    #[test]
    fn build_body_stream_flag() {
        let c = client();
        let body = c.build_body(&[Message::user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], serde_json::json!(true));
    }

    // endregion

    // Streaming: `get_streaming_response` delegates directly to
    // `agent_framework_openai::parse_sse_stream`, with no Mistral-specific
    // logic of its own -- the wiring is verified at compile time (this crate
    // wouldn't type-check against `ChatStream` otherwise), and SSE chunk
    // parsing itself (text deltas, tool-call argument accumulation, `[DONE]`
    // handling, trailing usage chunk, mid-stream error surfacing) is already
    // covered by `agent-framework-openai`'s own test suite, which this crate
    // depends on unchanged. `reqwest::Response` can't be constructed from raw
    // bytes outside an actual HTTP exchange, so reproducing those fixtures
    // here would require standing up a mock server rather than a plain unit
    // test (`agent-framework-azure`, which reuses the very same function,
    // documents this identically).

    // region: env-var constructor

    /// Guards `MISTRAL_API_KEY` / `MISTRAL_BASE_URL` mutation: tests within a
    /// crate run on multiple threads, and env vars are process-global, so
    /// this serializes access across the tests below.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::set_var("MISTRAL_API_KEY", "test-key-123");
            std::env::set_var("MISTRAL_BASE_URL", "https://example.test/v1");
        }
        let client = MistralChatClient::from_env("mistral-large-latest").unwrap();
        assert_eq!(client.inner.api_key, "test-key-123");
        assert_eq!(client.inner.base_url, "https://example.test/v1");
        unsafe {
            std::env::remove_var("MISTRAL_API_KEY");
            std::env::remove_var("MISTRAL_BASE_URL");
        }
    }

    #[test]
    fn from_env_errors_when_api_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("MISTRAL_API_KEY");
            std::env::remove_var("MISTRAL_BASE_URL");
        }
        let result = MistralChatClient::from_env("mistral-large-latest");
        assert!(result.is_err());
    }

    #[test]
    fn from_env_defaults_base_url_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var("MISTRAL_API_KEY", "test-key-123");
            std::env::remove_var("MISTRAL_BASE_URL");
        }
        let client = MistralChatClient::from_env("mistral-large-latest").unwrap();
        assert_eq!(client.inner.base_url, DEFAULT_BASE_URL);
        unsafe {
            std::env::remove_var("MISTRAL_API_KEY");
        }
    }

    // endregion

    #[test]
    fn model_returns_default_model() {
        let c = client();
        assert_eq!(c.model(), "mistral-large-latest");
        assert_eq!(ChatClient::model(&c), Some("mistral-large-latest"));
    }
}
