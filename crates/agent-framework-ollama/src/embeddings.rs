//! Ollama embeddings client.
//!
//! Rust counterpart of upstream's `OllamaEmbeddingClient`
//! (`agent_framework_ollama/_embedding_client.py`). Upstream drives Ollama's
//! native `/api/embed` endpoint via the `ollama` SDK; this crate — like its
//! chat client — speaks Ollama's OpenAI-compatible surface instead
//! (`POST {base_url}/embeddings`, default base URL
//! `http://localhost:11434/v1`), so request extras and response parsing are
//! shared with `agent-framework-openai`. No API key is required.

use std::sync::Arc;

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{EmbeddingGenerationOptions, GeneratedEmbeddings};
use serde_json::{json, Map, Value};

use crate::{base_url_from_host, DEFAULT_BASE_URL, OLLAMA_HOST_ENV};

/// An Ollama embeddings client (`POST {base_url}/embeddings`).
///
/// ```no_run
/// # use agent_framework_ollama::OllamaEmbeddingClient;
/// # use agent_framework_core::client::EmbeddingClient;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = OllamaEmbeddingClient::new("nomic-embed-text");
/// let batch = client.get_embeddings(vec!["hello".into()], None).await?;
/// println!("{} dims", batch[0].dimensions());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OllamaEmbeddingClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for OllamaEmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OllamaEmbeddingClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish()
    }
}

impl OllamaEmbeddingClient {
    /// Create a client for the given embedding model against the default
    /// local server (`http://localhost:11434/v1`).
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
            }),
        }
    }

    /// Build a client from the environment: `OLLAMA_HOST` (optional)
    /// overrides the default base URL, interpreted exactly as the chat
    /// client's `from_env` does.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let mut client = Self::new(model);
        if let Ok(host) = std::env::var(OLLAMA_HOST_ENV) {
            if !host.trim().is_empty() {
                client = client.with_base_url(base_url_from_host(&host));
            }
        }
        Ok(client)
    }

    /// Override the base URL (must include the `/v1` OpenAI-compatible
    /// prefix, e.g. `http://my-host:11434/v1`).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for OllamaEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        let mut body = Map::new();
        let model = options
            .as_ref()
            .and_then(|o| o.model.clone())
            .unwrap_or_else(|| self.inner.model.clone());
        body.insert("model".into(), json!(model));
        body.insert("input".into(), json!(values));
        if let Some(dimensions) = options.as_ref().and_then(|o| o.dimensions) {
            body.insert("dimensions".into(), json!(dimensions));
        }
        let url = format!("{}/embeddings", self.inner.base_url.trim_end_matches('/'));
        let resp = self
            .inner
            .http
            .post(&url)
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service_status(
                status.as_u16(),
                format!("Ollama API error {status}: {text}"),
                None,
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        agent_framework_openai::embeddings::parse_embeddings_response(&value)
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}
