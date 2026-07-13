//! Mistral embeddings client.
//!
//! Rust equivalent of upstream's `MistralEmbeddingClient`
//! (`agent_framework_mistral/_embedding_client.py`):
//! `POST {base_url}/embeddings` (`https://api.mistral.ai/v1` by default),
//! OpenAI-shaped request/response, default model `mistral-embed`. Response
//! parsing is shared with `agent-framework-openai`.

use std::sync::Arc;

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{EmbeddingGenerationOptions, GeneratedEmbeddings};
use serde_json::{json, Map, Value};

use crate::DEFAULT_BASE_URL;

/// Upstream's default Mistral embedding model.
pub const DEFAULT_EMBEDDING_MODEL: &str = "mistral-embed";

/// A Mistral embeddings client (`POST {base_url}/embeddings`).
///
/// ```no_run
/// # use agent_framework_mistral::MistralEmbeddingClient;
/// # use agent_framework_core::client::EmbeddingClient;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = MistralEmbeddingClient::from_env(None)?; // mistral-embed
/// let batch = client.get_embeddings(vec!["hello".into()], None).await?;
/// println!("{} dims", batch[0].dimensions());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct MistralEmbeddingClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

impl std::fmt::Debug for MistralEmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MistralEmbeddingClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .finish_non_exhaustive()
    }
}

impl MistralEmbeddingClient {
    /// Create a client for the given API key and embedding model
    /// (`None` → [`DEFAULT_EMBEDDING_MODEL`]).
    pub fn new(api_key: impl Into<String>, model: Option<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.unwrap_or_else(|| DEFAULT_EMBEDDING_MODEL.to_string()),
            }),
        }
    }

    /// Build a client from the `MISTRAL_API_KEY` (and optional
    /// `MISTRAL_BASE_URL`) environment variables.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let key = std::env::var("MISTRAL_API_KEY")
            .map_err(|_| Error::Configuration("MISTRAL_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("MISTRAL_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for MistralEmbeddingClient {
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
        // Mistral spells requested dimensionality `output_dimension`
        // (upstream maps options["dimensions"] onto it).
        if let Some(dimensions) = options.as_ref().and_then(|o| o.dimensions) {
            body.insert("output_dimension".into(), json!(dimensions));
        }
        if let Some(dtype) = options
            .as_ref()
            .and_then(|o| o.additional_properties.get("output_dtype"))
        {
            body.insert("output_dtype".into(), dtype.clone());
        }
        let url = format!("{}/embeddings", self.inner.base_url.trim_end_matches('/'));
        let resp = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&self.inner.api_key)
            .json(&Value::Object(body))
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(Error::service_status(
                status.as_u16(),
                format!("Mistral API error {status}: {text}"),
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
