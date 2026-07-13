//! OpenAI embeddings client.
//!
//! Rust equivalent of upstream's `OpenAIEmbeddingClient`
//! (`agent_framework_openai/_embedding_client.py`): the
//! [`POST /v1/embeddings`](https://platform.openai.com/docs/api-reference/embeddings)
//! endpoint, batching all input values into one request. Works against any
//! OpenAI-compatible server via [`OpenAIEmbeddingClient::with_base_url`].

use std::sync::Arc;

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings, UsageDetails,
};
use serde_json::{json, Map, Value};

use crate::{classify_service_error, parse_retry_after, DEFAULT_BASE_URL};

/// An OpenAI (or OpenAI-compatible) embeddings client.
///
/// ```no_run
/// # use agent_framework_openai::OpenAIEmbeddingClient;
/// # use agent_framework_core::client::EmbeddingClient;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = OpenAIEmbeddingClient::from_env("text-embedding-3-small")?;
/// let batch = client
///     .get_embeddings(vec!["Hello, world!".into()], None)
///     .await?;
/// println!("{} dims", batch[0].dimensions());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct OpenAIEmbeddingClient {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
}

impl OpenAIEmbeddingClient {
    /// Create a client for the given API key and default embedding model.
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

    /// Build a client from the environment: `OPENAI_API_KEY` (required) and
    /// optional `OPENAI_BASE_URL`. The model falls back through
    /// `OPENAI_EMBEDDING_MODEL` when the argument is empty — mirroring
    /// upstream's `model or OPENAI_EMBEDDING_MODEL` resolution.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("OPENAI_API_KEY is not set".into()))?;
        let mut model = model.into();
        if model.is_empty() {
            model = std::env::var("OPENAI_EMBEDDING_MODEL").map_err(|_| {
                Error::Configuration(
                    "no embedding model: pass one or set OPENAI_EMBEDDING_MODEL".into(),
                )
            })?;
        }
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for OpenAI-compatible servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).organization = Some(org.into());
        self
    }

    fn build_body(&self, values: &[String], options: Option<&EmbeddingGenerationOptions>) -> Value {
        let mut body = Map::new();
        let model = options
            .and_then(|o| o.model.clone())
            .unwrap_or_else(|| self.inner.model.clone());
        body.insert("model".into(), json!(model));
        body.insert("input".into(), json!(values));
        if let Some(options) = options {
            if let Some(dimensions) = options.dimensions {
                body.insert("dimensions".into(), json!(dimensions));
            }
            // Provider-specific extras understood by this endpoint
            // (`encoding_format`, `user`) are forwarded verbatim.
            for key in ["encoding_format", "user"] {
                if let Some(v) = options.additional_properties.get(key) {
                    body.insert(key.into(), v.clone());
                }
            }
        }
        Value::Object(body)
    }
}

/// `Arc::make_mut` over an `Inner` that is not `Clone`-derivable field-wise —
/// manual clone keeps the `reqwest::Client` (itself an `Arc` internally)
/// shared.
fn arc_inner(inner: &mut Arc<Inner>) -> &mut Inner {
    if Arc::strong_count(inner) != 1 {
        *inner = Arc::new(Inner {
            http: inner.http.clone(),
            api_key: inner.api_key.clone(),
            base_url: inner.base_url.clone(),
            model: inner.model.clone(),
            organization: inner.organization.clone(),
        });
    }
    Arc::get_mut(inner).expect("just ensured unique")
}

/// Parse an OpenAI-shaped embeddings response
/// (`{"data": [{"embedding": [...], "index": n}], "model": .., "usage": ..}`)
/// into a [`GeneratedEmbeddings`], restoring input order via `index`.
pub fn parse_embeddings_response(value: &Value) -> Result<GeneratedEmbeddings> {
    let model = value.get("model").and_then(Value::as_str);
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::service("embeddings response missing 'data' array"))?;

    let mut indexed: Vec<(usize, Embedding)> = Vec::with_capacity(data.len());
    for (position, item) in data.iter().enumerate() {
        let vector: Vec<f32> = item
            .get("embedding")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::service("embeddings item missing 'embedding' vector"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or_default() as f32)
            .collect();
        let index = item
            .get("index")
            .and_then(Value::as_u64)
            .map(|i| i as usize)
            .unwrap_or(position);
        indexed.push((
            index,
            Embedding {
                vector,
                model: model.map(String::from),
            },
        ));
    }
    indexed.sort_by_key(|(i, _)| *i);

    let mut batch = GeneratedEmbeddings::new(indexed.into_iter().map(|(_, e)| e).collect());
    if let Some(usage) = value.get("usage") {
        let input = usage.get("prompt_tokens").and_then(Value::as_u64);
        let total = usage.get("total_tokens").and_then(Value::as_u64);
        if input.is_some() || total.is_some() {
            batch.usage = Some(UsageDetails {
                input_token_count: input,
                total_token_count: total,
                ..Default::default()
            });
        }
    }
    Ok(batch)
}

#[async_trait::async_trait]
impl EmbeddingClient for OpenAIEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        let body = self.build_body(&values, options.as_ref());
        let url = format!("{}/embeddings", self.inner.base_url.trim_end_matches('/'));
        let mut req = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&self.inner.api_key)
            .json(&body);
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
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        parse_embeddings_response(&value)
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_includes_model_input_and_dimensions() {
        let client = OpenAIEmbeddingClient::new("sk-test", "text-embedding-3-small");
        let options = EmbeddingGenerationOptions::new().with_dimensions(256);
        let body = client.build_body(&["a".into(), "b".into()], Some(&options));
        assert_eq!(body["model"], "text-embedding-3-small");
        assert_eq!(body["input"], json!(["a", "b"]));
        assert_eq!(body["dimensions"], 256);
    }

    #[test]
    fn build_body_option_model_overrides_default() {
        let client = OpenAIEmbeddingClient::new("sk-test", "text-embedding-3-small");
        let options = EmbeddingGenerationOptions::new().with_model("text-embedding-3-large");
        let body = client.build_body(&["a".into()], Some(&options));
        assert_eq!(body["model"], "text-embedding-3-large");
    }

    #[test]
    fn build_body_forwards_known_additional_properties_only() {
        let client = OpenAIEmbeddingClient::new("sk-test", "m");
        let mut options = EmbeddingGenerationOptions::new();
        options
            .additional_properties
            .insert("encoding_format".into(), json!("float"));
        options
            .additional_properties
            .insert("unrelated".into(), json!(true));
        let body = client.build_body(&["a".into()], Some(&options));
        assert_eq!(body["encoding_format"], "float");
        assert!(body.get("unrelated").is_none());
    }

    #[test]
    fn parse_response_restores_index_order_and_usage() {
        let value = json!({
            "model": "text-embedding-3-small",
            "data": [
                { "index": 1, "embedding": [0.3, 0.4] },
                { "index": 0, "embedding": [0.1, 0.2] },
            ],
            "usage": { "prompt_tokens": 5, "total_tokens": 5 }
        });
        let batch = parse_embeddings_response(&value).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0].vector, vec![0.1, 0.2]);
        assert_eq!(batch[1].vector, vec![0.3, 0.4]);
        assert_eq!(batch[0].model.as_deref(), Some("text-embedding-3-small"));
        let usage = batch.usage.as_ref().unwrap();
        assert_eq!(usage.input_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(5));
    }

    #[test]
    fn parse_response_missing_data_errors() {
        assert!(parse_embeddings_response(&json!({})).is_err());
    }
}
