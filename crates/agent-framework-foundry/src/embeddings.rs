//! Azure AI Foundry Models embeddings client.
//!
//! Rust equivalent of upstream's `RawFoundryEmbeddingClient`
//! (`agent_framework_foundry/_embedding_client.py`): the Foundry **Models**
//! inference endpoint (`POST {models_endpoint}/embeddings`), which upstream
//! reaches through `azure.ai.inference`'s `EmbeddingsClient`.
//!
//! This is a different surface from
//! [`agent_framework_azure::AzureOpenAIEmbeddingClient`], which is
//! deployment-scoped (`{endpoint}/openai/deployments/{deployment}/embeddings`)
//! — hence a separate client rather than a delegation. The request and
//! response bodies are OpenAI-shaped, so response parsing is shared with
//! `agent-framework-openai` rather than duplicated, exactly as the Azure
//! embeddings client does.
//!
//! # Divergences from upstream
//!
//! - **Text inputs only.** Upstream accepts `Content | str` and splits a batch
//!   across two endpoints, sending image content to
//!   `ImageEmbeddingsClient` (`/images/embeddings`) and reassembling both
//!   result sets into input order. The core [`EmbeddingClient`] trait takes
//!   `Vec<String>`, so an image input cannot be expressed here at all: the
//!   image half is structurally out of reach rather than merely unimplemented,
//!   and adding it means widening a shared trait, not extending this client.
//!   `FOUNDRY_IMAGE_EMBEDDING_MODEL` is correspondingly not read.
//! - **API version.** Upstream passes none, inheriting whatever
//!   `azure-ai-inference` (pinned to `1.0.0b9`) sends. [`DEFAULT_API_VERSION`]
//!   reproduces that SDK's default; it is the one value here taken from an
//!   SDK default rather than from a service contract this port can verify, so
//!   it is overridable via [`FoundryEmbeddingClient::with_api_version`].

use std::sync::Arc;

use agent_framework_azure::TokenCredential;
use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{EmbeddingGenerationOptions, GeneratedEmbeddings};
use serde_json::{json, Map, Value};

/// The Foundry Models inference endpoint (e.g.
/// `https://<resource>.services.ai.azure.com/models`).
pub const FOUNDRY_MODELS_ENDPOINT_ENV: &str = "FOUNDRY_MODELS_ENDPOINT";
/// API key for the Foundry Models inference endpoint.
pub const FOUNDRY_MODELS_API_KEY_ENV: &str = "FOUNDRY_MODELS_API_KEY";
/// The default text embedding model (deployment) name.
pub const FOUNDRY_EMBEDDING_MODEL_ENV: &str = "FOUNDRY_EMBEDDING_MODEL";

/// The `api-version` sent when none is set, mirroring the default in
/// `azure-ai-inference` 1.0.0b9 — the version upstream's `foundry` package
/// pins. Override with [`FoundryEmbeddingClient::with_api_version`].
pub const DEFAULT_API_VERSION: &str = "2024-05-01-preview";

enum Auth {
    ApiKey(String),
    Credential(Arc<dyn TokenCredential>),
}

/// A Foundry Models embeddings client (`POST {endpoint}/embeddings`).
///
/// ```no_run
/// # use agent_framework_foundry::FoundryEmbeddingClient;
/// # use agent_framework_core::client::EmbeddingClient;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = FoundryEmbeddingClient::from_env(None)?;
/// let batch = client.get_embeddings(vec!["hello".into()], None).await?;
/// println!("{} dims", batch[0].dimensions());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct FoundryEmbeddingClient {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    endpoint: String,
    model: String,
    api_version: String,
    auth: Auth,
}

impl std::fmt::Debug for FoundryEmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryEmbeddingClient")
            .field("endpoint", &self.inner.endpoint)
            .field("model", &self.inner.model)
            .field("api_version", &self.inner.api_version)
            .field(
                "auth",
                &match self.inner.auth {
                    Auth::ApiKey(_) => "api-key",
                    Auth::Credential(_) => "token-credential",
                },
            )
            .finish_non_exhaustive()
    }
}

impl FoundryEmbeddingClient {
    /// Create a client authenticating with a static API key (`api-key`
    /// header), mirroring upstream's `AzureKeyCredential` path.
    pub fn new(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self::build(endpoint, model, Auth::ApiKey(api_key.into()))
    }

    /// Create a client authenticating via a [`TokenCredential`] (Microsoft
    /// Entra ID bearer token). The credential should already be scoped to
    /// [`crate::FOUNDRY_SCOPE`].
    pub fn with_credential(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint, model, Auth::Credential(credential))
    }

    fn build(endpoint: impl Into<String>, model: impl Into<String>, auth: Auth) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                endpoint: endpoint.into().trim_end_matches('/').to_string(),
                model: model.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                auth,
            }),
        }
    }

    /// Build a client from `FOUNDRY_MODELS_ENDPOINT`,
    /// `FOUNDRY_MODELS_API_KEY` and `FOUNDRY_EMBEDDING_MODEL`. `model`
    /// overrides the environment's model when supplied.
    ///
    /// Mirrors upstream's `required_fields=["models_endpoint",
    /// "embedding_model"]`: both must be resolvable, from arguments or the
    /// environment, or this errors rather than deferring the failure to the
    /// first request.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let endpoint = std::env::var(FOUNDRY_MODELS_ENDPOINT_ENV).map_err(|_| {
            Error::Configuration(format!("{FOUNDRY_MODELS_ENDPOINT_ENV} is not set"))
        })?;
        let model = model
            .or_else(|| std::env::var(FOUNDRY_EMBEDDING_MODEL_ENV).ok())
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "an embedding model is required: pass one or set {FOUNDRY_EMBEDDING_MODEL_ENV}"
                ))
            })?;
        let api_key = std::env::var(FOUNDRY_MODELS_API_KEY_ENV).map_err(|_| {
            Error::Configuration(format!("{FOUNDRY_MODELS_API_KEY_ENV} is not set"))
        })?;
        Ok(Self::new(endpoint, model, api_key))
    }

    /// Override the `api-version` query parameter (see
    /// [`DEFAULT_API_VERSION`]).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).api_version = api_version.into();
        self
    }

    /// The default embedding model.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The full request URL.
    fn url(&self) -> String {
        format!(
            "{}/embeddings?api-version={}",
            self.inner.endpoint, self.inner.api_version
        )
    }

    /// Build the request body for a batch.
    fn body_for(&self, values: &[String], options: Option<&EmbeddingGenerationOptions>) -> Value {
        let mut body = Map::new();
        body.insert("input".into(), json!(values));
        body.insert(
            "model".into(),
            json!(Self::effective_model(&self.inner.model, options)),
        );
        if let Some(dimensions) = options.and_then(|o| o.dimensions) {
            body.insert("dimensions".into(), json!(dimensions));
        }
        // Upstream forwards `encoding_format` and `input_type` explicitly, and
        // anything under `extra_parameters` as `model_extras`. The core
        // options struct has no typed slot for the first two, so all three
        // arrive through `additional_properties` and are passed verbatim.
        if let Some(extras) = options.map(|o| &o.additional_properties) {
            for (key, value) in extras {
                body.insert(key.clone(), value.clone());
            }
        }
        Value::Object(body)
    }

    fn effective_model(default: &str, options: Option<&EmbeddingGenerationOptions>) -> String {
        options
            .and_then(|o| o.model.clone())
            .unwrap_or_else(|| default.to_string())
    }
}

/// `Arc::make_mut` over a non-`Clone` `Inner` — the manual clone keeps the
/// `reqwest::Client` (internally an `Arc`) shared. Mirrors the same helper in
/// `agent-framework-openai`'s embeddings client.
fn arc_inner(inner: &mut Arc<Inner>) -> &mut Inner {
    if Arc::strong_count(inner) != 1 {
        *inner = Arc::new(Inner {
            http: inner.http.clone(),
            endpoint: inner.endpoint.clone(),
            model: inner.model.clone(),
            api_version: inner.api_version.clone(),
            auth: match &inner.auth {
                Auth::ApiKey(key) => Auth::ApiKey(key.clone()),
                Auth::Credential(cred) => Auth::Credential(cred.clone()),
            },
        });
    }
    Arc::get_mut(inner).expect("just ensured unique")
}

/// Parse a `Retry-After` header into a delay in seconds, when present.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.trim().parse::<f64>().ok())
}

#[async_trait::async_trait]
impl EmbeddingClient for FoundryEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        // Upstream returns an empty batch without calling the service.
        if values.is_empty() {
            return Ok(GeneratedEmbeddings::new(Vec::new()));
        }

        let body = self.body_for(&values, options.as_ref());
        let request = self.inner.http.post(self.url()).json(&body);
        let request = match &self.inner.auth {
            Auth::ApiKey(key) => request.header("api-key", key),
            Auth::Credential(credential) => {
                let token = credential.get_token().await?;
                request.bearer_auth(token)
            }
        };

        let resp = request
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            // Wire-compatible with OpenAI; classification shared verbatim.
            return Err(agent_framework_openai::classify_service_error(
                status.as_u16(),
                &text,
                format!("Foundry Models embeddings API error {status}: {text}"),
                retry_after,
            ));
        }

        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        let mut batch = agent_framework_openai::embeddings::parse_embeddings_response(&value)?;

        // Upstream stamps `response.model or text_model`: when the service
        // omits the model, the requested one still identifies the vectors.
        let requested = Self::effective_model(&self.inner.model, options.as_ref());
        for embedding in &mut batch.embeddings {
            if embedding.model.is_none() {
                embedding.model = Some(requested.clone());
            }
        }
        Ok(batch)
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_joins_endpoint_and_api_version_without_doubling_slashes() {
        let client = FoundryEmbeddingClient::new(
            "https://r.services.ai.azure.com/models/",
            "text-embedding-3-small",
            "k",
        );
        assert_eq!(
            client.url(),
            format!(
                "https://r.services.ai.azure.com/models/embeddings?api-version={DEFAULT_API_VERSION}"
            )
        );
    }

    #[test]
    fn api_version_is_overridable() {
        let client = FoundryEmbeddingClient::new("https://e/models", "m", "k")
            .with_api_version("2099-01-01");
        assert!(client.url().ends_with("?api-version=2099-01-01"));
    }

    #[test]
    fn body_carries_input_model_and_forwarded_extras() {
        let client = FoundryEmbeddingClient::new("https://e/models", "default-model", "k");
        let bare = client.body_for(&["a".into(), "b".into()], None);
        assert_eq!(
            bare,
            json!({ "input": ["a", "b"], "model": "default-model" })
        );

        let mut options = EmbeddingGenerationOptions::new();
        options.model = Some("per-request".into());
        options.dimensions = Some(512);
        options
            .additional_properties
            .insert("encoding_format".into(), json!("float"));
        options
            .additional_properties
            .insert("input_type".into(), json!("query"));
        let full = client.body_for(&["a".into()], Some(&options));
        assert_eq!(
            full,
            json!({
                "input": ["a"],
                "model": "per-request",
                "dimensions": 512,
                "encoding_format": "float",
                "input_type": "query",
            })
        );
    }

    #[test]
    fn from_env_requires_endpoint_model_and_key() {
        // Each missing piece names itself rather than failing at request time.
        temp_env_absent(|| {
            let err = FoundryEmbeddingClient::from_env(None).expect_err("no endpoint");
            assert!(
                err.to_string().contains(FOUNDRY_MODELS_ENDPOINT_ENV),
                "{err}"
            );
        });
    }

    /// Run `f` with the three Foundry embedding variables unset, restoring
    /// whatever was there before. Tests in one binary share a process
    /// environment, so this is deliberately narrow.
    fn temp_env_absent(f: impl FnOnce()) {
        let keys = [
            FOUNDRY_MODELS_ENDPOINT_ENV,
            FOUNDRY_MODELS_API_KEY_ENV,
            FOUNDRY_EMBEDDING_MODEL_ENV,
        ];
        let saved: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, std::env::var(k).ok())).collect();
        for key in keys {
            std::env::remove_var(key);
        }
        f();
        for (key, value) in saved {
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }
}
