//! Azure OpenAI embeddings client.
//!
//! Rust equivalent of upstream's Azure branch of the OpenAI embedding client
//! (`agent_framework_openai/_embedding_client.py`): the deployment-scoped
//! `POST {endpoint}/openai/deployments/{deployment}/embeddings` endpoint with
//! `api-key` or Entra ID auth. The wire body/response are OpenAI-shaped, so
//! request extras and response parsing are shared with
//! `agent-framework-openai` rather than duplicated.

use std::sync::Arc;

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    EmbeddingGenerationOptions, EmbeddingInput, GeneratedEmbeddings,
};
use serde_json::{json, Map, Value};

use crate::credential::TokenCredential;
use crate::{parse_retry_after, DEFAULT_EMBEDDING_API_VERSION};

enum Auth {
    ApiKey(String),
    Credential(Arc<dyn TokenCredential>),
}

/// An Azure OpenAI embeddings client
/// (`POST {endpoint}/openai/deployments/{deployment}/embeddings`).
#[derive(Clone)]
pub struct AzureOpenAIEmbeddingClient {
    inner: Arc<Inner>,
}

struct Inner {
    http: reqwest::Client,
    endpoint: String,
    deployment: String,
    api_version: String,
    auth: Auth,
}

impl AzureOpenAIEmbeddingClient {
    /// Create a client authenticating with a static API key
    /// (`api-key` header).
    pub fn new(
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                endpoint: endpoint.into(),
                deployment: deployment.into(),
                api_version: DEFAULT_EMBEDDING_API_VERSION.to_string(),
                auth: Auth::ApiKey(api_key.into()),
            }),
        }
    }

    /// Create a client authenticating via a [`TokenCredential`]
    /// (`Authorization: Bearer <token>`, e.g. Microsoft Entra ID).
    pub fn with_token_credential(
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                endpoint: endpoint.into(),
                deployment: deployment.into(),
                api_version: DEFAULT_EMBEDDING_API_VERSION.to_string(),
                auth: Auth::Credential(credential),
            }),
        }
    }

    /// Build an API-key-authenticated client from `AZURE_OPENAI_ENDPOINT`,
    /// `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`, and
    /// optional `AZURE_OPENAI_API_VERSION`.
    pub fn from_env() -> Result<Self> {
        Self::from_env_vars(|key| std::env::var(key).ok())
    }

    /// Implementation of [`from_env`](Self::from_env), parameterized over an
    /// environment lookup function so unit tests can exercise it against an
    /// in-memory map rather than the process-global environment.
    fn from_env_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let endpoint = get("AZURE_OPENAI_ENDPOINT")
            .ok_or_else(|| Error::Configuration("AZURE_OPENAI_ENDPOINT is not set".into()))?;
        let api_key = get("AZURE_OPENAI_API_KEY")
            .ok_or_else(|| Error::Configuration("AZURE_OPENAI_API_KEY is not set".into()))?;
        let deployment = get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME").ok_or_else(|| {
            Error::Configuration("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME is not set".into())
        })?;
        let mut client = Self::new(endpoint, deployment, api_key);
        // Blank leaves [`DEFAULT_EMBEDDING_API_VERSION`] in place: the
        // embeddings route requires `api-version`, so a blank one is a 400.
        // See `AzureOpenAIClient::from_env_vars` (#9098).
        if let Some(v) = get("AZURE_OPENAI_API_VERSION") {
            if !v.trim().is_empty() {
                client = client.with_api_version(v);
            }
        }
        Ok(client)
    }

    /// Override the API version (default [`DEFAULT_EMBEDDING_API_VERSION`]).
    ///
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).api_version = api_version.into();
        self
    }

    /// The deployment name this client targets.
    pub fn deployment(&self) -> &str {
        &self.inner.deployment
    }

    fn url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.inner.endpoint.trim_end_matches('/'),
            self.inner.deployment,
            self.inner.api_version,
        )
    }

    fn build_body(&self, values: &[String], options: Option<&EmbeddingGenerationOptions>) -> Value {
        let mut body = Map::new();
        // The deployment in the URL already selects the model; only send
        // `model` if the caller explicitly asked for a specific one (same
        // convention as the chat client).
        if let Some(model) = options.and_then(|o| o.model.as_ref()) {
            body.insert("model".into(), json!(model));
        }
        body.insert("input".into(), json!(values));
        if let Some(options) = options {
            if let Some(dimensions) = options.dimensions {
                body.insert("dimensions".into(), json!(dimensions));
            }
            for key in ["encoding_format", "user"] {
                if let Some(v) = options.additional_properties.get(key) {
                    body.insert(key.into(), v.clone());
                }
            }
        }
        Value::Object(body)
    }

    async fn auth_header(&self) -> Result<(&'static str, String)> {
        match &self.inner.auth {
            Auth::ApiKey(key) => Ok(("api-key", key.clone())),
            Auth::Credential(credential) => {
                let token = credential.get_token().await?;
                Ok(("Authorization", format!("Bearer {token}")))
            }
        }
    }
}

fn arc_inner(inner: &mut Arc<Inner>) -> &mut Inner {
    if Arc::strong_count(inner) != 1 {
        *inner = Arc::new(Inner {
            http: inner.http.clone(),
            endpoint: inner.endpoint.clone(),
            deployment: inner.deployment.clone(),
            api_version: inner.api_version.clone(),
            auth: match &inner.auth {
                Auth::ApiKey(k) => Auth::ApiKey(k.clone()),
                Auth::Credential(c) => Auth::Credential(c.clone()),
            },
        });
    }
    Arc::get_mut(inner).expect("just ensured unique")
}

#[async_trait::async_trait]
impl EmbeddingClient for AzureOpenAIEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<EmbeddingInput>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        let values = EmbeddingInput::into_texts(values, "Azure OpenAI")?;
        let body = self.build_body(&values, options.as_ref());
        let (header_name, header_value) = self.auth_header().await?;
        let resp = self
            .inner
            .http
            .post(self.url())
            .header(header_name, header_value)
            .json(&body)
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
                format!("Azure OpenAI API error {status}: {text}"),
                retry_after,
            ));
        }
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        agent_framework_openai::embeddings::parse_embeddings_response(&value)
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.deployment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn url_is_deployment_scoped_with_api_version() {
        let client = AzureOpenAIEmbeddingClient::new(
            "https://example.openai.azure.com/",
            "embed-dep",
            "key",
        )
        .with_api_version("2024-10-21");
        assert_eq!(
            client.url(),
            "https://example.openai.azure.com/openai/deployments/embed-dep/embeddings?api-version=2024-10-21"
        );
    }

    #[test]
    fn from_env_defaults_api_version_when_blank() {
        // Same rule as the chat client: the embeddings route requires
        // `api-version`, so a variable that is set but blank leaves the
        // dated default in place rather than emptying the query parameter
        // (upstream #9098).
        for blank in ["", "   "] {
            let client = AzureOpenAIEmbeddingClient::from_env_vars(|key| match key {
                "AZURE_OPENAI_ENDPOINT" => Some("https://res.openai.azure.com".into()),
                "AZURE_OPENAI_API_KEY" => Some("test-key-123".into()),
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME" => Some("embed-dep".into()),
                "AZURE_OPENAI_API_VERSION" => Some(blank.to_string()),
                _ => None,
            })
            .unwrap();
            assert_eq!(client.inner.api_version, DEFAULT_EMBEDDING_API_VERSION);
            assert!(
                client
                    .url()
                    .ends_with(&format!("api-version={DEFAULT_EMBEDDING_API_VERSION}")),
                "a blank variable must not reach the query string: {}",
                client.url()
            );
        }
    }

    #[test]
    fn body_omits_model_unless_explicit() {
        let client = AzureOpenAIEmbeddingClient::new("https://e", "dep", "key");
        let body = client.build_body(&["x".into()], None);
        assert!(body.get("model").is_none());
        assert_eq!(body["input"], json!(["x"]));

        let options = EmbeddingGenerationOptions::new()
            .with_model("text-embedding-3-large")
            .with_dimensions(64);
        let body = client.build_body(&["x".into()], Some(&options));
        assert_eq!(body["model"], "text-embedding-3-large");
        assert_eq!(body["dimensions"], 64);
    }
}
