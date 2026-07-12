//! [`AnthropicVertexClient`]: Anthropic (Claude) models hosted on
//! [Google Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude),
//! spoken over Vertex's `rawPredict` / `streamRawPredict` publisher-model
//! routes
//! (`POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:rawPredict`).
//!
//! Like [`crate::bedrock::AnthropicBedrockClient`], this is a *transport*, not
//! a new wire format: the request/response body is the same Anthropic
//! Messages API shape [`AnthropicClient`](crate::AnthropicClient) speaks
//! directly, built via [`crate::convert::build_cloud_request`] (which carries
//! `anthropic_version: "vertex-2023-10-16"` instead of a top-level `model`
//! field, since the model is already selected by the URL).
//!
//! ## Authentication
//!
//! Vertex AI authenticates with a Google OAuth2 access token
//! (`Authorization: Bearer <token>`) minted via Google's
//! [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials)
//! (ADC) flow. This workspace has no Google Cloud SDK dependency to perform
//! that flow, so instead of reimplementing it, token acquisition is factored
//! out behind the small, synchronous [`VertexTokenProvider`] trait: the
//! caller supplies a token however they like (most simply, the output of
//! `gcloud auth print-access-token`, wrapped in [`StaticVertexToken`]).
//! Wiring up full ADC auto-discovery (metadata-server tokens on GCE/GKE,
//! service-account JSON key files, `gcloud`'s cached ADC token, …) behind a
//! richer provider implementation is a documented extension point, not
//! attempted here.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_anthropic::vertex::{AnthropicVertexClient, StaticVertexToken};
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! // `$ gcloud auth print-access-token`
//! let token = std::env::var("VERTEX_ACCESS_TOKEN").unwrap_or_default();
//! let client = AnthropicVertexClient::new(
//!     "my-gcp-project",
//!     "us-east5",
//!     "claude-sonnet-4-5@20250929",
//!     Arc::new(StaticVertexToken::new(token)),
//! );
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, Message, Role, UsageContent,
};
use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::convert;

/// The `anthropic_version` Vertex AI's `rawPredict`/`streamRawPredict` routes
/// expect in the body of every Claude request (per Google's
/// [Claude-on-Vertex documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude)).
pub const ANTHROPIC_VERTEX_VERSION: &str = "vertex-2023-10-16";

/// `max_tokens` is required by the Anthropic Messages API; used whenever
/// neither `ChatOptions::max_tokens` nor a client-level override is set.
/// Matches [`crate::AnthropicClient`]'s default.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// The region [`AnthropicVertexClient::from_env`] falls back to when neither
/// `CLOUD_ML_REGION` nor `ANTHROPIC_VERTEX_REGION` is set — Anthropic's
/// models on Vertex AI are only available in a handful of regions, and
/// `us-east5` is the one Google's own quickstart documentation defaults to.
const DEFAULT_REGION: &str = "us-east5";

/// Supplies a Google OAuth2 access token to authenticate Vertex AI requests
/// (`Authorization: Bearer <token>`).
///
/// Deliberately synchronous and minimal — see the [module docs](self) for why
/// this crate doesn't perform Google's Application Default Credentials flow
/// itself. Implement this to plug in real token acquisition/refresh (a
/// metadata-server client, a cached `gcloud` token, a service-account JWT
/// exchange, …); [`StaticVertexToken`] covers the common case of a
/// caller-supplied, pre-fetched token.
pub trait VertexTokenProvider: Send + Sync {
    /// Return the current access token to send as `Authorization: Bearer
    /// <token>`. Implementations that cache/refresh a token should do so
    /// internally (e.g. behind a `Mutex`); this is called once per request.
    fn access_token(&self) -> Result<String>;
}

/// A [`VertexTokenProvider`] that always returns the same, pre-fetched token
/// — e.g. the output of `gcloud auth print-access-token`.
#[derive(Debug, Clone)]
pub struct StaticVertexToken(String);

impl StaticVertexToken {
    /// Wrap a fixed access token.
    pub fn new(token: impl Into<String>) -> Self {
        Self(token.into())
    }
}

impl VertexTokenProvider for StaticVertexToken {
    fn access_token(&self) -> Result<String> {
        Ok(self.0.clone())
    }
}

/// Classify a non-success Vertex AI HTTP response into a granular [`Error`].
/// Google's Vertex AI error bodies use a `{"error": {"code": ..., "status":
/// "PERMISSION_DENIED" | "INVALID_ARGUMENT" | ...}}` shape, but the HTTP
/// status alone is sufficient and unambiguous for the classification this
/// crate's other clients perform: `401`/`403` -> invalid auth, `400` ->
/// invalid request, everything else (notably `429` and `5xx`) stays a
/// generic, retry-layer-visible [`Error::ServiceStatus`].
fn classify_vertex_error(status: u16, message: impl Into<String>) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 => Error::service_invalid_request(message),
        _ => Error::service_status(status, message, None),
    }
}

/// An Anthropic Messages API transport for Claude models on Google Vertex AI
/// (`POST https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:rawPredict`).
///
/// See the [module docs](self) for how this relates to
/// [`AnthropicClient`](crate::AnthropicClient) and for the authentication
/// model.
#[derive(Clone)]
pub struct AnthropicVertexClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    project_id: String,
    region: String,
    model: String,
    token_provider: Arc<dyn VertexTokenProvider>,
    max_tokens: u32,
}

impl std::fmt::Debug for AnthropicVertexClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicVertexClient")
            .field("project_id", &self.inner.project_id)
            .field("region", &self.inner.region)
            .field("model", &self.inner.model)
            .field("max_tokens", &self.inner.max_tokens)
            .finish_non_exhaustive()
    }
}

impl AnthropicVertexClient {
    /// Create a client for the given GCP project, region, default model id,
    /// and token provider.
    pub fn new(
        project_id: impl Into<String>,
        region: impl Into<String>,
        model: impl Into<String>,
        token_provider: Arc<dyn VertexTokenProvider>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                project_id: project_id.into(),
                region: region.into(),
                model: model.into(),
                token_provider,
                max_tokens: DEFAULT_MAX_TOKENS,
            }),
        }
    }

    /// Build a client from `GOOGLE_CLOUD_PROJECT` (falling back to
    /// `ANTHROPIC_VERTEX_PROJECT_ID`) and `CLOUD_ML_REGION` (falling back to
    /// `ANTHROPIC_VERTEX_REGION`, defaulting to `us-east5` when neither is
    /// set).
    pub fn from_env(
        model: impl Into<String>,
        token_provider: Arc<dyn VertexTokenProvider>,
    ) -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("ANTHROPIC_VERTEX_PROJECT_ID"))
            .map_err(|_| {
                Error::Configuration(
                    "neither GOOGLE_CLOUD_PROJECT nor ANTHROPIC_VERTEX_PROJECT_ID is set".into(),
                )
            })?;
        let region = std::env::var("CLOUD_ML_REGION")
            .or_else(|_| std::env::var("ANTHROPIC_VERTEX_REGION"))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());
        Ok(Self::new(project_id, region, model, token_provider))
    }

    /// Override the default `max_tokens` sent when `ChatOptions::max_tokens`
    /// is unset (the Anthropic Messages API requires this field).
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        Arc::make_mut(&mut self.inner).max_tokens = max_tokens;
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The configured GCP project id.
    pub fn project_id(&self) -> &str {
        &self.inner.project_id
    }

    /// The configured Vertex AI region.
    pub fn region(&self) -> &str {
        &self.inner.region
    }

    fn effective_model(&self, options: &ChatOptions) -> String {
        options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone())
    }

    /// The `rawPredict` (or `streamRawPredict`) URL for a given model id.
    fn url(&self, model: &str, streaming: bool) -> String {
        let action = if streaming {
            "streamRawPredict"
        } else {
            "rawPredict"
        };
        format!(
            "https://{region}-aiplatform.googleapis.com/v1/projects/{project}/locations/{region}/publishers/anthropic/models/{model}:{action}",
            region = self.inner.region,
            project = self.inner.project_id,
        )
    }

    async fn send(&self, model: &str, body: &Value) -> Result<reqwest::Response> {
        let token = self.inner.token_provider.access_token()?;
        let resp = self
            .inner
            .http
            .post(self.url(model, false))
            .bearer_auth(token)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_vertex_error(
                status.as_u16(),
                format!("Vertex AI rawPredict error {status}: {text}"),
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for AnthropicVertexClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let model = self.effective_model(&options);
        let max_tokens = options.max_tokens.unwrap_or(self.inner.max_tokens);
        let body = convert::build_cloud_request(
            &messages,
            &options,
            max_tokens,
            false,
            ANTHROPIC_VERTEX_VERSION,
        );
        let resp = self.send(&model, &body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(convert::parse_response(&value))
    }

    /// Get a streaming response.
    ///
    /// Vertex AI's `streamRawPredict` route does stream Anthropic's usual SSE
    /// framing, but wiring up a second SSE consumer here (duplicating
    /// [`crate::AnthropicClient`]'s `parse_sse_stream`) plus a
    /// synchronous-token-provider-vs-long-lived-stream story is deferred as a
    /// documented extension point for a first cut of this transport. This
    /// method calls the non-streaming `rawPredict` route (the same request
    /// [`ChatClient::get_response`] sends) and adapts the complete
    /// [`ChatResponse`] into a single [`ChatResponseUpdate`] — the same
    /// tactic [`crate::bedrock::AnthropicBedrockClient::get_streaming_response`]
    /// and [`agent_framework_bedrock::BedrockChatClient::get_streaming_response`]
    /// use. Callers driving this client through
    /// [`ChatResponse::from_updates`](agent_framework_core::types::ChatResponse::from_updates)
    /// still get a correct aggregated result; they just don't see partial
    /// text arrive incrementally.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;

        let mut contents: Vec<Content> = response
            .messages
            .iter()
            .flat_map(|m| m.contents.iter().cloned())
            .collect();
        if let Some(usage) = response.usage_details.clone() {
            contents.push(Content::Usage(UsageContent { details: usage }));
        }

        let update = ChatResponseUpdate {
            contents,
            role: Some(Role::assistant()),
            response_id: response.response_id.clone(),
            model: response.model.clone(),
            finish_reason: response.finish_reason.clone(),
            ..Default::default()
        };
        Ok(stream::once(async move { Ok(update) }).boxed())
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> AnthropicVertexClient {
        AnthropicVertexClient::new(
            "my-project",
            "us-east5",
            "claude-sonnet-4-5@20250929",
            Arc::new(StaticVertexToken::new("test-token")),
        )
    }

    #[test]
    fn static_vertex_token_returns_configured_token() {
        let provider = StaticVertexToken::new("abc123");
        assert_eq!(provider.access_token().unwrap(), "abc123");
    }

    #[test]
    fn url_contains_publishers_anthropic_models_and_raw_predict() {
        let c = client();
        let url = c.url("claude-sonnet-4-5@20250929", false);
        assert!(url.contains("/projects/my-project/"));
        assert!(url.contains("/locations/us-east5/"));
        assert!(url.contains("/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"));
        assert_eq!(
            url,
            "https://us-east5-aiplatform.googleapis.com/v1/projects/my-project/locations/us-east5/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict"
        );
    }

    #[test]
    fn url_streaming_uses_stream_raw_predict() {
        let c = client();
        let url = c.url("claude-sonnet-4-5@20250929", true);
        assert!(url.ends_with(":streamRawPredict"));
    }

    #[test]
    fn accessors_expose_project_region_and_model() {
        let c = client();
        assert_eq!(c.project_id(), "my-project");
        assert_eq!(c.region(), "us-east5");
        assert_eq!(c.model(), "claude-sonnet-4-5@20250929");
        assert_eq!(ChatClient::model(&c), Some("claude-sonnet-4-5@20250929"));
    }

    #[test]
    fn request_body_has_vertex_anthropic_version_and_no_model_key() {
        let body = convert::build_cloud_request(
            &[Message::user("hi")],
            &ChatOptions::new(),
            1024,
            false,
            ANTHROPIC_VERTEX_VERSION,
        );
        assert_eq!(
            body["anthropic_version"],
            serde_json::json!("vertex-2023-10-16")
        );
        assert!(body.get("model").is_none());
    }

    // region: env-var constructor

    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_vertex_env() {
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module.
        unsafe {
            std::env::remove_var("GOOGLE_CLOUD_PROJECT");
            std::env::remove_var("ANTHROPIC_VERTEX_PROJECT_ID");
            std::env::remove_var("CLOUD_ML_REGION");
            std::env::remove_var("ANTHROPIC_VERTEX_REGION");
        }
    }

    #[test]
    fn from_env_errors_without_project() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_vertex_env();
        let result =
            AnthropicVertexClient::from_env("claude-x", Arc::new(StaticVertexToken::new("t")));
        assert!(matches!(result, Err(Error::Configuration(_))));
        clear_vertex_env();
    }

    #[test]
    fn from_env_reads_project_and_region() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_vertex_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_vertex_env.
        unsafe {
            std::env::set_var("GOOGLE_CLOUD_PROJECT", "proj-1");
            std::env::set_var("CLOUD_ML_REGION", "europe-west1");
        }
        let client =
            AnthropicVertexClient::from_env("claude-x", Arc::new(StaticVertexToken::new("t")))
                .unwrap();
        assert_eq!(client.project_id(), "proj-1");
        assert_eq!(client.region(), "europe-west1");
        clear_vertex_env();
    }

    #[test]
    fn from_env_falls_back_to_anthropic_specific_vars_and_default_region() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_vertex_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_vertex_env.
        unsafe {
            std::env::set_var("ANTHROPIC_VERTEX_PROJECT_ID", "proj-2");
        }
        let client =
            AnthropicVertexClient::from_env("claude-x", Arc::new(StaticVertexToken::new("t")))
                .unwrap();
        assert_eq!(client.project_id(), "proj-2");
        assert_eq!(client.region(), DEFAULT_REGION);
        clear_vertex_env();
    }

    // endregion
}
