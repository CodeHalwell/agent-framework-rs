//! [`AnthropicFoundryClient`]: Anthropic (Claude) models hosted on
//! [Azure AI Foundry](https://learn.microsoft.com/en-us/azure/ai-foundry/),
//! spoken over a Foundry Anthropic deployment's Messages-shaped endpoint
//! (`POST {base_url}{path}`, defaulting to `{base_url}/v1/messages`).
//!
//! Like [`crate::bedrock::AnthropicBedrockClient`] and
//! [`crate::vertex::AnthropicVertexClient`], this is a *transport*, not a new
//! wire format: the request body is the same Anthropic Messages API shape
//! [`AnthropicClient`](crate::AnthropicClient) speaks directly, built via
//! [`crate::convert::build_cloud_request`]. Authentication is Microsoft Entra
//! ID (`Authorization: Bearer <token>`), via the
//! [`agent_framework_azure::TokenCredential`] abstraction that crate's own
//! Azure OpenAI clients use — any of its real credential chains
//! (`ManagedIdentityCredential`, `ClientSecretCredential`,
//! `AzureCliCredential`, `ChainedTokenCredential`, …) work here unmodified.
//!
//! ## What's *not* stably documented
//!
//! Unlike Bedrock's `InvokeModel`/`anthropic_version` pairing or Vertex AI's
//! `rawPredict`/`anthropic_version` pairing, Azure AI Foundry's Anthropic
//! integration does not (as of this writing) have a single, stable, publicly
//! documented route path or `anthropic_version` tag the way the other two
//! clouds do — Foundry deployments vary in how they expose partner models.
//! Both are therefore **overridable defaults**, not hardcoded assumptions:
//!
//! * the path suffix defaults to [`DEFAULT_PATH`] (`/v1/messages`) and can be
//!   changed per deployment via [`with_path`](AnthropicFoundryClient::with_path);
//! * the `anthropic_version` body field defaults to
//!   [`ANTHROPIC_FOUNDRY_VERSION`] and can be changed via
//!   [`with_anthropic_version`](AnthropicFoundryClient::with_anthropic_version).
//!
//! Callers should confirm both against their specific Foundry deployment's
//! documentation before relying on the defaults in production.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::StaticTokenCredential;
//! use agent_framework_anthropic::foundry::AnthropicFoundryClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let credential = Arc::new(StaticTokenCredential::new("eyJ0eXAi..."));
//! let client = AnthropicFoundryClient::with_token_credential(
//!     "https://my-foundry-resource.services.ai.azure.com",
//!     "claude-sonnet-4-5",
//!     credential,
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

use agent_framework_azure::TokenCredential;
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Content, Message, Role, UsageContent,
};
use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::convert;

/// Default path suffix appended to `base_url`. Overridable via
/// [`AnthropicFoundryClient::with_path`] — see the [module docs](self) for
/// why this isn't assumed to be stable across Foundry deployments.
pub const DEFAULT_PATH: &str = "/v1/messages";

/// Default `anthropic_version` sent in the request body. This is **not**
/// drawn from stable, publicly documented Azure AI Foundry API reference the
/// way [`crate::bedrock::ANTHROPIC_BEDROCK_VERSION`] and
/// [`crate::vertex::ANTHROPIC_VERTEX_VERSION`] are — see the [module
/// docs](self). Overridable via
/// [`AnthropicFoundryClient::with_anthropic_version`].
pub const ANTHROPIC_FOUNDRY_VERSION: &str = "foundry-2025-01-01";

/// Default Entra ID scope requested for the bearer token, matching the scope
/// [`agent_framework_azure::AzureOpenAIClient`]'s own documentation uses for
/// Azure AI resources. Overridable via
/// [`AnthropicFoundryClient::with_scope`] if a specific Foundry deployment
/// requires a different audience.
pub const DEFAULT_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// `max_tokens` is required by the Anthropic Messages API; used whenever
/// neither `ChatOptions::max_tokens` nor a client-level override is set.
/// Matches [`crate::AnthropicClient`]'s default.
const DEFAULT_MAX_TOKENS: u32 = 1024;

/// Classify a non-success Foundry HTTP response into a granular [`Error`].
/// Status-only, matching [`crate::bedrock`]/[`crate::vertex`]: `401`/`403` ->
/// invalid auth, `400` -> invalid request, everything else (notably `429`
/// and `5xx`) stays a generic, retry-layer-visible [`Error::ServiceStatus`].
fn classify_foundry_error(status: u16, message: impl Into<String>) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 => Error::service_invalid_request(message),
        _ => Error::service_status(status, message, None),
    }
}

/// An Anthropic Messages API transport for Claude models hosted on Azure AI
/// Foundry (`POST {base_url}{path}`).
///
/// See the [module docs](self) for how this relates to
/// [`AnthropicClient`](crate::AnthropicClient) and for the caveats around the
/// default path/`anthropic_version`.
#[derive(Clone)]
pub struct AnthropicFoundryClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    base_url: String,
    path: String,
    model: String,
    credential: Arc<dyn TokenCredential>,
    scope: String,
    anthropic_version: String,
    max_tokens: u32,
}

impl std::fmt::Debug for AnthropicFoundryClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicFoundryClient")
            .field("base_url", &self.inner.base_url)
            .field("path", &self.inner.path)
            .field("model", &self.inner.model)
            .field("scope", &self.inner.scope)
            .field("anthropic_version", &self.inner.anthropic_version)
            .field("max_tokens", &self.inner.max_tokens)
            .finish_non_exhaustive()
    }
}

impl AnthropicFoundryClient {
    /// Create a client authenticating via a [`TokenCredential`] (Microsoft
    /// Entra ID) against a Foundry Anthropic deployment's `base_url`.
    pub fn with_token_credential(
        base_url: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                base_url: base_url.into(),
                path: DEFAULT_PATH.to_string(),
                model: model.into(),
                credential,
                scope: DEFAULT_SCOPE.to_string(),
                anthropic_version: ANTHROPIC_FOUNDRY_VERSION.to_string(),
                max_tokens: DEFAULT_MAX_TOKENS,
            }),
        }
    }

    /// Override the Entra ID scope requested for the bearer token (default
    /// [`DEFAULT_SCOPE`]).
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).scope = scope.into();
        self
    }

    /// Override the path suffix appended to `base_url` (default
    /// [`DEFAULT_PATH`]). See the [module docs](self) for why this may need
    /// adjusting per deployment.
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).path = path.into();
        self
    }

    /// Override the `anthropic_version` sent in the request body (default
    /// [`ANTHROPIC_FOUNDRY_VERSION`]). See the [module docs](self) for why
    /// this may need adjusting per deployment.
    pub fn with_anthropic_version(mut self, anthropic_version: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).anthropic_version = anthropic_version.into();
        self
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

    /// The full request URL (`{base_url}{path}`).
    fn url(&self) -> String {
        format!(
            "{}{}",
            self.inner.base_url.trim_end_matches('/'),
            self.inner.path
        )
    }

    /// Fetch the bearer token for the configured scope. Split out from
    /// [`send`](Self::send) so header attachment is unit-testable without an
    /// HTTP round trip.
    async fn bearer_token(&self) -> Result<String> {
        self.inner
            .credential
            .get_token_for_scope(&self.inner.scope)
            .await
    }

    async fn send(&self, body: &Value) -> Result<reqwest::Response> {
        let token = self.bearer_token().await?;
        let resp = self
            .inner
            .http
            .post(self.url())
            .bearer_auth(token)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_foundry_error(
                status.as_u16(),
                format!("Azure AI Foundry error {status}: {text}"),
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for AnthropicFoundryClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        // Unlike Bedrock/Vertex, a Foundry deployment's `base_url` is itself
        // model-scoped (it names one specific Anthropic deployment), so
        // there is no URL slot to route a per-request `ChatOptions::model`
        // override into; the configured `model` is descriptive only (see
        // `model()`/`Debug`).
        let max_tokens = options.max_tokens.unwrap_or(self.inner.max_tokens);
        let body = convert::build_cloud_request(
            &messages,
            &options,
            max_tokens,
            false,
            &self.inner.anthropic_version,
        );
        let resp = self.send(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(convert::parse_response(&value))
    }

    /// Get a streaming response.
    ///
    /// Whether (and how) a given Foundry Anthropic deployment streams is one
    /// more thing this crate can't assume stably across deployments (see the
    /// [module docs](self)), so this calls the same non-streaming endpoint
    /// [`ChatClient::get_response`] does and adapts the complete
    /// [`ChatResponse`] into a single [`ChatResponseUpdate`] — the same
    /// tactic [`crate::bedrock::AnthropicBedrockClient::get_streaming_response`]
    /// and [`crate::vertex::AnthropicVertexClient::get_streaming_response`]
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
    use agent_framework_azure::StaticTokenCredential;

    fn client() -> AnthropicFoundryClient {
        AnthropicFoundryClient::with_token_credential(
            "https://my-foundry-resource.services.ai.azure.com",
            "claude-sonnet-4-5",
            Arc::new(StaticTokenCredential::new("my-jwt-token")),
        )
    }

    #[test]
    fn url_is_base_url_plus_default_path() {
        assert_eq!(
            client().url(),
            "https://my-foundry-resource.services.ai.azure.com/v1/messages"
        );
    }

    #[test]
    fn url_trims_trailing_slash_on_base_url_before_appending_path() {
        let c = AnthropicFoundryClient::with_token_credential(
            "https://my-foundry-resource.services.ai.azure.com/",
            "claude-sonnet-4-5",
            Arc::new(StaticTokenCredential::new("tok")),
        );
        assert_eq!(
            c.url(),
            "https://my-foundry-resource.services.ai.azure.com/v1/messages"
        );
    }

    #[test]
    fn with_path_overrides_default_path() {
        let c = client().with_path("/anthropic/v1/messages");
        assert_eq!(
            c.url(),
            "https://my-foundry-resource.services.ai.azure.com/anthropic/v1/messages"
        );
    }

    #[tokio::test]
    async fn bearer_token_is_attached_from_credential() {
        let c = client();
        assert_eq!(c.bearer_token().await.unwrap(), "my-jwt-token");
    }

    #[test]
    fn with_scope_overrides_default_scope() {
        let c = client().with_scope("https://example.com/.default");
        assert_eq!(c.inner.scope, "https://example.com/.default");
    }

    #[test]
    fn request_body_has_configured_anthropic_version_and_no_model_key() {
        let c = client().with_anthropic_version("2024-99-99");
        let body = convert::build_cloud_request(
            &[Message::user("hi")],
            &ChatOptions::new(),
            1024,
            false,
            &c.inner.anthropic_version,
        );
        assert_eq!(body["anthropic_version"], serde_json::json!("2024-99-99"));
        assert!(body.get("model").is_none());
    }

    #[test]
    fn default_anthropic_version_matches_const() {
        let c = client();
        assert_eq!(c.inner.anthropic_version, ANTHROPIC_FOUNDRY_VERSION);
    }

    #[test]
    fn model_accessor() {
        let c = client();
        assert_eq!(c.model(), "claude-sonnet-4-5");
        assert_eq!(ChatClient::model(&c), Some("claude-sonnet-4-5"));
    }
}
