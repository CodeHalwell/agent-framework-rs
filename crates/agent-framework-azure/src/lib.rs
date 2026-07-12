//! # agent-framework-azure
//!
//! An Azure OpenAI [`ChatClient`] for `agent-framework-rs`, supporting both
//! static API-key and Microsoft Entra ID (OAuth bearer token) authentication.
//!
//! Azure OpenAI's Chat Completions wire format is identical to OpenAI's, so
//! request/response conversion is reused from
//! [`agent_framework_openai::convert`] rather than duplicated — only the URL
//! shape (`{endpoint}/openai/deployments/{deployment}/chat/completions`) and
//! authentication differ.
//!
//! ```no_run
//! use agent_framework_azure::AzureOpenAIClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = AzureOpenAIClient::new(
//!     "https://my-resource.openai.azure.com",
//!     "my-gpt4o-deployment",
//!     "my-api-key",
//! );
//! let agent = ChatAgent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Entra ID (bearer token) authentication instead of a static key:
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::{AzureOpenAIClient, StaticTokenCredential};
//!
//! let credential = Arc::new(StaticTokenCredential::new("eyJ0eXAi..."));
//! let client = AzureOpenAIClient::with_token_credential(
//!     "https://my-resource.openai.azure.com",
//!     "my-gpt4o-deployment",
//!     credential,
//! );
//! ```
//!
//! A real Microsoft Entra ID credential chain — try a managed identity, then a
//! client secret, then the Azure CLI, whichever succeeds first (each caches and
//! refreshes tokens for the configured scope):
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::{
//!     AzureCliCredential, ChainedTokenCredential, ClientSecretCredential,
//!     ManagedIdentityCredential, TokenCredential,
//! };
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let scope = "https://cognitiveservices.azure.com/.default";
//! let chain = ChainedTokenCredential::new(vec![
//!     Arc::new(ManagedIdentityCredential::new(scope)),
//!     Arc::new(ClientSecretCredential::new("tenant", "client", "secret", scope)),
//!     Arc::new(AzureCliCredential::new(scope)),
//! ]);
//! let token = chain.get_token().await?;
//! # let _ = token;
//! # Ok(())
//! # }
//! ```

mod credential;
mod credentials;
pub mod responses;

pub use credential::{StaticTokenCredential, TokenCredential};
pub use credentials::{
    AzureCliCredential, ChainedTokenCredential, ClientSecretCredential, DefaultAzureCredential,
    EnvironmentCredential, ManagedIdentityCredential, WorkloadIdentityCredential,
    DEFAULT_AUTHORITY, DEFAULT_IMDS_ENDPOINT, REFRESH_SKEW,
};
pub use responses::AzureOpenAIResponsesClient;

use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{ChatMessage, ChatOptions, ChatResponse};
use futures::StreamExt;
use serde_json::{json, Map, Value};

const DEFAULT_API_VERSION: &str = "2024-10-21";

/// Parse a `Retry-After` header into a delay in seconds.
///
/// Mirrors the OpenAI/Anthropic clients: Azure OpenAI returns the
/// integer/decimal-seconds form on `429`/`503`, which is what we honor so a
/// [`RetryingChatClient`](agent_framework_core::client::RetryingChatClient) can
/// wait exactly as long as the server asks. A date-form or unparseable value is
/// treated as absent.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// How a request authenticates against Azure OpenAI.
#[derive(Clone)]
enum Auth {
    /// `api-key: <key>` header.
    ApiKey(String),
    /// `Authorization: Bearer <token>`, fetched fresh per request from a
    /// [`TokenCredential`].
    Credential(Arc<dyn TokenCredential>),
}

/// An Azure OpenAI chat client
/// (`POST {endpoint}/openai/deployments/{deployment}/chat/completions`).
pub struct AzureOpenAIClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    endpoint: String,
    deployment: String,
    api_version: String,
    auth: Auth,
}

impl Clone for AzureOpenAIClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for AzureOpenAIClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAIClient")
            .field("endpoint", &self.inner.endpoint)
            .field("deployment", &self.inner.deployment)
            .field("api_version", &self.inner.api_version)
            .field(
                "auth",
                &match &self.inner.auth {
                    Auth::ApiKey(_) => "api-key",
                    Auth::Credential(_) => "token-credential",
                },
            )
            .finish_non_exhaustive()
    }
}

impl AzureOpenAIClient {
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
                api_version: DEFAULT_API_VERSION.to_string(),
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
                api_version: DEFAULT_API_VERSION.to_string(),
                auth: Auth::Credential(credential),
            }),
        }
    }

    /// Build an API-key-authenticated client from `AZURE_OPENAI_ENDPOINT`,
    /// `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, and
    /// optional `AZURE_OPENAI_API_VERSION`.
    pub fn from_env() -> Result<Self> {
        let endpoint = std::env::var("AZURE_OPENAI_ENDPOINT")
            .map_err(|_| Error::Configuration("AZURE_OPENAI_ENDPOINT is not set".into()))?;
        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("AZURE_OPENAI_API_KEY is not set".into()))?;
        let deployment = std::env::var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME").map_err(|_| {
            Error::Configuration("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME is not set".into())
        })?;
        let mut client = Self::new(endpoint, deployment, api_key);
        if let Ok(v) = std::env::var("AZURE_OPENAI_API_VERSION") {
            client = client.with_api_version(v);
        }
        Ok(client)
    }

    /// Override the API version (default `"2024-10-21"`).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).api_version = api_version.into();
        self
    }

    /// The deployment name this client targets.
    pub fn deployment(&self) -> &str {
        &self.inner.deployment
    }

    /// The API version this client sends.
    pub fn api_version(&self) -> &str {
        &self.inner.api_version
    }

    fn url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/chat/completions?api-version={}",
            self.inner.endpoint.trim_end_matches('/'),
            self.inner.deployment,
            self.inner.api_version,
        )
    }

    /// Build the Chat Completions request body, reusing conversion from
    /// `agent-framework-openai` verbatim.
    fn build_body(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
        let mut body = Map::new();
        // The deployment in the URL already selects the model; only send
        // `model` if the caller explicitly asked for a specific one.
        if let Some(model) = &options.model_id {
            body.insert("model".into(), json!(model));
        }
        body.insert(
            "messages".into(),
            json!(agent_framework_openai::convert::messages_to_openai(
                messages
            )),
        );
        agent_framework_openai::convert::apply_options(&mut body, options);
        let (tools, tool_choice) = agent_framework_openai::convert::tools_to_openai(options);
        if let Some(tools) = tools {
            body.insert("tools".into(), tools);
        }
        if let Some(choice) = tool_choice {
            body.insert("tool_choice".into(), choice);
        }
        if stream {
            body.insert("stream".into(), json!(true));
            body.insert("stream_options".into(), json!({ "include_usage": true }));
        }
        Value::Object(body)
    }

    /// The header name/value pair to authenticate a request, per the
    /// client's configured [`Auth`] mode.
    async fn auth_header(&self) -> Result<(&'static str, String)> {
        match &self.inner.auth {
            Auth::ApiKey(key) => Ok(("api-key", key.clone())),
            Auth::Credential(credential) => {
                let token = credential.get_token().await?;
                Ok(("Authorization", format!("Bearer {token}")))
            }
        }
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let (header_name, header_value) = self.auth_header().await?;
        let resp = self
            .inner
            .http
            .post(self.url())
            .header(header_name, header_value)
            .json(body)
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            // Azure OpenAI is wire-compatible with OpenAI's Chat Completions,
            // so status/body classification (401/403 -> auth, 400/404/422 ->
            // invalid request or content filter) is shared verbatim with
            // `agent-framework-openai` rather than duplicated.
            return Err(agent_framework_openai::classify_service_error(
                status.as_u16(),
                &text,
                format!("Azure OpenAI API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for AzureOpenAIClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
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
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(agent_framework_openai::parse_sse_stream(resp).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.deployment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{
        Content, FinishReason, FunctionArguments, FunctionCallContent,
    };

    fn client() -> AzureOpenAIClient {
        AzureOpenAIClient::new("https://my-resource.openai.azure.com", "gpt-4o", "test-key")
    }

    // region: URL building

    #[test]
    fn url_includes_deployment_and_api_version() {
        let c = client();
        assert_eq!(
            c.url(),
            "https://my-resource.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-21"
        );
    }

    #[test]
    fn url_trims_trailing_slash_on_endpoint() {
        let c = AzureOpenAIClient::new(
            "https://my-resource.openai.azure.com/",
            "gpt-4o",
            "test-key",
        );
        assert!(c
            .url()
            .starts_with("https://my-resource.openai.azure.com/openai/"));
        assert!(!c.url().contains("azure.com//openai"));
    }

    #[test]
    fn with_api_version_overrides_default() {
        let c = client().with_api_version("2025-01-01-preview");
        assert!(c.url().ends_with("api-version=2025-01-01-preview"));
    }

    // endregion

    // region: auth header selection

    #[tokio::test]
    async fn api_key_auth_uses_api_key_header() {
        let c = client();
        let (name, value) = c.auth_header().await.unwrap();
        assert_eq!(name, "api-key");
        assert_eq!(value, "test-key");
    }

    #[tokio::test]
    async fn token_credential_auth_uses_bearer_header() {
        let credential = Arc::new(credential::StaticTokenCredential::new("my-jwt-token"));
        let c = AzureOpenAIClient::with_token_credential(
            "https://my-resource.openai.azure.com",
            "gpt-4o",
            credential,
        );
        let (name, value) = c.auth_header().await.unwrap();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-jwt-token");
    }

    // endregion

    // region: request body building (reuses agent-framework-openai::convert)

    #[test]
    fn build_body_omits_model_by_default() {
        let c = client();
        let body = c.build_body(&[ChatMessage::user("hi")], &ChatOptions::new(), false);
        assert!(body.get("model").is_none());
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": "hi" }])
        );
    }

    #[test]
    fn build_body_includes_model_when_explicitly_set() {
        let c = client();
        let options = ChatOptions::new().with_model("gpt-4o-override");
        let body = c.build_body(&[ChatMessage::user("hi")], &options, false);
        assert_eq!(body["model"], json!("gpt-4o-override"));
    }

    #[test]
    fn build_body_stream_includes_usage_option() {
        let c = client();
        let body = c.build_body(&[ChatMessage::user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({ "include_usage": true }));
    }

    #[test]
    fn build_body_function_call_round_trip() {
        let c = client();
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Raw("{}".to_string())),
        );
        let assistant_msg = ChatMessage::with_contents(
            agent_framework_core::types::Role::assistant(),
            vec![Content::FunctionCall(call)],
        );
        let body = c.build_body(
            &[ChatMessage::user("weather?"), assistant_msg],
            &ChatOptions::new(),
            false,
        );
        assert_eq!(
            body["messages"][1]["tool_calls"][0]["function"]["name"],
            json!("get_weather")
        );
    }

    // endregion

    // region: response parsing (reuses agent-framework-openai::convert)

    #[test]
    fn parse_response_reuses_openai_convert() {
        let value = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
        });
        let resp = agent_framework_openai::convert::parse_response(&value);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        assert_eq!(resp.usage_details.unwrap().total_token_count, Some(15));
    }

    // endregion

    // Streaming: `get_streaming_response` calls
    // `agent_framework_openai::parse_sse_stream(resp).boxed()` directly, with
    // no azure-specific logic of its own — the wiring is verified at compile
    // time (this crate wouldn't type-check against `ChatStream` otherwise),
    // and SSE parsing itself (text-only and tool-call fixtures, `[DONE]`
    // handling, error surfacing) is already covered by
    // `agent-framework-openai`'s own test suite. `reqwest::Response` can't be
    // constructed from raw bytes outside an actual HTTP exchange, so
    // reproducing those fixtures here would require standing up a mock
    // server rather than a plain unit test.

    // region: env-var constructor

    /// Guards Azure env var mutation across the tests below: `cargo test`
    /// runs tests in the same process on multiple threads, and env vars are
    /// process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_all_four_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com");
            std::env::set_var("AZURE_OPENAI_API_KEY", "test-key-123");
            std::env::set_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-deployment");
            std::env::set_var("AZURE_OPENAI_API_VERSION", "2025-02-01");
        }
        let client = AzureOpenAIClient::from_env().unwrap();
        assert_eq!(client.inner.endpoint, "https://res.openai.azure.com");
        assert_eq!(client.inner.deployment, "gpt-4o-deployment");
        assert_eq!(client.inner.api_version, "2025-02-01");
        assert!(matches!(client.inner.auth, Auth::ApiKey(ref k) if k == "test-key-123"));
        unsafe {
            std::env::remove_var("AZURE_OPENAI_ENDPOINT");
            std::env::remove_var("AZURE_OPENAI_API_KEY");
            std::env::remove_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME");
            std::env::remove_var("AZURE_OPENAI_API_VERSION");
        }
    }

    #[test]
    fn from_env_defaults_api_version_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com");
            std::env::set_var("AZURE_OPENAI_API_KEY", "test-key-123");
            std::env::set_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o-deployment");
            std::env::remove_var("AZURE_OPENAI_API_VERSION");
        }
        let client = AzureOpenAIClient::from_env().unwrap();
        assert_eq!(client.inner.api_version, DEFAULT_API_VERSION);
        unsafe {
            std::env::remove_var("AZURE_OPENAI_ENDPOINT");
            std::env::remove_var("AZURE_OPENAI_API_KEY");
            std::env::remove_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME");
        }
    }

    #[test]
    fn from_env_errors_when_deployment_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::set_var("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com");
            std::env::set_var("AZURE_OPENAI_API_KEY", "test-key-123");
            std::env::remove_var("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME");
            std::env::remove_var("AZURE_OPENAI_API_VERSION");
        }
        let result = AzureOpenAIClient::from_env();
        assert!(result.is_err());
        unsafe {
            std::env::remove_var("AZURE_OPENAI_ENDPOINT");
            std::env::remove_var("AZURE_OPENAI_API_KEY");
        }
    }

    // endregion
}
