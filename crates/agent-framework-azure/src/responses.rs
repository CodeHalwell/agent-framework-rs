//! [`AzureOpenAIResponsesClient`]: a [`ChatClient`] for the Responses API on
//! Azure OpenAI (`POST {endpoint}/openai/v1/responses`).
//!
//! ## URL shape and api-version
//!
//! Unlike [`AzureOpenAIClient`](crate::AzureOpenAIClient) (Chat Completions,
//! which selects the model via a deployment-scoped URL —
//! `.../openai/deployments/{deployment}/chat/completions`), the Responses API
//! on Azure OpenAI is documented upstream as supported only through the
//! newer, OpenAI-compatible "v1 preview" surface: there is no deployment
//! segment in the URL at all, and the deployment instead flows into the
//! request body's `model` field, exactly like the plain
//! [`OpenAIChatClient`](agent_framework_openai::responses::OpenAIChatClient).
//!
//! This mirrors upstream `AzureOpenAIResponsesClient.__init__`
//! (`azure/_responses_client.py:99-146`), which:
//! * forces `default_api_version="preview"` when building its settings
//!   (`_responses_client.py:112`) — distinct from every other Azure OpenAI
//!   client's `"2024-10-21"` default (`azure/_shared.py:28`,
//!   [`crate::AzureOpenAIClient`]'s own default);
//! * auto-derives `base_url = urljoin(endpoint, "/openai/v1/")` for standard
//!   `*.openai.azure.com` endpoints when no explicit `base_url` is given
//!   (`_responses_client.py:117-123`), and documents that "currently, the
//!   base_url must end with `/openai/v1/`" and "the api_version must be
//!   `preview`" (`_responses_client.py:60-65`);
//! * requires a deployment name, raising if one isn't configured
//!   (`_responses_client.py:127-131`).
//!
//! This client always derives the `/openai/v1/` route from `endpoint`
//! (skipping upstream's `.openai.azure.com`-hostname sniff, which its own
//! comment flags as "a temporary hack" for a case the Rust port doesn't need
//! to special-case); [`with_base_url`](AzureOpenAIResponsesClient::with_base_url)
//! is the escape hatch upstream's `base_url` parameter provides for full
//! control.
//!
//! ## Conversion and streaming
//!
//! Request/response conversion (messages → `input` items, tool specs, output
//! parsing, SSE event parsing) is reused verbatim from
//! [`agent_framework_openai::responses`] rather than duplicated — only the
//! URL shape, api-version default, and authentication differ, exactly as
//! [`AzureOpenAIClient`](crate::AzureOpenAIClient) reuses
//! [`agent_framework_openai::convert`] for Chat Completions. `conversation_id`
//! ↔ `previous_response_id` and `store` ↔ auto-populated `conversation_id`
//! behave identically to [`OpenAIChatClient`](agent_framework_openai::responses::OpenAIChatClient)
//! because the same conversion functions are called.
//!
//! ```no_run
//! use agent_framework_azure::responses::AzureOpenAIResponsesClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = AzureOpenAIResponsesClient::new(
//!     "https://my-resource.openai.azure.com",
//!     "my-gpt4o-deployment",
//!     "my-api-key",
//! );
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```
//!
//! Entra ID (bearer token) authentication instead of a static key — the same
//! [`TokenCredential`](crate::TokenCredential) plumbing
//! [`AzureOpenAIClient`](crate::AzureOpenAIClient) uses (e.g. the
//! `"https://cognitiveservices.azure.com/.default"` scope):
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_azure::StaticTokenCredential;
//! use agent_framework_azure::responses::AzureOpenAIResponsesClient;
//!
//! let credential = Arc::new(StaticTokenCredential::new("eyJ0eXAi..."));
//! let client = AzureOpenAIResponsesClient::with_token_credential(
//!     "https://my-resource.openai.azure.com",
//!     "my-gpt4o-deployment",
//!     credential,
//! );
//! ```

use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{ChatOptions, ChatResponse, Message};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use crate::{Auth, TokenCredential};

/// Default api-version for the Responses API on Azure OpenAI: the "preview"
/// v1-surface identifier upstream forces today
/// (`azure/_responses_client.py:112`, `default_api_version="preview"`),
/// distinct from [`AzureOpenAIClient`](crate::AzureOpenAIClient)'s Chat
/// Completions default (`"2024-10-21"`, `azure/_shared.py:28`). Overridable
/// via [`with_api_version`](AzureOpenAIResponsesClient::with_api_version) or
/// `AZURE_OPENAI_API_VERSION`.
const DEFAULT_API_VERSION: &str = "preview";

/// An Azure OpenAI Responses API chat client
/// (`POST {endpoint}/openai/v1/responses`).
///
/// See the [module docs](self) for the URL/api-version rationale.
pub struct AzureOpenAIResponsesClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    /// The resource endpoint, e.g. `https://my-resource.openai.azure.com`.
    /// Used to derive the `/openai/v1/` base URL when `base_url` is `None`.
    endpoint: String,
    /// An explicit override of the full base URL (mirrors upstream's
    /// `base_url` parameter), taking precedence over `endpoint` when set.
    base_url: Option<String>,
    deployment: String,
    api_version: Option<String>,
    auth: Auth,
}

impl Clone for AzureOpenAIResponsesClient {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl std::fmt::Debug for AzureOpenAIResponsesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureOpenAIResponsesClient")
            .field("endpoint", &self.inner.endpoint)
            .field("base_url", &self.inner.base_url)
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

impl AzureOpenAIResponsesClient {
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
                base_url: None,
                deployment: deployment.into(),
                api_version: Some(DEFAULT_API_VERSION.to_string()),
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
                base_url: None,
                deployment: deployment.into(),
                api_version: Some(DEFAULT_API_VERSION.to_string()),
                auth: Auth::Credential(credential),
            }),
        }
    }

    /// Build an API-key-authenticated client from `AZURE_OPENAI_ENDPOINT`,
    /// `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME`, and
    /// optional `AZURE_OPENAI_API_VERSION`/`AZURE_OPENAI_BASE_URL` — the same
    /// generic `AZURE_OPENAI_*` variables
    /// [`AzureOpenAIClient::from_env`](crate::AzureOpenAIClient::from_env)
    /// reads, except for the Responses-specific deployment variable (mirrors
    /// upstream's `responses_deployment_name` settings field, distinct from
    /// Chat Completions' `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`, so a resource
    /// with differently named deployments per API surface works;
    /// `azure/_shared.py:102-103`, docstring at
    /// `azure/_responses_client.py:53-56`).
    pub fn from_env() -> Result<Self> {
        Self::from_env_vars(|key| std::env::var(key).ok())
    }

    /// Implementation of [`from_env`](Self::from_env), parameterized over an
    /// environment lookup function.
    ///
    /// Kept separate so unit tests can exercise the parsing/validation logic
    /// against an in-memory map instead of mutating real process environment
    /// variables: those are process-global, and `AZURE_OPENAI_ENDPOINT`/
    /// `AZURE_OPENAI_API_KEY`/`AZURE_OPENAI_API_VERSION` are also read by
    /// [`AzureOpenAIClient::from_env`](crate::AzureOpenAIClient::from_env)'s
    /// own tests in `lib.rs`, which run concurrently under `cargo test` and
    /// guard only against races among themselves.
    fn from_env_vars(get: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let endpoint = get("AZURE_OPENAI_ENDPOINT")
            .ok_or_else(|| Error::Configuration("AZURE_OPENAI_ENDPOINT is not set".into()))?;
        let api_key = get("AZURE_OPENAI_API_KEY")
            .ok_or_else(|| Error::Configuration("AZURE_OPENAI_API_KEY is not set".into()))?;
        let deployment = get("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME").ok_or_else(|| {
            Error::Configuration("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME is not set".into())
        })?;
        let mut client = Self::new(endpoint, deployment, api_key);
        match get("AZURE_OPENAI_API_VERSION") {
            // Empty string opts out of the query parameter entirely (GA v1 /
            // gateway targets); see `without_api_version`.
            Some(v) if v.is_empty() => client = client.without_api_version(),
            Some(v) => client = client.with_api_version(v),
            None => {}
        }
        if let Some(b) = get("AZURE_OPENAI_BASE_URL") {
            client = client.with_base_url(b);
        }
        Ok(client)
    }

    /// Override the API version (default `"preview"`).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).api_version = Some(api_version.into());
        self
    }

    /// Send no `api-version` query parameter at all.
    ///
    /// Microsoft's v1 Responses examples call the bare
    /// `{endpoint}/openai/v1/responses` URL, and the v1 lifecycle removes
    /// dated api-version parameters — GA v1 resources and some
    /// OpenAI-compatible gateways reject or misroute requests carrying one.
    /// The `"preview"` default mirrors upstream Python
    /// (`_responses_client.py:112`); use this to produce the documented
    /// query-less URL instead. (Also reachable by setting the
    /// `AZURE_OPENAI_API_VERSION` env var to an empty string with
    /// [`from_env`](Self::from_env).)
    pub fn without_api_version(mut self) -> Self {
        Arc::make_mut(&mut self.inner).api_version = None;
        self
    }

    /// Override the full base URL used to build requests, taking precedence
    /// over the endpoint-derived `/openai/v1/` route. Mirrors upstream's
    /// `base_url` parameter, which "must end with `/openai/v1/`"
    /// (`azure/_responses_client.py:60-62`) — e.g. for a differently-shaped
    /// gateway or proxy in front of Azure OpenAI.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = Some(base_url.into());
        self
    }

    /// The deployment name this client targets (sent as the request body's
    /// `model` field; see the [module docs](self)).
    pub fn deployment(&self) -> &str {
        &self.inner.deployment
    }

    /// The API version this client sends, or `None` when the query
    /// parameter is omitted (see [`without_api_version`](Self::without_api_version)).
    pub fn api_version(&self) -> Option<&str> {
        self.inner.api_version.as_deref()
    }

    /// The effective base URL requests are built against: the explicit
    /// [`with_base_url`](Self::with_base_url) override when set, otherwise
    /// `None` (the endpoint-derived `/openai/v1/` route is computed lazily by
    /// [`url`](Self::url)).
    pub fn base_url(&self) -> Option<&str> {
        self.inner.base_url.as_deref()
    }

    fn url(&self) -> String {
        let base = match &self.inner.base_url {
            Some(explicit) => explicit.trim_end_matches('/').to_string(),
            None => format!("{}/openai/v1", self.inner.endpoint.trim_end_matches('/')),
        };
        match &self.inner.api_version {
            Some(v) => format!("{base}/responses?api-version={v}"),
            None => format!("{base}/responses"),
        }
    }

    /// Build the Responses API request body, reusing conversion from
    /// [`agent_framework_openai::responses`] verbatim — see the
    /// [module docs](self).
    fn build_body(&self, messages: &[Message], options: &ChatOptions, stream: bool) -> Value {
        let mut body = Map::new();
        // Unlike Chat Completions (deployment selects the model via the URL
        // path), the `/openai/v1/responses` route carries no deployment
        // segment, so `model` is the *only* way to select it and is always
        // sent — mirroring `OpenAIChatClient::build_body` and upstream's
        // `run_options["model"] = self.model` fallback
        // (`openai/_responses_client.py:432-435`).
        let model = options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.deployment.clone());
        body.insert("model".into(), json!(model));

        let (instructions, rest) = agent_framework_openai::responses::extract_instructions(
            messages,
            options.instructions.as_deref(),
        );
        if let Some(instructions) = instructions {
            body.insert("instructions".into(), json!(instructions));
        }
        body.insert(
            "input".into(),
            json!(agent_framework_openai::responses::messages_to_input(rest)),
        );

        if let Some(conversation_id) = &options.conversation_id {
            body.insert("previous_response_id".into(), json!(conversation_id));
        }
        if let Some(t) = options.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(t) = options.top_p {
            body.insert("top_p".into(), json!(t));
        }
        if let Some(mt) = options.max_tokens {
            body.insert("max_output_tokens".into(), json!(mt));
        }
        if let Some(store) = options.store {
            body.insert("store".into(), json!(store));
        }
        if let Some(user) = &options.user {
            body.insert("user".into(), json!(user));
        }
        if let Some(metadata) = &options.metadata {
            body.insert("metadata".into(), json!(metadata));
        }

        if !options.tools.is_empty() {
            let tools: Vec<Value> = options
                .tools
                .iter()
                .map(agent_framework_openai::responses::tool_to_responses_spec)
                .collect();
            body.insert("tools".into(), json!(tools));
            if let Some(allow_multi) = options.allow_multiple_tool_calls {
                body.insert("parallel_tool_calls".into(), json!(allow_multi));
            }
        }
        if let Some(tool_choice) = &options.tool_choice {
            body.insert(
                "tool_choice".into(),
                agent_framework_openai::responses::tool_choice_to_responses(tool_choice),
            );
        }
        if let Some(fmt) = &options.response_format {
            body.insert(
                "text".into(),
                json!({ "format": agent_framework_openai::responses::response_format_to_text(fmt) }),
            );
        }

        for (k, v) in &options.additional_properties {
            body.entry(k.clone()).or_insert_with(|| v.clone());
        }

        if stream {
            body.insert("stream".into(), json!(true));
        }
        Value::Object(body)
    }

    /// The header name/value pair to authenticate a request, per the
    /// client's configured [`Auth`] mode — identical logic to
    /// [`AzureOpenAIClient`](crate::AzureOpenAIClient)'s own `auth_header`.
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
            let retry_after = crate::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            // Shared with `agent-framework-openai`'s Responses client — see
            // `AzureOpenAIClient::post`.
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
impl ChatClient for AzureOpenAIResponsesClient {
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
        // Mirrors `OpenAIChatClient::get_response`: a failed run reports
        // `status: "failed"` with a 2xx HTTP status, so the error has to be
        // pulled out of the body rather than the transport layer —
        // content-filter failures get the granular variant.
        if let Some(err) = agent_framework_openai::responses::response_failure_error(&value) {
            return Err(err);
        }
        Ok(agent_framework_openai::responses::parse_response(
            &value,
            options.store,
        ))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(
            agent_framework_openai::responses::parse_responses_sse_stream(resp, options.store)
                .boxed(),
        )
    }

    fn model(&self) -> Option<&str> {
        Some(&self.inner.deployment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
    use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, ToolMode};

    fn client() -> AzureOpenAIResponsesClient {
        AzureOpenAIResponsesClient::new(
            "https://my-resource.openai.azure.com",
            "my-gpt4o-deployment",
            "test-key",
        )
    }

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    // region: URL building

    #[test]
    fn url_uses_v1_responses_route_with_default_preview_api_version() {
        let c = client();
        assert_eq!(
            c.url(),
            "https://my-resource.openai.azure.com/openai/v1/responses?api-version=preview"
        );
    }

    #[test]
    fn url_trims_trailing_slash_on_endpoint() {
        let c = AzureOpenAIResponsesClient::new(
            "https://my-resource.openai.azure.com/",
            "my-gpt4o-deployment",
            "test-key",
        );
        assert_eq!(
            c.url(),
            "https://my-resource.openai.azure.com/openai/v1/responses?api-version=preview"
        );
    }

    #[test]
    fn url_has_no_deployment_segment_unlike_chat_completions() {
        // Contrast with `AzureOpenAIClient::url()`, which *does* embed the
        // deployment in the path; the Responses client never does.
        let c = client();
        assert!(!c.url().contains("deployments"));
        assert!(!c.url().contains("my-gpt4o-deployment"));
    }

    #[test]
    fn with_api_version_overrides_default() {
        let c = client().with_api_version("2025-04-01-preview");
        assert!(c.url().ends_with("api-version=2025-04-01-preview"));
    }

    #[test]
    fn without_api_version_omits_the_query_parameter() {
        // GA v1 resources / OpenAI-compatible gateways use the documented
        // bare URL with no api-version query.
        let c = client().without_api_version();
        assert_eq!(
            c.url(),
            "https://my-resource.openai.azure.com/openai/v1/responses"
        );
        assert_eq!(c.api_version(), None);

        let gateway = client()
            .with_base_url("https://gateway.example.com/openai/v1/")
            .without_api_version();
        assert_eq!(
            gateway.url(),
            "https://gateway.example.com/openai/v1/responses"
        );
    }

    #[test]
    fn from_env_empty_api_version_opts_out_of_the_query() {
        let c = AzureOpenAIResponsesClient::from_env_vars(|k| match k {
            "AZURE_OPENAI_ENDPOINT" => Some("https://my-resource.openai.azure.com".into()),
            "AZURE_OPENAI_API_KEY" => Some("key".into()),
            "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME" => Some("dep".into()),
            "AZURE_OPENAI_API_VERSION" => Some(String::new()),
            _ => None,
        })
        .unwrap();
        assert!(!c.url().contains("api-version"));
    }

    #[test]
    fn with_base_url_overrides_derived_route() {
        let c = client().with_base_url("https://gateway.example.com/openai/v1/");
        assert_eq!(
            c.url(),
            "https://gateway.example.com/openai/v1/responses?api-version=preview"
        );
        assert_eq!(c.base_url(), Some("https://gateway.example.com/openai/v1/"));
    }

    #[test]
    fn accessors_report_configured_deployment_and_api_version() {
        let c = client();
        assert_eq!(c.deployment(), "my-gpt4o-deployment");
        assert_eq!(c.api_version(), Some("preview"));
        assert_eq!(c.model(), Some("my-gpt4o-deployment"));
        assert_eq!(c.base_url(), None);
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
        let credential = Arc::new(crate::StaticTokenCredential::new("my-jwt-token"));
        let c = AzureOpenAIResponsesClient::with_token_credential(
            "https://my-resource.openai.azure.com",
            "my-gpt4o-deployment",
            credential,
        );
        let (name, value) = c.auth_header().await.unwrap();
        assert_eq!(name, "Authorization");
        assert_eq!(value, "Bearer my-jwt-token");
    }

    // endregion

    // region: request-body parity with `agent_framework_openai::responses::OpenAIChatClient`
    //
    // These mirror the equivalent `build_body_*` tests in
    // `agent-framework-openai/src/responses.rs` field-for-field (substituting
    // the deployment name for `model`), proving this client *reuses* — rather
    // than reimplements — `extract_instructions`, `messages_to_input`,
    // `tool_to_responses_spec`, `tool_choice_to_responses`, and
    // `response_format_to_text`: any divergence in those shared functions
    // would show up here exactly as it would in the OpenAI crate's own tests.

    #[test]
    fn build_body_simple_text_matches_openai_shape_with_deployment_as_model() {
        let c = client();
        let body = c.build_body(&[user("Hello there")], &ChatOptions::new(), false);
        assert_eq!(
            body,
            json!({
                "model": "my-gpt4o-deployment",
                "input": [
                    { "type": "message", "role": "user", "content": [
                        { "type": "input_text", "text": "Hello there" }
                    ]}
                ],
            })
        );
    }

    #[test]
    fn build_body_model_always_present_unlike_chat_completions() {
        // Contrast with `AzureOpenAIClient::build_body`, which *omits*
        // `model` unless explicitly overridden (the deployment in its URL
        // already selects it); the Responses route has no such URL segment,
        // so `model` must always be sent.
        let c = client();
        let body = c.build_body(&[user("hi")], &ChatOptions::new(), false);
        assert_eq!(body["model"], json!("my-gpt4o-deployment"));
    }

    #[test]
    fn build_body_model_override_wins_over_deployment() {
        let c = client();
        let options = ChatOptions::new().with_model("gpt-4o-override");
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(body["model"], json!("gpt-4o-override"));
    }

    #[test]
    fn build_body_extracts_leading_system_message_as_instructions() {
        let c = client();
        let messages = vec![Message::system("Be terse."), user("Hi")];
        let body = c.build_body(&messages, &ChatOptions::new(), false);
        assert_eq!(body["instructions"], json!("Be terse."));
        assert_eq!(
            body["input"],
            json!([
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "Hi" }
                ]}
            ])
        );
    }

    #[test]
    fn build_body_function_call_round_trip() {
        let c = client();
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Raw(r#"{"city":"Paris"}"#.to_string())),
        );
        let assistant_msg = Message::with_contents(
            agent_framework_core::types::Role::assistant(),
            vec![Content::FunctionCall(call)],
        );
        let tool_msg = Message::with_contents(
            agent_framework_core::types::Role::tool(),
            vec![Content::FunctionResult(
                agent_framework_core::types::FunctionResultContent::new(
                    "call_1",
                    Some(json!("18C and sunny")),
                ),
            )],
        );
        let body = c.build_body(
            &[user("weather?"), assistant_msg, tool_msg],
            &ChatOptions::new(),
            false,
        );
        assert_eq!(
            body["input"],
            json!([
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "weather?" }
                ]},
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "18C and sunny" },
            ])
        );
    }

    #[test]
    fn build_body_tools_are_flat_not_nested() {
        let c = client();
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get the weather".into(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        };
        let options = ChatOptions::new().with_tool(tool);
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": { "type": "object", "properties": {} },
            }])
        );
    }

    #[test]
    fn build_body_tool_choice_required_named() {
        let c = client();
        let options =
            ChatOptions::new().with_tool_choice(ToolMode::Required(Some("get_weather".into())));
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "function", "name": "get_weather" })
        );
    }

    #[test]
    fn build_body_conversation_id_becomes_previous_response_id() {
        let c = client();
        let mut options = ChatOptions::new();
        options.conversation_id = Some("resp_abc123".into());
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(body["previous_response_id"], json!("resp_abc123"));
    }

    #[test]
    fn build_body_max_tokens_becomes_max_output_tokens() {
        let c = client();
        let options = ChatOptions::new().with_max_tokens(256);
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(body["max_output_tokens"], json!(256));
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn build_body_stream_sets_stream_flag_without_stream_options() {
        // Unlike Chat Completions, the Responses API needs no
        // `stream_options.include_usage` toggle: usage arrives on the
        // `response.completed` event unconditionally.
        let c = client();
        let body = c.build_body(&[user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], json!(true));
        assert!(body.get("stream_options").is_none());
    }

    // endregion

    // region: response parsing (reuses agent_framework_openai::responses::parse_response)

    #[test]
    fn parse_response_reuses_openai_responses_convert() {
        let value = json!({
            "id": "resp_abc123",
            "model": "my-gpt4o-deployment",
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{ "type": "output_text", "text": "Hello!" }],
            }],
            "usage": { "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 },
        });
        let resp = agent_framework_openai::responses::parse_response(&value, None);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.response_id.as_deref(), Some("resp_abc123"));
        // `store != Some(false)` defaults `conversation_id` to the response
        // id, identical to `OpenAIChatClient`.
        assert_eq!(resp.conversation_id.as_deref(), Some("resp_abc123"));
        assert_eq!(resp.usage_details.unwrap().total_token_count, Some(15));
    }

    // endregion

    // Streaming: `get_streaming_response` calls
    // `agent_framework_openai::responses::parse_responses_sse_stream(resp,
    // options.store).boxed()` directly, with no azure-specific logic of its
    // own — the wiring is verified at compile time (this crate wouldn't
    // type-check against `ChatStream` otherwise). A real SSE round trip is
    // covered by the loopback test in `tests/credentials_loopback.rs`
    // (`azure_openai_responses_client_streams_sse_events`); the event-parsing
    // logic itself is exercised by `agent-framework-openai`'s own test suite.

    // region: env-var constructor

    #[test]
    fn from_env_reads_all_vars() {
        let vars = [
            ("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com"),
            ("AZURE_OPENAI_API_KEY", "test-key-123"),
            (
                "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME",
                "gpt-4o-responses-deployment",
            ),
            ("AZURE_OPENAI_API_VERSION", "2025-05-01-preview"),
            (
                "AZURE_OPENAI_BASE_URL",
                "https://gateway.example.com/openai/v1/",
            ),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

        let client =
            AzureOpenAIResponsesClient::from_env_vars(|k| vars.get(k).map(|v| v.to_string()))
                .unwrap();
        assert_eq!(client.inner.endpoint, "https://res.openai.azure.com");
        assert_eq!(client.inner.deployment, "gpt-4o-responses-deployment");
        assert_eq!(
            client.inner.api_version.as_deref(),
            Some("2025-05-01-preview")
        );
        assert_eq!(
            client.inner.base_url.as_deref(),
            Some("https://gateway.example.com/openai/v1/")
        );
        assert!(matches!(client.inner.auth, Auth::ApiKey(ref k) if k == "test-key-123"));
    }

    #[test]
    fn from_env_defaults_api_version_to_preview_when_unset() {
        let vars = [
            ("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com"),
            ("AZURE_OPENAI_API_KEY", "test-key-123"),
            (
                "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME",
                "gpt-4o-responses-deployment",
            ),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

        let client =
            AzureOpenAIResponsesClient::from_env_vars(|k| vars.get(k).map(|v| v.to_string()))
                .unwrap();
        assert_eq!(
            client.inner.api_version.as_deref(),
            Some(DEFAULT_API_VERSION)
        );
        assert_eq!(client.inner.base_url, None);
    }

    #[test]
    fn from_env_errors_when_responses_deployment_missing() {
        let vars = [
            ("AZURE_OPENAI_ENDPOINT", "https://res.openai.azure.com"),
            ("AZURE_OPENAI_API_KEY", "test-key-123"),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

        let err = AzureOpenAIResponsesClient::from_env_vars(|k| vars.get(k).map(|v| v.to_string()))
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME"));
    }

    #[test]
    fn from_env_errors_when_endpoint_missing() {
        let vars = [
            ("AZURE_OPENAI_API_KEY", "test-key-123"),
            (
                "AZURE_OPENAI_RESPONSES_DEPLOYMENT_NAME",
                "gpt-4o-responses-deployment",
            ),
        ]
        .into_iter()
        .collect::<std::collections::HashMap<_, _>>();

        let err = AzureOpenAIResponsesClient::from_env_vars(|k| vars.get(k).map(|v| v.to_string()))
            .unwrap_err();
        assert!(err.to_string().contains("AZURE_OPENAI_ENDPOINT"));
    }

    // `from_env()` itself is intentionally not exercised here against real
    // process env vars: it is a one-line call to `from_env_vars` (verified
    // exhaustively above), and env vars are process-global — this crate's
    // `AzureOpenAIClient::from_env` tests (`lib.rs`) already mutate the exact
    // same `AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_API_KEY`/
    // `AZURE_OPENAI_API_VERSION` names under their own mutex, and `cargo
    // test` runs both modules' tests concurrently in one binary. Testing
    // through the injectable `from_env_vars` seam instead gives the same
    // coverage with no risk of racing those tests.

    // endregion
}
