//! [`AnthropicBedrockClient`]: Anthropic (Claude) models hosted on
//! [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html),
//! spoken over Bedrock's `InvokeModel` API rather than Bedrock's own
//! model-agnostic Converse API
//! (`POST https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`).
//!
//! This is a *transport*, not a new wire format: the request/response body is
//! the same Anthropic Messages API shape [`AnthropicClient`](crate::AnthropicClient)
//! speaks directly, built via [`crate::convert::build_cloud_request`] (which
//! carries `anthropic_version: "bedrock-2023-05-31"` in the body instead of a
//! top-level `model` field, since the model is already selected by the URL).
//! Only the transport differs: requests are
//! [SigV4](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)-signed
//! rather than authenticated with an `x-api-key` header. Signing itself is
//! **not** reimplemented here — it's delegated to
//! [`agent_framework_bedrock::sigv4`], the same hand-rolled, test-vector-verified
//! implementation [`agent_framework_bedrock::BedrockChatClient`] uses for its
//! own (Converse API) requests.
//!
//! ```no_run
//! use agent_framework_anthropic::bedrock::AnthropicBedrockClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = AnthropicBedrockClient::from_env("anthropic.claude-3-5-sonnet-20241022-v2:0")?;
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_framework_bedrock::sigv4::{self, SigV4Params};
use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Message, Role, UsageContent,
};
use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::convert;

/// The `anthropic_version` Bedrock's `InvokeModel` API expects in the body of
/// every Claude request (Anthropic's
/// [Bedrock integration docs](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock)).
pub const ANTHROPIC_BEDROCK_VERSION: &str = "bedrock-2023-05-31";

/// `max_tokens` is required by the Anthropic Messages API; used whenever
/// neither `ChatOptions::max_tokens` nor a client-level override is set.
/// Matches [`crate::AnthropicClient`]'s default.
const DEFAULT_MAX_TOKENS: u32 = 1024;

const AWS_REGION_ENV: &str = "AWS_REGION";
const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";
const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";
const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";
const AWS_SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";
const DEFAULT_REGION: &str = "us-east-1";
const SIGV4_SERVICE: &str = "bedrock";

/// Build the `bedrock-runtime` host for a region (e.g.
/// `bedrock-runtime.us-east-1.amazonaws.com`).
fn bedrock_host(region: &str) -> String {
    format!("bedrock-runtime.{region}.amazonaws.com")
}

/// Percent-encode a Bedrock model id for use as a URI path segment, per
/// SigV4's `UriEncode` rules (uppercase `%XX` escapes; unreserved characters
/// `A-Za-z0-9-_.~` pass through unescaped). Hand-rolled rather than pulling
/// in the `percent-encoding` crate, matching [`sigv4`]'s own no-extra-crates
/// philosophy for small, self-contained encodings.
///
/// Model ids containing a literal `/` (e.g. cross-region inference-profile
/// ids) are not expected here — Anthropic's own models on Bedrock use plain
/// `provider.model-version` ids with no path separators — so `/` is escaped
/// like any other reserved character; this mirrors what SigV4 canonicalization
/// requires for a *single* path segment.
fn uri_encode_model(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for byte in model.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// The `InvokeModel` path for a given model id.
fn invoke_path(model: &str) -> String {
    format!("/model/{}/invoke", uri_encode_model(model))
}

/// Classify a non-success Bedrock `InvokeModel` HTTP response into a granular
/// [`Error`]. Mirrors [`agent_framework_bedrock`]'s own `classify_bedrock_error`
/// (not reusable directly — it's private to that crate): `401`/`403` ->
/// invalid auth, `400` -> invalid request, everything else (notably `429`
/// throttling and `5xx`) stays a generic, retry-layer-visible
/// [`Error::ServiceStatus`].
fn classify_invoke_error(status: u16, message: impl Into<String>) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 => Error::service_invalid_request(message),
        _ => Error::service_status(status, message, None),
    }
}

/// An Anthropic Messages API transport for Claude models on AWS Bedrock
/// (`POST https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke`).
///
/// See the [module docs](self) for how this relates to
/// [`AnthropicClient`](crate::AnthropicClient) and
/// [`agent_framework_bedrock::BedrockChatClient`].
#[derive(Clone)]
pub struct AnthropicBedrockClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    region: String,
    /// The `bedrock-runtime` host derived from `region`.
    host: String,
    model: String,
    access_key_id: String,
    secret_access_key: String,
    /// Present when using temporary/STS credentials.
    session_token: Option<String>,
    max_tokens: u32,
}

impl std::fmt::Debug for AnthropicBedrockClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicBedrockClient")
            .field("region", &self.inner.region)
            .field("model", &self.inner.model)
            .field("max_tokens", &self.inner.max_tokens)
            .field("has_session_token", &self.inner.session_token.is_some())
            .finish_non_exhaustive()
    }
}

impl AnthropicBedrockClient {
    /// Create a client for the given static AWS credentials, region, and
    /// default model id.
    pub fn new(
        access_key_id: impl Into<String>,
        secret_access_key: impl Into<String>,
        region: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        let region = region.into();
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                host: bedrock_host(&region),
                region,
                model: model.into(),
                access_key_id: access_key_id.into(),
                secret_access_key: secret_access_key.into(),
                session_token: None,
                max_tokens: DEFAULT_MAX_TOKENS,
            }),
        }
    }

    /// Build a client from the standard AWS environment variables:
    /// `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are required;
    /// `AWS_SESSION_TOKEN` is read when present (temporary/STS credentials);
    /// the region is `AWS_REGION`, falling back to `AWS_DEFAULT_REGION`,
    /// falling back to `us-east-1` when neither is set.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let access_key_id = std::env::var(AWS_ACCESS_KEY_ID_ENV)
            .map_err(|_| Error::Configuration(format!("{AWS_ACCESS_KEY_ID_ENV} is not set")))?;
        let secret_access_key = std::env::var(AWS_SECRET_ACCESS_KEY_ENV)
            .map_err(|_| Error::Configuration(format!("{AWS_SECRET_ACCESS_KEY_ENV} is not set")))?;
        let region = std::env::var(AWS_REGION_ENV)
            .or_else(|_| std::env::var(AWS_DEFAULT_REGION_ENV))
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());

        let mut client = Self::new(access_key_id, secret_access_key, region, model);
        if let Ok(token) = std::env::var(AWS_SESSION_TOKEN_ENV) {
            if !token.trim().is_empty() {
                client = client.with_session_token(token);
            }
        }
        Ok(client)
    }

    /// Set an AWS session token (for temporary/STS credentials).
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).session_token = Some(session_token.into());
        self
    }

    /// Override the region (and derive a new signing host from it).
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        let region = region.into();
        let inner = Arc::make_mut(&mut self.inner);
        inner.host = bedrock_host(&region);
        inner.region = region;
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

    /// The configured AWS region.
    pub fn region(&self) -> &str {
        &self.inner.region
    }

    fn effective_model(&self, options: &ChatOptions) -> String {
        options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone())
    }

    /// Sign and POST a request body to `/model/{model}/invoke`, returning the
    /// raw response on a success status.
    async fn send(&self, model: &str, body: &Value) -> Result<reqwest::Response> {
        let path = invoke_path(model);
        let url = format!("https://{}{path}", self.inner.host);
        let payload = serde_json::to_vec(body).map_err(|e| {
            Error::Serialization(format!("failed to serialize Bedrock invoke request: {e}"))
        })?;

        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let (amz_date, date_stamp) = sigv4::amz_dates_from_unix(secs);

        let params = SigV4Params {
            access_key: &self.inner.access_key_id,
            secret_key: &self.inner.secret_access_key,
            session_token: self.inner.session_token.as_deref(),
            region: &self.inner.region,
            service: SIGV4_SERVICE,
            host: &self.inner.host,
            method: "POST",
            canonical_uri: &path,
            canonical_query: "",
            payload: &payload,
            amz_date: &amz_date,
            date_stamp: &date_stamp,
        };
        let (authorization, extra_headers) = sigv4::authorization_header(&params);

        let mut request = self
            .inner
            .http
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, authorization)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .body(payload);
        for (name, value) in extra_headers {
            request = request.header(name, value);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(classify_invoke_error(
                status.as_u16(),
                format!("Bedrock InvokeModel error {status}: {text}"),
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for AnthropicBedrockClient {
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
            ANTHROPIC_BEDROCK_VERSION,
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
    /// Bedrock's `invoke-with-response-stream` counterpart uses AWS's binary
    /// event-stream framing (length-prefixed, CRC-checked frames), not
    /// line-delimited SSE — parsing it needs a dedicated binary frame
    /// decoder, which is a documented extension point rather than
    /// implemented here (the same tactic
    /// [`agent_framework_bedrock::BedrockChatClient::get_streaming_response`]
    /// takes for the Converse API). This method calls the non-streaming
    /// `invoke` endpoint (the same request [`ChatClient::get_response`]
    /// sends) and adapts the complete [`ChatResponse`] into a single
    /// [`ChatResponseUpdate`]. Callers driving this client through
    /// [`ChatResponse::from_updates`](agent_framework_core::types::ChatResponse::from_updates)
    /// still get a correct aggregated result; they just don't see partial
    /// text arrive incrementally.
    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let response = self.get_response(messages, options).await?;

        let mut contents: Vec<_> = response
            .messages
            .iter()
            .flat_map(|m| m.contents.iter().cloned())
            .collect();
        if let Some(usage) = response.usage_details.clone() {
            contents.push(agent_framework_core::types::Content::Usage(UsageContent {
                details: usage,
            }));
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

    fn client() -> AnthropicBedrockClient {
        AnthropicBedrockClient::new(
            "AKIDEXAMPLE",
            "secret",
            "us-east-1",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
        )
    }

    #[test]
    fn invoke_path_encodes_colon_but_preserves_dots() {
        assert_eq!(
            invoke_path("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke"
        );
    }

    #[test]
    fn new_derives_host_from_region() {
        assert_eq!(
            client().inner.host,
            "bedrock-runtime.us-east-1.amazonaws.com"
        );
    }

    #[test]
    fn with_region_overrides_region_and_host() {
        let c = client().with_region("eu-west-1");
        assert_eq!(c.region(), "eu-west-1");
        assert_eq!(c.inner.host, "bedrock-runtime.eu-west-1.amazonaws.com");
    }

    #[test]
    fn model_and_region_accessors() {
        let c = client();
        assert_eq!(c.model(), "anthropic.claude-3-5-sonnet-20241022-v2:0");
        assert_eq!(c.region(), "us-east-1");
        assert_eq!(
            ChatClient::model(&c),
            Some("anthropic.claude-3-5-sonnet-20241022-v2:0")
        );
    }

    #[test]
    fn request_url_targets_invoke_endpoint() {
        let c = client();
        let path = invoke_path(&c.effective_model(&ChatOptions::new()));
        let url = format!("https://{}{path}", c.inner.host);
        assert_eq!(
            url,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/\
             anthropic.claude-3-5-sonnet-20241022-v2%3A0/invoke"
        );
    }

    #[test]
    fn request_body_has_bedrock_anthropic_version_and_no_model_key() {
        let body = convert::build_cloud_request(
            &[Message::user("hi")],
            &ChatOptions::new(),
            1024,
            false,
            ANTHROPIC_BEDROCK_VERSION,
        );
        assert_eq!(
            body["anthropic_version"],
            serde_json::json!("bedrock-2023-05-31")
        );
        assert!(body.get("model").is_none());
    }

    // region: env-var constructor

    /// Guards AWS env-var mutation: tests within a crate run on multiple
    /// threads, and env vars are process-global.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_aws_env() {
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module.
        unsafe {
            std::env::remove_var(AWS_ACCESS_KEY_ID_ENV);
            std::env::remove_var(AWS_SECRET_ACCESS_KEY_ENV);
            std::env::remove_var(AWS_SESSION_TOKEN_ENV);
            std::env::remove_var(AWS_REGION_ENV);
            std::env::remove_var(AWS_DEFAULT_REGION_ENV);
        }
    }

    #[test]
    fn from_env_errors_without_credentials() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        let result = AnthropicBedrockClient::from_env("anthropic.claude-3-haiku");
        assert!(matches!(result, Err(Error::Configuration(_))));
        clear_aws_env();
    }

    #[test]
    fn from_env_reads_credentials_and_region() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_aws_env.
        unsafe {
            std::env::set_var(AWS_ACCESS_KEY_ID_ENV, "AKIDEXAMPLE");
            std::env::set_var(AWS_SECRET_ACCESS_KEY_ENV, "secret-123");
            std::env::set_var(AWS_REGION_ENV, "eu-central-1");
        }
        let client = AnthropicBedrockClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(client.inner.access_key_id, "AKIDEXAMPLE");
        assert_eq!(client.region(), "eu-central-1");
        clear_aws_env();
    }

    #[test]
    fn from_env_defaults_region_when_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_aws_env.
        unsafe {
            std::env::set_var(AWS_ACCESS_KEY_ID_ENV, "AKIDEXAMPLE");
            std::env::set_var(AWS_SECRET_ACCESS_KEY_ENV, "secret-123");
        }
        let client = AnthropicBedrockClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(client.region(), DEFAULT_REGION);
        clear_aws_env();
    }

    // endregion

    // region: error classification

    #[test]
    fn classify_invoke_error_maps_auth_and_invalid_request() {
        assert!(matches!(
            classify_invoke_error(401, "denied"),
            Error::ServiceInvalidAuth { .. }
        ));
        assert!(matches!(
            classify_invoke_error(403, "denied"),
            Error::ServiceInvalidAuth { .. }
        ));
        assert!(matches!(
            classify_invoke_error(400, "bad request"),
            Error::ServiceInvalidRequest { .. }
        ));
    }

    #[test]
    fn classify_invoke_error_leaves_throttling_and_5xx_as_service_status() {
        assert_eq!(classify_invoke_error(429, "throttled").status(), Some(429));
        assert_eq!(classify_invoke_error(500, "boom").status(), Some(500));
    }

    // endregion
}
