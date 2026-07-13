//! # agent-framework-bedrock
//!
//! An [AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock.html)
//! [`ChatClient`] for
//! `agent-framework-rs`, built on Bedrock's model-agnostic
//! [Converse API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
//! (`POST https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse`).
//!
//! Unlike the OpenAI-compatible providers in this workspace, Bedrock has its
//! own wire format (hence no dependency on `agent-framework-openai`) and
//! authenticates requests with
//! [AWS Signature Version 4](https://docs.aws.amazon.com/general/latest/gr/signature-version-4.html)
//! rather than a bearer token — see [`sigv4`] for the signing implementation
//! (hand-rolled SHA256/HMAC-SHA256-based canonical-request signing, verified
//! against AWS's own published SigV4 test-suite vector), and [`convert`] for
//! the request/response conversion between this crate's message types and
//! Bedrock's Converse API JSON shapes.
//!
//! ```no_run
//! use agent_framework_bedrock::BedrockChatClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = BedrockChatClient::from_env("anthropic.claude-3-5-sonnet-20241022-v2:0")?;
//! let agent = Agent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

pub mod convert;
pub mod sigv4;

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, ChatResponseUpdate, Message, Role, UsageContent,
};
use futures::stream::{self, StreamExt};
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use serde_json::Value;

/// The environment variable read for the AWS region (e.g. `us-east-1`).
const AWS_REGION_ENV: &str = "AWS_REGION";

/// The fallback environment variable for the AWS region, checked when
/// [`AWS_REGION_ENV`] is unset (mirrors the AWS CLI/SDKs, which honor both).
const AWS_DEFAULT_REGION_ENV: &str = "AWS_DEFAULT_REGION";

/// The environment variable read for the AWS access key id.
const AWS_ACCESS_KEY_ID_ENV: &str = "AWS_ACCESS_KEY_ID";

/// The environment variable read for the AWS secret access key.
const AWS_SECRET_ACCESS_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";

/// The environment variable read for an optional AWS session token
/// (required when using temporary/STS credentials).
const AWS_SESSION_TOKEN_ENV: &str = "AWS_SESSION_TOKEN";

/// The region used by [`BedrockChatClient::from_env`] when neither
/// [`AWS_REGION_ENV`] nor [`AWS_DEFAULT_REGION_ENV`] is set.
const DEFAULT_REGION: &str = "us-east-1";

/// The SigV4 service name for Bedrock Runtime.
const SIGV4_SERVICE: &str = "bedrock";

/// Build the `bedrock-runtime` host for a region (e.g.
/// `bedrock-runtime.us-east-1.amazonaws.com`).
fn bedrock_host(region: &str) -> String {
    format!("bedrock-runtime.{region}.amazonaws.com")
}

/// The characters SigV4's `UriEncode` leaves unescaped in a path segment:
/// alphanumerics plus `- _ . ~` (the unreserved set), and — for this
/// specific use, encoding a whole path rather than one segment — `/` itself,
/// so an inference-profile ARN's slashes stay literal path separators rather
/// than becoming `%2F`.
const AWS_PATH_UNRESERVED: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~')
    .remove(b'/');

/// URI-encode a Bedrock model id (or inference-profile ARN) for use as a
/// path segment, per SigV4's `UriEncode` rules (uppercase `%XX` escapes,
/// unreserved characters passed through). The *same* encoded string is used
/// both to build the actual request path and as the canonical URI fed to
/// [`sigv4::authorization_header`] — they must match exactly, or the
/// server-side signature recomputation fails.
fn uri_encode_model_id(model: &str) -> String {
    utf8_percent_encode(model, AWS_PATH_UNRESERVED).to_string()
}

/// The Converse API path for a given model id.
fn converse_path(model: &str) -> String {
    format!("/model/{}/converse", uri_encode_model_id(model))
}

/// Parse a `Retry-After` header into a delay in seconds, when present.
fn parse_retry_after(headers: &reqwest::header::HeaderMap) -> Option<f64> {
    headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
}

/// Classify a non-success Bedrock Runtime HTTP response into a granular
/// [`Error`].
///
/// Bedrock signals the exception type via the `x-amzn-errortype` response
/// header (e.g. `ValidationException`, `AccessDeniedException`,
/// `ThrottlingException`) rather than in the JSON body's `error.status` the
/// way Gemini does, so classification is driven by HTTP status plus that
/// header:
///
/// * `401` / `403` -> [`Error::ServiceInvalidAuth`] (`UnrecognizedClientException`,
///   `AccessDeniedException`)
/// * `400` -> [`Error::ServiceInvalidRequest`] (`ValidationException` is the
///   overwhelmingly common 400 cause; a 400 is never retryable regardless of
///   the specific exception name)
/// * anything else — notably `429` (`ThrottlingException`) and `5xx`, which
///   the retry layer depends on — -> [`Error::ServiceStatus`], unchanged
///
/// Bedrock has no content-filter-specific HTTP error either: a
/// guardrail-blocked response is a `200 OK` with `stopReason:
/// "guardrail_intervened"` or `"content_filtered"`, mapped to
/// `FinishReason::CONTENT_FILTER` by [`convert::parse_response`] rather than
/// raised as an error, so [`Error::ServiceContentFilter`] is never
/// constructed on this path.
fn classify_bedrock_error(
    status: u16,
    message: impl Into<String>,
    retry_after: Option<f64>,
) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 => Error::service_invalid_request(message),
        _ => Error::service_status(status, message, retry_after),
    }
}

/// An AWS Bedrock Converse API chat client
/// (`POST https://{host}/model/{model}/converse`).
#[derive(Clone)]
pub struct BedrockChatClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    region: String,
    model: String,
    access_key_id: String,
    secret_access_key: String,
    /// Present when using temporary/STS credentials.
    session_token: Option<String>,
    /// The `bedrock-runtime` host derived from `region`
    /// (e.g. `bedrock-runtime.us-east-1.amazonaws.com`).
    host: String,
}

impl std::fmt::Debug for BedrockChatClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockChatClient")
            .field("region", &self.inner.region)
            .field("model", &self.inner.model)
            .field("host", &self.inner.host)
            .field("has_session_token", &self.inner.session_token.is_some())
            .finish_non_exhaustive()
    }
}

impl BedrockChatClient {
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
            }),
        }
    }

    /// Build a client from the standard AWS environment variables:
    /// `AWS_ACCESS_KEY_ID_ENV` (`AWS_ACCESS_KEY_ID`) and
    /// `AWS_SECRET_ACCESS_KEY_ENV` (`AWS_SECRET_ACCESS_KEY`) are required;
    /// `AWS_SESSION_TOKEN_ENV` (`AWS_SESSION_TOKEN`) is read when present
    /// (temporary/STS credentials); the region is `AWS_REGION_ENV`
    /// (`AWS_REGION`), falling back to `AWS_DEFAULT_REGION_ENV`
    /// (`AWS_DEFAULT_REGION`), falling back to `DEFAULT_REGION`
    /// (`us-east-1`) when neither is set.
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

    /// Override the region (and derive a new `host` from it).
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        let region = region.into();
        let inner = Arc::make_mut(&mut self.inner);
        inner.host = bedrock_host(&region);
        inner.region = region;
        self
    }

    /// Set an AWS session token (for temporary/STS credentials).
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).session_token = Some(session_token.into());
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

    /// Sign and POST a Converse request body to `/model/{model}/converse`,
    /// returning the raw response on a success status (classifying a
    /// non-success status into a granular [`Error`] via
    /// [`classify_bedrock_error`]).
    async fn send(&self, model: &str, payload: &[u8]) -> Result<reqwest::Response> {
        let path = converse_path(model);
        let url = format!("https://{}{path}", self.inner.host);

        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let (amz_date, date_stamp) = sigv4::amz_dates_from_unix(secs);

        let params = sigv4::SigV4Params {
            access_key: &self.inner.access_key_id,
            secret_key: &self.inner.secret_access_key,
            session_token: self.inner.session_token.as_deref(),
            region: &self.inner.region,
            service: SIGV4_SERVICE,
            host: &self.inner.host,
            method: "POST",
            canonical_uri: &path,
            canonical_query: "",
            payload,
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
            .body(payload.to_vec());
        for (name, value) in extra_headers {
            request = request.header(name, value);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_type = resp
                .headers()
                .get("x-amzn-errortype")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let message = match &error_type {
                Some(et) => format!("Bedrock Converse API error {status} ({et}): {text}"),
                None => format!("Bedrock Converse API error {status}: {text}"),
            };
            return Err(classify_bedrock_error(
                status.as_u16(),
                message,
                retry_after,
            ));
        }
        Ok(resp)
    }

    fn effective_model(&self, options: &ChatOptions) -> String {
        options
            .model
            .clone()
            .unwrap_or_else(|| self.inner.model.clone())
    }
}

#[async_trait::async_trait]
impl ChatClient for BedrockChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let model = self.effective_model(&options);
        let body = convert::build_request(&messages, &options);
        let payload = serde_json::to_vec(&body).map_err(|e| {
            Error::Serialization(format!("failed to serialize Bedrock Converse request: {e}"))
        })?;

        let resp = self.send(&model, &payload).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        Ok(convert::parse_response(&value))
    }

    /// Get a streaming response.
    ///
    /// Bedrock's `converse-stream` endpoint uses AWS's binary
    /// [event-stream framing](https://docs.aws.amazon.com/AmazonS3/latest/API/RESTSelectObjectAppendix.html)
    /// (length-prefixed, CRC-checked frames), not line-delimited SSE like
    /// every other streaming provider in this workspace — parsing it
    /// correctly needs a dedicated binary frame decoder, which is
    /// significant additional surface for a first cut of this client. This
    /// method therefore calls the non-streaming `converse` endpoint (the
    /// same request [`ChatClient::get_response`] sends) and adapts the
    /// complete [`ChatResponse`] into a single [`ChatResponseUpdate`] rather
    /// than a true incremental stream. Callers driving this client through
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

    fn client() -> BedrockChatClient {
        BedrockChatClient::new(
            "AKIDEXAMPLE",
            "secret",
            "us-east-1",
            "anthropic.claude-3-haiku",
        )
    }

    // region: host / path construction

    #[test]
    fn new_derives_host_from_region() {
        let c = client();
        assert_eq!(c.inner.host, "bedrock-runtime.us-east-1.amazonaws.com");
    }

    #[test]
    fn with_region_overrides_region_and_host() {
        let c = client().with_region("eu-west-1");
        assert_eq!(c.region(), "eu-west-1");
        assert_eq!(c.inner.host, "bedrock-runtime.eu-west-1.amazonaws.com");
    }

    #[test]
    fn converse_path_encodes_colon_but_preserves_dots_and_slashes() {
        assert_eq!(
            converse_path("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"
        );
        // Inference-profile ARNs contain literal slashes and colons; only
        // the colons are percent-encoded.
        assert_eq!(
            converse_path("arn:aws:bedrock:us-east-1:111122223333:inference-profile/my-profile"),
            "/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A111122223333%3Ainference-profile/my-profile/converse"
        );
    }

    #[test]
    fn model_returns_default_model() {
        let c = client();
        assert_eq!(c.model(), "anthropic.claude-3-haiku");
        assert_eq!(ChatClient::model(&c), Some("anthropic.claude-3-haiku"));
    }

    #[test]
    fn effective_model_prefers_per_request_override() {
        let c = client();
        let options = ChatOptions::new().with_model("anthropic.claude-3-opus");
        assert_eq!(c.effective_model(&options), "anthropic.claude-3-opus");
        assert_eq!(
            c.effective_model(&ChatOptions::new()),
            "anthropic.claude-3-haiku"
        );
    }

    // endregion

    // region: error classification

    #[test]
    fn classify_bedrock_error_maps_auth_statuses() {
        assert!(matches!(
            classify_bedrock_error(401, "denied", None),
            Error::ServiceInvalidAuth { .. }
        ));
        assert!(matches!(
            classify_bedrock_error(403, "denied", None),
            Error::ServiceInvalidAuth { .. }
        ));
    }

    #[test]
    fn classify_bedrock_error_maps_400_to_invalid_request() {
        assert!(matches!(
            classify_bedrock_error(400, "bad request", None),
            Error::ServiceInvalidRequest { .. }
        ));
    }

    #[test]
    fn classify_bedrock_error_maps_throttling_and_5xx_to_service_status() {
        let throttled = classify_bedrock_error(429, "throttled", Some(2.0));
        assert_eq!(throttled.status(), Some(429));
        assert_eq!(throttled.retry_after(), Some(2.0));

        let server_error = classify_bedrock_error(500, "boom", None);
        assert_eq!(server_error.status(), Some(500));
    }

    // endregion

    // region: env-var constructor

    /// Guards AWS env-var mutation: tests within a crate run on multiple
    /// threads, and env vars are process-global, so this serializes access
    /// across the tests below.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_aws_env() {
        // SAFETY: serialized by ENV_MUTEX against the other env-var tests in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::remove_var(AWS_ACCESS_KEY_ID_ENV);
            std::env::remove_var(AWS_SECRET_ACCESS_KEY_ENV);
            std::env::remove_var(AWS_SESSION_TOKEN_ENV);
            std::env::remove_var(AWS_REGION_ENV);
            std::env::remove_var(AWS_DEFAULT_REGION_ENV);
        }
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
        let client = BedrockChatClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(client.inner.access_key_id, "AKIDEXAMPLE");
        assert_eq!(client.inner.secret_access_key, "secret-123");
        assert_eq!(client.region(), "eu-central-1");
        assert!(client.inner.session_token.is_none());
        clear_aws_env();
    }

    #[test]
    fn from_env_falls_back_to_aws_default_region() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_aws_env.
        unsafe {
            std::env::set_var(AWS_ACCESS_KEY_ID_ENV, "AKIDEXAMPLE");
            std::env::set_var(AWS_SECRET_ACCESS_KEY_ENV, "secret-123");
            std::env::set_var(AWS_DEFAULT_REGION_ENV, "ap-southeast-2");
        }
        let client = BedrockChatClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(client.region(), "ap-southeast-2");
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
        let client = BedrockChatClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(client.region(), DEFAULT_REGION);
        clear_aws_env();
    }

    #[test]
    fn from_env_reads_optional_session_token() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_aws_env.
        unsafe {
            std::env::set_var(AWS_ACCESS_KEY_ID_ENV, "AKIDEXAMPLE");
            std::env::set_var(AWS_SECRET_ACCESS_KEY_ENV, "secret-123");
            std::env::set_var(AWS_SESSION_TOKEN_ENV, "session-token-xyz");
        }
        let client = BedrockChatClient::from_env("anthropic.claude-3-haiku").unwrap();
        assert_eq!(
            client.inner.session_token.as_deref(),
            Some("session-token-xyz")
        );
        clear_aws_env();
    }

    #[test]
    fn from_env_errors_when_access_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        let result = BedrockChatClient::from_env("anthropic.claude-3-haiku");
        assert!(matches!(result, Err(Error::Configuration(_))));
        clear_aws_env();
    }

    #[test]
    fn from_env_errors_when_secret_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_aws_env();
        // SAFETY: serialized by ENV_MUTEX; see clear_aws_env.
        unsafe {
            std::env::set_var(AWS_ACCESS_KEY_ID_ENV, "AKIDEXAMPLE");
        }
        let result = BedrockChatClient::from_env("anthropic.claude-3-haiku");
        assert!(matches!(result, Err(Error::Configuration(_))));
        clear_aws_env();
    }

    // endregion
}
