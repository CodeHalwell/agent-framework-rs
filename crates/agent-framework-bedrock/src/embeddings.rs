//! AWS Bedrock embeddings client.
//!
//! Rust equivalent of upstream's `RawBedrockEmbeddingClient`
//! (`agent_framework_bedrock/_embedding_client.py`): the Bedrock Runtime
//! `InvokeModel` operation (`POST /model/{modelId}/invoke`) driving an Amazon
//! Titan Text Embeddings model.
//!
//! # One request per value
//!
//! Unlike the OpenAI-shaped embedding APIs, which take a whole batch in one
//! call, Titan's `InvokeModel` body carries a single `inputText`. Upstream
//! issues one call per value and gathers them concurrently
//! (`asyncio.gather`); this port does the same with
//! [`futures::future::try_join_all`], so a batch of *n* values costs *n*
//! signed requests. Results are reassembled in input order — `try_join_all`
//! preserves it — and `inputTextTokenCount` is summed across the batch into
//! [`UsageDetails::input_token_count`], reported only when non-zero (matching
//! upstream's `if total_input_tokens > 0`).
//!
//! # Divergences from upstream
//!
//! - Upstream reaches Bedrock through boto3, inheriting its whole credential
//!   chain (config files, instance metadata, SSO). This crate speaks HTTP
//!   directly and signs with [`crate::sigv4`], so credentials are the static
//!   ones the chat client already takes: explicit, or the standard
//!   `AWS_*` environment variables via [`BedrockEmbeddingClient::from_env`].
//! - Upstream's `normalize` option is forwarded here through
//!   [`EmbeddingGenerationOptions::additional_properties`] rather than a typed
//!   field, since it is Titan-specific and the core options struct is shared
//!   across providers.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::{
    Embedding, EmbeddingGenerationOptions, GeneratedEmbeddings, UsageDetails,
};
use serde_json::{json, Map, Value};

use crate::sigv4;
use crate::{
    bedrock_host, classify_bedrock_error, parse_retry_after, uri_encode_model_id,
    AWS_ACCESS_KEY_ID_ENV, AWS_DEFAULT_REGION_ENV, AWS_REGION_ENV, AWS_SECRET_ACCESS_KEY_ENV,
    AWS_SESSION_TOKEN_ENV, DEFAULT_REGION, SIGV4_SERVICE,
};

/// The `InvokeModel` path for a given model id.
fn invoke_path(model: &str) -> String {
    format!("/model/{}/invoke", uri_encode_model_id(model))
}

/// An AWS Bedrock embeddings client (Titan Text Embeddings via `InvokeModel`).
///
/// ```no_run
/// # use agent_framework_bedrock::BedrockEmbeddingClient;
/// # use agent_framework_core::client::EmbeddingClient;
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = BedrockEmbeddingClient::from_env("amazon.titan-embed-text-v2:0")?;
/// let batch = client.get_embeddings(vec!["hello".into()], None).await?;
/// println!("{} dims", batch[0].dimensions());
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct BedrockEmbeddingClient {
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
    /// Base URL including scheme, e.g.
    /// `https://bedrock-runtime.us-east-1.amazonaws.com`. Derived from
    /// `region` unless overridden by
    /// [`BedrockEmbeddingClient::with_endpoint`].
    endpoint: String,
}

impl std::fmt::Debug for BedrockEmbeddingClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockEmbeddingClient")
            .field("region", &self.inner.region)
            .field("model", &self.inner.model)
            .field("endpoint", &self.inner.endpoint)
            .field("has_session_token", &self.inner.session_token.is_some())
            .finish_non_exhaustive()
    }
}

impl BedrockEmbeddingClient {
    /// Create a client for the given static AWS credentials, region, and
    /// default embedding model id.
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
                endpoint: format!("https://{}", bedrock_host(&region)),
                region,
                model: model.into(),
                access_key_id: access_key_id.into(),
                secret_access_key: secret_access_key.into(),
                session_token: None,
            }),
        }
    }

    /// Build a client from the standard AWS environment variables, exactly as
    /// [`crate::BedrockChatClient::from_env`] does: `AWS_ACCESS_KEY_ID` and
    /// `AWS_SECRET_ACCESS_KEY` are required, `AWS_SESSION_TOKEN` is read when
    /// present, and the region comes from `AWS_REGION`, then
    /// `AWS_DEFAULT_REGION`, then `us-east-1`.
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

    /// Override the region (and derive a new endpoint from it).
    ///
    /// This resets any endpoint set by
    /// [`BedrockEmbeddingClient::with_endpoint`], so call them in that order
    /// if you need both.
    pub fn with_region(mut self, region: impl Into<String>) -> Self {
        let region = region.into();
        let inner = Arc::make_mut(&mut self.inner);
        inner.endpoint = format!("https://{}", bedrock_host(&region));
        inner.region = region;
        self
    }

    /// Set an AWS session token (for temporary/STS credentials).
    pub fn with_session_token(mut self, session_token: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).session_token = Some(session_token.into());
        self
    }

    /// Override the base URL, scheme included — for a VPC interface endpoint
    /// (AWS PrivateLink), a region this crate's host derivation does not
    /// cover, or a local test server.
    ///
    /// The `Host` header signed into the SigV4 canonical request is taken
    /// from this URL, so the signature stays consistent with where the
    /// request is actually sent. The region used to derive the *signing
    /// scope* is unchanged — override it separately with
    /// [`BedrockEmbeddingClient::with_region`] first if the endpoint belongs
    /// to another region.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).endpoint = endpoint.into().trim_end_matches('/').to_string();
        self
    }

    /// The default embedding model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The host to sign and send to: the endpoint's authority, with the
    /// scheme and any path stripped.
    fn host(&self) -> &str {
        Self::split_endpoint(&self.inner.endpoint).0
    }

    /// The endpoint's path prefix, if it carries one (`""` otherwise), with
    /// no trailing slash.
    fn path_prefix(&self) -> &str {
        Self::split_endpoint(&self.inner.endpoint).1
    }

    /// Split an endpoint into `(authority, path prefix)`, dropping the scheme.
    /// The prefix keeps its leading `/` and loses any trailing one, so it
    /// concatenates cleanly with a path that starts with `/`.
    fn split_endpoint(endpoint: &str) -> (&str, &str) {
        let after_scheme = endpoint
            .split_once("://")
            .map_or(endpoint, |(_, rest)| rest);
        match after_scheme.find('/') {
            Some(slash) => (
                &after_scheme[..slash],
                after_scheme[slash..].trim_end_matches('/'),
            ),
            None => (after_scheme, ""),
        }
    }

    /// Sign and POST one `InvokeModel` request, returning the parsed body.
    async fn invoke(&self, model: &str, payload: Vec<u8>) -> Result<Value> {
        // The path actually requested and the canonical URI signed must be
        // byte-identical, so an endpoint path prefix has to be folded in
        // *before* signing — otherwise a proxy or VPC endpoint that preserves
        // the prefix forwards a request whose signature covers a different
        // path, and AWS rejects it.
        let path = format!("{}{}", self.path_prefix(), invoke_path(model));
        let host = self.host();
        let scheme = self
            .inner
            .endpoint
            .split_once("://")
            .map_or("https", |(scheme, _)| scheme);
        let url = format!("{scheme}://{host}{path}");

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
            host,
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
            .header(reqwest::header::ACCEPT, "application/json")
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
            let error_type = resp
                .headers()
                .get("x-amzn-errortype")
                .and_then(|v| v.to_str().ok())
                .map(str::to_string);
            let retry_after = parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            let message = match &error_type {
                Some(et) => format!("Bedrock InvokeModel API error {status} ({et}): {text}"),
                None => format!("Bedrock InvokeModel API error {status}: {text}"),
            };
            return Err(classify_bedrock_error(
                status.as_u16(),
                message,
                retry_after,
            ));
        }

        resp.json::<Value>()
            .await
            .map_err(|e| Error::service(format!("invalid Bedrock embedding response: {e}")))
    }

    /// Build the Titan request body for one value.
    fn body_for(value: &str, options: Option<&EmbeddingGenerationOptions>) -> Vec<u8> {
        let mut body = Map::new();
        body.insert("inputText".into(), json!(value));
        if let Some(dimensions) = options.and_then(|o| o.dimensions) {
            body.insert("dimensions".into(), json!(dimensions));
        }
        // Titan-specific; forwarded verbatim when the caller supplies it.
        if let Some(normalize) = options.and_then(|o| o.additional_properties.get("normalize")) {
            body.insert("normalize".into(), normalize.clone());
        }
        Value::Object(body).to_string().into_bytes()
    }
}

#[async_trait::async_trait]
impl EmbeddingClient for BedrockEmbeddingClient {
    async fn get_embeddings(
        &self,
        values: Vec<String>,
        options: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        // Upstream returns an empty batch without calling the service.
        if values.is_empty() {
            return Ok(GeneratedEmbeddings::new(Vec::new()));
        }

        let model = options
            .as_ref()
            .and_then(|o| o.model.clone())
            .unwrap_or_else(|| self.inner.model.clone());

        let results = futures::future::try_join_all(values.iter().map(|value| {
            let payload = Self::body_for(value, options.as_ref());
            let model = model.clone();
            async move { self.invoke(&model, payload).await }
        }))
        .await?;

        let mut embeddings = Vec::with_capacity(results.len());
        let mut total_input_tokens = 0u64;
        for parsed in results {
            let vector = parsed
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Error::service("Bedrock embedding response has no `embedding` array")
                })?
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        Error::service("Bedrock embedding vector holds a non-numeric value")
                    })
                })
                .collect::<Result<Vec<f32>>>()?;
            total_input_tokens += parsed
                .get("inputTextTokenCount")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            embeddings.push(Embedding {
                vector,
                model: Some(model.clone()),
            });
        }

        let mut batch = GeneratedEmbeddings::new(embeddings);
        if total_input_tokens > 0 {
            batch.usage = Some(UsageDetails {
                input_token_count: Some(total_input_tokens),
                ..Default::default()
            });
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
    fn invoke_path_encodes_the_model_id_like_converse_does() {
        // Titan ids carry a `:` version suffix, which SigV4's UriEncode
        // escapes; the same encoded string must be used for the request path
        // and the canonical URI, so this shares `uri_encode_model_id`.
        assert_eq!(
            invoke_path("amazon.titan-embed-text-v2:0"),
            "/model/amazon.titan-embed-text-v2%3A0/invoke"
        );
    }

    #[test]
    fn host_is_the_endpoint_authority_without_scheme_or_path() {
        let client = BedrockEmbeddingClient::new("ak", "sk", "us-east-1", "m");
        assert_eq!(client.host(), "bedrock-runtime.us-east-1.amazonaws.com");
        assert_eq!(client.path_prefix(), "");

        let overridden = client.clone().with_endpoint("http://127.0.0.1:8080");
        assert_eq!(overridden.host(), "127.0.0.1:8080");
        assert_eq!(overridden.path_prefix(), "");

        let with_path = client
            .clone()
            .with_endpoint("https://vpce-abc.bedrock-runtime.aws/prefix");
        assert_eq!(with_path.host(), "vpce-abc.bedrock-runtime.aws");
        assert_eq!(with_path.path_prefix(), "/prefix");

        // A trailing slash on the endpoint must not double up against the
        // leading slash of the invoke path.
        let trailing = client.with_endpoint("https://vpce-abc.bedrock-runtime.aws/prefix/");
        assert_eq!(trailing.path_prefix(), "/prefix");
    }

    #[test]
    fn with_region_redefines_the_endpoint() {
        let client =
            BedrockEmbeddingClient::new("ak", "sk", "us-east-1", "m").with_region("eu-west-1");
        assert_eq!(client.host(), "bedrock-runtime.eu-west-1.amazonaws.com");
    }

    #[test]
    fn body_carries_input_text_and_optional_titan_knobs() {
        let bare: Value = serde_json::from_slice(&BedrockEmbeddingClient::body_for("hi", None))
            .expect("valid json");
        assert_eq!(bare, json!({ "inputText": "hi" }));

        let mut options = EmbeddingGenerationOptions::new();
        options.dimensions = Some(256);
        options
            .additional_properties
            .insert("normalize".into(), json!(false));
        let full: Value =
            serde_json::from_slice(&BedrockEmbeddingClient::body_for("hi", Some(&options)))
                .expect("valid json");
        assert_eq!(
            full,
            json!({ "inputText": "hi", "dimensions": 256, "normalize": false })
        );
    }

    #[tokio::test]
    async fn empty_input_short_circuits_without_a_request() {
        // No server is listening on the overridden endpoint, so if this
        // issued a request at all it would fail rather than return empty.
        let client = BedrockEmbeddingClient::new("ak", "sk", "us-east-1", "m")
            .with_endpoint("http://127.0.0.1:1");
        let batch = client.get_embeddings(Vec::new(), None).await.expect("ok");
        assert!(batch.embeddings.is_empty());
        assert!(batch.usage.is_none());
    }
}
