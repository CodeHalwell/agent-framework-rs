//! Azure AI Foundry embeddings client.
//!
//! Rust equivalent of upstream's `RawFoundryEmbeddingClient`
//! (`agent_framework_foundry/_embedding_client.py`), which reaches text
//! embeddings over **two** surfaces:
//!
//! - the Foundry **Models** inference endpoint (`POST
//!   {models_endpoint}/embeddings`), which upstream reaches through
//!   `azure.ai.inference`'s `EmbeddingsClient`
//!   ([`FoundryEmbeddingClient::new`] / [`FoundryEmbeddingClient::with_credential`]);
//! - a Foundry **project**'s OpenAI model deployments (upstream #8454), by
//!   deriving the resource-scoped `POST {resource}/openai/v1/embeddings` route
//!   from the project endpoint ([`FoundryEmbeddingClient::with_project_endpoint`],
//!   [`openai_model_base_url`]).
//!
//! The second matters because the project endpoint is the one a Foundry user
//! already has — [`crate::FoundryChatClient`] takes it. Without it, a project
//! with an embedding deployment still needed a separately-provisioned Models
//! inference endpoint before it could be used for embeddings.
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
/// The Foundry **project** endpoint (e.g.
/// `https://<resource>.services.ai.azure.com/api/projects/<project>`) — the
/// same endpoint [`crate::FoundryChatClient`] takes.
pub const FOUNDRY_PROJECT_ENDPOINT_ENV: &str = "FOUNDRY_PROJECT_ENDPOINT";
/// Alias for [`FOUNDRY_PROJECT_ENDPOINT_ENV`], read second.
///
/// [`crate::FoundryChatClient::from_env`] takes this as its *primary* name,
/// so an environment configured for the chat client is picked up here too
/// rather than reporting that no endpoint is set while one plainly is.
pub const FOUNDRY_ENDPOINT_ENV: &str = "FOUNDRY_ENDPOINT";

/// Derive the resource-scoped OpenAI **v1** model route from a Foundry
/// project endpoint, as upstream's `_get_openai_model_base_url` does.
///
/// `https://<resource>.services.ai.azure.com/api/projects/<project>`
/// becomes `https://<resource>.openai.azure.com/openai/v1`. The project path
/// is dropped: the model route is scoped to the *resource*, not to the
/// project underneath it.
///
/// A host that does not carry `.services.ai.` is passed through unchanged
/// apart from the path, matching upstream — a custom or sovereign-cloud
/// domain is a caller's business, and refusing it here would reject a
/// deployment that works.
pub fn openai_model_base_url(project_endpoint: &str) -> Result<String> {
    let trimmed = project_endpoint.trim();
    let invalid = || {
        Error::Configuration(format!(
            "invalid Foundry project endpoint: {project_endpoint:?}"
        ))
    };
    let (scheme, rest) = trimmed.split_once("://").ok_or_else(invalid)?;
    if scheme.is_empty() {
        return Err(invalid());
    }
    let host = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if host.is_empty() {
        return Err(invalid());
    }
    let host = host.replacen(".services.ai.", ".openai.", 1);
    Ok(format!("{scheme}://{host}/openai/v1"))
}

/// Which Foundry surface a [`FoundryEmbeddingClient`] talks to.
///
/// The two differ in more than a hostname: the Models endpoint is
/// query-versioned (`?api-version=`) while the project's OpenAI route is
/// path-versioned (`/openai/v1`) and rejects the query parameter — the same
/// split [`crate::FoundryChatClient`] handles with `without_api_version`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Route {
    /// `POST {models_endpoint}/embeddings?api-version=…`
    Models,
    /// `POST {resource}/openai/v1/embeddings`
    ProjectOpenAI,
}

/// The `api-version` sent when none is set, mirroring the default in
/// `azure-ai-inference` 1.0.0b9 — the version upstream's `foundry` package
/// pins. Override with [`FoundryEmbeddingClient::with_api_version`].
pub const DEFAULT_API_VERSION: &str = "2024-05-01-preview";

/// Default Entra ID scope requested for the bearer token.
///
/// This is the **data plane** audience for Azure AI Services, the same one
/// `agent-framework-anthropic`'s Foundry client and
/// [`agent_framework_azure::AzureOpenAIClient`] use — *not*
/// [`crate::FOUNDRY_SCOPE`] (`https://ai.azure.com/.default`), which is the
/// Foundry **project** audience that [`crate::FoundryChatClient`] needs for
/// the Responses API. The two are different surfaces with different
/// audiences: a token minted for the project scope is rejected by the Models
/// inference endpoint. Override with [`FoundryEmbeddingClient::with_scope`]
/// if a particular deployment wants something else.
pub const DEFAULT_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Header that tells Azure AI Inference what to do with body fields outside
/// its own schema. It defaults to erroring; `pass-through` forwards them to
/// the model instead. The `azure-ai-inference` SDK sets it whenever
/// `model_extras` is supplied, which is what upstream maps
/// `extra_parameters` onto — so expanding those extras into the body without
/// this header would turn a call that works upstream into a 4xx here.
const EXTRA_PARAMETERS_HEADER: &str = "extra-parameters";
const EXTRA_PARAMETERS_PASS_THROUGH: &str = "pass-through";

/// Keys in [`EmbeddingGenerationOptions::additional_properties`] this endpoint
/// understands directly, matching the kwargs upstream builds. `dimensions` has
/// a typed slot on the options struct and is handled separately; anything
/// outside this list belongs under
/// [`FoundryEmbeddingClient::EXTRA_PARAMETERS_KEY`].
const SUPPORTED_PROPERTIES: &[&str] = &["encoding_format", "input_type"];

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
    scope: String,
    auth: Auth,
    route: Route,
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
    /// Entra ID bearer token).
    ///
    /// The token is requested for [`DEFAULT_SCOPE`] through
    /// [`TokenCredential::get_token_for_scope`], so a credential that mints
    /// per-audience tokens gets the right one without the caller having to
    /// know which. A credential wrapping a single fixed token ignores the
    /// scope, as that trait method's default does — such a token must already
    /// carry the Models-inference audience.
    pub fn with_credential(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Self {
        Self::build(endpoint, model, Auth::Credential(credential))
    }

    fn build(endpoint: impl Into<String>, model: impl Into<String>, auth: Auth) -> Self {
        Self::build_on(endpoint, model, auth, Route::Models)
    }

    fn build_on(
        endpoint: impl Into<String>,
        model: impl Into<String>,
        auth: Auth,
        route: Route,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                endpoint: endpoint.into().trim_end_matches('/').to_string(),
                model: model.into(),
                api_version: DEFAULT_API_VERSION.to_string(),
                scope: DEFAULT_SCOPE.to_string(),
                auth,
                route,
            }),
        }
    }

    /// Create a client against a Foundry **project** endpoint, reaching the
    /// resource's OpenAI v1 model route (upstream #8454).
    ///
    /// This is the endpoint a Foundry user already has — the one
    /// [`crate::FoundryChatClient`] takes. Before this, embeddings were
    /// reachable only through a separately-provisioned Foundry *Models*
    /// inference endpoint, so a project that had an embedding deployment
    /// still could not be used for embeddings from here.
    ///
    /// Entra ID only: the project route has no API-key form, which is why
    /// there is no key-taking counterpart. The token is requested for
    /// [`DEFAULT_SCOPE`] — the Azure OpenAI data-plane audience the derived
    /// `*.openai.azure.com` host expects, *not* [`crate::FOUNDRY_SCOPE`].
    pub fn with_project_endpoint(
        project_endpoint: &str,
        model: impl Into<String>,
        credential: Arc<dyn TokenCredential>,
    ) -> Result<Self> {
        Ok(Self::build_on(
            openai_model_base_url(project_endpoint)?,
            model,
            Auth::Credential(credential),
            Route::ProjectOpenAI,
        ))
    }

    /// Build a client from `FOUNDRY_MODELS_ENDPOINT`,
    /// `FOUNDRY_MODELS_API_KEY` and `FOUNDRY_EMBEDDING_MODEL`. `model`
    /// overrides the environment's model when supplied.
    ///
    /// Mirrors upstream's `required_fields=["models_endpoint",
    /// "embedding_model"]`: only those two are required, and they must be
    /// resolvable from arguments or the environment or this errors rather
    /// than deferring the failure to the first request.
    ///
    /// The API key is **not** required. Without it this falls back to
    /// `DefaultAzureCredential` scoped to [`DEFAULT_SCOPE`], matching both
    /// upstream (whose key is optional beside a credential) and
    /// [`crate::FoundryChatClient::from_env`], so a managed-identity or
    /// `az login` environment works with no key configured.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let models_endpoint = std::env::var(FOUNDRY_MODELS_ENDPOINT_ENV)
            .ok()
            .filter(|e| !e.trim().is_empty());
        let project_endpoint = std::env::var(FOUNDRY_PROJECT_ENDPOINT_ENV)
            .ok()
            .or_else(|| std::env::var(FOUNDRY_ENDPOINT_ENV).ok())
            .filter(|e| !e.trim().is_empty());
        if models_endpoint.is_none() && project_endpoint.is_none() {
            return Err(Error::Configuration(format!(
                "a Foundry endpoint is required: set {FOUNDRY_MODELS_ENDPOINT_ENV} for a Models \
                 inference endpoint, or {FOUNDRY_PROJECT_ENDPOINT_ENV} for a project"
            )));
        }
        let model = model
            .or_else(|| std::env::var(FOUNDRY_EMBEDDING_MODEL_ENV).ok())
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "an embedding model is required: pass one or set {FOUNDRY_EMBEDDING_MODEL_ENV}"
                ))
            })?;

        // The Models endpoint wins when both are set, which is the
        // conservative order: it is the surface this client has always spoken,
        // so an environment that already worked keeps working unchanged, and
        // it is the only one of the two with an API-key path.
        let Some(endpoint) = models_endpoint else {
            let project_endpoint = project_endpoint.expect("checked above");
            let credential: Arc<dyn TokenCredential> = Arc::new(
                agent_framework_azure::DefaultAzureCredential::new(DEFAULT_SCOPE),
            );
            return Self::with_project_endpoint(&project_endpoint, model, credential);
        };
        match std::env::var(FOUNDRY_MODELS_API_KEY_ENV) {
            Ok(api_key) if !api_key.trim().is_empty() => Ok(Self::new(endpoint, model, api_key)),
            _ => {
                let credential: Arc<dyn TokenCredential> = Arc::new(
                    agent_framework_azure::DefaultAzureCredential::new(DEFAULT_SCOPE),
                );
                Ok(Self::with_credential(endpoint, model, credential))
            }
        }
    }

    /// Override the `api-version` query parameter (see
    /// [`DEFAULT_API_VERSION`]).
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).api_version = api_version.into();
        self
    }

    /// Override the Entra ID scope requested for the bearer token (see
    /// [`DEFAULT_SCOPE`]). No effect on the API-key path.
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        arc_inner(&mut self.inner).scope = scope.into();
        self
    }

    /// The default embedding model.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The full request URL.
    ///
    /// The project route is path-versioned, so it carries no `api-version`
    /// query parameter; appending one there is rejected by the service.
    fn url(&self) -> String {
        match self.inner.route {
            Route::Models => format!(
                "{}/embeddings?api-version={}",
                self.inner.endpoint, self.inner.api_version
            ),
            Route::ProjectOpenAI => format!("{}/embeddings", self.inner.endpoint),
        }
    }

    /// The option key upstream maps onto the Azure AI Inference SDK's
    /// `model_extras`: a map of model-specific parameters that belong in the
    /// request body under **their own names**, not nested under this key.
    const EXTRA_PARAMETERS_KEY: &'static str = "extra_parameters";

    /// Build the request body for a batch, and report whether any
    /// model-specific extras were expanded into it.
    ///
    /// Upstream forwards `encoding_format` and `input_type` explicitly and
    /// passes `extra_parameters` as `model_extras`, which the SDK *expands*
    /// into the body. The core options struct has no typed slot for any of
    /// them, so all three arrive through `additional_properties`: the first
    /// two are forwarded verbatim, while `extra_parameters` has its entries
    /// merged in — sending it as a literal `extra_parameters` field would
    /// mean the model never sees those parameters under the names it expects.
    fn body_for(
        &self,
        values: &[String],
        options: Option<&EmbeddingGenerationOptions>,
    ) -> (Value, bool) {
        let mut body = Map::new();
        body.insert("input".into(), json!(values));
        body.insert(
            "model".into(),
            json!(Self::effective_model(&self.inner.model, options)),
        );
        if let Some(dimensions) = options.and_then(|o| o.dimensions) {
            body.insert("dimensions".into(), json!(dimensions));
        }

        let mut expanded_extras = false;
        if let Some(properties) = options.map(|o| &o.additional_properties) {
            // An allowlist, not a passthrough: upstream builds its kwargs from
            // exactly these keys, and the sibling OpenAI/Azure clients forward
            // a fixed list the same way. Anything else would reach the wire as
            // a field outside the inference schema, which the service rejects
            // unless pass-through is set — so a stray OpenAI-ism like `user`
            // would turn the whole request into a 4xx. Dropping it matches
            // upstream, which never forwards it either.
            for key in SUPPORTED_PROPERTIES {
                if let Some(value) = properties.get(*key) {
                    body.insert((*key).into(), value.clone());
                }
            }
            // `extra_parameters` is the deliberate escape hatch for anything
            // outside that list: its entries are expanded under their own
            // names and carry the pass-through header. A non-object value has
            // nothing to expand, and forwarding it as a literal field would be
            // the very 4xx this allowlist exists to avoid, so it is dropped
            // like any other unsupported entry.
            if let Some(Value::Object(extras)) = properties.get(Self::EXTRA_PARAMETERS_KEY) {
                for (extra_key, extra_value) in extras {
                    body.insert(extra_key.clone(), extra_value.clone());
                    expanded_extras = true;
                }
            }
        }
        (Value::Object(body), expanded_extras)
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
            scope: inner.scope.clone(),
            auth: match &inner.auth {
                Auth::ApiKey(key) => Auth::ApiKey(key.clone()),
                Auth::Credential(cred) => Auth::Credential(cred.clone()),
            },
            route: inner.route,
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

        let (body, expanded_extras) = self.body_for(&values, options.as_ref());
        let mut request = self.inner.http.post(self.url()).json(&body);
        if expanded_extras {
            // Azure AI Inference rejects body fields it does not recognise
            // unless this header opts into passing them to the model. The SDK
            // sets it whenever `model_extras` is supplied, so expanding the
            // extras without it would turn a working upstream call into a
            // 4xx here.
            request = request.header(EXTRA_PARAMETERS_HEADER, EXTRA_PARAMETERS_PASS_THROUGH);
        }
        let request = match &self.inner.auth {
            Auth::ApiKey(key) => request.header("api-key", key),
            Auth::Credential(credential) => {
                let token = credential.get_token_for_scope(&self.inner.scope).await?;
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
        let (bare, extras) = client.body_for(&["a".into(), "b".into()], None);
        assert_eq!(
            bare,
            json!({ "input": ["a", "b"], "model": "default-model" })
        );
        assert!(!extras);

        let mut options = EmbeddingGenerationOptions::new();
        options.model = Some("per-request".into());
        options.dimensions = Some(512);
        options
            .additional_properties
            .insert("encoding_format".into(), json!("float"));
        options
            .additional_properties
            .insert("input_type".into(), json!("query"));
        let (full, extras) = client.body_for(&["a".into()], Some(&options));
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
        assert!(!extras, "no extra_parameters means no pass-through header");
    }

    /// Upstream maps `extra_parameters` onto the SDK's `model_extras`, which
    /// *expands* the map into the request body. Sending it as a literal
    /// `extra_parameters` field instead would mean the model never sees those
    /// parameters under the names it expects — the call would succeed and
    /// quietly ignore them.
    #[test]
    fn extra_parameters_are_expanded_rather_than_nested() {
        let client = FoundryEmbeddingClient::new("https://e/models", "m", "k");
        let mut options = EmbeddingGenerationOptions::new();
        options.additional_properties.insert(
            "extra_parameters".into(),
            json!({ "truncate": "END", "custom_knob": 7 }),
        );
        options
            .additional_properties
            .insert("input_type".into(), json!("document"));

        let (body, extras) = client.body_for(&["a".into()], Some(&options));
        assert_eq!(
            body,
            json!({
                "input": ["a"],
                "model": "m",
                // Expanded to the top level, under their own names...
                "truncate": "END",
                "custom_knob": 7,
                // ...while the explicitly-forwarded option is untouched.
                "input_type": "document",
            })
        );
        assert!(body.get("extra_parameters").is_none());
        assert!(extras, "expanding extras must request pass-through");
    }

    #[test]
    fn a_non_object_extra_parameters_is_dropped_rather_than_sent() {
        let client = FoundryEmbeddingClient::new("https://e/models", "m", "k");
        let mut options = EmbeddingGenerationOptions::new();
        options
            .additional_properties
            .insert("extra_parameters".into(), json!("not-a-map"));

        let (body, extras) = client.body_for(&["a".into()], Some(&options));
        // Nothing to expand. Sending it as a literal field would be a field
        // outside the inference schema without pass-through — the 4xx the
        // allowlist exists to prevent — so it is dropped.
        assert!(body.get("extra_parameters").is_none());
        assert!(!extras);
    }

    /// Upstream builds its request from a fixed set of keys and forwards
    /// nothing else, and the sibling OpenAI/Azure clients use the same
    /// allowlist shape. Copying arbitrary entries through would put fields
    /// outside the inference schema on the wire, which the service rejects
    /// unless pass-through is set — so a caller reusing one options struct
    /// across providers would get a 4xx from an option that is merely
    /// irrelevant here.
    #[test]
    fn unsupported_additional_properties_are_not_forwarded() {
        let client = FoundryEmbeddingClient::new("https://e/models", "m", "k");
        let mut options = EmbeddingGenerationOptions::new();
        // Supported here...
        options
            .additional_properties
            .insert("encoding_format".into(), json!("float"));
        options
            .additional_properties
            .insert("input_type".into(), json!("query"));
        // ...and an OpenAI-ism this endpoint does not take.
        options
            .additional_properties
            .insert("user".into(), json!("alice"));

        let (body, extras) = client.body_for(&["a".into()], Some(&options));
        assert_eq!(
            body,
            json!({
                "input": ["a"],
                "model": "m",
                "encoding_format": "float",
                "input_type": "query",
            })
        );
        assert!(body.get("user").is_none(), "must not reach the wire");
        assert!(!extras, "dropping a key never claims pass-through");
    }

    #[test]
    fn from_env_requires_only_the_endpoint_and_model() {
        temp_env_absent(|| {
            // Each missing required piece names itself rather than failing at
            // request time.
            let err = FoundryEmbeddingClient::from_env(None).expect_err("no endpoint");
            assert!(
                err.to_string().contains(FOUNDRY_MODELS_ENDPOINT_ENV)
                    && err.to_string().contains(FOUNDRY_PROJECT_ENDPOINT_ENV),
                "the error should name both endpoints a caller could set: {err}"
            );

            std::env::set_var(FOUNDRY_MODELS_ENDPOINT_ENV, "https://e/models");
            let err = FoundryEmbeddingClient::from_env(None).expect_err("no model");
            assert!(
                err.to_string().contains(FOUNDRY_EMBEDDING_MODEL_ENV),
                "{err}"
            );

            // The API key is *not* required: with endpoint and model present,
            // a keyless managed-identity or `az login` environment resolves to
            // a credential rather than erroring, matching FoundryChatClient.
            let client = FoundryEmbeddingClient::from_env(Some("m".into()))
                .expect("keyless env must fall back to a credential");
            assert!(
                matches!(client.inner.auth, Auth::Credential(_)),
                "expected a credential, got the api-key path"
            );
            assert_eq!(client.inner.scope, DEFAULT_SCOPE);

            // A key present still takes the api-key path.
            std::env::set_var(FOUNDRY_MODELS_API_KEY_ENV, "k");
            let keyed = FoundryEmbeddingClient::from_env(Some("m".into())).expect("keyed");
            assert!(matches!(keyed.inner.auth, Auth::ApiKey(_)));

            // An empty key is treated as absent rather than sent as "".
            std::env::set_var(FOUNDRY_MODELS_API_KEY_ENV, "   ");
            let blank = FoundryEmbeddingClient::from_env(Some("m".into())).expect("blank key");
            assert!(matches!(blank.inner.auth, Auth::Credential(_)));
        });
    }

    #[test]
    fn the_default_scope_is_the_models_data_plane_not_the_project_audience() {
        // Regression guard for a real mix-up: FOUNDRY_SCOPE
        // (https://ai.azure.com/.default) is the project audience the
        // Responses API needs; the Models inference endpoint rejects it.
        assert_eq!(
            DEFAULT_SCOPE,
            "https://cognitiveservices.azure.com/.default"
        );
        assert_ne!(DEFAULT_SCOPE, crate::FOUNDRY_SCOPE);

        let client = FoundryEmbeddingClient::new("https://e/models", "m", "k")
            .with_scope("https://custom/.default");
        assert_eq!(client.inner.scope, "https://custom/.default");
    }

    /// Run `f` with the three Foundry embedding variables unset, restoring
    /// whatever was there before. Tests in one binary share a process
    /// environment, so this is deliberately narrow.
    /// The derivation upstream #8454 added: the model route is scoped to the
    /// *resource*, so the project path is dropped and the host moves from
    /// `.services.ai.` to `.openai.`.
    #[test]
    fn a_project_endpoint_becomes_the_resource_openai_v1_route() {
        assert_eq!(
            openai_model_base_url("https://myres.services.ai.azure.com/api/projects/proj").unwrap(),
            "https://myres.openai.azure.com/openai/v1"
        );
        // Trailing slashes, queries and fragments are all part of the path
        // being replaced, not of the host.
        assert_eq!(
            openai_model_base_url("https://myres.services.ai.azure.com/api/projects/proj/?x=1#f")
                .unwrap(),
            "https://myres.openai.azure.com/openai/v1"
        );
        // A host without the Foundry marker keeps its name, as upstream's
        // single-shot replace does — a custom domain is not an error.
        assert_eq!(
            openai_model_base_url("https://custom.example.com/api/projects/p").unwrap(),
            "https://custom.example.com/openai/v1"
        );
    }

    #[test]
    fn a_malformed_project_endpoint_is_refused() {
        for bad in [
            "",
            "   ",
            "myres.services.ai.azure.com",
            "https://",
            "://host",
        ] {
            let err = openai_model_base_url(bad).expect_err("{bad:?} should be refused");
            assert!(err.to_string().contains("invalid Foundry project endpoint"));
        }
    }

    /// The project route is path-versioned. Carrying the Models endpoint's
    /// `?api-version=` onto it is rejected by the service, so the two URLs
    /// differ in more than their host.
    #[test]
    fn the_project_route_sends_no_api_version() {
        let client = FoundryEmbeddingClient::with_project_endpoint(
            "https://myres.services.ai.azure.com/api/projects/proj",
            "text-embedding-3-small",
            Arc::new(agent_framework_azure::StaticTokenCredential::new("t")),
        )
        .unwrap();
        assert_eq!(
            client.url(),
            "https://myres.openai.azure.com/openai/v1/embeddings"
        );
        assert!(!client.url().contains("api-version"));

        // The Models route is unchanged.
        let models = FoundryEmbeddingClient::new("https://e/models", "m", "k");
        assert!(models.url().contains("api-version"));
    }

    /// A project endpoint alone is now enough to build a client; before this
    /// the only way in was a separately-provisioned Models endpoint.
    #[test]
    fn from_env_accepts_a_project_endpoint_and_prefers_a_models_one() {
        temp_env_absent(|| {
            std::env::set_var(FOUNDRY_EMBEDDING_MODEL_ENV, "text-embedding-3-small");
            std::env::set_var(
                FOUNDRY_PROJECT_ENDPOINT_ENV,
                "https://myres.services.ai.azure.com/api/projects/proj",
            );
            let client = FoundryEmbeddingClient::from_env(None).expect("project endpoint suffices");
            assert_eq!(
                client.url(),
                "https://myres.openai.azure.com/openai/v1/embeddings"
            );
            assert!(matches!(client.inner.auth, Auth::Credential(_)));

            // The chat client's own primary name works too, so one Foundry
            // environment configures both clients.
            std::env::remove_var(FOUNDRY_PROJECT_ENDPOINT_ENV);
            std::env::set_var(
                FOUNDRY_ENDPOINT_ENV,
                "https://myres.services.ai.azure.com/api/projects/proj",
            );
            let aliased = FoundryEmbeddingClient::from_env(None).expect("alias is accepted");
            assert_eq!(
                aliased.url(),
                "https://myres.openai.azure.com/openai/v1/embeddings"
            );

            // With both set the Models endpoint wins, so an environment that
            // already worked is untouched.
            std::env::set_var(FOUNDRY_MODELS_ENDPOINT_ENV, "https://e/models");
            let client = FoundryEmbeddingClient::from_env(None).expect("models endpoint wins");
            assert!(client.url().starts_with("https://e/models/embeddings?"));
        });
    }

    /// `std::env::set_var` mutates process-global state, so two tests using
    /// this helper on different threads see each other's variables. Taking one
    /// lock makes them sequential; a poisoned lock is recovered rather than
    /// propagating one test's panic into every other.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn temp_env_absent(f: impl FnOnce()) {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let keys = [
            FOUNDRY_MODELS_ENDPOINT_ENV,
            FOUNDRY_MODELS_API_KEY_ENV,
            FOUNDRY_EMBEDDING_MODEL_ENV,
            FOUNDRY_PROJECT_ENDPOINT_ENV,
            FOUNDRY_ENDPOINT_ENV,
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
