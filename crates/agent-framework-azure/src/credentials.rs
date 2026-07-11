//! A small Microsoft Entra ID (Azure AD) credential chain implementing
//! [`TokenCredential`], with per-credential, per-scope token caching.
//!
//! These are hand-rolled analogues of `azure_identity`'s credentials, built on
//! `reqwest`/`tokio` with no Azure SDK dependency:
//!
//! * [`AzureCliCredential`] — shells out to `az account get-access-token`.
//! * [`ClientSecretCredential`] — OAuth2 client-credentials flow against Entra.
//! * [`EnvironmentCredential`] — the same flow, configured from
//!   `AZURE_TENANT_ID`/`AZURE_CLIENT_ID`/`AZURE_CLIENT_SECRET`.
//! * [`ManagedIdentityCredential`] — the IMDS token endpoint.
//! * [`WorkloadIdentityCredential`] — exchanges a Kubernetes federated
//!   service-account token (AKS workload identity) for an Entra token.
//! * [`ChainedTokenCredential`] — tries each in order; the first to succeed is
//!   remembered and preferred thereafter.
//! * [`DefaultAzureCredential`] — a prebuilt chain of the above four, in
//!   `azure_identity`'s documented `DefaultAzureCredential` order.
//!
//! Every credential is bound to a default scope (audience) supplied at
//! construction and used by [`TokenCredential::get_token`]; a different scope
//! can be requested per call via [`TokenCredential::get_token_for_scope`].
//! Tokens are cached per scope and refreshed [`REFRESH_SKEW`] before expiry.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use agent_framework_core::error::{Error, Result};
use async_trait::async_trait;
use serde_json::Value;

use crate::credential::TokenCredential;

/// How long before a token's stated expiry it is considered stale and
/// proactively refreshed (2 minutes), so an in-flight request never races the
/// exact expiry instant.
pub const REFRESH_SKEW: Duration = Duration::from_secs(120);

/// The IMDS token endpoint used by [`ManagedIdentityCredential`] when no
/// endpoint is configured or discovered from the environment.
pub const DEFAULT_IMDS_ENDPOINT: &str = "http://169.254.169.254/metadata/identity/oauth2/token";

/// The IMDS token API version.
const IMDS_API_VERSION: &str = "2018-02-01";

/// The default Entra ID authority (login endpoint) for
/// [`ClientSecretCredential`].
pub const DEFAULT_AUTHORITY: &str = "https://login.microsoftonline.com";

/// Fallback token lifetime, in seconds, used only when a token response omits
/// any expiry hint (1 hour is the Entra default access-token lifetime).
const DEFAULT_TOKEN_TTL_SECS: u64 = 3600;

/// Conservative lifetime, in seconds, assumed for an Azure CLI token whose
/// JSON carries no machine-readable `expires_on` epoch (older `az` builds emit
/// only a local-time `expiresOn` string, which cannot be converted to an
/// instant without a timezone database). Kept short so a real refresh happens
/// soon rather than trusting a possibly-stale token.
const CLI_FALLBACK_TTL_SECS: u64 = 300;

// ---------------------------------------------------------------------------
// Token cache
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct CachedToken {
    token: String,
    expires_at: SystemTime,
}

/// A thread-safe, per-scope token cache shared by each credential.
#[derive(Default)]
struct TokenCache {
    entries: Mutex<HashMap<String, CachedToken>>,
}

impl TokenCache {
    /// Return the cached token for `scope` if one is present and not within
    /// [`REFRESH_SKEW`] of expiry.
    fn get(&self, scope: &str) -> Option<String> {
        let entries = self.entries.lock().unwrap();
        let entry = entries.get(scope)?;
        if SystemTime::now() + REFRESH_SKEW < entry.expires_at {
            Some(entry.token.clone())
        } else {
            None
        }
    }

    /// Store `token` for `scope`, valid until `expires_at`.
    fn put(&self, scope: &str, token: String, expires_at: SystemTime) {
        self.entries
            .lock()
            .unwrap()
            .insert(scope.to_string(), CachedToken { token, expires_at });
    }
}

// ---------------------------------------------------------------------------
// Shared parsing helpers
// ---------------------------------------------------------------------------

/// Read a JSON value that may be a number or a numeric string as `u64`.
///
/// IMDS returns `expires_in`/`expires_on` as strings; the OAuth2 token endpoint
/// returns `expires_in` as a number — both are accepted.
fn json_u64(v: Option<&Value>) -> Option<u64> {
    match v {
        Some(Value::Number(n)) => n.as_u64().or_else(|| n.as_f64().map(|f| f as u64)),
        Some(Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    }
}

/// Parse an OAuth2 / IMDS token response body: `access_token` plus a relative
/// `expires_in` (seconds from now). Shared by the client-secret and
/// managed-identity credentials.
fn parse_oauth_token(body: &str) -> Result<(String, SystemTime)> {
    let v: Value = serde_json::from_str(body)
        .map_err(|e| Error::other(format!("invalid token response json: {e}")))?;
    let token = v
        .get("access_token")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::other("token response missing 'access_token'"))?
        .to_string();
    let ttl = json_u64(v.get("expires_in")).unwrap_or(DEFAULT_TOKEN_TTL_SECS);
    Ok((token, SystemTime::now() + Duration::from_secs(ttl)))
}

/// Parse the JSON emitted by `az account get-access-token --output json`.
///
/// Extracts `accessToken` and computes expiry from the integer epoch
/// `expires_on` when present (newer `az`); otherwise falls back to a short
/// conservative TTL (see [`CLI_FALLBACK_TTL_SECS`]).
fn parse_cli_output(stdout: &[u8]) -> Result<(String, SystemTime)> {
    let v: Value = serde_json::from_slice(stdout)
        .map_err(|e| Error::other(format!("invalid Azure CLI token json: {e}")))?;
    let token = v
        .get("accessToken")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::other("Azure CLI token json missing 'accessToken'"))?
        .to_string();
    let expires_at = match v.get("expires_on").and_then(Value::as_i64) {
        Some(epoch) if epoch > 0 => UNIX_EPOCH + Duration::from_secs(epoch as u64),
        _ => SystemTime::now() + Duration::from_secs(CLI_FALLBACK_TTL_SECS),
    };
    Ok((token, expires_at))
}

/// Convert an Entra ID scope (`"<resource>/.default"`) to the bare resource URI
/// IMDS expects (`resource=<resource>`).
fn resource_from_scope(scope: &str) -> &str {
    scope.strip_suffix("/.default").unwrap_or(scope)
}

/// Names of `vars` whose resolved value is `None`, in the given order — used
/// by [`EnvironmentCredential::new`] and [`WorkloadIdentityCredential`] to
/// build a single "missing: X, Y" error message instead of failing on just
/// the first missing variable.
fn missing_env_vars<'a>(vars: &[(&'a str, &Option<String>)]) -> Vec<&'a str> {
    vars.iter()
        .filter(|(_, v)| v.is_none())
        .map(|(name, _)| *name)
        .collect()
}

// ---------------------------------------------------------------------------
// AzureCliCredential
// ---------------------------------------------------------------------------

/// Authenticates by shelling out to the Azure CLI
/// (`az account get-access-token --scope <scope> --output json`).
///
/// Useful for local development where a developer is already signed in via
/// `az login`. Requires the `az` binary on `PATH`; a clear error is returned
/// when it is missing.
pub struct AzureCliCredential {
    program: String,
    default_scope: String,
    cache: TokenCache,
}

impl AzureCliCredential {
    /// Create a credential that acquires tokens for `scope` (e.g.
    /// `"https://ai.azure.com/.default"`).
    pub fn new(scope: impl Into<String>) -> Self {
        Self {
            program: "az".to_string(),
            default_scope: scope.into(),
            cache: TokenCache::default(),
        }
    }

    /// Override the CLI executable (default `"az"`), e.g. an absolute path or
    /// `"az.cmd"` on Windows.
    pub fn with_command(mut self, program: impl Into<String>) -> Self {
        self.program = program.into();
        self
    }

    async fn token(&self, scope: &str) -> Result<String> {
        if let Some(t) = self.cache.get(scope) {
            return Ok(t);
        }
        let (token, expires_at) = self.fetch(scope).await?;
        self.cache.put(scope, token.clone(), expires_at);
        Ok(token)
    }

    async fn fetch(&self, scope: &str) -> Result<(String, SystemTime)> {
        let output = tokio::process::Command::new(&self.program)
            .arg("account")
            .arg("get-access-token")
            .arg("--scope")
            .arg(scope)
            .arg("--output")
            .arg("json")
            .output()
            .await;
        match output {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(Error::Configuration(format!(
                    "Azure CLI ('{}') was not found on PATH; run `az login` after installing it, \
                     or use a different credential: {e}",
                    self.program
                )))
            }
            Err(e) => Err(Error::service(format!("failed to run Azure CLI: {e}"))),
            Ok(out) if !out.status.success() => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                Err(Error::other(format!(
                    "Azure CLI token request failed ({}): {}",
                    out.status,
                    stderr.trim()
                )))
            }
            Ok(out) => parse_cli_output(&out.stdout),
        }
    }
}

#[async_trait]
impl TokenCredential for AzureCliCredential {
    async fn get_token(&self) -> Result<String> {
        self.token(&self.default_scope).await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.token(scope).await
    }
}

// ---------------------------------------------------------------------------
// ClientSecretCredential
// ---------------------------------------------------------------------------

/// Authenticates a confidential client via the OAuth2 client-credentials flow
/// (`POST {authority}/{tenant}/oauth2/v2.0/token`).
pub struct ClientSecretCredential {
    http: reqwest::Client,
    authority: String,
    tenant_id: String,
    client_id: String,
    client_secret: String,
    default_scope: String,
    cache: TokenCache,
}

impl ClientSecretCredential {
    /// Create a client-secret credential for the given tenant/app registration,
    /// acquiring tokens for `scope`.
    pub fn new(
        tenant_id: impl Into<String>,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        scope: impl Into<String>,
    ) -> Self {
        Self {
            http: reqwest::Client::new(),
            authority: DEFAULT_AUTHORITY.to_string(),
            tenant_id: tenant_id.into(),
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            default_scope: scope.into(),
            cache: TokenCache::default(),
        }
    }

    /// Override the Entra authority (default
    /// [`DEFAULT_AUTHORITY`]) — e.g. a sovereign cloud, or a loopback in tests.
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authority = authority.into();
        self
    }

    fn token_url(&self) -> String {
        format!(
            "{}/{}/oauth2/v2.0/token",
            self.authority.trim_end_matches('/'),
            self.tenant_id
        )
    }

    async fn token(&self, scope: &str) -> Result<String> {
        if let Some(t) = self.cache.get(scope) {
            return Ok(t);
        }
        let (token, expires_at) = self.fetch(scope).await?;
        self.cache.put(scope, token.clone(), expires_at);
        Ok(token)
    }

    async fn fetch(&self, scope: &str) -> Result<(String, SystemTime)> {
        let params = [
            ("grant_type", "client_credentials"),
            ("client_id", self.client_id.as_str()),
            ("client_secret", self.client_secret.as_str()),
            ("scope", scope),
        ];
        let resp = self
            .http
            .post(self.token_url())
            .form(&params)
            .send()
            .await
            .map_err(|e| Error::service(format!("client-secret token request failed: {e}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::other(format!(
                "client-secret token request rejected ({}): {}",
                status,
                body.trim()
            )));
        }
        parse_oauth_token(&body)
    }
}

#[async_trait]
impl TokenCredential for ClientSecretCredential {
    async fn get_token(&self) -> Result<String> {
        self.token(&self.default_scope).await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.token(scope).await
    }
}

// ---------------------------------------------------------------------------
// EnvironmentCredential
// ---------------------------------------------------------------------------

/// Authenticates a service principal configured entirely by environment
/// variables, via the client-secret flow: `AZURE_TENANT_ID`,
/// `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`.
///
/// A restricted port of `azure_identity`'s `EnvironmentCredential`: upstream
/// also supports a certificate flow and a (deprecated) username/password
/// flow, neither of which this crate implements, so only the client-secret
/// configuration is recognized. Unlike upstream — which defers the
/// "not configured" failure to the first
/// [`get_token`](TokenCredential::get_token) call — [`new`](Self::new) fails
/// immediately, listing every missing variable, so misconfiguration is caught
/// at startup rather than on first use.
pub struct EnvironmentCredential {
    inner: ClientSecretCredential,
}

impl EnvironmentCredential {
    /// Read `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and `AZURE_CLIENT_SECRET`
    /// from the environment and build a client-secret credential acquiring
    /// tokens for `scope`.
    ///
    /// # Errors
    /// [`Error::Configuration`] naming every missing variable, when one or
    /// more of the three are unset.
    pub fn new(scope: impl Into<String>) -> Result<Self> {
        let tenant_id = std::env::var("AZURE_TENANT_ID").ok();
        let client_id = std::env::var("AZURE_CLIENT_ID").ok();
        let client_secret = std::env::var("AZURE_CLIENT_SECRET").ok();

        let missing = missing_env_vars(&[
            ("AZURE_TENANT_ID", &tenant_id),
            ("AZURE_CLIENT_ID", &client_id),
            ("AZURE_CLIENT_SECRET", &client_secret),
        ]);
        if !missing.is_empty() {
            return Err(Error::Configuration(format!(
                "EnvironmentCredential: missing required environment variable(s): {}",
                missing.join(", ")
            )));
        }

        Ok(Self {
            inner: ClientSecretCredential::new(
                tenant_id.unwrap(),
                client_id.unwrap(),
                client_secret.unwrap(),
                scope,
            ),
        })
    }

    /// Override the Entra authority used by the inner client-secret
    /// credential (default [`DEFAULT_AUTHORITY`]).
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.inner = self.inner.with_authority(authority);
        self
    }
}

#[async_trait]
impl TokenCredential for EnvironmentCredential {
    async fn get_token(&self) -> Result<String> {
        self.inner.get_token().await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.inner.get_token_for_scope(scope).await
    }
}

// ---------------------------------------------------------------------------
// ManagedIdentityCredential
// ---------------------------------------------------------------------------

/// Authenticates via an Azure Managed Identity by calling the Instance Metadata
/// Service (IMDS) token endpoint
/// (`GET {endpoint}?api-version=2018-02-01&resource=<resource>` with the
/// `Metadata: true` header).
///
/// The endpoint defaults to the IMDS address but is overridden from the
/// `IDENTITY_ENDPOINT`/`MSI_ENDPOINT` environment variables (as set by App
/// Service / Functions) or via [`with_endpoint`](Self::with_endpoint) — which
/// also makes it loopback-testable. An optional user-assigned identity client
/// id may be supplied.
pub struct ManagedIdentityCredential {
    http: reqwest::Client,
    endpoint: String,
    client_id: Option<String>,
    identity_header: Option<String>,
    default_scope: String,
    cache: TokenCache,
}

impl ManagedIdentityCredential {
    /// Create a managed-identity credential acquiring tokens for `scope`.
    ///
    /// The IMDS endpoint is taken from `IDENTITY_ENDPOINT`, then `MSI_ENDPOINT`,
    /// then [`DEFAULT_IMDS_ENDPOINT`]; an `IDENTITY_HEADER`, when present, is
    /// forwarded as `X-IDENTITY-HEADER`.
    pub fn new(scope: impl Into<String>) -> Self {
        let endpoint = std::env::var("IDENTITY_ENDPOINT")
            .or_else(|_| std::env::var("MSI_ENDPOINT"))
            .unwrap_or_else(|_| DEFAULT_IMDS_ENDPOINT.to_string());
        let identity_header = std::env::var("IDENTITY_HEADER").ok();
        Self {
            http: reqwest::Client::new(),
            endpoint,
            client_id: None,
            identity_header,
            default_scope: scope.into(),
            cache: TokenCache::default(),
        }
    }

    /// Override the token endpoint (default [`DEFAULT_IMDS_ENDPOINT`] or the
    /// environment) — the seam used by loopback tests.
    pub fn with_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.endpoint = endpoint.into();
        self
    }

    /// Pin a user-assigned managed identity by its client id.
    pub fn with_client_id(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// Set the `X-IDENTITY-HEADER` value (App Service / Functions secret).
    pub fn with_identity_header(mut self, header: impl Into<String>) -> Self {
        self.identity_header = Some(header.into());
        self
    }

    async fn token(&self, scope: &str) -> Result<String> {
        if let Some(t) = self.cache.get(scope) {
            return Ok(t);
        }
        let (token, expires_at) = self.fetch(scope).await?;
        self.cache.put(scope, token.clone(), expires_at);
        Ok(token)
    }

    async fn fetch(&self, scope: &str) -> Result<(String, SystemTime)> {
        let resource = resource_from_scope(scope);
        let mut params: Vec<(&str, &str)> =
            vec![("api-version", IMDS_API_VERSION), ("resource", resource)];
        if let Some(cid) = &self.client_id {
            params.push(("client_id", cid.as_str()));
        }
        let mut req = self
            .http
            .get(&self.endpoint)
            .query(&params)
            .header("Metadata", "true");
        if let Some(h) = &self.identity_header {
            req = req.header("X-IDENTITY-HEADER", h);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("managed-identity token request failed: {e}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::other(format!(
                "managed-identity token request rejected ({}): {}",
                status,
                body.trim()
            )));
        }
        parse_oauth_token(&body)
    }
}

#[async_trait]
impl TokenCredential for ManagedIdentityCredential {
    async fn get_token(&self) -> Result<String> {
        self.token(&self.default_scope).await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.token(scope).await
    }
}

// ---------------------------------------------------------------------------
// WorkloadIdentityCredential
// ---------------------------------------------------------------------------

/// Authenticates via Microsoft Entra Workload ID (Azure Kubernetes Service
/// workload identity federation): exchanges a Kubernetes projected
/// service-account token for an Entra access token using the OAuth2
/// client-credentials grant with a JWT client assertion
/// (`POST {authority}/{tenant}/oauth2/v2.0/token`,
/// `client_assertion_type=urn:ietf:params:oauth:client-assertion-type:jwt-bearer`).
///
/// By default `tenant_id`, `client_id`, and the federated token file path are
/// read from `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and
/// `AZURE_FEDERATED_TOKEN_FILE` — the variables the AKS workload identity
/// webhook injects into a pod — mirroring `azure_identity`'s
/// `WorkloadIdentityCredential`. The token file is re-read on every token
/// *request* (not cached once at startup), since the kubelet periodically
/// rotates the projected token.
pub struct WorkloadIdentityCredential {
    http: reqwest::Client,
    authority: String,
    tenant_id: String,
    client_id: String,
    token_file_path: String,
    default_scope: String,
    cache: TokenCache,
}

impl WorkloadIdentityCredential {
    /// Read `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, and
    /// `AZURE_FEDERATED_TOKEN_FILE` from the environment, acquiring tokens for
    /// `scope`.
    ///
    /// # Errors
    /// [`Error::Configuration`] naming every missing variable, when one or
    /// more of the three are unset.
    pub fn new(scope: impl Into<String>) -> Result<Self> {
        Self::with_overrides(scope, None, None, None)
    }

    /// Same as [`new`](Self::new), but the tenant id, client id, and/or
    /// federated-token-file path can be supplied directly instead of read
    /// from the environment. Pass `None` for a field to keep reading it from
    /// the usual environment variable.
    ///
    /// # Errors
    /// [`Error::Configuration`] naming every value that is neither passed
    /// here nor found in the environment.
    pub fn with_overrides(
        scope: impl Into<String>,
        tenant_id: Option<String>,
        client_id: Option<String>,
        token_file_path: Option<String>,
    ) -> Result<Self> {
        let tenant_id = tenant_id.or_else(|| std::env::var("AZURE_TENANT_ID").ok());
        let client_id = client_id.or_else(|| std::env::var("AZURE_CLIENT_ID").ok());
        let token_file_path =
            token_file_path.or_else(|| std::env::var("AZURE_FEDERATED_TOKEN_FILE").ok());

        let missing = missing_env_vars(&[
            ("AZURE_TENANT_ID", &tenant_id),
            ("AZURE_CLIENT_ID", &client_id),
            ("AZURE_FEDERATED_TOKEN_FILE", &token_file_path),
        ]);
        if !missing.is_empty() {
            return Err(Error::Configuration(format!(
                "WorkloadIdentityCredential: missing required configuration: {}",
                missing.join(", ")
            )));
        }

        Ok(Self {
            http: reqwest::Client::new(),
            authority: DEFAULT_AUTHORITY.to_string(),
            tenant_id: tenant_id.unwrap(),
            client_id: client_id.unwrap(),
            token_file_path: token_file_path.unwrap(),
            default_scope: scope.into(),
            cache: TokenCache::default(),
        })
    }

    /// Same as [`new`](Self::new), overriding only the client id (default:
    /// `AZURE_CLIENT_ID`) — e.g. to authenticate as an app registration whose
    /// federated credential trusts this pod's service account token, distinct
    /// from the identity the AKS webhook injects by default.
    pub fn with_client_id(scope: impl Into<String>, client_id: Option<String>) -> Result<Self> {
        Self::with_overrides(scope, None, client_id, None)
    }

    /// Override the Entra authority (default [`DEFAULT_AUTHORITY`]) — e.g. a
    /// sovereign cloud, or a loopback in tests.
    pub fn with_authority(mut self, authority: impl Into<String>) -> Self {
        self.authority = authority.into();
        self
    }

    fn token_url(&self) -> String {
        format!(
            "{}/{}/oauth2/v2.0/token",
            self.authority.trim_end_matches('/'),
            self.tenant_id
        )
    }

    async fn token(&self, scope: &str) -> Result<String> {
        if let Some(t) = self.cache.get(scope) {
            return Ok(t);
        }
        let (token, expires_at) = self.fetch(scope).await?;
        self.cache.put(scope, token.clone(), expires_at);
        Ok(token)
    }

    /// Build the token-request form body, re-reading the federated token file
    /// fresh every time so a rotated projected token is always picked up.
    /// Factored out from [`fetch`](Self::fetch) so the exact request shape is
    /// unit-testable without a network.
    fn request_params(&self, scope: &str) -> Result<Vec<(String, String)>> {
        let assertion = std::fs::read_to_string(&self.token_file_path).map_err(|e| {
            Error::Configuration(format!(
                "WorkloadIdentityCredential: failed to read federated token file '{}': {e}",
                self.token_file_path
            ))
        })?;
        Ok(vec![
            ("grant_type".to_string(), "client_credentials".to_string()),
            ("client_id".to_string(), self.client_id.clone()),
            ("client_assertion".to_string(), assertion.trim().to_string()),
            (
                "client_assertion_type".to_string(),
                "urn:ietf:params:oauth:client-assertion-type:jwt-bearer".to_string(),
            ),
            ("scope".to_string(), scope.to_string()),
        ])
    }

    async fn fetch(&self, scope: &str) -> Result<(String, SystemTime)> {
        let params = self.request_params(scope)?;
        let resp = self
            .http
            .post(self.token_url())
            .form(&params)
            .send()
            .await
            .map_err(|e| Error::service(format!("workload-identity token request failed: {e}")))?;
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(Error::other(format!(
                "workload-identity token request rejected ({}): {}",
                status,
                body.trim()
            )));
        }
        parse_oauth_token(&body)
    }
}

#[async_trait]
impl TokenCredential for WorkloadIdentityCredential {
    async fn get_token(&self) -> Result<String> {
        self.token(&self.default_scope).await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.token(scope).await
    }
}

// ---------------------------------------------------------------------------
// ChainedTokenCredential
// ---------------------------------------------------------------------------

/// Tries a sequence of credentials in order, returning the first token
/// obtained. The index of the first credential to succeed is remembered and
/// tried first on subsequent calls (re-remembered if a different one later
/// wins), mirroring `azure_identity`'s `ChainedTokenCredential`.
pub struct ChainedTokenCredential {
    credentials: Vec<Arc<dyn TokenCredential>>,
    remembered: Mutex<Option<usize>>,
}

impl ChainedTokenCredential {
    /// Build a chain from an ordered list of credentials.
    pub fn new(credentials: Vec<Arc<dyn TokenCredential>>) -> Self {
        Self {
            credentials,
            remembered: Mutex::new(None),
        }
    }

    async fn acquire(&self, scope: Option<&str>) -> Result<String> {
        // Copy the remembered index out before awaiting; never hold the lock
        // across an `.await`.
        let remembered = *self.remembered.lock().unwrap();

        if let Some(idx) = remembered {
            if let Some(cred) = self.credentials.get(idx) {
                if let Ok(token) = Self::call(cred.as_ref(), scope).await {
                    return Ok(token);
                }
            }
        }

        let mut errors = Vec::new();
        for (i, cred) in self.credentials.iter().enumerate() {
            if Some(i) == remembered {
                // Already attempted above.
                continue;
            }
            match Self::call(cred.as_ref(), scope).await {
                Ok(token) => {
                    *self.remembered.lock().unwrap() = Some(i);
                    return Ok(token);
                }
                Err(e) => errors.push(format!("credential[{i}]: {e}")),
            }
        }

        Err(Error::Configuration(format!(
            "ChainedTokenCredential: no credential in the chain returned a token ({})",
            if errors.is_empty() {
                "the chain is empty".to_string()
            } else {
                errors.join("; ")
            }
        )))
    }

    async fn call(cred: &dyn TokenCredential, scope: Option<&str>) -> Result<String> {
        match scope {
            Some(s) => cred.get_token_for_scope(s).await,
            None => cred.get_token().await,
        }
    }
}

#[async_trait]
impl TokenCredential for ChainedTokenCredential {
    async fn get_token(&self) -> Result<String> {
        self.acquire(None).await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.acquire(Some(scope)).await
    }
}

// ---------------------------------------------------------------------------
// DefaultAzureCredential
// ---------------------------------------------------------------------------

/// A prebuilt [`ChainedTokenCredential`], in (a subset of) `azure_identity`'s
/// documented `DefaultAzureCredential` order.
///
/// Upstream (Python/.NET) `DefaultAzureCredential` tries, in order:
/// `EnvironmentCredential`, `WorkloadIdentityCredential`,
/// `ManagedIdentityCredential`, `SharedTokenCacheCredential` (Windows only),
/// `VisualStudioCodeCredential`, `AzureCliCredential`,
/// `AzurePowerShellCredential`, `AzureDeveloperCliCredential`, and
/// (Windows/WSL only, off by default) `InteractiveBrowserCredential`/broker
/// authentication.
///
/// This crate implements four of those credential types, so this chain is
/// **`EnvironmentCredential` → `WorkloadIdentityCredential` →
/// `ManagedIdentityCredential` → `AzureCliCredential`** — the same relative
/// order upstream uses among the credentials that exist here. Every other
/// link in upstream's list (the shared token cache, IDE-signed-in
/// credentials, PowerShell, the Azure Developer CLI, interactive/broker auth)
/// is simply absent rather than approximated by something else; callers that
/// need one of those should build their own [`ChainedTokenCredential`].
///
/// `EnvironmentCredential` and `WorkloadIdentityCredential` are included only
/// when their required environment variables are present — construction
/// failure just excludes them from the chain, mirroring upstream's
/// "unavailable credentials are skipped, not fatal" behavior (there, via a
/// placeholder that defers the error to `get_token`; here, by omission, since
/// [`ChainedTokenCredential`] already only holds constructed credentials).
/// `ManagedIdentityCredential` and `AzureCliCredential` need no such
/// configuration and are always present, so the chain is never empty. As with
/// any [`ChainedTokenCredential`], whichever credential first returns a token
/// is remembered and tried first on subsequent calls.
pub struct DefaultAzureCredential {
    chain: ChainedTokenCredential,
}

impl DefaultAzureCredential {
    /// Build the chain, acquiring tokens for `scope` by default.
    pub fn new(scope: impl Into<String>) -> Self {
        let scope = scope.into();
        Self {
            chain: ChainedTokenCredential::new(default_chain(&scope)),
        }
    }
}

/// The ordered, environment-filtered credential list backing
/// [`DefaultAzureCredential`]. A free function so the inclusion logic is
/// independently unit-testable without a network.
fn default_chain(scope: &str) -> Vec<Arc<dyn TokenCredential>> {
    let mut chain: Vec<Arc<dyn TokenCredential>> = Vec::new();
    if let Ok(cred) = EnvironmentCredential::new(scope) {
        chain.push(Arc::new(cred));
    }
    if let Ok(cred) = WorkloadIdentityCredential::new(scope) {
        chain.push(Arc::new(cred));
    }
    chain.push(Arc::new(ManagedIdentityCredential::new(scope)));
    chain.push(Arc::new(AzureCliCredential::new(scope)));
    chain
}

#[async_trait]
impl TokenCredential for DefaultAzureCredential {
    async fn get_token(&self) -> Result<String> {
        self.chain.get_token().await
    }
    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        self.chain.get_token_for_scope(scope).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // region: token cache

    #[test]
    fn cache_returns_token_before_skew_window() {
        let cache = TokenCache::default();
        cache.put(
            "scope",
            "tok".into(),
            SystemTime::now() + Duration::from_secs(3600),
        );
        assert_eq!(cache.get("scope").as_deref(), Some("tok"));
    }

    #[test]
    fn cache_misses_within_refresh_skew() {
        let cache = TokenCache::default();
        // Expires in 60s, inside the 120s skew → treated as stale.
        cache.put(
            "scope",
            "tok".into(),
            SystemTime::now() + Duration::from_secs(60),
        );
        assert!(cache.get("scope").is_none());
    }

    #[test]
    fn cache_is_per_scope() {
        let cache = TokenCache::default();
        cache.put(
            "a",
            "tok-a".into(),
            SystemTime::now() + Duration::from_secs(3600),
        );
        assert_eq!(cache.get("a").as_deref(), Some("tok-a"));
        assert!(cache.get("b").is_none());
    }

    // endregion

    // region: parsing helpers

    #[test]
    fn parse_cli_output_reads_token_and_epoch_expiry() {
        let json = br#"{"accessToken":"cli-token","expiresOn":"2099-01-01 00:00:00.000000","expires_on":4070908800,"tokenType":"Bearer"}"#;
        let (token, expires_at) = parse_cli_output(json).unwrap();
        assert_eq!(token, "cli-token");
        assert_eq!(expires_at, UNIX_EPOCH + Duration::from_secs(4070908800));
    }

    #[test]
    fn parse_cli_output_falls_back_to_ttl_without_epoch() {
        let before = SystemTime::now();
        let (token, expires_at) = parse_cli_output(br#"{"accessToken":"t"}"#).unwrap();
        assert_eq!(token, "t");
        assert!(expires_at >= before + Duration::from_secs(CLI_FALLBACK_TTL_SECS));
    }

    #[test]
    fn parse_cli_output_errors_without_token() {
        assert!(parse_cli_output(br#"{"tokenType":"Bearer"}"#).is_err());
    }

    #[test]
    fn parse_oauth_token_number_and_string_expiry() {
        let (t, _) = parse_oauth_token(r#"{"access_token":"a","expires_in":3600}"#).unwrap();
        assert_eq!(t, "a");
        // IMDS returns expires_in as a string.
        let (t2, _) = parse_oauth_token(r#"{"access_token":"b","expires_in":"3600"}"#).unwrap();
        assert_eq!(t2, "b");
    }

    #[test]
    fn resource_strips_default_suffix() {
        assert_eq!(
            resource_from_scope("https://ai.azure.com/.default"),
            "https://ai.azure.com"
        );
        assert_eq!(
            resource_from_scope("https://vault.azure.net"),
            "https://vault.azure.net"
        );
    }

    // endregion

    // region: azure cli missing-binary path

    #[tokio::test]
    async fn azure_cli_missing_binary_gives_clear_error() {
        let cred = AzureCliCredential::new("https://ai.azure.com/.default")
            .with_command("definitely-not-a-real-az-binary-xyz");
        let err = cred.get_token().await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("was not found on PATH"),
            "expected a clear missing-CLI message, got: {msg}"
        );
    }

    // endregion

    // region: chain order + remembering

    struct FakeCredential {
        result: std::result::Result<String, ()>,
        calls: AtomicUsize,
    }

    impl FakeCredential {
        fn ok(token: &str) -> Self {
            Self {
                result: Ok(token.to_string()),
                calls: AtomicUsize::new(0),
            }
        }
        fn fail() -> Self {
            Self {
                result: Err(()),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl TokenCredential for FakeCredential {
        async fn get_token(&self) -> Result<String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.result
                .clone()
                .map_err(|_| Error::other("fake credential failure"))
        }
    }

    #[tokio::test]
    async fn chain_returns_first_success_and_remembers_it() {
        let first = Arc::new(FakeCredential::fail());
        let second = Arc::new(FakeCredential::ok("winner"));
        let third = Arc::new(FakeCredential::ok("unused"));
        let chain = ChainedTokenCredential::new(vec![first.clone(), second.clone(), third.clone()]);

        assert_eq!(chain.get_token().await.unwrap(), "winner");
        // First and second were tried; third never reached.
        assert_eq!(first.calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.calls.load(Ordering::SeqCst), 1);
        assert_eq!(third.calls.load(Ordering::SeqCst), 0);

        // Second is remembered: the next call goes straight to it, skipping the
        // failing first credential.
        assert_eq!(chain.get_token().await.unwrap(), "winner");
        assert_eq!(first.calls.load(Ordering::SeqCst), 1);
        assert_eq!(second.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn chain_reports_all_failures_when_none_succeed() {
        let chain = ChainedTokenCredential::new(vec![
            Arc::new(FakeCredential::fail()),
            Arc::new(FakeCredential::fail()),
        ]);
        let err = chain.get_token().await.unwrap_err();
        assert!(err.to_string().contains("no credential in the chain"));
    }

    #[tokio::test]
    async fn empty_chain_errors() {
        let chain = ChainedTokenCredential::new(vec![]);
        assert!(chain.get_token().await.is_err());
    }

    // endregion

    // region: env-var-backed credentials (Environment / WorkloadIdentity / Default)

    /// Guards mutation of the Entra env vars consumed by
    /// `EnvironmentCredential`/`WorkloadIdentityCredential`/
    /// `DefaultAzureCredential`: tests in this module run on multiple
    /// threads, and env vars are process-global (same pattern as
    /// `agent-framework-mem0`'s `ENV_MUTEX`).
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    const ENTRA_ENV_VARS: &[&str] = &[
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_FEDERATED_TOKEN_FILE",
    ];

    fn clear_entra_env() {
        for key in ENTRA_ENV_VARS {
            // SAFETY: serialized by ENV_MUTEX; no other test in this crate
            // touches these variables.
            unsafe { std::env::remove_var(key) };
        }
    }

    const TEST_SCOPE: &str = "https://ai.azure.com/.default";

    #[test]
    fn environment_credential_errors_listing_all_missing_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        // `.err().unwrap()` rather than `.unwrap_err()`: the credential types
        // deliberately don't derive `Debug` (they can hold secrets), and only
        // `unwrap_err()` requires the `Ok` side to be `Debug`.
        let err = EnvironmentCredential::new(TEST_SCOPE).err().unwrap();
        clear_entra_env();
        let msg = err.to_string();
        assert!(msg.contains("AZURE_TENANT_ID"), "{msg}");
        assert!(msg.contains("AZURE_CLIENT_ID"), "{msg}");
        assert!(msg.contains("AZURE_CLIENT_SECRET"), "{msg}");
    }

    #[test]
    fn environment_credential_lists_only_the_vars_still_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_CLIENT_ID", "client");
        }
        let err = EnvironmentCredential::new(TEST_SCOPE).err().unwrap();
        clear_entra_env();
        let msg = err.to_string();
        assert!(msg.contains("AZURE_CLIENT_SECRET"), "{msg}");
        assert!(!msg.contains("AZURE_TENANT_ID"), "{msg}");
        assert!(!msg.contains("AZURE_CLIENT_ID"), "{msg}");
    }

    #[test]
    fn environment_credential_succeeds_with_all_three_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_CLIENT_ID", "client");
            std::env::set_var("AZURE_CLIENT_SECRET", "secret");
        }
        let cred = EnvironmentCredential::new(TEST_SCOPE);
        clear_entra_env();
        assert!(cred.is_ok());
    }

    #[test]
    fn workload_identity_errors_listing_all_missing_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        let err = WorkloadIdentityCredential::new(TEST_SCOPE).err().unwrap();
        clear_entra_env();
        let msg = err.to_string();
        assert!(msg.contains("AZURE_TENANT_ID"), "{msg}");
        assert!(msg.contains("AZURE_CLIENT_ID"), "{msg}");
        assert!(msg.contains("AZURE_FEDERATED_TOKEN_FILE"), "{msg}");
    }

    #[test]
    fn workload_identity_client_id_override_fills_in_for_missing_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_FEDERATED_TOKEN_FILE", "placeholder-token-file");
        }
        let cred = WorkloadIdentityCredential::with_client_id(
            TEST_SCOPE,
            Some("explicit-client".to_string()),
        );
        clear_entra_env();
        assert!(cred.is_ok());
    }

    #[test]
    fn workload_identity_request_params_build_client_assertion_shape() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // A real scratch file: `request_params` reads it fresh every call.
        let path = std::env::temp_dir().join(format!(
            "af-workload-identity-test-{}-{:?}.jwt",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, "fake.jwt.token").unwrap();

        let cred = WorkloadIdentityCredential::with_overrides(
            TEST_SCOPE,
            Some("tenant-1".to_string()),
            Some("client-1".to_string()),
            Some(path.to_str().unwrap().to_string()),
        )
        .unwrap();

        let params = cred.request_params(TEST_SCOPE).unwrap();
        let get = |key: &str| {
            params
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("grant_type").as_deref(), Some("client_credentials"));
        assert_eq!(get("client_id").as_deref(), Some("client-1"));
        assert_eq!(get("client_assertion").as_deref(), Some("fake.jwt.token"));
        assert_eq!(
            get("client_assertion_type").as_deref(),
            Some("urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
        );
        assert_eq!(get("scope").as_deref(), Some(TEST_SCOPE));
        assert_eq!(
            cred.token_url(),
            "https://login.microsoftonline.com/tenant-1/oauth2/v2.0/token"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn workload_identity_rereads_token_file_on_each_request() {
        let path = std::env::temp_dir().join(format!(
            "af-workload-identity-rotate-{}-{:?}.jwt",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, "first-token").unwrap();

        let cred = WorkloadIdentityCredential::with_overrides(
            TEST_SCOPE,
            Some("tenant-1".to_string()),
            Some("client-1".to_string()),
            Some(path.to_str().unwrap().to_string()),
        )
        .unwrap();

        let first = cred.request_params("scope").unwrap();
        let assertion_of = |params: &[(String, String)]| {
            params
                .iter()
                .find(|(k, _)| k == "client_assertion")
                .map(|(_, v)| v.clone())
        };
        assert_eq!(assertion_of(&first).as_deref(), Some("first-token"));

        std::fs::write(&path, "rotated-token").unwrap();
        let second = cred.request_params("scope").unwrap();
        assert_eq!(assertion_of(&second).as_deref(), Some("rotated-token"));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn workload_identity_missing_token_file_is_a_clear_error() {
        let cred = WorkloadIdentityCredential::with_overrides(
            TEST_SCOPE,
            Some("tenant-1".to_string()),
            Some("client-1".to_string()),
            Some("/definitely/not/a/real/path/af-test.jwt".to_string()),
        )
        .unwrap();
        let err = cred.request_params(TEST_SCOPE).unwrap_err();
        assert!(err.to_string().contains("federated token file"), "{err}");
    }

    // endregion

    // region: DefaultAzureCredential chain composition/order

    #[test]
    fn default_chain_includes_only_always_on_credentials_without_env() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        let chain = default_chain(TEST_SCOPE);
        clear_entra_env();
        // Only ManagedIdentityCredential + AzureCliCredential need no
        // configuration, so they're the only two present.
        assert_eq!(chain.len(), 2);
    }

    #[test]
    fn default_chain_adds_environment_credential_when_configured() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_CLIENT_ID", "client");
            std::env::set_var("AZURE_CLIENT_SECRET", "secret");
        }
        let chain = default_chain(TEST_SCOPE);
        clear_entra_env();
        assert_eq!(chain.len(), 3, "Environment + ManagedIdentity + AzureCli");
    }

    #[test]
    fn default_chain_adds_workload_identity_when_configured() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_CLIENT_ID", "client");
            std::env::set_var("AZURE_FEDERATED_TOKEN_FILE", "placeholder-token-file");
        }
        let chain = default_chain(TEST_SCOPE);
        clear_entra_env();
        assert_eq!(
            chain.len(),
            3,
            "WorkloadIdentity + ManagedIdentity + AzureCli"
        );
    }

    #[test]
    fn default_chain_adds_both_environment_and_workload_identity_in_order() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        unsafe {
            std::env::set_var("AZURE_TENANT_ID", "tenant");
            std::env::set_var("AZURE_CLIENT_ID", "client");
            std::env::set_var("AZURE_CLIENT_SECRET", "secret");
            std::env::set_var("AZURE_FEDERATED_TOKEN_FILE", "placeholder-token-file");
        }
        let chain = default_chain(TEST_SCOPE);
        clear_entra_env();
        assert_eq!(chain.len(), 4, "all four credentials configured");
    }

    #[test]
    fn default_azure_credential_wraps_a_never_empty_chain() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_entra_env();
        let cred = DefaultAzureCredential::new(TEST_SCOPE);
        clear_entra_env();
        assert_eq!(cred.chain.credentials.len(), 2);
    }

    // endregion
}
