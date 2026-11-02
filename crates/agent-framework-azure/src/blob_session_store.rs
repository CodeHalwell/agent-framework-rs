//! An [`AgentSessionStore`] backed by Azure Blob Storage — the port of
//! upstream's `AzureBlobAgentSessionStore`
//! (`Microsoft.Agents.AI.Hosting.AzureStorage`). Behind the `session-store`
//! feature, which enables core's `experimental-session-store`.
//!
//! Each session is one JSON blob. The blob name follows upstream's scheme
//! exactly, so a container can be shared with the .NET store:
//!
//! ```text
//! [<prefix>/]v2/<SHA-256(agent namespace)>/<SHA-256(session key)>.json
//! ```
//!
//! where the session key is the session id and each partition name and value,
//! length-prefixed (`s<len>:<id>|n<len>:<name>|v<len>:<value>|…`) so that no
//! choice of ids can make two different keys hash the same input. Hashes are
//! upper-case hex, as .NET's `Convert.ToHexString` writes them.
//!
//! It talks to the Blob REST API directly through this crate's HTTP client,
//! authenticating with any [`TokenCredential`] (scoped to
//! [`STORAGE_SCOPE`]) or a SAS token, so it adds no Azure SDK dependency.
//! `azure_storage_blob` is stable and could replace the transport later (see
//! `docs/azure-rust-sdk.md`); the blob layout would not change.
//!
//! # Divergences
//!
//! - **Agent namespace.** Like upstream, sessions are scoped by an
//!   `agent_namespace` given to the store, not by the agent's id, so stores
//!   on different instances of one agent share sessions. The `agent`
//!   argument of each call is used only to restore the session.
//! - **Snapshot shape.** The blob holds this port's
//!   [`serialize_session`] snapshot, not .NET's `SerializeSessionAsync`
//!   output; the two stores share a naming scheme, not a session format.

use std::sync::Arc;

use async_trait::async_trait;
use reqwest::{Method, StatusCode};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::OnceCell;

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{error_chain, Error, Result};
use agent_framework_core::session::AgentSession;
use agent_framework_core::session_store::{
    deserialize_session, serialize_session, AgentSessionStore, AgentSessionStoreKey,
};

use crate::TokenCredential;

/// The Entra ID scope for Azure Storage data-plane access.
pub const STORAGE_SCOPE: &str = "https://storage.azure.com/.default";

/// The Blob service REST API version sent on every request.
const BLOB_API_VERSION: &str = "2023-11-03";

/// The maximum blob name length the service accepts.
const MAX_BLOB_NAME_LENGTH: usize = 1024;

/// `v2/` + 64 hex + `/` + 64 hex + `.json`.
const BASE_BLOB_NAME_LENGTH: usize = 137;

/// How requests to the container authenticate.
#[derive(Clone)]
pub enum BlobAuth {
    /// `Authorization: Bearer <token>` from an Entra ID credential, requested
    /// for [`STORAGE_SCOPE`]. The identity needs a data role on the
    /// container (e.g. *Storage Blob Data Contributor*).
    Credential(Arc<dyn TokenCredential>),
    /// A shared access signature, appended to every request's query string.
    /// Leading `?` is optional.
    Sas(String),
}

impl std::fmt::Debug for BlobAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlobAuth::Credential(_) => f.write_str("BlobAuth::Credential(..)"),
            BlobAuth::Sas(_) => f.write_str("BlobAuth::Sas(<redacted>)"),
        }
    }
}

/// Stores agent sessions as JSON blobs in one Azure Blob Storage container.
#[derive(Debug)]
pub struct AzureBlobAgentSessionStore {
    container_url: String,
    auth: BlobAuth,
    agent_key: String,
    blob_name_prefix: Option<String>,
    create_container_if_not_exists: bool,
    container_ready: OnceCell<()>,
    http: reqwest::Client,
}

impl AzureBlobAgentSessionStore {
    /// A store over the container at `container_url`
    /// (`https://<account>.blob.core.windows.net/<container>`), scoping its
    /// sessions by `agent_namespace`.
    ///
    /// The container is created on first use if it does not exist; turn that
    /// off with [`Self::create_container_if_not_exists`] when the identity
    /// has data access but cannot create containers.
    ///
    /// # Errors
    /// [`Error::Configuration`] for an empty `agent_namespace` or a
    /// `container_url` that is not `http(s)`.
    pub fn new(
        container_url: impl Into<String>,
        auth: BlobAuth,
        agent_namespace: &str,
    ) -> Result<Self> {
        let container_url = container_url.into().trim_end_matches('/').to_string();
        if !(container_url.starts_with("https://") || container_url.starts_with("http://")) {
            return Err(Error::Configuration(format!(
                "Blob container URL must be http(s): {container_url}"
            )));
        }
        if agent_namespace.trim().is_empty() {
            return Err(Error::Configuration(
                "agent_namespace must be a non-empty string".into(),
            ));
        }
        Ok(Self {
            container_url,
            auth,
            agent_key: sha256_hex(agent_namespace),
            blob_name_prefix: None,
            create_container_if_not_exists: true,
            container_ready: OnceCell::new(),
            http: crate::http_client(),
        })
    }

    /// Namespace this store's blobs under `prefix` (surrounding `/` are
    /// trimmed; an empty prefix means none).
    ///
    /// # Errors
    /// [`Error::Configuration`] when the prefix would push a blob name past
    /// the service's 1,024-character limit (it may be at most 886).
    pub fn with_blob_name_prefix(mut self, prefix: &str) -> Result<Self> {
        let normalized = prefix.trim_matches('/');
        let max = MAX_BLOB_NAME_LENGTH - BASE_BLOB_NAME_LENGTH - 1;
        if normalized.chars().count() > max {
            return Err(Error::Configuration(format!(
                "the blob name prefix must not exceed {max} characters"
            )));
        }
        self.blob_name_prefix = (!normalized.is_empty()).then(|| normalized.to_string());
        Ok(self)
    }

    /// Whether to create the container on first use when it does not exist
    /// (default `true`, as upstream).
    pub fn create_container_if_not_exists(mut self, create: bool) -> Self {
        self.create_container_if_not_exists = create;
        self
    }

    /// The blob name a key is stored under — upstream's scheme exactly.
    pub fn blob_name(&self, key: &AgentSessionStoreKey) -> String {
        let base = format!("v2/{}/{}.json", self.agent_key, session_key_hash(key));
        match &self.blob_name_prefix {
            Some(prefix) => format!("{prefix}/{base}"),
            None => base,
        }
    }

    fn url(&self, path: &str, query: &str) -> String {
        let mut url = format!("{}{path}", self.container_url);
        let mut params: Vec<&str> = Vec::new();
        if !query.is_empty() {
            params.push(query);
        }
        let sas;
        if let BlobAuth::Sas(token) = &self.auth {
            sas = token.trim_start_matches('?').to_string();
            params.push(&sas);
        }
        if !params.is_empty() {
            url.push('?');
            url.push_str(&params.join("&"));
        }
        url
    }

    async fn request(&self, method: Method, url: String) -> Result<reqwest::RequestBuilder> {
        let mut req = self
            .http
            .request(method, url)
            .header("x-ms-version", BLOB_API_VERSION);
        if let BlobAuth::Credential(credential) = &self.auth {
            let token = credential.get_token_for_scope(STORAGE_SCOPE).await?;
            req = req.bearer_auth(token);
        }
        Ok(req)
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> Result<reqwest::Response> {
        req.send()
            .await
            .map_err(|e| Error::service(format!("Azure Blob request failed: {}", error_chain(&e))))
    }

    /// Create the container once, tolerating "already exists". A failure is
    /// not cached, so a later call retries — as upstream resets its
    /// initialization task on failure.
    async fn ensure_container(&self) -> Result<()> {
        if !self.create_container_if_not_exists {
            return Ok(());
        }
        self.container_ready
            .get_or_try_init(|| async {
                let req = self
                    .request(Method::PUT, self.url("", "restype=container"))
                    .await?
                    .header("Content-Length", "0");
                let resp = self.send(req).await?;
                match resp.status() {
                    StatusCode::CREATED | StatusCode::CONFLICT => Ok(()),
                    status => Err(blob_error("create container", status, resp).await),
                }
            })
            .await
            .map(|_| ())
    }

    fn blob_path(&self, key: &AgentSessionStoreKey) -> String {
        format!("/{}", self.blob_name(key))
    }
}

#[async_trait]
impl AgentSessionStore for AzureBlobAgentSessionStore {
    async fn save_session(
        &self,
        _agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
        session: &AgentSession,
    ) -> Result<()> {
        self.ensure_container().await?;
        let body = serde_json::to_vec(&serialize_session(session))
            .map_err(|e| Error::Serialization(format!("failed to serialize session: {e}")))?;
        let req = self
            .request(Method::PUT, self.url(&self.blob_path(key), ""))
            .await?
            .header("x-ms-blob-type", "BlockBlob")
            .header("Content-Type", "application/json")
            .body(body);
        let resp = self.send(req).await?;
        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            Err(blob_error("save session", status, resp).await)
        }
    }

    async fn get_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<Option<AgentSession>> {
        self.ensure_container().await?;
        let req = self
            .request(Method::GET, self.url(&self.blob_path(key), ""))
            .await?;
        let resp = self.send(req).await?;
        let status = resp.status();
        if status == StatusCode::NOT_FOUND && error_code(&resp).as_deref() == Some("BlobNotFound") {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(blob_error("read session", status, resp).await);
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| Error::service(format!("Azure Blob read failed: {}", error_chain(&e))))?;
        let snapshot: Value = serde_json::from_slice(&bytes)
            .map_err(|e| Error::Serialization(format!("failed to parse session blob: {e}")))?;
        deserialize_session(agent, &snapshot).map(Some)
    }

    async fn delete_session(
        &self,
        _agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<()> {
        let req = self
            .request(Method::DELETE, self.url(&self.blob_path(key), ""))
            .await?
            .header("x-ms-delete-snapshots", "include");
        let resp = self.send(req).await?;
        let status = resp.status();
        // A missing blob and a missing container both mean "nothing to
        // delete", as upstream's `DeleteIfExistsAsync`.
        if status.is_success() || status == StatusCode::NOT_FOUND {
            if status == StatusCode::NOT_FOUND {
                tracing::debug!(
                    session_id = key.session_id(),
                    "session not found in the Blob container; nothing to delete"
                );
            }
            return Ok(());
        }
        Err(blob_error("delete session", status, resp).await)
    }
}

fn error_code(resp: &reqwest::Response) -> Option<String> {
    resp.headers()
        .get("x-ms-error-code")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
}

async fn blob_error(op: &str, status: StatusCode, resp: reqwest::Response) -> Error {
    let code = error_code(&resp).unwrap_or_default();
    let body = resp.text().await.unwrap_or_default();
    let detail = if code.is_empty() { body } else { code };
    Error::service_status(
        status.as_u16(),
        format!("Azure Blob {op} failed ({status}): {detail}"),
        None,
    )
}

fn sha256_hex(value: &str) -> String {
    let digest = Sha256::digest(value.as_bytes());
    digest.iter().map(|b| format!("{b:02X}")).collect()
}

/// Upstream's `ComputeSessionKey`: each component length-prefixed so no
/// combination of ids can collide, then hashed.
fn session_key_hash(key: &AgentSessionStoreKey) -> String {
    fn push(out: &mut String, prefix: char, value: &str) {
        out.push(prefix);
        // Length in UTF-16 code units, as .NET's `string.Length`, so the
        // hash matches upstream's for any id.
        out.push_str(&value.encode_utf16().count().to_string());
        out.push(':');
        out.push_str(value);
        out.push('|');
    }
    let mut input = String::new();
    push(&mut input, 's', key.session_id());
    for (name, value) in key.partitions() {
        push(&mut input, 'n', name);
        push(&mut input, 'v', value);
    }
    sha256_hex(&input)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> AzureBlobAgentSessionStore {
        AzureBlobAgentSessionStore::new(
            "https://acct.blob.core.windows.net/sessions/",
            BlobAuth::Sas("?sv=1&sig=x".into()),
            "my-agent",
        )
        .unwrap()
    }

    #[test]
    fn the_blob_name_follows_upstreams_scheme() {
        let key = AgentSessionStoreKey::new("s1").unwrap();
        let name = store().blob_name(&key);
        assert_eq!(name.len(), BASE_BLOB_NAME_LENGTH);
        assert!(name.starts_with("v2/") && name.ends_with(".json"), "{name}");
        // Upstream: SHA-256 of the namespace and of "s2:s1|", upper-case hex.
        assert_eq!(&name[3..67], sha256_hex("my-agent"));
        assert_eq!(&name[68..132], sha256_hex("s2:s1|"));
    }

    #[test]
    fn length_prefixing_keeps_distinct_keys_distinct() {
        // Without the length prefix both would hash "s3:a|b|"-like inputs.
        let a = AgentSessionStoreKey::new("a|v1:b").unwrap();
        let b = AgentSessionStoreKey::new("a")
            .unwrap()
            .with_partition("b", "c")
            .unwrap();
        assert_ne!(session_key_hash(&a), session_key_hash(&b));
    }

    #[test]
    fn partitions_change_the_blob_name() {
        let plain = AgentSessionStoreKey::new("s").unwrap();
        let tenant = plain.clone().with_partition("tenant", "t1").unwrap();
        assert_ne!(store().blob_name(&plain), store().blob_name(&tenant));
    }

    #[test]
    fn a_prefix_is_trimmed_and_prepended() {
        let s = store().with_blob_name_prefix("/prod/").unwrap();
        let name = s.blob_name(&AgentSessionStoreKey::new("s").unwrap());
        assert!(name.starts_with("prod/v2/"), "{name}");
        let none = store().with_blob_name_prefix("//").unwrap();
        assert!(none
            .blob_name(&AgentSessionStoreKey::new("s").unwrap())
            .starts_with("v2/"));
    }

    #[test]
    fn a_prefix_that_would_overflow_the_blob_name_is_refused() {
        assert!(store().with_blob_name_prefix(&"p".repeat(886)).is_ok());
        assert!(store().with_blob_name_prefix(&"p".repeat(887)).is_err());
    }

    #[test]
    fn the_sas_token_is_appended_to_every_url() {
        let s = store();
        assert_eq!(
            s.url("/b.json", ""),
            "https://acct.blob.core.windows.net/sessions/b.json?sv=1&sig=x"
        );
        assert_eq!(
            s.url("", "restype=container"),
            "https://acct.blob.core.windows.net/sessions?restype=container&sv=1&sig=x"
        );
    }

    #[test]
    fn construction_validates_its_inputs() {
        assert!(
            AzureBlobAgentSessionStore::new("ftp://x/c", BlobAuth::Sas(String::new()), "a")
                .is_err()
        );
        assert!(
            AzureBlobAgentSessionStore::new("https://x/c", BlobAuth::Sas(String::new()), " ")
                .is_err()
        );
    }
}
