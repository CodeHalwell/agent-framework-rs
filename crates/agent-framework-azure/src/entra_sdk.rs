//! Entra ID credentials backed by the official [Azure SDK for Rust].
//!
//! Behind the `entra-sdk` feature. This module does not replace the
//! hand-rolled credentials this crate already ships (such as
//! [`ChainedTokenCredential`](crate::ChainedTokenCredential) and
//! [`ManagedIdentityCredential`](crate::ManagedIdentityCredential)) — it
//! adapts *any* credential from the official [`azure_identity`] crate onto
//! this crate's [`TokenCredential`] trait, so both can be used
//! interchangeably wherever a credential is accepted.
//!
//! # Why both exist
//!
//! The hand-rolled chain stays the default: it is dependency-free, and it is
//! what every existing caller already builds against. The SDK path is worth
//! opting into when you want credential types this crate does not implement —
//! [`azure_identity`] ships `ClientCertificateCredential`,
//! `ClientAssertionCredential`, `AzurePipelinesCredential` and
//! `AzureDeveloperCliCredential`, none of which have a hand-rolled equivalent
//! here — or when you would rather Microsoft owned the IMDS quirks, sovereign
//! cloud endpoints and token-lifetime handling under a semver guarantee.
//!
//! # Dependency shape
//!
//! `azure_core` depends on **reqwest 0.13**, a different major version from
//! the 0.12 the rest of this workspace uses. Cargo does not unify features
//! across semver-incompatible versions, so the two are separate packages and
//! the workspace's `rustls-tls` does not reach the Azure SDK's client: it must
//! enable TLS for itself, which is why `azure_core` carries the
//! `reqwest_rustls` feature here. Without it, reqwest 0.13 resolves with no
//! TLS backend and every Entra token request fails before leaving the process
//! — not at compile time, only on the first call.
//!
//! That TLS comes from `aws-lc-rs`, a C library needing `cmake` at build time,
//! so a build with this feature on needs a C toolchain and the graph carries
//! two rustls crypto providers: `ring` behind reqwest 0.12 and `aws-lc-rs`
//! behind reqwest 0.13. A real handshake against Entra completes under that
//! arrangement (see the ignored `tls_probe` test), so the shared rustls has no
//! trouble picking a provider. All of it is gated behind `entra-sdk`, leaving
//! the default build unaffected.
//!
//! # Example
//!
//! ```no_run
//! # use std::sync::Arc;
//! use agent_framework_azure::{AzureOpenAIClient, SdkTokenCredential};
//! use azure_identity::DeveloperToolsCredential;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let credential = DeveloperToolsCredential::new(None)?;
//! let credential = SdkTokenCredential::azure_openai(credential);
//!
//! let client = AzureOpenAIClient::with_token_credential(
//!     "https://my-resource.openai.azure.com",
//!     "gpt-4o",
//!     Arc::new(credential),
//! );
//! # Ok(())
//! # }
//! ```
//!
//! [Azure SDK for Rust]: https://github.com/Azure/azure-sdk-for-rust

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use async_trait::async_trait;
use azure_core::credentials::TokenCredential as AzureTokenCredential;

use crate::TokenCredential;

/// Entra ID scope for the Azure OpenAI data plane.
pub const AZURE_OPENAI_SCOPE: &str = "https://cognitiveservices.azure.com/.default";

/// Entra ID scope for the Azure AI Foundry data plane.
pub const FOUNDRY_SCOPE: &str = "https://ai.azure.com/.default";

/// Adapts any [`azure_identity`] credential onto this crate's
/// [`TokenCredential`].
///
/// Carries a *default scope* because this crate's trait has a no-argument
/// [`get_token`](TokenCredential::get_token) that the Azure SDK's
/// scope-per-call signature has no equivalent for. Callers that need a token
/// for a different audience use
/// [`get_token_for_scope`](TokenCredential::get_token_for_scope), which is
/// forwarded to the SDK unchanged.
///
/// Token caching and refresh are the wrapped credential's responsibility —
/// [`azure_identity`] caches internally, so this adapter deliberately adds no
/// caching layer of its own on top and cannot serve a stale token past the
/// SDK's own refresh window.
#[derive(Debug, Clone)]
pub struct SdkTokenCredential {
    inner: Arc<dyn AzureTokenCredential>,
    default_scope: String,
}

impl SdkTokenCredential {
    /// Wrap an Azure SDK credential, using `default_scope` for
    /// [`get_token`](TokenCredential::get_token).
    pub fn new(inner: Arc<dyn AzureTokenCredential>, default_scope: impl Into<String>) -> Self {
        Self {
            inner,
            default_scope: default_scope.into(),
        }
    }

    /// Wrap a credential defaulting to the Azure OpenAI scope
    /// ([`AZURE_OPENAI_SCOPE`]).
    pub fn azure_openai(inner: Arc<dyn AzureTokenCredential>) -> Self {
        Self::new(inner, AZURE_OPENAI_SCOPE)
    }

    /// Wrap a credential defaulting to the Azure AI Foundry scope
    /// ([`FOUNDRY_SCOPE`]).
    pub fn foundry(inner: Arc<dyn AzureTokenCredential>) -> Self {
        Self::new(inner, FOUNDRY_SCOPE)
    }

    /// The scope used by [`get_token`](TokenCredential::get_token).
    pub fn default_scope(&self) -> &str {
        &self.default_scope
    }

    /// The wrapped Azure SDK credential, for callers that also need to use it
    /// directly against an Azure SDK client.
    pub fn inner(&self) -> &Arc<dyn AzureTokenCredential> {
        &self.inner
    }
}

#[async_trait]
impl TokenCredential for SdkTokenCredential {
    async fn get_token(&self) -> Result<String> {
        self.get_token_for_scope(&self.default_scope).await
    }

    async fn get_token_for_scope(&self, scope: &str) -> Result<String> {
        let token = self.inner.get_token(&[scope], None).await.map_err(|e| {
            Error::service(format!(
                "Azure SDK credential failed for scope '{scope}': {e}"
            ))
        })?;
        Ok(token.token.secret().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use azure_core::credentials::AccessToken;
    use azure_core::time::OffsetDateTime;
    use std::sync::Mutex;

    /// Records the scopes it is asked for, so the tests can assert which scope
    /// actually reached the SDK rather than only what came back.
    #[derive(Debug, Default)]
    struct RecordingCredential {
        scopes: Mutex<Vec<String>>,
        fail: bool,
    }

    #[async_trait]
    impl AzureTokenCredential for RecordingCredential {
        async fn get_token(
            &self,
            scopes: &[&str],
            _options: Option<azure_core::credentials::TokenRequestOptions<'_>>,
        ) -> azure_core::Result<AccessToken> {
            self.scopes.lock().unwrap().push(scopes.join(","));
            if self.fail {
                return Err(azure_core::Error::with_message(
                    azure_core::error::ErrorKind::Credential,
                    "no credential available",
                ));
            }
            Ok(AccessToken::new(
                "token-value",
                OffsetDateTime::now_utc() + azure_core::time::Duration::hours(1),
            ))
        }
    }

    #[tokio::test]
    async fn get_token_uses_the_default_scope() {
        let recorder = Arc::new(RecordingCredential::default());
        let cred = SdkTokenCredential::azure_openai(recorder.clone());

        assert_eq!(cred.get_token().await.unwrap(), "token-value");
        assert_eq!(
            recorder.scopes.lock().unwrap().as_slice(),
            [AZURE_OPENAI_SCOPE]
        );
    }

    /// The whole reason the trait carries a scope override: a single credential
    /// serving two audiences must not silently pin every call to its default.
    #[tokio::test]
    async fn get_token_for_scope_overrides_the_default() {
        let recorder = Arc::new(RecordingCredential::default());
        let cred = SdkTokenCredential::azure_openai(recorder.clone());

        cred.get_token_for_scope(FOUNDRY_SCOPE).await.unwrap();

        assert_eq!(recorder.scopes.lock().unwrap().as_slice(), [FOUNDRY_SCOPE]);
    }

    /// An SDK credential failure must arrive as this crate's error type, and
    /// name the scope — a chained credential failing for one audience while
    /// working for another is otherwise very hard to read from a log.
    #[tokio::test]
    async fn credential_failure_maps_to_a_service_error_naming_the_scope() {
        let cred = SdkTokenCredential::foundry(Arc::new(RecordingCredential {
            fail: true,
            ..Default::default()
        }));

        let err = cred.get_token().await.unwrap_err();
        let rendered = err.to_string();
        assert!(
            rendered.contains(FOUNDRY_SCOPE),
            "error should name the scope, got: {rendered}"
        );
    }

    /// The adapter's reason to exist: real `azure_identity` types must satisfy
    /// the trait object it takes, and it must itself be usable as this crate's
    /// `TokenCredential` trait object wherever a credential is accepted.
    #[test]
    fn real_sdk_credentials_fit_the_trait() {
        let secret = azure_identity::ClientSecretCredential::new(
            "cc7d0b33-84c6-4bd1-bbfc-1b5b1cd8ca3a",
            "client-id".into(),
            "client-secret".into(),
            None,
        )
        .expect("client-secret credential constructs");

        let boxed: Box<dyn TokenCredential> = Box::new(SdkTokenCredential::azure_openai(secret));
        let _ = boxed;
    }
}
