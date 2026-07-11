//! Token credential abstraction for Microsoft Entra ID (Azure AD) bearer-token
//! authentication.

use agent_framework_core::error::Result;
use async_trait::async_trait;

/// Supplies bearer tokens for Microsoft Entra ID authentication.
///
/// Implement this to integrate a real credential chain (e.g. the
/// [`AzureCliCredential`](crate::AzureCliCredential),
/// [`ClientSecretCredential`](crate::ClientSecretCredential),
/// [`ManagedIdentityCredential`](crate::ManagedIdentityCredential), and
/// [`ChainedTokenCredential`](crate::ChainedTokenCredential) that ship with
/// this crate); [`StaticTokenCredential`] is provided for a fixed/pre-fetched
/// token (useful in tests, short-lived scripts, or when the caller manages
/// token refresh externally).
///
/// Most credentials are bound to a single configured *scope* (audience) and
/// [`get_token`](Self::get_token) fetches a token for it. A caller that needs a
/// token for a *different* scope from the same credential uses
/// [`get_token_for_scope`](Self::get_token_for_scope); the default
/// implementation ignores the scope and delegates to
/// [`get_token`](Self::get_token), which is correct for fixed-token credentials
/// but is overridden by the real credentials so each scope is fetched (and
/// cached) independently.
#[async_trait]
pub trait TokenCredential: Send + Sync {
    /// Fetch a bearer token to send as `Authorization: Bearer <token>`.
    async fn get_token(&self) -> Result<String>;

    /// Fetch a bearer token for a specific `scope` (audience), e.g.
    /// `"https://ai.azure.com/.default"`.
    ///
    /// The default implementation ignores `scope` and delegates to
    /// [`get_token`](Self::get_token) — appropriate for credentials that wrap a
    /// single fixed token. Credentials that mint tokens per audience override
    /// this to honor the requested scope.
    async fn get_token_for_scope(&self, _scope: &str) -> Result<String> {
        self.get_token().await
    }
}

/// A [`TokenCredential`] that always returns the same, pre-fetched token.
#[derive(Debug, Clone)]
pub struct StaticTokenCredential {
    token: String,
}

impl StaticTokenCredential {
    /// Wrap a fixed bearer token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl TokenCredential for StaticTokenCredential {
    async fn get_token(&self) -> Result<String> {
        Ok(self.token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_token_credential_returns_configured_token() {
        let cred = StaticTokenCredential::new("my-token");
        assert_eq!(cred.get_token().await.unwrap(), "my-token");
    }
}
