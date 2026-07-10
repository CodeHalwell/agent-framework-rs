//! Token credential abstraction for Microsoft Entra ID (Azure AD) bearer-token
//! authentication.

use agent_framework_core::error::Result;
use async_trait::async_trait;

/// Supplies bearer tokens for Microsoft Entra ID authentication.
///
/// Implement this to integrate a real credential chain (e.g. wrapping
/// `azure_identity`'s `TokenCredential`); [`StaticTokenCredential`] is
/// provided for a fixed/pre-fetched token (useful in tests, short-lived
/// scripts, or when the caller manages token refresh externally).
#[async_trait]
pub trait TokenCredential: Send + Sync {
    /// Fetch a bearer token to send as `Authorization: Bearer <token>`.
    async fn get_token(&self) -> Result<String>;
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
