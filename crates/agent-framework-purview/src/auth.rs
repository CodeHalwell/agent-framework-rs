//! Authentication: a self-contained, bring-your-own bearer token trait.
//!
//! # Auth burden
//!
//! The Python reference (`PurviewClient`) accepts an `azure-identity`
//! `TokenCredential` or `AsyncTokenCredential` directly, so it inherits
//! whatever credential chain the caller already has configured
//! (`DefaultAzureCredential`, `InteractiveBrowserCredential`, a managed
//! identity, ...). This crate has no `azure_identity`-equivalent dependency
//! to accept, and per this work package's brief is deliberately
//! **self-contained**: it defines its own minimal [`TokenProvider`] trait
//! rather than depending on `agent-framework-azure`'s `TokenCredential` (a
//! near-identical trait one layer up), so this crate has no dependency on
//! any other provider crate in this workspace.
//!
//! Bring a Microsoft Graph bearer token with the
//! `https://graph.microsoft.com/.default` scope (or the equivalent for a
//! custom [`PurviewSettings::graph_base_uri`](crate::settings::PurviewSettings::graph_base_uri) —
//! see [`PurviewSettings::get_scopes`](crate::settings::PurviewSettings::get_scopes)),
//! carrying the `dataSecurityAndGovernance` Graph permission described in
//! the Python package's README. [`StaticTokenProvider`] is provided for a
//! fixed/pre-fetched token (tests, short-lived scripts, or externally-managed
//! refresh).

use async_trait::async_trait;

use agent_framework_core::error::Result;

/// Supplies bearer tokens for Microsoft Graph (Purview) requests. See the
/// module docs for why this trait exists instead of reusing another crate's.
#[async_trait]
pub trait TokenProvider: Send + Sync {
    /// Fetch a bearer token to send as `Authorization: Bearer <token>`.
    async fn get_token(&self) -> Result<String>;
}

/// A [`TokenProvider`] that always returns the same, pre-fetched token.
#[derive(Debug, Clone)]
pub struct StaticTokenProvider {
    token: String,
}

impl StaticTokenProvider {
    /// Wrap a fixed bearer token.
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

#[async_trait]
impl TokenProvider for StaticTokenProvider {
    async fn get_token(&self) -> Result<String> {
        Ok(self.token.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn static_token_provider_returns_configured_token() {
        let provider = StaticTokenProvider::new("my-token");
        assert_eq!(provider.get_token().await.unwrap(), "my-token");
    }
}
