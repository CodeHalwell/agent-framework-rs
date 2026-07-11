//! Authentication: bring-your-own bearer token.
//!
//! # Auth burden
//!
//! The Python reference (`agent_framework_copilotstudio._acquire_token`)
//! acquires a Power Platform API token itself, via MSAL
//! (`msal.PublicClientApplication`): it tries a silent, cached-account token
//! first, and falls back to an *interactive* (browser-popup) login if that
//! fails. That flow has no meaningful equivalent in a headless Rust library —
//! there is no MSAL-for-Rust in this workspace, and an interactive browser
//! login is not something a library crate should perform on a caller's
//! behalf.
//!
//! This port therefore pushes the entire auth burden onto the caller: bring
//! a valid Power Platform API bearer token (scope
//! `https://api.powerplatform.com/.default`, or the equivalent for your
//! [`PowerPlatformCloud`](crate::settings::PowerPlatformCloud)) via a
//! [`TokenProvider`] implementation. In practice, callers will typically
//! wrap an MSAL confidential/managed-identity flow performed elsewhere (e.g.
//! `azure_identity`, a sidecar token service, or a cached token refreshed out
//! of band) — [`StaticTokenProvider`] is provided for a fixed/pre-fetched
//! token (tests, short-lived scripts, or externally-managed refresh).

use async_trait::async_trait;

use agent_framework_core::error::Result;

/// Supplies bearer tokens for Copilot Studio's Direct-to-Engine API.
///
/// See the module docs for why this port does not acquire tokens itself the
/// way the Python reference's MSAL-based `acquire_token` does.
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
