//! [`PurviewClient`]: calls the Microsoft Graph `processContent` endpoint.
//!
//! # Scope
//!
//! Python's `PurviewClient` exposes three Graph calls
//! (`get_protection_scopes`, `process_content`, `send_content_activities`),
//! orchestrated by a separate `ScopedContentProcessor` that: computes
//! applicable protection scopes first (cached, with ETag-based
//! invalidation), only calls `process_content` when a scope actually applies
//! (inline) or queues it in the background (offline execution mode), and
//! logs a `contentActivities` entry in the background when no scope applies
//! at all. This work package's brief scopes this crate down to just
//! `processContent` — "POST the processContent route Python uses ... parse
//! verdict" — so **only that one endpoint is called here**: no protection-
//! scopes precheck, no response caching, no background content-activity
//! logging. This is a deliberate, brief-directed scope cut, not an
//! oversight; see [`crate::processor`] for where the request-building logic
//! that *is* ported lives, and the crate docs for the full list of what's
//! out of scope.

use std::sync::Arc;

use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, USER_AGENT};

use agent_framework_core::error::{Error, Result};

use crate::auth::TokenProvider;
use crate::models::{ProcessContentRequest, ProcessContentResponse};
use crate::settings::PurviewSettings;

const PURVIEW_USER_AGENT: &str = concat!("agent-framework-rs-purview/", env!("CARGO_PKG_VERSION"));

/// Calls the Microsoft Graph `dataSecurityAndGovernance/processContent`
/// endpoint. See the module docs for how this differs in scope from
/// Python's `PurviewClient`.
pub struct PurviewClient {
    http: reqwest::Client,
    token_provider: Arc<dyn TokenProvider>,
    /// `graph_base_uri`, trailing slash trimmed (mirrors Python's
    /// `self._graph_uri = settings.graph_base_uri.rstrip("/")`).
    graph_base_uri: String,
    /// Captured from [`PurviewSettings::ignore_payment_required`] at
    /// construction time: on a 402, this client itself returns an empty
    /// [`ProcessContentResponse`] instead of an error when `true` —
    /// mirrors Python's `PurviewClient._post`, which performs this same
    /// check (not the middleware) before ever raising
    /// `PurviewPaymentRequiredError`.
    ignore_payment_required: bool,
}

impl PurviewClient {
    pub fn new(token_provider: impl TokenProvider + 'static, settings: &PurviewSettings) -> Self {
        Self {
            http: reqwest::Client::new(),
            token_provider: Arc::new(token_provider),
            graph_base_uri: settings.graph_base_uri.trim_end_matches('/').to_string(),
            ignore_payment_required: settings.ignore_payment_required,
        }
    }

    /// `POST {graph_base_uri}/users/{userId}/dataSecurityAndGovernance/processContent`.
    ///
    /// On a non-2xx status: 402 is swallowed into an empty
    /// [`ProcessContentResponse`] when `ignore_payment_required` was set at
    /// construction time (see the field docs); every other non-2xx status
    /// (401/403/402-when-not-ignored/429/anything else) becomes an
    /// [`Error::ServiceStatus`] carrying the real status code, so callers
    /// (namely [`crate::middleware`]) can distinguish "payment required" via
    /// [`Error::status`] the same way Python's `except
    /// PurviewPaymentRequiredError` clause does.
    pub async fn process_content(
        &self,
        request: &ProcessContentRequest,
    ) -> Result<ProcessContentResponse> {
        let token = self.token_provider.get_token().await?;
        let url = format!(
            "{}/users/{}/dataSecurityAndGovernance/processContent",
            self.graph_base_uri, request.user_id
        );
        let resp = self
            .http
            .post(&url)
            .header(AUTHORIZATION, format!("Bearer {token}"))
            .header(CONTENT_TYPE, "application/json")
            .header(USER_AGENT, PURVIEW_USER_AGENT)
            .json(request)
            .send()
            .await
            .map_err(|e| Error::service(format!("Purview request to {url} failed: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| Error::service(format!("failed reading Purview response body: {e}")))?;

        if status.as_u16() == 402 && self.ignore_payment_required {
            return Ok(ProcessContentResponse::default());
        }
        if !status.is_success() {
            return Err(Error::service_status(status.as_u16(), text, None));
        }

        serde_json::from_str(&text).map_err(|e| {
            Error::service(format!(
                "invalid Purview processContent response JSON: {e} (body: {text})"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::StaticTokenProvider;

    #[test]
    fn graph_base_uri_is_trimmed_of_trailing_slash() {
        let settings = PurviewSettings::new("Test App");
        let client = PurviewClient::new(StaticTokenProvider::new("t"), &settings);
        assert_eq!(client.graph_base_uri, "https://graph.microsoft.com/v1.0");
    }

    #[test]
    fn ignore_payment_required_is_captured_from_settings() {
        let settings = PurviewSettings::new("Test App").with_ignore_payment_required(true);
        let client = PurviewClient::new(StaticTokenProvider::new("t"), &settings);
        assert!(client.ignore_payment_required);

        let settings2 = PurviewSettings::new("Test App");
        let client2 = PurviewClient::new(StaticTokenProvider::new("t"), &settings2);
        assert!(!client2.ignore_payment_required);
    }
}
