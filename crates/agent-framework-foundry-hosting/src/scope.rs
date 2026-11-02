//! Per-request platform identity: the request context the platform sends as
//! headers, and the trusted scope hosted state is keyed by.

use agent_framework_core::error::{Error, Result};
use axum::http::HeaderMap;
use sha2::{Digest, Sha256};

use crate::config::FoundryHostConfig;

/// Platform header names.
pub mod headers {
    /// Request correlation id (echoed on every response).
    pub const REQUEST_ID: &str = "x-request-id";
    /// The server SDK stack, set on every response.
    pub const SERVER_VERSION: &str = "x-platform-server";
    /// The resolved agent session id.
    pub const SESSION_ID: &str = "x-agent-session-id";
    /// The platform-injected user id (trusted only when hosted).
    pub const USER_ID: &str = "x-agent-user-id";
    /// The platform-minted opaque call id, forwarded on outbound 1P calls.
    pub const FOUNDRY_CALL_ID: &str = "x-agent-foundry-call-id";
    /// The invocation id (Invocations protocol).
    pub const INVOCATION_ID: &str = "x-agent-invocation-id";
    /// Classifies an error response (`user`, `platform`, `upstream`).
    pub const ERROR_SOURCE: &str = "x-platform-error-source";
}

/// What the platform said about one request (upstream
/// `FoundryAgentRequestContext`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FoundryRequestContext {
    /// The request id (`x-request-id`), generated when absent.
    pub request_id: String,
    /// The caller-visible session id, if any.
    pub session_id: Option<String>,
    /// `x-agent-user-id`.
    pub user_id: Option<String>,
    /// `x-agent-foundry-call-id`.
    pub call_id: Option<String>,
}

fn header(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(str::to_string)
}

impl FoundryRequestContext {
    /// Read the context from request headers; `session_id` is the caller's
    /// (a query parameter or body field) when the protocol has one.
    pub fn from_headers(headers: &HeaderMap, session_id: Option<String>) -> Self {
        Self {
            request_id: header(headers, headers::REQUEST_ID)
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            session_id: session_id
                .filter(|s| !s.is_empty())
                .or_else(|| header(headers, headers::SESSION_ID)),
            user_id: header(headers, headers::USER_ID),
            call_id: header(headers, headers::FOUNDRY_CALL_ID),
        }
    }
}

/// The trusted identity hosted state is partitioned by: the platform
/// sandbox session and user, separate from caller-facing continuation ids
/// (upstream `FoundryRequestScope`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundryRequestScope {
    /// The sandbox session id.
    pub session_id: String,
    /// The platform user id.
    pub user_id: Option<String>,
    /// The platform call id.
    pub call_id: Option<String>,
    /// Whether the host runs in Foundry.
    pub is_hosted: bool,
}

impl FoundryRequestScope {
    /// Resolve the scope. Hosted, the session comes from the platform's
    /// `FOUNDRY_AGENT_SESSION_ID` (a caller-supplied one must match it) and
    /// a user id and call id are required; locally, the request's session id
    /// or `local_session_id` is used.
    pub fn from_context(
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
        local_session_id: Option<&str>,
    ) -> Result<Self> {
        let session_id = if config.is_hosted {
            if config.session_id.trim().is_empty() {
                return Err(Error::Configuration(
                    "Foundry hosted requests require a platform FOUNDRY_AGENT_SESSION_ID.".into(),
                ));
            }
            if context
                .session_id
                .as_deref()
                .is_some_and(|s| s != config.session_id)
            {
                return Err(Error::Configuration(
                    "The request agent_session_id does not match the platform session ID.".into(),
                ));
            }
            if context.user_id.is_none() || context.call_id.is_none() {
                return Err(Error::Configuration(
                    "Foundry hosted requests require a trusted user ID and call ID.".into(),
                ));
            }
            config.session_id.clone()
        } else {
            context
                .session_id
                .clone()
                .or_else(|| local_session_id.map(str::to_string))
                .ok_or_else(|| {
                    Error::Configuration(
                        "A Foundry agent session ID is required to handle the request.".into(),
                    )
                })?
        };
        Ok(Self {
            session_id,
            user_id: context.user_id.clone(),
            call_id: context.call_id.clone(),
            is_hosted: config.is_hosted,
        })
    }

    /// The SHA-256 (hex) of the framed `[user_id, session_id]` identity, so
    /// store names reveal neither id.
    pub fn storage_key(&self) -> String {
        let framed = serde_json::to_string(&serde_json::json!([self.user_id, self.session_id]))
            .unwrap_or_default();
        hex(&Sha256::digest(framed.as_bytes()))
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The state-store name for `root` under `scope` (upstream `_store_name`):
/// `root[/context]` locally, `root/v2/<storage key>[/<hashed context>]`
/// hosted.
pub fn store_name(
    root: &str,
    scope: Option<&FoundryRequestScope>,
    context_id: Option<&str>,
) -> String {
    match scope {
        None => match context_id {
            Some(id) => format!("{root}/{id}"),
            None => root.to_string(),
        },
        Some(scope) => {
            let mut name = format!("{root}/v2/{}", scope.storage_key());
            if let Some(id) = context_id {
                name.push('/');
                name.push_str(&hex(&Sha256::digest(id.as_bytes()))[..24]);
            }
            name
        }
    }
}

/// Ids in the agentserver's `{prefix}_{partition key}{entropy}` format: an
/// 18-character partition key (16 hex digits and `00`) followed by 32
/// alphanumeric characters (upstream `IdGenerator`).
pub mod ids {
    fn entropy() -> String {
        format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        )[..32]
            .to_string()
    }

    /// The partition key embedded in `id`, if it has one (new 50-character
    /// bodies carry it first; legacy 48-character bodies carry 16 characters
    /// last, which are suffixed with `00`).
    pub fn partition_key(id: &str) -> Option<String> {
        let (_, body) = id.split_once('_')?;
        match body.len() {
            50 => Some(body[..18].to_string()),
            48 => Some(format!("{}00", &body[32..])),
            _ => None,
        }
    }

    /// A new id with `prefix`, co-located with `hint`'s partition when it
    /// has one.
    pub fn new_id(prefix: &str, hint: Option<&str>) -> String {
        let key = hint
            .and_then(partition_key)
            .unwrap_or_else(|| format!("{}00", &uuid::Uuid::new_v4().simple().to_string()[..16]));
        format!("{prefix}_{key}{}", entropy())
    }

    /// A response id (`caresp_…`).
    pub fn response_id(hint: Option<&str>) -> String {
        new_id("caresp", hint)
    }

    /// A message item id (`msg_…`).
    pub fn message_id(hint: Option<&str>) -> String {
        new_id("msg", hint)
    }

    /// A function-call item id (`fc_…`).
    pub fn function_call_id(hint: Option<&str>) -> String {
        new_id("fc", hint)
    }
}
