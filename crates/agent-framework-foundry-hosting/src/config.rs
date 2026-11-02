//! The hosted-agent container's configuration (the agentserver's
//! `AgentConfig`), resolved from the platform's environment variables.

use agent_framework_core::error::{Error, Result};

/// The default container port (`PORT` unset).
pub const DEFAULT_PORT: u16 = 8088;
/// The default SSE keep-alive interval, in seconds.
pub const DEFAULT_SSE_KEEPALIVE_SECONDS: u64 = 15;

/// Environment variable names the Foundry platform sets.
pub mod env {
    /// The agent's name.
    pub const FOUNDRY_AGENT_NAME: &str = "FOUNDRY_AGENT_NAME";
    /// The agent's stable GUID.
    pub const FOUNDRY_AGENT_ID: &str = "FOUNDRY_AGENT_ID";
    /// The agent's version.
    pub const FOUNDRY_AGENT_VERSION: &str = "FOUNDRY_AGENT_VERSION";
    /// Non-empty when running in a Foundry-hosted container.
    pub const FOUNDRY_HOSTING_ENVIRONMENT: &str = "FOUNDRY_HOSTING_ENVIRONMENT";
    /// The Foundry project endpoint.
    pub const FOUNDRY_PROJECT_ENDPOINT: &str = "FOUNDRY_PROJECT_ENDPOINT";
    /// The project's ARM resource id.
    pub const FOUNDRY_PROJECT_ARM_ID: &str = "FOUNDRY_PROJECT_ARM_ID";
    /// The platform session id (the sandbox's session).
    pub const FOUNDRY_AGENT_SESSION_ID: &str = "FOUNDRY_AGENT_SESSION_ID";
    /// The session incarnation GUID (32 lowercase hex characters).
    pub const FOUNDRY_AGENT_SESSION_GUID: &str = "FOUNDRY_AGENT_SESSION_GUID";
    /// The port to listen on.
    pub const PORT: &str = "PORT";
    /// Application Insights connection string.
    pub const APPLICATIONINSIGHTS_CONNECTION_STRING: &str = "APPLICATIONINSIGHTS_CONNECTION_STRING";
    /// OTLP exporter endpoint.
    pub const OTEL_EXPORTER_OTLP_ENDPOINT: &str = "OTEL_EXPORTER_OTLP_ENDPOINT";
    /// SSE keep-alive interval in seconds (`0` disables it).
    pub const SSE_KEEPALIVE_INTERVAL: &str = "SSE_KEEPALIVE_INTERVAL";
}

/// Resolved configuration of a hosted-agent container (upstream
/// `AgentConfig`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FoundryHostConfig {
    /// `FOUNDRY_AGENT_NAME`.
    pub agent_name: String,
    /// `FOUNDRY_AGENT_VERSION`.
    pub agent_version: String,
    /// `name:version`, `name`, or empty.
    pub agent_id: String,
    /// `FOUNDRY_AGENT_ID`.
    pub agent_guid: String,
    /// Whether `FOUNDRY_HOSTING_ENVIRONMENT` is set (non-empty).
    pub is_hosted: bool,
    /// `FOUNDRY_PROJECT_ENDPOINT`.
    pub project_endpoint: String,
    /// `FOUNDRY_PROJECT_ARM_ID`.
    pub project_id: String,
    /// `FOUNDRY_AGENT_SESSION_ID`.
    pub session_id: String,
    /// `FOUNDRY_AGENT_SESSION_GUID`.
    pub session_guid: String,
    /// `PORT` (default 8088).
    pub port: u16,
    /// `APPLICATIONINSIGHTS_CONNECTION_STRING`.
    pub appinsights_connection_string: String,
    /// `OTEL_EXPORTER_OTLP_ENDPOINT`.
    pub otlp_endpoint: String,
    /// `SSE_KEEPALIVE_INTERVAL` seconds (default 15, `0` disables).
    pub sse_keepalive_interval: u64,
}

impl FoundryHostConfig {
    /// A local (not hosted) configuration with defaults.
    pub fn local() -> Self {
        Self {
            port: DEFAULT_PORT,
            sse_keepalive_interval: DEFAULT_SSE_KEEPALIVE_SECONDS,
            ..Default::default()
        }
    }

    /// Resolve from the process environment.
    pub fn from_env() -> Result<Self> {
        Self::from_lookup(|name| std::env::var(name).ok())
    }

    /// Resolve from `lookup` (a variable name to its value). Invalid numbers
    /// and, when hosted, a malformed session GUID are errors, surfacing
    /// misconfiguration at startup as upstream does.
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Result<Self> {
        let get = |name: &str| lookup(name).unwrap_or_default();
        let agent_name = get(env::FOUNDRY_AGENT_NAME);
        let agent_version = get(env::FOUNDRY_AGENT_VERSION);
        let agent_id = match (agent_name.is_empty(), agent_version.is_empty()) {
            (false, false) => format!("{agent_name}:{agent_version}"),
            (false, true) => agent_name.clone(),
            _ => String::new(),
        };
        let is_hosted = !get(env::FOUNDRY_HOSTING_ENVIRONMENT).is_empty();
        let session_guid = get(env::FOUNDRY_AGENT_SESSION_GUID);
        if is_hosted
            && !session_guid.is_empty()
            && !(session_guid.len() == 32
                && session_guid
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
        {
            return Err(Error::Configuration(
                "FOUNDRY_AGENT_SESSION_GUID must be a 32-character lowercase hexadecimal GUID"
                    .into(),
            ));
        }
        let int = |name: &str, default: u64| -> Result<u64> {
            match lookup(name) {
                None => Ok(default),
                Some(raw) => raw.trim().parse::<u64>().map_err(|_| {
                    Error::Configuration(format!(
                        "Invalid value for {name}: {raw:?} (expected a non-negative integer)"
                    ))
                }),
            }
        };
        let port = int(env::PORT, u64::from(DEFAULT_PORT))?;
        let port = u16::try_from(port)
            .ok()
            .filter(|p| *p > 0)
            .ok_or_else(|| Error::Configuration(format!("Invalid value for PORT: {port}")))?;
        Ok(Self {
            agent_name,
            agent_version,
            agent_id,
            agent_guid: get(env::FOUNDRY_AGENT_ID),
            is_hosted,
            project_endpoint: get(env::FOUNDRY_PROJECT_ENDPOINT),
            project_id: get(env::FOUNDRY_PROJECT_ARM_ID),
            session_id: get(env::FOUNDRY_AGENT_SESSION_ID),
            session_guid,
            port,
            appinsights_connection_string: get(env::APPLICATIONINSIGHTS_CONNECTION_STRING),
            otlp_endpoint: get(env::OTEL_EXPORTER_OTLP_ENDPOINT),
            sse_keepalive_interval: int(
                env::SSE_KEEPALIVE_INTERVAL,
                DEFAULT_SSE_KEEPALIVE_SECONDS,
            )?,
        })
    }
}
