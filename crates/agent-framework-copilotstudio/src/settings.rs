//! Environment settings, Power Platform cloud/agent-type configuration, and
//! Direct-to-Engine (D2E) conversation URL construction.
//!
//! # Fidelity
//!
//! Python's `agent_framework_copilotstudio` package does none of this itself
//! — it constructs a `microsoft_agents.copilotstudio.client.ConnectionSettings`
//! and hands it to `CopilotClient` (from the separate
//! `microsoft-agents-copilotstudio-client` PyPI package), which does the URL
//! construction internally. That package is not part of this repository, so
//! it is normally invisible to a from-scratch port. This module was written
//! against that SDK's actual source (`microsoft_agents/copilotstudio/client/{connection_settings,power_platform_environment,power_platform_cloud,agent_type}.py`,
//! version 1.1.0) rather than a guessed convention, so URL construction below
//! is a faithful, line-by-line port of
//! `PowerPlatformEnvironment.get_copilot_studio_connection_url` /
//! `.get_environment_endpoint` / `.get_endpoint_suffix` — **high fidelity**,
//! not "the convention it implies". See the crate docs for how that source
//! was obtained (it was not fetched over the network by this port).

use agent_framework_core::error::{Error, Result};

/// `COPILOTSTUDIOAGENT__*` environment settings, mirroring Python's
/// `CopilotStudioSettings` (an `AFBaseSettings` with `env_prefix =
/// "COPILOTSTUDIOAGENT__"`).
///
/// Only the four fields Python's settings type actually declares are here;
/// `cloud` / `agent_type` are configured on [`CopilotStudioConnectionSettings`]
/// instead (builder-set, not environment-driven — matching Python, where
/// those live on the separate, non-env-prefixed `ConnectionSettings` type).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CopilotStudioSettings {
    /// `COPILOTSTUDIOAGENT__ENVIRONMENTID` — the Power Platform environment
    /// hosting the Copilot Studio agent.
    pub environment_id: Option<String>,
    /// `COPILOTSTUDIOAGENT__SCHEMANAME` — the agent identifier / schema name.
    pub schema_name: Option<String>,
    /// `COPILOTSTUDIOAGENT__AGENTAPPID` — the App Registration client id used
    /// for authentication (informational here; this port's [`TokenProvider`](crate::auth::TokenProvider)
    /// brings its own token rather than acquiring one from this id).
    pub agent_app_id: Option<String>,
    /// `COPILOTSTUDIOAGENT__TENANTID` — the App Registration's tenant id.
    pub tenant_id: Option<String>,
}

impl CopilotStudioSettings {
    /// The environment-variable prefix Python's settings type uses.
    pub const ENV_PREFIX: &'static str = "COPILOTSTUDIOAGENT__";

    /// Read settings from `COPILOTSTUDIOAGENT__ENVIRONMENTID` /
    /// `_SCHEMANAME` / `_AGENTAPPID` / `_TENANTID`. Absent or empty variables
    /// map to `None` (mirrors Python's optional-field settings; nothing is
    /// validated as required here — see [`CopilotStudioConnectionSettings::new`]
    /// for where a missing environment id / schema name actually surfaces as
    /// an error, exactly like the Python constructor).
    pub fn from_env() -> Self {
        Self {
            environment_id: env_nonempty("COPILOTSTUDIOAGENT__ENVIRONMENTID"),
            schema_name: env_nonempty("COPILOTSTUDIOAGENT__SCHEMANAME"),
            agent_app_id: env_nonempty("COPILOTSTUDIOAGENT__AGENTAPPID"),
            tenant_id: env_nonempty("COPILOTSTUDIOAGENT__TENANTID"),
        }
    }
}

fn env_nonempty(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|v| !v.is_empty())
}

/// The Power Platform cloud hosting the environment. Mirrors
/// `microsoft_agents.copilotstudio.client.PowerPlatformCloud`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PowerPlatformCloud {
    Unknown,
    Exp,
    Dev,
    Test,
    Preprod,
    FirstRelease,
    #[default]
    Prod,
    Gov,
    High,
    #[allow(clippy::upper_case_acronyms)]
    DoD,
    Mooncake,
    Ex,
    Rx,
    Prv,
    Local,
    GovFr,
    /// A custom cloud; requires
    /// [`CopilotStudioConnectionSettings::custom_power_platform_cloud`] to be
    /// set to the base hostname.
    Other,
}

impl PowerPlatformCloud {
    /// The `id_suffix_length` used to split the (dash-stripped, lowercased)
    /// environment id into `{prefix}.{suffix}`. Mirrors
    /// `PowerPlatformEnvironment.get_id_suffix_length`: 2 for
    /// Prod/FirstRelease, 1 for every other cloud.
    fn id_suffix_length(self) -> usize {
        match self {
            PowerPlatformCloud::FirstRelease | PowerPlatformCloud::Prod => 2,
            _ => 1,
        }
    }

    /// The API host suffix for this cloud. Mirrors
    /// `PowerPlatformEnvironment.get_endpoint_suffix`.
    fn endpoint_suffix(self, custom_power_platform_cloud: Option<&str>) -> Result<String> {
        let suffix = match self {
            PowerPlatformCloud::Local => "api.powerplatform.localhost",
            PowerPlatformCloud::Exp => "api.exp.powerplatform.com",
            PowerPlatformCloud::Dev => "api.dev.powerplatform.com",
            PowerPlatformCloud::Prv => "api.prv.powerplatform.com",
            PowerPlatformCloud::Test => "api.test.powerplatform.com",
            PowerPlatformCloud::Preprod => "api.preprod.powerplatform.com",
            PowerPlatformCloud::FirstRelease | PowerPlatformCloud::Prod => "api.powerplatform.com",
            PowerPlatformCloud::Gov | PowerPlatformCloud::GovFr => {
                "api.gov.powerplatform.microsoft.us"
            }
            PowerPlatformCloud::High => "api.high.powerplatform.microsoft.us",
            PowerPlatformCloud::DoD => "api.appsplatform.us",
            PowerPlatformCloud::Mooncake => "api.powerplatform.partner.microsoftonline.cn",
            PowerPlatformCloud::Ex => "api.powerplatform.eaglex.ic.gov",
            PowerPlatformCloud::Rx => "api.powerplatform.microsoft.scloud",
            PowerPlatformCloud::Other => {
                return custom_power_platform_cloud
                    .map(str::to_string)
                    .ok_or_else(|| {
                        Error::Configuration(
                            "PowerPlatformCloud::Other requires custom_power_platform_cloud \
                             to be set"
                                .into(),
                        )
                    });
            }
            PowerPlatformCloud::Unknown => {
                return Err(Error::Configuration(
                    "PowerPlatformCloud::Unknown cannot be resolved to a host; set an explicit \
                     cloud"
                        .into(),
                ));
            }
        };
        Ok(suffix.to_string())
    }
}

/// The kind of Copilot Studio agent being addressed. Mirrors
/// `microsoft_agents.copilotstudio.client.AgentType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AgentType {
    /// A published, Dataverse-backed Copilot Studio agent (the common case).
    #[default]
    Published,
    /// A prebuilt agent.
    Prebuilt,
}

impl AgentType {
    fn path_segment(self) -> &'static str {
        match self {
            AgentType::Published => "dataverse-backed",
            AgentType::Prebuilt => "prebuilt",
        }
    }
}

/// The Direct-to-Engine API version pinned by the reference SDK
/// (`PowerPlatformEnvironment.API_VERSION`).
pub const API_VERSION: &str = "2022-03-01-preview";

/// Connection parameters for one Copilot Studio agent: which environment/
/// agent to talk to, which Power Platform cloud it's hosted in, and how to
/// build the Direct-to-Engine conversation URL. Mirrors
/// `microsoft_agents.copilotstudio.client.ConnectionSettings`.
#[derive(Debug, Clone)]
pub struct CopilotStudioConnectionSettings {
    pub environment_id: String,
    pub agent_identifier: String,
    pub cloud: PowerPlatformCloud,
    pub agent_type: AgentType,
    pub custom_power_platform_cloud: Option<String>,
    /// A full base URL that bypasses environment-id/cloud-based host
    /// construction entirely (mirrors Python's `direct_connect_url`) — a
    /// host override for e.g. a local test double or a pre-resolved
    /// "island" endpoint.
    pub direct_connect_url: Option<String>,
}

impl CopilotStudioConnectionSettings {
    /// `cloud` defaults to [`PowerPlatformCloud::Prod`] and `agent_type` to
    /// [`AgentType::Published`], matching Python's `ConnectionSettings.__init__`
    /// defaults (`cloud or PowerPlatformCloud.PROD`, `copilot_agent_type or
    /// AgentType.PUBLISHED`).
    pub fn new(environment_id: impl Into<String>, agent_identifier: impl Into<String>) -> Self {
        Self {
            environment_id: environment_id.into(),
            agent_identifier: agent_identifier.into(),
            cloud: PowerPlatformCloud::default(),
            agent_type: AgentType::default(),
            custom_power_platform_cloud: None,
            direct_connect_url: None,
        }
    }

    /// Build connection settings from [`CopilotStudioSettings`] (as read via
    /// [`CopilotStudioSettings::from_env`] or constructed directly). Fails
    /// exactly where Python's `CopilotStudioAgent.__init__` does: a missing
    /// `environment_id` or `schema_name`.
    pub fn from_settings(settings: &CopilotStudioSettings) -> Result<Self> {
        let environment_id = settings.environment_id.clone().ok_or_else(|| {
            Error::Configuration(
                "Copilot Studio environment ID is required. Set via 'environment_id' or the \
                 'COPILOTSTUDIOAGENT__ENVIRONMENTID' environment variable."
                    .into(),
            )
        })?;
        let agent_identifier = settings.schema_name.clone().ok_or_else(|| {
            Error::Configuration(
                "Copilot Studio agent identifier/schema name is required. Set via \
                 'agent_identifier' or the 'COPILOTSTUDIOAGENT__SCHEMANAME' environment \
                 variable."
                    .into(),
            )
        })?;
        Ok(Self::new(environment_id, agent_identifier))
    }

    /// Set the Power Platform cloud (builder style).
    pub fn with_cloud(mut self, cloud: PowerPlatformCloud) -> Self {
        self.cloud = cloud;
        self
    }

    /// Set the agent type (builder style).
    pub fn with_agent_type(mut self, agent_type: AgentType) -> Self {
        self.agent_type = agent_type;
        self
    }

    /// Set the custom cloud base hostname, required when `cloud ==
    /// PowerPlatformCloud::Other` (builder style).
    pub fn with_custom_power_platform_cloud(mut self, host: impl Into<String>) -> Self {
        self.custom_power_platform_cloud = Some(host.into());
        self
    }

    /// Override the connection URL's base entirely (builder style). See
    /// [`Self::direct_connect_url`].
    pub fn with_direct_connect_url(mut self, url: impl Into<String>) -> Self {
        self.direct_connect_url = Some(url.into());
        self
    }

    /// The environment host: `{hex_prefix}.{hex_suffix}.environment.{suffix}`.
    /// Mirrors `PowerPlatformEnvironment.get_environment_endpoint`.
    fn environment_endpoint(&self) -> Result<String> {
        let suffix = self
            .cloud
            .endpoint_suffix(self.custom_power_platform_cloud.as_deref())?;
        let normalized: String = self
            .environment_id
            .chars()
            .filter(|c| *c != '-')
            .flat_map(char::to_lowercase)
            .collect();
        let id_suffix_len = self.cloud.id_suffix_length();
        if normalized.len() <= id_suffix_len {
            return Err(Error::Configuration(format!(
                "environment_id '{}' is too short to derive a Power Platform host",
                self.environment_id
            )));
        }
        let split_at = normalized.len() - id_suffix_len;
        let (prefix, id_suffix) = normalized.split_at(split_at);
        Ok(format!("{prefix}.{id_suffix}.environment.{suffix}"))
    }

    /// Build the Direct-to-Engine conversation URL: the "create conversation"
    /// endpoint when `conversation_id` is `None`, else the "execute turn"
    /// endpoint for that conversation. Mirrors
    /// `PowerPlatformEnvironment.get_copilot_studio_connection_url` (the
    /// `create_subscribe_link` / `cloud_base_address` parameters are not
    /// reproduced — `/subscribe` is out of scope for this port; see the
    /// crate docs).
    pub fn conversation_url(&self, conversation_id: Option<&str>) -> Result<String> {
        if let Some(direct) = &self.direct_connect_url {
            return Self::direct_conversation_url(direct, conversation_id);
        }
        if self.environment_id.is_empty() {
            return Err(Error::Configuration(
                "environment_id must be provided".into(),
            ));
        }
        if self.agent_identifier.is_empty() {
            return Err(Error::Configuration(
                "agent_identifier must be provided".into(),
            ));
        }
        let host = self.environment_endpoint()?;
        let path = match conversation_id {
            None => format!(
                "/copilotstudio/{}/authenticated/bots/{}/conversations",
                self.agent_type.path_segment(),
                self.agent_identifier
            ),
            Some(id) => format!(
                "/copilotstudio/{}/authenticated/bots/{}/conversations/{id}",
                self.agent_type.path_segment(),
                self.agent_identifier
            ),
        };
        Ok(format!("https://{host}{path}?api-version={API_VERSION}"))
    }

    /// Mirrors `PowerPlatformEnvironment._create_uri_direct`: strip trailing
    /// slashes, drop anything from an existing `/conversations` segment
    /// onward, then append the conversation path.
    fn direct_conversation_url(base: &str, conversation_id: Option<&str>) -> Result<String> {
        let (scheme, rest) = base.split_once("://").ok_or_else(|| {
            Error::Configuration(format!(
                "direct_connect_url '{base}' is not an absolute URL"
            ))
        })?;
        let (authority, raw_path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], &rest[idx..]),
            None => (rest, ""),
        };
        if authority.is_empty() {
            return Err(Error::Configuration(format!(
                "direct_connect_url '{base}' is not an absolute URL"
            )));
        }
        let mut path = raw_path.trim_end_matches(['/', '\\']).to_string();
        if let Some(idx) = path.find("/conversations") {
            path.truncate(idx);
        }
        let path = match conversation_id {
            None => format!("{path}/conversations"),
            Some(id) => format!("{path}/conversations/{id}"),
        };
        Ok(format!(
            "{scheme}://{authority}{path}?api-version={API_VERSION}"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- CopilotStudioSettings::from_env ---------------------------------

    /// Guards env var mutation: tests run on multiple threads within a crate,
    /// and env vars are process-global (same pattern as
    /// `agent-framework-mem0`'s `ENV_MUTEX`).
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn clear_env() {
        for key in [
            "COPILOTSTUDIOAGENT__ENVIRONMENTID",
            "COPILOTSTUDIOAGENT__SCHEMANAME",
            "COPILOTSTUDIOAGENT__AGENTAPPID",
            "COPILOTSTUDIOAGENT__TENANTID",
        ] {
            // SAFETY: serialized by ENV_MUTEX; no other test in this crate
            // touches these variables.
            unsafe { std::env::remove_var(key) };
        }
    }

    #[test]
    fn from_env_reads_all_four_prefixed_vars() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        unsafe {
            std::env::set_var("COPILOTSTUDIOAGENT__ENVIRONMENTID", "env-123");
            std::env::set_var("COPILOTSTUDIOAGENT__SCHEMANAME", "my-agent");
            std::env::set_var("COPILOTSTUDIOAGENT__AGENTAPPID", "client-abc");
            std::env::set_var("COPILOTSTUDIOAGENT__TENANTID", "tenant-xyz");
        }
        let settings = CopilotStudioSettings::from_env();
        clear_env();
        assert_eq!(settings.environment_id.as_deref(), Some("env-123"));
        assert_eq!(settings.schema_name.as_deref(), Some("my-agent"));
        assert_eq!(settings.agent_app_id.as_deref(), Some("client-abc"));
        assert_eq!(settings.tenant_id.as_deref(), Some("tenant-xyz"));
    }

    #[test]
    fn from_env_missing_vars_are_none() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        let settings = CopilotStudioSettings::from_env();
        assert_eq!(settings, CopilotStudioSettings::default());
    }

    #[test]
    fn from_env_empty_string_treated_as_unset() {
        let _guard = ENV_MUTEX.lock().unwrap();
        clear_env();
        unsafe { std::env::set_var("COPILOTSTUDIOAGENT__ENVIRONMENTID", "") };
        let settings = CopilotStudioSettings::from_env();
        clear_env();
        assert!(settings.environment_id.is_none());
    }

    // -- CopilotStudioConnectionSettings::from_settings ------------------

    #[test]
    fn from_settings_requires_environment_id() {
        let settings = CopilotStudioSettings {
            schema_name: Some("agent".into()),
            ..Default::default()
        };
        let err = CopilotStudioConnectionSettings::from_settings(&settings).unwrap_err();
        assert!(err.to_string().contains("environment ID is required"));
    }

    #[test]
    fn from_settings_requires_schema_name() {
        let settings = CopilotStudioSettings {
            environment_id: Some("env".into()),
            ..Default::default()
        };
        let err = CopilotStudioConnectionSettings::from_settings(&settings).unwrap_err();
        assert!(err.to_string().contains("agent identifier"));
    }

    #[test]
    fn from_settings_builds_connection_settings() {
        let settings = CopilotStudioSettings {
            environment_id: Some("52d1e846-e080-4341-a63f-58f2ab72fb28".into()),
            schema_name: Some("my-schema".into()),
            ..Default::default()
        };
        let conn = CopilotStudioConnectionSettings::from_settings(&settings).unwrap();
        assert_eq!(conn.environment_id, "52d1e846-e080-4341-a63f-58f2ab72fb28");
        assert_eq!(conn.agent_identifier, "my-schema");
        assert_eq!(conn.cloud, PowerPlatformCloud::Prod);
        assert_eq!(conn.agent_type, AgentType::Published);
    }

    // -- conversation_url (environment/cloud based) -----------------------

    #[test]
    fn conversation_url_prod_hashes_environment_id_and_splits_last_two_chars() {
        // 32 lowercase hex chars (a real Power Platform environment id,
        // dashes included as it would be copy-pasted from the Power
        // Platform admin center).
        let conn = CopilotStudioConnectionSettings::new(
            "52d1e846-e080-4341-a63f-58f2ab72fb28",
            "my-agent-schema",
        );
        let url = conn.conversation_url(None).unwrap();
        // normalized (dashes stripped, lowercased): 52d1e846e0804341a63f58f2ab72fb28
        // Prod => id_suffix_length 2 => prefix = ...fb2, suffix = 8? Actually
        // split from the *end*: last 2 chars = "28", rest is the prefix.
        assert_eq!(
            url,
            "https://52d1e846e0804341a63f58f2ab72fb.28.environment.api.powerplatform.com\
             /copilotstudio/dataverse-backed/authenticated/bots/my-agent-schema/conversations\
             ?api-version=2022-03-01-preview"
        );
    }

    #[test]
    fn conversation_url_with_conversation_id_appends_conversation_segment() {
        let conn = CopilotStudioConnectionSettings::new(
            "52d1e846-e080-4341-a63f-58f2ab72fb28",
            "my-agent-schema",
        );
        let url = conn.conversation_url(Some("conv-1")).unwrap();
        assert!(url.ends_with(
            "/copilotstudio/dataverse-backed/authenticated/bots/my-agent-schema/conversations/conv-1\
             ?api-version=2022-03-01-preview"
        ));
        assert!(!url.contains("conversations?"));
    }

    #[test]
    fn conversation_url_prebuilt_agent_type_uses_prebuilt_path_segment() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "s")
            .with_agent_type(AgentType::Prebuilt);
        let url = conn.conversation_url(None).unwrap();
        assert!(url.contains("/copilotstudio/prebuilt/authenticated/bots/s/conversations"));
    }

    #[test]
    fn conversation_url_first_release_also_uses_two_char_id_suffix_and_prod_host() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "s")
            .with_cloud(PowerPlatformCloud::FirstRelease);
        let url = conn.conversation_url(None).unwrap();
        assert!(url.starts_with(
            "https://52d1e846e0804341a63f58f2ab72fb.28.environment.api.powerplatform.com/"
        ));
    }

    #[test]
    fn conversation_url_gov_cloud_uses_one_char_id_suffix_and_gov_host() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "s")
            .with_cloud(PowerPlatformCloud::Gov);
        let url = conn.conversation_url(None).unwrap();
        assert!(url.starts_with(
            "https://52d1e846e0804341a63f58f2ab72fb2.8.environment.api.gov.powerplatform.microsoft.us/"
        ));
    }

    #[test]
    fn conversation_url_other_cloud_requires_custom_host() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "s")
            .with_cloud(PowerPlatformCloud::Other);
        let err = conn.conversation_url(None).unwrap_err();
        assert!(err.to_string().contains("custom_power_platform_cloud"));
    }

    #[test]
    fn conversation_url_other_cloud_with_custom_host_succeeds() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "s")
            .with_cloud(PowerPlatformCloud::Other)
            .with_custom_power_platform_cloud("api.contoso-private.example.com");
        let url = conn.conversation_url(None).unwrap();
        // Other, like every non-Prod/FirstRelease cloud, uses a 1-char id
        // suffix (see `conversation_url_gov_cloud_uses_one_char_id_suffix...`).
        assert!(url.starts_with(
            "https://52d1e846e0804341a63f58f2ab72fb2.8.environment.api.contoso-private.example.com/"
        ));
    }

    #[test]
    fn conversation_url_empty_environment_id_errors() {
        let conn = CopilotStudioConnectionSettings::new("", "s");
        let err = conn.conversation_url(None).unwrap_err();
        assert!(err.to_string().contains("environment_id"));
    }

    #[test]
    fn conversation_url_empty_agent_identifier_errors() {
        let conn = CopilotStudioConnectionSettings::new("52d1e846e0804341a63f58f2ab72fb28", "");
        let err = conn.conversation_url(None).unwrap_err();
        assert!(err.to_string().contains("agent_identifier"));
    }

    // -- conversation_url (direct_connect_url override) --------------------

    #[test]
    fn conversation_url_direct_connect_override_bypasses_environment_hashing() {
        let conn = CopilotStudioConnectionSettings::new("unused-env", "unused-agent")
            .with_direct_connect_url("https://island.example.com/some/base/");
        let url = conn.conversation_url(None).unwrap();
        assert_eq!(
            url,
            "https://island.example.com/some/base/conversations?api-version=2022-03-01-preview"
        );
    }

    #[test]
    fn conversation_url_direct_connect_strips_existing_conversations_segment() {
        let conn = CopilotStudioConnectionSettings::new("unused", "unused")
            .with_direct_connect_url("https://island.example.com/base/conversations/old-conv-id");
        let url = conn.conversation_url(Some("new-conv")).unwrap();
        assert_eq!(
            url,
            "https://island.example.com/base/conversations/new-conv?api-version=2022-03-01-preview"
        );
    }

    #[test]
    fn conversation_url_direct_connect_invalid_url_errors() {
        let conn = CopilotStudioConnectionSettings::new("unused", "unused")
            .with_direct_connect_url("not-a-url");
        let err = conn.conversation_url(None).unwrap_err();
        assert!(err.to_string().contains("not an absolute URL"));
    }
}
