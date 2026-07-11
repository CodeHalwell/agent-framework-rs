//! [`PurviewSettings`] and the location-scoping types it carries. Mirrors
//! Python's `agent_framework_purview._settings`.

use crate::models::PolicyLocation;

/// Default Microsoft Graph base URI, matching `PurviewSettings.graph_base_uri`'s
/// Python default.
pub const DEFAULT_GRAPH_BASE_URI: &str = "https://graph.microsoft.com/v1.0/";

/// Default message returned when a prompt is blocked by policy.
pub const DEFAULT_BLOCKED_PROMPT_MESSAGE: &str = "Prompt blocked by policy";

/// Default message returned when a response is blocked by policy.
pub const DEFAULT_BLOCKED_RESPONSE_MESSAGE: &str = "Response blocked by policy";

/// Default protection-scopes cache TTL Python declares (14400s = 4 hours).
/// Carried here for settings parity even though this port does not cache —
/// see the crate docs' "Scope" section.
pub const DEFAULT_CACHE_TTL_SECONDS: u64 = 14_400;

/// Default max cache size Python declares (200 MiB). See
/// [`DEFAULT_CACHE_TTL_SECONDS`].
pub const DEFAULT_MAX_CACHE_SIZE_BYTES: u64 = 200 * 1024 * 1024;

/// The kind of location a [`PurviewAppLocation`] identifies. Mirrors
/// Python's `PurviewLocationType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PurviewLocationType {
    Application,
    Uri,
    Domain,
}

/// Identifies the calling application's location for Purview policy
/// evaluation (an application id, a URL, or a domain). Mirrors Python's
/// `PurviewAppLocation`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurviewAppLocation {
    pub location_type: PurviewLocationType,
    pub location_value: String,
}

impl PurviewAppLocation {
    pub fn new(location_type: PurviewLocationType, location_value: impl Into<String>) -> Self {
        Self {
            location_type,
            location_value: location_value.into(),
        }
    }

    /// Build the Graph `@odata.type` + `value` pair for this location.
    /// Mirrors `PurviewAppLocation.get_policy_location`.
    pub fn to_policy_location(&self) -> PolicyLocation {
        let data_type = match self.location_type {
            PurviewLocationType::Application => "microsoft.graph.policyLocationApplication",
            PurviewLocationType::Uri => "microsoft.graph.policyLocationUrl",
            PurviewLocationType::Domain => "microsoft.graph.policyLocationDomain",
        };
        PolicyLocation {
            data_type: data_type.to_string(),
            value: self.location_value.clone(),
        }
    }
}

/// Settings for Purview integration. Mirrors Python's `PurviewSettings`
/// (an `AFBaseSettings`/pydantic model there; a plain builder-style struct
/// here — this port has no environment-variable-driven construction for it,
/// since Python's `PurviewSettings` isn't env-prefixed either, unlike
/// `agent-framework-copilotstudio`'s `CopilotStudioSettings`).
#[derive(Debug, Clone)]
pub struct PurviewSettings {
    /// Required: the calling application's display/logical name. Sent as
    /// both `integratedAppMetadata.name` and `protectedAppMetadata.name`.
    pub app_name: String,
    pub app_version: Option<String>,
    /// The tenant id (GUID) evaluated requests are scoped to. Python can
    /// also infer this from the bearer token's `tid` claim when unset; this
    /// port requires it to be set explicitly — see the crate docs' "Scope"
    /// section (no JWT introspection here).
    pub tenant_id: Option<String>,
    /// Where the calling application is considered to run, for policy
    /// location matching. Python can also infer an application-type
    /// location from the bearer token's `appid` claim when unset; this port
    /// requires it to be set explicitly, for the same reason as `tenant_id`.
    pub purview_app_location: Option<PurviewAppLocation>,
    pub graph_base_uri: String,
    pub blocked_prompt_message: String,
    pub blocked_response_message: String,
    /// If `true`, a policy-evaluation failure (any error other than a 402 —
    /// see [`Self::ignore_payment_required`]) is logged and swallowed
    /// (fail-open: the run proceeds as if evaluation passed) instead of
    /// propagated.
    pub ignore_exceptions: bool,
    /// If `true`, a 402 Payment Required response is logged and swallowed
    /// instead of propagated — checked independently of
    /// [`Self::ignore_exceptions`], mirroring Python's separate
    /// `except PurviewPaymentRequiredError` clause (checked *before*, and
    /// independently of, the generic exception handler).
    pub ignore_payment_required: bool,
    /// Carried for settings parity with Python; this port does not cache
    /// protection-scopes responses (see the crate docs), so this value is
    /// currently unused.
    pub cache_ttl_seconds: u64,
    /// See [`Self::cache_ttl_seconds`].
    pub max_cache_size_bytes: u64,
}

impl PurviewSettings {
    /// `app_name` is Purview's only required setting; everything else takes
    /// Python's documented defaults.
    pub fn new(app_name: impl Into<String>) -> Self {
        Self {
            app_name: app_name.into(),
            app_version: None,
            tenant_id: None,
            purview_app_location: None,
            graph_base_uri: DEFAULT_GRAPH_BASE_URI.to_string(),
            blocked_prompt_message: DEFAULT_BLOCKED_PROMPT_MESSAGE.to_string(),
            blocked_response_message: DEFAULT_BLOCKED_RESPONSE_MESSAGE.to_string(),
            ignore_exceptions: false,
            ignore_payment_required: false,
            cache_ttl_seconds: DEFAULT_CACHE_TTL_SECONDS,
            max_cache_size_bytes: DEFAULT_MAX_CACHE_SIZE_BYTES,
        }
    }

    pub fn with_app_version(mut self, version: impl Into<String>) -> Self {
        self.app_version = Some(version.into());
        self
    }
    pub fn with_tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }
    pub fn with_purview_app_location(mut self, location: PurviewAppLocation) -> Self {
        self.purview_app_location = Some(location);
        self
    }
    pub fn with_graph_base_uri(mut self, uri: impl Into<String>) -> Self {
        self.graph_base_uri = uri.into();
        self
    }
    pub fn with_blocked_prompt_message(mut self, message: impl Into<String>) -> Self {
        self.blocked_prompt_message = message.into();
        self
    }
    pub fn with_blocked_response_message(mut self, message: impl Into<String>) -> Self {
        self.blocked_response_message = message.into();
        self
    }
    pub fn with_ignore_exceptions(mut self, value: bool) -> Self {
        self.ignore_exceptions = value;
        self
    }
    pub fn with_ignore_payment_required(mut self, value: bool) -> Self {
        self.ignore_payment_required = value;
        self
    }
    pub fn with_cache_ttl_seconds(mut self, ttl: u64) -> Self {
        self.cache_ttl_seconds = ttl;
        self
    }
    pub fn with_max_cache_size_bytes(mut self, bytes: u64) -> Self {
        self.max_cache_size_bytes = bytes;
        self
    }

    /// The Microsoft Graph OAuth scope(s) required for this base URI:
    /// `https://{host}/.default`. Mirrors `PurviewSettings.get_scopes`.
    pub fn get_scopes(&self) -> Vec<String> {
        let host =
            extract_host(&self.graph_base_uri).unwrap_or_else(|| "graph.microsoft.com".to_string());
        vec![format!("https://{host}/.default")]
    }
}

/// Extract the host from an `http(s)://host[:port]/path` URI without a full
/// URL-parsing dependency — this crate only ever needs the host, and Graph
/// base URIs are always simple `https://graph.microsoft.com/...`-shaped.
fn extract_host(uri: &str) -> Option<String> {
    let after_scheme = uri.split_once("://").map(|(_, rest)| rest).unwrap_or(uri);
    let host = after_scheme.split(['/', '?', '#']).next()?;
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_applies_python_documented_defaults() {
        let settings = PurviewSettings::new("Test App");
        assert_eq!(settings.app_name, "Test App");
        assert_eq!(settings.graph_base_uri, "https://graph.microsoft.com/v1.0/");
        assert!(settings.tenant_id.is_none());
        assert!(settings.purview_app_location.is_none());
        assert_eq!(settings.blocked_prompt_message, "Prompt blocked by policy");
        assert_eq!(
            settings.blocked_response_message,
            "Response blocked by policy"
        );
        assert!(!settings.ignore_exceptions);
        assert!(!settings.ignore_payment_required);
        assert_eq!(settings.cache_ttl_seconds, 14_400);
        assert_eq!(settings.max_cache_size_bytes, 200 * 1024 * 1024);
    }

    #[test]
    fn builders_override_defaults() {
        let location = PurviewAppLocation::new(PurviewLocationType::Application, "app-123");
        let settings = PurviewSettings::new("Test App")
            .with_graph_base_uri("https://graph.microsoft-ppe.com")
            .with_tenant_id("test-tenant-id")
            .with_purview_app_location(location.clone());
        assert_eq!(settings.graph_base_uri, "https://graph.microsoft-ppe.com");
        assert_eq!(settings.tenant_id.as_deref(), Some("test-tenant-id"));
        assert_eq!(settings.purview_app_location, Some(location));
    }

    #[test]
    fn get_scopes_derives_default_suffix_from_graph_base_uri() {
        let settings = PurviewSettings::new("Test App");
        assert_eq!(
            settings.get_scopes(),
            vec!["https://graph.microsoft.com/.default".to_string()]
        );
    }

    #[test]
    fn get_scopes_derives_suffix_from_custom_graph_base_uri() {
        let settings = PurviewSettings::new("Test App")
            .with_graph_base_uri("https://graph.microsoft-ppe.com/v1.0/");
        assert_eq!(
            settings.get_scopes(),
            vec!["https://graph.microsoft-ppe.com/.default".to_string()]
        );
    }

    #[test]
    fn get_policy_location_maps_all_three_location_types() {
        let cases = [
            (
                PurviewLocationType::Application,
                "microsoft.graph.policyLocationApplication",
            ),
            (
                PurviewLocationType::Uri,
                "microsoft.graph.policyLocationUrl",
            ),
            (
                PurviewLocationType::Domain,
                "microsoft.graph.policyLocationDomain",
            ),
        ];
        for (location_type, expected) in cases {
            let location = PurviewAppLocation::new(location_type, "value-1");
            let policy_location = location.to_policy_location();
            assert_eq!(policy_location.data_type, expected);
            assert_eq!(policy_location.value, "value-1");
        }
    }

    #[test]
    fn extract_host_handles_scheme_path_and_bare_host() {
        assert_eq!(
            extract_host("https://graph.microsoft.com/v1.0/"),
            Some("graph.microsoft.com".to_string())
        );
        assert_eq!(
            extract_host("https://graph.microsoft-ppe.com"),
            Some("graph.microsoft-ppe.com".to_string())
        );
        assert_eq!(
            extract_host("graph.microsoft.com/v1.0/"),
            Some("graph.microsoft.com".to_string())
        );
    }
}
