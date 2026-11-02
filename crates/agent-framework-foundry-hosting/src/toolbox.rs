//! `FoundryToolbox`: a Foundry toolbox's MCP endpoint as an MCP tool source
//! (upstream `FoundryToolbox`, an `MCPStreamableHTTPTool`).
//!
//! The endpoint is `url`, else `TOOLBOX_ENDPOINT`, else
//! `{FOUNDRY_PROJECT_ENDPOINT}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1`.
//! Every request carries a bearer token for the Foundry data plane, the
//! `Foundry-Features` header (`Toolboxes=V1Preview` merged with
//! `FOUNDRY_AGENT_TOOLSET_FEATURES`) and, inside a hosted request, the
//! platform call id.

use std::sync::Arc;
use std::time::Duration;

use agent_framework_azure::TokenCredential;
use agent_framework_core::error::{Error, Result};
use agent_framework_mcp::McpStreamableHttpTool;

use crate::scope::{headers, FoundryRequestContext};

/// The default token scope for toolbox requests.
pub const DEFAULT_TOOLBOX_SCOPE: &str = "https://ai.azure.com/.default";
/// The preview feature every toolbox request declares.
pub const MANDATORY_TOOLBOX_FEATURE: &str = "Toolboxes=V1Preview";
/// Platform-provided toolbox feature flags.
pub const TOOLSET_FEATURES_ENV: &str = "FOUNDRY_AGENT_TOOLSET_FEATURES";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);

/// The `Foundry-Features` value: the mandatory feature merged with the
/// platform's (kept as given when it already names the mandatory one).
pub fn features_header(additional: Option<&str>) -> String {
    match additional.map(str::trim).filter(|a| !a.is_empty()) {
        None => MANDATORY_TOOLBOX_FEATURE.to_string(),
        Some(extra)
            if extra
                .split(',')
                .any(|f| f.trim().eq_ignore_ascii_case(MANDATORY_TOOLBOX_FEATURE)) =>
        {
            extra.to_string()
        }
        Some(extra) => format!("{MANDATORY_TOOLBOX_FEATURE},{extra}"),
    }
}

/// The toolbox endpoint from `lookup` (an environment).
pub fn resolve_endpoint(lookup: impl Fn(&str) -> Option<String>) -> Result<String> {
    if let Some(endpoint) = lookup("TOOLBOX_ENDPOINT") {
        if endpoint.is_empty() {
            return Err(Error::Configuration(
                "TOOLBOX_ENDPOINT is set but empty.".into(),
            ));
        }
        return Ok(endpoint);
    }
    match (lookup("FOUNDRY_PROJECT_ENDPOINT"), lookup("TOOLBOX_NAME")) {
        (Some(project), Some(name)) if !project.is_empty() && !name.is_empty() => Ok(format!(
            "{}/toolboxes/{name}/mcp?api-version=v1",
            project.trim_end_matches('/')
        )),
        _ => Err(Error::Configuration(
            "Pass 'url', or set TOOLBOX_ENDPOINT, or set both FOUNDRY_PROJECT_ENDPOINT and \
             TOOLBOX_NAME to build the toolbox MCP endpoint."
                .into(),
        )),
    }
}

/// The toolbox name in an endpoint's path (`…/toolboxes/<name>/…`), else
/// `"toolbox"`.
pub fn toolbox_name_from_endpoint(endpoint: &str) -> String {
    let path = endpoint
        .split_once("://")
        .map(|(_, rest)| rest)
        .unwrap_or(endpoint);
    let path = path.split(['?', '#']).next().unwrap_or_default();
    let segments: Vec<&str> = path.split('/').collect();
    segments
        .iter()
        .position(|s| *s == "toolboxes")
        .and_then(|i| segments.get(i + 1))
        .filter(|s| !s.is_empty())
        .map_or_else(|| "toolbox".to_string(), |s| s.to_string())
}

/// Builds toolbox tools. See the [module docs](self).
#[derive(Clone)]
pub struct FoundryToolbox {
    credential: Arc<dyn TokenCredential>,
    url: Option<String>,
    name: Option<String>,
    token_scope: String,
    timeout: Duration,
    context: Option<FoundryRequestContext>,
}

impl std::fmt::Debug for FoundryToolbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryToolbox")
            .field("url", &self.url)
            .field("name", &self.name)
            .field("token_scope", &self.token_scope)
            .finish_non_exhaustive()
    }
}

impl FoundryToolbox {
    /// A toolbox authenticating with `credential`.
    pub fn new(credential: Arc<dyn TokenCredential>) -> Self {
        Self {
            credential,
            url: None,
            name: None,
            token_scope: DEFAULT_TOOLBOX_SCOPE.into(),
            timeout: DEFAULT_TIMEOUT,
            context: None,
        }
    }

    /// The endpoint (default: from the environment).
    pub fn url(mut self, url: impl Into<String>) -> Self {
        self.url = Some(url.into());
        self
    }

    /// The local tool-source name (default `TOOLBOX_NAME`, else the
    /// endpoint's toolbox segment).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// The token scope (default [`DEFAULT_TOOLBOX_SCOPE`]).
    pub fn token_scope(mut self, scope: impl Into<String>) -> Self {
        self.token_scope = scope.into();
        self
    }

    /// The per-request timeout (default 120 s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Forward this request's platform call id.
    pub fn request_context(mut self, context: FoundryRequestContext) -> Self {
        self.context = Some(context);
        self
    }

    /// The MCP tool, with the token fetched now. Tokens expire, so build a
    /// toolbox per request (or per agent instance from an agent factory)
    /// rather than once per process.
    pub async fn build(&self) -> Result<McpStreamableHttpTool> {
        self.build_with(|name| std::env::var(name).ok()).await
    }

    /// [`Self::build`] reading configuration from `lookup`.
    pub async fn build_with(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<McpStreamableHttpTool> {
        let endpoint = match &self.url {
            Some(url) => url.clone(),
            None => resolve_endpoint(&lookup)?,
        };
        let name = self
            .name
            .clone()
            .or_else(|| lookup("TOOLBOX_NAME").filter(|n| !n.is_empty()))
            .unwrap_or_else(|| toolbox_name_from_endpoint(&endpoint));
        let token = self
            .credential
            .get_token_for_scope(&self.token_scope)
            .await?;
        let mut headers = vec![
            ("Authorization".to_string(), format!("Bearer {token}")),
            (
                "Foundry-Features".to_string(),
                features_header(lookup(TOOLSET_FEATURES_ENV).as_deref()),
            ),
        ];
        if let Some(call_id) = self.context.as_ref().and_then(|c| c.call_id.clone()) {
            headers.push((headers::FOUNDRY_CALL_ID.to_string(), call_id));
        }
        Ok(McpStreamableHttpTool::new(name, endpoint)
            .headers(headers)
            .timeout(self.timeout))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_and_endpoints() {
        assert_eq!(features_header(None), "Toolboxes=V1Preview");
        assert_eq!(features_header(Some(" ")), "Toolboxes=V1Preview");
        assert_eq!(features_header(Some("A=1")), "Toolboxes=V1Preview,A=1");
        assert_eq!(
            features_header(Some("toolboxes=v1preview,B")),
            "toolboxes=v1preview,B"
        );
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |n: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == n)
                    .map(|(_, v)| v.to_string())
            }
        };
        assert_eq!(
            resolve_endpoint(env(&[
                ("FOUNDRY_PROJECT_ENDPOINT", "https://p/"),
                ("TOOLBOX_NAME", "tb")
            ]))
            .unwrap(),
            "https://p/toolboxes/tb/mcp?api-version=v1"
        );
        assert_eq!(
            resolve_endpoint(env(&[("TOOLBOX_ENDPOINT", "https://e")])).unwrap(),
            "https://e"
        );
        assert!(resolve_endpoint(env(&[("TOOLBOX_ENDPOINT", "")])).is_err());
        assert!(resolve_endpoint(env(&[])).is_err());
        assert_eq!(
            toolbox_name_from_endpoint("https://p/toolboxes/tb/versions/2/mcp?x=1"),
            "tb"
        );
        assert_eq!(toolbox_name_from_endpoint("https://p/mcp"), "toolbox");
    }
}
