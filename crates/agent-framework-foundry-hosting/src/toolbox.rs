//! `FoundryToolbox`: a Foundry toolbox's MCP endpoint as an MCP tool source
//! (upstream `FoundryToolbox`, an `MCPStreamableHTTPTool`).
//!
//! The endpoint is `url`, else `TOOLBOX_ENDPOINT`, else
//! `{FOUNDRY_PROJECT_ENDPOINT}/toolboxes/{TOOLBOX_NAME}/mcp?api-version=v1`.
//! Every request carries a bearer token for the Foundry data plane (fetched
//! from the credential for each request, so a long-lived tool never sends an
//! expired token; credentials cache and refresh their own tokens), the
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
    #[cfg(feature = "skills")]
    skills: ToolboxSkillsOptions,
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
            #[cfg(feature = "skills")]
            skills: ToolboxSkillsOptions::default(),
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

    /// The MCP tool. The bearer token is fetched from the credential for
    /// every MCP request, so the tool may be built once and reused.
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
        let credential = self.credential.clone();
        let scope = self.token_scope.clone();
        let token_provider: agent_framework_mcp::HeaderProvider = Arc::new(move || {
            let credential = credential.clone();
            let scope = scope.clone();
            Box::pin(async move {
                let token = credential.get_token_for_scope(&scope).await?;
                Ok(vec![(
                    "Authorization".to_string(),
                    format!("Bearer {token}"),
                )])
            })
        });
        let mut headers = vec![(
            "Foundry-Features".to_string(),
            features_header(lookup(TOOLSET_FEATURES_ENV).as_deref()),
        )];
        if let Some(call_id) = self.context.as_ref().and_then(|c| c.call_id.clone()) {
            headers.push((headers::FOUNDRY_CALL_ID.to_string(), call_id));
        }
        Ok(McpStreamableHttpTool::new(name, endpoint)
            .headers(headers)
            .header_provider(token_provider)
            .timeout(self.timeout))
    }
}

/// How a toolbox's skills are discovered and cached (feature `skills`;
/// upstream `as_skills_provider`'s keyword arguments).
#[cfg(feature = "skills")]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolboxSkillsOptions {
    /// How `archive` entries are unpacked (upstream `archive_*`).
    pub archive: agent_framework_mcp::skills::ArchiveOptions,
    /// [`ToolboxSkillsProvider`] re-discovers on every run (upstream
    /// `disable_caching`).
    pub disable_caching: bool,
    /// How long [`ToolboxSkillsProvider`]'s discovery stays fresh; `None`
    /// keeps it for the provider's life (upstream `cache_refresh_interval`).
    pub refresh_interval: Option<Duration>,
}

#[cfg(feature = "skills")]
impl FoundryToolbox {
    /// Skill discovery and caching options (default: upstream's).
    pub fn skills_options(mut self, options: ToolboxSkillsOptions) -> Self {
        self.skills = options;
        self
    }

    /// A [`SkillsProvider`](agent_framework_core::skills::SkillsProvider)
    /// over the Agent Skills this toolbox serves over MCP (upstream
    /// `as_skills_provider`): discovery reads `skill://index.json` on the
    /// toolbox's MCP session (see [`agent_framework_mcp::skills`]), with
    /// [`ToolboxSkillsOptions::archive`]. The skills are discovered now and
    /// kept; [`Self::refreshing_skills_provider`] re-discovers.
    pub async fn skills_provider(&self) -> Result<agent_framework_core::skills::SkillsProvider> {
        self.skills_provider_with(|name| std::env::var(name).ok())
            .await
    }

    /// [`Self::skills_provider`] reading configuration from `lookup`.
    pub async fn skills_provider_with(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<agent_framework_core::skills::SkillsProvider> {
        let tool = self.build_with(lookup).await?;
        Ok(agent_framework_core::skills::SkillsProvider::new(
            tool.skills_with(&self.skills.archive).await?,
        ))
    }

    /// A [`ToolboxSkillsProvider`]: discovery on first run, then cached as
    /// [`ToolboxSkillsOptions`] direct.
    pub async fn refreshing_skills_provider(&self) -> Result<ToolboxSkillsProvider> {
        self.refreshing_skills_provider_with(|name| std::env::var(name).ok())
            .await
    }

    /// [`Self::refreshing_skills_provider`] reading configuration from
    /// `lookup`.
    pub async fn refreshing_skills_provider_with(
        &self,
        lookup: impl Fn(&str) -> Option<String>,
    ) -> Result<ToolboxSkillsProvider> {
        Ok(ToolboxSkillsProvider {
            tool: self.build_with(lookup).await?,
            options: self.skills.clone(),
            configure: None,
            cache: tokio::sync::Mutex::new(None),
        })
    }
}

/// Customizes each [`SkillsProvider`](agent_framework_core::skills::SkillsProvider)
/// a [`ToolboxSkillsProvider`] builds.
#[cfg(feature = "skills")]
type ConfigureSkills = Arc<
    dyn Fn(
            agent_framework_core::skills::SkillsProvider,
        ) -> agent_framework_core::skills::SkillsProvider
        + Send
        + Sync,
>;

/// A context provider over a toolbox's skills that discovers them lazily
/// and caches them (upstream `as_skills_provider` with its caching source):
///
/// - The first run discovers; a failure fails that run, and the next retries.
/// - Then the discovery is reused until
///   [`ToolboxSkillsOptions::refresh_interval`] passes (never, by default),
///   when the next run re-discovers. A failed refresh keeps the previous
///   skills (logged) and retries on the next run.
/// - With [`ToolboxSkillsOptions::disable_caching`] every run discovers.
///
/// Concurrent runs share one discovery. Divergence: each discovery builds a
/// new [`SkillsProvider`](agent_framework_core::skills::SkillsProvider), so
/// skills loaded before a refresh are announced again only once loaded again
/// (their earlier `load_skill` results stay in the conversation). A
/// toolbox's platform-header identity is fixed when it is built, so there is
/// no per-identity cache to replace.
#[cfg(feature = "skills")]
pub struct ToolboxSkillsProvider {
    tool: McpStreamableHttpTool,
    options: ToolboxSkillsOptions,
    configure: Option<ConfigureSkills>,
    cache: tokio::sync::Mutex<
        Option<(
            tokio::time::Instant,
            agent_framework_core::skills::SkillsProvider,
        )>,
    >,
}

#[cfg(feature = "skills")]
impl std::fmt::Debug for ToolboxSkillsProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolboxSkillsProvider")
            .field("options", &self.options)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "skills")]
impl ToolboxSkillsProvider {
    /// Customize each discovered provider (approvals, filters, the
    /// instructions prompt): upstream's `disable_*_approval`,
    /// `instruction_template`.
    pub fn configure(
        mut self,
        configure: impl Fn(
                agent_framework_core::skills::SkillsProvider,
            ) -> agent_framework_core::skills::SkillsProvider
            + Send
            + Sync
            + 'static,
    ) -> Self {
        self.configure = Some(Arc::new(configure));
        self
    }

    /// The current skills provider, discovering as the options direct.
    pub async fn current(&self) -> Result<agent_framework_core::skills::SkillsProvider> {
        let mut cache = self.cache.lock().await;
        let fresh = match (&*cache, self.options.refresh_interval) {
            _ if self.options.disable_caching => None,
            (Some((_, provider)), None) => Some(provider),
            (Some((at, provider)), Some(interval)) if at.elapsed() < interval => Some(provider),
            _ => None,
        };
        if let Some(provider) = fresh {
            return Ok(provider.clone());
        }
        match self.tool.skills_with(&self.options.archive).await {
            Ok(skills) => {
                let mut provider = agent_framework_core::skills::SkillsProvider::new(skills);
                if let Some(configure) = &self.configure {
                    provider = configure(provider);
                }
                *cache = Some((tokio::time::Instant::now(), provider.clone()));
                Ok(provider)
            }
            Err(e) => match &*cache {
                Some((_, previous)) if !self.options.disable_caching => {
                    tracing::warn!(error = %e, "toolbox skill refresh failed; keeping the previous skills");
                    Ok(previous.clone())
                }
                _ => Err(e),
            },
        }
    }
}

#[cfg(feature = "skills")]
#[async_trait::async_trait]
impl agent_framework_core::memory::ContextProvider for ToolboxSkillsProvider {
    async fn before_run(
        &self,
        ctx: &mut agent_framework_core::memory::SessionContext,
    ) -> Result<()> {
        self.current().await?.before_run(ctx).await
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
