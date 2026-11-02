//! `SecureAgentConfig`: one object wiring the whole FIDES layer into an
//! agent.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use super::labels::{ConfidentialityLabel, IntegrityLabel};
use super::policy::{
    PendingPolicyApproval, PolicyApprovalHandler, PolicyEnforcementFunctionMiddleware,
    DEFAULT_MAX_PENDING_APPROVALS, DEFAULT_PENDING_APPROVAL_TTL,
};
use super::scope::{ScopedVariableStore, SecurityScope, ToolSecurity, ToolSecurityRegistry};
use super::tools::{get_security_tools, SECURITY_TOOL_INSTRUCTIONS};
use super::tracking::LabelTrackingFunctionMiddleware;
use crate::agent::AgentBuilder;
use crate::client::ChatClient;
use crate::error::Result;
use crate::memory::{ContextProvider, SessionContext};
use crate::middleware::FunctionMiddleware;
use crate::session::AgentSession;
use crate::tools::ToolDefinition;

/// The default context-provider source id (and session-state key).
pub const DEFAULT_SOURCE_ID: &str = "secure_agent";

/// Prompt-injection defense for an agent in one object.
///
/// As a [`ContextProvider`] it adds the security tools and their
/// instructions to every run; [`middleware`](Self::middleware) gives the
/// label tracker and the policy enforcer that must also be installed (a
/// context provider cannot add middleware in this port).
/// [`apply`](Self::apply) does both on an [`AgentBuilder`]. Security state
/// lives in the run's session, under the config's source id.
///
/// ```no_run
/// use agent_framework_core::prelude::*;
/// use agent_framework_core::security::{SecureAgentConfig, ToolSecurity, IntegrityLabel};
///
/// # fn demo(client: impl ChatClient + 'static) -> Result<()> {
/// let security = std::sync::Arc::new(
///     SecureAgentConfig::builder()
///         .tool_security(
///             "fetch_page",
///             ToolSecurity::new().source_integrity(IntegrityLabel::Untrusted),
///         )
///         .build()?,
/// );
/// let agent = security
///     .apply(Agent::builder(client).instructions("Be helpful."))
///     .build();
/// # let _ = agent;
/// # Ok(())
/// # }
/// ```
pub struct SecureAgentConfig {
    source_id: String,
    label_tracker: Arc<LabelTrackingFunctionMiddleware>,
    policy_enforcer: Option<Arc<PolicyEnforcementFunctionMiddleware>>,
    tools: Vec<ToolDefinition>,
    registry: ToolSecurityRegistry,
    scope: SecurityScope,
    quarantine_client: Option<Arc<dyn ChatClient>>,
}

impl std::fmt::Debug for SecureAgentConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecureAgentConfig")
            .field("source_id", &self.source_id)
            .field("label_tracker", &self.label_tracker)
            .field("policy_enforcer", &self.policy_enforcer)
            .finish_non_exhaustive()
    }
}

/// Builds a [`SecureAgentConfig`].
pub struct SecureAgentConfigBuilder {
    auto_hide_untrusted: bool,
    default_integrity: IntegrityLabel,
    default_confidentiality: ConfidentialityLabel,
    allow_untrusted_tools: HashSet<String>,
    block_on_violation: bool,
    approval_on_violation: bool,
    approval_handler: Option<PolicyApprovalHandler>,
    enable_audit_log: bool,
    enable_policy_enforcement: bool,
    quarantine_client: Option<Arc<dyn ChatClient>>,
    source_id: String,
    max_pending_approvals: usize,
    pending_approval_ttl: Option<Duration>,
    registry: ToolSecurityRegistry,
}

impl Default for SecureAgentConfigBuilder {
    fn default() -> Self {
        Self {
            auto_hide_untrusted: true,
            default_integrity: IntegrityLabel::Untrusted,
            default_confidentiality: ConfidentialityLabel::Public,
            allow_untrusted_tools: ["quarantined_llm", "inspect_variable"]
                .into_iter()
                .map(String::from)
                .collect(),
            block_on_violation: true,
            approval_on_violation: false,
            approval_handler: None,
            enable_audit_log: true,
            enable_policy_enforcement: true,
            quarantine_client: None,
            source_id: DEFAULT_SOURCE_ID.into(),
            max_pending_approvals: DEFAULT_MAX_PENDING_APPROVALS,
            pending_approval_ttl: Some(DEFAULT_PENDING_APPROVAL_TTL),
            registry: ToolSecurityRegistry::new(),
        }
    }
}

impl SecureAgentConfigBuilder {
    /// Hide untrusted results (default `true`).
    pub fn auto_hide_untrusted(mut self, hide: bool) -> Self {
        self.auto_hide_untrusted = hide;
        self
    }

    /// The integrity of undeclared tools' results (default `UNTRUSTED`).
    pub fn default_integrity(mut self, integrity: IntegrityLabel) -> Self {
        self.default_integrity = integrity;
        self
    }

    /// The confidentiality of undeclared tools' results (default `PUBLIC`).
    pub fn default_confidentiality(mut self, confidentiality: ConfidentialityLabel) -> Self {
        self.default_confidentiality = confidentiality;
        self
    }

    /// Tools allowed in an untrusted context (the security tools always
    /// are).
    pub fn allow_untrusted_tools(
        mut self,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allow_untrusted_tools
            .extend(tools.into_iter().map(Into::into));
        self
    }

    /// Block violations (default `true`).
    pub fn block_on_violation(mut self, block: bool) -> Self {
        self.block_on_violation = block;
        self
    }

    /// Put violations to approval instead of blocking.
    pub fn approval_on_violation(mut self, approval: bool) -> Self {
        self.approval_on_violation = approval;
        self
    }

    /// Decide approvals inline.
    pub fn approval_handler(mut self, handler: PolicyApprovalHandler) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    /// Keep an audit log (default `true`).
    pub fn enable_audit_log(mut self, enable: bool) -> Self {
        self.enable_audit_log = enable;
        self
    }

    /// Install the policy enforcer (default `true`).
    pub fn enable_policy_enforcement(mut self, enable: bool) -> Self {
        self.enable_policy_enforcement = enable;
        self
    }

    /// The client `quarantined_llm` calls for this config's runs. Without
    /// one, those runs get placeholder responses (the global client is
    /// not inherited).
    pub fn quarantine_client(mut self, client: Arc<dyn ChatClient>) -> Self {
        self.quarantine_client = Some(client);
        self
    }

    /// The provider source id, also the session-state key.
    pub fn source_id(mut self, id: impl Into<String>) -> Self {
        self.source_id = id.into();
        self
    }

    /// The bound on pending approvals per session.
    pub fn max_pending_approvals(mut self, max: usize) -> Self {
        self.max_pending_approvals = max;
        self
    }

    /// The lifetime of an unconsumed approval.
    pub fn pending_approval_ttl(mut self, ttl: Option<Duration>) -> Self {
        self.pending_approval_ttl = ttl;
        self
    }

    /// Declare a tool's security properties.
    pub fn tool_security(self, tool: impl Into<String>, security: ToolSecurity) -> Self {
        self.registry.declare(tool, security);
        self
    }

    /// Build; fails on an invalid approval bound or TTL.
    pub fn build(self) -> Result<SecureAgentConfig> {
        let scope = SecurityScope::new(self.source_id.clone());
        let label_tracker = LabelTrackingFunctionMiddleware::new()
            .auto_hide_untrusted(self.auto_hide_untrusted)
            .default_integrity(self.default_integrity)
            .default_confidentiality(self.default_confidentiality)
            .scope(scope.clone())
            .tool_registry(self.registry.clone())
            .quarantine_client(self.quarantine_client.clone());
        let mut policy = PolicyEnforcementFunctionMiddleware::new()
            .allow_untrusted_tools(self.allow_untrusted_tools)
            .block_on_violation(self.block_on_violation)
            .approval_on_violation(self.approval_on_violation)
            .enable_audit_log(self.enable_audit_log)
            .max_pending_approvals(self.max_pending_approvals)?
            .pending_approval_ttl(self.pending_approval_ttl)?
            .scope(scope.clone())
            .tool_registry(self.registry.clone());
        if let Some(handler) = self.approval_handler {
            policy = policy.approval_handler(handler);
        }
        Ok(SecureAgentConfig {
            source_id: self.source_id,
            label_tracker: Arc::new(label_tracker),
            policy_enforcer: self.enable_policy_enforcement.then(|| Arc::new(policy)),
            tools: get_security_tools(),
            registry: self.registry,
            scope,
            quarantine_client: self.quarantine_client,
        })
    }
}

impl SecureAgentConfig {
    /// A builder with the defaults.
    pub fn builder() -> SecureAgentConfigBuilder {
        SecureAgentConfigBuilder::default()
    }

    /// The label tracker.
    pub fn label_tracker(&self) -> &Arc<LabelTrackingFunctionMiddleware> {
        &self.label_tracker
    }

    /// The policy enforcer, when enabled.
    pub fn policy_enforcer(&self) -> Option<&Arc<PolicyEnforcementFunctionMiddleware>> {
        self.policy_enforcer.as_ref()
    }

    /// The function middleware to install, in order.
    pub fn middleware(&self) -> Vec<Arc<FunctionMiddleware>> {
        let mut out: Vec<Arc<FunctionMiddleware>> = vec![self.label_tracker.clone()];
        if let Some(p) = &self.policy_enforcer {
            out.push(p.clone());
        }
        out
    }

    /// The security tools this config adds.
    pub fn tools(&self) -> Vec<ToolDefinition> {
        self.tools.clone()
    }

    /// The instructions this config adds.
    pub fn instructions(&self) -> &'static str {
        SECURITY_TOOL_INSTRUCTIONS
    }

    /// The tool declarations.
    pub fn registry(&self) -> &ToolSecurityRegistry {
        &self.registry
    }

    /// The bound quarantine client.
    pub fn quarantine_client(&self) -> Option<&Arc<dyn ChatClient>> {
        self.quarantine_client.as_ref()
    }

    /// Install the config on an agent: as a context provider, plus its
    /// middleware.
    pub fn apply(self: &Arc<Self>, builder: AgentBuilder) -> AgentBuilder {
        let mut builder = builder.context_provider(self.clone());
        for mw in self.middleware() {
            builder = builder.function_middleware(mw);
        }
        builder
    }

    /// The policy audit log for `session`.
    pub fn audit_log(&self, session: &AgentSession) -> Vec<Value> {
        self.scope.audit_log(Some(session))
    }

    /// The hidden variables of `session`.
    pub fn variable_store(&self, session: &AgentSession) -> ScopedVariableStore {
        self.scope.variable_store(Some(session))
    }

    /// The hidden variable ids of `session`.
    pub fn list_variables(&self, session: &AgentSession) -> Vec<String> {
        self.variable_store(session).list_variables()
    }

    /// The pending policy approvals of `session`.
    pub fn pending_approvals(&self, session: &AgentSession) -> Vec<PendingPolicyApproval> {
        self.policy_enforcer
            .as_ref()
            .map(|p| p.pending_approvals(Some(session)))
            .unwrap_or_default()
    }

    /// Approve a pending policy approval in `session`.
    pub fn approve(&self, session: &AgentSession, approval_id: &str) -> bool {
        self.policy_enforcer
            .as_ref()
            .is_some_and(|p| p.approve(Some(session), approval_id))
    }
}

#[async_trait]
impl ContextProvider for SecureAgentConfig {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        for tool in &self.tools {
            if !ctx.tools.iter().any(|t| t.name == tool.name) {
                ctx.tools.push(tool.clone());
            }
        }
        ctx.add_instructions(SECURITY_TOOL_INSTRUCTIONS);
        Ok(())
    }

    fn source_id(&self) -> String {
        self.source_id.clone()
    }
}
