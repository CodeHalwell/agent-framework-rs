//! FIDES: information-flow control against prompt injection
//! (experimental, upstream `ExperimentalFeature.FIDES`).
//!
//! Port of upstream's `agent_framework.security`. Every piece of content
//! carries a [`ContentLabel`]: an integrity ([`IntegrityLabel`], trusted or
//! untrusted) and a confidentiality ([`ConfidentialityLabel`], public,
//! private, or restricted to a set of [`Principal`]s). Labels combine
//! towards the most restrictive ([`combine_labels`]).
//!
//! - [`LabelTrackingFunctionMiddleware`] labels every tool result (from
//!   the result's own label, the tool's declared [`ToolSecurity`], or its
//!   inputs), hides untrusted results behind [`VariableReferenceContent`]
//!   so the model never reads them, expands `[var_<id>]` references the
//!   model passes back into tool arguments, and accumulates the
//!   conversation's context label.
//! - [`PolicyEnforcementFunctionMiddleware`] blocks, warns on, or puts to
//!   approval calls made in an untrusted context, with untrusted
//!   arguments, or that would send data to a less confidential (or
//!   unauthorized) destination, with an audit log.
//! - The security tools ([`get_security_tools`]): `quarantined_llm`
//!   processes hidden content in an isolated model call (no tools), and
//!   `inspect_variable` reveals it (tainting the context).
//! - [`SecureAgentConfig`] wires all of it into an agent.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agent_framework_core::prelude::*;
//! use agent_framework_core::security::{
//!     ConfidentialityLabel, IntegrityLabel, SecureAgentConfig, ToolSecurity,
//! };
//!
//! # fn demo(client: impl ChatClient + 'static, fetch: ToolDefinition, post: ToolDefinition) -> Result<()> {
//! let security = Arc::new(
//!     SecureAgentConfig::builder()
//!         .tool_security("fetch_page", ToolSecurity::new().source_integrity(IntegrityLabel::Untrusted))
//!         .tool_security(
//!             "post_public",
//!             ToolSecurity::new().max_allowed_confidentiality(ConfidentialityLabel::Public),
//!         )
//!         .build()?,
//! );
//! let agent = security
//!     .apply(Agent::builder(client).tool(fetch).tool(post))
//!     .build();
//! # let _ = agent;
//! # Ok(())
//! # }
//! ```
//!
//! # State
//!
//! Security state (context label, hidden variables, audit log, pending
//! approvals) is JSON in the run's [`AgentSession`](crate::session::AgentSession)
//! state, under the [`SecurityScope`]'s key, so it is isolated per session
//! and survives session serialization. Without a session it lives in the
//! scope itself.
//!
//! # Divergences
//!
//! - **Tool declarations** (`source_integrity`, `confidentiality`,
//!   `max_allowed_confidentiality`, `accepts_untrusted`, principals,
//!   `standing_guidance`) are registered by tool name in a
//!   [`ToolSecurityRegistry`] shared by the middleware, since a Rust
//!   [`ToolDefinition`](crate::tools::ToolDefinition) has no
//!   `additional_properties`. The framework's own security tools are
//!   recognized by their executor, so a same-named tool cannot claim their
//!   authority.
//! - **Results are one JSON value.** A tool result is one item (upstream
//!   handles a list of `Content` items); an embedded label is a
//!   `security_label` member of the result object. Labels are reported in
//!   the invocation metadata ([`RESULT_LABEL_KEY`]) rather than stamped
//!   on the result. Standing guidance turns the result into a JSON array
//!   `[result, sentence, ...]`. Framework labels (from `quarantined_llm`
//!   and `inspect_variable`) travel out of band, not in the result.
//! - **Policy approval.** Function middleware cannot raise a
//!   `FunctionApprovalRequest` into this port's tool loop, so approval is
//!   either an inline [`PolicyApprovalHandler`] or a pending record: the
//!   call returns `{"approval_required": true, "approval_id", ...}` to the
//!   model, the host calls
//!   [`approve`](PolicyEnforcementFunctionMiddleware::approve), and the
//!   identical call (tool, arguments as sent and as resolved, labels,
//!   destination principals, session, disclosed violations) then runs once.
//! - **[`SecureAgentConfig`]** is a context provider for the tools and
//!   instructions, but its middleware must be installed on the agent too
//!   ([`SecureAgentConfig::apply`] does both); it never sets the
//!   process-global quarantine client.
//! - **[`LabeledMessage`]** wraps a [`Message`](crate::types::Message)
//!   (Rust has no subclassing).
//! - Variable references embedded in a longer string are replaced with the
//!   content's text (a JSON value is serialized as JSON, where upstream
//!   uses Python's `str`).
//!
//! # Not ported
//!
//! - **`SecureMCPToolProxy`** and server-provided IFC labels in MCP
//!   results (`_meta`): this crate cannot depend on `agent-framework-mcp`.
//!   The annotation mapping is ported ([`map_mcp_annotations_to_labels`],
//!   [`apply_mcp_security_labels`]) for declaring MCP tools in a registry.
//! - Pydantic argument re-validation (`_degrade_rewritten_arguments`): Rust
//!   tools deserialize their own arguments.

mod config;
mod labels;
mod mcp;
mod policy;
mod scope;
mod tools;
mod tracking;

pub use config::{SecureAgentConfig, SecureAgentConfigBuilder, DEFAULT_SOURCE_ID};
pub use labels::{
    check_confidentiality_allowed, combine_labels, ConfidentialityLabel, ContentLabel,
    ContentVariableStore, IntegrityLabel, LabeledMessage, Principal, VariableReferenceContent,
    PRINCIPAL_METADATA_KEY,
};
pub use mcp::{apply_mcp_security_labels, map_mcp_annotations_to_labels, mcp_tool_security};
pub use policy::{
    PendingPolicyApproval, PolicyApprovalHandler, PolicyApprovalRequest,
    PolicyEnforcementFunctionMiddleware, PolicyViolation, DEFAULT_MAX_PENDING_APPROVALS,
    DEFAULT_PENDING_APPROVAL_TTL,
};
pub use scope::{
    ScopedVariableStore, SecurityScope, ToolSecurity, ToolSecurityRegistry,
    DEFAULT_SESSION_STATE_KEY,
};
pub use tools::{
    get_quarantine_client, get_security_tools, get_variable_store, set_authoritative_result_label,
    set_quarantine_client, set_variable_store, store_untrusted_content, SECURITY_TOOL_INSTRUCTIONS,
};
pub use tracking::{
    rewritten_arguments, LabelTrackingFunctionMiddleware, ARGUMENT_LABEL_KEY, CONTEXT_LABEL_KEY,
    EFFECTIVE_LABEL_KEY, ORIGINAL_ARGUMENTS_KEY, RESULT_LABEL_KEY, REWRITTEN_ARGUMENTS_KEY,
};
