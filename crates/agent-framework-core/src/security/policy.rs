//! `PolicyEnforcementFunctionMiddleware`: blocks (or puts to approval)
//! tool calls in an untrusted context, with untrusted arguments, or that
//! would leak data to a less confidential destination.

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use super::labels::{
    canonical_principals, combine_labels, ConfidentialityLabel, ContentLabel, IntegrityLabel,
    Principal, PRINCIPAL_METADATA_KEY,
};
use super::scope::{
    is_internal_tool, object_entry, SecurityScope, ToolSecurity, ToolSecurityRegistry,
    STATE_PENDING_APPROVALS,
};
use super::tracking::{
    ARGUMENT_LABEL_KEY, CONTEXT_LABEL_KEY, EFFECTIVE_LABEL_KEY, ORIGINAL_ARGUMENTS_KEY,
    POLICY_TERMINATED_KEY,
};
use crate::error::Result;
use crate::middleware::{FunctionInvocationContext, Middleware, Next};
use crate::session::AgentSession;
use crate::tools::BoxFuture;

/// The default bound on pending approvals per scope.
pub const DEFAULT_MAX_PENDING_APPROVALS: usize = 256;
/// The default lifetime of an unconsumed approval.
pub const DEFAULT_PENDING_APPROVAL_TTL: Duration = Duration::from_secs(3600);

/// One detected policy violation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyViolation {
    /// `untrusted_context`, `untrusted_arguments`,
    /// `max_allowed_confidentiality` or `principal_mismatch`.
    pub violation_type: String,
    /// Why, for the approver.
    pub reason: String,
}

/// What an approver is asked.
#[derive(Debug, Clone)]
pub struct PolicyApprovalRequest {
    /// The approval's id.
    pub approval_id: String,
    /// The tool.
    pub function_name: String,
    /// The arguments as the model sent them (references unexpanded).
    pub arguments: Value,
    /// The conversation's context label.
    pub context_label: ContentLabel,
    /// Every violation, disclosed together.
    pub violations: Vec<PolicyViolation>,
    /// The session, if any.
    pub session_id: Option<String>,
}

/// Decides a policy approval inline: `true` lets the call run.
pub type PolicyApprovalHandler =
    Arc<dyn Fn(PolicyApprovalRequest) -> BoxFuture<bool> + Send + Sync>;

/// A pending (or host-approved, not yet used) policy approval.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingPolicyApproval {
    /// Its id.
    pub approval_id: String,
    /// The tool.
    pub function_name: String,
    /// The disclosed violations.
    pub violations: Vec<PolicyViolation>,
    /// Whether the host approved it.
    pub approved: bool,
    /// When it was requested (seconds since the epoch).
    pub created_at: f64,
}

/// One violation with its block and audit forms.
struct Detected {
    violation_type: &'static str,
    approval_reason: String,
    block_error: String,
    block_type: Option<&'static str>,
    audit: Value,
}

/// Enforces security policy on tool calls, using the labels
/// [`LabelTrackingFunctionMiddleware`](super::LabelTrackingFunctionMiddleware)
/// publishes (install that one first).
///
/// A call violates policy when the context is untrusted, when its arguments
/// resolve to untrusted hidden content (unless the tool is allow-listed or
/// declares `accepts_untrusted`), or when the effective label exceeds the
/// tool's `max_allowed_confidentiality` or, for `USER_IDENTITY` data, its
/// authorized principals. A violation is blocked (the default), allowed
/// with a warning, or, with `approval_on_violation`, put to approval: to an
/// inline [`PolicyApprovalHandler`] when one is set, otherwise recorded as
/// pending (the model is told approval is required) until the host calls
/// [`approve`](Self::approve); the identical call (same tool, arguments,
/// labels, destination, session and violations) then runs once.
pub struct PolicyEnforcementFunctionMiddleware {
    allow_untrusted_tools: HashSet<String>,
    block_on_violation: bool,
    enable_audit_log: bool,
    approval_on_violation: bool,
    max_pending_approvals: usize,
    pending_approval_ttl: Option<Duration>,
    scope: SecurityScope,
    tools: ToolSecurityRegistry,
    approval_handler: Option<PolicyApprovalHandler>,
}

impl std::fmt::Debug for PolicyEnforcementFunctionMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyEnforcementFunctionMiddleware")
            .field("allow_untrusted_tools", &self.allow_untrusted_tools)
            .field("block_on_violation", &self.block_on_violation)
            .field("approval_on_violation", &self.approval_on_violation)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl Default for PolicyEnforcementFunctionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PolicyEnforcementFunctionMiddleware {
    /// Block on violation, with an audit log.
    pub fn new() -> Self {
        Self {
            allow_untrusted_tools: HashSet::new(),
            block_on_violation: true,
            enable_audit_log: true,
            approval_on_violation: false,
            max_pending_approvals: DEFAULT_MAX_PENDING_APPROVALS,
            pending_approval_ttl: Some(DEFAULT_PENDING_APPROVAL_TTL),
            scope: SecurityScope::default(),
            tools: ToolSecurityRegistry::new(),
            approval_handler: None,
        }
    }

    /// Builder: tools allowed to run in an untrusted context and receive
    /// untrusted arguments.
    pub fn allow_untrusted_tools(
        mut self,
        tools: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.allow_untrusted_tools
            .extend(tools.into_iter().map(Into::into));
        self
    }

    /// Builder: block violations (`true`, default) or only warn.
    pub fn block_on_violation(mut self, block: bool) -> Self {
        self.block_on_violation = block;
        self
    }

    /// Builder: keep an audit log of violations (default on).
    pub fn enable_audit_log(mut self, enable: bool) -> Self {
        self.enable_audit_log = enable;
        self
    }

    /// Builder: put violations to approval instead of blocking.
    pub fn approval_on_violation(mut self, approval: bool) -> Self {
        self.approval_on_violation = approval;
        self
    }

    /// Builder: decide approvals inline (implies `approval_on_violation`).
    pub fn approval_handler(mut self, handler: PolicyApprovalHandler) -> Self {
        self.approval_on_violation = true;
        self.approval_handler = Some(handler);
        self
    }

    /// Builder: the bound on pending approvals (oldest evicted first).
    pub fn max_pending_approvals(mut self, max: usize) -> crate::Result<Self> {
        if max == 0 {
            return Err(crate::Error::Configuration(
                "max_pending_approvals must be a positive integer.".into(),
            ));
        }
        self.max_pending_approvals = max;
        Ok(self)
    }

    /// Builder: the lifetime of an unconsumed approval (`None`: no expiry).
    pub fn pending_approval_ttl(mut self, ttl: Option<Duration>) -> crate::Result<Self> {
        if ttl.is_some_and(|t| t.is_zero()) {
            return Err(crate::Error::Configuration(
                "pending_approval_ttl must be positive or None.".into(),
            ));
        }
        self.pending_approval_ttl = ttl;
        Ok(self)
    }

    /// Builder: the state scope (share one with the label tracker).
    pub fn scope(mut self, scope: SecurityScope) -> Self {
        self.scope = scope;
        self
    }

    /// Builder: the tool declarations (share them with the label tracker).
    pub fn tool_registry(mut self, tools: ToolSecurityRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Builder: declare one tool's security properties.
    pub fn tool_security(self, tool: impl Into<String>, security: ToolSecurity) -> Self {
        self.tools.declare(tool, security);
        self
    }

    /// The audit log for `session`.
    pub fn audit_log(&self, session: Option<&AgentSession>) -> Vec<Value> {
        self.scope.audit_log(session)
    }

    /// Clear the audit log for `session`.
    pub fn clear_audit_log(&self, session: Option<&AgentSession>) {
        self.scope.clear_audit_log(session)
    }

    /// The live pending approvals for `session`, oldest first.
    pub fn pending_approvals(&self, session: Option<&AgentSession>) -> Vec<PendingPolicyApproval> {
        self.scope.with_state(session, |s| {
            self.prune(s);
            let mut out: Vec<(u64, PendingPolicyApproval)> = s
                .get(STATE_PENDING_APPROVALS)
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .filter_map(|(id, r)| {
                            Some((
                                r.get("seq")?.as_u64()?,
                                PendingPolicyApproval {
                                    approval_id: id.clone(),
                                    function_name: r.get("function_name")?.as_str()?.into(),
                                    violations: r
                                        .get("violations")?
                                        .as_array()?
                                        .iter()
                                        .filter_map(|v| {
                                            Some(PolicyViolation {
                                                violation_type: v
                                                    .get("violation_type")?
                                                    .as_str()?
                                                    .into(),
                                                reason: v.get("reason")?.as_str()?.into(),
                                            })
                                        })
                                        .collect(),
                                    approved: r.get("approved")?.as_bool()?,
                                    created_at: r.get("created_at")?.as_f64()?,
                                },
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default();
            out.sort_by_key(|(seq, _)| *seq);
            out.into_iter().map(|(_, p)| p).collect()
        })
    }

    /// Approve a pending approval: the identical call runs once. `false`
    /// when no live approval has that id.
    pub fn approve(&self, session: Option<&AgentSession>, approval_id: &str) -> bool {
        self.scope.with_state(session, |s| {
            self.prune(s);
            match object_entry(s, STATE_PENDING_APPROVALS)
                .get_mut(approval_id)
                .and_then(Value::as_object_mut)
            {
                Some(record) => {
                    record.insert("approved".into(), json!(true));
                    true
                }
                None => false,
            }
        })
    }

    /// Reject (discard) a pending approval. `false` when none has that id.
    pub fn reject(&self, session: Option<&AgentSession>, approval_id: &str) -> bool {
        self.scope.with_state(session, |s| {
            object_entry(s, STATE_PENDING_APPROVALS)
                .remove(approval_id)
                .is_some()
        })
    }

    /// Drop malformed and expired records and enforce the bound.
    fn prune(&self, state: &mut Map<String, Value>) {
        let now = now_secs();
        let ttl = self.pending_approval_ttl.map(|t| t.as_secs_f64());
        let pending = object_entry(state, STATE_PENDING_APPROVALS);
        pending.retain(|_, r| {
            let created = r.get("created_at").and_then(Value::as_f64);
            let valid = r.get("binding").is_some_and(Value::is_object)
                && r.get("seq").is_some_and(Value::is_u64);
            match (valid, created) {
                (true, Some(c)) => ttl.is_none_or(|t| now - c < t),
                _ => false,
            }
        });
        while pending.len() > self.max_pending_approvals {
            let oldest = pending
                .iter()
                .min_by_key(|(_, r)| r.get("seq").and_then(Value::as_u64).unwrap_or(0))
                .map(|(k, _)| k.clone());
            match oldest {
                Some(k) => {
                    pending.remove(&k);
                }
                None => break,
            }
        }
    }

    fn log(&self, session: Option<&AgentSession>, audit: &Value) {
        tracing::warn!("security policy violation detected");
        if self.enable_audit_log {
            self.scope.append_audit_entry(session, audit.clone());
        }
    }
}

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// Sort object keys recursively, so the serialized form is canonical
/// whatever map ordering serde_json was built with.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut out = Map::new();
            for k in keys {
                out.insert(k.clone(), canonical(&map[k]));
            }
            Value::Object(out)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

fn canonical_string(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                out.push('{');
                for (i, k) in keys.into_iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push_str(&Value::String(k.clone()).to_string());
                    out.push(':');
                    write(&map[k], out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write(v, out);
                }
                out.push(']');
            }
            other => out.push_str(&other.to_string()),
        }
    }
    let mut out = String::new();
    write(&canonical(value), &mut out);
    out
}

fn signature(name: &str, arguments: &Value) -> String {
    let payload = canonical_string(&json!({"name": name, "arguments": arguments}));
    Sha256::digest(payload.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn principal_key(value: Option<&Value>, source: &str) -> std::result::Result<String, String> {
    let set = canonical_principals(value, source)?;
    Ok(canonical_string(
        &serde_json::to_value(&set).unwrap_or_default(),
    ))
}

fn label_key(label: &ContentLabel) -> std::result::Result<String, String> {
    let principals = if label.confidentiality == ConfidentialityLabel::UserIdentity {
        principal_key(label.metadata.get(PRINCIPAL_METADATA_KEY), "approval label")?
    } else {
        String::new()
    };
    Ok(format!(
        "{}/{}/{}",
        label.integrity, label.confidentiality, principals
    ))
}

fn parse_label(value: Option<&Value>) -> ContentLabel {
    value
        .and_then(|v| ContentLabel::from_value(v).ok())
        .unwrap_or_default()
}

/// The confidentiality check: `Err((failure_type, reason))` on violation.
fn check_confidentiality(
    label: &ContentLabel,
    declaration: &ToolSecurity,
    internal: bool,
) -> std::result::Result<(), (&'static str, String)> {
    let max_allowed = declaration.max_allowed_confidentiality;
    if let Some(max) = max_allowed {
        if label.confidentiality > max {
            return Err((
                "max_allowed_confidentiality",
                format!(
                    "Cannot write {} data to {} destination (data exfiltration blocked)",
                    label.confidentiality.as_str().to_uppercase(),
                    max.as_str().to_uppercase()
                ),
            ));
        }
    }
    if label.confidentiality == ConfidentialityLabel::UserIdentity {
        let invalid = || {
            (
                "principal_mismatch",
                "USER_IDENTITY source or destination principals are missing or invalid".to_string(),
            )
        };
        let source =
            canonical_principals(label.metadata.get(PRINCIPAL_METADATA_KEY), "source label")
                .map_err(|_| invalid())?;
        let destination: BTreeSet<Principal> = if internal {
            source.clone()
        } else {
            if max_allowed != Some(ConfidentialityLabel::UserIdentity) {
                return Err((
                    "principal_mismatch",
                    "USER_IDENTITY destination does not declare an authorized principal set".into(),
                ));
            }
            let declared = serde_json::to_value(&declaration.principals).unwrap_or_default();
            canonical_principals(Some(&declared), "tool").map_err(|_| invalid())?
        };
        if !source.is_subset(&destination) {
            return Err((
                "principal_mismatch",
                "USER_IDENTITY source principals are not authorized for the destination".into(),
            ));
        }
    }
    Ok(())
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for PolicyEnforcementFunctionMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        let name = ctx.function_name.clone();
        let Some(context_label) = ctx
            .metadata
            .get(CONTEXT_LABEL_KEY)
            .and_then(|v| ContentLabel::from_value(v).ok())
        else {
            tracing::warn!(
                tool = %name,
                "no context label; install LabelTrackingFunctionMiddleware before the policy"
            );
            return next.run(ctx).await;
        };
        let argument_label = parse_label(ctx.metadata.get(ARGUMENT_LABEL_KEY));
        let effective = combine_labels([&context_label, &argument_label]);
        ctx.metadata
            .insert(EFFECTIVE_LABEL_KEY.into(), effective.to_value());
        let declaration = self.tools.for_invocation(&ctx);
        let internal = is_internal_tool(&ctx);
        let accepts_untrusted =
            self.allow_untrusted_tools.contains(&name) || declaration.accepts_untrusted;
        let turn = ctx
            .metadata
            .get("turn_number")
            .cloned()
            .unwrap_or(json!(-1));

        let mut violations: Vec<Detected> = Vec::new();
        if context_label.integrity == IntegrityLabel::Untrusted && !accepts_untrusted {
            violations.push(Detected {
                violation_type: "untrusted_context",
                approval_reason: format!(
                    "Tool '{name}' is being called in an UNTRUSTED context. \
                     Approve to proceed despite possible untrusted influence."
                ),
                block_error: "Policy violation: Tool cannot be called in untrusted context".into(),
                block_type: None,
                audit: json!({
                    "type": "untrusted_context",
                    "function": name,
                    "context_label": context_label.to_value(),
                    "turn": turn,
                    "reason": "Context is UNTRUSTED and the tool does not accept untrusted input",
                }),
            });
        }
        if argument_label.integrity == IntegrityLabel::Untrusted && !accepts_untrusted {
            violations.push(Detected {
                violation_type: "untrusted_arguments",
                approval_reason: format!(
                    "Tool '{name}' would receive UNTRUSTED content resolved from a hidden \
                     variable. Approve to forward that content."
                ),
                block_error: "Policy violation: Tool cannot receive untrusted variable content"
                    .into(),
                block_type: Some("untrusted_arguments"),
                audit: json!({
                    "type": "untrusted_arguments",
                    "function": name,
                    "context_label": context_label.to_value(),
                    "argument_label": argument_label.to_value(),
                    "effective_label": effective.to_value(),
                    "turn": turn,
                    "reason": "Arguments resolve to UNTRUSTED content and the tool does not accept untrusted input",
                }),
            });
        }
        if let Err((failure, reason)) = check_confidentiality(&effective, &declaration, internal) {
            violations.push(Detected {
                violation_type: failure,
                approval_reason: format!(
                    "Tool '{name}' violates confidentiality policy: {reason}. Approve to proceed anyway."
                ),
                block_error: format!("Policy violation: {reason}"),
                block_type: Some(failure),
                audit: json!({
                    "type": "confidentiality_violation",
                    "subtype": failure,
                    "function": name,
                    "context_label": context_label.to_value(),
                    "argument_label": argument_label.to_value(),
                    "effective_label": effective.to_value(),
                    "reason": reason,
                    "turn": turn,
                }),
            });
        }
        if violations.is_empty() {
            return next.run(ctx).await;
        }
        let session = ctx.session.clone();
        for v in &violations {
            self.log(session.as_ref(), &v.audit);
        }
        let disclosed = violations
            .iter()
            .map(|v| v.violation_type)
            .collect::<Vec<_>>()
            .join(", ");

        if self.approval_on_violation {
            let binding = match self.binding(
                &ctx,
                &context_label,
                &effective,
                &declaration,
                &violations,
            ) {
                Ok(b) => b,
                Err(_) => {
                    return Ok(terminate(
                        ctx,
                        json!({
                            "error": "Policy violation: approval cannot be safely bound to this invocation",
                            "function": name,
                            "context_label": context_label.to_value(),
                            "violation_type": "unsafe_approval_binding",
                        }),
                    ));
                }
            };
            let public: Vec<PolicyViolation> = violations
                .iter()
                .map(|v| PolicyViolation {
                    violation_type: v.violation_type.into(),
                    reason: v.approval_reason.clone(),
                })
                .collect();
            let approved = if let Some(handler) = &self.approval_handler {
                let request = PolicyApprovalRequest {
                    approval_id: format!("policy_{}", uuid::Uuid::new_v4().simple()),
                    function_name: name.clone(),
                    arguments: original_arguments(&ctx),
                    context_label: context_label.clone(),
                    violations: public.clone(),
                    session_id: session.as_ref().map(|s| s.session_id().to_string()),
                };
                handler(request).await
            } else {
                self.consume_approved(session.as_ref(), &binding)
            };
            if approved {
                tracing::warn!(tool = %name, "APPROVED BY USER: executing despite [{disclosed}]");
                ctx.metadata
                    .insert("user_approved_violation".into(), json!(true));
                return next.run(ctx).await;
            }
            if self.approval_handler.is_some() {
                return Ok(self.block(ctx, &context_label, &violations));
            }
            let approval_id = format!("policy_{}", uuid::Uuid::new_v4().simple());
            self.store_pending(session.as_ref(), &approval_id, &name, &public, binding);
            let primary = &violations[0];
            let mut result = json!({
                "approval_required": true,
                "approval_id": approval_id,
                "policy_violation": true,
                "violation_type": primary.violation_type,
                "reason": if violations.len() == 1 {
                    primary.approval_reason.clone()
                } else {
                    violations.iter().map(|v| v.approval_reason.as_str()).collect::<Vec<_>>().join(" ")
                },
                "function": name,
                "context_label": context_label.to_value(),
            });
            if violations.len() > 1 {
                result["violations"] = json!(public
                    .iter()
                    .map(|v| json!({"violation_type": v.violation_type, "reason": v.reason}))
                    .collect::<Vec<_>>());
            }
            tracing::info!(tool = %name, "policy approval requested for [{disclosed}]");
            return Ok(terminate(ctx, result));
        }
        if self.block_on_violation {
            return Ok(self.block(ctx, &context_label, &violations));
        }
        tracing::warn!(tool = %name, "policy violation(s) [{disclosed}] (allowed)");
        next.run(ctx).await
    }
}

fn original_arguments(ctx: &FunctionInvocationContext) -> Value {
    ctx.metadata
        .get(ORIGINAL_ARGUMENTS_KEY)
        .cloned()
        .unwrap_or_else(|| ctx.arguments.clone())
}

fn terminate(mut ctx: FunctionInvocationContext, result: Value) -> FunctionInvocationContext {
    ctx.result = Some(result);
    ctx.terminate = true;
    ctx.metadata
        .insert(POLICY_TERMINATED_KEY.into(), json!(true));
    ctx
}

impl PolicyEnforcementFunctionMiddleware {
    fn block(
        &self,
        ctx: FunctionInvocationContext,
        context_label: &ContentLabel,
        violations: &[Detected],
    ) -> FunctionInvocationContext {
        let primary = &violations[0];
        let mut result = json!({
            "error": primary.block_error,
            "function": ctx.function_name,
            "context_label": context_label.to_value(),
        });
        if let Some(t) = primary.block_type {
            result["violation_type"] = json!(t);
        }
        if violations.len() > 1 {
            result["violations"] = json!(violations
                .iter()
                .map(|v| v.violation_type)
                .collect::<Vec<_>>());
        }
        terminate(ctx, result)
    }

    /// Every dimension an approval is bound to.
    fn binding(
        &self,
        ctx: &FunctionInvocationContext,
        context_label: &ContentLabel,
        effective: &ContentLabel,
        declaration: &ToolSecurity,
        violations: &[Detected],
    ) -> std::result::Result<Value, String> {
        let destination = if declaration.max_allowed_confidentiality
            == Some(ConfidentialityLabel::UserIdentity)
        {
            let declared = serde_json::to_value(&declaration.principals).unwrap_or_default();
            principal_key(Some(&declared), "approval destination")?
        } else {
            String::new()
        };
        let mut disclosed: Vec<String> = violations
            .iter()
            .map(|v| {
                format!(
                    "{}\u{0}{}",
                    v.violation_type,
                    v.audit.get("reason").and_then(Value::as_str).unwrap_or("")
                )
            })
            .collect();
        disclosed.sort();
        Ok(json!({
            "body_signature": signature(&ctx.function_name, &original_arguments(ctx)),
            "resolved_signature": signature(&ctx.function_name, &json!({"arguments": ctx.arguments})),
            "label_key": label_key(context_label)?,
            "effective_label_key": label_key(effective)?,
            "destination_principal_key": destination,
            "session_key": ctx.session.as_ref().map(|s| s.session_id().to_string()).unwrap_or_default(),
            "disclosed_violations": disclosed,
        }))
    }

    /// Consume a host-approved record bound to exactly this call.
    fn consume_approved(&self, session: Option<&AgentSession>, binding: &Value) -> bool {
        self.scope.with_state(session, |s| {
            self.prune(s);
            let pending = object_entry(s, STATE_PENDING_APPROVALS);
            let found = pending
                .iter()
                .find(|(_, r)| {
                    r.get("approved").and_then(Value::as_bool) == Some(true)
                        && r.get("binding") == Some(binding)
                })
                .map(|(k, _)| k.clone());
            found.is_some_and(|k| pending.remove(&k).is_some())
        })
    }

    fn store_pending(
        &self,
        session: Option<&AgentSession>,
        approval_id: &str,
        function_name: &str,
        violations: &[PolicyViolation],
        binding: Value,
    ) {
        self.scope.with_state(session, |s| {
            let seq = s.get("pending_seq").and_then(Value::as_u64).unwrap_or(0) + 1;
            s.insert("pending_seq".into(), json!(seq));
            object_entry(s, STATE_PENDING_APPROVALS).insert(
                approval_id.to_string(),
                json!({
                    "binding": binding,
                    "function_name": function_name,
                    "violations": violations
                        .iter()
                        .map(|v| json!({"violation_type": v.violation_type, "reason": v.reason}))
                        .collect::<Vec<_>>(),
                    "approved": false,
                    "created_at": now_secs(),
                    "seq": seq,
                }),
            );
            self.prune(s);
        });
    }
}
