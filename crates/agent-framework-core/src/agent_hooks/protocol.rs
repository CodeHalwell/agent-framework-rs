//! The AGENT-HOOKS-0.1 protocol: interception points, agent contexts,
//! verdicts, the interceptor trait, the emitter and the interception record.
//!
//! Upstream depends on the external `agent-hooks` SDK for this layer (Python
//! `agent_hooks`, .NET `ResponsibleAI.AgentHooks`). No Rust crate exists, so
//! the subset the enforcement bundle needs is implemented here, following the
//! SDK's emitter: one `sequential/first_deny` composition profile with
//! `on_approval: "stop"`, no approval resolver, and no identity provider (see
//! the module docs of [`crate::agent_hooks`] for what is left out).

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::FutureExt;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// The spec version this host targets (§4.1 `spec`).
pub const SPEC_VERSION: &str = "agent-hooks/0.1";

/// The spec-recommended per-interceptor timeout (§7): 5 seconds.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);

/// The composition profile this emitter implements (§7.2).
pub const COMPOSITION_PROFILE: &str = "sequential/first_deny";

/// Recommended bound on a context's serialized size (§12.3): 5 MiB.
const MAX_CONTEXT_BYTES: usize = 5 * 1024 * 1024;
/// Recommended bound on a context's nesting depth (§12.3).
const MAX_CONTEXT_DEPTH: usize = 128;
/// §5.3 cap on a verdict's serialized `evidence`.
const MAX_EVIDENCE_BYTES: usize = 10240;
/// §10.3 cap on record messages.
const MAX_RECORD_MESSAGE_BYTES: usize = 256;

/// The reserved `host_error:*` reasons a host synthesizes (§11).
///
/// An interceptor must never return a reason starting with `host_error:`;
/// one that does fails verdict validation.
pub mod host_error {
    /// The prefix every host-synthesized reason starts with.
    pub const PREFIX: &str = "host_error:";
    /// The host could not construct a valid context (or it broke a bound).
    pub const CONTEXT_INVALID: &str = "host_error:context_invalid";
    /// An interceptor returned an error or panicked.
    pub const INTERCEPTOR_FAILED: &str = "host_error:interceptor_failed";
    /// An interceptor exceeded the emitter's timeout.
    pub const INTERCEPTOR_TIMEOUT: &str = "host_error:interceptor_timeout";
    /// An interceptor returned a verdict that fails §5 validation.
    pub const VERDICT_INVALID: &str = "host_error:verdict_invalid";
    /// A transform path did not resolve.
    pub const TRANSFORM_INVALID: &str = "host_error:transform_invalid";
    /// A transform path is not rooted at `$target`, or the point forbids
    /// transforms.
    pub const TRANSFORM_TARGET_FORBIDDEN: &str = "host_error:transform_target_forbidden";
    /// An `enforce`-mode emission with no interceptor registered.
    pub const NO_INTERCEPTOR: &str = "host_error:no_interceptor";
}

/// One of the eight interception points of the control contract (§3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterceptionPoint {
    /// Once, before the first `input` of a session. Target: `agent_init`.
    AgentStartup,
    /// On ingress of each request into the session. Target: `input`.
    Input,
    /// Before each model request is dispatched. Target: `messages`.
    PreModelCall,
    /// After each model response is received. Target: `response`.
    PostModelCall,
    /// Before each tool invocation. Target: `tool_call.args`.
    PreToolCall,
    /// After each tool invocation completes. Target: `tool_result.value`.
    PostToolCall,
    /// Before the final response is returned to the caller. Target: `output`.
    Output,
    /// Once, after the last `output` of a session. Target: `summary`.
    AgentShutdown,
}

impl InterceptionPoint {
    /// The wire name (`"pre_tool_call"`, ...).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AgentStartup => "agent_startup",
            Self::Input => "input",
            Self::PreModelCall => "pre_model_call",
            Self::PostModelCall => "post_model_call",
            Self::PreToolCall => "pre_tool_call",
            Self::PostToolCall => "post_tool_call",
            Self::Output => "output",
            Self::AgentShutdown => "agent_shutdown",
        }
    }

    /// Whether a `transform` verdict is permitted at this point (§3).
    pub fn permits_transform(self) -> bool {
        !matches!(self, Self::AgentStartup | Self::AgentShutdown)
    }

    /// Where the target is mirrored in the context: the conditional field
    /// (and, for the tool points, its member) that `target` aliases (§4.2).
    fn target_field(self) -> (&'static str, Option<&'static str>) {
        match self {
            Self::AgentStartup => ("agent_init", None),
            Self::Input => ("input", None),
            Self::PreModelCall => ("messages", None),
            Self::PostModelCall => ("response", None),
            Self::PreToolCall => ("tool_call", Some("args")),
            Self::PostToolCall => ("tool_result", Some("value")),
            Self::Output => ("output", None),
            Self::AgentShutdown => ("summary", None),
        }
    }
}

impl std::fmt::Display for InterceptionPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Why a session ended (§4.2 `summary.reason`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShutdownReason {
    /// The run completed.
    Completed,
    /// The run failed (including a deny).
    Error,
    /// The run was cancelled.
    Cancelled,
}

impl ShutdownReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Error => "error",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Whether verdicts are enforced or only recorded (§8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnforcementMode {
    /// Honour combined verdicts: denies block, transforms rewrite.
    #[default]
    Enforce,
    /// Invoke interceptors and record their verdicts, but proceed with every
    /// action as if the verdict were `allow`. Transforms are validated, never
    /// applied.
    EvaluateOnly,
}

/// The agent context handed to interceptors (the spec's `AgentContext`, §4).
///
/// Wire-shaped JSON: the required core (`spec`, `interception_point`,
/// `timestamp`, `sequence`, `agent`, `session`, `target`) plus the point's
/// conditional fields. Build one with [`InterceptionContextBuilder`].
/// Each interceptor receives its own copy, so mutating it cannot affect
/// enforcement.
#[derive(Debug, Clone, PartialEq)]
pub struct InterceptionContext {
    point: InterceptionPoint,
    json: Map<String, Value>,
}

impl InterceptionContext {
    /// The interception point.
    pub fn point(&self) -> InterceptionPoint {
        self.point
    }

    /// The session-scoped sequence number.
    pub fn sequence(&self) -> u64 {
        self.json
            .get("sequence")
            .and_then(Value::as_u64)
            .unwrap_or_default()
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        self.json
            .get("session")
            .and_then(|s| s.get("id"))
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The value under evaluation: what a `transform` verdict rewrites.
    pub fn target(&self) -> &Value {
        self.json.get("target").unwrap_or(&Value::Null)
    }

    /// A top-level field of the context (`"tool_call"`, `"model"`, ...).
    pub fn get(&self, field: &str) -> Option<&Value> {
        self.json.get(field)
    }

    /// The whole context as a JSON object.
    pub fn as_json(&self) -> &Map<String, Value> {
        &self.json
    }

    /// Replace `target` and its mirror field (§4.3: the target is a deep
    /// reference into the context, so both must agree).
    fn set_target(&mut self, value: Value) {
        let (field, member) = self.point.target_field();
        match member {
            None => {
                self.json.insert(field.to_string(), value.clone());
            }
            Some(member) => {
                if let Some(Value::Object(inner)) = self.json.get_mut(field) {
                    inner.insert(member.to_string(), value.clone());
                }
            }
        }
        self.json.insert("target".to_string(), value);
    }
}

impl Serialize for InterceptionContext {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.json.serialize(serializer)
    }
}

/// Builds the contexts of one agent-hooks session (one per session).
///
/// Owns the envelope (`agent`, `session`) and the session's `sequence`
/// counter, which is assigned atomically when each context is built
/// (§12.2), so concurrent emissions stay totally ordered.
#[derive(Debug)]
pub struct InterceptionContextBuilder {
    agent: Map<String, Value>,
    session: Map<String, Value>,
    sequence: AtomicU64,
}

impl InterceptionContextBuilder {
    /// A builder for session `session_id` of agent `agent_id` running on
    /// `framework` (lowercase, `^[a-z0-9_-]+$`; contexts with another value
    /// fail closed as `host_error:context_invalid`).
    pub fn new(
        agent_id: impl Into<String>,
        framework: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        let mut agent = Map::new();
        agent.insert("id".into(), Value::String(agent_id.into()));
        agent.insert("framework".into(), Value::String(framework.into()));
        let mut session = Map::new();
        session.insert("id".into(), Value::String(session_id.into()));
        Self {
            agent,
            session,
            sequence: AtomicU64::new(0),
        }
    }

    /// Builder: the agent's human-readable name (§4.5 `agent.name`).
    pub fn with_agent_name(mut self, name: impl Into<String>) -> Self {
        self.agent.insert("name".into(), Value::String(name.into()));
        self
    }

    /// The session id.
    pub fn session_id(&self) -> &str {
        self.session
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// Consume the next sequence number without building a context — for
    /// [`InterceptionEmitter::record_host_failure`], so the failed
    /// emission's record still takes its slot in the session order.
    pub fn next_sequence(&self) -> u64 {
        self.sequence.fetch_add(1, Ordering::SeqCst)
    }

    fn envelope(&self, point: InterceptionPoint, target: Value) -> InterceptionContext {
        let mut json = Map::new();
        json.insert("spec".into(), Value::String(SPEC_VERSION.into()));
        json.insert(
            "interception_point".into(),
            Value::String(point.as_str().into()),
        );
        json.insert("timestamp".into(), Value::String(now_rfc3339()));
        json.insert("sequence".into(), Value::from(self.next_sequence()));
        json.insert("agent".into(), Value::Object(self.agent.clone()));
        json.insert("session".into(), Value::Object(self.session.clone()));
        json.insert("target".into(), target);
        InterceptionContext { point, json }
    }

    fn with_field(
        point: InterceptionPoint,
        mut ctx: InterceptionContext,
        field: &str,
        value: Value,
    ) -> InterceptionContext {
        debug_assert_eq!(ctx.point, point);
        ctx.json.insert(field.to_string(), value);
        ctx
    }

    /// `agent_startup` with the run-start tool names.
    pub fn agent_startup(&self, tools_registered: Vec<String>) -> InterceptionContext {
        let init = serde_json::json!({ "tools_registered": tools_registered });
        let p = InterceptionPoint::AgentStartup;
        Self::with_field(p, self.envelope(p, init.clone()), "agent_init", init)
    }

    /// `input` with the request content and its role (`user`, `system` or
    /// `external`).
    pub fn input(&self, content: Value, role: &str) -> InterceptionContext {
        let input = serde_json::json!({ "content": content, "role": role });
        let p = InterceptionPoint::Input;
        Self::with_field(p, self.envelope(p, input.clone()), "input", input)
    }

    /// `pre_model_call` with the outgoing messages and, optionally, the
    /// call's tools (`{name, description?}` entries).
    pub fn pre_model_call(
        &self,
        model_id: &str,
        messages: Vec<Value>,
        tools: Option<Vec<Value>>,
    ) -> InterceptionContext {
        let p = InterceptionPoint::PreModelCall;
        let messages = Value::Array(messages);
        let mut ctx = Self::with_field(p, self.envelope(p, messages.clone()), "messages", messages);
        ctx.json
            .insert("model".into(), serde_json::json!({ "id": model_id }));
        if let Some(tools) = tools {
            ctx.json.insert("tools".into(), Value::Array(tools));
        }
        ctx
    }

    /// `post_model_call` with the assembled response.
    pub fn post_model_call(
        &self,
        model_id: &str,
        content: Value,
        tool_calls: Vec<Value>,
        finish_reason: &str,
        usage: Option<Value>,
    ) -> InterceptionContext {
        let p = InterceptionPoint::PostModelCall;
        let response = serde_json::json!({
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        });
        let mut ctx = Self::with_field(p, self.envelope(p, response.clone()), "response", response);
        ctx.json
            .insert("model".into(), serde_json::json!({ "id": model_id }));
        if let Some(usage) = usage {
            ctx.json.insert("usage".into(), usage);
        }
        ctx
    }

    /// `pre_tool_call` for call `call_id` of tool `name`.
    pub fn pre_tool_call(
        &self,
        call_id: &str,
        name: &str,
        args: Map<String, Value>,
    ) -> InterceptionContext {
        let p = InterceptionPoint::PreToolCall;
        let args = Value::Object(args);
        let call = serde_json::json!({ "id": call_id, "name": name, "args": args });
        Self::with_field(p, self.envelope(p, args), "tool_call", call)
    }

    /// `post_tool_call` with the arguments actually passed and the result
    /// (or, when `is_error`, a payload-free error description).
    pub fn post_tool_call(
        &self,
        call_id: &str,
        name: &str,
        args: Map<String, Value>,
        value: Value,
        is_error: bool,
    ) -> InterceptionContext {
        let p = InterceptionPoint::PostToolCall;
        let call = serde_json::json!({ "id": call_id, "name": name, "args": args });
        let result = serde_json::json!({ "value": value, "is_error": is_error });
        let mut ctx = Self::with_field(p, self.envelope(p, value), "tool_call", call);
        ctx.json.insert("tool_result".into(), result);
        ctx
    }

    /// `output` with the final response content.
    pub fn output(&self, content: Value) -> InterceptionContext {
        let output = serde_json::json!({ "content": content });
        let p = InterceptionPoint::Output;
        Self::with_field(p, self.envelope(p, output.clone()), "output", output)
    }

    /// `agent_shutdown` with the session's end reason.
    pub fn agent_shutdown(&self, reason: ShutdownReason) -> InterceptionContext {
        let summary = serde_json::json!({ "reason": reason.as_str() });
        let p = InterceptionPoint::AgentShutdown;
        Self::with_field(p, self.envelope(p, summary.clone()), "summary", summary)
    }
}

/// An interceptor's decision (§5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Proceed with the target unchanged.
    Allow,
    /// Do not proceed.
    Deny,
    /// Proceed with the target rewritten.
    Transform,
}

/// A recorded concern that does not change control flow (§5.1).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Warning {
    /// Machine-readable reason.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Human-readable message.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// The rewrite a `transform` verdict carries (§5.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    /// A path rooted at `$target` (`$policy_target` is accepted as a
    /// deprecated alias): dot members, `[index]` and `["member"]` only.
    pub path: String,
    /// The replacement value. An absent value is `null`.
    #[serde(default)]
    pub value: Value,
}

/// An interceptor's verdict (§5).
///
/// Construct with [`Verdict::allow`], [`Verdict::deny`],
/// [`Verdict::transform`], [`Verdict::warn`] or [`Verdict::escalate`]. A
/// verdict that breaks the §5 rules (a `host_error:` reason, a transform on a
/// non-transform decision, oversized evidence, ...) is replaced by a
/// `deny host_error:verdict_invalid`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Verdict {
    /// The decision.
    pub decision: Decision,
    /// A free-form machine identifier. Must not start with `host_error:`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Free-form human-readable text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    /// Warnings; permitted on any decision.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Warning>,
    /// Marks a deny as liftable by an approval seam. This host registers no
    /// approval resolver, so a liftable deny is enforced as a plain deny
    /// (which §9 allows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub approval: Option<Map<String, Value>>,
    /// The rewrite; present iff the decision is `transform`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<Transform>,
    /// An opaque pointer to offline evidence (an object, at most 10 KiB
    /// serialized).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<Value>,
    /// Labels for label-flow tracking (§5.4).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub result_labels: Vec<String>,
}

impl Verdict {
    fn bare(decision: Decision) -> Self {
        Self {
            decision,
            reason: None,
            message: None,
            warnings: Vec::new(),
            approval: None,
            transform: None,
            evidence: None,
            result_labels: Vec::new(),
        }
    }

    /// Proceed unchanged.
    pub fn allow() -> Self {
        Self::bare(Decision::Allow)
    }

    /// Block, with a machine-readable reason.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self {
            reason: Some(reason.into()),
            ..Self::bare(Decision::Deny)
        }
    }

    /// Proceed with the value at `path` (rooted at `$target`) replaced by
    /// `value`.
    pub fn transform(path: impl Into<String>, value: Value) -> Self {
        Self {
            transform: Some(Transform {
                path: path.into(),
                value,
            }),
            ..Self::bare(Decision::Transform)
        }
    }

    /// Allow, recording a warning (the spec has no separate `warn`).
    pub fn warn(reason: impl Into<String>, message: impl Into<String>) -> Self {
        Self::allow().with_warning(Warning {
            reason: Some(reason.into()),
            message: Some(message.into()),
        })
    }

    /// A liftable deny (a deny with an empty `approval` block).
    pub fn escalate(reason: impl Into<String>) -> Self {
        Self {
            approval: Some(Map::new()),
            ..Self::deny(reason)
        }
    }

    /// Builder: set the human-readable message.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }

    /// Builder: add a warning.
    pub fn with_warning(mut self, warning: Warning) -> Self {
        self.warnings.push(warning);
        self
    }

    /// Builder: add a result label.
    pub fn with_result_label(mut self, label: impl Into<String>) -> Self {
        self.result_labels.push(label.into());
        self
    }

    /// Parse a wire verdict (for interceptors that delegate to a remote
    /// service). Validation happens at emission time.
    pub fn from_json(value: Value) -> crate::Result<Self> {
        serde_json::from_value(value).map_err(Into::into)
    }

    /// Whether this verdict lets the action proceed (`allow`/`transform`).
    pub fn is_permit(&self) -> bool {
        self.decision != Decision::Deny
    }

    /// Whether the reason is a host-synthesized `host_error:*` reason.
    pub fn is_host_error(&self) -> bool {
        self.reason
            .as_deref()
            .is_some_and(|r| r.starts_with(host_error::PREFIX))
    }

    /// Check the §5 rules; `Err` carries a payload-free description.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.is_host_error() {
            return Err("reason must not start with host_error:".into());
        }
        if (self.decision == Decision::Transform) != self.transform.is_some() {
            return Err("transform must be present iff decision is transform".into());
        }
        if self.approval.is_some() && self.decision != Decision::Deny {
            return Err("approval is only valid on a deny".into());
        }
        if let Some(evidence) = &self.evidence {
            if !evidence.is_object() {
                return Err("evidence must be an object".into());
            }
            let size = serde_json::to_vec(evidence).map(|v| v.len()).unwrap_or(0);
            if size > MAX_EVIDENCE_BYTES {
                return Err("evidence exceeds 10240 bytes".into());
            }
        }
        Ok(())
    }

    pub(crate) fn host_error(reason: &str, message: Option<String>) -> Self {
        Self {
            reason: Some(reason.to_string()),
            message,
            ..Self::bare(Decision::Deny)
        }
    }

    /// The §10.3 payload-free projection carried on records.
    fn project(&self) -> Self {
        Self {
            decision: self.decision,
            reason: self.reason.clone(),
            message: self.message.as_deref().map(truncate_message),
            warnings: self
                .warnings
                .iter()
                .map(|w| Warning {
                    reason: w.reason.clone(),
                    message: w.message.as_deref().map(truncate_message),
                })
                .collect(),
            approval: self.approval.as_ref().map(|_| Map::new()),
            transform: self.transform.as_ref().map(|t| Transform {
                path: t.path.clone(),
                value: Value::Null,
            }),
            evidence: self.evidence.clone(),
            result_labels: self.result_labels.clone(),
        }
    }
}

/// Something that inspects an [`InterceptionContext`] and returns a
/// [`Verdict`] (§7).
///
/// Returning `Err` (or panicking, or exceeding the emitter's timeout) fails
/// the emission closed as a `host_error:*` deny. Interceptors may be invoked
/// concurrently across emissions (parallel tool calls); a stateful one owns
/// its own synchronization.
#[async_trait]
pub trait Interceptor: Send + Sync {
    /// Decide on one emission. `context` is this interceptor's own copy.
    async fn intercept(&self, context: InterceptionContext) -> crate::Result<Verdict>;
}

/// An [`Interceptor`] from an async closure; see [`interceptor_fn`].
pub struct FnInterceptor<F>(F);

/// Build an [`Interceptor`] from an async closure.
pub fn interceptor_fn<F, Fut>(f: F) -> FnInterceptor<F>
where
    F: Fn(InterceptionContext) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = crate::Result<Verdict>> + Send,
{
    FnInterceptor(f)
}

#[async_trait]
impl<F, Fut> Interceptor for FnInterceptor<F>
where
    F: Fn(InterceptionContext) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = crate::Result<Verdict>> + Send,
{
    async fn intercept(&self, context: InterceptionContext) -> crate::Result<Verdict> {
        (self.0)(context).await
    }
}

/// The composition block of a record (§10.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CompositionRecord {
    /// Always `sequential/first_deny`.
    pub profile: &'static str,
    /// Always `stop` (the normative default).
    pub on_approval: &'static str,
}

/// A payload-free per-interceptor verdict summary (§10.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct VerdictSummary {
    /// Registration index.
    pub index: usize,
    /// The interceptor's (or its substituted failure's) decision.
    pub decision: Decision,
    /// Its reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The registration name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The payload-free record of one emission (§10.3).
///
/// This host declares no identity provider, so `input_identity`,
/// `enforced_identity` and `identity_provider` are always `null` — the
/// record self-describes as identity-unbound.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct InterceptionRecord {
    /// The interception point.
    pub interception_point: InterceptionPoint,
    /// The enforcement mode in effect.
    pub mode: EnforcementMode,
    /// The payload-free projection of the combined verdict.
    pub verdict: Verdict,
    /// Always `None` (no identity provider).
    pub input_identity: Option<String>,
    /// Always `None` (no identity provider).
    pub enforced_identity: Option<String>,
    /// Always `None` (no identity provider).
    pub identity_provider: Option<String>,
    /// The session id (`""` when unknown).
    pub session_id: String,
    /// The emission's sequence number (`-1` when unknown).
    pub sequence: i64,
    /// The context's timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    /// Index of the interceptor whose verdict won, if any.
    pub decided_by: Option<usize>,
    /// The composition profile in effect.
    pub composition: CompositionRecord,
    /// Per-interceptor summaries, in registration order.
    pub verdicts: Vec<VerdictSummary>,
    /// Whether registered interceptors were left uninvoked.
    pub fold_truncated: bool,
    /// Number of interceptors registered.
    pub interceptors_registered: usize,
}

impl InterceptionRecord {
    /// Whether the guarded action may proceed: always in `evaluate_only`,
    /// otherwise iff the combined verdict is a permit.
    pub fn proceeds(&self) -> bool {
        self.mode == EnforcementMode::EvaluateOnly || self.verdict.is_permit()
    }
}

/// An emission whose combined verdict blocks the guarded action.
///
/// Surfaces from an enforced agent run as
/// [`Error::InterceptionBlocked`](crate::Error::InterceptionBlocked).
#[derive(Debug, Clone, PartialEq)]
pub struct InterceptionBlocked {
    /// The combined verdict (in process; may carry interceptor text).
    pub verdict: Verdict,
    /// The payload-free record of the emission.
    pub record: InterceptionRecord,
}

impl InterceptionBlocked {
    /// The point that blocked.
    pub fn point(&self) -> InterceptionPoint {
        self.record.interception_point
    }

    /// The deny's reason, if any.
    pub fn reason(&self) -> Option<&str> {
        self.verdict.reason.as_deref()
    }

    /// Whether the block was host-synthesized (`host_error:*`): the
    /// enforcement layer itself failed rather than a policy denying.
    pub fn is_host_error(&self) -> bool {
        self.verdict.is_host_error()
    }
}

impl std::fmt::Display for InterceptionBlocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "blocked by agent-hooks at {} ({})",
            self.point(),
            self.reason().unwrap_or("deny")
        )
    }
}

impl std::error::Error for InterceptionBlocked {}

/// A proceeding emission: the record plus the effective (post-composition)
/// target the guarded action must consume (§4.3).
#[derive(Debug, Clone, PartialEq)]
pub struct EmitOutcome {
    /// The combined verdict.
    pub verdict: Verdict,
    /// The target after any applied transforms.
    pub target: Value,
    /// The payload-free record.
    pub record: InterceptionRecord,
}

/// A callback receiving every interception record.
pub type RecordSink = Arc<dyn Fn(&InterceptionRecord) + Send + Sync>;

/// One registered interceptor and its optional, payload-free name.
type Registration = (Option<String>, Arc<dyn Interceptor>);

/// Dispatches contexts to interceptors and composes their verdicts
/// (`sequential/first_deny`), producing a record per emission (§6–§10).
///
/// Fail-closed throughout: zero interceptors in `enforce` mode, an
/// interceptor error, panic or timeout, an invalid verdict, an unappliable
/// transform and an out-of-bounds context all become `host_error:*` denies.
pub struct InterceptionEmitter {
    interceptors: Vec<Registration>,
    mode: EnforcementMode,
    timeout: Option<Duration>,
    record_sink: Option<RecordSink>,
}

impl Default for InterceptionEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for InterceptionEmitter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InterceptionEmitter")
            .field("interceptors", &self.interceptors.len())
            .field("mode", &self.mode)
            .field("timeout", &self.timeout)
            .field("record_sink", &self.record_sink.is_some())
            .finish()
    }
}

/// The internal outcome of one dispatch.
struct Dispatch {
    combined: Verdict,
    decided_by: Option<usize>,
    verdicts: Vec<VerdictSummary>,
    fold_truncated: bool,
}

impl Dispatch {
    fn synthesized(reason: &str, message: Option<String>) -> Self {
        Self {
            combined: Verdict::host_error(reason, message),
            decided_by: None,
            verdicts: Vec::new(),
            fold_truncated: false,
        }
    }
}

impl InterceptionEmitter {
    /// An `enforce`-mode emitter with the default 5-second timeout and no
    /// interceptors.
    pub fn new() -> Self {
        Self {
            interceptors: Vec::new(),
            mode: EnforcementMode::Enforce,
            timeout: Some(DEFAULT_TIMEOUT),
            record_sink: None,
        }
    }

    /// Builder: the enforcement mode.
    pub fn with_mode(mut self, mode: EnforcementMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: the per-interceptor timeout (`None` disables it).
    pub fn with_timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: a callback receiving every record. It runs synchronously
    /// after each emission; a panicking sink is contained.
    pub fn with_record_sink(mut self, sink: RecordSink) -> Self {
        self.record_sink = Some(sink);
        self
    }

    /// Builder: register an interceptor, optionally with a payload-free name
    /// recorded on the record's verdict summaries.
    pub fn register(mut self, interceptor: Arc<dyn Interceptor>, name: Option<String>) -> Self {
        self.interceptors.push((name, interceptor));
        self
    }

    /// The enforcement mode.
    pub fn mode(&self) -> EnforcementMode {
        self.mode
    }

    /// Run one emission; `Err` when the guarded action must not proceed.
    ///
    /// The block is boxed because it carries the full record.
    pub async fn emit(
        &self,
        context: InterceptionContext,
    ) -> std::result::Result<EmitOutcome, Box<InterceptionBlocked>> {
        let outcome = self.emit_unchecked(context).await;
        if outcome.record.proceeds() {
            Ok(outcome)
        } else {
            Err(Box::new(InterceptionBlocked {
                verdict: outcome.verdict,
                record: outcome.record,
            }))
        }
    }

    /// Run one emission and return its outcome even when it blocks. The
    /// caller must check [`InterceptionRecord::proceeds`]; prefer
    /// [`InterceptionEmitter::emit`].
    pub async fn emit_unchecked(&self, mut context: InterceptionContext) -> EmitOutcome {
        let original_target = context.target().clone();
        let dispatch = match check_envelope(&context) {
            Err(message) => Dispatch::synthesized(host_error::CONTEXT_INVALID, Some(message)),
            Ok(()) => self.dispatch(&mut context).await,
        };
        let target = if self.mode == EnforcementMode::Enforce {
            context.target().clone()
        } else {
            original_target
        };
        let record = InterceptionRecord {
            interception_point: context.point,
            mode: self.mode,
            verdict: dispatch.combined.project(),
            input_identity: None,
            enforced_identity: None,
            identity_provider: None,
            session_id: context.session_id().to_string(),
            sequence: context.sequence() as i64,
            timestamp: context
                .json
                .get("timestamp")
                .and_then(Value::as_str)
                .map(str::to_string),
            decided_by: dispatch.decided_by,
            composition: composition_record(),
            verdicts: dispatch.verdicts,
            fold_truncated: dispatch.fold_truncated,
            interceptors_registered: self.interceptors.len(),
        };
        self.deliver(&record);
        EmitOutcome {
            verdict: dispatch.combined,
            target,
            record,
        }
    }

    /// Produce (and deliver) the §10.3 record for an emission whose context
    /// the host could not construct at all, e.g. a projection fault. The
    /// record is a `deny host_error:context_invalid` with a payload-free
    /// `detail`. The caller must still fail the guarded action closed.
    pub fn record_host_failure(
        &self,
        point: InterceptionPoint,
        detail: impl Into<String>,
        session_id: &str,
        sequence: Option<u64>,
    ) -> InterceptionRecord {
        let verdict = Verdict::host_error(host_error::CONTEXT_INVALID, Some(detail.into()));
        let record = InterceptionRecord {
            interception_point: point,
            mode: self.mode,
            verdict: verdict.project(),
            input_identity: None,
            enforced_identity: None,
            identity_provider: None,
            session_id: session_id.to_string(),
            sequence: sequence.map(|s| s as i64).unwrap_or(-1),
            timestamp: Some(now_rfc3339()),
            decided_by: None,
            composition: composition_record(),
            verdicts: Vec::new(),
            fold_truncated: false,
            interceptors_registered: self.interceptors.len(),
        };
        self.deliver(&record);
        record
    }

    fn deliver(&self, record: &InterceptionRecord) {
        if let Some(sink) = &self.record_sink {
            // Audit delivery must not take down the control plane: the
            // outcome is already decided.
            let _ = std::panic::catch_unwind(AssertUnwindSafe(|| sink(record)));
        }
    }

    /// `sequential/first_deny` (§7.4), `on_approval: stop`, no resolver.
    async fn dispatch(&self, context: &mut InterceptionContext) -> Dispatch {
        if self.interceptors.is_empty() {
            if self.mode == EnforcementMode::Enforce {
                return Dispatch::synthesized(host_error::NO_INTERCEPTOR, None);
            }
            return Dispatch {
                combined: Verdict::allow(),
                decided_by: None,
                verdicts: Vec::new(),
                fold_truncated: false,
            };
        }
        let n = self.interceptors.len();
        let mut summaries = Vec::with_capacity(n);
        let mut pool: Vec<Verdict> = Vec::with_capacity(n);
        let mut last_transform: Option<(usize, Verdict)> = None;
        for (i, (name, interceptor)) in self.interceptors.iter().enumerate() {
            let verdict = self.invoke(interceptor.as_ref(), context.clone()).await;
            summaries.push(VerdictSummary {
                index: i,
                decision: verdict.decision,
                reason: verdict.reason.clone(),
                name: name.clone(),
            });
            let truncated = i + 1 < n;
            pool.push(verdict.clone());
            match verdict.decision {
                // A deny (liftable or not: no resolver is registered, so a
                // liftable deny stands, §9) short-circuits. A host-synthesized
                // failure deny takes the failing interceptor's slot.
                Decision::Deny => {
                    return Dispatch {
                        combined: with_unions(verdict, &pool),
                        decided_by: Some(i),
                        verdicts: summaries,
                        fold_truncated: truncated,
                    };
                }
                Decision::Transform => {
                    if let Err(reason) = self.fold(context, &verdict) {
                        return Dispatch {
                            combined: with_unions(Verdict::host_error(reason, None), &pool),
                            decided_by: None,
                            verdicts: summaries,
                            fold_truncated: truncated,
                        };
                    }
                    last_transform = Some((i, verdict));
                }
                Decision::Allow => {}
            }
        }
        let (decided_by, combined) = match last_transform {
            Some((i, v)) => (Some(i), v),
            None => (None, Verdict::allow()),
        };
        Dispatch {
            combined: with_unions(combined, &pool),
            decided_by,
            verdicts: summaries,
            fold_truncated: false,
        }
    }

    /// Apply a transform before the next interceptor runs (§7.4). In
    /// `evaluate_only` the transform is validated against a scratch copy
    /// and never applied.
    fn fold(
        &self,
        context: &mut InterceptionContext,
        verdict: &Verdict,
    ) -> std::result::Result<(), &'static str> {
        if !context.point.permits_transform() {
            return Err(host_error::TRANSFORM_TARGET_FORBIDDEN);
        }
        let transform = verdict
            .transform
            .as_ref()
            .ok_or(host_error::VERDICT_INVALID)?;
        let mut target = context.target().clone();
        apply_transform(&mut target, &transform.path, transform.value.clone())?;
        if self.mode == EnforcementMode::Enforce {
            context.set_target(target);
        }
        Ok(())
    }

    /// Invoke one interceptor on its own copy and normalize every failure to
    /// the §6.3 deny.
    async fn invoke(&self, interceptor: &dyn Interceptor, context: InterceptionContext) -> Verdict {
        let call = AssertUnwindSafe(interceptor.intercept(context)).catch_unwind();
        let result = match self.timeout {
            Some(timeout) => match tokio::time::timeout(timeout, call).await {
                Ok(r) => r,
                Err(_) => return Verdict::host_error(host_error::INTERCEPTOR_TIMEOUT, None),
            },
            None => call.await,
        };
        match result {
            Err(_panic) => {
                Verdict::host_error(host_error::INTERCEPTOR_FAILED, Some("panic".into()))
            }
            // Only a type-level description crosses into the record (§14): an
            // error's text may carry target content.
            Ok(Err(_)) => Verdict::host_error(host_error::INTERCEPTOR_FAILED, Some("error".into())),
            Ok(Ok(verdict)) => match verdict.validate() {
                Ok(()) => verdict,
                Err(why) => Verdict::host_error(host_error::VERDICT_INVALID, Some(why)),
            },
        }
    }
}

fn composition_record() -> CompositionRecord {
    CompositionRecord {
        profile: COMPOSITION_PROFILE,
        on_approval: "stop",
    }
}

/// §7.3 metadata unions: warnings from every verdict (first seen, deduped),
/// result labels from every permit verdict, kept only when the combined
/// verdict permits.
fn with_unions(mut combined: Verdict, pool: &[Verdict]) -> Verdict {
    let mut warnings: Vec<Warning> = Vec::new();
    for w in pool.iter().flat_map(|v| v.warnings.iter()) {
        if !warnings.contains(w) {
            warnings.push(w.clone());
        }
    }
    combined.warnings = warnings;
    let mut labels: Vec<String> = Vec::new();
    if combined.is_permit() {
        for l in pool
            .iter()
            .filter(|v| v.is_permit())
            .flat_map(|v| v.result_labels.iter())
        {
            if !labels.contains(l) {
                labels.push(l.clone());
            }
        }
    }
    combined.result_labels = labels;
    combined
}

/// Validate the §4 envelope and the §12.3 bounds before any interceptor
/// runs. `Err` carries a payload-free message.
fn check_envelope(context: &InterceptionContext) -> std::result::Result<(), String> {
    let json = &context.json;
    if json.get("spec").and_then(Value::as_str) != Some(SPEC_VERSION) {
        return Err("spec is missing or unsupported".into());
    }
    let framework = json
        .get("agent")
        .and_then(|a| a.get("framework"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if framework.is_empty()
        || !framework
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
    {
        return Err("agent.framework must match ^[a-z0-9_-]+$".into());
    }
    if json
        .get("agent")
        .and_then(|a| a.get("id"))
        .and_then(Value::as_str)
        .is_none()
    {
        return Err("agent.id is missing".into());
    }
    if json
        .get("session")
        .and_then(|s| s.get("id"))
        .and_then(Value::as_str)
        .is_none()
    {
        return Err("session.id is missing".into());
    }
    let (field, _) = context.point.target_field();
    if !json.contains_key(field) || !json.contains_key("target") {
        return Err(format!("{field} is missing"));
    }
    if depth_exceeds(&Value::Object(json.clone()), MAX_CONTEXT_DEPTH) {
        return Err("context nesting depth exceeds 128".into());
    }
    match serde_json::to_vec(json) {
        Ok(bytes) if bytes.len() <= MAX_CONTEXT_BYTES => Ok(()),
        Ok(_) => Err("context exceeds 5 MiB".into()),
        Err(_) => Err("context is not serializable".into()),
    }
}

/// Whether `value` nests deeper than `limit` (iterative, so a hostile value
/// cannot overflow the stack here).
fn depth_exceeds(value: &Value, limit: usize) -> bool {
    let mut stack: Vec<(&Value, usize)> = vec![(value, 1)];
    while let Some((v, depth)) = stack.pop() {
        let children: Box<dyn Iterator<Item = &Value>> = match v {
            Value::Array(items) => Box::new(items.iter()),
            Value::Object(map) => Box::new(map.values()),
            _ => continue,
        };
        if depth > limit {
            return true;
        }
        stack.extend(children.map(|c| (c, depth + 1)));
    }
    false
}

/// One parsed transform path segment.
#[derive(Debug, PartialEq)]
enum Segment {
    Member(String),
    Index(usize),
}

fn is_member_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'-'
}

/// Parse a §5.2 path (`$target` root; `.member`, `[index]`, `["member"]`).
fn parse_path(path: &str) -> std::result::Result<Vec<Segment>, &'static str> {
    let rest = path
        .strip_prefix("$policy_target")
        .or_else(|| path.strip_prefix("$target"))
        .ok_or(host_error::TRANSFORM_TARGET_FORBIDDEN)?;
    let bytes = rest.as_bytes();
    let mut i = 0;
    let mut segments = Vec::new();
    let member = |start: usize| {
        let mut end = start;
        while end < bytes.len() && is_member_byte(bytes[end]) {
            end += 1;
        }
        end
    };
    while i < bytes.len() {
        match bytes[i] {
            b'.' => {
                let end = member(i + 1);
                if end == i + 1 {
                    return Err(host_error::TRANSFORM_INVALID);
                }
                segments.push(Segment::Member(rest[i + 1..end].to_string()));
                i = end;
            }
            b'[' if bytes.get(i + 1) == Some(&b'"') => {
                let end = member(i + 2);
                if end == i + 2
                    || bytes.get(end) != Some(&b'"')
                    || bytes.get(end + 1) != Some(&b']')
                {
                    return Err(host_error::TRANSFORM_INVALID);
                }
                segments.push(Segment::Member(rest[i + 2..end].to_string()));
                i = end + 2;
            }
            b'[' => {
                let mut end = i + 1;
                while end < bytes.len() && bytes[end].is_ascii_digit() {
                    end += 1;
                }
                if end == i + 1 || bytes.get(end) != Some(&b']') {
                    return Err(host_error::TRANSFORM_INVALID);
                }
                let index = rest[i + 1..end]
                    .parse()
                    .map_err(|_| host_error::TRANSFORM_INVALID)?;
                segments.push(Segment::Index(index));
                i = end + 1;
            }
            _ => return Err(host_error::TRANSFORM_INVALID),
        }
    }
    // A root token followed directly by member bytes (`$targetx`) is not the
    // `$target` root.
    Ok(segments)
}

/// Replace the location `path` addresses within `target` with `value`
/// (§5.2). Every segment, including the last, must resolve: a transform
/// replaces, it never creates.
pub(crate) fn apply_transform(
    target: &mut Value,
    path: &str,
    value: Value,
) -> std::result::Result<(), &'static str> {
    let segments = parse_path(path)?;
    let mut cursor = target;
    for segment in &segments {
        cursor = match (segment, cursor) {
            (Segment::Member(name), Value::Object(map)) => map
                .get_mut(name.as_str())
                .ok_or(host_error::TRANSFORM_INVALID)?,
            (Segment::Index(index), Value::Array(items)) => {
                items.get_mut(*index).ok_or(host_error::TRANSFORM_INVALID)?
            }
            _ => return Err(host_error::TRANSFORM_INVALID),
        };
    }
    *cursor = value;
    Ok(())
}

fn truncate_message(message: &str) -> String {
    if message.len() <= MAX_RECORD_MESSAGE_BYTES {
        return message.to_string();
    }
    // Leave room for the 3-byte ellipsis within the cap.
    let mut end = MAX_RECORD_MESSAGE_BYTES - '…'.len_utf8();
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &message[..end])
}

/// The current UTC time as RFC 3339 with millisecond precision.
pub(crate) fn now_rfc3339() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_rfc3339(now.as_secs() as i64, now.subsec_millis())
}

fn format_rfc3339(secs: i64, millis: u32) -> String {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rfc3339_formats_known_instants() {
        assert_eq!(format_rfc3339(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(
            format_rfc3339(1_782_303_791, 123),
            "2026-06-24T12:23:11.123Z"
        );
        assert_eq!(format_rfc3339(951_782_400, 7), "2000-02-29T00:00:00.007Z");
    }

    #[test]
    fn path_grammar() {
        assert_eq!(parse_path("$target"), Ok(vec![]));
        assert_eq!(
            parse_path("$target.a[0][\"b-c\"]"),
            Ok(vec![
                Segment::Member("a".into()),
                Segment::Index(0),
                Segment::Member("b-c".into())
            ])
        );
        assert_eq!(
            parse_path("$policy_target.x"),
            Ok(vec![Segment::Member("x".into())])
        );
        assert_eq!(
            parse_path("$.x"),
            Err(host_error::TRANSFORM_TARGET_FORBIDDEN)
        );
        assert_eq!(
            parse_path("$input.x"),
            Err(host_error::TRANSFORM_TARGET_FORBIDDEN)
        );
        for bad in [
            "$targetx",
            "$target.",
            "$target[]",
            "$target[\"a\"",
            "$target[a]",
            "$target.a b",
        ] {
            assert_eq!(parse_path(bad), Err(host_error::TRANSFORM_INVALID), "{bad}");
        }
    }

    #[test]
    fn transforms_replace_and_never_create() {
        let mut t = json!({"a": [1, {"b": 2}]});
        apply_transform(&mut t, "$target.a[1].b", json!(3)).unwrap();
        assert_eq!(t, json!({"a": [1, {"b": 3}]}));
        apply_transform(&mut t, "$target", json!("whole")).unwrap();
        assert_eq!(t, json!("whole"));
        let mut t = json!({"a": [1]});
        assert_eq!(
            apply_transform(&mut t, "$target.missing", json!(1)),
            Err(host_error::TRANSFORM_INVALID)
        );
        assert_eq!(
            apply_transform(&mut t, "$target.a[5]", json!(1)),
            Err(host_error::TRANSFORM_INVALID)
        );
        assert_eq!(
            apply_transform(&mut t, "$target.a.b", json!(1)),
            Err(host_error::TRANSFORM_INVALID)
        );
        assert_eq!(t, json!({"a": [1]}));
    }

    #[test]
    fn verdict_validation() {
        assert!(Verdict::allow().validate().is_ok());
        assert!(Verdict::escalate("x").validate().is_ok());
        assert!(Verdict::deny("host_error:no_interceptor")
            .validate()
            .is_err());
        let mut v = Verdict::allow();
        v.transform = Some(Transform {
            path: "$target".into(),
            value: Value::Null,
        });
        assert!(v.validate().is_err());
        let mut v = Verdict::allow();
        v.approval = Some(Map::new());
        assert!(v.validate().is_err());
        let mut v = Verdict::deny("x");
        v.evidence = Some(json!("not an object"));
        assert!(v.validate().is_err());
        v.evidence = Some(json!({"artefact": "x".repeat(20_000)}));
        assert!(v.validate().is_err());
        let mut v = Verdict::transform("$target", json!(1));
        v.decision = Decision::Allow;
        assert!(v.validate().is_err());
    }

    #[test]
    fn wire_verdicts_parse() {
        let v = Verdict::from_json(json!({
            "decision": "transform",
            "transform": {"path": "$target.content"}
        }))
        .unwrap();
        assert_eq!(v.decision, Decision::Transform);
        assert_eq!(v.transform.unwrap().value, Value::Null);
        assert!(Verdict::from_json(json!({"reason": "no decision"})).is_err());
    }

    #[test]
    fn record_projection_drops_payload() {
        let mut v = Verdict::transform("$target.secret", json!("redacted value"))
            .with_message("é".repeat(300));
        v.warnings.push(Warning {
            reason: Some("w".into()),
            message: Some("m".into()),
        });
        let p = v.project();
        assert_eq!(p.transform.unwrap().value, Value::Null);
        let message = p.message.unwrap();
        assert!(message.len() <= 256);
        assert!(message.ends_with('…'));
        let mut e = Verdict::escalate("x");
        e.approval
            .as_mut()
            .unwrap()
            .insert("ticket".into(), json!("t"));
        assert_eq!(e.project().approval, Some(Map::new()));
    }

    #[test]
    fn depth_bound() {
        let mut v = json!(1);
        for _ in 0..200 {
            v = json!([v]);
        }
        assert!(depth_exceeds(&v, 128));
        assert!(!depth_exceeds(&json!({"a": [1, 2, {"b": 3}]}), 128));
    }

    fn builder() -> InterceptionContextBuilder {
        InterceptionContextBuilder::new("agent-1", "agent-framework", "s1")
    }

    #[tokio::test]
    async fn zero_interceptors_fail_closed_in_enforce_mode() {
        let emitter = InterceptionEmitter::new();
        let blocked = emitter
            .emit(builder().output(json!("hi")))
            .await
            .unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::NO_INTERCEPTOR));
        let observe = InterceptionEmitter::new().with_mode(EnforcementMode::EvaluateOnly);
        assert!(observe.emit(builder().output(json!("hi"))).await.is_ok());
    }

    #[tokio::test]
    async fn sequence_is_strictly_increasing() {
        let b = builder();
        let a = b.agent_startup(vec![]);
        let c = b.input(json!("x"), "user");
        assert_eq!(a.sequence(), 0);
        assert_eq!(c.sequence(), 1);
        assert_eq!(b.next_sequence(), 2);
        assert_eq!(c.session_id(), "s1");
    }

    #[tokio::test]
    async fn invalid_framework_is_context_invalid() {
        let emitter = InterceptionEmitter::new().register(
            Arc::new(interceptor_fn(|_| async { Ok(Verdict::allow()) })),
            None,
        );
        let b = InterceptionContextBuilder::new("a", "Agent Framework", "s");
        let blocked = emitter.emit(b.output(json!("x"))).await.unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::CONTEXT_INVALID));
        assert!(blocked.record.verdicts.is_empty());
    }

    #[tokio::test]
    async fn sequential_fold_and_first_deny() {
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let seen2 = seen.clone();
        let emitter = InterceptionEmitter::new()
            .register(
                Arc::new(interceptor_fn(|_| async {
                    Ok(Verdict::transform("$target.content", json!("[redacted]"))
                        .with_result_label("pii"))
                })),
                Some("redactor".into()),
            )
            .register(
                Arc::new(interceptor_fn(move |ctx: InterceptionContext| {
                    let seen = seen2.clone();
                    async move {
                        seen.lock().unwrap().push(ctx.target().clone());
                        // The mirror field is folded too.
                        seen.lock()
                            .unwrap()
                            .push(ctx.get("output").cloned().unwrap());
                        Ok(Verdict::warn("w", "careful"))
                    }
                })),
                None,
            );
        let out = emitter
            .emit(builder().output(json!("secret")))
            .await
            .unwrap();
        assert_eq!(out.target, json!({"content": "[redacted]"}));
        assert_eq!(out.verdict.decision, Decision::Transform);
        assert_eq!(out.record.decided_by, Some(0));
        assert_eq!(out.verdict.result_labels, vec!["pii".to_string()]);
        assert_eq!(out.verdict.warnings.len(), 1);
        assert_eq!(out.record.verdicts[0].name.as_deref(), Some("redactor"));
        assert_eq!(
            *seen.lock().unwrap(),
            vec![
                json!({"content": "[redacted]"}),
                json!({"content": "[redacted]"})
            ]
        );

        let called = Arc::new(AtomicU64::new(0));
        let called2 = called.clone();
        let emitter = InterceptionEmitter::new()
            .register(
                Arc::new(interceptor_fn(|_| async {
                    Ok(Verdict::escalate("needs_review").with_result_label("x"))
                })),
                None,
            )
            .register(
                Arc::new(interceptor_fn(move |_| {
                    called2.fetch_add(1, Ordering::SeqCst);
                    async { Ok(Verdict::allow()) }
                })),
                None,
            );
        let blocked = emitter
            .emit(builder().output(json!("x")))
            .await
            .unwrap_err();
        assert_eq!(blocked.reason(), Some("needs_review"));
        assert!(blocked.record.fold_truncated);
        assert!(blocked.verdict.result_labels.is_empty());
        assert_eq!(called.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn interceptor_failures_map_to_host_errors() {
        struct Panics;
        #[async_trait]
        impl Interceptor for Panics {
            async fn intercept(&self, _: InterceptionContext) -> crate::Result<Verdict> {
                panic!("boom")
            }
        }
        let cases: Vec<(Arc<dyn Interceptor>, &str)> = vec![
            (
                Arc::new(interceptor_fn(|_| async {
                    Err(crate::Error::other("secret text"))
                })),
                host_error::INTERCEPTOR_FAILED,
            ),
            (Arc::new(Panics), host_error::INTERCEPTOR_FAILED),
            (
                Arc::new(interceptor_fn(|_| async {
                    Ok(Verdict::deny("host_error:spoofed"))
                })),
                host_error::VERDICT_INVALID,
            ),
            (
                Arc::new(interceptor_fn(|_| async {
                    Ok(Verdict::transform("$target.nope", json!(1)))
                })),
                host_error::TRANSFORM_INVALID,
            ),
            (
                Arc::new(interceptor_fn(|_| async {
                    Ok(Verdict::transform("$.content", json!(1)))
                })),
                host_error::TRANSFORM_TARGET_FORBIDDEN,
            ),
        ];
        for (interceptor, reason) in cases {
            let records = Arc::new(std::sync::Mutex::new(Vec::new()));
            let sink_records = records.clone();
            let emitter = InterceptionEmitter::new()
                .register(interceptor, None)
                .with_record_sink(Arc::new(move |r: &InterceptionRecord| {
                    sink_records.lock().unwrap().push(r.clone())
                }));
            let blocked = emitter
                .emit(builder().output(json!("x")))
                .await
                .unwrap_err();
            assert_eq!(blocked.reason(), Some(reason));
            let records = records.lock().unwrap();
            assert_eq!(records.len(), 1);
            let wire = serde_json::to_string(&records[0]).unwrap();
            assert!(!wire.contains("secret text"), "{wire}");
            assert!(wire.contains("\"identity_provider\":null"));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn slow_interceptors_time_out() {
        let emitter = InterceptionEmitter::new().register(
            Arc::new(interceptor_fn(|_| async {
                tokio::time::sleep(Duration::from_secs(60)).await;
                Ok(Verdict::allow())
            })),
            None,
        );
        let blocked = emitter
            .emit(builder().output(json!("x")))
            .await
            .unwrap_err();
        assert_eq!(blocked.reason(), Some(host_error::INTERCEPTOR_TIMEOUT));
    }

    #[tokio::test]
    async fn session_points_forbid_transforms_and_evaluate_only_never_applies() {
        let transformer: Arc<dyn Interceptor> = Arc::new(interceptor_fn(|_| async {
            Ok(Verdict::transform("$target", json!("changed")))
        }));
        let emitter = InterceptionEmitter::new().register(transformer.clone(), None);
        let blocked = emitter
            .emit(builder().agent_shutdown(ShutdownReason::Completed))
            .await
            .unwrap_err();
        assert_eq!(
            blocked.reason(),
            Some(host_error::TRANSFORM_TARGET_FORBIDDEN)
        );

        let observe = InterceptionEmitter::new()
            .with_mode(EnforcementMode::EvaluateOnly)
            .register(transformer, None)
            .register(
                Arc::new(interceptor_fn(|_| async { Ok(Verdict::deny("nope")) })),
                None,
            );
        let out = observe
            .emit(builder().input(json!("x"), "user"))
            .await
            .unwrap();
        assert_eq!(out.target, json!({"content": "x", "role": "user"}));
        assert_eq!(out.record.verdict.decision, Decision::Deny);
        assert!(out.record.proceeds());
    }

    #[test]
    fn host_failure_record() {
        let emitter = InterceptionEmitter::new();
        let r =
            emitter.record_host_failure(InterceptionPoint::PreModelCall, "projection", "s", None);
        assert_eq!(r.sequence, -1);
        assert_eq!(
            r.verdict.reason.as_deref(),
            Some(host_error::CONTEXT_INVALID)
        );
        assert!(!r.proceeds());
    }
}
