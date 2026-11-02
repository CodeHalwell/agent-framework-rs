//! Security state (per session or per scope), tool declarations, and the
//! per-invocation task-local the security tools read.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock, Weak};

use serde_json::{json, Map, Value};

use super::labels::{
    canonical_principals, new_variable_id, principal_list, ConfidentialityLabel, ContentLabel,
    IntegrityLabel, Principal,
};
use crate::client::ChatClient;
use crate::middleware::FunctionInvocationContext;
use crate::session::AgentSession;
use crate::tools::Tool;

/// The session-state key standalone security middleware stores its state
/// under.
pub const DEFAULT_SESSION_STATE_KEY: &str = "__agent_framework_fides_security__";

const STATE_CONTEXT_LABEL: &str = "context_label";
const STATE_VARIABLES: &str = "variables";
const STATE_VARIABLE_METADATA: &str = "variable_metadata";
const STATE_AUDIT_LOG: &str = "audit_log";
pub(crate) const STATE_PENDING_APPROVALS: &str = "pending_policy_approvals";

/// Serializes read-modify-write of security state held in session state
/// bags (each operation is short and never awaits).
static STATE_LOCK: Mutex<()> = Mutex::new(());

/// Where security state lives.
///
/// With a session (an agent run always has one) the state is JSON in the
/// session's state bag under [`Self::key`], so it travels with the session
/// and is isolated per session. Without one it lives in this scope's own
/// map. Middleware sharing a scope (as [`SecureAgentConfig`](super::SecureAgentConfig)
/// arranges) share the session-less state too.
#[derive(Clone, Debug)]
pub struct SecurityScope {
    key: String,
    fallback: Arc<Mutex<Map<String, Value>>>,
}

impl Default for SecurityScope {
    fn default() -> Self {
        Self::new(DEFAULT_SESSION_STATE_KEY)
    }
}

impl SecurityScope {
    /// A scope storing its session state under `key`.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            fallback: Arc::new(Mutex::new(Map::new())),
        }
    }

    /// The session-state key.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Run `f` on the state for `session` (or the session-less state).
    pub(crate) fn with_state<R>(
        &self,
        session: Option<&AgentSession>,
        f: impl FnOnce(&mut Map<String, Value>) -> R,
    ) -> R {
        match session {
            Some(session) => {
                let _guard = STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
                let mut state = match session.state.get(&self.key) {
                    Some(Value::Object(map)) => map,
                    _ => {
                        let mut map = Map::new();
                        map.insert("scope_id".into(), json!(session.session_id()));
                        map
                    }
                };
                let out = f(&mut state);
                session.state.insert(self.key.clone(), Value::Object(state));
                out
            }
            None => {
                let mut state = self.fallback.lock().unwrap_or_else(|e| e.into_inner());
                f(&mut state)
            }
        }
    }

    /// The cumulative context label (`TRUSTED` + `PUBLIC` initially).
    pub fn context_label(&self, session: Option<&AgentSession>) -> ContentLabel {
        self.with_state(session, |s| {
            s.get(STATE_CONTEXT_LABEL)
                .and_then(|v| ContentLabel::from_value(v).ok())
                .unwrap_or_default()
        })
    }

    pub(crate) fn set_context_label(&self, session: Option<&AgentSession>, label: &ContentLabel) {
        self.with_state(session, |s| {
            s.insert(STATE_CONTEXT_LABEL.into(), label.to_value());
        });
    }

    /// The scope's variable store.
    pub fn variable_store(&self, session: Option<&AgentSession>) -> ScopedVariableStore {
        ScopedVariableStore {
            scope: self.clone(),
            session: session.cloned(),
        }
    }

    /// Metadata recorded for a hidden variable (producing tool, time).
    pub fn variable_metadata(&self, session: Option<&AgentSession>, id: &str) -> Option<Value> {
        self.with_state(session, |s| {
            if !s.get(STATE_VARIABLES).and_then(|v| v.get(id)).is_some() {
                return None;
            }
            s.get(STATE_VARIABLE_METADATA)
                .and_then(|m| m.get(id))
                .cloned()
        })
    }

    pub(crate) fn set_variable_metadata(
        &self,
        session: Option<&AgentSession>,
        id: &str,
        metadata: Value,
    ) {
        self.with_state(session, |s| {
            object_entry(s, STATE_VARIABLE_METADATA).insert(id.to_string(), metadata);
        });
    }

    /// The policy audit log.
    pub fn audit_log(&self, session: Option<&AgentSession>) -> Vec<Value> {
        self.with_state(session, |s| {
            s.get(STATE_AUDIT_LOG)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
    }

    pub(crate) fn append_audit_entry(&self, session: Option<&AgentSession>, entry: Value) {
        self.with_state(session, |s| {
            let log = s
                .entry(STATE_AUDIT_LOG)
                .or_insert_with(|| Value::Array(Vec::new()));
            if !log.is_array() {
                *log = Value::Array(Vec::new());
            }
            if let Value::Array(items) = log {
                items.push(entry);
            }
        });
    }

    /// Clear the audit log.
    pub fn clear_audit_log(&self, session: Option<&AgentSession>) {
        self.with_state(session, |s| {
            s.remove(STATE_AUDIT_LOG);
        });
    }
}

/// The object stored at `key`, created (or replaced, when malformed) empty.
pub(crate) fn object_entry<'a>(
    state: &'a mut Map<String, Value>,
    key: &str,
) -> &'a mut Map<String, Value> {
    let entry = state
        .entry(key.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !entry.is_object() {
        *entry = Value::Object(Map::new());
    }
    entry.as_object_mut().expect("an object")
}

/// A [`SecurityScope`]'s variable store for one session: only variables
/// stored in this scope resolve, so ids from another session stay literal.
#[derive(Clone, Debug)]
pub struct ScopedVariableStore {
    scope: SecurityScope,
    session: Option<AgentSession>,
}

impl ScopedVariableStore {
    /// Store `content` and return its new variable id.
    pub fn store(&self, content: Value, label: &ContentLabel) -> String {
        let id = new_variable_id();
        let entry = json!({"content": content, "label": label.to_value()});
        self.scope.with_state(self.session.as_ref(), |s| {
            object_entry(s, STATE_VARIABLES).insert(id.clone(), entry);
        });
        id
    }

    /// The content and label behind `id`, if this scope owns it.
    pub fn retrieve(&self, id: &str) -> Option<(Value, ContentLabel)> {
        self.scope.with_state(self.session.as_ref(), |s| {
            let entry = s.get(STATE_VARIABLES)?.get(id)?;
            let label = ContentLabel::from_value(entry.get("label")?).ok()?;
            Some((entry.get("content").cloned().unwrap_or(Value::Null), label))
        })
    }

    /// Whether this scope owns `id`.
    pub fn exists(&self, id: &str) -> bool {
        self.retrieve(id).is_some()
    }

    /// The owned ids.
    pub fn list_variables(&self) -> Vec<String> {
        self.scope.with_state(self.session.as_ref(), |s| {
            let mut entries: Vec<String> = s
                .get(STATE_VARIABLES)
                .and_then(Value::as_object)
                .map(|m| m.keys().cloned().collect())
                .unwrap_or_default();
            entries.sort();
            entries
        })
    }

    /// Remove every variable and its metadata.
    pub fn clear(&self) {
        self.scope.with_state(self.session.as_ref(), |s| {
            s.remove(STATE_VARIABLES);
            s.remove(STATE_VARIABLE_METADATA);
        });
    }
}

/// A tool's security declarations (upstream's tool `additional_properties`:
/// `source_integrity`, `confidentiality`, `max_allowed_confidentiality`,
/// `accepts_untrusted`, the principal set and `standing_guidance`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolSecurity {
    /// The integrity of what the tool produces (tier 2). `None` inherits
    /// from its arguments or the configured default (tier 3).
    pub source_integrity: Option<IntegrityLabel>,
    /// The confidentiality of what it produces (default: the middleware's
    /// `default_confidentiality`).
    pub confidentiality: Option<ConfidentialityLabel>,
    /// The highest confidentiality it may receive (a sink's limit).
    pub max_allowed_confidentiality: Option<ConfidentialityLabel>,
    /// Whether it may run in an untrusted context and receive untrusted
    /// arguments.
    pub accepts_untrusted: bool,
    /// For `USER_IDENTITY`: the principals its results belong to (as a
    /// source) and that it is authorized to receive data for (as a sink).
    pub principals: Vec<Principal>,
    /// Trusted sentences appended to every result, explaining what a hidden
    /// or labeled result means. Fixed at declaration time.
    pub standing_guidance: Vec<String>,
}

impl ToolSecurity {
    /// No declarations.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: `source_integrity`.
    pub fn source_integrity(mut self, integrity: IntegrityLabel) -> Self {
        self.source_integrity = Some(integrity);
        self
    }

    /// Builder: `confidentiality`.
    pub fn confidentiality(mut self, confidentiality: ConfidentialityLabel) -> Self {
        self.confidentiality = Some(confidentiality);
        self
    }

    /// Builder: `max_allowed_confidentiality`.
    pub fn max_allowed_confidentiality(mut self, confidentiality: ConfidentialityLabel) -> Self {
        self.max_allowed_confidentiality = Some(confidentiality);
        self
    }

    /// Builder: `accepts_untrusted`.
    pub fn accepts_untrusted(mut self, accepts: bool) -> Self {
        self.accepts_untrusted = accepts;
        self
    }

    /// Builder: the `USER_IDENTITY` principal set.
    pub fn principals(mut self, principals: impl IntoIterator<Item = Principal>) -> Self {
        self.principals = principals.into_iter().collect();
        self
    }

    /// Builder: standing guidance (empty sentences are dropped).
    pub fn standing_guidance(mut self, sentences: impl IntoIterator<Item = String>) -> Self {
        self.standing_guidance = sentences.into_iter().filter(|s| !s.is_empty()).collect();
        self
    }

    /// The principal set as label metadata, when declared and valid.
    pub(crate) fn principal_metadata(&self) -> Option<Value> {
        let value = serde_json::to_value(&self.principals).ok()?;
        let set = canonical_principals(Some(&value), "tool").ok()?;
        Some(principal_list(&set))
    }
}

/// Tool declarations by tool name, shared by the middleware that read them.
#[derive(Clone, Debug, Default)]
pub struct ToolSecurityRegistry {
    inner: Arc<RwLock<HashMap<String, ToolSecurity>>>,
}

impl ToolSecurityRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `tool`'s security properties (replacing earlier ones).
    pub fn declare(&self, tool: impl Into<String>, security: ToolSecurity) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(tool.into(), security);
    }

    /// The declarations for `tool`.
    pub fn get(&self, tool: &str) -> Option<ToolSecurity> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(tool)
            .cloned()
    }

    /// The declarations that govern this invocation: the framework's own
    /// for a proven internal security tool, else the registered ones.
    pub(crate) fn for_invocation(&self, ctx: &FunctionInvocationContext) -> ToolSecurity {
        if let Some(builtin) = internal_tool_security(ctx) {
            return builtin;
        }
        self.get(&ctx.function_name).unwrap_or_default()
    }
}

/// The names of the security tools that consume variable ids literally.
pub(crate) const VARIABLE_ID_CONSUMERS: [&str; 2] = ["inspect_variable", "quarantined_llm"];

/// The executors of the framework's own security tools, so a same-named
/// user tool cannot claim their authority.
static INTERNAL_TOOLS: Mutex<Vec<Weak<dyn Tool>>> = Mutex::new(Vec::new());

pub(crate) fn register_internal_tool(tool: &Arc<dyn Tool>) {
    let mut tools = INTERNAL_TOOLS.lock().unwrap_or_else(|e| e.into_inner());
    tools.retain(|w| w.strong_count() > 0);
    tools.push(Arc::downgrade(tool));
}

/// Whether the tool this invocation dispatches is one of the framework's
/// security tools (the definition in the run's tool list is one of ours).
pub(crate) fn is_internal_tool(ctx: &FunctionInvocationContext) -> bool {
    let name = ctx.tool_name.as_deref().unwrap_or(&ctx.function_name);
    if !VARIABLE_ID_CONSUMERS.contains(&name) || ctx.function_name != name {
        return false;
    }
    let Some(tools) = ctx.tools.as_ref() else {
        return false;
    };
    let Some(executor) = tools
        .snapshot()
        .into_iter()
        .find(|d| d.name == name)
        .and_then(|d| d.executor)
    else {
        return false;
    };
    let target = Arc::as_ptr(&executor) as *const ();
    INTERNAL_TOOLS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .filter_map(Weak::upgrade)
        .any(|t| Arc::as_ptr(&t) as *const () == target)
}

fn internal_tool_security(ctx: &FunctionInvocationContext) -> Option<ToolSecurity> {
    if !is_internal_tool(ctx) {
        return None;
    }
    Some(match ctx.function_name.as_str() {
        "quarantined_llm" => ToolSecurity::new()
            .confidentiality(ConfidentialityLabel::Private)
            .accepts_untrusted(true)
            .source_integrity(IntegrityLabel::Untrusted),
        _ => ToolSecurity::new().confidentiality(ConfidentialityLabel::Private),
    })
}

/// What the label-tracking middleware publishes to the tools it runs.
pub(crate) struct Invocation {
    pub(crate) scope: SecurityScope,
    pub(crate) session: Option<AgentSession>,
    /// `Some` when a config bound a quarantine client (even `None`).
    pub(crate) quarantine_client: Option<Option<Arc<dyn ChatClient>>>,
    /// A label a framework tool stamps on its own result.
    pub(crate) authoritative: Mutex<Option<ContentLabel>>,
    /// Set by `inspect_variable` when its result is an error.
    pub(crate) inspect_error: Mutex<bool>,
}

tokio::task_local! {
    pub(crate) static CURRENT: Arc<Invocation>;
}

/// The invocation the current tool runs in, if any.
pub(crate) fn current_invocation() -> Option<Arc<Invocation>> {
    CURRENT.try_with(Arc::clone).ok()
}
