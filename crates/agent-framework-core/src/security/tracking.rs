//! `LabelTrackingFunctionMiddleware`: labels every tool result, hides
//! untrusted ones behind variable references, and expands references the
//! model passes back into tool arguments.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Map, Value};

use super::labels::{
    canonical_principals, combine_labels, parse_content_label, principal_list,
    ConfidentialityLabel, ContentLabel, IntegrityLabel, VariableReferenceContent,
    PRINCIPAL_METADATA_KEY,
};
use super::scope::{
    Invocation, ScopedVariableStore, SecurityScope, ToolSecurityRegistry, CURRENT,
    VARIABLE_ID_CONSUMERS,
};
use crate::client::ChatClient;
use crate::error::{Error, Result};
use crate::middleware::{FunctionInvocationContext, Middleware, Next};
use crate::session::AgentSession;
use crate::tools::ToolDefinition;

/// Metadata key: the arguments as the model sent them (references
/// unexpanded), restored on every pass so re-entry never double-expands.
pub const ORIGINAL_ARGUMENTS_KEY: &str = "original_arguments_for_messages";
/// Metadata key: the conversation's context label at call time.
pub const CONTEXT_LABEL_KEY: &str = "context_label";
/// Metadata key: the combined label of the call's arguments.
pub const ARGUMENT_LABEL_KEY: &str = "argument_label";
/// Metadata key: the context and argument labels combined.
pub const EFFECTIVE_LABEL_KEY: &str = "effective_invocation_label";
/// Metadata key: the combined label of the call's result.
pub const RESULT_LABEL_KEY: &str = "result_label";
/// Metadata key: per top-level argument, the list positions (or `-1` for
/// the whole value) the middleware rewrote from hidden content.
pub const REWRITTEN_ARGUMENTS_KEY: &str = "_rewritten_argument_indices";
/// Metadata key set when the policy middleware short-circuited the call.
pub(crate) const POLICY_TERMINATED_KEY: &str = "_security_policy_terminated";

const MAX_REFERENCE_DEPTH: usize = 16;
const MAX_REFERENCE_COUNT: usize = 100;

/// The argument positions rewritten from hidden content in this invocation
/// (empty when none), so a tool can tell which inputs came from a hidden
/// variable.
pub fn rewritten_arguments(ctx: &FunctionInvocationContext) -> BTreeMap<String, BTreeSet<i64>> {
    ctx.metadata
        .get(REWRITTEN_ARGUMENTS_KEY)
        .and_then(Value::as_object)
        .map(|m| {
            m.iter()
                .map(|(k, v)| {
                    let set = v
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_i64).collect())
                        .unwrap_or_default();
                    (k.clone(), set)
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Tracks and propagates security labels through tool invocations.
///
/// A result's label comes from, in priority order:
///
/// 1. a label embedded in the result (`{"security_label": {...}}`), which
///    may only restrict the fallback below, unless a framework security
///    tool stamped it;
/// 2. the tool's declared `source_integrity` ([`ToolSecurity`](super::ToolSecurity));
/// 3. the labels of the hidden variables its arguments resolved (or the
///    configured default integrity), restricted by any labels in its
///    arguments.
///
/// Confidentiality is the tool's declared confidentiality joined with its
/// arguments'. Untrusted results are replaced by a
/// [`VariableReferenceContent`] while the context is still trusted; the
/// content stays in the session's variable store. The model can pass
/// `[var_<id>]` back in a later tool argument, and the middleware expands
/// it (with its label) before the tool runs. The conversation's context
/// label accumulates what the model has seen.
///
/// Install it first in the agent's function middleware, ahead of
/// [`PolicyEnforcementFunctionMiddleware`](super::PolicyEnforcementFunctionMiddleware).
pub struct LabelTrackingFunctionMiddleware {
    default_integrity: IntegrityLabel,
    default_confidentiality: ConfidentialityLabel,
    auto_hide_untrusted: bool,
    hide_threshold: IntegrityLabel,
    scope: SecurityScope,
    tools: ToolSecurityRegistry,
    quarantine_client: Option<Option<Arc<dyn ChatClient>>>,
}

impl std::fmt::Debug for LabelTrackingFunctionMiddleware {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LabelTrackingFunctionMiddleware")
            .field("default_integrity", &self.default_integrity)
            .field("default_confidentiality", &self.default_confidentiality)
            .field("auto_hide_untrusted", &self.auto_hide_untrusted)
            .field("hide_threshold", &self.hide_threshold)
            .field("scope", &self.scope)
            .finish_non_exhaustive()
    }
}

impl Default for LabelTrackingFunctionMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelTrackingFunctionMiddleware {
    /// Defaults: untrusted results by default, public confidentiality,
    /// automatic hiding of untrusted results.
    pub fn new() -> Self {
        Self {
            default_integrity: IntegrityLabel::Untrusted,
            default_confidentiality: ConfidentialityLabel::Public,
            auto_hide_untrusted: true,
            hide_threshold: IntegrityLabel::Untrusted,
            scope: SecurityScope::default(),
            tools: ToolSecurityRegistry::new(),
            quarantine_client: None,
        }
    }

    /// Builder: the integrity of results from tools with no declaration.
    pub fn default_integrity(mut self, integrity: IntegrityLabel) -> Self {
        self.default_integrity = integrity;
        self
    }

    /// Builder: the confidentiality of results from tools with no
    /// declaration.
    pub fn default_confidentiality(mut self, confidentiality: ConfidentialityLabel) -> Self {
        self.default_confidentiality = confidentiality;
        self
    }

    /// Builder: whether untrusted results are hidden behind variables.
    pub fn auto_hide_untrusted(mut self, hide: bool) -> Self {
        self.auto_hide_untrusted = hide;
        self
    }

    /// Builder: the integrity at which results are hidden.
    pub fn hide_threshold(mut self, threshold: IntegrityLabel) -> Self {
        self.hide_threshold = threshold;
        self
    }

    /// Builder: the state scope (share one with the policy middleware).
    pub fn scope(mut self, scope: SecurityScope) -> Self {
        self.scope = scope;
        self
    }

    /// Builder: the tool declarations (share them with the policy
    /// middleware).
    pub fn tool_registry(mut self, tools: ToolSecurityRegistry) -> Self {
        self.tools = tools;
        self
    }

    /// Builder: declare one tool's security properties.
    pub fn tool_security(self, tool: impl Into<String>, security: super::ToolSecurity) -> Self {
        self.tools.declare(tool, security);
        self
    }

    /// Builder: the client `quarantined_llm` uses for calls this middleware
    /// mediates (`None` disables real calls even if a global client is
    /// set).
    pub fn quarantine_client(mut self, client: Option<Arc<dyn ChatClient>>) -> Self {
        self.quarantine_client = Some(client);
        self
    }

    /// The state scope.
    pub fn security_scope(&self) -> &SecurityScope {
        &self.scope
    }

    /// The tool declarations.
    pub fn registry(&self) -> &ToolSecurityRegistry {
        &self.tools
    }

    /// The cumulative context label for `session` (or the session-less
    /// state).
    pub fn context_label(&self, session: Option<&AgentSession>) -> ContentLabel {
        self.scope.context_label(session)
    }

    /// Reset the context label to `TRUSTED` + `PUBLIC`.
    pub fn reset_context_label(&self, session: Option<&AgentSession>) {
        self.scope.set_context_label(
            session,
            &ContentLabel::trusted().with_metadata("reset", json!(true)),
        );
    }

    /// The variable store for `session`.
    pub fn variable_store(&self, session: Option<&AgentSession>) -> ScopedVariableStore {
        self.scope.variable_store(session)
    }

    /// Metadata recorded for a hidden variable.
    pub fn variable_metadata(&self, id: &str, session: Option<&AgentSession>) -> Option<Value> {
        self.scope.variable_metadata(session, id)
    }

    /// The hidden variable ids.
    pub fn list_variables(&self, session: Option<&AgentSession>) -> Vec<String> {
        self.variable_store(session).list_variables()
    }

    /// The security tools (`quarantined_llm`, `inspect_variable`).
    pub fn security_tools(&self) -> Vec<ToolDefinition> {
        super::get_security_tools()
    }

    /// Instructions teaching the model to work with hidden content.
    pub fn security_instructions(&self) -> &'static str {
        super::SECURITY_TOOL_INSTRUCTIONS
    }

    fn update_context_label(&self, session: Option<&AgentSession>, new: &ContentLabel) {
        let old = self.scope.context_label(session);
        let updated = combine_labels([&old, new]);
        if updated != old {
            tracing::info!(
                from = %old,
                to = %updated,
                "security context label tainted"
            );
        }
        self.scope.set_context_label(session, &updated);
    }

    fn should_hide(
        &self,
        label: &ContentLabel,
        function_name: &str,
        context: &ContentLabel,
    ) -> bool {
        self.auto_hide_untrusted
            && label.integrity == self.hide_threshold
            && context.integrity == IntegrityLabel::Trusted
            && function_name != "inspect_variable"
    }
}

/// Labels found in the arguments themselves: variable references and any
/// object carrying a `security_label` (or legacy `label`). They may only
/// restrict.
fn input_labels(value: &Value, out: &mut Vec<ContentLabel>) {
    match value {
        Value::Object(map) => {
            if let Some(label) = map.get("security_label").filter(|l| l.is_object()) {
                if let Ok(l) = parse_content_label(label, "input", false) {
                    out.push(l);
                }
            } else if let Some(label) = map.get("label").filter(|l| l.is_object()) {
                if let Ok(l) = parse_content_label(label, "input", false) {
                    out.push(l);
                }
            }
            for v in map.values() {
                input_labels(v, out);
            }
        }
        Value::Array(items) => items.iter().for_each(|v| input_labels(v, out)),
        _ => {}
    }
}

fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// One reference found in a string: its byte range, id, and whether it was
/// bare (`var_x`) rather than bracketed (`[var_x]`).
struct Reference {
    start: usize,
    end: usize,
    id: String,
    bare: bool,
}

fn hex_run(s: &str) -> usize {
    s.bytes().take_while(u8::is_ascii_hexdigit).count()
}

/// A bracketed reference at the start of `s` (`[ var_<hex> ]`), returning
/// its length and id.
fn bracketed_at(s: &str) -> Option<(usize, String)> {
    let rest = s.strip_prefix('[')?;
    let inner = rest.trim_start();
    let lead = rest.len() - inner.len();
    let body = inner.strip_prefix("var_")?;
    let hex = hex_run(body);
    if hex == 0 {
        return None;
    }
    let after = &body[hex..];
    let trimmed = after.trim_start();
    let gap = after.len() - trimmed.len();
    trimmed.strip_prefix(']')?;
    let len = 1 + lead + 4 + hex + gap + 1;
    Some((len, format!("var_{}", &body[..hex])))
}

/// A bare reference at the start of `s` (`var_` and at least 8 hex digits
/// ending at a word boundary).
fn bare_at(s: &str) -> Option<(usize, String)> {
    let body = s.strip_prefix("var_")?;
    let hex = hex_run(body);
    if hex < 8 || body[hex..].chars().next().is_some_and(is_word) {
        return None;
    }
    Some((4 + hex, s[..4 + hex].to_string()))
}

fn find_references(s: &str) -> Vec<Reference> {
    let mut out = Vec::new();
    let mut prev: Option<char> = None;
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        let found = bracketed_at(rest)
            .map(|(len, id)| (len, id, false))
            .or_else(|| {
                (!prev.is_some_and(is_word))
                    .then(|| bare_at(rest).map(|(len, id)| (len, id, true)))
                    .flatten()
            });
        if let Some((len, id, bare)) = found {
            out.push(Reference {
                start: i,
                end: i + len,
                id,
                bare,
            });
            prev = s[..i + len].chars().next_back();
            i += len;
            continue;
        }
        let c = rest.chars().next().expect("non-empty");
        prev = Some(c);
        i += c.len_utf8();
    }
    out
}

/// The whole string is one reference (surrounding whitespace allowed).
fn whole_reference(s: &str) -> Option<(String, bool)> {
    let t = s.trim();
    if let Some((len, id)) = bracketed_at(t) {
        return (len == t.len()).then_some((id, false));
    }
    let body = t.strip_prefix("var_")?;
    let hex = hex_run(body);
    (hex >= 8 && hex == body.len()).then(|| (t.to_string(), true))
}

/// The primary response of a hidden `quarantined_llm` result.
fn primary_content(value: Value, from_quarantined_llm: bool) -> Value {
    if !from_quarantined_llm {
        return value;
    }
    match value {
        Value::Object(ref map) => map.get("response").cloned().unwrap_or(value),
        Value::String(ref s) => {
            let t = s.trim();
            if t.starts_with('{') && t.ends_with('}') {
                if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(t) {
                    if let Some(r) = map.get("response") {
                        return r.clone();
                    }
                }
            }
            value
        }
        other => other,
    }
}

type Path = Vec<PathSegment>;

#[derive(Clone, PartialEq, Eq, Hash)]
enum PathSegment {
    Key(String),
    Index(usize),
}

/// Expands owned variable references, collecting their stored labels.
struct Resolver<'a> {
    store: &'a ScopedVariableStore,
    scope: &'a SecurityScope,
    session: Option<&'a AgentSession>,
    labels: Vec<ContentLabel>,
    active: HashSet<String>,
    count: usize,
}

impl Resolver<'_> {
    fn lookup(&mut self, id: &str, depth: usize) -> std::result::Result<Option<Value>, String> {
        if self.active.contains(id) {
            return Err("Variable reference cycle detected".into());
        }
        if depth >= MAX_REFERENCE_DEPTH {
            return Err("Variable reference depth limit exceeded".into());
        }
        let Some((content, label)) = self.store.retrieve(id) else {
            return Ok(None);
        };
        self.count += 1;
        if self.count > MAX_REFERENCE_COUNT {
            return Err("Variable reference count limit exceeded".into());
        }
        self.labels.push(label);
        let from_quarantine = self
            .scope
            .variable_metadata(self.session, id)
            .and_then(|m| {
                m.get("function_name")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .is_some_and(|f| f == "quarantined_llm");
        let expanded = primary_content(content, from_quarantine);
        self.active.insert(id.to_string());
        let out = self.resolve(expanded, depth + 1, &mut Vec::new(), None);
        self.active.remove(id);
        out.map(Some)
    }

    fn resolve_string(
        &mut self,
        value: String,
        depth: usize,
        path: &mut Path,
        rewritten: Option<&mut HashSet<Path>>,
    ) -> std::result::Result<Value, String> {
        let references = find_references(&value);
        if references.is_empty() {
            return Ok(Value::String(value));
        }
        if let Some((id, bare)) = whole_reference(&value) {
            return Ok(match self.lookup(&id, depth)? {
                None => Value::String(value),
                Some(resolved) => {
                    if bare {
                        tracing::warn!("{BARE_REFERENCE_WARNING}");
                    }
                    if let Some(r) = rewritten {
                        r.insert(path.clone());
                    }
                    resolved
                }
            });
        }
        let mut out = String::with_capacity(value.len());
        let mut last = 0;
        let mut any = false;
        for reference in references {
            out.push_str(&value[last..reference.start]);
            match self.lookup(&reference.id, depth)? {
                None => out.push_str(&value[reference.start..reference.end]),
                Some(resolved) => {
                    if reference.bare {
                        tracing::warn!("{BARE_REFERENCE_WARNING}");
                    }
                    any = true;
                    match resolved {
                        Value::String(s) => out.push_str(&s),
                        other => out.push_str(&other.to_string()),
                    }
                }
            }
            last = reference.end;
        }
        out.push_str(&value[last..]);
        if any {
            if let Some(r) = rewritten {
                r.insert(path.clone());
            }
        }
        Ok(Value::String(out))
    }

    fn resolve(
        &mut self,
        value: Value,
        depth: usize,
        path: &mut Path,
        mut rewritten: Option<&mut HashSet<Path>>,
    ) -> std::result::Result<Value, String> {
        match value {
            Value::String(s) => self.resolve_string(s, depth, path, rewritten),
            Value::Object(map) => {
                let mut out = Map::new();
                for (k, v) in map {
                    path.push(PathSegment::Key(k.clone()));
                    let resolved = self.resolve(v, depth, path, rewritten.as_deref_mut());
                    path.pop();
                    out.insert(k, resolved?);
                }
                Ok(Value::Object(out))
            }
            Value::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for (i, v) in items.into_iter().enumerate() {
                    path.push(PathSegment::Index(i));
                    let resolved = self.resolve(v, depth, path, rewritten.as_deref_mut());
                    path.pop();
                    out.push(resolved?);
                }
                Ok(Value::Array(out))
            }
            other => Ok(other),
        }
    }
}

const BARE_REFERENCE_WARNING: &str =
    "Expanded a bare variable reference in a tool argument; models should use [var_<id>] instead.";

/// What one result item is.
enum Item {
    /// The tool's own result.
    Result(Value),
    /// A trusted standing-guidance sentence.
    Guidance(String),
}

#[async_trait]
impl Middleware<FunctionInvocationContext> for LabelTrackingFunctionMiddleware {
    async fn process(
        &self,
        mut ctx: FunctionInvocationContext,
        next: Next<FunctionInvocationContext>,
    ) -> Result<FunctionInvocationContext> {
        match ctx.metadata.get(ORIGINAL_ARGUMENTS_KEY) {
            Some(original) => ctx.arguments = original.clone(),
            None => {
                ctx.metadata
                    .insert(ORIGINAL_ARGUMENTS_KEY.into(), ctx.arguments.clone());
            }
        }
        let session = ctx.session.clone();
        let function_name = ctx.function_name.clone();
        let declaration = self.tools.for_invocation(&ctx);

        let mut found = Vec::new();
        input_labels(&ctx.arguments, &mut found);
        let input = found;
        let confidentiality = declaration
            .confidentiality
            .unwrap_or(self.default_confidentiality);

        // Expand hidden references before execution, keeping their labels.
        let mut resolved_labels = Vec::new();
        if !VARIABLE_ID_CONSUMERS.contains(&function_name.as_str()) {
            let store = self.scope.variable_store(session.as_ref());
            let mut resolver = Resolver {
                store: &store,
                scope: &self.scope,
                session: session.as_ref(),
                labels: Vec::new(),
                active: HashSet::new(),
                count: 0,
            };
            let mut rewritten = HashSet::new();
            let arguments = std::mem::take(&mut ctx.arguments);
            ctx.arguments = resolver
                .resolve(arguments, 0, &mut Vec::new(), Some(&mut rewritten))
                .map_err(Error::Tool)?;
            resolved_labels = resolver.labels;
            let mut indices: BTreeMap<String, BTreeSet<i64>> = BTreeMap::new();
            for path in rewritten {
                let Some(PathSegment::Key(arg)) = path.first() else {
                    continue;
                };
                let entry = indices.entry(arg.clone()).or_default();
                let value = ctx.arguments.get(arg);
                match (path.get(1), value) {
                    (None, Some(Value::Array(items))) => entry.extend(0..items.len() as i64),
                    (Some(PathSegment::Index(i)), Some(Value::Array(_))) => {
                        entry.insert(*i as i64);
                    }
                    _ => {
                        entry.insert(-1);
                    }
                }
            }
            ctx.metadata
                .insert(REWRITTEN_ARGUMENTS_KEY.into(), json!(indices));
        }

        let argument_labels: Vec<&ContentLabel> = input.iter().chain(&resolved_labels).collect();
        let argument_label = combine_labels(argument_labels.iter().copied());
        let mut declared = ContentLabel::new(IntegrityLabel::Trusted, confidentiality);
        if confidentiality == ConfidentialityLabel::UserIdentity {
            match declaration.principal_metadata() {
                Some(p) => {
                    declared.metadata.insert(PRINCIPAL_METADATA_KEY.into(), p);
                }
                None => tracing::warn!(
                    tool = %function_name,
                    "invalid USER_IDENTITY source principals for tool"
                ),
            }
        }
        let policy_label = combine_labels([&declared, &argument_label]);
        let mut fallback_metadata = Map::new();
        fallback_metadata.insert("function_name".into(), json!(function_name));
        if let Some(p) = policy_label.metadata.get(PRINCIPAL_METADATA_KEY) {
            fallback_metadata.insert(PRINCIPAL_METADATA_KEY.into(), p.clone());
        }
        let (integrity, source) = if let Some(declared) = declaration.source_integrity {
            (declared, "source_integrity")
        } else if !argument_labels.is_empty() {
            // Argument labels can restrict, never establish, integrity.
            let baseline = if resolved_labels.is_empty() {
                ContentLabel::new(self.default_integrity, ConfidentialityLabel::Public)
            } else {
                combine_labels(&resolved_labels)
            };
            let combined = combine_labels(std::iter::once(&baseline).chain(&input));
            (combined.integrity, "input_labels_join")
        } else {
            (self.default_integrity, "default")
        };
        fallback_metadata.insert("source".into(), json!(source));
        let fallback = ContentLabel {
            integrity,
            confidentiality: policy_label.confidentiality,
            metadata: fallback_metadata,
        };

        let context_label = self.scope.context_label(session.as_ref());
        let effective = combine_labels([&context_label, &argument_label]);
        ctx.metadata
            .insert(CONTEXT_LABEL_KEY.into(), context_label.to_value());
        ctx.metadata
            .insert(ARGUMENT_LABEL_KEY.into(), argument_label.to_value());
        ctx.metadata
            .insert(EFFECTIVE_LABEL_KEY.into(), effective.to_value());

        let invocation = Arc::new(Invocation {
            scope: self.scope.clone(),
            session: session.clone(),
            quarantine_client: self.quarantine_client.clone(),
            authoritative: Mutex::new(None),
            inspect_error: Mutex::new(false),
        });
        let mut ctx = CURRENT.scope(invocation.clone(), next.run(ctx)).await?;
        if ctx.metadata.contains_key(POLICY_TERMINATED_KEY) {
            return Ok(ctx);
        }
        self.label_result(
            &mut ctx,
            &invocation,
            &function_name,
            &fallback,
            &declaration.standing_guidance,
        );
        Ok(ctx)
    }
}

impl LabelTrackingFunctionMiddleware {
    fn label_result(
        &self,
        ctx: &mut FunctionInvocationContext,
        invocation: &Invocation,
        function_name: &str,
        fallback: &ContentLabel,
        guidance: &[String],
    ) {
        let session = ctx.session.clone();
        let session = session.as_ref();
        let mut items: Vec<Item> = Vec::new();
        if let Some(result) = ctx.result.take() {
            items.push(Item::Result(result));
        }
        items.extend(guidance.iter().cloned().map(Item::Guidance));
        if items.is_empty() {
            ctx.metadata
                .insert(RESULT_LABEL_KEY.into(), fallback.to_value());
            return;
        }
        let authoritative = invocation
            .authoritative
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        let inspect_error = function_name == "inspect_variable"
            && *invocation
                .inspect_error
                .lock()
                .unwrap_or_else(|e| e.into_inner());
        let context_label = self.scope.context_label(session);

        let mut processed = Vec::with_capacity(items.len());
        let mut labels = Vec::with_capacity(items.len());
        let mut visible = Vec::new();
        for item in items {
            let (value, label) = match item {
                Item::Result(value) => {
                    let label =
                        extract_label(&value, fallback, authoritative.clone(), inspect_error);
                    (value, label)
                }
                Item::Guidance(sentence) => {
                    let label = ContentLabel {
                        integrity: IntegrityLabel::Trusted,
                        confidentiality: fallback.confidentiality,
                        metadata: fallback.metadata.clone(),
                    };
                    (Value::String(sentence), label)
                }
            };
            if self.should_hide(&label, function_name, &context_label) {
                processed.push(self.hide(session, value, &label, function_name));
            } else {
                processed.push(value);
                visible.push(label.clone());
            }
            labels.push(label);
        }
        let result_label = combine_labels(&labels);
        ctx.result = Some(if processed.len() == 1 {
            processed.pop().expect("one item")
        } else {
            Value::Array(processed)
        });
        ctx.metadata
            .insert(RESULT_LABEL_KEY.into(), result_label.to_value());

        // Hidden items never reach the model, so only visible ones taint
        // integrity; confidentiality covers the whole result.
        if visible.is_empty() {
            let current = self.scope.context_label(session);
            if result_label.confidentiality != current.confidentiality
                || result_label.confidentiality == ConfidentialityLabel::UserIdentity
            {
                let hidden = ContentLabel {
                    integrity: current.integrity,
                    confidentiality: result_label.confidentiality,
                    metadata: result_label.metadata.clone(),
                };
                self.update_context_label(session, &hidden);
            }
        } else {
            let visible_label = combine_labels(&visible);
            let exposed = ContentLabel {
                integrity: visible_label.integrity,
                confidentiality: result_label.confidentiality,
                metadata: result_label.metadata.clone(),
            };
            self.update_context_label(session, &exposed);
        }
    }

    fn hide(
        &self,
        session: Option<&AgentSession>,
        value: Value,
        label: &ContentLabel,
        function_name: &str,
    ) -> Value {
        let id = self.scope.variable_store(session).store(value, label);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or_default();
        self.scope.set_variable_metadata(
            session,
            &id,
            json!({"function_name": function_name, "timestamp": now}),
        );
        tracing::info!(tool = %function_name, "auto-hid an untrusted result behind a variable reference");
        // The model sees only the label's levels: its metadata may carry
        // strings derived from the hidden content, which must not reach the
        // model while the context is trusted. The full label stays with the
        // stored value.
        VariableReferenceContent::new(
            id,
            ContentLabel::new(label.integrity, label.confidentiality),
            Some(format!("Result from {function_name}")),
        )
        .to_value()
    }
}

/// The label of one result item: an authoritative framework label as is;
/// an embedded label joined with (restricting) the fallback; an
/// `inspect_variable` error at the invocation's integrity; else the
/// fallback.
fn extract_label(
    value: &Value,
    fallback: &ContentLabel,
    authoritative: Option<ContentLabel>,
    inspect_error: bool,
) -> ContentLabel {
    if let Some(label) = authoritative {
        if label.confidentiality != ConfidentialityLabel::UserIdentity
            || label.principals().is_some()
        {
            return label;
        }
        return combine_labels([
            fallback,
            &ContentLabel::new(label.integrity, label.confidentiality),
        ]);
    }
    if let Some(embedded) = value.get("security_label").filter(|l| l.is_object()) {
        match parse_content_label(embedded, "embedded", false) {
            Ok(embedded) => {
                // A result's own label is an untrusted claim: only its
                // levels count (they can only restrict). Its metadata is
                // attacker-controlled and is dropped.
                let embedded = ContentLabel::new(embedded.integrity, embedded.confidentiality);
                let mut combined = combine_labels([fallback, &embedded]);
                if combined.confidentiality == ConfidentialityLabel::UserIdentity
                    && fallback.confidentiality == ConfidentialityLabel::UserIdentity
                {
                    match canonical_principals(
                        fallback.metadata.get(PRINCIPAL_METADATA_KEY),
                        "fallback label",
                    ) {
                        Ok(p) => {
                            combined
                                .metadata
                                .insert(PRINCIPAL_METADATA_KEY.into(), principal_list(&p));
                        }
                        Err(_) => {
                            combined.metadata.remove(PRINCIPAL_METADATA_KEY);
                        }
                    }
                }
                return combined;
            }
            Err(e) => tracing::warn!("failed to parse an embedded security_label: {e}"),
        }
    }
    if inspect_error {
        let integrity =
            if fallback.metadata.get("source").and_then(Value::as_str) == Some("default") {
                IntegrityLabel::Trusted
            } else {
                fallback.integrity
            };
        return ContentLabel {
            integrity,
            confidentiality: fallback.confidentiality,
            metadata: fallback.metadata.clone(),
        };
    }
    fallback.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &str) -> Vec<(String, bool)> {
        find_references(s)
            .into_iter()
            .map(|r| (r.id, r.bare))
            .collect()
    }

    #[test]
    fn references_are_found_like_upstream_patterns() {
        assert_eq!(
            ids("see [var_ab] and [ var_0123 ]"),
            [
                ("var_ab".to_string(), false),
                ("var_0123".to_string(), false)
            ]
        );
        assert_eq!(
            ids("bare var_0123abcd here"),
            [("var_0123abcd".to_string(), true)]
        );
        // Too short, glued to a word, or followed by a word character.
        assert!(ids("var_0123abc xvar_0123abcd var_0123abcdz").is_empty());
        assert_eq!(
            whole_reference("  [var_ab]  "),
            Some(("var_ab".into(), false))
        );
        assert_eq!(
            whole_reference("var_0123abcd"),
            Some(("var_0123abcd".into(), true))
        );
        assert_eq!(whole_reference("x [var_ab]"), None);
    }
}
