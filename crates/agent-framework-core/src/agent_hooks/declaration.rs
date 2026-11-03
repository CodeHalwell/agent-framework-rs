//! Host declaration documents (§7.7): one JSON document that fixes the
//! enforcement configuration (mode, composition, identity provider,
//! approval resolver and redactor, timeouts, record bound), the declared
//! surface and the interceptor bindings.
//!
//! Port of the agent-hooks SDK's declaration loader and registry, which
//! upstream's `create_agent_hooks_middleware_from_declaration` (Python)
//! uses. Loading is a pipeline; each step runs only when the one before it
//! passed, so a document yields exactly one refusal class
//! ([`DeclarationErrorClass`]):
//!
//! | Step | Where | Check | Class |
//! | --- | --- | --- | --- |
//! | 1 | [`HostDeclaration::from_path`] | regular file, size, UTF-8 | `unreadable` |
//! | 2 | [`HostDeclaration::from_json`] | JSON object, no duplicate keys, depth | `malformed` |
//! | 3 | same | `declaration` in the accepted set | `version_unsupported` |
//! | 4 | same | `spec` compatible with this host | `spec_unsupported` |
//! | 5 | same | no unknown member at a closed level | `unknown_field` |
//! | 6 | same | types, enums, patterns, ranges, integers within ±(2⁵³−1) | `invalid_field` |
//! | 7 | same, and again on the filled surface | internal consistency | `inconsistent` |
//! | 8 | [`HostRegistry::resolve`] | surface and configuration honoured by this host | `surface_unsupported` |
//! | 9 | same | named identity provider, resolver and redactor registered | `reference_unresolved` |
//! | 10 | same | every binding kind registered | `kind_unknown` |
//! | 11 | [`AgentHooksOptions::from_declaration`](super::AgentHooksOptions::from_declaration) | kind resolvers run | `binding_rejected` |
//!
//! Nothing in the pipeline drops or defaults a member the document stated;
//! defaults fill absent members only.
//!
//! This host's surface ([`HostSurface::agent_framework`]) is upstream's:
//! all eight points; `model_calls`, `tool_calls`, `int64_json`,
//! `bigint_json` and `host_declaration`; buffered output; every profile and
//! knob value; and posture `tool_seam_host_error: terminate` (a
//! `host_error:*` deny at the tool seam halts the run). The spec default
//! posture is `continue`, so a document for this host must state
//! `"posture": {"tool_seam_host_error": "terminate"}`, as upstream requires.
//!
//! ```
//! use std::sync::Arc;
//! use agent_framework_core::agent_hooks::{
//!     interceptor_fn, AgentHooks, HostDeclaration, HostRegistry, Verdict,
//! };
//!
//! let declaration = HostDeclaration::from_json(r#"{
//!     "declaration": "agent-hooks-declaration/1.0",
//!     "configuration": {"posture": {"tool_seam_host_error": "terminate"}},
//!     "bindings": [{"id": "egress", "kind": "agent_framework.interceptor"}]
//! }"#)?;
//! let allow = Arc::new(interceptor_fn(|_| async { Ok(Verdict::allow()) }));
//! let registry = HostRegistry::new().interceptors([("egress", allow as _)])?;
//! let hooks = AgentHooks::from_declaration(&declaration, &registry)?;
//! # let _ = hooks;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::io::Read as _;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::composition::{
    ApprovalRedactor, ApprovalResolver, Composition, CompositionProfile, OnApproval,
    SynthesisPolicy,
};
use super::identity::{IdentityProvider, JCS_SHA256};
use super::protocol::{
    EnforcementMode, InterceptionContext, InterceptionPoint, Interceptor, SPEC_VERSION,
};
use super::seams::{AgentHooks, AgentHooksOptions};

/// The declaration contract version this host writes and reads.
pub const DECLARATION_VERSION: &str = "agent-hooks-declaration/1.0";
/// The declaration versions this loader accepts.
pub const SUPPORTED_DECLARATION_VERSIONS: &[&str] = &[DECLARATION_VERSION];
/// The kind upstream registers for interceptors supplied by name
/// ([`HostRegistry::interceptors`]).
pub const INTERCEPTOR_KIND: &str = "agent_framework.interceptor";

/// Largest document the text paths accept, in bytes.
pub const MAX_DOCUMENT_BYTES: usize = 1 << 20;
/// Deepest container nesting (the root object is depth 1).
pub const MAX_DEPTH: usize = 32;
/// Most bindings one document may carry.
pub const MAX_BINDINGS: usize = 256;
/// Longest binding `id`, reference name and document `id`.
pub const MAX_ID_LEN: usize = 64;
/// Longest binding `kind`.
pub const MAX_KIND_LEN: usize = 128;
/// Longest `exposure_bound` and longest refusal detail.
pub const MAX_DETAIL_LEN: usize = 512;
/// Largest timeout, in milliseconds (one hour).
pub const MAX_TIMEOUT_MS: u64 = 3_600_000;
/// Largest `records.max_buffered`: 2⁵³−1.
pub const MAX_BUFFERED_RECORDS: u64 = 9_007_199_254_740_991;
/// Smallest declared surface: the §3.2 lifecycle floor.
pub const FLOOR_POINTS: [InterceptionPoint; 4] = [
    InterceptionPoint::AgentStartup,
    InterceptionPoint::Input,
    InterceptionPoint::Output,
    InterceptionPoint::AgentShutdown,
];
/// The closed capability vocabulary a surface may name (§13.1).
pub const CAPABILITIES: &[&str] = &[
    "model_calls",
    "tool_calls",
    "parallel_tool_calls",
    "streaming",
    "multi_turn",
    "int64_json",
    "bigint_json",
    "incremental_output",
    "host_declaration",
];

const ALL_POINTS: [InterceptionPoint; 8] = [
    InterceptionPoint::AgentStartup,
    InterceptionPoint::Input,
    InterceptionPoint::PreModelCall,
    InterceptionPoint::PostModelCall,
    InterceptionPoint::PreToolCall,
    InterceptionPoint::PostToolCall,
    InterceptionPoint::Output,
    InterceptionPoint::AgentShutdown,
];
const ALL_PROFILES: [CompositionProfile; 4] = [
    CompositionProfile::SequentialFirstDeny,
    CompositionProfile::SequentialRunAll,
    CompositionProfile::ParallelStrictest,
    CompositionProfile::ParallelUnanimous,
];
const RESERVED_KIND_SEGMENTS: [&str; 2] = ["agent_hooks", "ctk"];

fn profile_from_wire(s: &str) -> Option<CompositionProfile> {
    ALL_PROFILES.into_iter().find(|p| p.as_str() == s)
}

/// The knob members a profile consults (§7.2).
fn consulted_knobs(profile: CompositionProfile) -> &'static [&'static str] {
    match profile {
        CompositionProfile::SequentialFirstDeny => &["on_approval"],
        CompositionProfile::SequentialRunAll => &[],
        CompositionProfile::ParallelStrictest => &["on_transform_conflict"],
        CompositionProfile::ParallelUnanimous => &["on_disagreement"],
    }
}

fn knob_values(knob: &str) -> &'static [&'static str] {
    match knob {
        "on_approval" => &["stop", "resume"],
        "on_disagreement" | "on_transform_conflict" => &["deny", "approval"],
        _ => &[],
    }
}

fn knob_default(knob: &str) -> Option<&'static str> {
    match knob {
        "on_approval" => Some("stop"),
        "on_disagreement" | "on_transform_conflict" => Some("deny"),
        _ => None,
    }
}

fn point_from_wire(s: &str) -> Option<InterceptionPoint> {
    ALL_POINTS.into_iter().find(|p| p.as_str() == s)
}

// ---- errors -------------------------------------------------------------------

/// The eleven refusal classes (§7.7.6), in pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclarationErrorClass {
    /// Step 1: the file cannot be read.
    Unreadable,
    /// Step 2: not one bounded JSON object.
    Malformed,
    /// Step 3: `declaration` not accepted.
    VersionUnsupported,
    /// Step 4: `spec` not compatible.
    SpecUnsupported,
    /// Step 5: an unknown member at a closed level.
    UnknownField,
    /// Step 6: a type, enum, pattern or range violation.
    InvalidField,
    /// Step 7: internally inconsistent.
    Inconsistent,
    /// Step 8: more than this host can honour.
    SurfaceUnsupported,
    /// Step 9: a named provider, resolver or redactor is not registered.
    ReferenceUnresolved,
    /// Step 10: a binding kind is not registered.
    KindUnknown,
    /// Step 11: a kind resolver refused its binding.
    BindingRejected,
}

impl DeclarationErrorClass {
    /// The bare class name (`unknown_field`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unreadable => "unreadable",
            Self::Malformed => "malformed",
            Self::VersionUnsupported => "version_unsupported",
            Self::SpecUnsupported => "spec_unsupported",
            Self::UnknownField => "unknown_field",
            Self::InvalidField => "invalid_field",
            Self::Inconsistent => "inconsistent",
            Self::SurfaceUnsupported => "surface_unsupported",
            Self::ReferenceUnresolved => "reference_unresolved",
            Self::KindUnknown => "kind_unknown",
            Self::BindingRejected => "binding_rejected",
        }
    }

    /// The namespaced code (`declaration_error:unknown_field`).
    pub fn code(self) -> String {
        format!("declaration_error:{}", self.as_str())
    }
}

impl fmt::Display for DeclarationErrorClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One problem a load step found: a JSON pointer into the document (the
/// empty string is the root) and a detail naming members, kinds and ids
/// but never binding configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// RFC 6901 pointer.
    pub pointer: String,
    /// Payload-free detail.
    pub detail: String,
}

/// A refused declaration: one class, every finding that step produced and,
/// for `version_unsupported`, the accepted version set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarationError {
    /// The refusal class.
    pub class: DeclarationErrorClass,
    /// The findings.
    pub findings: Vec<Finding>,
    /// The accepted versions (`version_unsupported` only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub accepted: Vec<String>,
}

impl DeclarationError {
    fn new(
        class: DeclarationErrorClass,
        pointer: impl Into<String>,
        detail: impl Into<String>,
    ) -> Self {
        Self::many(
            class,
            vec![Finding {
                pointer: pointer.into(),
                detail: truncate(detail.into()),
            }],
        )
    }

    fn many(class: DeclarationErrorClass, findings: Vec<Finding>) -> Self {
        Self {
            class,
            findings,
            accepted: Vec::new(),
        }
    }

    fn version(detail: impl Into<String>) -> Self {
        Self {
            class: DeclarationErrorClass::VersionUnsupported,
            findings: vec![Finding {
                pointer: "/declaration".into(),
                detail: truncate(format!(
                    "{}; accepted: {}",
                    detail.into(),
                    SUPPORTED_DECLARATION_VERSIONS.join(", ")
                )),
            }],
            accepted: SUPPORTED_DECLARATION_VERSIONS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }

    /// The namespaced code (`declaration_error:<class>`).
    pub fn code(&self) -> String {
        self.class.code()
    }
}

impl fmt::Display for DeclarationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.class.code())?;
        for (i, finding) in self.findings.iter().enumerate() {
            let sep = if i == 0 { ": " } else { "; " };
            if finding.pointer.is_empty() {
                write!(f, "{sep}{}", finding.detail)?;
            } else {
                write!(f, "{sep}{}: {}", finding.pointer, finding.detail)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for DeclarationError {}

impl From<DeclarationError> for crate::Error {
    fn from(e: DeclarationError) -> Self {
        crate::Error::Configuration(e.to_string())
    }
}

/// A host registry programming error (an invalid, reserved or duplicate
/// name). The document is not at fault.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistryError {
    /// What is wrong.
    pub detail: String,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "host registry: {}", self.detail)
    }
}

impl std::error::Error for RegistryError {}

impl From<RegistryError> for crate::Error {
    fn from(e: RegistryError) -> Self {
        crate::Error::Configuration(e.to_string())
    }
}

/// Bound a detail at [`MAX_DETAIL_LEN`] characters, the ellipsis included.
fn truncate(s: String) -> String {
    if s.chars().count() <= MAX_DETAIL_LEN {
        return s;
    }
    let mut out: String = s.chars().take(MAX_DETAIL_LEN - 1).collect();
    out.push('…');
    out
}

/// RFC 6901 pointer segment escaping.
fn ptr(parent: &str, key: &str) -> String {
    format!("{parent}/{}", key.replace('~', "~0").replace('/', "~1"))
}

// ---- host surface --------------------------------------------------------------

/// What the host does with the run after a `host_error:*` deny at the tool
/// seam (§6.2, §13.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSeamPosture {
    /// The loop continues with a tool error.
    Continue,
    /// The run halts.
    Terminate,
}

impl ToolSeamPosture {
    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::Terminate => "terminate",
        }
    }
}

/// The knob values a host supports for one profile.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnobSupport {
    /// `on_approval` values.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub on_approval: BTreeSet<String>,
    /// `on_disagreement` values.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub on_disagreement: BTreeSet<String>,
    /// `on_transform_conflict` values.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub on_transform_conflict: BTreeSet<String>,
}

impl KnobSupport {
    /// Every value of every knob the profile consults.
    pub fn full(profile: CompositionProfile) -> Self {
        let mut k = Self::default();
        for knob in consulted_knobs(profile) {
            *k.knob_mut(knob) = knob_values(knob).iter().map(|v| (*v).to_owned()).collect();
        }
        k
    }

    /// Only the §7.2 default value of every knob the profile consults.
    pub fn defaults_only(profile: CompositionProfile) -> Self {
        let mut k = Self::default();
        for knob in consulted_knobs(profile) {
            if let Some(d) = knob_default(knob) {
                k.knob_mut(knob).insert(d.to_owned());
            }
        }
        k
    }

    fn knob(&self, knob: &str) -> &BTreeSet<String> {
        match knob {
            "on_approval" => &self.on_approval,
            "on_disagreement" => &self.on_disagreement,
            _ => &self.on_transform_conflict,
        }
    }

    fn knob_mut(&mut self, knob: &str) -> &mut BTreeSet<String> {
        match knob {
            "on_approval" => &mut self.on_approval,
            "on_disagreement" => &mut self.on_disagreement,
            _ => &mut self.on_transform_conflict,
        }
    }

    fn subset_of(&self, other: &Self) -> bool {
        self.on_approval.is_subset(&other.on_approval)
            && self.on_disagreement.is_subset(&other.on_disagreement)
            && self
                .on_transform_conflict
                .is_subset(&other.on_transform_conflict)
    }
}

/// What this host's code can honour (§7.7.4, §13.1). A document may select
/// a subset of it, never more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostSurface {
    /// The points the host emits.
    pub interception_points: BTreeSet<InterceptionPoint>,
    /// Its capabilities.
    pub capabilities: BTreeSet<String>,
    /// The profiles and knob values it supports.
    pub profiles: BTreeMap<CompositionProfile, KnobSupport>,
    /// Its tool-seam posture.
    pub tool_seam_host_error: ToolSeamPosture,
    /// Whether it can stream caller-bound output unbuffered.
    pub streams_unbuffered: bool,
    /// The declaration versions it accepts.
    pub declaration_versions: BTreeSet<String>,
}

impl HostSurface {
    /// This framework's surface (upstream's `_host_surface()`): every point,
    /// `model_calls`, `tool_calls`, `int64_json`, `bigint_json` and
    /// `host_declaration`, buffered output, every profile and knob value,
    /// posture `terminate`.
    pub fn agent_framework() -> Self {
        Self {
            interception_points: ALL_POINTS.into_iter().collect(),
            capabilities: [
                "model_calls",
                "tool_calls",
                "int64_json",
                "bigint_json",
                "host_declaration",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            profiles: ALL_PROFILES
                .into_iter()
                .map(|p| (p, KnobSupport::full(p)))
                .collect(),
            tool_seam_host_error: ToolSeamPosture::Terminate,
            streams_unbuffered: false,
            declaration_versions: SUPPORTED_DECLARATION_VERSIONS
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        }
    }
}

// ---- registry ------------------------------------------------------------------

/// What a kind resolver learns about the binding it builds (§7.7.5).
#[derive(Debug, Clone)]
pub struct BindingContext<'a> {
    /// The binding id.
    pub id: &'a str,
    /// The binding kind.
    pub kind: &'a str,
    /// The points it runs at.
    pub at: &'a BTreeSet<InterceptionPoint>,
    /// Its resolved timeout; `None` is unbounded.
    pub timeout: Option<Duration>,
    /// The document's informative `host` block.
    pub host: Option<&'a HostInfo>,
    /// The document's contract version.
    pub declaration_version: &'a str,
}

/// Host code that turns one binding's `config` into one interceptor, or
/// refuses it with a message (which must not echo the config).
pub type KindResolver = Arc<
    dyn Fn(&Value, &BindingContext<'_>) -> std::result::Result<Arc<dyn Interceptor>, String>
        + Send
        + Sync,
>;

type IdentityFn = Arc<dyn Fn(&InterceptionContext) -> String + Send + Sync>;

/// Everything a host registers in code for a declaration to reference
/// (§7.7.5): kind resolvers, custom identity providers, approval resolvers
/// and approval redactors, over this host's [`HostSurface`]. A kind is a
/// lookup key, never code the loader imports.
#[derive(Clone)]
pub struct HostRegistry {
    surface: HostSurface,
    kinds: BTreeMap<String, KindResolver>,
    identity_providers: BTreeMap<String, IdentityFn>,
    approval_resolvers: BTreeMap<String, Arc<dyn ApprovalResolver>>,
    approval_redactors: BTreeMap<String, ApprovalRedactor>,
}

impl fmt::Debug for HostRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostRegistry")
            .field("kinds", &self.kinds.keys().collect::<Vec<_>>())
            .field(
                "identity_providers",
                &self.identity_providers.keys().collect::<Vec<_>>(),
            )
            .field(
                "approval_resolvers",
                &self.approval_resolvers.keys().collect::<Vec<_>>(),
            )
            .field(
                "approval_redactors",
                &self.approval_redactors.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl Default for HostRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Reference grammar for resolver and redactor names and binding ids.
fn valid_reference(s: &str) -> bool {
    s.len() <= MAX_ID_LEN && valid_segment(s)
}

fn valid_segment(s: &str) -> bool {
    let mut chars = s.chars();
    chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

/// Binding kind grammar: dot-separated lowercase segments, at least two.
fn valid_kind(s: &str) -> bool {
    s.len() <= MAX_KIND_LEN && {
        let segs: Vec<&str> = s.split('.').collect();
        segs.len() >= 2 && segs.iter().all(|seg| valid_segment(seg))
    }
}

/// Custom identity provider name rules (§10.1).
fn valid_provider_name(s: &str) -> bool {
    valid_reference(s) && !s.starts_with("jcs")
}

impl HostRegistry {
    /// An empty registry over [`HostSurface::agent_framework`].
    pub fn new() -> Self {
        Self {
            surface: HostSurface::agent_framework(),
            kinds: BTreeMap::new(),
            identity_providers: BTreeMap::new(),
            approval_resolvers: BTreeMap::new(),
            approval_redactors: BTreeMap::new(),
        }
    }

    /// The surface documents are checked against.
    pub fn surface(&self) -> &HostSurface {
        &self.surface
    }

    /// Register a kind resolver. Kinds under the reserved `agent_hooks`
    /// and `ctk` segments are refused.
    pub fn kind(
        mut self,
        kind: &str,
        resolver: KindResolver,
    ) -> std::result::Result<Self, RegistryError> {
        if !valid_kind(kind) {
            return Err(RegistryError {
                detail: format!("kind {kind:?} does not match the kind grammar (see spec §7.7.5)"),
            });
        }
        let head = kind.split('.').next().unwrap_or_default();
        if RESERVED_KIND_SEGMENTS.contains(&head) {
            return Err(RegistryError {
                detail: format!("kind {kind:?} uses the reserved segment {head:?}"),
            });
        }
        if self.kinds.contains_key(kind) {
            return Err(RegistryError {
                detail: format!("kind {kind:?} registered twice"),
            });
        }
        self.kinds.insert(kind.to_owned(), resolver);
        Ok(self)
    }

    /// Register [`INTERCEPTOR_KIND`] over interceptors supplied by name.
    /// Its `config` is `{}` or `{"name": "<key>"}`; the key defaults to the
    /// binding id. Anything else refuses the binding.
    pub fn interceptors<'a>(
        self,
        interceptors: impl IntoIterator<Item = (&'a str, Arc<dyn Interceptor>)>,
    ) -> std::result::Result<Self, RegistryError> {
        let table: HashMap<String, Arc<dyn Interceptor>> = interceptors
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v))
            .collect();
        let resolver: KindResolver = Arc::new(move |config, binding| {
            let Some(members) = config.as_object() else {
                return Err(format!(
                    "binding {:?}: config must be an object with an optional 'name' member",
                    binding.id
                ));
            };
            if members.keys().any(|k| k != "name") {
                return Err(format!(
                    "binding {:?}: config accepts only the 'name' member",
                    binding.id
                ));
            }
            let name = match members.get("name") {
                None => binding.id,
                Some(Value::String(s)) => s.as_str(),
                Some(_) => {
                    return Err(format!(
                        "binding {:?}: config.name must be a string",
                        binding.id
                    ))
                }
            };
            table.get(name).cloned().ok_or_else(|| {
                format!(
                    "binding {:?}: no interceptor was supplied under the name it binds",
                    binding.id
                )
            })
        });
        self.kind(INTERCEPTOR_KIND, resolver)
    }

    /// Register a custom identity provider (§10.1 name rules apply).
    pub fn identity_provider(
        mut self,
        name: &str,
        f: impl Fn(&InterceptionContext) -> String + Send + Sync + 'static,
    ) -> std::result::Result<Self, RegistryError> {
        if !valid_provider_name(name) {
            return Err(RegistryError {
                detail: format!(
                    "identity provider name {name:?} must match ^[a-z][a-z0-9_-]{{0,63}}$ and not begin with jcs"
                ),
            });
        }
        if self.identity_providers.contains_key(name) {
            return Err(RegistryError {
                detail: format!("identity provider {name:?} registered twice"),
            });
        }
        self.identity_providers.insert(name.to_owned(), Arc::new(f));
        Ok(self)
    }

    /// Register an approval resolver under a reference name.
    pub fn approval_resolver(
        mut self,
        name: &str,
        resolver: Arc<dyn ApprovalResolver>,
    ) -> std::result::Result<Self, RegistryError> {
        check_reference(name, "approval resolver")?;
        if self.approval_resolvers.contains_key(name) {
            return Err(RegistryError {
                detail: format!("approval resolver {name:?} registered twice"),
            });
        }
        self.approval_resolvers.insert(name.to_owned(), resolver);
        Ok(self)
    }

    /// Register an approval redactor under a reference name.
    pub fn approval_redactor(
        mut self,
        name: &str,
        redactor: ApprovalRedactor,
    ) -> std::result::Result<Self, RegistryError> {
        check_reference(name, "approval redactor")?;
        if self.approval_redactors.contains_key(name) {
            return Err(RegistryError {
                detail: format!("approval redactor {name:?} registered twice"),
            });
        }
        self.approval_redactors.insert(name.to_owned(), redactor);
        Ok(self)
    }

    /// Steps 7 (on the filled surface), 8, 9 and 10: resolve a validated
    /// declaration against this host's surface and registered names.
    pub fn resolve(
        &self,
        declaration: &HostDeclaration,
    ) -> std::result::Result<ResolvedDeclaration, DeclarationError> {
        let doc = &declaration.doc;
        check_consistency(doc, Some(&self.surface))?;
        let resolved = fill(doc, &self.surface);
        check_surface(doc, &resolved, &self.surface)?;
        self.check_names(&resolved)?;
        Ok(resolved)
    }

    fn check_names(
        &self,
        resolved: &ResolvedDeclaration,
    ) -> std::result::Result<(), DeclarationError> {
        let mut f = Vec::new();
        let cfg = &resolved.configuration;
        if let Some(p) = cfg.identity_provider.as_deref() {
            if p != JCS_SHA256 && !self.identity_providers.contains_key(p) {
                f.push(Finding {
                    pointer: "/configuration/identity_provider".into(),
                    detail: format!("no identity provider named {p:?} is registered"),
                });
            }
        }
        if let Some(r) = &cfg.approval.resolver {
            if !self.approval_resolvers.contains_key(r) {
                f.push(Finding {
                    pointer: "/configuration/approval/resolver".into(),
                    detail: format!("no approval resolver named {r:?} is registered"),
                });
            }
        }
        if let Some(r) = &cfg.approval.redactor {
            if !self.approval_redactors.contains_key(r) {
                f.push(Finding {
                    pointer: "/configuration/approval/redactor".into(),
                    detail: format!("no approval redactor named {r:?} is registered"),
                });
            }
        }
        if !f.is_empty() {
            return Err(DeclarationError::many(
                DeclarationErrorClass::ReferenceUnresolved,
                f,
            ));
        }
        for (i, b) in resolved.bindings.iter().enumerate() {
            if !self.kinds.contains_key(&b.kind) {
                f.push(Finding {
                    pointer: format!("/bindings/{i}/kind"),
                    detail: format!(
                        "no resolver is registered for kind {:?} (binding {:?})",
                        b.kind, b.id
                    ),
                });
            }
        }
        if f.is_empty() {
            Ok(())
        } else {
            Err(DeclarationError::many(
                DeclarationErrorClass::KindUnknown,
                f,
            ))
        }
    }
}

fn check_reference(name: &str, what: &str) -> std::result::Result<(), RegistryError> {
    if valid_reference(name) {
        Ok(())
    } else {
        Err(RegistryError {
            detail: format!("{what} name {name:?} must match ^[a-z][a-z0-9_-]{{0,63}}$"),
        })
    }
}

// ---- the document ----------------------------------------------------------------

/// The informative `host` block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostInfo {
    /// The host's name.
    pub name: String,
    /// Its version.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// A declaration that passed steps 2 to 7: parsed, of an accepted version,
/// closed, well-typed and internally consistent as stated. Not yet checked
/// against a host; see [`HostRegistry::resolve`].
#[derive(Debug, Clone, PartialEq)]
pub struct HostDeclaration {
    doc: Map<String, Value>,
}

impl HostDeclaration {
    /// Step 1, then [`Self::from_json`]: a regular file of at most
    /// [`MAX_DOCUMENT_BYTES`], strict UTF-8 without a
    /// byte-order mark.
    pub fn from_path(path: impl AsRef<Path>) -> std::result::Result<Self, DeclarationError> {
        let path = path.as_ref();
        let unreadable =
            |detail: String| DeclarationError::new(DeclarationErrorClass::Unreadable, "", detail);
        let meta = std::fs::metadata(path)
            .map_err(|e| unreadable(format!("cannot stat: {:?}", e.kind())))?;
        if !meta.is_file() {
            return Err(unreadable("not a regular file".into()));
        }
        let mut bytes = Vec::new();
        std::fs::File::open(path)
            .and_then(|f| {
                f.take(MAX_DOCUMENT_BYTES as u64 + 1)
                    .read_to_end(&mut bytes)
            })
            .map_err(|e| unreadable(format!("cannot read: {:?}", e.kind())))?;
        if bytes.len() > MAX_DOCUMENT_BYTES {
            return Err(unreadable(format!(
                "document exceeds the bound of {MAX_DOCUMENT_BYTES} bytes"
            )));
        }
        if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return Err(unreadable("document starts with a byte-order mark".into()));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|_| unreadable("document is not valid UTF-8".into()))?;
        Self::from_json(text)
    }

    /// Steps 2 to 7 over JSON text.
    pub fn from_json(text: &str) -> std::result::Result<Self, DeclarationError> {
        let Value::Object(doc) = parse_checked(text)? else {
            return Err(DeclarationError::new(
                DeclarationErrorClass::Malformed,
                "",
                "document root is not a JSON object",
            ));
        };
        check_version(&doc)?;
        check_spec(&doc)?;
        check_unknown_fields(&doc)?;
        check_fields(&doc, text)?;
        check_consistency(&doc, None)?;
        Ok(Self { doc })
    }

    /// Steps 2 to 7 over an in-memory value (serialized and handed to
    /// [`Self::from_json`], so the bounds are the same).
    pub fn from_value(value: Value) -> std::result::Result<Self, DeclarationError> {
        let text = serde_json::to_string(&value).map_err(|e| {
            DeclarationError::new(
                DeclarationErrorClass::Malformed,
                "",
                format!("cannot serialize: {e}"),
            )
        })?;
        Self::from_json(&text)
    }

    /// The document's `declaration` value.
    pub fn version(&self) -> &str {
        self.doc
            .get("declaration")
            .and_then(Value::as_str)
            .unwrap_or_default()
    }

    /// The validated document, verbatim.
    pub fn as_value(&self) -> Value {
        Value::Object(self.doc.clone())
    }
}

// ---- step 2: checked parse ----------------------------------------------------------

struct Checked {
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for Checked {
    type Value = Value;
    fn deserialize<D: de::Deserializer<'de>>(self, d: D) -> std::result::Result<Value, D::Error> {
        d.deserialize_any(self)
    }
}

impl<'de> Visitor<'de> for Checked {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }
    fn visit_bool<E: de::Error>(self, v: bool) -> std::result::Result<Value, E> {
        Ok(Value::Bool(v))
    }
    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Value, E> {
        Ok(Value::from(v))
    }
    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Value, E> {
        Ok(Value::from(v))
    }
    fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Value, E> {
        Ok(serde_json::Number::from_f64(v)
            .map(Value::Number)
            .unwrap_or(Value::Null))
    }
    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Value, E> {
        Ok(Value::String(v.to_owned()))
    }
    fn visit_string<E: de::Error>(self, v: String) -> std::result::Result<Value, E> {
        Ok(Value::String(v))
    }
    fn visit_unit<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_none<E: de::Error>(self) -> std::result::Result<Value, E> {
        Ok(Value::Null)
    }
    fn visit_some<D: de::Deserializer<'de>>(self, d: D) -> std::result::Result<Value, D::Error> {
        d.deserialize_any(self)
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Value, A::Error> {
        if self.depth + 1 > MAX_DEPTH {
            return Err(de::Error::custom(format!(
                "nesting deeper than {MAX_DEPTH}"
            )));
        }
        let mut out = Vec::new();
        while let Some(v) = seq.next_element_seed(Checked {
            depth: self.depth + 1,
        })? {
            out.push(v);
        }
        Ok(Value::Array(out))
    }
    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Value, A::Error> {
        if self.depth + 1 > MAX_DEPTH {
            return Err(de::Error::custom(format!(
                "nesting deeper than {MAX_DEPTH}"
            )));
        }
        let mut out = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if out.contains_key(&key) {
                return Err(de::Error::custom(format!("duplicate key {key:?}")));
            }
            let v = map.next_value_seed(Checked {
                depth: self.depth + 1,
            })?;
            out.insert(key, v);
        }
        Ok(Value::Object(out))
    }
}

fn parse_checked(text: &str) -> std::result::Result<Value, DeclarationError> {
    let malformed = |d: String| DeclarationError::new(DeclarationErrorClass::Malformed, "", d);
    if text.len() > MAX_DOCUMENT_BYTES {
        return Err(malformed(format!(
            "document exceeds the bound of {MAX_DOCUMENT_BYTES} bytes"
        )));
    }
    let mut de = serde_json::Deserializer::from_str(text);
    let value = Checked { depth: 0 }
        .deserialize(&mut de)
        .map_err(|e| malformed(format!("not a valid JSON document: {e}")))?;
    de.end()
        .map_err(|e| malformed(format!("trailing content after the document: {e}")))?;
    Ok(value)
}

// ---- steps 3 and 4 ---------------------------------------------------------------------

fn check_version(doc: &Map<String, Value>) -> std::result::Result<(), DeclarationError> {
    match doc.get("declaration") {
        None => Err(DeclarationError::version("declaration is missing")),
        Some(Value::String(v)) if SUPPORTED_DECLARATION_VERSIONS.contains(&v.as_str()) => Ok(()),
        Some(Value::String(v)) => Err(DeclarationError::version(format!(
            "declaration {v:?} is not accepted by this loader"
        ))),
        Some(_) => Err(DeclarationError::version("declaration is not a string")),
    }
}

fn parse_version(s: &str, prefix: &str) -> Option<(u64, u64)> {
    let rest = s.strip_prefix(prefix)?;
    let (maj, min) = rest.split_once('.')?;
    let ok = |p: &str| !p.is_empty() && p.len() <= 9 && p.chars().all(|c| c.is_ascii_digit());
    if !ok(maj) || !ok(min) {
        return None;
    }
    Some((maj.parse().ok()?, min.parse().ok()?))
}

fn check_spec(doc: &Map<String, Value>) -> std::result::Result<(), DeclarationError> {
    let Some(spec) = doc.get("spec") else {
        return Ok(());
    };
    let unsupported =
        |d: String| DeclarationError::new(DeclarationErrorClass::SpecUnsupported, "/spec", d);
    let Some(s) = spec.as_str() else {
        return Err(unsupported("spec is not a string".into()));
    };
    let Some((maj, min)) = parse_version(s, "agent-hooks/") else {
        return Err(unsupported(format!(
            "spec {s:?} is not of the form agent-hooks/<major>.<minor>"
        )));
    };
    let (lmaj, lmin) = parse_version(SPEC_VERSION, "agent-hooks/").unwrap_or((0, 1));
    if maj != lmaj || min > lmin {
        return Err(unsupported(format!(
            "spec {s:?} is not supported by this host (it implements {SPEC_VERSION})"
        )));
    }
    Ok(())
}

// ---- step 5 -----------------------------------------------------------------------------

const ROOT_KEYS: &[&str] = &[
    "$schema",
    "declaration",
    "spec",
    "id",
    "host",
    "configuration",
    "surface",
    "bindings",
    "extensions",
];
const HOST_KEYS: &[&str] = &["name", "version"];
const CONFIGURATION_KEYS: &[&str] = &[
    "mode",
    "composition",
    "identity_provider",
    "approval",
    "posture",
    "timeouts",
    "records",
];
const COMPOSITION_KEYS: &[&str] = &[
    "profile",
    "on_approval",
    "on_disagreement",
    "on_transform_conflict",
];
const APPROVAL_KEYS: &[&str] = &["resolver", "redactor"];
const POSTURE_KEYS: &[&str] = &["tool_seam_host_error"];
const TIMEOUT_KEYS: &[&str] = &["interceptor_ms", "approval_resolver_ms"];
const RECORD_KEYS: &[&str] = &["max_buffered"];
const SURFACE_KEYS: &[&str] = &[
    "interception_points",
    "capabilities",
    "profiles",
    "buffered_output",
    "exposure_bound",
    "declaration_versions",
];
const BINDING_KEYS: &[&str] = &["id", "kind", "config", "at", "timeout_ms"];

fn unknown_in(obj: &Map<String, Value>, allowed: &[&str], pointer: &str, out: &mut Vec<Finding>) {
    for k in obj.keys() {
        if !allowed.contains(&k.as_str()) {
            out.push(Finding {
                pointer: ptr(pointer, k),
                detail: format!("unknown member {k:?}"),
            });
        }
    }
}

fn get_obj<'a>(m: &'a Map<String, Value>, k: &str) -> Option<&'a Map<String, Value>> {
    m.get(k).and_then(Value::as_object)
}

fn get_str<'a>(m: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(Value::as_str)
}

fn check_unknown_fields(doc: &Map<String, Value>) -> std::result::Result<(), DeclarationError> {
    let mut f = Vec::new();
    unknown_in(doc, ROOT_KEYS, "", &mut f);
    if let Some(host) = get_obj(doc, "host") {
        unknown_in(host, HOST_KEYS, "/host", &mut f);
    }
    if let Some(cfg) = get_obj(doc, "configuration") {
        unknown_in(cfg, CONFIGURATION_KEYS, "/configuration", &mut f);
        for (key, allowed) in [
            ("composition", COMPOSITION_KEYS),
            ("approval", APPROVAL_KEYS),
            ("posture", POSTURE_KEYS),
            ("timeouts", TIMEOUT_KEYS),
            ("records", RECORD_KEYS),
        ] {
            if let Some(sub) = get_obj(cfg, key) {
                unknown_in(sub, allowed, &ptr("/configuration", key), &mut f);
            }
        }
    }
    if let Some(surface) = get_obj(doc, "surface") {
        unknown_in(surface, SURFACE_KEYS, "/surface", &mut f);
        if let Some(profiles) = get_obj(surface, "profiles") {
            for (name, knobs) in profiles {
                let p = ptr("/surface/profiles", name);
                match profile_from_wire(name) {
                    None => f.push(Finding {
                        pointer: p,
                        detail: format!("unknown profile {name:?}"),
                    }),
                    Some(profile) => {
                        if let Some(obj) = knobs.as_object() {
                            unknown_in(obj, consulted_knobs(profile), &p, &mut f);
                        }
                    }
                }
            }
        }
    }
    if let Some(bindings) = doc.get("bindings").and_then(Value::as_array) {
        for (i, b) in bindings.iter().enumerate() {
            if let Some(obj) = b.as_object() {
                unknown_in(obj, BINDING_KEYS, &format!("/bindings/{i}"), &mut f);
            }
        }
    }
    if f.is_empty() {
        Ok(())
    } else {
        Err(DeclarationError::many(
            DeclarationErrorClass::UnknownField,
            f,
        ))
    }
}

// ---- step 6 -----------------------------------------------------------------------------

const INTEGER_DOMAIN_DETAIL: &str =
    "integer exceeds 2^53 - 1 and has no RFC 8785 canonical form; string-encode 64-bit values";

/// The JSON pointer of every integer-syntax token beyond ±(2⁵³−1), read
/// from the text (serde_json turns a literal beyond i64/u64 into an f64,
/// after which it cannot be told from a double).
fn scan_integer_tokens(text: &str) -> Vec<String> {
    const LIMIT: &[u8] = b"9007199254740991";
    enum Frame<'a> {
        Object {
            key: Option<&'a str>,
            want_key: bool,
        },
        Array {
            index: usize,
        },
    }
    fn pointer_of(stack: &[Frame<'_>]) -> String {
        let mut p = String::new();
        for frame in stack {
            match frame {
                Frame::Object { key, .. } => {
                    let raw = key.unwrap_or_default();
                    let decoded: String = serde_json::from_str(&format!("\"{raw}\""))
                        .unwrap_or_else(|_| raw.to_owned());
                    p = ptr(&p, &decoded);
                }
                Frame::Array { index } => {
                    p.push('/');
                    p.push_str(&index.to_string());
                }
            }
        }
        p
    }
    let b = text.as_bytes();
    let mut stack: Vec<Frame<'_>> = Vec::new();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'{' => {
                stack.push(Frame::Object {
                    key: None,
                    want_key: true,
                });
                i += 1;
            }
            b'[' => {
                stack.push(Frame::Array { index: 0 });
                i += 1;
            }
            b'}' | b']' => {
                stack.pop();
                i += 1;
            }
            b',' => {
                match stack.last_mut() {
                    Some(Frame::Object { key, want_key }) => {
                        *key = None;
                        *want_key = true;
                    }
                    Some(Frame::Array { index }) => *index += 1,
                    None => {}
                }
                i += 1;
            }
            b'"' => {
                let start = i + 1;
                i += 1;
                while i < b.len() {
                    match b[i] {
                        b'\\' => i += 2,
                        b'"' => break,
                        _ => i += 1,
                    }
                }
                let end = i.min(b.len());
                i += 1;
                if let Some(Frame::Object { key, want_key }) = stack.last_mut() {
                    if *want_key {
                        *key = Some(&text[start..end]);
                        *want_key = false;
                    }
                }
            }
            b'-' | b'0'..=b'9' => {
                let start = i;
                let mut integer_syntax = true;
                while i < b.len() {
                    match b[i] {
                        b'0'..=b'9' | b'-' | b'+' => i += 1,
                        b'.' | b'e' | b'E' => {
                            integer_syntax = false;
                            i += 1;
                        }
                        _ => break,
                    }
                }
                if integer_syntax {
                    let digits = &b[start..i];
                    let digits = digits.strip_prefix(b"-").unwrap_or(digits);
                    if digits.len() > LIMIT.len() || (digits.len() == LIMIT.len() && digits > LIMIT)
                    {
                        out.push(pointer_of(&stack));
                    }
                }
            }
            _ => i += 1,
        }
    }
    out
}

struct Fields {
    out: Vec<Finding>,
}

impl Fields {
    fn bad(&mut self, pointer: impl Into<String>, detail: impl Into<String>) {
        self.out.push(Finding {
            pointer: pointer.into(),
            detail: truncate(detail.into()),
        });
    }

    fn string<'a>(&mut self, v: Option<&'a Value>, pointer: &str) -> Option<&'a str> {
        match v {
            None => None,
            Some(Value::String(s)) => Some(s.as_str()),
            Some(_) => {
                self.bad(pointer, "must be a string");
                None
            }
        }
    }

    fn object<'a>(
        &mut self,
        v: Option<&'a Value>,
        pointer: &str,
    ) -> Option<&'a Map<String, Value>> {
        match v {
            None => None,
            Some(Value::Object(o)) => Some(o),
            Some(_) => {
                self.bad(pointer, "must be an object");
                None
            }
        }
    }

    fn array<'a>(&mut self, v: Option<&'a Value>, pointer: &str) -> Option<&'a [Value]> {
        match v {
            None => None,
            Some(Value::Array(a)) => Some(a.as_slice()),
            Some(_) => {
                self.bad(pointer, "must be an array");
                None
            }
        }
    }

    fn enum_str(&mut self, v: Option<&Value>, pointer: &str, allowed: &[&str]) {
        if let Some(s) = self.string(v, pointer) {
            if !allowed.contains(&s) {
                self.bad(pointer, format!("{s:?} is not one of {allowed:?}"));
            }
        }
    }

    fn int_or_null(&mut self, v: Option<&Value>, pointer: &str, lo: u64, hi: u64) {
        match v {
            None | Some(Value::Null) => {}
            Some(n) => match n.as_u64() {
                Some(x) if (lo..=hi).contains(&x) => {}
                _ => self.bad(
                    pointer,
                    format!("must be an integer from {lo} to {hi}, or null"),
                ),
            },
        }
    }

    fn string_set(
        &mut self,
        v: Option<&Value>,
        pointer: &str,
        allowed: &[&str],
        min: usize,
    ) -> Option<BTreeSet<String>> {
        let arr = self.array(v, pointer)?;
        let mut seen = BTreeSet::new();
        let mut ok = true;
        for (i, item) in arr.iter().enumerate() {
            let p = format!("{pointer}/{i}");
            match item.as_str() {
                None => {
                    self.bad(&p, "must be a string");
                    ok = false;
                }
                Some(s) if !allowed.is_empty() && !allowed.contains(&s) => {
                    self.bad(&p, format!("{s:?} is not one of {allowed:?}"));
                    ok = false;
                }
                Some(s) => {
                    if !seen.insert(s.to_owned()) {
                        self.bad(&p, format!("{s:?} is listed twice; the member is a set"));
                        ok = false;
                    }
                }
            }
        }
        if arr.len() < min {
            self.bad(pointer, format!("must list at least {min} item(s)"));
            ok = false;
        }
        ok.then_some(seen)
    }
}

fn check_fields(doc: &Map<String, Value>, text: &str) -> std::result::Result<(), DeclarationError> {
    let mut f = Fields { out: Vec::new() };
    let point_names = ALL_POINTS.map(|p| p.as_str());
    f.string(doc.get("$schema"), "/$schema");
    if let Some(id) = f.string(doc.get("id"), "/id") {
        let mut chars = id.chars();
        let ok = id.len() <= MAX_ID_LEN
            && chars
                .next()
                .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
            && chars.all(|c| {
                c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '_' || c == '-'
            });
        if !ok {
            f.bad("/id", "must match ^[a-z0-9][a-z0-9._-]{0,63}$");
        }
    }
    if let Some(host) = f.object(doc.get("host"), "/host") {
        match f.string(host.get("name"), "/host/name") {
            None if !host.contains_key("name") => {
                f.bad("/host/name", "required when host is present")
            }
            Some(n) => {
                let ok = !n.is_empty()
                    && n.len() <= 64
                    && n.chars().all(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-'
                    });
                if !ok {
                    f.bad("/host/name", "must match ^[a-z0-9_-]{1,64}$");
                }
            }
            None => {}
        }
        if let Some(v) = f.string(host.get("version"), "/host/version") {
            if v.is_empty() || v.chars().count() > 64 {
                f.bad("/host/version", "must be 1 to 64 characters");
            }
        }
    }
    if let Some(cfg) = f.object(doc.get("configuration"), "/configuration") {
        f.enum_str(
            cfg.get("mode"),
            "/configuration/mode",
            &["enforce", "evaluate_only"],
        );
        if let Some(c) = f.object(cfg.get("composition"), "/configuration/composition") {
            f.enum_str(
                c.get("profile"),
                "/configuration/composition/profile",
                &ALL_PROFILES.map(|p| p.as_str()),
            );
            for knob in ["on_approval", "on_disagreement", "on_transform_conflict"] {
                f.enum_str(
                    c.get(knob),
                    &ptr("/configuration/composition", knob),
                    knob_values(knob),
                );
            }
        }
        match cfg.get("identity_provider") {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s == JCS_SHA256 => {}
            Some(Value::String(s)) => {
                if !valid_provider_name(s) {
                    f.bad(
                        "/configuration/identity_provider",
                        "custom provider names must match ^[a-z][a-z0-9_-]{0,63}$ and not begin with jcs (see spec §10.1)",
                    );
                }
            }
            Some(_) => f.bad(
                "/configuration/identity_provider",
                "must be a string or null",
            ),
        }
        if let Some(a) = f.object(cfg.get("approval"), "/configuration/approval") {
            for key in APPROVAL_KEYS {
                let p = ptr("/configuration/approval", key);
                match a.get(*key) {
                    None | Some(Value::Null) => {}
                    Some(Value::String(s)) if valid_reference(s) => {}
                    Some(Value::String(_)) => f.bad(&p, "must match ^[a-z][a-z0-9_-]{0,63}$"),
                    Some(_) => f.bad(&p, "must be a string or null"),
                }
            }
        }
        if let Some(p) = f.object(cfg.get("posture"), "/configuration/posture") {
            f.enum_str(
                p.get("tool_seam_host_error"),
                "/configuration/posture/tool_seam_host_error",
                &["continue", "terminate"],
            );
        }
        if let Some(t) = f.object(cfg.get("timeouts"), "/configuration/timeouts") {
            for key in TIMEOUT_KEYS {
                f.int_or_null(
                    t.get(*key),
                    &ptr("/configuration/timeouts", key),
                    1,
                    MAX_TIMEOUT_MS,
                );
            }
        }
        if let Some(r) = f.object(cfg.get("records"), "/configuration/records") {
            f.int_or_null(
                r.get("max_buffered"),
                "/configuration/records/max_buffered",
                1,
                MAX_BUFFERED_RECORDS,
            );
        }
    }
    if let Some(surface) = f.object(doc.get("surface"), "/surface") {
        f.string_set(
            surface.get("interception_points"),
            "/surface/interception_points",
            &point_names,
            4,
        );
        f.string_set(
            surface.get("capabilities"),
            "/surface/capabilities",
            CAPABILITIES,
            0,
        );
        if let Some(profiles) = f.object(surface.get("profiles"), "/surface/profiles") {
            for (name, knobs) in profiles {
                let p = ptr("/surface/profiles", name);
                if let (Some(profile), Some(obj)) =
                    (profile_from_wire(name), f.object(Some(knobs), &p))
                {
                    for knob in consulted_knobs(profile) {
                        if obj.contains_key(*knob) {
                            f.string_set(obj.get(*knob), &ptr(&p, knob), knob_values(knob), 1);
                        }
                    }
                }
            }
        }
        let buffered = match surface.get("buffered_output") {
            None => true,
            Some(Value::Bool(b)) => *b,
            Some(_) => {
                f.bad("/surface/buffered_output", "must be a boolean");
                true
            }
        };
        let has_bound = surface.contains_key("exposure_bound");
        if let Some(b) = f.string(surface.get("exposure_bound"), "/surface/exposure_bound") {
            if b.is_empty() || b.chars().count() > MAX_DETAIL_LEN {
                f.bad(
                    "/surface/exposure_bound",
                    format!("must be 1 to {MAX_DETAIL_LEN} characters"),
                );
            }
        }
        if has_bound && buffered {
            f.bad(
                "/surface/exposure_bound",
                "permitted only when buffered_output is false",
            );
        }
        if !has_bound && !buffered {
            f.bad(
                "/surface/exposure_bound",
                "required when buffered_output is false (see spec §12.1a)",
            );
        }
        if let Some(arr) = f.array(
            surface.get("declaration_versions"),
            "/surface/declaration_versions",
        ) {
            let mut seen = BTreeSet::new();
            for (i, v) in arr.iter().enumerate() {
                let p = format!("/surface/declaration_versions/{i}");
                match v.as_str() {
                    Some(s) if parse_version(s, "agent-hooks-declaration/").is_some() => {
                        if !seen.insert(s) {
                            f.bad(&p, format!("{s:?} is listed twice; the member is a set"));
                        }
                    }
                    Some(_) => f.bad(&p, "must match ^agent-hooks-declaration/[0-9]+\\.[0-9]+$"),
                    None => f.bad(&p, "must be a string"),
                }
            }
            if arr.is_empty() {
                f.bad(
                    "/surface/declaration_versions",
                    "must list at least one version",
                );
            }
        }
    }
    match doc.get("bindings") {
        None => f.bad("/bindings", "required"),
        Some(Value::Array(bindings)) => {
            if bindings.len() > MAX_BINDINGS {
                f.bad(
                    "/bindings",
                    format!("must list at most {MAX_BINDINGS} bindings"),
                );
            }
            for (i, b) in bindings.iter().enumerate() {
                let bp = format!("/bindings/{i}");
                let Some(obj) = f.object(Some(b), &bp) else {
                    continue;
                };
                match f.string(obj.get("id"), &ptr(&bp, "id")) {
                    None if !obj.contains_key("id") => f.bad(ptr(&bp, "id"), "required"),
                    Some(id) if !valid_reference(id) => {
                        f.bad(ptr(&bp, "id"), "must match ^[a-z][a-z0-9_-]{0,63}$")
                    }
                    _ => {}
                }
                match f.string(obj.get("kind"), &ptr(&bp, "kind")) {
                    None if !obj.contains_key("kind") => f.bad(ptr(&bp, "kind"), "required"),
                    Some(k) if !valid_kind(k) => f.bad(
                        ptr(&bp, "kind"),
                        format!(
                            "must match ^[a-z][a-z0-9_-]*(\\.[a-z][a-z0-9_-]*)+$ with at most {MAX_KIND_LEN} characters"
                        ),
                    ),
                    _ => {}
                }
                f.string_set(obj.get("at"), &ptr(&bp, "at"), &point_names, 1);
                f.int_or_null(
                    obj.get("timeout_ms"),
                    &ptr(&bp, "timeout_ms"),
                    1,
                    MAX_TIMEOUT_MS,
                );
            }
        }
        Some(_) => f.bad("/bindings", "must be an array"),
    }
    if let Some(ext) = f.object(doc.get("extensions"), "/extensions") {
        for k in ext.keys() {
            let mut chars = k.chars();
            let ok = chars.next().is_some_and(|c| c.is_ascii_lowercase())
                && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
            if !ok {
                f.bad(
                    ptr("/extensions", k),
                    "extension keys must match ^[a-z][a-z0-9_]*$ (see spec §4.6)",
                );
            }
        }
    }
    for pointer in scan_integer_tokens(text) {
        if !f.out.iter().any(|x| x.pointer == pointer) {
            f.bad(pointer, INTEGER_DOMAIN_DETAIL);
        }
    }
    if f.out.is_empty() {
        Ok(())
    } else {
        Err(DeclarationError::many(
            DeclarationErrorClass::InvalidField,
            f.out,
        ))
    }
}

// ---- resolution ----------------------------------------------------------------------------

/// The resolved configuration: every default filled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedConfiguration {
    /// The enforcement mode.
    pub mode: EnforcementMode,
    /// The composition, knobs resolved.
    pub composition: Composition,
    /// The identity provider name (`None` is unbound).
    pub identity_provider: Option<String>,
    /// The approval references.
    pub approval: ResolvedApproval,
    /// The tool-seam posture.
    pub tool_seam_host_error: ToolSeamPosture,
    /// The interceptor timeout in milliseconds (`None` unbounded).
    pub interceptor_ms: Option<u64>,
    /// The approval resolver timeout in milliseconds (`None` unbounded).
    pub approval_resolver_ms: Option<u64>,
    /// The record buffer bound.
    pub max_buffered: Option<u64>,
}

/// The approval references.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedApproval {
    /// The resolver name.
    pub resolver: Option<String>,
    /// The redactor name.
    pub redactor: Option<String>,
}

/// The resolved surface: the document's, or the host's when it stated
/// none.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedSurface {
    /// The declared points.
    pub interception_points: BTreeSet<InterceptionPoint>,
    /// The declared capabilities.
    pub capabilities: BTreeSet<String>,
    /// The declared profiles.
    pub profiles: BTreeMap<CompositionProfile, KnobSupport>,
    /// Whether caller-bound output is buffered.
    pub buffered_output: bool,
    /// The exposure bound (unbuffered surfaces only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exposure_bound: Option<String>,
    /// The accepted declaration versions.
    pub declaration_versions: BTreeSet<String>,
}

/// One binding with `at` and `timeout_ms` filled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedBinding {
    /// The id (recorded as the interceptor's name).
    pub id: String,
    /// The kind.
    pub kind: String,
    /// The kind's configuration.
    pub config: Value,
    /// The points it runs at.
    pub at: BTreeSet<InterceptionPoint>,
    /// Its timeout in milliseconds (`None` unbounded).
    pub timeout_ms: Option<u64>,
}

/// A declaration resolved against this host: every default filled.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResolvedDeclaration {
    /// The contract version.
    pub declaration: String,
    /// The spec version.
    pub spec: String,
    /// The document id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The informative host block.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host: Option<HostInfo>,
    /// The configuration.
    pub configuration: ResolvedConfiguration,
    /// The surface.
    pub surface: ResolvedSurface,
    /// The bindings.
    pub bindings: Vec<ResolvedBinding>,
    /// The extensions, verbatim.
    pub extensions: Map<String, Value>,
}

fn int_or_null(m: &Map<String, Value>, k: &str) -> Option<Option<u64>> {
    match m.get(k) {
        None => None,
        Some(Value::Null) => Some(None),
        Some(v) => Some(v.as_u64()),
    }
}

fn str_set(m: &Map<String, Value>, k: &str) -> Option<BTreeSet<String>> {
    m.get(k).and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()
    })
}

fn point_set(m: &Map<String, Value>, k: &str) -> Option<BTreeSet<InterceptionPoint>> {
    m.get(k).and_then(Value::as_array).map(|a| {
        a.iter()
            .filter_map(Value::as_str)
            .filter_map(point_from_wire)
            .collect()
    })
}

/// The composition as stated (knobs not yet defaulted).
fn stated_composition(doc: &Map<String, Value>) -> Composition {
    let c = get_obj(doc, "configuration").and_then(|c| get_obj(c, "composition"));
    let knob = |k: &str| c.and_then(|c| get_str(c, k));
    let policy = |k: &str| match knob(k) {
        Some("deny") => Some(SynthesisPolicy::Deny),
        Some("approval") => Some(SynthesisPolicy::Approval),
        _ => None,
    };
    Composition {
        profile: c
            .and_then(|c| get_str(c, "profile"))
            .and_then(profile_from_wire)
            .unwrap_or_default(),
        on_approval: match knob("on_approval") {
            Some("stop") => Some(OnApproval::Stop),
            Some("resume") => Some(OnApproval::Resume),
            _ => None,
        },
        on_disagreement: policy("on_disagreement"),
        on_transform_conflict: policy("on_transform_conflict"),
    }
}

fn stated_profiles(p: &Map<String, Value>) -> BTreeMap<CompositionProfile, KnobSupport> {
    let mut out = BTreeMap::new();
    for (name, knobs) in p {
        let Some(profile) = profile_from_wire(name) else {
            continue;
        };
        let mut support = KnobSupport::defaults_only(profile);
        if let Some(obj) = knobs.as_object() {
            for knob in consulted_knobs(profile) {
                if let Some(set) = str_set(obj, knob) {
                    *support.knob_mut(knob) = set;
                }
            }
        }
        out.insert(profile, support);
    }
    out
}

fn resolve_surface(
    doc: &Map<String, Value>,
    code: Option<&HostSurface>,
) -> Option<ResolvedSurface> {
    let code = code?;
    let Some(s) = get_obj(doc, "surface") else {
        return Some(ResolvedSurface {
            interception_points: code.interception_points.clone(),
            capabilities: code.capabilities.clone(),
            profiles: code.profiles.clone(),
            buffered_output: true,
            exposure_bound: None,
            declaration_versions: code.declaration_versions.clone(),
        });
    };
    Some(ResolvedSurface {
        interception_points: point_set(s, "interception_points")
            .unwrap_or_else(|| code.interception_points.clone()),
        capabilities: str_set(s, "capabilities").unwrap_or_else(|| code.capabilities.clone()),
        profiles: get_obj(s, "profiles")
            .map(stated_profiles)
            .unwrap_or_else(|| code.profiles.clone()),
        buffered_output: s
            .get("buffered_output")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        exposure_bound: get_str(s, "exposure_bound").map(str::to_owned),
        declaration_versions: str_set(s, "declaration_versions")
            .unwrap_or_else(|| code.declaration_versions.clone()),
    })
}

fn composition_knob_values(c: &Composition) -> [(&'static str, Option<&'static str>); 3] {
    let policy = |p: Option<SynthesisPolicy>| {
        p.map(|p| match p {
            SynthesisPolicy::Deny => "deny",
            SynthesisPolicy::Approval => "approval",
        })
    };
    [
        (
            "on_approval",
            c.on_approval.map(|v| match v {
                OnApproval::Stop => "stop",
                OnApproval::Resume => "resume",
            }),
        ),
        ("on_disagreement", policy(c.on_disagreement)),
        ("on_transform_conflict", policy(c.on_transform_conflict)),
    ]
}

/// Step 7. Without `code` only members the document states are checked;
/// with it, the filled surface is checked too.
fn check_consistency(
    doc: &Map<String, Value>,
    code: Option<&HostSurface>,
) -> std::result::Result<(), DeclarationError> {
    let mut f: Vec<Finding> = Vec::new();
    let stated_surface = get_obj(doc, "surface");
    let filled_note = |pointer: &str| -> Option<String> {
        let member = pointer.strip_prefix("/surface/")?.split('/').next()?;
        if stated_surface.is_some_and(|s| s.contains_key(member)) {
            return None;
        }
        Some(format!(
            " (the document does not state /surface/{member}; it was filled from the host surface)"
        ))
    };
    let mut bad = |pointer: &str, detail: String| {
        let detail = match filled_note(pointer) {
            Some(note) => format!("{detail}{note}"),
            None => detail,
        };
        f.push(Finding {
            pointer: pointer.to_owned(),
            detail: truncate(detail),
        })
    };
    let surface = resolve_surface(doc, code);
    let points = surface
        .as_ref()
        .map(|s| s.interception_points.clone())
        .or_else(|| stated_surface.and_then(|s| point_set(s, "interception_points")));
    let caps = surface
        .as_ref()
        .map(|s| s.capabilities.clone())
        .or_else(|| stated_surface.and_then(|s| str_set(s, "capabilities")));
    let profiles = stated_surface
        .and_then(|s| get_obj(s, "profiles"))
        .map(stated_profiles);
    let versions = surface
        .as_ref()
        .map(|s| s.declaration_versions.clone())
        .or_else(|| stated_surface.and_then(|s| str_set(s, "declaration_versions")));
    let buffered = surface
        .as_ref()
        .map(|s| s.buffered_output)
        .or_else(|| {
            stated_surface
                .and_then(|s| s.get("buffered_output"))
                .and_then(Value::as_bool)
        })
        .unwrap_or(true);

    if let Some(points) = &points {
        for p in FLOOR_POINTS {
            if !points.contains(&p) {
                bad(
                    "/surface/interception_points",
                    format!("the §3.2 floor requires {p}; omit only the model or tool points"),
                );
            }
        }
        let has = |p: InterceptionPoint| points.contains(&p);
        if has(InterceptionPoint::PreModelCall) != has(InterceptionPoint::PostModelCall) {
            bad(
                "/surface/interception_points",
                "pre_model_call and post_model_call are omitted together or not at all (see spec §3.2)".into(),
            );
        }
        if has(InterceptionPoint::PreToolCall) != has(InterceptionPoint::PostToolCall) {
            bad(
                "/surface/interception_points",
                "pre_tool_call and post_tool_call are omitted together or not at all (see spec §3.2)".into(),
            );
        }
        if let Some(caps) = &caps {
            let model =
                has(InterceptionPoint::PreModelCall) && has(InterceptionPoint::PostModelCall);
            let tool = has(InterceptionPoint::PreToolCall) && has(InterceptionPoint::PostToolCall);
            if caps.contains("model_calls") != model {
                bad(
                    "/surface/capabilities",
                    "model_calls is listed iff both model points are listed (see spec §3.2)".into(),
                );
            }
            if caps.contains("tool_calls") != tool {
                bad(
                    "/surface/capabilities",
                    "tool_calls is listed iff both tool points are listed (see spec §3.2)".into(),
                );
            }
        }
    }
    if let Some(caps) = &caps {
        if caps.contains("incremental_output") && buffered {
            bad(
                "/surface/capabilities",
                "incremental_output requires buffered_output: false (see spec §12.1)".into(),
            );
        }
    }

    let stated = stated_composition(doc);
    let consulted = consulted_knobs(stated.profile);
    let knob_present = |k: &str| {
        get_obj(doc, "configuration")
            .and_then(|c| get_obj(c, "composition"))
            .is_some_and(|c| c.contains_key(k))
    };
    for knob in ["on_approval", "on_disagreement", "on_transform_conflict"] {
        if knob_present(knob) && !consulted.contains(&knob) {
            bad(
                &ptr("/configuration/composition", knob),
                format!(
                    "{} does not consult {knob}; the member is refused, not cleared",
                    stated.profile.as_str()
                ),
            );
        }
    }
    let resolved = stated.normalized();
    if let Some(profiles) = &profiles {
        match profiles.get(&resolved.profile) {
            None => bad(
                "/configuration/composition/profile",
                format!("{} is not in surface.profiles", resolved.profile.as_str()),
            ),
            Some(support) => {
                for (knob, value) in composition_knob_values(&resolved) {
                    if let Some(v) = value {
                        if !support.knob(knob).contains(v) {
                            bad(
                                &ptr("/configuration/composition", knob),
                                format!(
                                    "{v:?} is not in surface.profiles for {}",
                                    resolved.profile.as_str()
                                ),
                            );
                        }
                    }
                }
            }
        }
    }
    if let (Some(versions), Some(own)) = (&versions, get_str(doc, "declaration")) {
        if !versions.contains(own) {
            bad(
                "/surface/declaration_versions",
                format!("must include the document's own declaration {own:?}"),
            );
        }
    }
    let mut ids = BTreeSet::new();
    for (i, b) in doc
        .get("bindings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let Some(obj) = b.as_object() else { continue };
        if let Some(id) = get_str(obj, "id") {
            if !ids.insert(id) {
                bad(
                    &format!("/bindings/{i}/id"),
                    format!("binding id {id:?} is used twice"),
                );
            }
        }
        if let (Some(at), Some(points)) = (point_set(obj, "at"), &points) {
            for p in at.difference(points) {
                bad(
                    &format!("/bindings/{i}/at"),
                    format!("{p} is not in surface.interception_points"),
                );
            }
        }
    }
    if f.is_empty() {
        Ok(())
    } else {
        Err(DeclarationError::many(
            DeclarationErrorClass::Inconsistent,
            f,
        ))
    }
}

/// Step 8: the resolved surface and configuration against the host.
fn check_surface(
    _doc: &Map<String, Value>,
    resolved: &ResolvedDeclaration,
    code: &HostSurface,
) -> std::result::Result<(), DeclarationError> {
    let mut f = Vec::new();
    let mut bad = |pointer: &str, detail: String| {
        f.push(Finding {
            pointer: pointer.to_owned(),
            detail: truncate(detail),
        })
    };
    let s = &resolved.surface;
    for p in s.interception_points.difference(&code.interception_points) {
        bad(
            "/surface/interception_points",
            format!("the host does not emit {p}"),
        );
    }
    for c in s.capabilities.difference(&code.capabilities) {
        bad(
            "/surface/capabilities",
            format!("the host does not have capability {c:?}"),
        );
    }
    for (profile, support) in &s.profiles {
        match code.profiles.get(profile) {
            None => bad(
                &ptr("/surface/profiles", profile.as_str()),
                "the host does not support this profile".into(),
            ),
            Some(code_support) if !support.subset_of(code_support) => bad(
                &ptr("/surface/profiles", profile.as_str()),
                "the host does not support every listed knob value".into(),
            ),
            Some(_) => {}
        }
    }
    for v in s
        .declaration_versions
        .difference(&code.declaration_versions)
    {
        bad(
            "/surface/declaration_versions",
            format!("the host does not accept {v:?}"),
        );
    }
    let composition = &resolved.configuration.composition;
    match code.profiles.get(&composition.profile) {
        None => bad(
            "/configuration/composition/profile",
            format!(
                "the host does not support profile {}",
                composition.profile.as_str()
            ),
        ),
        Some(support) => {
            for (knob, value) in composition_knob_values(composition) {
                if let Some(v) = value {
                    if !support.knob(knob).contains(v) {
                        bad(
                            &ptr("/configuration/composition", knob),
                            format!(
                                "the host does not support {knob} value {v:?} under {}",
                                composition.profile.as_str()
                            ),
                        );
                    }
                }
            }
        }
    }
    if !s.buffered_output && !code.streams_unbuffered {
        bad(
            "/surface/buffered_output",
            "the host buffers caller-bound output and cannot declare false".into(),
        );
    }
    let posture = resolved.configuration.tool_seam_host_error;
    if posture != code.tool_seam_host_error {
        bad(
            "/configuration/posture/tool_seam_host_error",
            format!(
                "the host implements {}, not {}",
                code.tool_seam_host_error.as_str(),
                posture.as_str()
            ),
        );
    }
    if f.is_empty() {
        Ok(())
    } else {
        Err(DeclarationError::many(
            DeclarationErrorClass::SurfaceUnsupported,
            f,
        ))
    }
}

fn fill(doc: &Map<String, Value>, code: &HostSurface) -> ResolvedDeclaration {
    let cfg = get_obj(doc, "configuration");
    let sub = |k: &str| cfg.and_then(|c| get_obj(c, k));
    let mode = match cfg.and_then(|c| get_str(c, "mode")) {
        Some("evaluate_only") => EnforcementMode::EvaluateOnly,
        _ => EnforcementMode::Enforce,
    };
    let identity_provider = match cfg.and_then(|c| c.get("identity_provider")) {
        None => Some(JCS_SHA256.to_owned()),
        Some(Value::Null) => None,
        Some(v) => v.as_str().map(str::to_owned),
    };
    let approval = sub("approval");
    let interceptor_ms = sub("timeouts")
        .and_then(|t| int_or_null(t, "interceptor_ms"))
        .unwrap_or(Some(5000));
    let approval_resolver_ms = sub("timeouts")
        .and_then(|t| int_or_null(t, "approval_resolver_ms"))
        .unwrap_or(interceptor_ms);
    let surface = resolve_surface(doc, Some(code)).unwrap_or_else(|| ResolvedSurface {
        interception_points: code.interception_points.clone(),
        capabilities: code.capabilities.clone(),
        profiles: code.profiles.clone(),
        buffered_output: true,
        exposure_bound: None,
        declaration_versions: code.declaration_versions.clone(),
    });
    let bindings = doc
        .get("bindings")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_object)
        .map(|b| ResolvedBinding {
            id: get_str(b, "id").unwrap_or_default().to_owned(),
            kind: get_str(b, "kind").unwrap_or_default().to_owned(),
            config: b
                .get("config")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
            at: point_set(b, "at").unwrap_or_else(|| surface.interception_points.clone()),
            timeout_ms: int_or_null(b, "timeout_ms").unwrap_or(interceptor_ms),
        })
        .collect();
    ResolvedDeclaration {
        declaration: get_str(doc, "declaration").unwrap_or_default().to_owned(),
        spec: get_str(doc, "spec").unwrap_or(SPEC_VERSION).to_owned(),
        id: get_str(doc, "id").map(str::to_owned),
        host: get_obj(doc, "host").map(|h| HostInfo {
            name: get_str(h, "name").unwrap_or_default().to_owned(),
            version: get_str(h, "version").map(str::to_owned),
        }),
        configuration: ResolvedConfiguration {
            mode,
            composition: stated_composition(doc).normalized(),
            identity_provider,
            approval: ResolvedApproval {
                resolver: approval
                    .and_then(|a| get_str(a, "resolver"))
                    .map(str::to_owned),
                redactor: approval
                    .and_then(|a| get_str(a, "redactor"))
                    .map(str::to_owned),
            },
            tool_seam_host_error: match sub("posture")
                .and_then(|p| get_str(p, "tool_seam_host_error"))
            {
                Some("terminate") => ToolSeamPosture::Terminate,
                _ => ToolSeamPosture::Continue,
            },
            interceptor_ms,
            approval_resolver_ms,
            max_buffered: sub("records")
                .and_then(|r| int_or_null(r, "max_buffered"))
                .unwrap_or(None),
        },
        surface,
        bindings,
        extensions: get_obj(doc, "extensions").cloned().unwrap_or_default(),
    }
}

// ---- building the options --------------------------------------------------------------------

impl AgentHooksOptions {
    /// Steps 7 to 11: resolve `declaration` against `registry` and build
    /// the options it fixes. Each binding's interceptor runs only at its
    /// `at` points with its own timeout and is named by its id on records;
    /// every record carries the declaration version.
    ///
    /// `records.max_buffered` is accepted but has nothing to bound: the
    /// bundle delivers records to the sink and buffers none.
    pub fn from_declaration(
        declaration: &HostDeclaration,
        registry: &HostRegistry,
    ) -> std::result::Result<Self, DeclarationError> {
        let resolved = registry.resolve(declaration)?;
        let cfg = &resolved.configuration;
        let ms = |v: Option<u64>| v.map(Duration::from_millis);
        let mut options = AgentHooksOptions::new()
            .mode(cfg.mode)
            .composition(cfg.composition)
            .timeout(ms(cfg.interceptor_ms))
            .approval_resolver_timeout(ms(cfg.approval_resolver_ms))
            .declaration_version(resolved.declaration.clone());
        let vanished = |pointer: &str, what: &str| {
            DeclarationError::new(
                DeclarationErrorClass::ReferenceUnresolved,
                pointer,
                format!("{what} vanished from the registry"),
            )
        };
        options = options.identity_provider(match cfg.identity_provider.as_deref() {
            None => IdentityProvider::Null,
            Some(JCS_SHA256) => IdentityProvider::JcsSha256,
            Some(name) => {
                let f = registry
                    .identity_providers
                    .get(name)
                    .cloned()
                    .ok_or_else(|| vanished("/configuration/identity_provider", name))?;
                IdentityProvider::Custom {
                    name: name.to_owned(),
                    f,
                }
            }
        });
        if let Some(name) = &cfg.approval.resolver {
            let resolver = registry
                .approval_resolvers
                .get(name)
                .cloned()
                .ok_or_else(|| vanished("/configuration/approval/resolver", name))?;
            options = options.approval_resolver(resolver);
        }
        if let Some(name) = &cfg.approval.redactor {
            let redactor = registry
                .approval_redactors
                .get(name)
                .cloned()
                .ok_or_else(|| vanished("/configuration/approval/redactor", name))?;
            options = options.approval_redactor(redactor);
        }
        for (i, b) in resolved.bindings.iter().enumerate() {
            let pointer = format!("/bindings/{i}");
            let resolver = registry.kinds.get(&b.kind).ok_or_else(|| {
                DeclarationError::new(
                    DeclarationErrorClass::KindUnknown,
                    format!("{pointer}/kind"),
                    format!("kind {:?} vanished from the registry", b.kind),
                )
            })?;
            let timeout = ms(b.timeout_ms);
            let context = BindingContext {
                id: &b.id,
                kind: &b.kind,
                at: &b.at,
                timeout,
                host: resolved.host.as_ref(),
                declaration_version: &resolved.declaration,
            };
            // A resolver error or panic refuses the document; the config is
            // never echoed by the loader.
            let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                resolver(&b.config, &context)
            }));
            let interceptor = match built {
                Ok(Ok(i)) => i,
                Ok(Err(msg)) => {
                    return Err(DeclarationError::new(
                        DeclarationErrorClass::BindingRejected,
                        pointer,
                        format!("binding {:?} (kind {:?}) rejected: {msg}", b.id, b.kind),
                    ))
                }
                Err(_) => {
                    return Err(DeclarationError::new(
                        DeclarationErrorClass::BindingRejected,
                        pointer,
                        format!(
                            "binding {:?} (kind {:?}) rejected: resolver panicked",
                            b.id, b.kind
                        ),
                    ))
                }
            };
            options =
                options.bound_interceptor(b.id.clone(), interceptor, b.at.iter().copied(), timeout);
        }
        Ok(options)
    }
}

impl AgentHooks {
    /// A bundle configured by a host declaration (upstream
    /// `create_agent_hooks_middleware_from_declaration`): the document is
    /// resolved against `registry` once here, so a refused document fails at
    /// construction, before any run. A declaration with no bindings is
    /// refused, since an emitter with none fails closed on every emission.
    pub fn from_declaration(
        declaration: &HostDeclaration,
        registry: &HostRegistry,
    ) -> crate::Result<Self> {
        let options = AgentHooksOptions::from_declaration(declaration, registry)?;
        AgentHooks::new(options)
    }
}

#[cfg(test)]
mod tests;
