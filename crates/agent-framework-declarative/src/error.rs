//! Errors produced while parsing declarative specs and building agents/workflows.

use agent_framework_core::error::Error as CoreError;

/// The result type for the declarative crate.
pub type Result<T> = std::result::Result<T, DeclarativeError>;

/// Errors surfaced by the declarative loader.
///
/// Every variant carries an actionable message: unknown `kind` values,
/// unsupported fields, unresolved environment variables, and missing registry
/// bindings all report *what* was wrong and *where*, rather than being silently
/// dropped.
#[derive(Debug, thiserror::Error)]
pub enum DeclarativeError {
    /// The YAML/JSON was syntactically invalid or did not match the spec shape.
    /// Wraps the underlying `serde_yaml` message (which includes a field path).
    #[error("failed to parse declarative spec: {0}")]
    Parse(String),

    /// Serializing a spec back to YAML failed.
    #[error("failed to serialize declarative spec: {0}")]
    Serialize(String),

    /// A spec used a `kind` this loader does not support.
    #[error("unsupported {what} kind {kind:?}{expected}", expected = expected_hint(.expected))]
    UnsupportedKind {
        /// What the kind was for (e.g. "agent", "tool", "connection").
        what: &'static str,
        /// The offending kind value.
        kind: String,
        /// The set of accepted kinds, for the hint.
        expected: Vec<&'static str>,
    },

    /// A required field was missing for the selected `kind`.
    #[error("{context}: missing required field {field:?}")]
    MissingField {
        /// A human-readable context (e.g. `tool 'search' (kind: mcp)`).
        context: String,
        /// The missing field name (matching the schema vocabulary).
        field: &'static str,
    },

    /// A `${VAR}` interpolation referenced an env var that is not set and has no
    /// `:-default`.
    #[error(
        "environment variable {0:?} is not set and no default was provided (use ${{{0}:-default}})"
    )]
    MissingEnvVar(String),

    /// A `${...}` placeholder was malformed (e.g. unterminated).
    #[error("malformed environment placeholder in value {value:?}: {reason}")]
    MalformedPlaceholder {
        /// The offending string value.
        value: String,
        /// Why it was rejected.
        reason: String,
    },

    /// A spec referenced a name that was not found in the relevant registry.
    #[error("{kind} {name:?} was not found in the {registry} registry")]
    UnknownReference {
        /// The category of the missing reference (e.g. "agent", "predicate").
        kind: &'static str,
        /// The referenced name.
        name: String,
        /// Which registry was consulted.
        registry: &'static str,
    },

    /// No [`ChatClientFactory`](crate::ChatClientFactory) was registered for the
    /// provider implied by the spec's `model` block.
    #[error("no chat-client factory registered for provider {0:?}")]
    NoClientFactory(String),

    /// The spec's structure was internally inconsistent (e.g. a workflow that is
    /// neither orchestration shorthand nor an explicit graph).
    #[error("invalid declarative spec: {0}")]
    Invalid(String),

    /// A condition mini-expression could not be parsed.
    #[error("invalid edge condition {expr:?}: {reason}")]
    InvalidCondition {
        /// The offending expression.
        expr: String,
        /// Why it was rejected.
        reason: String,
    },

    /// An error bubbled up from the core framework (e.g. workflow build/run).
    #[error(transparent)]
    Core(#[from] CoreError),
}

fn expected_hint(expected: &[&'static str]) -> String {
    if expected.is_empty() {
        String::new()
    } else {
        format!("; expected one of: {}", expected.join(", "))
    }
}

impl DeclarativeError {
    /// Convenience constructor for [`DeclarativeError::MissingField`].
    pub(crate) fn missing_field(context: impl Into<String>, field: &'static str) -> Self {
        DeclarativeError::MissingField {
            context: context.into(),
            field,
        }
    }
}
