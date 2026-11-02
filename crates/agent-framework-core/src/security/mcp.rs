//! Mapping MCP tool annotations to restriction-only security declarations.

use serde_json::Value;

use super::labels::{ConfidentialityLabel, IntegrityLabel};
use super::scope::{ToolSecurity, ToolSecurityRegistry};

/// Map MCP `ToolAnnotations` to `(integrity, max_confidentiality,
/// accepts_untrusted)`.
///
/// Server annotations are hints, not policy authority: they may only make
/// the local default stricter. `openWorldHint: true` makes results
/// untrusted; every remote tool keeps the `PUBLIC` cap and rejects an
/// untrusted context.
pub fn map_mcp_annotations_to_labels(
    annotations: Option<&Value>,
    default_integrity: IntegrityLabel,
) -> (IntegrityLabel, Option<ConfidentialityLabel>, bool) {
    let open_world = annotations
        .and_then(|a| a.get("openWorldHint").or_else(|| a.get("open_world_hint")))
        .and_then(Value::as_bool);
    let integrity = if open_world == Some(true) {
        IntegrityLabel::Untrusted
    } else {
        default_integrity
    };
    (integrity, Some(ConfidentialityLabel::Public), false)
}

/// The declarations for one MCP tool: a local `override_labels` wins over
/// the server's annotations; `mark_write_tools_as_sinks` applies the
/// confidentiality cap as `max_allowed_confidentiality`.
pub fn mcp_tool_security(
    annotations: Option<&Value>,
    default_integrity: IntegrityLabel,
    override_labels: Option<(IntegrityLabel, Option<ConfidentialityLabel>)>,
    mark_write_tools_as_sinks: bool,
) -> ToolSecurity {
    let (integrity, max_confidentiality, accepts_untrusted) = match override_labels {
        Some((integrity, max)) => (integrity, max, false),
        None => map_mcp_annotations_to_labels(annotations, default_integrity),
    };
    let mut security = ToolSecurity::new()
        .source_integrity(integrity)
        .accepts_untrusted(accepts_untrusted);
    if mark_write_tools_as_sinks {
        security.max_allowed_confidentiality = max_confidentiality;
    }
    security
}

/// Declare every `(tool name, annotations)` pair (an MCP server's
/// `tools/list`) in `registry` (upstream `apply_mcp_security_labels`).
pub fn apply_mcp_security_labels<'a>(
    registry: &ToolSecurityRegistry,
    tools: impl IntoIterator<Item = (&'a str, Option<&'a Value>)>,
    default_integrity: IntegrityLabel,
    overrides: &std::collections::HashMap<String, (IntegrityLabel, Option<ConfidentialityLabel>)>,
    mark_write_tools_as_sinks: bool,
) {
    for (name, annotations) in tools {
        registry.declare(
            name,
            mcp_tool_security(
                annotations,
                default_integrity,
                overrides.get(name).copied(),
                mark_write_tools_as_sinks,
            ),
        );
    }
}
