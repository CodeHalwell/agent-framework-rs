//! Wire (de)serialization for [`ToolDefinition`] within a
//! [`super::PromptAgentDefinition`].
//!
//! `ToolDefinition` does not itself implement `Serialize`/`Deserialize`: a
//! function tool carries a `dyn Tool` local executor, which isn't
//! serializable. This module captures only the declarative fields a Prompt
//! Agent definition needs — name, description, JSON-schema parameters, kind
//! (including a hosted MCP tool's server URL/allow-list), and whether it
//! requires approval — and round-trips them through a private wire struct.
//! Deserializing a tool always produces `executor: None`: a definition read
//! back describes what a tool *is*, not a live local implementation to call
//! it.

use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The wire shape of one [`ToolDefinition`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ToolDefinitionWire {
    name: String,
    description: String,
    #[serde(default)]
    parameters: Value,
    #[serde(flatten)]
    kind: ToolKindWire,
    #[serde(default, skip_serializing_if = "is_false")]
    requires_approval: bool,
}

fn is_false(b: &bool) -> bool {
    !*b
}

/// The wire shape of [`ToolKind`]: an internally-tagged (`"kind"`) enum,
/// flattened into [`ToolDefinitionWire`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ToolKindWire {
    Function,
    HostedCodeInterpreter,
    HostedImageGeneration,
    HostedWebSearch,
    HostedFileSearch {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_results: Option<u32>,
    },
    HostedMcp {
        url: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        allowed_tools: Option<Vec<String>>,
    },
}

impl From<&ToolKind> for ToolKindWire {
    fn from(kind: &ToolKind) -> Self {
        match kind {
            ToolKind::Function => ToolKindWire::Function,
            ToolKind::HostedCodeInterpreter => ToolKindWire::HostedCodeInterpreter,
            ToolKind::HostedImageGeneration => ToolKindWire::HostedImageGeneration,
            ToolKind::HostedWebSearch => ToolKindWire::HostedWebSearch,
            ToolKind::HostedFileSearch { max_results } => ToolKindWire::HostedFileSearch {
                max_results: *max_results,
            },
            ToolKind::HostedMcp { url, allowed_tools } => ToolKindWire::HostedMcp {
                url: url.clone(),
                allowed_tools: allowed_tools.clone(),
            },
        }
    }
}

impl From<ToolKindWire> for ToolKind {
    fn from(wire: ToolKindWire) -> Self {
        match wire {
            ToolKindWire::Function => ToolKind::Function,
            ToolKindWire::HostedCodeInterpreter => ToolKind::HostedCodeInterpreter,
            ToolKindWire::HostedImageGeneration => ToolKind::HostedImageGeneration,
            ToolKindWire::HostedWebSearch => ToolKind::HostedWebSearch,
            ToolKindWire::HostedFileSearch { max_results } => {
                ToolKind::HostedFileSearch { max_results }
            }
            ToolKindWire::HostedMcp { url, allowed_tools } => {
                ToolKind::HostedMcp { url, allowed_tools }
            }
        }
    }
}

impl From<&ToolDefinition> for ToolDefinitionWire {
    fn from(tool: &ToolDefinition) -> Self {
        Self {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            kind: ToolKindWire::from(&tool.kind),
            requires_approval: tool.requires_approval(),
        }
    }
}

impl From<ToolDefinitionWire> for ToolDefinition {
    fn from(wire: ToolDefinitionWire) -> Self {
        let approval_mode = if wire.requires_approval {
            ApprovalMode::AlwaysRequire
        } else {
            ApprovalMode::NeverRequire
        };
        ToolDefinition {
            name: wire.name,
            description: wire.description,
            parameters: wire.parameters,
            kind: ToolKind::from(wire.kind),
            approval_mode,
            executor: None,
        }
    }
}

/// `#[serde(with = "tool_definition_wire::vec")]` for a `Vec<ToolDefinition>`
/// field — see the [module docs](self).
pub(crate) mod vec {
    use super::ToolDefinitionWire;
    use agent_framework_core::tools::ToolDefinition;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(tools: &[ToolDefinition], s: S) -> Result<S::Ok, S::Error> {
        let wire: Vec<ToolDefinitionWire> = tools.iter().map(ToolDefinitionWire::from).collect();
        wire.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<ToolDefinition>, D::Error> {
        let wire = Vec::<ToolDefinitionWire>::deserialize(d)?;
        Ok(wire.into_iter().map(ToolDefinition::from).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{
        hosted_file_search, hosted_mcp, hosted_web_search, FunctionTool,
    };
    use serde_json::json;

    #[test]
    fn function_tool_round_trips_through_wire_shape() {
        let tool = FunctionTool::new(
            "get_weather",
            "Get the weather",
            json!({"type": "object", "properties": {}}),
            |_| async { Ok(json!("ok")) },
        )
        .into_definition()
        .require_approval();
        let wire = ToolDefinitionWire::from(&tool);
        let json = serde_json::to_value(&wire).unwrap();
        assert_eq!(json["kind"], json!("function"));
        assert_eq!(json["requires_approval"], json!(true));

        let back: ToolDefinition = wire.into();
        assert_eq!(back.name, "get_weather");
        assert_eq!(back.kind, ToolKind::Function);
        assert!(back.requires_approval());
        assert!(
            back.executor.is_none(),
            "wire tools carry no local executor"
        );
    }

    #[test]
    fn hosted_mcp_tool_round_trips_url_and_allowed_tools() {
        let tool = hosted_mcp(
            "my_server",
            "https://mcp.example.com",
            Some(vec!["a".to_string(), "b".to_string()]),
        );
        let wire = ToolDefinitionWire::from(&tool);
        let back: ToolDefinition = wire.into();
        assert_eq!(
            back.kind,
            ToolKind::HostedMcp {
                url: "https://mcp.example.com".into(),
                allowed_tools: Some(vec!["a".into(), "b".into()]),
            }
        );
    }

    #[test]
    fn hosted_file_search_and_web_search_kinds_round_trip() {
        for tool in [hosted_file_search(Some(5)), hosted_web_search()] {
            let wire = ToolDefinitionWire::from(&tool);
            let back: ToolDefinition = wire.into();
            assert_eq!(back.kind, tool.kind);
        }
    }
}
