//! Per-request chat options and tool-choice mode.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::tools::ToolDefinition;

/// If and how tools may be used for a request. Mirrors the Python `ToolMode`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolMode {
    /// The model decides whether to call tools.
    Auto,
    /// The model must call a tool; optionally a specific named function.
    Required(Option<String>),
    /// Tools are disabled.
    #[default]
    None,
}

impl ToolMode {
    pub fn required(function_name: Option<String>) -> Self {
        ToolMode::Required(function_name)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ToolMode::Auto => "auto",
            ToolMode::Required(_) => "required",
            ToolMode::None => "none",
        }
    }
}

impl Serialize for ToolMode {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ToolMode {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Ok(match s.as_str() {
            "auto" => ToolMode::Auto,
            "required" => ToolMode::Required(None),
            _ => ToolMode::None,
        })
    }
}

/// Common per-request settings for a chat/AI service.
///
/// The [`ChatOptions::merge`] method implements the Python `&` operator: the
/// right-hand side wins per scalar field, list/map fields are combined, and
/// `instructions` are newline-concatenated.
#[derive(Debug, Clone, Default)]
pub struct ChatOptions {
    pub model_id: Option<String>,
    pub allow_multiple_tool_calls: Option<bool>,
    pub conversation_id: Option<String>,
    pub frequency_penalty: Option<f32>,
    pub instructions: Option<String>,
    pub logit_bias: Option<HashMap<String, f32>>,
    pub max_tokens: Option<u32>,
    pub metadata: Option<HashMap<String, String>>,
    pub presence_penalty: Option<f32>,
    /// A JSON schema describing the desired structured output.
    pub response_format: Option<serde_json::Value>,
    pub seed: Option<i64>,
    pub stop: Option<Vec<String>>,
    pub store: Option<bool>,
    pub temperature: Option<f32>,
    pub tool_choice: Option<ToolMode>,
    pub tools: Vec<ToolDefinition>,
    pub top_p: Option<f32>,
    pub user: Option<String>,
    pub additional_properties: HashMap<String, serde_json::Value>,
}

impl ChatOptions {
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the instructions (system prompt).
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Builder: set the model id.
    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = Some(model_id.into());
        self
    }

    /// Builder: set the temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder: set the max output tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder: add a tool.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    /// Builder: set the tool choice.
    pub fn with_tool_choice(mut self, mode: ToolMode) -> Self {
        self.tool_choice = Some(mode);
        self
    }

    /// Merge `other` into `self`, with `other` taking precedence per the
    /// Python `ChatOptions.__and__` semantics.
    pub fn merge(mut self, other: ChatOptions) -> ChatOptions {
        macro_rules! take {
            ($field:ident) => {
                if other.$field.is_some() {
                    self.$field = other.$field;
                }
            };
        }
        take!(model_id);
        take!(allow_multiple_tool_calls);
        take!(conversation_id);
        take!(frequency_penalty);
        take!(logit_bias);
        take!(max_tokens);
        take!(metadata);
        take!(presence_penalty);
        take!(response_format);
        take!(seed);
        take!(stop);
        take!(store);
        take!(temperature);
        take!(tool_choice);
        take!(top_p);
        take!(user);

        // instructions: newline-concatenate.
        self.instructions = match (self.instructions.take(), other.instructions) {
            (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
            (Some(a), None) => Some(a),
            (None, b) => b,
        };

        // tools: combine, de-duplicating by name.
        for t in other.tools {
            if !self.tools.iter().any(|existing| existing.name == t.name) {
                self.tools.push(t);
            }
        }

        // additional_properties: combine.
        self.additional_properties
            .extend(other.additional_properties);
        self
    }
}
