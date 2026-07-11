//! Per-request chat options, structured-output format, and tool-choice mode.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::tools::ToolDefinition;

/// The requested structured-output format for a response.
///
/// Mirrors the Python `ChatOptions.response_format` / .NET `ChatResponseFormat`.
/// The serialized form matches the OpenAI `response_format` request object, so a
/// provider can map it with `serde_json::to_value(&format)`:
///
/// * [`ResponseFormat::Text`] → `{"type":"text"}`
/// * [`ResponseFormat::JsonObject`] → `{"type":"json_object"}`
/// * [`ResponseFormat::JsonSchema`] →
///   `{"type":"json_schema","json_schema":{"name":…,"schema":…}}`
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    /// Free-form text (the default provider behavior).
    Text,
    /// A syntactically valid JSON object, without an enforced schema.
    JsonObject,
    /// A JSON object conforming to the given JSON Schema.
    JsonSchema {
        name: String,
        description: Option<String>,
        schema: Value,
        strict: Option<bool>,
    },
}

impl ResponseFormat {
    /// Build a [`ResponseFormat::JsonSchema`] from a name and schema.
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        ResponseFormat::JsonSchema {
            name: name.into(),
            description: None,
            schema,
            strict: None,
        }
    }

    /// The provider-facing wire object (same as `serde_json::to_value(self)`).
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

impl Serialize for ResponseFormat {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        match self {
            ResponseFormat::Text => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("type", "text")?;
                m.end()
            }
            ResponseFormat::JsonObject => {
                let mut m = s.serialize_map(Some(1))?;
                m.serialize_entry("type", "json_object")?;
                m.end()
            }
            ResponseFormat::JsonSchema {
                name,
                description,
                schema,
                strict,
            } => {
                let mut inner = serde_json::Map::new();
                inner.insert("name".into(), Value::String(name.clone()));
                if let Some(d) = description {
                    inner.insert("description".into(), Value::String(d.clone()));
                }
                inner.insert("schema".into(), schema.clone());
                if let Some(st) = strict {
                    inner.insert("strict".into(), Value::Bool(*st));
                }
                let mut m = s.serialize_map(Some(2))?;
                m.serialize_entry("type", "json_schema")?;
                m.serialize_entry("json_schema", &Value::Object(inner))?;
                m.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for ResponseFormat {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Value::deserialize(d)?;
        let ty = value.get("type").and_then(Value::as_str).unwrap_or("text");
        match ty {
            "json_object" => Ok(ResponseFormat::JsonObject),
            "json_schema" => {
                let inner = value.get("json_schema").cloned().unwrap_or(Value::Null);
                let name = inner
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let description = inner
                    .get("description")
                    .and_then(Value::as_str)
                    .map(String::from);
                let schema = inner
                    .get("schema")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({}));
                let strict = inner.get("strict").and_then(Value::as_bool);
                Ok(ResponseFormat::JsonSchema {
                    name,
                    description,
                    schema,
                    strict,
                })
            }
            _ => Ok(ResponseFormat::Text),
        }
    }
}

/// If and how tools may be used for a request. Mirrors the Python `ToolMode`.
///
/// The four modes are `Auto`, "required any" ([`ToolMode::required_any`], i.e.
/// `Required(None)`), a specific required function
/// ([`ToolMode::required_function`], i.e. `Required(Some(name))`), and `None`
/// (tools disabled). Like Python's `serialize_model`, serialization emits only
/// the mode string; the specific function name is applied by the provider
/// mapping, not persisted on the mode itself.
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
    /// Construct a `Required` mode, optionally pinning a specific function.
    pub fn required(function_name: Option<String>) -> Self {
        ToolMode::Required(function_name)
    }

    /// The model decides whether to call tools.
    pub fn auto() -> Self {
        ToolMode::Auto
    }

    /// Tools are disabled.
    pub fn none() -> Self {
        ToolMode::None
    }

    /// The model must call some tool, but any tool is acceptable.
    pub fn required_any() -> Self {
        ToolMode::Required(None)
    }

    /// The model must call the named function.
    pub fn required_function(function_name: impl Into<String>) -> Self {
        ToolMode::Required(Some(function_name.into()))
    }

    /// The specific function name pinned by a `Required` mode, if any.
    pub fn required_function_name(&self) -> Option<&str> {
        match self {
            ToolMode::Required(Some(name)) => Some(name.as_str()),
            _ => None,
        }
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
    /// The requested structured-output format, if any.
    pub response_format: Option<ResponseFormat>,
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

    /// Builder: set the structured-output response format.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = Some(format);
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
        take!(max_tokens);
        take!(presence_penalty);
        // Map-valued fields merge per key (the overriding side wins on
        // conflicts) rather than replacing the whole map, so client-level
        // defaults survive request-level additions.
        if let Some(other_map) = other.logit_bias {
            match &mut self.logit_bias {
                Some(mine) => mine.extend(other_map),
                None => self.logit_bias = Some(other_map),
            }
        }
        if let Some(other_map) = other.metadata {
            match &mut self.metadata {
                Some(mine) => mine.extend(other_map),
                None => self.metadata = Some(other_map),
            }
        }
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
