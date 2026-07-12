//! Content items exchanged with AI services.
//!
//! Mirrors `agent_framework._types` content classes. All variants are unified
//! under the [`Content`] enum, discriminated by the serde `type` tag exactly
//! like the Python `Contents` union.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::error::{Error, Result};

/// Token usage counts for a request/response.
///
/// Supports element-wise addition to accumulate usage across streamed updates.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageDetails {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub input_token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_token_count: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_token_count: Option<u64>,
    /// The number of input tokens written to a provider-managed cache.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_creation_input_token_count: Option<u64>,
    /// The number of input tokens served from a provider-managed cache.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cache_read_input_token_count: Option<u64>,
    /// The number of output tokens used for reasoning.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub reasoning_output_token_count: Option<u64>,
    /// Additional, provider-specific counts merged into the root on serialize.
    #[serde(flatten, default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_counts: HashMap<String, u64>,
}

impl UsageDetails {
    pub fn new() -> Self {
        Self::default()
    }

    /// Element-wise add another usage into this one, treating `None` as `0`
    /// only when the other side has a value.
    pub fn add_assign(&mut self, other: &UsageDetails) {
        fn merge(a: &mut Option<u64>, b: Option<u64>) {
            if let Some(bv) = b {
                *a = Some(a.unwrap_or(0) + bv);
            }
        }
        merge(&mut self.input_token_count, other.input_token_count);
        merge(&mut self.output_token_count, other.output_token_count);
        merge(&mut self.total_token_count, other.total_token_count);
        merge(
            &mut self.cache_creation_input_token_count,
            other.cache_creation_input_token_count,
        );
        merge(
            &mut self.cache_read_input_token_count,
            other.cache_read_input_token_count,
        );
        merge(
            &mut self.reasoning_output_token_count,
            other.reasoning_output_token_count,
        );
        for (k, v) in &other.additional_counts {
            *self.additional_counts.entry(k.clone()).or_insert(0) += *v;
        }
    }
}

impl std::ops::Add for UsageDetails {
    type Output = UsageDetails;
    fn add(mut self, rhs: UsageDetails) -> Self::Output {
        self.add_assign(&rhs);
        self
    }
}

/// An annotated region of text, addressed by index range.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextSpanRegion {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub start_index: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub end_index: Option<i64>,
}

/// A citation annotation attached to content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Annotation {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_regions: Option<Vec<TextSpanRegion>>,
}

/// Plain text content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotations: Option<Vec<Annotation>>,
}

impl TextContent {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            annotations: None,
        }
    }
}

/// Reasoning / chain-of-thought text, distinct from user-facing text.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TextReasoningContent {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotations: Option<Vec<Annotation>>,
    /// The raw provider reasoning item this was decoded from, when it must be
    /// replayed verbatim. Reasoning models (OpenAI Responses with
    /// `store: false`) require the original reasoning item — id and encrypted
    /// content, not just the summary — on the follow-up tool-call turn; the
    /// Responses input mapper re-emits it from here. Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw_representation: Option<Value>,
}

/// Inline binary data encoded as a `data:` URI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataContent {
    pub uri: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub media_type: Option<String>,
}

impl DataContent {
    /// Build a data URI from raw bytes and a media type.
    pub fn from_bytes(data: &[u8], media_type: impl Into<String>) -> Self {
        use base64_lite::encode;
        let media_type = media_type.into();
        let uri = format!("data:{};base64,{}", media_type, encode(data));
        Self {
            uri,
            media_type: Some(media_type),
        }
    }
}

/// A reference to a remote resource by URI (not inline data).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UriContent {
    pub uri: String,
    pub media_type: String,
}

/// A non-fatal error surfaced as content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ErrorContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub error_code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub details: Option<String>,
}

/// A request from the model to call a tool/function.
///
/// `arguments` may be a partial JSON string (during streaming) or a completed
/// JSON object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCallContent {
    pub call_id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub arguments: Option<FunctionArguments>,
}

/// Arguments to a function call: either a raw (possibly partial) string, or a
/// parsed object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FunctionArguments {
    Raw(String),
    Object(HashMap<String, Value>),
}

impl FunctionCallContent {
    pub fn new(
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: Option<FunctionArguments>,
    ) -> Self {
        Self {
            call_id: call_id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Parse the arguments into a JSON object map.
    pub fn parse_arguments(&self) -> Result<HashMap<String, Value>> {
        match &self.arguments {
            None => Ok(HashMap::new()),
            Some(FunctionArguments::Object(m)) => Ok(m.clone()),
            Some(FunctionArguments::Raw(s)) => {
                if s.trim().is_empty() {
                    return Ok(HashMap::new());
                }
                match serde_json::from_str::<Value>(s) {
                    Ok(Value::Object(map)) => Ok(map.into_iter().collect()),
                    Ok(other) => {
                        let mut m = HashMap::new();
                        m.insert("raw".to_string(), other);
                        Ok(m)
                    }
                    Err(e) => Err(Error::Content(format!("invalid function arguments: {e}"))),
                }
            }
        }
    }

    /// Merge a streamed continuation of the same call into this one.
    pub fn merge(&mut self, other: &FunctionCallContent) -> Result<()> {
        if !other.call_id.is_empty() && !self.call_id.is_empty() && self.call_id != other.call_id {
            return Err(Error::AdditionItemMismatch(format!(
                "cannot merge function calls with different call_ids: {} != {}",
                self.call_id, other.call_id
            )));
        }
        if self.call_id.is_empty() {
            self.call_id = other.call_id.clone();
        }
        // The function name is not fragmented by real providers: it arrives once
        // in the first chunk. Set it if we don't have one yet; otherwise only
        // append a genuinely different fragment so a repeated full name does not
        // produce e.g. "get_weatherget_weather".
        if self.name.is_empty() {
            self.name = other.name.clone();
        } else if !other.name.is_empty() && other.name != self.name {
            self.name.push_str(&other.name);
        }
        match (&mut self.arguments, &other.arguments) {
            (Some(FunctionArguments::Raw(a)), Some(FunctionArguments::Raw(b))) => a.push_str(b),
            (None, Some(o)) => self.arguments = Some(o.clone()),
            (Some(FunctionArguments::Object(a)), Some(FunctionArguments::Object(b))) => {
                a.extend(b.clone());
            }
            _ => {}
        }
        Ok(())
    }
}

/// The result of executing a tool/function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionResultContent {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub exception: Option<String>,
}

impl FunctionResultContent {
    pub fn new(call_id: impl Into<String>, result: Option<Value>) -> Self {
        Self {
            call_id: call_id.into(),
            result,
            exception: None,
        }
    }
}

/// Token usage carried inline as a content item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UsageContent {
    pub details: UsageDetails,
}

/// A reference to a file hosted by the service.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostedFileContent {
    pub file_id: String,
}

/// A reference to a vector store hosted by the service.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HostedVectorStoreContent {
    pub vector_store_id: String,
}

/// A request to the user to approve a function call (human-in-the-loop).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionApprovalRequestContent {
    pub id: String,
    pub function_call: FunctionCallContent,
}

impl FunctionApprovalRequestContent {
    pub fn create_response(&self, approved: bool) -> FunctionApprovalResponseContent {
        FunctionApprovalResponseContent {
            approved,
            id: self.id.clone(),
            function_call: self.function_call.clone(),
        }
    }
}

/// A user's response approving or denying a function call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionApprovalResponseContent {
    pub approved: bool,
    pub id: String,
    pub function_call: FunctionCallContent,
}

/// A provider-hosted code-interpreter tool call (the model's request to run
/// code). `inputs` carries the submitted code / files as nested content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CodeInterpreterToolCallContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub inputs: Option<Vec<Content>>,
}

/// The result of a provider-hosted code-interpreter tool call. `outputs`
/// carries the produced logs / files as nested content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CodeInterpreterToolResultContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub outputs: Option<Vec<Content>>,
}

/// A provider-hosted image-generation tool call.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationToolCallContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub image_id: Option<String>,
}

/// The result of a provider-hosted image-generation tool call. `outputs` is
/// the provider-specific payload (e.g. the generated image data or references).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationToolResultContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub image_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub outputs: Option<Value>,
}

/// A provider-hosted MCP server tool call the service already routed to a
/// remote MCP server. Recorded for transcript fidelity; not a local function
/// invocation request (always `informational_only`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerToolCallContent {
    pub call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub server_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub arguments: Option<FunctionArguments>,
}

/// The result of a provider-hosted MCP server tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServerToolResultContent {
    pub call_id: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output: Option<Value>,
}

/// A provider-hosted search tool call (e.g. web/file search).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchToolCallContent {
    pub call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub arguments: Option<FunctionArguments>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub status: Option<String>,
}

/// The result of a provider-hosted search tool call. `items` carries the
/// retrieved results as nested content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchToolResultContent {
    pub call_id: String,
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub items: Option<Vec<Content>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub status: Option<String>,
}

/// A shell tool call: the model's request to run one or more shell commands.
/// This is request metadata, not command output.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShellToolCallContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub commands: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timeout_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_output_length: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub status: Option<String>,
}

/// The aggregate result of a shell tool call. Each per-command output is a
/// [`ShellCommandOutputContent`] carried in `outputs`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShellToolResultContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub outputs: Option<Vec<Content>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_output_length: Option<i64>,
}

/// The output of a single shell command execution.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ShellCommandOutputContent {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stdout: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stderr: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub exit_code: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub timed_out: Option<bool>,
}

/// A request for the user to complete an OAuth consent flow (human-in-the-loop).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OauthConsentRequestContent {
    pub consent_link: String,
}

/// The unified content union, discriminated by the `type` tag.
///
/// This is the Rust equivalent of the Python `Contents` union.
///
/// The [`Content::Unknown`] variant makes deserialization forward-compatible:
/// a content item whose `type` tag is not one of the known variants
/// deserializes to `Unknown` rather than failing the whole message. This
/// mirrors Python's parse-and-skip behavior for unknown content
/// (`_types.py:2205-2210`), except the item is retained as an inert
/// placeholder instead of being dropped. `Unknown` carries no data, so it
/// re-serializes as `{"type":"unknown"}` (the original tag/fields are not
/// preserved) and is treated as inert everywhere: it yields no text, no
/// function call, and is ignored by aggregation/coalescing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text(TextContent),
    TextReasoning(TextReasoningContent),
    Data(DataContent),
    Uri(UriContent),
    Error(ErrorContent),
    FunctionCall(FunctionCallContent),
    FunctionResult(FunctionResultContent),
    Usage(UsageContent),
    HostedFile(HostedFileContent),
    HostedVectorStore(HostedVectorStoreContent),
    CodeInterpreterToolCall(CodeInterpreterToolCallContent),
    CodeInterpreterToolResult(CodeInterpreterToolResultContent),
    ImageGenerationToolCall(ImageGenerationToolCallContent),
    ImageGenerationToolResult(ImageGenerationToolResultContent),
    McpServerToolCall(McpServerToolCallContent),
    McpServerToolResult(McpServerToolResultContent),
    SearchToolCall(SearchToolCallContent),
    SearchToolResult(SearchToolResultContent),
    ShellToolCall(ShellToolCallContent),
    ShellToolResult(ShellToolResultContent),
    ShellCommandOutput(ShellCommandOutputContent),
    FunctionApprovalRequest(FunctionApprovalRequestContent),
    FunctionApprovalResponse(FunctionApprovalResponseContent),
    OauthConsentRequest(OauthConsentRequestContent),
    /// A content item whose `type` tag is unknown to this version of the
    /// library. Deserialization falls back to this inert variant instead of
    /// erroring; see the type-level docs.
    #[serde(other)]
    Unknown,
}

impl Content {
    /// Convenience constructor for plain text.
    pub fn text(text: impl Into<String>) -> Self {
        Content::Text(TextContent::new(text))
    }

    /// The text of this item, if it is text or reasoning content.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(t) => Some(&t.text),
            Content::TextReasoning(t) => Some(&t.text),
            _ => None,
        }
    }

    /// The function call carried by this item, if any.
    pub fn as_function_call(&self) -> Option<&FunctionCallContent> {
        match self {
            Content::FunctionCall(fc) => Some(fc),
            _ => None,
        }
    }

    /// The function result carried by this item, if any.
    pub fn as_function_result(&self) -> Option<&FunctionResultContent> {
        match self {
            Content::FunctionResult(fr) => Some(fr),
            _ => None,
        }
    }

    /// The function-approval request carried by this item, if any.
    pub fn as_function_approval_request(&self) -> Option<&FunctionApprovalRequestContent> {
        match self {
            Content::FunctionApprovalRequest(r) => Some(r),
            _ => None,
        }
    }

    /// The function-approval response carried by this item, if any.
    pub fn as_function_approval_response(&self) -> Option<&FunctionApprovalResponseContent> {
        match self {
            Content::FunctionApprovalResponse(r) => Some(r),
            _ => None,
        }
    }
}

impl From<FunctionApprovalRequestContent> for Content {
    fn from(v: FunctionApprovalRequestContent) -> Self {
        Content::FunctionApprovalRequest(v)
    }
}
impl From<FunctionApprovalResponseContent> for Content {
    fn from(v: FunctionApprovalResponseContent) -> Self {
        Content::FunctionApprovalResponse(v)
    }
}

impl From<TextContent> for Content {
    fn from(v: TextContent) -> Self {
        Content::Text(v)
    }
}
impl From<FunctionCallContent> for Content {
    fn from(v: FunctionCallContent) -> Self {
        Content::FunctionCall(v)
    }
}
impl From<FunctionResultContent> for Content {
    fn from(v: FunctionResultContent) -> Self {
        Content::FunctionResult(v)
    }
}

/// Serialize content(s) into a JSON string suitable for a tool result payload.
///
/// Equivalent to `prepare_function_call_results` in Python.
pub fn prepare_function_call_results(contents: &[Content]) -> String {
    if contents.len() == 1 {
        if let Some(t) = contents[0].as_text() {
            return t.to_string();
        }
    }
    serde_json::to_string(contents).unwrap_or_default()
}

/// Minimal, dependency-free base64 used for data URIs.
mod base64_lite {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub fn encode(input: &[u8]) -> String {
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let b = [
                chunk[0],
                *chunk.get(1).unwrap_or(&0),
                *chunk.get(2).unwrap_or(&0),
            ];
            let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
            out.push(CHARS[((n >> 18) & 63) as usize] as char);
            out.push(CHARS[((n >> 12) & 63) as usize] as char);
            out.push(if chunk.len() > 1 {
                CHARS[((n >> 6) & 63) as usize] as char
            } else {
                '='
            });
            out.push(if chunk.len() > 2 {
                CHARS[(n & 63) as usize] as char
            } else {
                '='
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn unknown_content_type_deserializes_inertly_and_keeps_siblings() {
        // A message carrying a content variant this version doesn't know about,
        // sandwiched between known variants.
        let json = serde_json::json!({
            "role": "assistant",
            "contents": [
                {"type": "text", "text": "before"},
                {"type": "super_future_content", "payload": {"x": 1}, "note": "hi"},
                {"type": "function_call", "call_id": "c1", "name": "f"}
            ]
        });
        let msg: Message =
            serde_json::from_value(json).expect("unknown content must not fail the message");
        assert_eq!(
            msg.contents.len(),
            3,
            "unknown content is retained, not dropped"
        );
        assert!(matches!(msg.contents[0], Content::Text(_)));
        assert_eq!(msg.contents[1], Content::Unknown, "novel type -> Unknown");
        assert!(matches!(msg.contents[2], Content::FunctionCall(_)));

        // The known siblings are still usable / inspectable.
        assert_eq!(msg.contents[0].as_text(), Some("before"));
        assert_eq!(msg.contents[2].as_function_call().unwrap().name, "f");
        // Unknown is inert: no text, no function call.
        assert_eq!(msg.contents[1].as_text(), None);
        assert!(msg.contents[1].as_function_call().is_none());
    }

    #[test]
    fn unknown_content_reserializes_without_panicking() {
        let c = Content::Unknown;
        let v = serde_json::to_value(&c).expect("Unknown must serialize");
        assert_eq!(v, serde_json::json!({"type": "unknown"}));
        // And a full message round-trips through serialization without panic.
        let msg = Message::with_contents(
            crate::types::Role::assistant(),
            vec![Content::text("keep"), Content::Unknown],
        );
        let s = serde_json::to_string(&msg).expect("message serializes");
        assert!(s.contains("\"unknown\""));
    }

    #[test]
    fn new_hosted_tool_variants_use_upstream_wire_tags() {
        // These `type` tags are load-bearing for cross-language interop with
        // the Python/.NET `Content` union; assert each one exactly.
        let cases: Vec<(Content, &str)> = vec![
            (
                Content::CodeInterpreterToolCall(CodeInterpreterToolCallContent::default()),
                "code_interpreter_tool_call",
            ),
            (
                Content::CodeInterpreterToolResult(CodeInterpreterToolResultContent::default()),
                "code_interpreter_tool_result",
            ),
            (
                Content::ImageGenerationToolCall(ImageGenerationToolCallContent::default()),
                "image_generation_tool_call",
            ),
            (
                Content::ImageGenerationToolResult(ImageGenerationToolResultContent::default()),
                "image_generation_tool_result",
            ),
            (
                Content::McpServerToolCall(McpServerToolCallContent {
                    call_id: "c".into(),
                    tool_name: "t".into(),
                    server_name: None,
                    arguments: None,
                }),
                "mcp_server_tool_call",
            ),
            (
                Content::McpServerToolResult(McpServerToolResultContent {
                    call_id: "c".into(),
                    output: None,
                }),
                "mcp_server_tool_result",
            ),
            (
                Content::SearchToolCall(SearchToolCallContent {
                    call_id: "c".into(),
                    tool_name: "t".into(),
                    arguments: None,
                    status: None,
                }),
                "search_tool_call",
            ),
            (
                Content::SearchToolResult(SearchToolResultContent {
                    call_id: "c".into(),
                    tool_name: "t".into(),
                    result: None,
                    items: None,
                    status: None,
                }),
                "search_tool_result",
            ),
            (
                Content::ShellToolCall(ShellToolCallContent::default()),
                "shell_tool_call",
            ),
            (
                Content::ShellToolResult(ShellToolResultContent::default()),
                "shell_tool_result",
            ),
            (
                Content::ShellCommandOutput(ShellCommandOutputContent::default()),
                "shell_command_output",
            ),
            (
                Content::OauthConsentRequest(OauthConsentRequestContent {
                    consent_link: "https://example/consent".into(),
                }),
                "oauth_consent_request",
            ),
        ];
        for (content, tag) in cases {
            let v = serde_json::to_value(&content).unwrap();
            assert_eq!(
                v.get("type").and_then(serde_json::Value::as_str),
                Some(tag),
                "wire tag mismatch for {content:?}"
            );
            // And it must round-trip to the same variant (not fall back to Unknown).
            let back: Content = serde_json::from_value(v).unwrap();
            assert_eq!(back, content);
            assert_ne!(back, Content::Unknown);
        }
    }

    #[test]
    fn usage_details_typed_cache_reasoning_fields_roundtrip_and_add() {
        let a = UsageDetails {
            input_token_count: Some(10),
            cache_creation_input_token_count: Some(3),
            cache_read_input_token_count: Some(4),
            reasoning_output_token_count: Some(5),
            ..Default::default()
        };
        // Serialize at the top level with the upstream key names.
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v.get("cache_creation_input_token_count"), Some(&3.into()));
        assert_eq!(v.get("cache_read_input_token_count"), Some(&4.into()));
        assert_eq!(v.get("reasoning_output_token_count"), Some(&5.into()));
        let back: UsageDetails = serde_json::from_value(v).unwrap();
        assert_eq!(back, a);
        // Element-wise addition folds the typed fields too.
        let sum = a.clone() + a;
        assert_eq!(sum.cache_creation_input_token_count, Some(6));
        assert_eq!(sum.cache_read_input_token_count, Some(8));
        assert_eq!(sum.reasoning_output_token_count, Some(10));
    }

    #[test]
    fn known_variants_roundtrip_unchanged() {
        let contents = vec![
            Content::text("hello"),
            Content::FunctionCall(FunctionCallContent::new("id", "name", None)),
            Content::FunctionResult(FunctionResultContent::new(
                "id",
                Some(Value::String("ok".into())),
            )),
        ];
        for c in contents {
            let s = serde_json::to_string(&c).unwrap();
            let back: Content = serde_json::from_str(&s).unwrap();
            assert_eq!(c, back);
            assert_ne!(back, Content::Unknown);
        }
    }
}
