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
pub struct CitationAnnotation {
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
    pub annotations: Option<Vec<CitationAnnotation>>,
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
    pub annotations: Option<Vec<CitationAnnotation>>,
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

/// The unified content union, discriminated by the `type` tag.
///
/// This is the Rust equivalent of the Python `Contents` union.
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
    FunctionApprovalRequest(FunctionApprovalRequestContent),
    FunctionApprovalResponse(FunctionApprovalResponseContent),
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
