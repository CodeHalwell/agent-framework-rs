//! [`OpenAIResponsesClient`]: a [`ChatClient`] for the OpenAI Responses API
//! (`POST /v1/responses`).
//!
//! The Responses API uses an item-based `input`/`output` shape rather than
//! the `messages` array used by Chat Completions, and supports a dedicated
//! `previous_response_id` for service-side conversation state. Wire framing
//! (SSE parsing style, error handling) mirrors [`crate::OpenAIClient`].
//!
//! ```no_run
//! use agent_framework_openai::responses::OpenAIResponsesClient;
//! use agent_framework_core::prelude::*;
//!
//! # async fn demo() -> Result<()> {
//! let client = OpenAIResponsesClient::new("sk-...", "gpt-4o-mini");
//! let agent = ChatAgent::builder(client)
//!     .instructions("You are concise.")
//!     .build();
//! let reply = agent.run_once("Say hi").await?;
//! println!("{}", reply.text());
//! # Ok(())
//! # }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::tools::ToolDefinition;
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, ChatResponseUpdate, CitationAnnotation, Content,
    DataContent, FinishReason, FunctionApprovalRequestContent, FunctionArguments,
    FunctionCallContent, FunctionResultContent, ResponseFormat, Role, TextContent,
    TextReasoningContent, TextSpanRegion, ToolMode, UriContent, UsageContent, UsageDetails,
};
use futures::StreamExt;
use serde_json::{json, Map, Value};

use crate::convert::{
    audio_format, data_content_media_type, function_arguments_to_string, result_to_string,
    top_level_media_type, DEFAULT_FILENAME,
};
use crate::{ByteStream, DEFAULT_BASE_URL};

/// An OpenAI Responses API chat client (`POST /v1/responses`).
#[derive(Clone)]
pub struct OpenAIResponsesClient {
    inner: Arc<Inner>,
}

#[derive(Clone)]
struct Inner {
    http: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    organization: Option<String>,
}

impl std::fmt::Debug for OpenAIResponsesClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAIResponsesClient")
            .field("base_url", &self.inner.base_url)
            .field("model", &self.inner.model)
            .field("organization", &self.inner.organization)
            .finish_non_exhaustive()
    }
}

impl OpenAIResponsesClient {
    /// Create a client for the given API key and default model.
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            inner: Arc::new(Inner {
                http: reqwest::Client::new(),
                api_key: api_key.into(),
                base_url: DEFAULT_BASE_URL.to_string(),
                model: model.into(),
                organization: None,
            }),
        }
    }

    /// Build a client from the `OPENAI_API_KEY` (and optional
    /// `OPENAI_BASE_URL`) environment variables.
    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let key = std::env::var("OPENAI_API_KEY")
            .map_err(|_| Error::Configuration("OPENAI_API_KEY is not set".into()))?;
        let mut client = Self::new(key, model);
        if let Ok(base) = std::env::var("OPENAI_BASE_URL") {
            client = client.with_base_url(base);
        }
        Ok(client)
    }

    /// Override the base URL (for Azure OpenAI or compatible servers).
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).base_url = base_url.into();
        self
    }

    /// Set the organization header.
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        Arc::make_mut(&mut self.inner).organization = Some(org.into());
        self
    }

    /// The default model id.
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    fn build_body(&self, messages: &[ChatMessage], options: &ChatOptions, stream: bool) -> Value {
        let mut body = Map::new();
        let model = options
            .model_id
            .clone()
            .unwrap_or_else(|| self.inner.model.clone());
        body.insert("model".into(), json!(model));

        let (instructions, rest) = extract_instructions(messages, options.instructions.as_deref());
        if let Some(instructions) = instructions {
            body.insert("instructions".into(), json!(instructions));
        }
        body.insert("input".into(), json!(messages_to_input(rest)));

        if let Some(conversation_id) = &options.conversation_id {
            body.insert("previous_response_id".into(), json!(conversation_id));
        }
        if let Some(t) = options.temperature {
            body.insert("temperature".into(), json!(t));
        }
        if let Some(t) = options.top_p {
            body.insert("top_p".into(), json!(t));
        }
        if let Some(mt) = options.max_tokens {
            body.insert("max_output_tokens".into(), json!(mt));
        }
        if let Some(store) = options.store {
            body.insert("store".into(), json!(store));
        }
        if let Some(user) = &options.user {
            body.insert("user".into(), json!(user));
        }
        if let Some(metadata) = &options.metadata {
            body.insert("metadata".into(), json!(metadata));
        }

        if !options.tools.is_empty() {
            let tools: Vec<Value> = options.tools.iter().map(tool_to_responses_spec).collect();
            body.insert("tools".into(), json!(tools));
            if let Some(allow_multi) = options.allow_multiple_tool_calls {
                body.insert("parallel_tool_calls".into(), json!(allow_multi));
            }
        }
        if let Some(tool_choice) = &options.tool_choice {
            body.insert("tool_choice".into(), tool_choice_to_responses(tool_choice));
        }
        if let Some(fmt) = &options.response_format {
            body.insert(
                "text".into(),
                json!({ "format": response_format_to_text(fmt) }),
            );
        }

        for (k, v) in &options.additional_properties {
            body.entry(k.clone()).or_insert_with(|| v.clone());
        }

        if stream {
            body.insert("stream".into(), json!(true));
        }
        Value::Object(body)
    }

    async fn post(&self, body: &Value) -> Result<reqwest::Response> {
        let url = format!("{}/responses", self.inner.base_url.trim_end_matches('/'));
        let mut req = self
            .inner
            .http
            .post(&url)
            .bearer_auth(&self.inner.api_key)
            .json(body);
        if let Some(org) = &self.inner.organization {
            req = req.header("OpenAI-Organization", org);
        }
        let resp = req
            .send()
            .await
            .map_err(|e| Error::service(format!("request failed: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let retry_after = crate::parse_retry_after(resp.headers());
            let text = resp.text().await.unwrap_or_default();
            return Err(crate::classify_service_error(
                status.as_u16(),
                &text,
                format!("OpenAI API error {status}: {text}"),
                retry_after,
            ));
        }
        Ok(resp)
    }
}

#[async_trait::async_trait]
impl ChatClient for OpenAIResponsesClient {
    async fn get_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let body = self.build_body(&messages, &options, false);
        let resp = self.post(&body).await?;
        let value: Value = resp
            .json()
            .await
            .map_err(|e| Error::service(format!("invalid response json: {e}")))?;
        if value.get("status").and_then(Value::as_str) == Some("failed") {
            let msg = value
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("response failed")
                .to_string();
            return Err(Error::service(msg));
        }
        Ok(parse_response(&value, options.store))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<ChatMessage>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let body = self.build_body(&messages, &options, true);
        let resp = self.post(&body).await?;
        Ok(parse_responses_sse_stream(resp, options.store).boxed())
    }

    fn model_id(&self) -> Option<&str> {
        Some(&self.inner.model)
    }
}

// region: request conversion

/// Split a leading system message (and/or `ChatOptions::instructions`) out
/// into the Responses API's top-level `instructions` field, returning the
/// remaining messages to convert into `input` items.
///
/// `pub` (rather than private) so `agent-framework-azure`'s Responses client
/// can reuse this exact instructions-extraction step ahead of
/// [`messages_to_input`] when building the Azure OpenAI Responses request
/// body, instead of reimplementing it.
pub fn extract_instructions<'a>(
    messages: &'a [ChatMessage],
    options_instructions: Option<&str>,
) -> (Option<String>, &'a [ChatMessage]) {
    let mut parts = Vec::new();
    if let Some(instr) = options_instructions {
        if !instr.is_empty() {
            parts.push(instr.to_string());
        }
    }
    let mut rest = messages;
    if let Some(first) = messages.first() {
        if first.role == Role::system() {
            let text = first.text();
            if !text.is_empty() {
                parts.push(text);
            }
            rest = &messages[1..];
        }
    }
    if parts.is_empty() {
        (None, rest)
    } else {
        (Some(parts.join("\n\n")), rest)
    }
}

/// Convert framework messages into the Responses API's `input` item array.
///
/// `pub` so `agent-framework-azure`'s Responses client can reuse this
/// conversion verbatim rather than reimplementing it (Azure OpenAI's
/// Responses API shares the exact same `input` item wire shape).
pub fn messages_to_input(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::new();
    for msg in messages {
        let role = msg.role.as_str();
        if role == Role::TOOL {
            for content in &msg.contents {
                if let Content::FunctionResult(fr) = content {
                    out.push(function_result_to_item(fr));
                }
            }
            continue;
        }

        let mut buffered: Vec<Value> = Vec::new();
        for content in &msg.contents {
            match content {
                Content::Text(t) => {
                    let text_type = if role == Role::ASSISTANT {
                        "output_text"
                    } else {
                        "input_text"
                    };
                    buffered.push(json!({ "type": text_type, "text": t.text }));
                }
                Content::Uri(u) => {
                    if let Some(part) = content_to_input_part(&u.uri, Some(&u.media_type)) {
                        buffered.push(part);
                    }
                }
                Content::Data(d) => {
                    if let Some(part) =
                        content_to_input_part(&d.uri, data_content_media_type(d).as_deref())
                    {
                        buffered.push(part);
                    }
                }
                Content::HostedFile(h) => {
                    buffered.push(json!({ "type": "input_file", "file_id": h.file_id }));
                }
                Content::FunctionCall(fc) => {
                    flush_text(&mut out, &mut buffered, role);
                    out.push(json!({
                        "type": "function_call",
                        "call_id": fc.call_id,
                        "name": fc.name,
                        "arguments": function_arguments_to_string(&fc.arguments),
                    }));
                }
                Content::FunctionResult(fr) => {
                    flush_text(&mut out, &mut buffered, role);
                    out.push(function_result_to_item(fr));
                }
                Content::FunctionApprovalResponse(r) => {
                    flush_text(&mut out, &mut buffered, role);
                    out.push(json!({
                        "type": "mcp_approval_response",
                        "approval_request_id": r.id,
                        "approve": r.approved,
                    }));
                }
                Content::FunctionApprovalRequest(r) => {
                    flush_text(&mut out, &mut buffered, role);
                    out.push(json!({
                        "type": "mcp_approval_request",
                        "id": r.id,
                        "name": r.function_call.name,
                        "arguments": function_arguments_to_string(&r.function_call.arguments),
                    }));
                }
                _ => {}
            }
        }
        flush_text(&mut out, &mut buffered, role);
    }
    out
}

/// Map a URI/data content item to a Responses API input content part, or `None`
/// when it has no wire mapping (mirrors upstream `_openai_content_parser`).
/// Handles images (`input_image`), audio (`input_audio`), and `application/*`
/// data (`input_file`).
fn content_to_input_part(uri: &str, media_type: Option<&str>) -> Option<Value> {
    let media_type = media_type?;
    match top_level_media_type(media_type).as_str() {
        // `detail` defaults to "auto"; the Rust content types carry no override.
        "image" => Some(json!({
            "type": "input_image",
            "image_url": uri,
            "detail": "auto",
        })),
        "audio" => {
            let format = audio_format(media_type)?;
            Some(json!({
                "type": "input_audio",
                "input_audio": { "data": uri, "format": format },
            }))
        }
        "application" => Some(json!({
            "type": "input_file",
            "file_data": uri,
            "filename": DEFAULT_FILENAME,
        })),
        _ => None,
    }
}

fn flush_text(out: &mut Vec<Value>, buffered: &mut Vec<Value>, role: &str) {
    if !buffered.is_empty() {
        out.push(json!({ "type": "message", "role": role, "content": std::mem::take(buffered) }));
    }
}

fn function_result_to_item(fr: &FunctionResultContent) -> Value {
    json!({
        "type": "function_call_output",
        "call_id": fr.call_id,
        "output": result_to_string(fr),
    })
}

/// The flat Responses-API tool spec: `{"type":"function","name":...}`, unlike
/// Chat Completions' `{"type":"function","function":{...}}` nesting.
///
/// `pub` so `agent-framework-azure`'s Responses client can reuse this
/// mapping rather than reimplementing it.
pub fn tool_to_responses_spec(tool: &ToolDefinition) -> Value {
    use agent_framework_core::tools::ToolKind;
    match &tool.kind {
        ToolKind::HostedWebSearch => {
            let mut spec = Map::new();
            spec.insert("type".into(), json!("web_search"));
            if let Some(loc) = tool.parameters.get("user_location") {
                spec.insert("user_location".into(), user_location_to_responses(loc));
            }
            Value::Object(spec)
        }
        ToolKind::HostedCodeInterpreter => {
            let mut spec = Map::new();
            spec.insert("type".into(), json!("code_interpreter"));
            // A caller-supplied `container` wins; otherwise default to `auto`
            // and attach any `file_ids` (`_responses_client.py:264-278`).
            if let Some(container) = tool.parameters.get("container") {
                spec.insert("container".into(), container.clone());
            } else {
                let mut container = Map::new();
                container.insert("type".into(), json!("auto"));
                if let Some(file_ids) = tool.parameters.get("file_ids") {
                    container.insert("file_ids".into(), file_ids.clone());
                }
                spec.insert("container".into(), Value::Object(container));
            }
            Value::Object(spec)
        }
        ToolKind::HostedFileSearch { max_results } => {
            let mut spec = Map::new();
            spec.insert("type".into(), json!("file_search"));
            // The Responses API requires vector-store ids; the marker itself
            // carries none, so honor ids supplied via the definition's
            // parameters object when present.
            if let Some(ids) = tool.parameters.get("vector_store_ids") {
                spec.insert("vector_store_ids".into(), ids.clone());
            }
            // Prefer the marker's `max_results`; fall back to a parameters key.
            let max = (*max_results)
                .map(|n| json!(n))
                .or_else(|| tool.parameters.get("max_results").cloned());
            if let Some(n) = max {
                spec.insert("max_num_results".into(), n);
            }
            Value::Object(spec)
        }
        ToolKind::HostedMcp { url, allowed_tools } => {
            let mut spec = Map::new();
            spec.insert("type".into(), json!("mcp"));
            spec.insert("server_label".into(), json!(tool.name.replace(' ', "_")));
            spec.insert("server_url".into(), json!(url));
            if !tool.description.is_empty() {
                spec.insert("server_description".into(), json!(tool.description));
            }
            if let Some(headers) = tool.parameters.get("headers") {
                spec.insert("headers".into(), headers.clone());
            }
            if let Some(allowed) = allowed_tools {
                spec.insert("allowed_tools".into(), json!(allowed));
            }
            spec.insert("require_approval".into(), mcp_require_approval(tool));
            Value::Object(spec)
        }
        ToolKind::Function => json!({
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }),
    }
}

/// Build the Responses `web_search.user_location` object from a hosted-tool
/// `user_location` parameter (`_responses_client.py:310-329`).
fn user_location_to_responses(location: &Value) -> Value {
    let mut loc = Map::new();
    loc.insert("type".into(), json!("approximate"));
    for key in ["city", "country", "region", "timezone"] {
        if let Some(v) = location.get(key) {
            loc.insert(key.into(), v.clone());
        }
    }
    Value::Object(loc)
}

/// Build the Responses MCP `require_approval` value. A `parameters.approval_mode`
/// override — either the string `"always_require"`/`"never_require"` or an
/// object `{"always": [...], "never": [...]}` — takes precedence over the
/// definition's [`ApprovalMode`]; mirrors `get_mcp_tool`
/// (`_responses_client.py:365-386`).
fn mcp_require_approval(tool: &ToolDefinition) -> Value {
    use agent_framework_core::tools::ApprovalMode;
    match tool.parameters.get("approval_mode") {
        Some(Value::String(s)) => {
            return json!(if s == "always_require" {
                "always"
            } else {
                "never"
            });
        }
        Some(Value::Object(modes)) => {
            let mut req = Map::new();
            if let Some(always) = modes.get("always") {
                req.insert("always".into(), json!({ "tool_names": always }));
            }
            if let Some(never) = modes.get("never") {
                req.insert("never".into(), json!({ "tool_names": never }));
            }
            if !req.is_empty() {
                return Value::Object(req);
            }
        }
        _ => {}
    }
    json!(match tool.approval_mode {
        ApprovalMode::AlwaysRequire => "always",
        ApprovalMode::NeverRequire => "never",
    })
}

/// `pub` so `agent-framework-azure`'s Responses client can reuse this
/// mapping rather than reimplementing it.
pub fn tool_choice_to_responses(mode: &ToolMode) -> Value {
    match mode {
        ToolMode::Auto => json!("auto"),
        ToolMode::None => json!("none"),
        ToolMode::Required(Some(name)) => json!({ "type": "function", "name": name }),
        ToolMode::Required(None) => json!("required"),
    }
}

/// Convert a `ChatOptions::response_format` into a Responses API
/// `text.format` object. Unlike Chat Completions (which nests the schema
/// under `json_schema`), the Responses API uses a flat object.
///
/// `pub` so `agent-framework-azure`'s Responses client can reuse this
/// mapping rather than reimplementing it.
pub fn response_format_to_text(format: &ResponseFormat) -> Value {
    match format {
        ResponseFormat::Text => json!({ "type": "text" }),
        ResponseFormat::JsonObject => json!({ "type": "json_object" }),
        ResponseFormat::JsonSchema {
            name,
            description,
            schema,
            strict,
        } => {
            let mut obj = Map::new();
            obj.insert("type".into(), json!("json_schema"));
            obj.insert("name".into(), json!(name));
            if let Some(d) = description {
                obj.insert("description".into(), json!(d));
            }
            obj.insert("schema".into(), schema.clone());
            if let Some(st) = strict {
                obj.insert("strict".into(), json!(st));
            }
            Value::Object(obj)
        }
    }
}

// endregion

// region: response conversion

/// Parse a full (non-streaming) Responses API response.
///
/// `pub` so `agent-framework-azure`'s Responses client (whose wire format is
/// otherwise identical) can reuse this parser — including `parse_output_item`,
/// `parse_annotations`, and usage/finish-reason handling — rather than
/// reimplementing it.
pub fn parse_response(value: &Value, store: Option<bool>) -> ChatResponse {
    let mut response = ChatResponse {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model_id: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    let mut contents: Vec<Content> = Vec::new();
    if let Some(items) = value.get("output").and_then(Value::as_array) {
        for item in items {
            parse_output_item(item, &mut contents);
        }
    }

    let mut message = ChatMessage::with_contents(Role::assistant(), contents);
    message.message_id = response.response_id.clone();
    response.messages.push(message);

    response.finish_reason = finish_reason_from_response(value);

    if let Some(usage) = value.get("usage") {
        response.usage_details = Some(parse_responses_usage(usage));
    }
    if store != Some(false) {
        response.conversation_id = response.response_id.clone();
    }
    response
}

fn parse_output_item(item: &Value, contents: &mut Vec<Content>) {
    match item.get("type").and_then(Value::as_str) {
        Some("message") => {
            if let Some(parts) = item.get("content").and_then(Value::as_array) {
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("output_text") => {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                let mut tc = TextContent::new(text);
                                tc.annotations = parse_annotations(part);
                                contents.push(Content::Text(tc));
                            }
                        }
                        Some("refusal") => {
                            if let Some(text) = part.get("refusal").and_then(Value::as_str) {
                                contents.push(Content::Text(TextContent::new(text)));
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Some("function_call") => {
            let call_id = item
                .get("call_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string();
            contents.push(Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Raw(args)),
            )));
        }
        Some("reasoning") => {
            if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                for s in summary {
                    if let Some(text) = s.get("text").and_then(Value::as_str) {
                        contents.push(Content::TextReasoning(TextReasoningContent {
                            text: text.to_string(),
                            annotations: None,
                        }));
                    }
                }
            }
        }
        // Code-interpreter runs surface `logs` as text and `image` outputs as
        // URIs; a bare `code` (no outputs) is a text fallback
        // (`_create_response_content:748-764`).
        Some("code_interpreter_call") => {
            let outputs = item
                .get("outputs")
                .and_then(Value::as_array)
                .filter(|a| !a.is_empty());
            if let Some(outputs) = outputs {
                for output in outputs {
                    match output.get("type").and_then(Value::as_str) {
                        Some("logs") => {
                            if let Some(logs) = output.get("logs").and_then(Value::as_str) {
                                contents.push(Content::Text(TextContent::new(logs)));
                            }
                        }
                        Some("image") => {
                            if let Some(url) = output.get("url").and_then(Value::as_str) {
                                contents.push(Content::Uri(UriContent {
                                    uri: url.to_string(),
                                    media_type: "image".to_string(),
                                }));
                            }
                        }
                        _ => {}
                    }
                }
            } else if let Some(code) = item.get("code").and_then(Value::as_str) {
                contents.push(Content::Text(TextContent::new(code)));
            }
        }
        // A generated image is returned as base64; default to image/png unless
        // the result is a data URI that states its own type
        // (`_create_response_content:788-811`).
        Some("image_generation_call") => {
            if let Some(result) = item.get("result").and_then(Value::as_str) {
                let (uri, media_type) = if result.starts_with("data:") {
                    let media_type = if result.contains(';') {
                        result
                            .strip_prefix("data:")
                            .and_then(|r| r.split(';').next())
                            .filter(|s| !s.is_empty())
                            .map(String::from)
                    } else {
                        None
                    };
                    (result.to_string(), media_type)
                } else {
                    (
                        format!("data:image/png;base64,{result}"),
                        Some("image/png".to_string()),
                    )
                };
                contents.push(Content::Data(DataContent { uri, media_type }));
            }
        }
        // An MCP approval request round-trips its `id` as the call id so a
        // later `FunctionApprovalResponse` refers back to it
        // (`_create_response_content:775-787`).
        Some("mcp_approval_request") => {
            let id = item
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = item
                .get("arguments")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_string();
            contents.push(Content::FunctionApprovalRequest(
                FunctionApprovalRequestContent {
                    id: id.clone(),
                    function_call: FunctionCallContent::new(
                        id,
                        name,
                        Some(FunctionArguments::Raw(args)),
                    ),
                },
            ));
        }
        _ => {}
    }
}

/// Parse the `annotations` on an `output_text` part into [`CitationAnnotation`]s
/// (`_create_response_content:667-724`). The core annotation type has no free
/// `additional_properties`, so upstream's `index`/`container_id` extras are
/// dropped.
fn parse_annotations(part: &Value) -> Option<Vec<CitationAnnotation>> {
    let arr = part.get("annotations").and_then(Value::as_array)?;
    let mut out = Vec::new();
    for ann in arr {
        let str_field = |k: &str| ann.get(k).and_then(Value::as_str).map(String::from);
        let regions = || {
            Some(vec![TextSpanRegion {
                start_index: ann.get("start_index").and_then(Value::as_i64),
                end_index: ann.get("end_index").and_then(Value::as_i64),
            }])
        };
        match ann.get("type").and_then(Value::as_str) {
            Some("file_path") => out.push(CitationAnnotation {
                file_id: str_field("file_id"),
                ..Default::default()
            }),
            Some("file_citation") => out.push(CitationAnnotation {
                url: str_field("filename"),
                file_id: str_field("file_id"),
                ..Default::default()
            }),
            Some("url_citation") => out.push(CitationAnnotation {
                title: str_field("title"),
                url: str_field("url"),
                annotated_regions: regions(),
                ..Default::default()
            }),
            Some("container_file_citation") => out.push(CitationAnnotation {
                file_id: str_field("file_id"),
                url: str_field("filename"),
                annotated_regions: regions(),
                ..Default::default()
            }),
            _ => {}
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn finish_reason_from_response(value: &Value) -> Option<FinishReason> {
    let has_function_call = value
        .get("output")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|i| i.get("type").and_then(Value::as_str) == Some("function_call"))
        })
        .unwrap_or(false);
    if has_function_call {
        return Some(FinishReason::tool_calls());
    }
    let status = value.get("status").and_then(Value::as_str)?;
    Some(match status {
        "completed" => FinishReason::stop(),
        "incomplete" => match value
            .get("incomplete_details")
            .and_then(|d| d.get("reason"))
            .and_then(Value::as_str)
        {
            Some("max_output_tokens") => FinishReason::new(FinishReason::LENGTH),
            Some("content_filter") => FinishReason::new(FinishReason::CONTENT_FILTER),
            Some(other) => FinishReason::new(other),
            None => FinishReason::new("incomplete"),
        },
        other => FinishReason::new(other),
    })
}

fn parse_responses_usage(usage: &Value) -> UsageDetails {
    let mut details = UsageDetails {
        input_token_count: usage.get("input_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("output_tokens").and_then(Value::as_u64),
        total_token_count: usage.get("total_tokens").and_then(Value::as_u64),
        additional_counts: Default::default(),
    };
    if let Some(cached) = usage
        .get("input_tokens_details")
        .and_then(|d| d.get("cached_tokens"))
        .and_then(Value::as_u64)
    {
        details
            .additional_counts
            .insert("openai.cached_input_tokens".into(), cached);
    }
    if let Some(reasoning) = usage
        .get("output_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(Value::as_u64)
    {
        details
            .additional_counts
            .insert("openai.reasoning_tokens".into(), reasoning);
    }
    details
}

// endregion

// region: streaming

/// Turn a Responses API SSE HTTP response into a stream of updates.
///
/// `pub` so `agent-framework-azure`'s Responses client can reuse this exact
/// SSE event parser (Azure OpenAI's Responses API streams the same event
/// shapes) rather than reimplementing it.
pub fn parse_responses_sse_stream(
    resp: reqwest::Response,
    store: Option<bool>,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let byte_stream: ByteStream = Box::pin(resp.bytes_stream());
    futures::stream::unfold(
        ResponsesSseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            call_ids: HashMap::new(),
            done: false,
            store,
        },
        |mut state| async move {
            loop {
                if let Some(update) = state.queued.pop_front() {
                    return Some((Ok(update), state));
                }
                if state.done {
                    return None;
                }
                match state.byte_stream.next().await {
                    Some(Ok(bytes)) => {
                        let decoded = state.utf8.push(&bytes);
                        state.buffer.push_str(&decoded);
                        while let Some(pos) = state.buffer.find('\n') {
                            let line = state.buffer[..pos].trim().to_string();
                            state.buffer.drain(..=pos);
                            let Some(data) = line.strip_prefix("data:") else {
                                continue;
                            };
                            let data = data.trim();
                            if data.is_empty() {
                                continue;
                            }
                            let Ok(value) = serde_json::from_str::<Value>(data) else {
                                continue;
                            };
                            match parse_responses_event(&value, &mut state.call_ids, state.store) {
                                EventOutcome::Update(update) => state.queued.push_back(update),
                                EventOutcome::Error(e) => {
                                    state.done = true;
                                    return Some((Err(e), state));
                                }
                                EventOutcome::None => {}
                            }
                        }
                    }
                    Some(Err(e)) => {
                        state.done = true;
                        return Some((Err(Error::service(format!("stream error: {e}"))), state));
                    }
                    None => return None,
                }
            }
        },
    )
}

struct ResponsesSseState {
    byte_stream: ByteStream,
    buffer: String,
    utf8: Utf8StreamDecoder,
    queued: VecDeque<ChatResponseUpdate>,
    /// `output_index` -> `call_id`, resolved when the call is first announced
    /// via `response.output_item.added` and used to route later
    /// `response.function_call_arguments.delta` fragments.
    call_ids: HashMap<i64, String>,
    done: bool,
    store: Option<bool>,
}

// A transient control-flow value: produced per SSE event and immediately
// destructured in the stream loop, never stored in bulk. Boxing the `Update`
// variant to equalize sizes would add a heap allocation on every streamed
// token, so the size skew is accepted here.
#[allow(clippy::large_enum_variant)]
enum EventOutcome {
    Update(ChatResponseUpdate),
    Error(Error),
    None,
}

/// Wrap streamed reasoning text as a [`TextReasoningContent`] update, or
/// [`EventOutcome::None`] when empty.
fn reasoning_update(text: &str) -> EventOutcome {
    if text.is_empty() {
        return EventOutcome::None;
    }
    EventOutcome::Update(ChatResponseUpdate {
        contents: vec![Content::TextReasoning(TextReasoningContent {
            text: text.to_string(),
            annotations: None,
        })],
        role: Some(Role::assistant()),
        ..Default::default()
    })
}

/// Parse one Responses API SSE event (already-decoded JSON `data:` payload).
fn parse_responses_event(
    value: &Value,
    call_ids: &mut HashMap<i64, String>,
    store: Option<bool>,
) -> EventOutcome {
    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");
    match event_type {
        "response.created" => {
            let resp = value.get("response");
            let response_id = resp
                .and_then(|r| r.get("id"))
                .and_then(Value::as_str)
                .map(String::from);
            let model_id = resp
                .and_then(|r| r.get("model"))
                .and_then(Value::as_str)
                .map(String::from);
            if response_id.is_none() && model_id.is_none() {
                return EventOutcome::None;
            }
            EventOutcome::Update(ChatResponseUpdate {
                role: Some(Role::assistant()),
                response_id,
                model_id,
                ..Default::default()
            })
        }
        "response.output_text.delta" => {
            let text = value.get("delta").and_then(Value::as_str).unwrap_or("");
            if text.is_empty() {
                return EventOutcome::None;
            }
            EventOutcome::Update(ChatResponseUpdate {
                contents: vec![Content::Text(TextContent::new(text))],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        }
        // Reasoning (chain-of-thought / summary) streams as its own text
        // channel. Both the incremental `.delta` and the terminal `.done`
        // (full text) map to `TextReasoningContent`, mirroring upstream
        // `_create_streaming_response_content` (`_responses_client.py:917-928`).
        "response.reasoning_text.delta" | "response.reasoning_summary_text.delta" => {
            reasoning_update(value.get("delta").and_then(Value::as_str).unwrap_or(""))
        }
        "response.reasoning_text.done" | "response.reasoning_summary_text.done" => {
            reasoning_update(value.get("text").and_then(Value::as_str).unwrap_or(""))
        }
        "response.output_item.added" => {
            let item = value.get("item");
            if item.and_then(|i| i.get("type")).and_then(Value::as_str) != Some("function_call") {
                return EventOutcome::None;
            }
            let output_index = value
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let call_id = item
                .and_then(|i| i.get("call_id"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let name = item
                .and_then(|i| i.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            call_ids.insert(output_index, call_id.clone());
            EventOutcome::Update(ChatResponseUpdate {
                contents: vec![Content::FunctionCall(FunctionCallContent::new(
                    call_id, name, None,
                ))],
                role: Some(Role::assistant()),
                ..Default::default()
            })
        }
        "response.function_call_arguments.delta" => {
            let output_index = value
                .get("output_index")
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let delta = value.get("delta").and_then(Value::as_str).unwrap_or("");
            match call_ids.get(&output_index) {
                Some(call_id) => EventOutcome::Update(ChatResponseUpdate {
                    contents: vec![Content::FunctionCall(FunctionCallContent::new(
                        call_id.clone(),
                        "",
                        Some(FunctionArguments::Raw(delta.to_string())),
                    ))],
                    role: Some(Role::assistant()),
                    ..Default::default()
                }),
                None => EventOutcome::None,
            }
        }
        "response.completed" => {
            let resp = value.get("response");
            let response_id = resp
                .and_then(|r| r.get("id"))
                .and_then(Value::as_str)
                .map(String::from);
            let model_id = resp
                .and_then(|r| r.get("model"))
                .and_then(Value::as_str)
                .map(String::from);
            let mut contents = Vec::new();
            if let Some(usage) = resp.and_then(|r| r.get("usage")) {
                contents.push(Content::Usage(UsageContent {
                    details: parse_responses_usage(usage),
                }));
            }
            let finish_reason = resp.and_then(finish_reason_from_response);
            let conversation_id = if store != Some(false) {
                response_id.clone()
            } else {
                None
            };
            EventOutcome::Update(ChatResponseUpdate {
                contents,
                role: Some(Role::assistant()),
                response_id,
                model_id,
                conversation_id,
                finish_reason,
                ..Default::default()
            })
        }
        "response.failed" | "error" => {
            let resp = value.get("response");
            let err_obj = resp
                .and_then(|r| r.get("error"))
                .or_else(|| value.get("error"));
            let msg = err_obj
                .and_then(|e| e.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("response failed")
                .to_string();
            EventOutcome::Error(Error::service(msg))
        }
        // Recognized but carry no additional content: the arguments are
        // already fully accumulated via `.delta` events, and item/part
        // lifecycle markers don't themselves map to a `Content`.
        "response.function_call_arguments.done"
        | "response.output_item.done"
        | "response.content_part.added"
        | "response.content_part.done"
        | "response.in_progress" => EventOutcome::None,
        _ => EventOutcome::None,
    }
}

// endregion

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
    use agent_framework_core::types::{
        FunctionApprovalResponseContent, FunctionResultContent, HostedFileContent,
    };

    fn user(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }

    fn user_with(contents: Vec<Content>) -> ChatMessage {
        ChatMessage::with_contents(Role::user(), contents)
    }

    /// Parse a single Responses output item into its content list.
    fn parse_item(item: Value) -> Vec<Content> {
        let mut contents = Vec::new();
        parse_output_item(&item, &mut contents);
        contents
    }

    fn client() -> OpenAIResponsesClient {
        OpenAIResponsesClient::new("test-key", "gpt-4o-mini")
    }

    // region: request body building

    #[test]
    fn build_body_simple_text() {
        let c = client();
        let body = c.build_body(&[user("Hello there")], &ChatOptions::new(), false);
        assert_eq!(
            body,
            json!({
                "model": "gpt-4o-mini",
                "input": [
                    { "type": "message", "role": "user", "content": [
                        { "type": "input_text", "text": "Hello there" }
                    ]}
                ],
            })
        );
    }

    #[test]
    fn build_body_extracts_leading_system_message_as_instructions() {
        let c = client();
        let messages = vec![ChatMessage::system("Be terse."), user("Hi")];
        let body = c.build_body(&messages, &ChatOptions::new(), false);
        assert_eq!(body["instructions"], json!("Be terse."));
        assert_eq!(
            body["input"],
            json!([
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "Hi" }
                ]}
            ])
        );
    }

    #[test]
    fn build_body_combines_options_instructions_and_system_message() {
        let c = client();
        let messages = vec![ChatMessage::system("Also be nice."), user("Hi")];
        let options = ChatOptions::new().with_instructions("Be terse.");
        let body = c.build_body(&messages, &options, false);
        assert_eq!(body["instructions"], json!("Be terse.\n\nAlso be nice."));
    }

    #[test]
    fn build_body_assistant_text_uses_output_text_type() {
        let c = client();
        let messages = vec![user("Hi"), ChatMessage::assistant("Hello!")];
        let body = c.build_body(&messages, &ChatOptions::new(), false);
        assert_eq!(
            body["input"][1],
            json!({ "type": "message", "role": "assistant", "content": [
                { "type": "output_text", "text": "Hello!" }
            ]})
        );
    }

    #[test]
    fn build_body_function_call_round_trip() {
        let c = client();
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Raw(r#"{"city":"Paris"}"#.to_string())),
        );
        let assistant_msg =
            ChatMessage::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]);
        let tool_msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "call_1",
                Some(json!("18C and sunny")),
            ))],
        );
        let body = c.build_body(
            &[user("weather?"), assistant_msg, tool_msg],
            &ChatOptions::new(),
            false,
        );
        assert_eq!(
            body["input"],
            json!([
                { "type": "message", "role": "user", "content": [
                    { "type": "input_text", "text": "weather?" }
                ]},
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "18C and sunny" },
            ])
        );
    }

    #[test]
    fn build_body_tools_are_flat_not_nested() {
        let c = client();
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get the weather".into(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        };
        let options = ChatOptions::new().with_tool(tool);
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(
            body["tools"],
            json!([{
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": { "type": "object", "properties": {} },
            }])
        );
    }

    #[test]
    fn build_body_tool_choice_required_named() {
        let c = client();
        let options =
            ChatOptions::new().with_tool_choice(ToolMode::Required(Some("get_weather".into())));
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "function", "name": "get_weather" })
        );
    }

    #[test]
    fn build_body_conversation_id_becomes_previous_response_id() {
        let c = client();
        let mut options = ChatOptions::new();
        options.conversation_id = Some("resp_abc123".into());
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(body["previous_response_id"], json!("resp_abc123"));
    }

    #[test]
    fn build_body_max_tokens_becomes_max_output_tokens() {
        let c = client();
        let options = ChatOptions::new().with_max_tokens(256);
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(body["max_output_tokens"], json!(256));
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn build_body_response_format_becomes_text_format() {
        let c = client();
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::JsonSchema {
            name: "answer".into(),
            description: None,
            schema: json!({"type": "object"}),
            strict: Some(true),
        });
        let body = c.build_body(&[user("hi")], &options, false);
        assert_eq!(
            body["text"]["format"],
            json!({ "type": "json_schema", "name": "answer", "schema": {"type": "object"}, "strict": true })
        );
    }

    #[test]
    fn build_body_stream_flag() {
        let c = client();
        let body = c.build_body(&[user("hi")], &ChatOptions::new(), true);
        assert_eq!(body["stream"], json!(true));
    }

    // endregion

    // region: response parsing

    #[test]
    fn parse_response_text_and_usage() {
        let value = json!({
            "id": "resp_123",
            "model": "gpt-4o-mini",
            "status": "completed",
            "output": [
                { "type": "message", "role": "assistant", "content": [
                    { "type": "output_text", "text": "Hello!" }
                ]}
            ],
            "usage": { "input_tokens": 10, "output_tokens": 5, "total_tokens": 15 },
        });
        let resp = parse_response(&value, None);
        assert_eq!(resp.response_id.as_deref(), Some("resp_123"));
        assert_eq!(resp.conversation_id.as_deref(), Some("resp_123"));
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(10));
        assert_eq!(usage.output_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(15));
    }

    #[test]
    fn parse_response_store_false_omits_conversation_id() {
        let value = json!({
            "id": "resp_123",
            "status": "completed",
            "output": [],
        });
        let resp = parse_response(&value, Some(false));
        assert_eq!(resp.conversation_id, None);
    }

    #[test]
    fn parse_response_function_call_sets_tool_calls_finish_reason() {
        let value = json!({
            "id": "resp_123",
            "status": "completed",
            "output": [
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" }
            ],
        });
        let resp = parse_response(&value, None);
        assert_eq!(resp.finish_reason, Some(FinishReason::tool_calls()));
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
    }

    #[test]
    fn parse_response_incomplete_max_output_tokens_is_length() {
        let value = json!({
            "id": "resp_123",
            "status": "incomplete",
            "incomplete_details": { "reason": "max_output_tokens" },
            "output": [],
        });
        let resp = parse_response(&value, None);
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::new(FinishReason::LENGTH))
        );
    }

    #[test]
    fn parse_response_reasoning_becomes_text_reasoning() {
        let value = json!({
            "id": "resp_123",
            "status": "completed",
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "thinking..." }] },
                { "type": "message", "role": "assistant", "content": [{ "type": "output_text", "text": "done" }] },
            ],
        });
        let resp = parse_response(&value, None);
        let contents = &resp.messages[0].contents;
        assert!(matches!(&contents[0], Content::TextReasoning(t) if t.text == "thinking..."));
        assert!(matches!(&contents[1], Content::Text(t) if t.text == "done"));
    }

    // endregion

    // region: streaming

    fn sse_body(events: &[(&str, Value)]) -> String {
        let mut out = String::new();
        for (event, data) in events {
            out.push_str(&format!("event: {event}\ndata: {}\n\n", data));
        }
        out
    }

    async fn collect_updates(text: String) -> Vec<ChatResponseUpdate> {
        // Build a fake reqwest::Response backed by the given SSE text using a
        // tiny local HTTP server would be overkill; instead we drive the
        // event parser directly through the same state machine by feeding
        // the byte stream via `futures::stream::once`.
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = ResponsesSseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            call_ids: HashMap::new(),
            done: false,
            store: None,
        };
        let mut updates = Vec::new();
        // Drain the single chunk manually (mirrors the real unfold body).
        if let Some(Ok(bytes)) = state.byte_stream.next().await {
            let decoded = state.utf8.push(&bytes);
            state.buffer.push_str(&decoded);
            while let Some(pos) = state.buffer.find('\n') {
                let line = state.buffer[..pos].trim().to_string();
                state.buffer.drain(..=pos);
                let Some(data) = line.strip_prefix("data:") else {
                    continue;
                };
                let data = data.trim();
                if data.is_empty() {
                    continue;
                }
                let value: Value = serde_json::from_str(data).unwrap();
                if let EventOutcome::Update(u) =
                    parse_responses_event(&value, &mut state.call_ids, state.store)
                {
                    updates.push(u);
                }
            }
        }
        updates
    }

    #[tokio::test]
    async fn stream_text_only_accumulates() {
        let text = sse_body(&[
            (
                "response.created",
                json!({ "type": "response.created", "response": { "id": "resp_1", "model": "gpt-4o-mini" } }),
            ),
            (
                "response.output_text.delta",
                json!({ "type": "response.output_text.delta", "delta": "Hel" }),
            ),
            (
                "response.output_text.delta",
                json!({ "type": "response.output_text.delta", "delta": "lo!" }),
            ),
            (
                "response.completed",
                json!({ "type": "response.completed", "response": { "id": "resp_1", "model": "gpt-4o-mini", "status": "completed", "output": [], "usage": { "input_tokens": 3, "output_tokens": 2 } } }),
            ),
        ]);
        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.response_id.as_deref(), Some("resp_1"));
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(3));
        assert_eq!(usage.output_token_count, Some(2));
    }

    #[tokio::test]
    async fn stream_tool_call_accumulates_arguments() {
        let text = sse_body(&[
            (
                "response.output_item.added",
                json!({ "type": "response.output_item.added", "output_index": 0, "item": { "type": "function_call", "call_id": "call_1", "name": "get_weather" } }),
            ),
            (
                "response.function_call_arguments.delta",
                json!({ "type": "response.function_call_arguments.delta", "output_index": 0, "delta": "{\"city\":" }),
            ),
            (
                "response.function_call_arguments.delta",
                json!({ "type": "response.function_call_arguments.delta", "output_index": 0, "delta": "\"Paris\"}" }),
            ),
            (
                "response.completed",
                json!({ "type": "response.completed", "response": { "id": "resp_2", "status": "completed", "output": [{"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"}] } }),
            ),
        ]);
        let updates = collect_updates(text).await;
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &json!("Paris")
        );
        assert_eq!(resp.finish_reason, Some(FinishReason::tool_calls()));
    }

    #[tokio::test]
    async fn stream_failed_event_is_error() {
        let text = sse_body(&[(
            "response.failed",
            json!({ "type": "response.failed", "response": { "error": { "message": "boom" } } }),
        )]);
        let stream =
            futures::stream::once(async move { Ok::<_, reqwest::Error>(bytes::Bytes::from(text)) });
        let byte_stream: ByteStream = Box::pin(stream);
        let mut state = ResponsesSseState {
            byte_stream,
            buffer: String::new(),
            utf8: Utf8StreamDecoder::new(),
            queued: VecDeque::new(),
            call_ids: HashMap::new(),
            done: false,
            store: None,
        };
        let bytes = state.byte_stream.next().await.unwrap().unwrap();
        let decoded = state.utf8.push(&bytes);
        state.buffer.push_str(&decoded);
        let mut saw_error = false;
        while let Some(pos) = state.buffer.find('\n') {
            let line = state.buffer[..pos].trim().to_string();
            state.buffer.drain(..=pos);
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data.is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(data).unwrap();
            if let EventOutcome::Error(e) =
                parse_responses_event(&value, &mut state.call_ids, state.store)
            {
                assert!(e.to_string().contains("boom"));
                saw_error = true;
            }
        }
        assert!(
            saw_error,
            "expected a response.failed event to surface an error"
        );
    }

    // endregion

    // region: env-var constructor

    /// Guards `OPENAI_API_KEY` / `OPENAI_BASE_URL` mutation: `cargo test` runs
    /// tests in the same process on multiple threads, and env vars are
    /// process-global, so concurrent set/remove across tests would be racy
    /// without serializing access.
    static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn from_env_reads_api_key_and_base_url() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX against the other env-var test in
        // this module; no other test in this crate touches these variables.
        unsafe {
            std::env::set_var("OPENAI_API_KEY", "sk-test-123");
            std::env::set_var("OPENAI_BASE_URL", "https://example.test/v1");
        }
        let client = OpenAIResponsesClient::from_env("gpt-4o-mini").unwrap();
        assert_eq!(client.inner.api_key, "sk-test-123");
        assert_eq!(client.inner.base_url, "https://example.test/v1");
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_BASE_URL");
        }
    }

    #[test]
    fn from_env_errors_when_api_key_missing() {
        let _guard = ENV_MUTEX.lock().unwrap();
        // SAFETY: serialized by ENV_MUTEX; see above.
        unsafe {
            std::env::remove_var("OPENAI_API_KEY");
            std::env::remove_var("OPENAI_BASE_URL");
        }
        let result = OpenAIResponsesClient::from_env("gpt-4o-mini");
        assert!(result.is_err());
    }
    #[test]
    fn build_body_maps_hosted_tools_to_responses_types() {
        use agent_framework_core::tools::{
            hosted_code_interpreter, hosted_file_search, hosted_mcp, hosted_web_search,
        };
        let c = client();
        let mut options = ChatOptions::new();
        options.tools = vec![
            hosted_web_search(),
            hosted_code_interpreter(),
            hosted_file_search(Some(7)),
            hosted_mcp(
                "docs",
                "https://mcp.example/sse",
                Some(vec!["search".into()]),
            ),
        ];
        let body = c.build_body(&[user("hi")], &options, false);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools[0], json!({ "type": "web_search" }));
        assert_eq!(
            tools[1],
            json!({ "type": "code_interpreter", "container": { "type": "auto" } })
        );
        assert_eq!(tools[2]["type"], "file_search");
        assert_eq!(tools[2]["max_num_results"], json!(7));
        assert_eq!(tools[3]["type"], "mcp");
        assert_eq!(tools[3]["server_url"], "https://mcp.example/sse");
        assert_eq!(tools[3]["allowed_tools"], json!(["search"]));
        assert_eq!(tools[3]["require_approval"], "never");
    }

    // endregion

    // region: multimodal input (Responses)

    #[test]
    fn input_image_uri_becomes_input_image_part() {
        let msg = user_with(vec![Content::Uri(UriContent {
            uri: "https://example.com/cat.png".into(),
            media_type: "image/png".into(),
        })]);
        let input = messages_to_input(&[msg]);
        assert_eq!(
            input[0],
            json!({ "type": "message", "role": "user", "content": [
                { "type": "input_image", "image_url": "https://example.com/cat.png", "detail": "auto" }
            ]})
        );
    }

    #[test]
    fn input_audio_file_and_hosted_file_parts() {
        let msg = user_with(vec![
            Content::Data(DataContent {
                uri: "data:audio/wav;base64,QQ".into(),
                media_type: Some("audio/wav".into()),
            }),
            Content::Data(DataContent {
                uri: "data:application/pdf;base64,JV".into(),
                media_type: Some("application/pdf".into()),
            }),
            Content::HostedFile(HostedFileContent {
                file_id: "file-123".into(),
            }),
        ]);
        let input = messages_to_input(&[msg]);
        // The Responses API keeps the full data URI for audio (unlike Chat
        // Completions, which strips the prefix).
        assert_eq!(
            input[0]["content"],
            json!([
                { "type": "input_audio", "input_audio": { "data": "data:audio/wav;base64,QQ", "format": "wav" } },
                { "type": "input_file", "file_data": "data:application/pdf;base64,JV", "filename": "file" },
                { "type": "input_file", "file_id": "file-123" },
            ])
        );
    }

    #[test]
    fn approval_response_becomes_mcp_approval_response_item() {
        let resp = FunctionApprovalResponseContent {
            approved: true,
            id: "appr_1".into(),
            function_call: FunctionCallContent::new("appr_1", "search", None),
        };
        let msg = user_with(vec![Content::FunctionApprovalResponse(resp)]);
        let input = messages_to_input(&[msg]);
        assert_eq!(
            input[0],
            json!({
                "type": "mcp_approval_response",
                "approval_request_id": "appr_1",
                "approve": true,
            })
        );
    }

    #[test]
    fn approval_request_becomes_mcp_approval_request_item() {
        let req = FunctionApprovalRequestContent {
            id: "appr_1".into(),
            function_call: FunctionCallContent::new(
                "appr_1",
                "search",
                Some(FunctionArguments::Raw(r#"{"q":"x"}"#.into())),
            ),
        };
        let msg = ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionApprovalRequest(req)],
        );
        let input = messages_to_input(&[msg]);
        assert_eq!(
            input[0],
            json!({
                "type": "mcp_approval_request",
                "id": "appr_1",
                "name": "search",
                "arguments": r#"{"q":"x"}"#,
            })
        );
    }

    // endregion

    // region: output parsing (annotations, code interpreter, images, approvals)

    #[test]
    fn output_text_url_citation_annotation() {
        let contents = parse_item(json!({
            "type": "message", "role": "assistant", "content": [{
                "type": "output_text", "text": "See source.",
                "annotations": [{
                    "type": "url_citation", "title": "Src", "url": "https://ex.com",
                    "start_index": 0, "end_index": 3,
                }],
            }],
        }));
        let Content::Text(t) = &contents[0] else {
            panic!("expected text content");
        };
        let ann = t.annotations.as_ref().unwrap();
        assert_eq!(ann[0].title.as_deref(), Some("Src"));
        assert_eq!(ann[0].url.as_deref(), Some("https://ex.com"));
        let region = &ann[0].annotated_regions.as_ref().unwrap()[0];
        assert_eq!(region.start_index, Some(0));
        assert_eq!(region.end_index, Some(3));
    }

    #[test]
    fn output_text_file_and_container_citations() {
        let contents = parse_item(json!({
            "type": "message", "role": "assistant", "content": [{
                "type": "output_text", "text": "x",
                "annotations": [
                    { "type": "file_citation", "filename": "doc.pdf", "file_id": "file-1", "index": 2 },
                    { "type": "file_path", "file_id": "file-2", "index": 0 },
                    { "type": "container_file_citation", "filename": "c.txt", "file_id": "file-3",
                      "container_id": "cont-1", "start_index": 1, "end_index": 4 },
                ],
            }],
        }));
        let Content::Text(t) = &contents[0] else {
            panic!("expected text content");
        };
        let ann = t.annotations.as_ref().unwrap();
        assert_eq!(ann[0].url.as_deref(), Some("doc.pdf"));
        assert_eq!(ann[0].file_id.as_deref(), Some("file-1"));
        assert_eq!(ann[1].file_id.as_deref(), Some("file-2"));
        assert_eq!(ann[2].file_id.as_deref(), Some("file-3"));
        assert_eq!(ann[2].url.as_deref(), Some("c.txt"));
        assert_eq!(
            ann[2].annotated_regions.as_ref().unwrap()[0].end_index,
            Some(4)
        );
    }

    #[test]
    fn code_interpreter_outputs_become_text_and_uri() {
        let contents = parse_item(json!({
            "type": "code_interpreter_call",
            "outputs": [
                { "type": "logs", "logs": "hello stdout" },
                { "type": "image", "url": "https://ex.com/plot.png" },
            ],
        }));
        assert!(matches!(&contents[0], Content::Text(t) if t.text == "hello stdout"));
        assert!(
            matches!(&contents[1], Content::Uri(u) if u.uri == "https://ex.com/plot.png" && u.media_type == "image")
        );
    }

    #[test]
    fn code_interpreter_without_outputs_falls_back_to_code() {
        let contents = parse_item(json!({
            "type": "code_interpreter_call", "code": "print(1)",
        }));
        assert!(matches!(&contents[0], Content::Text(t) if t.text == "print(1)"));
    }

    #[test]
    fn image_generation_raw_base64_becomes_png_data() {
        let contents = parse_item(json!({
            "type": "image_generation_call", "result": "AAAABBBB",
        }));
        let Content::Data(d) = &contents[0] else {
            panic!("expected data content");
        };
        assert_eq!(d.uri, "data:image/png;base64,AAAABBBB");
        assert_eq!(d.media_type.as_deref(), Some("image/png"));
    }

    #[test]
    fn image_generation_data_uri_keeps_stated_media_type() {
        let contents = parse_item(json!({
            "type": "image_generation_call", "result": "data:image/webp;base64,ZZZ",
        }));
        let Content::Data(d) = &contents[0] else {
            panic!("expected data content");
        };
        assert_eq!(d.uri, "data:image/webp;base64,ZZZ");
        assert_eq!(d.media_type.as_deref(), Some("image/webp"));
    }

    #[test]
    fn mcp_approval_request_output_round_trips_into_response() {
        let contents = parse_item(json!({
            "type": "mcp_approval_request",
            "id": "appr_9", "name": "search", "arguments": r#"{"q":"rust"}"#,
            "server_label": "docs",
        }));
        let Content::FunctionApprovalRequest(req) = &contents[0] else {
            panic!("expected approval request");
        };
        assert_eq!(req.id, "appr_9");
        assert_eq!(req.function_call.call_id, "appr_9");
        assert_eq!(req.function_call.name, "search");

        // The id round-trips into the request's response item (item 5).
        let msg = user_with(vec![Content::FunctionApprovalResponse(
            req.create_response(true),
        )]);
        let input = messages_to_input(&[msg]);
        assert_eq!(input[0]["type"], json!("mcp_approval_response"));
        assert_eq!(input[0]["approval_request_id"], json!("appr_9"));
        assert_eq!(input[0]["approve"], json!(true));
    }

    // endregion

    // region: streaming reasoning

    fn reasoning_event(value: Value) -> EventOutcome {
        let mut ids = HashMap::new();
        parse_responses_event(&value, &mut ids, None)
    }

    #[test]
    fn reasoning_text_delta_and_done_map_to_reasoning_content() {
        let EventOutcome::Update(delta) =
            reasoning_event(json!({ "type": "response.reasoning_text.delta", "delta": "Th" }))
        else {
            panic!("expected update");
        };
        assert!(matches!(&delta.contents[0], Content::TextReasoning(t) if t.text == "Th"));

        let EventOutcome::Update(done) =
            reasoning_event(json!({ "type": "response.reasoning_text.done", "text": "Think" }))
        else {
            panic!("expected update");
        };
        assert!(matches!(&done.contents[0], Content::TextReasoning(t) if t.text == "Think"));
    }

    #[test]
    fn reasoning_summary_text_events_map_to_reasoning_content() {
        let EventOutcome::Update(delta) = reasoning_event(
            json!({ "type": "response.reasoning_summary_text.delta", "delta": "sum" }),
        ) else {
            panic!("expected update");
        };
        assert!(matches!(&delta.contents[0], Content::TextReasoning(t) if t.text == "sum"));

        let EventOutcome::Update(done) = reasoning_event(
            json!({ "type": "response.reasoning_summary_text.done", "text": "summary" }),
        ) else {
            panic!("expected update");
        };
        assert!(matches!(&done.contents[0], Content::TextReasoning(t) if t.text == "summary"));
    }

    // endregion

    // region: hosted-tool config passthrough

    fn hosted(kind: ToolKind, name: &str, params: Value) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: String::new(),
            parameters: params,
            kind,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        }
    }

    #[test]
    fn web_search_passes_through_user_location() {
        let tool = hosted(
            ToolKind::HostedWebSearch,
            "web_search",
            json!({ "user_location": { "city": "Paris", "country": "FR" } }),
        );
        assert_eq!(
            tool_to_responses_spec(&tool),
            json!({
                "type": "web_search",
                "user_location": { "type": "approximate", "city": "Paris", "country": "FR" },
            })
        );
    }

    #[test]
    fn file_search_passes_vector_store_ids_and_max_results_param() {
        let tool = hosted(
            ToolKind::HostedFileSearch { max_results: None },
            "file_search",
            json!({ "vector_store_ids": ["vs_1"], "max_results": 12 }),
        );
        let spec = tool_to_responses_spec(&tool);
        assert_eq!(spec["vector_store_ids"], json!(["vs_1"]));
        assert_eq!(spec["max_num_results"], json!(12));
    }

    #[test]
    fn code_interpreter_passes_file_ids_and_container_override() {
        let with_files = hosted(
            ToolKind::HostedCodeInterpreter,
            "ci",
            json!({ "file_ids": ["file-1", "file-2"] }),
        );
        assert_eq!(
            tool_to_responses_spec(&with_files)["container"],
            json!({ "type": "auto", "file_ids": ["file-1", "file-2"] })
        );
        let with_container = hosted(
            ToolKind::HostedCodeInterpreter,
            "ci",
            json!({ "container": { "type": "secure", "id": "c1" } }),
        );
        assert_eq!(
            tool_to_responses_spec(&with_container)["container"],
            json!({ "type": "secure", "id": "c1" })
        );
    }

    #[test]
    fn mcp_passes_headers_and_string_approval_mode_override() {
        let tool = hosted(
            ToolKind::HostedMcp {
                url: "https://mcp/sse".into(),
                allowed_tools: None,
            },
            "docs",
            json!({ "headers": { "Authorization": "Bearer x" }, "approval_mode": "always_require" }),
        );
        // The enum default is NeverRequire; the parameter overrides it.
        let spec = tool_to_responses_spec(&tool);
        assert_eq!(spec["headers"], json!({ "Authorization": "Bearer x" }));
        assert_eq!(spec["require_approval"], json!("always"));
    }

    #[test]
    fn mcp_object_approval_mode_maps_to_tool_name_lists() {
        let tool = hosted(
            ToolKind::HostedMcp {
                url: "https://mcp/sse".into(),
                allowed_tools: None,
            },
            "docs",
            json!({ "approval_mode": { "always": ["delete"], "never": ["read"] } }),
        );
        assert_eq!(
            tool_to_responses_spec(&tool)["require_approval"],
            json!({ "always": { "tool_names": ["delete"] }, "never": { "tool_names": ["read"] } })
        );
    }

    // endregion
}
