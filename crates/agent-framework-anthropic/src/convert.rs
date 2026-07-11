//! Conversion between framework types and the Anthropic Messages API wire
//! format.

use std::collections::{BTreeSet, HashMap};

use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, CitationAnnotation, Content, DataContent, FinishReason,
    FunctionArguments, FunctionCallContent, FunctionResultContent, HostedFileContent,
    ResponseFormat, Role, TextContent, TextReasoningContent, TextSpanRegion, ToolMode, UriContent,
    UsageContent, UsageDetails,
};
use serde_json::{json, Map, Value};

/// The beta flags upstream's Python `AnthropicClient` unconditionally enables
/// on every request (`BETA_FLAGS` in `agent_framework_anthropic/_chat_client.py`
/// ~line 54), unioned with any client- or request-level additions and sent via
/// the `anthropic-beta` header. Verified against `_create_run_options`
/// (~line 254-264): `"betas": {*BETA_FLAGS, *self.additional_beta_flags, *betas}`
/// is built for *every* `beta.messages.create` call, not only ones that pass
/// hosted tools/MCP servers -- there is no conditional gating on tool
/// presence.
pub const DEFAULT_BETA_FLAGS: &[&str] = &["mcp-client-2025-04-04", "code-execution-2025-08-25"];

/// The `ChatOptions::additional_properties` key upstream pops per-request
/// additional beta flags from (`_create_run_options`, ~line 254-255).
const ADDITIONAL_BETA_FLAGS_KEY: &str = "additional_beta_flags";

/// Compute the full, deduplicated set of `anthropic-beta` flags for a
/// request: the always-on [`DEFAULT_BETA_FLAGS`], the client-level
/// `additional_beta_flags` (constructor option, mirroring upstream's
/// `self.additional_beta_flags`), and any per-request `additional_beta_flags`
/// found under `ChatOptions::additional_properties`.
///
/// Mirrors upstream's
/// `chat_options.additional_properties.pop("additional_beta_flags")`: the key
/// is removed from `options.additional_properties` so it is not also copied
/// into the request body as a stray top-level field by [`build_request`].
pub(crate) fn compute_beta_flags(
    options: &mut ChatOptions,
    client_additional: &[String],
) -> Vec<String> {
    let mut flags: BTreeSet<String> = DEFAULT_BETA_FLAGS.iter().map(|s| s.to_string()).collect();
    flags.extend(client_additional.iter().cloned());
    if let Some(value) = options
        .additional_properties
        .remove(ADDITIONAL_BETA_FLAGS_KEY)
    {
        if let Some(arr) = value.as_array() {
            flags.extend(arr.iter().filter_map(Value::as_str).map(str::to_string));
        }
    }
    flags.into_iter().collect()
}

/// Build a full Anthropic `POST /v1/messages` request body.
pub fn build_request(
    messages: &[ChatMessage],
    options: &ChatOptions,
    model: &str,
    max_tokens: u32,
    stream: bool,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("max_tokens".into(), json!(max_tokens));

    let (system, rest) = extract_system(messages, options.instructions.as_deref());
    let system = append_response_format_instructions(system, options.response_format.as_ref());
    if let Some(system) = system {
        body.insert("system".into(), json!(system));
    }
    body.insert("messages".into(), json!(messages_to_anthropic(rest)));

    if let Some(t) = options.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(t) = options.top_p {
        body.insert("top_p".into(), json!(t));
    }
    if let Some(stop) = &options.stop {
        body.insert("stop_sequences".into(), json!(stop));
    }

    if !options.tools.is_empty() {
        let (tools, mcp_servers) = tools_to_anthropic(&options.tools);
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
        }
        if !mcp_servers.is_empty() {
            body.insert("mcp_servers".into(), json!(mcp_servers));
        }
    }
    if let Some(tool_choice) = &options.tool_choice {
        body.insert(
            "tool_choice".into(),
            tool_choice_to_anthropic(tool_choice, options.allow_multiple_tool_calls),
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

/// Split a leading system message (and/or `ChatOptions::instructions`) out
/// into Anthropic's top-level `system` field, returning the remaining
/// messages to convert into the `messages` array.
///
/// Mirrors the Python `AnthropicClient`, which only pulls `messages[0]` when
/// it is a system message; any other content keeps its original position and
/// maps to a `user` turn (see [`messages_to_anthropic`]'s role mapping).
pub fn extract_system<'a>(
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

/// Fold a requested [`ResponseFormat`] into the system prompt.
///
/// The Anthropic Messages API has **no** native `response_format` /
/// structured-output parameter — confirmed against the upstream Python
/// `AnthropicClient._create_run_options` (`agent_framework_anthropic/_chat_client.py`),
/// which builds its `run_options` dict from `temperature`, `top_p`, `stop`,
/// `tool_choice`, tools, etc. but never reads `chat_options.response_format`
/// at all, and against .NET's `Microsoft.Agents.AI.Anthropic` extensions,
/// which likewise have no `ResponseFormat` handling. So this isn't a Rust
/// port gap to close by mapping onto a wire field that doesn't exist; it's a
/// gap in the underlying API. Rather than silently dropping the option (the
/// previous behavior here, and the *actual* behavior of both reference
/// implementations today), this appends an explicit natural-language
/// instruction to the system prompt as a pragmatic, observable fallback:
///
/// * [`ResponseFormat::Text`] (or no format): no-op.
/// * [`ResponseFormat::JsonObject`]: instructs the model to respond with a
///   bare JSON object.
/// * [`ResponseFormat::JsonSchema`]: instructs the model to respond with a
///   JSON object conforming to the embedded schema, and includes the schema
///   itself (pretty-printed) in the prompt.
fn append_response_format_instructions(
    system: Option<String>,
    format: Option<&ResponseFormat>,
) -> Option<String> {
    let instruction = match format {
        None | Some(ResponseFormat::Text) => return system,
        Some(ResponseFormat::JsonObject) => {
            "Respond only with a single valid JSON object. Do not include any \
             explanation, preamble, or markdown code fences before or after the JSON."
                .to_string()
        }
        Some(ResponseFormat::JsonSchema { name, schema, .. }) => {
            let pretty =
                serde_json::to_string_pretty(schema).unwrap_or_else(|_| schema.to_string());
            format!(
                "Respond only with a single valid JSON object that conforms exactly to \
                 the following JSON Schema (named \"{name}\"). Do not include any \
                 explanation, preamble, or markdown code fences before or after the JSON.\n\n\
                 JSON Schema:\n{pretty}"
            )
        }
    };
    Some(match system {
        Some(existing) if !existing.is_empty() => format!("{existing}\n\n{instruction}"),
        _ => instruction,
    })
}

/// Convert framework messages into Anthropic's `messages` array.
///
/// Anthropic has no `system` or `tool` role: everything that isn't
/// `assistant` (including tool results) is sent as a `user` turn, matching
/// the Python client's `ROLE_MAP`.
pub fn messages_to_anthropic(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = if msg.role == Role::assistant() {
            "assistant"
        } else {
            "user"
        };
        let mut blocks: Vec<Value> = Vec::new();
        for content in &msg.contents {
            match content {
                Content::Text(t) => blocks.push(json!({ "type": "text", "text": t.text })),
                Content::TextReasoning(t) => {
                    blocks.push(json!({ "type": "thinking", "thinking": t.text }))
                }
                Content::FunctionCall(fc) => blocks.push(function_call_block(fc)),
                Content::FunctionResult(fr) => blocks.push(function_result_block(fr)),
                Content::Data(dc) => {
                    if let Some(block) = image_block_from_data(dc) {
                        blocks.push(block);
                    }
                }
                Content::Uri(uc) => {
                    if let Some(block) = image_block_from_uri(uc) {
                        blocks.push(block);
                    }
                }
                _ => {}
            }
        }
        if blocks.is_empty() {
            // Anthropic rejects messages with an empty content array.
            continue;
        }
        out.push(json!({ "role": role, "content": blocks }));
    }
    normalize_role_alternation(out)
}

/// Enforce the Messages API's conversation-shape rules: messages must
/// alternate between `user` and `assistant`, starting with `user`.
/// Consecutive same-role messages (common in orchestration transcripts,
/// e.g. several user turns from a group chat) are merged by concatenating
/// their content blocks; a leading assistant message gets a minimal
/// synthetic user turn inserted before it so the greeting is preserved.
fn normalize_role_alternation(messages: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    for msg in messages {
        match out.last_mut() {
            Some(prev) if prev["role"] == msg["role"] => {
                if let (Some(prev_blocks), Some(new_blocks)) =
                    (prev["content"].as_array_mut(), msg["content"].as_array())
                {
                    prev_blocks.extend(new_blocks.iter().cloned());
                }
            }
            _ => out.push(msg),
        }
    }
    if out.first().map(|m| m["role"] == "assistant") == Some(true) {
        out.insert(
            0,
            json!({
                "role": "user",
                "content": [{ "type": "text", "text": "(continuing the conversation)" }]
            }),
        );
    }
    out
}

fn function_call_block(fc: &FunctionCallContent) -> Value {
    let input = fc.parse_arguments().unwrap_or_default();
    json!({
        "type": "tool_use",
        "id": fc.call_id,
        "name": fc.name,
        "input": Value::Object(input.into_iter().collect()),
    })
}

fn function_result_block(fr: &FunctionResultContent) -> Value {
    let mut block = Map::new();
    block.insert("type".into(), json!("tool_result"));
    block.insert("tool_use_id".into(), json!(fr.call_id));
    block.insert("content".into(), json!(result_text(fr)));
    if fr.exception.is_some() {
        block.insert("is_error".into(), json!(true));
    }
    Value::Object(block)
}

fn result_text(fr: &FunctionResultContent) -> String {
    if let Some(exc) = &fr.exception {
        return exc.clone();
    }
    match &fr.result {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

/// Build an `{"type":"image","source":{"type":"base64",...}}` block from a
/// `data:` URI, without needing a base64 encoder: [`DataContent::uri`] is
/// already base64 text after the `base64,` marker (per
/// `DataContent::from_bytes` in `agent-framework-core`), so we just slice it
/// out.
fn image_block_from_data(dc: &DataContent) -> Option<Value> {
    let is_image = dc
        .media_type
        .as_deref()
        .map(is_image_media_type)
        .unwrap_or_else(|| dc.uri.starts_with("data:image/"));
    if !is_image {
        return None;
    }
    let (parsed_media_type, data) = split_data_uri(&dc.uri)?;
    let media_type = dc.media_type.clone().unwrap_or(parsed_media_type);
    Some(json!({
        "type": "image",
        "source": { "type": "base64", "media_type": media_type, "data": data }
    }))
}

fn image_block_from_uri(uc: &UriContent) -> Option<Value> {
    if !is_image_media_type(&uc.media_type) {
        return None;
    }
    Some(json!({ "type": "image", "source": { "type": "url", "url": uc.uri } }))
}

fn split_data_uri(uri: &str) -> Option<(String, String)> {
    let rest = uri.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta
        .split(';')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    Some((media_type, data.to_string()))
}

fn is_image_media_type(media_type: &str) -> bool {
    media_type.starts_with("image/")
}

/// Convert tool definitions into Anthropic's request shape.
///
/// Returns `(tools, mcp_servers)`: ordinary function tools and most hosted
/// tool markers become entries in the returned `tools` list (destined for the
/// request's top-level `tools` field), while [`ToolKind::HostedMcp`] tools
/// become entries in `mcp_servers` instead -- Anthropic's MCP connector is a
/// separate top-level `mcp_servers` request field, not a `tools[]` entry.
///
/// Mirrors upstream's `_convert_tools_to_anthropic_format`
/// (`_chat_client.py` ~379-430):
///
/// * [`ToolKind::Function`] -> `{"type":"custom","name":...,"description":...,"input_schema":...}`
///   (~390-396).
/// * [`ToolKind::HostedWebSearch`] -> `{"type":"web_search_20250305","name":"web_search"}`,
///   optionally merged with extra config (~397-404). Upstream reads arbitrary
///   keys from `tool.additional_properties`; Rust's [`ToolKind::HostedWebSearch`]
///   has no such bag (core, out of scope here), so `"max_uses"` /
///   `"user_location"` are read from [`ToolDefinition::parameters`] instead,
///   the closest stand-in this crate can reach without touching core.
/// * [`ToolKind::HostedCodeInterpreter`] -> `{"type":"code_execution_20250825","name":"code_execution"}`
///   (~405-410), no extra config.
/// * [`ToolKind::HostedMcp`] -> an `mcp_servers[]` entry
///   `{"type":"url","name":...,"url":...}`, plus `tool_configuration.allowed_tools`
///   when non-empty and `authorization_token` when an `"authorization"` header
///   is present (~411-421). Rust's [`ToolKind::HostedMcp`] has no `headers`
///   field (core), so the authorization header is read from
///   `ToolDefinition::parameters["headers"]["authorization"]` instead.
/// * [`ToolKind::HostedFileSearch`]: unknown to the Anthropic API (upstream
///   has no case for it either -- it would fall through to the `case _:`
///   debug log), so it is skipped with a `tracing::warn!`.
///
/// The `MutableMapping()` case upstream uses for raw pass-through dict tools
/// has no Rust equivalent ([`ToolDefinition`] is always structured) and is
/// not applicable here.
pub fn tools_to_anthropic(tools: &[ToolDefinition]) -> (Vec<Value>, Vec<Value>) {
    let mut tool_list = Vec::new();
    let mut mcp_servers = Vec::new();
    for t in tools {
        match &t.kind {
            ToolKind::Function => {
                tool_list.push(json!({
                    "type": "custom",
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }));
            }
            ToolKind::HostedWebSearch => {
                let mut search_tool = Map::new();
                search_tool.insert("type".into(), json!("web_search_20250305"));
                search_tool.insert("name".into(), json!("web_search"));
                if let Some(max_uses) = t.parameters.get("max_uses") {
                    search_tool.insert("max_uses".into(), max_uses.clone());
                }
                if let Some(user_location) = t.parameters.get("user_location") {
                    search_tool.insert("user_location".into(), user_location.clone());
                }
                tool_list.push(Value::Object(search_tool));
            }
            ToolKind::HostedCodeInterpreter => {
                tool_list.push(json!({
                    "type": "code_execution_20250825",
                    "name": "code_execution",
                }));
            }
            ToolKind::HostedMcp { url, allowed_tools } => {
                let mut server_def = Map::new();
                server_def.insert("type".into(), json!("url"));
                server_def.insert("name".into(), json!(t.name));
                server_def.insert("url".into(), json!(url));
                if let Some(allowed) = allowed_tools {
                    if !allowed.is_empty() {
                        server_def.insert(
                            "tool_configuration".into(),
                            json!({ "allowed_tools": allowed }),
                        );
                    }
                }
                if let Some(auth) = t
                    .parameters
                    .get("headers")
                    .and_then(|h| h.get("authorization"))
                    .and_then(Value::as_str)
                {
                    server_def.insert("authorization_token".into(), json!(auth));
                }
                mcp_servers.push(Value::Object(server_def));
            }
            ToolKind::HostedFileSearch { .. } => {
                tracing::warn!(
                    tool = %t.name,
                    "Anthropic: hosted file-search tools are not supported by the Anthropic Messages API; skipping"
                );
            }
        }
    }
    (tool_list, mcp_servers)
}

fn tool_choice_to_anthropic(mode: &ToolMode, allow_multiple: Option<bool>) -> Value {
    let mut obj = Map::new();
    match mode {
        ToolMode::Auto => {
            obj.insert("type".into(), json!("auto"));
        }
        ToolMode::Required(Some(name)) => {
            obj.insert("type".into(), json!("tool"));
            obj.insert("name".into(), json!(name));
        }
        ToolMode::Required(None) => {
            obj.insert("type".into(), json!("any"));
        }
        ToolMode::None => {
            obj.insert("type".into(), json!("none"));
        }
    }
    if !matches!(mode, ToolMode::None) {
        if let Some(allow) = allow_multiple {
            obj.insert("disable_parallel_tool_use".into(), json!(!allow));
        }
    }
    Value::Object(obj)
}

/// Parse a full (non-streaming) Anthropic `Message` response.
pub fn parse_response(value: &Value) -> ChatResponse {
    let mut response = ChatResponse {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model_id: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    let contents = value
        .get("content")
        .and_then(Value::as_array)
        .map(|blocks| parse_content_blocks(blocks))
        .unwrap_or_default();

    let mut message = ChatMessage::with_contents(Role::assistant(), contents);
    message.message_id = response.response_id.clone();
    response.messages.push(message);

    if let Some(reason) = value.get("stop_reason").and_then(Value::as_str) {
        response.finish_reason = Some(map_stop_reason(reason));
    }
    if let Some(usage) = value.get("usage") {
        response.usage_details = Some(parse_usage(usage));
    }
    response
}

/// Parse Anthropic content blocks (a full response's `content` array, or a
/// single-element slice built from a streaming `content_block_start`'s
/// `content_block`) into framework [`Content`] items.
///
/// Mirrors upstream's `_parse_message_contents` (`_chat_client.py` ~521-609),
/// which takes and returns a list so that a single hosted-tool block can
/// expand into more than one [`Content`] item (see the
/// `code_execution_tool_result` case below). Handled block types:
///
/// * `text` -> [`Content::Text`], with citations parsed via
///   [`parse_citations`] (~528-535).
/// * `tool_use` | `mcp_tool_use` | `server_tool_use` -> [`Content::FunctionCall`]
///   (~536-545): hosted-tool invocations (web search, code execution, MCP)
///   surface the same way a plain function call does.
/// * `mcp_tool_result` -> [`Content::FunctionResult`] (~546-557): if the
///   block's `content` is a JSON array it is recursively parsed through this
///   same function (mirroring `self._parse_message_contents(content_block.content)`);
///   otherwise the raw value is used as-is.
/// * `web_search_tool_result` | `web_fetch_tool_result` -> [`Content::FunctionResult`]
///   (~558-567) with the raw `content` value (NOT recursively parsed --
///   upstream only recurses for `mcp_tool_result`).
/// * `code_execution_tool_result` | `bash_code_execution_tool_result` |
///   `text_editor_code_execution_tool_result` -> (~568-594): when the nested
///   `content.type` is `bash_code_execution_result` or
///   `code_execution_result`, each item of its nested `content` array that
///   carries a `file_id` becomes a [`Content::HostedFile`] emitted *before*
///   the trailing [`Content::FunctionResult`] (whose `result` is always the
///   whole nested `content` object, unparsed). `text_editor_code_execution_tool_result`'s
///   nested content is never one of those two types, so it only ever
///   produces the trailing `FunctionResult` -- verified against the
///   `anthropic` Python SDK's block schemas (`BetaTextEditorCodeExecutionToolResultBlock`'s
///   `content` union has no `code_execution_result`/`bash_code_execution_result`
///   member).
/// * `thinking` -> [`Content::TextReasoning`].
/// * anything else -> skipped with a `tracing::debug!`, mirroring upstream's
///   trailing `case _: logger.debug(...)`.
///
/// Two upstream behaviors have no Rust equivalent and are intentionally
/// dropped:
///
/// * Upstream tracks `self._last_call_id_name` to backfill a `name` onto the
///   `FunctionResultContent` produced for `mcp_tool_result` /
///   `web_search_tool_result` / `code_execution_tool_result` blocks. Rust's
///   [`FunctionResultContent`] (core, out of scope here) has no `name` field
///   at all, so there is nothing to backfill.
/// * Upstream also uses `self._last_call_id_name` to recover the `call_id`
///   for a streaming `input_json_delta`. Rust's streaming path
///   (`parse_stream_event` in `lib.rs`) already threads `call_id` per content
///   block *index* via `tool_use_ids`, which is strictly more correct for
///   interleaved concurrent tool calls, so it is left as-is rather than
///   downgraded to upstream's single-slot tracking.
pub(crate) fn parse_content_blocks(blocks: &[Value]) -> Vec<Content> {
    let mut out = Vec::with_capacity(blocks.len());
    for block in blocks {
        let Some(block_type) = block.get("type").and_then(Value::as_str) else {
            continue;
        };
        match block_type {
            "text" => {
                let text = block
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                out.push(Content::Text(TextContent {
                    text: text.to_string(),
                    annotations: parse_citations(block),
                }));
            }
            "tool_use" | "mcp_tool_use" | "server_tool_use" => {
                let id = block
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let input = match block.get("input") {
                    Some(Value::Object(m)) => m.clone().into_iter().collect(),
                    _ => HashMap::new(),
                };
                out.push(Content::FunctionCall(FunctionCallContent::new(
                    id,
                    name,
                    Some(FunctionArguments::Object(input)),
                )));
            }
            "mcp_tool_result" => {
                let call_id = tool_use_id(block);
                let result = match block.get("content") {
                    Some(Value::Array(items)) => {
                        serde_json::to_value(parse_content_blocks(items)).unwrap_or(Value::Null)
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                out.push(Content::FunctionResult(FunctionResultContent::new(
                    call_id,
                    Some(result),
                )));
            }
            "web_search_tool_result" | "web_fetch_tool_result" => {
                let call_id = tool_use_id(block);
                let result = block.get("content").cloned().unwrap_or(Value::Null);
                out.push(Content::FunctionResult(FunctionResultContent::new(
                    call_id,
                    Some(result),
                )));
            }
            "code_execution_tool_result"
            | "bash_code_execution_tool_result"
            | "text_editor_code_execution_tool_result" => {
                let call_id = tool_use_id(block);
                let nested = block.get("content");
                if let Some(nc) = nested {
                    let nc_type = nc.get("type").and_then(Value::as_str);
                    if matches!(
                        nc_type,
                        Some("bash_code_execution_result") | Some("code_execution_result")
                    ) {
                        if let Some(items) = nc.get("content").and_then(Value::as_array) {
                            for item in items {
                                if let Some(file_id) = item.get("file_id").and_then(Value::as_str) {
                                    out.push(Content::HostedFile(HostedFileContent {
                                        file_id: file_id.to_string(),
                                    }));
                                }
                            }
                        }
                    }
                }
                out.push(Content::FunctionResult(FunctionResultContent::new(
                    call_id,
                    Some(nested.cloned().unwrap_or(Value::Null)),
                )));
            }
            "thinking" => {
                out.push(Content::TextReasoning(TextReasoningContent {
                    text: block
                        .get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    annotations: None,
                }));
            }
            other => {
                tracing::debug!(block_type = %other, "Anthropic: ignoring unsupported content block type");
            }
        }
    }
    out
}

/// The `tool_use_id` field shared by every hosted-tool result block type.
fn tool_use_id(block: &Value) -> String {
    block
        .get("tool_use_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Parse the `citations` array on a text content block into
/// [`CitationAnnotation`]s. Mirrors upstream's `_parse_citations`
/// (`_chat_client.py` ~611-670), including which field feeds `title` for
/// each citation type:
///
/// * `char_location` / `page_location` / `content_block_location`: `snippet`
///   from `cited_text`, `file_id` when present, and one `annotated_regions`
///   span from the type's start/end pair (char index, page number, or block
///   index respectively).
/// * `web_search_result_location`: `title`, `snippet` from `cited_text`, and
///   `url`.
/// * `search_result_location`: `title`, `snippet` from `cited_text`, `url`
///   from `source`, and an `annotated_regions` span from the block index
///   pair.
/// * An unrecognized citation `type` still produces an (empty) annotation --
///   upstream unconditionally appends `cit` after the `match` regardless of
///   which arm (or the fallback `case _`) ran (~667-669).
///
/// `title` note: upstream's `page_location` and `content_block_location`
/// cases read `citation.document_title`, but `char_location` reads
/// `citation.title` (~622) -- and per the `anthropic` Python SDK's
/// `BetaCitationCharLocation` model, `char_location` citations have *no*
/// `title` field, only `document_title` (identical to its two siblings).
/// This looks like an upstream copy/paste bug (`char_location`'s branch
/// resembles `web_search_result_location`/`search_result_location`, which
/// legitimately use `.title`) rather than intentional behavior, since the
/// real API never sends a `title` key on a `char_location` citation. It is
/// mirrored here literally (`char_location` reads wire key `"title"`, which
/// in practice is always absent) rather than "corrected" to `document_title`,
/// per this task's mandate to match upstream's exact behavior; flagged in
/// the implementation report.
pub(crate) fn parse_citations(block: &Value) -> Option<Vec<CitationAnnotation>> {
    let citations = block.get("citations").and_then(Value::as_array)?;
    if citations.is_empty() {
        return None;
    }
    let mut annotations = Vec::with_capacity(citations.len());
    for citation in citations {
        let mut cit = CitationAnnotation::default();
        // Plain (possibly-absent) string field, assigned unconditionally --
        // mirrors upstream's bare `cit.title = citation.xxx` /
        // `cit.snippet = citation.cited_text` / `cit.url = citation.xxx`,
        // which set `None`/`""` through just as readily as a real value.
        let str_field = |key: &str| {
            citation
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        // `file_id` is the one field upstream gates on truthiness
        // (`if citation.file_id: cit.file_id = citation.file_id`), so an
        // empty string is treated the same as absent.
        let truthy_str = |key: &str| {
            citation
                .get(key)
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };
        match citation.get("type").and_then(Value::as_str) {
            Some("char_location") => {
                // See doc comment: upstream reads `citation.title` here, not
                // `citation.document_title` (likely a bug), mirrored as-is.
                cit.title = str_field("title");
                cit.snippet = str_field("cited_text");
                cit.file_id = truthy_str("file_id");
                cit.annotated_regions = Some(vec![TextSpanRegion {
                    start_index: citation.get("start_char_index").and_then(Value::as_i64),
                    end_index: citation.get("end_char_index").and_then(Value::as_i64),
                }]);
            }
            Some("page_location") => {
                cit.title = str_field("document_title");
                cit.snippet = str_field("cited_text");
                cit.file_id = truthy_str("file_id");
                cit.annotated_regions = Some(vec![TextSpanRegion {
                    start_index: citation.get("start_page_number").and_then(Value::as_i64),
                    end_index: citation.get("end_page_number").and_then(Value::as_i64),
                }]);
            }
            Some("content_block_location") => {
                cit.title = str_field("document_title");
                cit.snippet = str_field("cited_text");
                cit.file_id = truthy_str("file_id");
                cit.annotated_regions = Some(vec![TextSpanRegion {
                    start_index: citation.get("start_block_index").and_then(Value::as_i64),
                    end_index: citation.get("end_block_index").and_then(Value::as_i64),
                }]);
            }
            Some("web_search_result_location") => {
                cit.title = str_field("title");
                cit.snippet = str_field("cited_text");
                cit.url = str_field("url");
            }
            Some("search_result_location") => {
                cit.title = str_field("title");
                cit.snippet = str_field("cited_text");
                cit.url = str_field("source");
                cit.annotated_regions = Some(vec![TextSpanRegion {
                    start_index: citation.get("start_block_index").and_then(Value::as_i64),
                    end_index: citation.get("end_block_index").and_then(Value::as_i64),
                }]);
            }
            other => {
                tracing::debug!(
                    citation_type = ?other,
                    "Anthropic: unknown citation type encountered"
                );
            }
        }
        annotations.push(cit);
    }
    if annotations.is_empty() {
        None
    } else {
        Some(annotations)
    }
}

/// Map Anthropic's `stop_reason` to the shared [`FinishReason`].
pub(crate) fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::stop(),
        "max_tokens" => FinishReason::new(FinishReason::LENGTH),
        "tool_use" => FinishReason::tool_calls(),
        "refusal" => FinishReason::new(FinishReason::CONTENT_FILTER),
        "pause_turn" => FinishReason::stop(),
        other => FinishReason::new(other),
    }
}

/// Parse an Anthropic `usage` object (input/output tokens plus prompt-cache
/// counts) into [`UsageDetails`].
pub(crate) fn parse_usage(usage: &Value) -> UsageDetails {
    let mut details = UsageDetails {
        input_token_count: usage.get("input_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("output_tokens").and_then(Value::as_u64),
        total_token_count: None,
        additional_counts: Default::default(),
    };
    if let (Some(i), Some(o)) = (details.input_token_count, details.output_token_count) {
        details.total_token_count = Some(i + o);
    }
    if let Some(v) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        details
            .additional_counts
            .insert("anthropic.cache_creation_input_tokens".into(), v);
    }
    if let Some(v) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        details
            .additional_counts
            .insert("anthropic.cache_read_input_tokens".into(), v);
    }
    details
}

/// Parse `usage` at `message_start` time: only `input_tokens` (plus cache
/// counts) are taken. `message_start.usage.output_tokens` is a small
/// in-progress placeholder, not a real count — `message_delta.usage` later
/// carries the authoritative final `output_tokens`. Emitting both as
/// additive [`UsageContent`] (as `ChatResponse::absorb_update` does when
/// aggregating a stream) would double-count output tokens, so this
/// deliberately omits `output_tokens` here.
pub(crate) fn parse_message_start_usage(usage: &Value) -> Option<UsageContent> {
    let mut details = UsageDetails {
        input_token_count: usage.get("input_tokens").and_then(Value::as_u64),
        ..Default::default()
    };
    if let Some(v) = usage
        .get("cache_creation_input_tokens")
        .and_then(Value::as_u64)
    {
        details
            .additional_counts
            .insert("anthropic.cache_creation_input_tokens".into(), v);
    }
    if let Some(v) = usage.get("cache_read_input_tokens").and_then(Value::as_u64) {
        details
            .additional_counts
            .insert("anthropic.cache_read_input_tokens".into(), v);
    }
    if details.input_token_count.is_none() && details.additional_counts.is_empty() {
        return None;
    }
    Some(UsageContent { details })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::ApprovalMode;

    fn user(text: &str) -> ChatMessage {
        ChatMessage::user(text)
    }

    // region: request building

    #[test]
    fn build_request_simple_text() {
        let body = build_request(
            &[user("Hello there")],
            &ChatOptions::new(),
            "claude-x",
            4096,
            false,
        );
        assert_eq!(
            body,
            json!({
                "model": "claude-x",
                "max_tokens": 4096,
                "messages": [
                    { "role": "user", "content": [{ "type": "text", "text": "Hello there" }] }
                ],
            })
        );
    }

    #[test]
    fn build_request_extracts_leading_system_message() {
        let messages = vec![ChatMessage::system("Be terse."), user("Hi")];
        let body = build_request(&messages, &ChatOptions::new(), "claude-x", 4096, false);
        assert_eq!(body["system"], json!("Be terse."));
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": [{ "type": "text", "text": "Hi" }] }])
        );
    }

    #[test]
    fn build_request_combines_options_instructions_and_system_message() {
        let messages = vec![ChatMessage::system("Also be nice."), user("Hi")];
        let options = ChatOptions::new().with_instructions("Be terse.");
        let body = build_request(&messages, &options, "claude-x", 4096, false);
        assert_eq!(body["system"], json!("Be terse.\n\nAlso be nice."));
    }

    #[test]
    fn build_request_tool_role_message_becomes_user_tool_result() {
        let tool_msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "call_1",
                Some(json!("18C and sunny")),
            ))],
        );
        let body = build_request(&[tool_msg], &ChatOptions::new(), "claude-x", 4096, false);
        assert_eq!(
            body["messages"],
            json!([{
                "role": "user",
                "content": [{ "type": "tool_result", "tool_use_id": "call_1", "content": "18C and sunny" }]
            }])
        );
    }

    #[test]
    fn build_request_tool_result_error_sets_is_error() {
        let mut result = FunctionResultContent::new("call_1", None);
        result.exception = Some("boom".into());
        let tool_msg =
            ChatMessage::with_contents(Role::tool(), vec![Content::FunctionResult(result)]);
        let body = build_request(&[tool_msg], &ChatOptions::new(), "claude-x", 4096, false);
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({ "type": "tool_result", "tool_use_id": "call_1", "content": "boom", "is_error": true })
        );
    }

    #[test]
    fn build_request_assistant_function_call() {
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Object(HashMap::from([(
                "city".to_string(),
                json!("Paris"),
            )]))),
        );
        let assistant_msg =
            ChatMessage::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]);
        let body = build_request(
            &[assistant_msg],
            &ChatOptions::new(),
            "claude-x",
            4096,
            false,
        );
        assert_eq!(
            body["messages"],
            json!([
                {
                    "role": "user",
                    "content": [{ "type": "text", "text": "(continuing the conversation)" }]
                },
                {
                    "role": "assistant",
                    "content": [{ "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } }]
                }
            ])
        );
    }

    #[test]
    fn build_request_data_content_image_uses_embedded_base64() {
        let dc = DataContent::from_bytes(b"hello", "image/png");
        let msg = ChatMessage::with_contents(Role::user(), vec![Content::Data(dc.clone())]);
        let body = build_request(&[msg], &ChatOptions::new(), "claude-x", 4096, false);
        let (_, expected_data) = split_data_uri(&dc.uri).unwrap();
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({ "type": "image", "source": { "type": "base64", "media_type": "image/png", "data": expected_data } })
        );
    }

    #[test]
    fn build_request_uri_content_image_uses_url_source() {
        let uc = UriContent {
            uri: "https://example.com/cat.png".into(),
            media_type: "image/png".into(),
        };
        let msg = ChatMessage::with_contents(Role::user(), vec![Content::Uri(uc)]);
        let body = build_request(&[msg], &ChatOptions::new(), "claude-x", 4096, false);
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({ "type": "image", "source": { "type": "url", "url": "https://example.com/cat.png" } })
        );
    }

    #[test]
    fn build_request_tools_and_tool_choice() {
        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: "Get the weather".into(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        };
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::Required(Some("get_weather".into())));
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        assert_eq!(
            body["tools"],
            json!([{ "type": "custom", "name": "get_weather", "description": "Get the weather", "input_schema": { "type": "object", "properties": {} } }])
        );
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "tool", "name": "get_weather" })
        );
    }

    #[test]
    fn build_request_tool_choice_auto_with_disabled_parallel() {
        let mut options = ChatOptions::new().with_tool_choice(ToolMode::Auto);
        options.allow_multiple_tool_calls = Some(false);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "auto", "disable_parallel_tool_use": true })
        );
    }

    #[test]
    fn build_request_temperature_top_p_stop_sequences() {
        let mut options = ChatOptions::new().with_temperature(0.5);
        options.top_p = Some(0.9);
        options.stop = Some(vec!["STOP".into()]);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        // `temperature`/`top_p` are `f32` on `ChatOptions`; compare against
        // `f32` literals too so the widened-to-f64 JSON values match exactly
        // (0.9_f32 as f64 != 0.9_f64).
        assert_eq!(body["temperature"], json!(0.5_f32));
        assert_eq!(body["top_p"], json!(0.9_f32));
        assert_eq!(body["stop_sequences"], json!(["STOP"]));
    }

    #[test]
    fn build_request_stream_flag() {
        let body = build_request(&[user("hi")], &ChatOptions::new(), "claude-x", 4096, true);
        assert_eq!(body["stream"], json!(true));
    }

    #[test]
    fn build_request_uses_given_max_tokens() {
        let body = build_request(&[user("hi")], &ChatOptions::new(), "claude-x", 2048, false);
        assert_eq!(body["max_tokens"], json!(2048));
    }

    // endregion

    // region: response_format (structured output)
    //
    // Anthropic's Messages API has no native `response_format` field (see
    // `append_response_format_instructions`'s doc comment for the upstream
    // Python/.NET investigation), so these assert the pragmatic fallback:
    // the request body's `system` string, rather than a silent no-op.

    #[test]
    fn build_request_response_format_none_leaves_system_untouched() {
        let body = build_request(&[user("hi")], &ChatOptions::new(), "claude-x", 4096, false);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn build_request_response_format_text_is_a_noop() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::Text);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        assert!(body.get("system").is_none());
    }

    #[test]
    fn build_request_response_format_json_object_appends_system_instruction() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::JsonObject);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        let system = body["system"].as_str().expect("system must be a string");
        assert!(
            system.to_lowercase().contains("json"),
            "expected a JSON instruction, got: {system}"
        );
    }

    #[test]
    fn build_request_response_format_json_schema_embeds_schema_in_system() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::json_schema(
            "Person",
            json!({ "type": "object", "properties": { "name": { "type": "string" } } }),
        ));
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        let system = body["system"].as_str().expect("system must be a string");
        assert!(system.contains("Person"), "system: {system}");
        assert!(system.contains("\"name\""), "system: {system}");
        assert!(system.contains("\"type\": \"object\""), "system: {system}");
    }

    #[test]
    fn build_request_response_format_json_schema_appends_after_existing_system() {
        // A leading system message and `response_format` must combine, not
        // clobber one another.
        let messages = vec![ChatMessage::system("Be terse."), user("Hi")];
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::JsonObject);
        let body = build_request(&messages, &options, "claude-x", 4096, false);
        let system = body["system"].as_str().expect("system must be a string");
        assert!(
            system.starts_with("Be terse."),
            "existing system text must be preserved first: {system}"
        );
        assert!(system.to_lowercase().contains("json"), "system: {system}");
    }

    // endregion

    // region: response parsing

    #[test]
    fn parse_response_text_and_usage() {
        let value = json!({
            "id": "msg_123",
            "model": "claude-x",
            "stop_reason": "end_turn",
            "content": [{ "type": "text", "text": "Hello!" }],
            "usage": { "input_tokens": 10, "output_tokens": 5 },
        });
        let resp = parse_response(&value);
        assert_eq!(resp.response_id.as_deref(), Some("msg_123"));
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(10));
        assert_eq!(usage.output_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(15));
    }

    #[test]
    fn parse_response_tool_use() {
        let value = json!({
            "id": "msg_123",
            "stop_reason": "tool_use",
            "content": [
                { "type": "text", "text": "Let me check." },
                { "type": "tool_use", "id": "call_1", "name": "get_weather", "input": { "city": "Paris" } },
            ],
        });
        let resp = parse_response(&value);
        assert_eq!(resp.finish_reason, Some(FinishReason::tool_calls()));
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &json!("Paris")
        );
    }

    #[test]
    fn parse_response_cache_usage_fields() {
        let value = json!({
            "id": "msg_123",
            "content": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 10,
                "cache_creation_input_tokens": 50,
                "cache_read_input_tokens": 20,
            },
        });
        let resp = parse_response(&value);
        let usage = resp.usage_details.unwrap();
        assert_eq!(
            usage
                .additional_counts
                .get("anthropic.cache_creation_input_tokens"),
            Some(&50)
        );
        assert_eq!(
            usage
                .additional_counts
                .get("anthropic.cache_read_input_tokens"),
            Some(&20)
        );
    }

    #[test]
    fn map_stop_reason_covers_documented_mapping() {
        assert_eq!(map_stop_reason("end_turn"), FinishReason::stop());
        assert_eq!(map_stop_reason("stop_sequence"), FinishReason::stop());
        assert_eq!(
            map_stop_reason("max_tokens"),
            FinishReason::new(FinishReason::LENGTH)
        );
        assert_eq!(map_stop_reason("tool_use"), FinishReason::tool_calls());
    }

    #[test]
    fn message_start_usage_omits_output_tokens() {
        let usage = json!({ "input_tokens": 25, "output_tokens": 1 });
        let content = parse_message_start_usage(&usage).unwrap();
        assert_eq!(content.details.input_token_count, Some(25));
        assert_eq!(content.details.output_token_count, None);
    }

    // endregion
    #[test]
    fn consecutive_same_role_messages_are_merged() {
        let msgs = vec![
            ChatMessage::user("first"),
            ChatMessage::user("second"),
            ChatMessage::assistant("reply"),
            ChatMessage::assistant("more"),
            ChatMessage::user("third"),
        ];
        let out = messages_to_anthropic(&msgs);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"].as_array().unwrap().len(), 2);
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"].as_array().unwrap().len(), 2);
        assert_eq!(out[2]["role"], "user");
    }

    #[test]
    fn leading_assistant_message_gets_synthetic_user_turn() {
        let msgs = vec![
            ChatMessage::assistant("greeting"),
            ChatMessage::user("hello"),
        ];
        let out = messages_to_anthropic(&msgs);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(
            out[0]["content"][0]["text"],
            "(continuing the conversation)"
        );
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[2]["role"], "user");
    }

    // region: beta flags

    #[test]
    fn compute_beta_flags_default_includes_both_upstream_flags() {
        let mut options = ChatOptions::new();
        let flags = compute_beta_flags(&mut options, &[]);
        assert!(flags.contains(&"mcp-client-2025-04-04".to_string()));
        assert!(flags.contains(&"code-execution-2025-08-25".to_string()));
        assert_eq!(flags.len(), 2);
    }

    #[test]
    fn compute_beta_flags_merges_client_level_additional_flags() {
        let mut options = ChatOptions::new();
        let flags = compute_beta_flags(&mut options, &["my-beta-flag".to_string()]);
        assert!(flags.contains(&"my-beta-flag".to_string()));
        assert_eq!(flags.len(), 3);
    }

    #[test]
    fn compute_beta_flags_merges_and_removes_per_request_additional_flags() {
        let mut options = ChatOptions::new();
        options.additional_properties.insert(
            "additional_beta_flags".into(),
            json!(["request-level-flag"]),
        );
        let flags = compute_beta_flags(&mut options, &[]);
        assert!(flags.contains(&"request-level-flag".to_string()));
        // Popped out, like upstream's `.pop(...)` -- must not leak into the
        // body via `additional_properties`.
        assert!(!options
            .additional_properties
            .contains_key("additional_beta_flags"));
    }

    #[test]
    fn compute_beta_flags_deduplicates_overlapping_flags() {
        let mut options = ChatOptions::new();
        options.additional_properties.insert(
            "additional_beta_flags".into(),
            json!(["mcp-client-2025-04-04"]),
        );
        let flags = compute_beta_flags(&mut options, &["mcp-client-2025-04-04".to_string()]);
        assert_eq!(flags.len(), 2);
    }

    #[test]
    fn compute_beta_flags_does_not_leak_into_request_body() {
        let mut options = ChatOptions::new();
        options.additional_properties.insert(
            "additional_beta_flags".into(),
            json!(["request-level-flag"]),
        );
        let _ = compute_beta_flags(&mut options, &[]);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        assert!(body.get("additional_beta_flags").is_none());
    }

    // endregion

    // region: hosted tool mapping (Anthropic wire shape)

    fn make_tool(kind: ToolKind, name: &str, parameters: Value) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: String::new(),
            parameters,
            kind,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        }
    }

    #[test]
    fn tools_to_anthropic_web_search_basic() {
        let tool = make_tool(ToolKind::HostedWebSearch, "web_search", json!({}));
        let (tools, mcp_servers) = tools_to_anthropic(&[tool]);
        assert_eq!(
            tools,
            vec![json!({ "type": "web_search_20250305", "name": "web_search" })]
        );
        assert!(mcp_servers.is_empty());
    }

    #[test]
    fn tools_to_anthropic_web_search_reads_max_uses_and_user_location_from_parameters() {
        let tool = make_tool(
            ToolKind::HostedWebSearch,
            "web_search",
            json!({ "max_uses": 3, "user_location": { "type": "approximate", "city": "Seattle" } }),
        );
        let (tools, _) = tools_to_anthropic(&[tool]);
        assert_eq!(
            tools[0],
            json!({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
                "user_location": { "type": "approximate", "city": "Seattle" },
            })
        );
    }

    #[test]
    fn tools_to_anthropic_code_interpreter() {
        let tool = make_tool(
            ToolKind::HostedCodeInterpreter,
            "code_interpreter",
            json!({}),
        );
        let (tools, mcp_servers) = tools_to_anthropic(&[tool]);
        assert_eq!(
            tools,
            vec![json!({ "type": "code_execution_20250825", "name": "code_execution" })]
        );
        assert!(mcp_servers.is_empty());
    }

    #[test]
    fn tools_to_anthropic_mcp_goes_to_mcp_servers_not_tools() {
        let tool = make_tool(
            ToolKind::HostedMcp {
                url: "https://example.com/mcp".into(),
                allowed_tools: None,
            },
            "my-mcp",
            json!({}),
        );
        let (tools, mcp_servers) = tools_to_anthropic(&[tool]);
        assert!(tools.is_empty());
        assert_eq!(
            mcp_servers,
            vec![json!({ "type": "url", "name": "my-mcp", "url": "https://example.com/mcp" })]
        );
    }

    #[test]
    fn tools_to_anthropic_mcp_with_allowed_tools() {
        let tool = make_tool(
            ToolKind::HostedMcp {
                url: "https://example.com/mcp".into(),
                allowed_tools: Some(vec!["a".into(), "b".into()]),
            },
            "my-mcp",
            json!({}),
        );
        let (_, mcp_servers) = tools_to_anthropic(&[tool]);
        assert_eq!(
            mcp_servers[0]["tool_configuration"],
            json!({ "allowed_tools": ["a", "b"] })
        );
    }

    #[test]
    fn tools_to_anthropic_mcp_empty_allowed_tools_is_omitted() {
        let tool = make_tool(
            ToolKind::HostedMcp {
                url: "https://example.com/mcp".into(),
                allowed_tools: Some(vec![]),
            },
            "my-mcp",
            json!({}),
        );
        let (_, mcp_servers) = tools_to_anthropic(&[tool]);
        assert!(mcp_servers[0].get("tool_configuration").is_none());
    }

    #[test]
    fn tools_to_anthropic_mcp_authorization_header_becomes_authorization_token() {
        let tool = make_tool(
            ToolKind::HostedMcp {
                url: "https://example.com/mcp".into(),
                allowed_tools: None,
            },
            "my-mcp",
            json!({ "headers": { "authorization": "Bearer token123" } }),
        );
        let (_, mcp_servers) = tools_to_anthropic(&[tool]);
        assert_eq!(
            mcp_servers[0]["authorization_token"],
            json!("Bearer token123")
        );
    }

    #[test]
    fn tools_to_anthropic_function_tool_has_custom_type() {
        let tool = make_tool(
            ToolKind::Function,
            "get_weather",
            json!({ "type": "object", "properties": {} }),
        );
        let (tools, _) = tools_to_anthropic(&[tool]);
        assert_eq!(tools[0]["type"], json!("custom"));
    }

    #[test]
    fn tools_to_anthropic_unknown_hosted_kind_is_skipped() {
        let tool = make_tool(
            ToolKind::HostedFileSearch { max_results: None },
            "file_search",
            json!({}),
        );
        let (tools, mcp_servers) = tools_to_anthropic(&[tool]);
        assert!(tools.is_empty());
        assert!(mcp_servers.is_empty());
    }

    #[test]
    fn tools_to_anthropic_mixed_tools_and_mcp_servers_both_populate_body() {
        let function_tool = make_tool(ToolKind::Function, "get_weather", json!({}));
        let mcp_tool = make_tool(
            ToolKind::HostedMcp {
                url: "https://example.com/mcp".into(),
                allowed_tools: None,
            },
            "my-mcp",
            json!({}),
        );
        let options = ChatOptions::new()
            .with_tool(function_tool)
            .with_tool(mcp_tool);
        let body = build_request(&[user("hi")], &options, "claude-x", 4096, false);
        assert_eq!(body["tools"].as_array().unwrap().len(), 1);
        assert_eq!(body["mcp_servers"].as_array().unwrap().len(), 1);
    }

    // endregion

    // region: hosted result block parsing

    #[test]
    fn parse_content_blocks_server_tool_use_is_function_call() {
        let blocks = vec![json!({
            "type": "server_tool_use",
            "id": "srvtoolu_1",
            "name": "web_search",
            "input": { "query": "rust" }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 1);
        match &contents[0] {
            Content::FunctionCall(fc) => {
                assert_eq!(fc.call_id, "srvtoolu_1");
                assert_eq!(fc.name, "web_search");
                assert_eq!(
                    fc.parse_arguments().unwrap().get("query").unwrap(),
                    &json!("rust")
                );
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_blocks_mcp_tool_use_is_function_call() {
        let blocks = vec![json!({
            "type": "mcp_tool_use",
            "id": "mcptoolu_1",
            "name": "search_docs",
            "input": {}
        })];
        let contents = parse_content_blocks(&blocks);
        assert!(matches!(
            &contents[0],
            Content::FunctionCall(fc) if fc.call_id == "mcptoolu_1" && fc.name == "search_docs"
        ));
    }

    #[test]
    fn parse_content_blocks_mcp_tool_result_with_list_content_is_recursively_parsed() {
        let blocks = vec![json!({
            "type": "mcp_tool_result",
            "tool_use_id": "mcptoolu_1",
            "is_error": false,
            "content": [{ "type": "text", "text": "result text" }]
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 1);
        match &contents[0] {
            Content::FunctionResult(fr) => {
                assert_eq!(fr.call_id, "mcptoolu_1");
                assert_eq!(fr.exception, None);
                // The nested `content` array is itself parsed via
                // `parse_content_blocks` and serialized back to JSON.
                assert_eq!(
                    fr.result,
                    Some(json!([{ "type": "text", "text": "result text" }]))
                );
            }
            other => panic!("expected FunctionResult, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_blocks_mcp_tool_result_with_string_content_passes_through() {
        let blocks = vec![json!({
            "type": "mcp_tool_result",
            "tool_use_id": "mcptoolu_1",
            "content": "plain string result"
        })];
        let contents = parse_content_blocks(&blocks);
        match &contents[0] {
            Content::FunctionResult(fr) => {
                assert_eq!(fr.result, Some(json!("plain string result")));
            }
            other => panic!("expected FunctionResult, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_blocks_web_search_tool_result_is_not_recursively_parsed() {
        let blocks = vec![json!({
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_1",
            "content": [{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }]
        })];
        let contents = parse_content_blocks(&blocks);
        match &contents[0] {
            Content::FunctionResult(fr) => {
                assert_eq!(fr.call_id, "srvtoolu_1");
                // Raw content passed through as-is, unlike `mcp_tool_result`.
                assert_eq!(
                    fr.result,
                    Some(
                        json!([{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }])
                    )
                );
            }
            other => panic!("expected FunctionResult, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_blocks_web_fetch_tool_result_uses_same_mapping() {
        let blocks = vec![json!({
            "type": "web_fetch_tool_result",
            "tool_use_id": "srvtoolu_2",
            "content": { "type": "web_fetch_result", "url": "https://example.com" }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 1);
        assert!(matches!(&contents[0], Content::FunctionResult(fr) if fr.call_id == "srvtoolu_2"));
    }

    #[test]
    fn parse_content_blocks_code_execution_tool_result_extracts_hosted_files_before_result() {
        let blocks = vec![json!({
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_3",
            "content": {
                "type": "code_execution_result",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "content": [
                    { "type": "code_execution_output", "file_id": "file_abc" },
                    { "type": "code_execution_output", "file_id": "file_def" }
                ]
            }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 3);
        assert_eq!(
            contents[0],
            Content::HostedFile(HostedFileContent {
                file_id: "file_abc".into()
            })
        );
        assert_eq!(
            contents[1],
            Content::HostedFile(HostedFileContent {
                file_id: "file_def".into()
            })
        );
        match &contents[2] {
            Content::FunctionResult(fr) => assert_eq!(fr.call_id, "srvtoolu_3"),
            other => panic!("expected FunctionResult, got {other:?}"),
        }
    }

    #[test]
    fn parse_content_blocks_bash_code_execution_tool_result_extracts_hosted_files() {
        let blocks = vec![json!({
            "type": "bash_code_execution_tool_result",
            "tool_use_id": "srvtoolu_4",
            "content": {
                "type": "bash_code_execution_result",
                "stdout": "",
                "stderr": "",
                "return_code": 0,
                "content": [{ "type": "bash_code_execution_output", "file_id": "file_ghi" }]
            }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 2);
        assert_eq!(
            contents[0],
            Content::HostedFile(HostedFileContent {
                file_id: "file_ghi".into()
            })
        );
    }

    #[test]
    fn parse_content_blocks_code_execution_tool_result_no_files_only_function_result() {
        let blocks = vec![json!({
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_5",
            "content": { "type": "code_execution_result", "stdout": "hi", "stderr": "", "return_code": 0, "content": [] }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 1);
        assert!(matches!(&contents[0], Content::FunctionResult(_)));
    }

    #[test]
    fn parse_content_blocks_text_editor_code_execution_tool_result_never_extracts_files() {
        // `text_editor_code_execution_tool_result`'s nested content type is
        // never `code_execution_result`/`bash_code_execution_result`
        // (verified against the `anthropic` SDK's
        // `BetaTextEditorCodeExecutionToolResultBlock`), so this only ever
        // produces the trailing FunctionResult.
        let blocks = vec![json!({
            "type": "text_editor_code_execution_tool_result",
            "tool_use_id": "srvtoolu_6",
            "content": { "type": "text_editor_code_execution_view_result", "file_type": "text", "content": "print('hi')" }
        })];
        let contents = parse_content_blocks(&blocks);
        assert_eq!(contents.len(), 1);
        assert!(matches!(&contents[0], Content::FunctionResult(_)));
    }

    #[test]
    fn parse_content_blocks_unknown_block_type_is_skipped() {
        let blocks = vec![json!({ "type": "totally_unknown_block" })];
        let contents = parse_content_blocks(&blocks);
        assert!(contents.is_empty());
    }

    #[test]
    fn parse_response_includes_server_tool_use_and_web_search_result() {
        let value = json!({
            "id": "msg_1",
            "content": [
                { "type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": { "query": "rust" } },
                { "type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [{ "type": "web_search_result", "url": "https://example.com", "title": "Example" }] },
            ],
        });
        let resp = parse_response(&value);
        let contents = &resp.messages[0].contents;
        assert_eq!(contents.len(), 2);
        assert!(matches!(&contents[0], Content::FunctionCall(_)));
        assert!(matches!(&contents[1], Content::FunctionResult(_)));
    }

    // endregion

    // region: citations

    #[test]
    fn parse_citations_char_location() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "char_location",
                "cited_text": "The grass is green.",
                "document_index": 0,
                "document_title": "Example Document",
                "start_char_index": 0,
                "end_char_index": 20,
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        assert_eq!(annotations.len(), 1);
        let cit = &annotations[0];
        // Mirrors upstream's `char_location` branch, which reads
        // `citation.title` rather than `citation.document_title` -- absent
        // on the real wire payload, so `title` ends up `None` here too. See
        // `parse_citations`'s doc comment for the upstream-bug analysis.
        assert_eq!(cit.title, None);
        assert_eq!(cit.snippet.as_deref(), Some("The grass is green."));
        assert_eq!(
            cit.annotated_regions,
            Some(vec![TextSpanRegion {
                start_index: Some(0),
                end_index: Some(20)
            }])
        );
    }

    #[test]
    fn parse_citations_page_location_uses_document_title() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "page_location",
                "cited_text": "Water is essential for life.",
                "document_index": 1,
                "document_title": "PDF Document",
                "start_page_number": 5,
                "end_page_number": 6,
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        let cit = &annotations[0];
        assert_eq!(cit.title.as_deref(), Some("PDF Document"));
        assert_eq!(cit.snippet.as_deref(), Some("Water is essential for life."));
        assert_eq!(
            cit.annotated_regions,
            Some(vec![TextSpanRegion {
                start_index: Some(5),
                end_index: Some(6)
            }])
        );
    }

    #[test]
    fn parse_citations_content_block_location_uses_document_title() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "content_block_location",
                "cited_text": "These are important findings.",
                "document_index": 2,
                "document_title": "Custom Content Document",
                "start_block_index": 0,
                "end_block_index": 1,
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        let cit = &annotations[0];
        assert_eq!(cit.title.as_deref(), Some("Custom Content Document"));
        assert_eq!(
            cit.annotated_regions,
            Some(vec![TextSpanRegion {
                start_index: Some(0),
                end_index: Some(1)
            }])
        );
    }

    #[test]
    fn parse_citations_file_id_only_set_when_present() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "page_location",
                "cited_text": "text",
                "document_index": 0,
                "document_title": "Doc",
                "start_page_number": 1,
                "end_page_number": 2,
                "file_id": "file_123",
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        assert_eq!(annotations[0].file_id.as_deref(), Some("file_123"));
    }

    #[test]
    fn parse_citations_web_search_result_location() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "web_search_result_location",
                "cited_text": "some cited snippet",
                "url": "https://example.com/page",
                "title": "Example Page",
                "encrypted_index": "abc123",
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        let cit = &annotations[0];
        assert_eq!(cit.title.as_deref(), Some("Example Page"));
        assert_eq!(cit.snippet.as_deref(), Some("some cited snippet"));
        assert_eq!(cit.url.as_deref(), Some("https://example.com/page"));
        assert_eq!(cit.annotated_regions, None);
    }

    #[test]
    fn parse_citations_search_result_location_uses_source_as_url() {
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{
                "type": "search_result_location",
                "cited_text": "some cited snippet",
                "source": "https://example.com/doc",
                "title": "Search Result",
                "search_result_index": 0,
                "start_block_index": 0,
                "end_block_index": 1,
            }]
        });
        let annotations = parse_citations(&block).unwrap();
        let cit = &annotations[0];
        assert_eq!(cit.title.as_deref(), Some("Search Result"));
        assert_eq!(cit.url.as_deref(), Some("https://example.com/doc"));
        assert_eq!(
            cit.annotated_regions,
            Some(vec![TextSpanRegion {
                start_index: Some(0),
                end_index: Some(1)
            }])
        );
    }

    #[test]
    fn parse_citations_unknown_type_still_produces_empty_annotation() {
        // Mirrors upstream: `annotations.append(cit)` runs unconditionally
        // after the match, even for an unrecognized `citation.type`
        // (~667-669).
        let block = json!({
            "type": "text",
            "text": "cited",
            "citations": [{ "type": "some_future_citation_type" }]
        });
        let annotations = parse_citations(&block).unwrap();
        assert_eq!(annotations.len(), 1);
        assert_eq!(annotations[0], CitationAnnotation::default());
    }

    #[test]
    fn parse_citations_absent_returns_none() {
        let block = json!({ "type": "text", "text": "no citations here" });
        assert_eq!(parse_citations(&block), None);
    }

    #[test]
    fn parse_citations_empty_array_returns_none() {
        let block = json!({ "type": "text", "text": "no citations here", "citations": [] });
        assert_eq!(parse_citations(&block), None);
    }

    #[test]
    fn parse_response_text_block_carries_citations_as_annotations() {
        let value = json!({
            "id": "msg_1",
            "content": [{
                "type": "text",
                "text": "the grass is green",
                "citations": [{
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "Example Document",
                    "start_char_index": 0,
                    "end_char_index": 20,
                }]
            }]
        });
        let resp = parse_response(&value);
        match &resp.messages[0].contents[0] {
            Content::Text(t) => {
                assert_eq!(t.text, "the grass is green");
                assert!(t.annotations.is_some());
            }
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // endregion
}
