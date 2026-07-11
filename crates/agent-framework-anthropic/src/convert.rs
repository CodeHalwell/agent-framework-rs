//! Conversion between framework types and the Anthropic Messages API wire
//! format.

use std::collections::HashMap;

use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, Content, DataContent, FinishReason, FunctionArguments,
    FunctionCallContent, FunctionResultContent, ResponseFormat, Role, TextContent,
    TextReasoningContent, ToolMode, UriContent, UsageContent, UsageDetails,
};
use serde_json::{json, Map, Value};

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
        let tools = tools_to_anthropic(&options.tools);
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
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

/// Convert executable/spec function tools into Anthropic's flat tool spec.
/// Hosted tool markers (web search, code interpreter, MCP, ...) aren't in
/// scope here and are skipped.
pub fn tools_to_anthropic(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .filter(|t| t.kind == ToolKind::Function)
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        })
        .collect()
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
        .map(|blocks| blocks.iter().filter_map(parse_content_block).collect())
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

/// Parse a single Anthropic content block (from a full response's `content`
/// array, or a streaming `content_block_start`'s `content_block`).
pub(crate) fn parse_content_block(block: &Value) -> Option<Content> {
    match block.get("type").and_then(Value::as_str)? {
        "text" => Some(Content::Text(TextContent::new(
            block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ))),
        "tool_use" => {
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
            Some(Content::FunctionCall(FunctionCallContent::new(
                id,
                name,
                Some(FunctionArguments::Object(input)),
            )))
        }
        "thinking" => Some(Content::TextReasoning(TextReasoningContent {
            text: block
                .get("thinking")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            annotations: None,
        })),
        _ => None,
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
            json!([{ "name": "get_weather", "description": "Get the weather", "input_schema": { "type": "object", "properties": {} } }])
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
}
