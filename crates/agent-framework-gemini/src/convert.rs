//! Conversion between framework types and the Google Gemini
//! `generateContent` wire format.
//!
//! Reference: <https://ai.google.dev/api/generate-content>. Unlike the
//! OpenAI-shaped Chat Completions wire format, Gemini's request is
//! `{contents:[{role,parts:[...]}], generationConfig:{...},
//! systemInstruction:{parts:[...]}}` and its response is
//! `{candidates:[{content:{parts:[...]},finishReason}],usageMetadata:{...}}`.

use std::collections::HashMap;

use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatOptions, Content, DataContent, FinishReason, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, ResponseFormat, Role, TextContent, TextReasoningContent,
    ToolMode, UriContent, UsageDetails,
};
use serde_json::{json, Map, Value};

/// Build a full Gemini `generateContent` / `streamGenerateContent` request
/// body (everything except the `model` path segment, which the caller embeds
/// in the URL, and the `stream`/`alt=sse` selection, which is a query
/// parameter rather than a body field for this API).
pub fn build_request(messages: &[Message], options: &ChatOptions) -> Value {
    let mut body = Map::new();

    if let Some(system) = build_system_instruction(messages, options.instructions.as_deref()) {
        body.insert("systemInstruction".into(), system);
    }
    body.insert("contents".into(), json!(messages_to_gemini(messages)));

    if let Some(cfg) = build_generation_config(options) {
        body.insert("generationConfig".into(), cfg);
    }

    if !options.tools.is_empty() {
        let tools = tools_to_gemini(&options.tools);
        if !tools.is_empty() {
            body.insert("tools".into(), json!(tools));
        }
    }
    if let Some(mode) = &options.tool_choice {
        body.insert("toolConfig".into(), tool_config_to_gemini(mode));
    }

    for (k, v) in &options.additional_properties {
        body.entry(k.clone()).or_insert_with(|| v.clone());
    }

    Value::Object(body)
}

/// Build the top-level `systemInstruction` field from every `system`-role
/// message plus `ChatOptions::instructions`, joined with blank lines.
/// Unlike a turn-taking role, Gemini's system instruction is a single
/// out-of-band field, so every system message contributes regardless of its
/// position in the conversation (not just a leading one).
fn build_system_instruction(
    messages: &[Message],
    options_instructions: Option<&str>,
) -> Option<Value> {
    let mut parts = Vec::new();
    if let Some(instr) = options_instructions {
        if !instr.is_empty() {
            parts.push(instr.to_string());
        }
    }
    for msg in messages {
        if msg.role == Role::system() {
            let text = msg.text();
            if !text.is_empty() {
                parts.push(text);
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(json!({ "parts": [{ "text": parts.join("\n\n") }] }))
    }
}

/// Convert framework messages into Gemini's `contents` array.
///
/// Role mapping: `assistant` -> `model`, `tool` -> `function` (Gemini's
/// dedicated role for `functionResponse` parts), everything else (`user` and
/// any custom role) -> `user`. `system`-role messages are excluded here; they
/// feed [`build_system_instruction`] instead.
pub fn messages_to_gemini(messages: &[Message]) -> Vec<Value> {
    let call_names = collect_call_names(messages);
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        if msg.role == Role::system() {
            continue;
        }
        let role = if msg.role == Role::assistant() {
            "model"
        } else if msg.role == Role::tool() {
            "function"
        } else {
            "user"
        };
        let parts: Vec<Value> = msg
            .contents
            .iter()
            .filter_map(|c| content_to_part(c, &call_names))
            .collect();
        if parts.is_empty() {
            // Gemini rejects a content entry with an empty `parts` array.
            continue;
        }
        out.push(json!({ "role": role, "parts": parts }));
    }
    out
}

/// Build a `call_id -> function name` map from every [`FunctionCallContent`]
/// in the conversation. Gemini's `functionResponse` part identifies the call
/// it answers by *name*, not by an opaque id (the wire format has no call-id
/// concept at all) — since the framework's [`FunctionResultContent`] only
/// carries `call_id`, this recovers the name from the same conversation
/// history that is resent on every (stateless) request.
fn collect_call_names(messages: &[Message]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    for msg in messages {
        for content in &msg.contents {
            if let Content::FunctionCall(fc) = content {
                if !fc.call_id.is_empty() {
                    map.insert(fc.call_id.clone(), fc.name.clone());
                }
            }
        }
    }
    map
}

fn content_to_part(content: &Content, call_names: &HashMap<String, String>) -> Option<Value> {
    match content {
        Content::Text(t) => Some(json!({ "text": t.text })),
        Content::TextReasoning(t) => Some(json!({ "text": t.text, "thought": true })),
        Content::FunctionCall(fc) => Some(function_call_part(fc)),
        Content::FunctionResult(fr) => Some(function_response_part(fr, call_names)),
        Content::Data(dc) => data_part(dc),
        Content::Uri(uc) => Some(uri_part(uc)),
        _ => None,
    }
}

fn function_call_part(fc: &FunctionCallContent) -> Value {
    let args = fc.parse_arguments().unwrap_or_default();
    json!({
        "functionCall": {
            "name": fc.name,
            "args": Value::Object(args.into_iter().collect()),
        }
    })
}

fn function_response_part(
    fr: &FunctionResultContent,
    call_names: &HashMap<String, String>,
) -> Value {
    let name = call_names
        .get(&fr.call_id)
        .cloned()
        .unwrap_or_else(|| fr.call_id.clone());
    json!({
        "functionResponse": {
            "name": name,
            "response": function_response_value(fr),
        }
    })
}

/// Gemini requires `functionResponse.response` to be a JSON object. A tool
/// error is wrapped as `{"error": ...}`; a non-object success result (a bare
/// string/number/array, or no result at all) is wrapped as `{"result": ...}`
/// / `{}` so the field is always an object.
fn function_response_value(fr: &FunctionResultContent) -> Value {
    if let Some(exc) = &fr.exception {
        return json!({ "error": exc });
    }
    match &fr.result {
        Some(Value::Object(m)) => Value::Object(m.clone()),
        Some(v) => json!({ "result": v }),
        None => json!({}),
    }
}

/// Build an `{"inlineData":{"mimeType":...,"data":...}}` part from a `data:`
/// URI, without needing a base64 encoder: [`DataContent::uri`] is already
/// base64 text after the `base64,` marker (per `DataContent::from_bytes` in
/// `agent-framework-core`), so it is just sliced out.
fn data_part(dc: &DataContent) -> Option<Value> {
    let (parsed_media_type, data) = split_data_uri(&dc.uri)?;
    let media_type = dc.media_type.clone().unwrap_or(parsed_media_type);
    Some(json!({ "inlineData": { "mimeType": media_type, "data": data } }))
}

fn uri_part(uc: &UriContent) -> Value {
    json!({ "fileData": { "mimeType": uc.media_type, "fileUri": uc.uri } })
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

/// Build the `generationConfig` object from the request's scalar/structured
/// options. Returns `None` when no such option is set, so it is omitted
/// entirely rather than sent as `{}`.
fn build_generation_config(options: &ChatOptions) -> Option<Value> {
    let mut cfg = Map::new();
    if let Some(t) = options.temperature {
        cfg.insert("temperature".into(), json!(t));
    }
    if let Some(mt) = options.max_tokens {
        cfg.insert("maxOutputTokens".into(), json!(mt));
    }
    if let Some(tp) = options.top_p {
        cfg.insert("topP".into(), json!(tp));
    }
    if let Some(stop) = &options.stop {
        cfg.insert("stopSequences".into(), json!(stop));
    }
    match &options.response_format {
        None | Some(ResponseFormat::Text) => {}
        Some(ResponseFormat::JsonObject) => {
            cfg.insert("responseMimeType".into(), json!("application/json"));
        }
        Some(ResponseFormat::JsonSchema { schema, .. }) => {
            cfg.insert("responseMimeType".into(), json!("application/json"));
            cfg.insert("responseSchema".into(), schema.clone());
        }
    }
    if cfg.is_empty() {
        None
    } else {
        Some(Value::Object(cfg))
    }
}

/// Map a [`ToolMode`] to Gemini's `toolConfig.functionCallingConfig`.
fn tool_config_to_gemini(mode: &ToolMode) -> Value {
    let mut fcc = Map::new();
    match mode {
        ToolMode::Auto => {
            fcc.insert("mode".into(), json!("AUTO"));
        }
        ToolMode::Required(Some(name)) => {
            fcc.insert("mode".into(), json!("ANY"));
            fcc.insert("allowedFunctionNames".into(), json!([name]));
        }
        ToolMode::Required(None) => {
            fcc.insert("mode".into(), json!("ANY"));
        }
        ToolMode::None => {
            fcc.insert("mode".into(), json!("NONE"));
        }
    }
    json!({ "functionCallingConfig": Value::Object(fcc) })
}

/// Convert tool definitions into Gemini's `tools` array.
///
/// * [`ToolKind::Function`] entries are collected into a single
///   `{"functionDeclarations":[...]}` tool entry (Gemini allows at most one
///   `functionDeclarations` list per request, unlike Anthropic/OpenAI's
///   flat per-tool entries).
/// * [`ToolKind::HostedWebSearch`] -> a `{"googleSearch":{}}` tool entry.
/// * [`ToolKind::HostedCodeInterpreter`] -> a `{"codeExecution":{}}` tool
///   entry.
/// * [`ToolKind::HostedFileSearch`], [`ToolKind::HostedMcp`], and
///   [`ToolKind::HostedImageGeneration`] have no Gemini `generateContent`
///   tool equivalent and are skipped with a `tracing::warn!`.
pub fn tools_to_gemini(tools: &[ToolDefinition]) -> Vec<Value> {
    let mut declarations = Vec::new();
    let mut extra_tools = Vec::new();
    for t in tools {
        match &t.kind {
            ToolKind::Function => {
                declarations.push(json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }));
            }
            ToolKind::HostedWebSearch => {
                extra_tools.push(json!({ "googleSearch": {} }));
            }
            ToolKind::HostedCodeInterpreter => {
                extra_tools.push(json!({ "codeExecution": {} }));
            }
            ToolKind::HostedFileSearch { .. } => {
                tracing::warn!(
                    tool = %t.name,
                    "Gemini: hosted file-search tools are not supported by the generateContent API; skipping"
                );
            }
            ToolKind::HostedMcp { .. } => {
                tracing::warn!(
                    tool = %t.name,
                    "Gemini: hosted MCP tools are not supported by the generateContent API; skipping"
                );
            }
            ToolKind::HostedImageGeneration => {
                tracing::warn!(
                    tool = %t.name,
                    "Gemini: hosted image-generation tools are not supported by the generateContent API; skipping"
                );
            }
        }
    }
    let mut out = Vec::new();
    if !declarations.is_empty() {
        out.push(json!({ "functionDeclarations": declarations }));
    }
    out.extend(extra_tools);
    out
}

/// Parse a full (non-streaming) Gemini `GenerateContentResponse`.
pub fn parse_response(value: &Value) -> agent_framework_core::types::ChatResponse {
    use agent_framework_core::types::ChatResponse;

    let mut response = ChatResponse {
        response_id: value
            .get("responseId")
            .and_then(Value::as_str)
            .map(String::from),
        model: value
            .get("modelVersion")
            .and_then(Value::as_str)
            .map(String::from),
        ..Default::default()
    };

    if let Some(candidate) = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    {
        let contents = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
            .map(|parts| parse_parts(parts))
            .unwrap_or_default();
        let raw_finish_reason = candidate.get("finishReason").and_then(Value::as_str);
        response.finish_reason = finalize_finish_reason(raw_finish_reason, &contents);
        let mut message = Message::with_contents(Role::assistant(), contents);
        message.message_id = response.response_id.clone();
        response.messages.push(message);
    } else if let Some(block_reason) = value
        .get("promptFeedback")
        .and_then(|pf| pf.get("blockReason"))
        .and_then(Value::as_str)
    {
        // The prompt itself was blocked: Gemini returns a 200 with no
        // `candidates` at all, only `promptFeedback.blockReason`. Surface
        // this as a content-filter finish reason on an empty assistant
        // message rather than an error, mirroring how a `stop_reason:
        // "refusal"` 200 is handled for Anthropic.
        tracing::debug!(
            block_reason,
            "Gemini: prompt blocked, no candidates returned"
        );
        response.finish_reason = Some(FinishReason::new(FinishReason::CONTENT_FILTER));
        response
            .messages
            .push(Message::with_contents(Role::assistant(), Vec::new()));
    }

    if let Some(usage) = value.get("usageMetadata") {
        response.usage_details = Some(parse_usage(usage));
    }
    response
}

/// Parse one Gemini `content.parts` array into framework [`Content`] items.
pub(crate) fn parse_parts(parts: &[Value]) -> Vec<Content> {
    let mut out = Vec::with_capacity(parts.len());
    for part in parts {
        if let Some(fc) = part.get("functionCall") {
            let name = fc
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let args = match fc.get("args") {
                Some(Value::Object(m)) => m.clone().into_iter().collect(),
                _ => HashMap::new(),
            };
            // Gemini's wire format carries no call id at all; synthesize one
            // so the framework's call/result correlation has something to
            // key on. `messages_to_gemini` recovers the name from this id
            // via `collect_call_names` when the call is answered.
            let call_id = format!("call_{}", uuid::Uuid::new_v4());
            out.push(Content::FunctionCall(FunctionCallContent::new(
                call_id,
                name,
                Some(FunctionArguments::Object(args)),
            )));
            continue;
        }
        if let Some(fr) = part.get("functionResponse") {
            // Not expected in a model response, but handled defensively
            // (e.g. an echoed turn) rather than silently dropped.
            let name = fr
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let result = fr.get("response").cloned();
            out.push(Content::FunctionResult(FunctionResultContent::new(
                name, result,
            )));
            continue;
        }
        if let Some(text) = part.get("text").and_then(Value::as_str) {
            let is_thought = part
                .get("thought")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if is_thought {
                out.push(Content::TextReasoning(TextReasoningContent {
                    text: text.to_string(),
                    annotations: None,
                    ..Default::default()
                }));
            } else {
                out.push(Content::Text(TextContent::new(text)));
            }
            continue;
        }
        if let Some(inline) = part.get("inlineData") {
            if let (Some(mime), Some(data)) = (
                inline.get("mimeType").and_then(Value::as_str),
                inline.get("data").and_then(Value::as_str),
            ) {
                out.push(Content::Data(DataContent {
                    uri: format!("data:{mime};base64,{data}"),
                    media_type: Some(mime.to_string()),
                }));
            }
            continue;
        }
        tracing::debug!(?part, "Gemini: ignoring unsupported content part");
    }
    out
}

/// Map Gemini's `finishReason` to the shared [`FinishReason`], with a
/// `tool_calls` override.
///
/// Gemini has no dedicated "the model wants to call a function" finish
/// reason the way Anthropic's `stop_reason: "tool_use"` or OpenAI's
/// `finish_reason: "tool_calls"` do — a turn that ends in a function call
/// still reports `STOP` (see the `FinishReason` enum in the Gemini API
/// reference). Reporting `stop` on a response that actually carries
/// unresolved function calls would be misleading to any caller keying off
/// `finish_reason` directly (the function-invocation loop itself only looks
/// at `function_calls()`, so this is purely for caller-facing accuracy), so
/// a raw `STOP`/absent reason is upgraded to `tool_calls` whenever the parsed
/// contents include a `FunctionCall`. Any other raw reason (`MAX_TOKENS`,
/// `SAFETY`, ...) is left as-is even alongside a function call, since those
/// describe a more specific/severe outcome.
fn finalize_finish_reason(raw: Option<&str>, contents: &[Content]) -> Option<FinishReason> {
    let has_call = contents
        .iter()
        .any(|c| matches!(c, Content::FunctionCall(_)));
    match raw {
        Some(r) => {
            let mapped = map_finish_reason(r);
            if has_call && mapped == FinishReason::stop() {
                Some(FinishReason::tool_calls())
            } else {
                Some(mapped)
            }
        }
        None => has_call.then(FinishReason::tool_calls),
    }
}

pub(crate) fn map_finish_reason(reason: &str) -> FinishReason {
    match reason {
        "STOP" => FinishReason::stop(),
        "MAX_TOKENS" => FinishReason::new(FinishReason::LENGTH),
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            FinishReason::new(FinishReason::CONTENT_FILTER)
        }
        other => FinishReason::new(other.to_lowercase()),
    }
}

/// Parse a Gemini `usageMetadata` object into [`UsageDetails`].
pub(crate) fn parse_usage(usage: &Value) -> UsageDetails {
    UsageDetails {
        input_token_count: usage.get("promptTokenCount").and_then(Value::as_u64),
        output_token_count: usage.get("candidatesTokenCount").and_then(Value::as_u64),
        total_token_count: usage.get("totalTokenCount").and_then(Value::as_u64),
        reasoning_output_token_count: usage.get("thoughtsTokenCount").and_then(Value::as_u64),
        ..Default::default()
    }
}

/// Parse a single Gemini SSE `data:` chunk (a full, self-contained
/// `GenerateContentResponse`, not an incremental delta protocol) into a
/// [`agent_framework_core::types::ChatResponseUpdate`]. Returns `None` for a
/// chunk that carries nothing new (defensive; not expected in practice).
pub(crate) fn parse_stream_chunk(
    value: &Value,
) -> Option<agent_framework_core::types::ChatResponseUpdate> {
    use agent_framework_core::types::{ChatResponseUpdate, UsageContent};

    let mut contents = Vec::new();
    let mut finish_reason = None;

    if let Some(candidate) = value
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
    {
        if let Some(parts) = candidate
            .get("content")
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
        {
            contents.extend(parse_parts(parts));
        }
        let raw_finish_reason = candidate.get("finishReason").and_then(Value::as_str);
        finish_reason = finalize_finish_reason(raw_finish_reason, &contents);
    }

    let response_id = value
        .get("responseId")
        .and_then(Value::as_str)
        .map(String::from);
    let model = value
        .get("modelVersion")
        .and_then(Value::as_str)
        .map(String::from);
    let usage = value.get("usageMetadata").map(parse_usage);

    if contents.is_empty()
        && finish_reason.is_none()
        && response_id.is_none()
        && model.is_none()
        && usage.is_none()
    {
        return None;
    }

    if let Some(details) = usage {
        contents.push(Content::Usage(UsageContent { details }));
    }

    Some(ChatResponseUpdate {
        contents,
        role: Some(Role::assistant()),
        response_id,
        model,
        finish_reason,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::ApprovalMode;

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    // region: request building

    #[test]
    fn build_request_simple_text() {
        let body = build_request(&[user("Hello there")], &ChatOptions::new());
        assert_eq!(
            body,
            json!({
                "contents": [
                    { "role": "user", "parts": [{ "text": "Hello there" }] }
                ],
            })
        );
    }

    #[test]
    fn build_request_extracts_system_message() {
        let messages = vec![Message::system("Be terse."), user("Hi")];
        let body = build_request(&messages, &ChatOptions::new());
        assert_eq!(
            body["systemInstruction"],
            json!({ "parts": [{ "text": "Be terse." }] })
        );
        assert_eq!(
            body["contents"],
            json!([{ "role": "user", "parts": [{ "text": "Hi" }] }])
        );
    }

    #[test]
    fn build_request_combines_options_instructions_and_system_message() {
        let messages = vec![Message::system("Also be nice."), user("Hi")];
        let options = ChatOptions::new().with_instructions("Be terse.");
        let body = build_request(&messages, &options);
        assert_eq!(
            body["systemInstruction"]["parts"][0]["text"],
            "Be terse.\n\nAlso be nice."
        );
    }

    #[test]
    fn build_request_assistant_role_maps_to_model() {
        let messages = vec![user("hi"), Message::assistant("hello")];
        let body = build_request(&messages, &ChatOptions::new());
        assert_eq!(body["contents"][0]["role"], "user");
        assert_eq!(body["contents"][1]["role"], "model");
    }

    #[test]
    fn build_request_tool_role_message_becomes_function_response() {
        let assistant_call = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let tool_msg = Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "call_1",
                Some(json!({ "temp": 18 })),
            ))],
        );
        let body = build_request(&[assistant_call, tool_msg], &ChatOptions::new());
        assert_eq!(body["contents"][1]["role"], "function");
        assert_eq!(
            body["contents"][1]["parts"][0],
            json!({ "functionResponse": { "name": "get_weather", "response": { "temp": 18 } } })
        );
    }

    #[test]
    fn build_request_tool_result_error_wraps_in_error_object() {
        let assistant_call = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let mut result = FunctionResultContent::new("call_1", None);
        result.exception = Some("boom".into());
        let tool_msg = Message::with_contents(Role::tool(), vec![Content::FunctionResult(result)]);
        let body = build_request(&[assistant_call, tool_msg], &ChatOptions::new());
        assert_eq!(
            body["contents"][1]["parts"][0]["functionResponse"]["response"],
            json!({ "error": "boom" })
        );
    }

    #[test]
    fn build_request_tool_result_scalar_wraps_in_result_object() {
        let assistant_call = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let tool_msg = Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "call_1",
                Some(json!("18C and sunny")),
            ))],
        );
        let body = build_request(&[assistant_call, tool_msg], &ChatOptions::new());
        assert_eq!(
            body["contents"][1]["parts"][0]["functionResponse"]["response"],
            json!({ "result": "18C and sunny" })
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
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]);
        let body = build_request(&[assistant_msg], &ChatOptions::new());
        assert_eq!(
            body["contents"],
            json!([{
                "role": "model",
                "parts": [{ "functionCall": { "name": "get_weather", "args": { "city": "Paris" } } }]
            }])
        );
    }

    #[test]
    fn build_request_data_content_image_uses_embedded_base64() {
        let dc = DataContent::from_bytes(b"hello", "image/png");
        let msg = Message::with_contents(Role::user(), vec![Content::Data(dc.clone())]);
        let body = build_request(&[msg], &ChatOptions::new());
        let (_, expected_data) = split_data_uri(&dc.uri).unwrap();
        assert_eq!(
            body["contents"][0]["parts"][0],
            json!({ "inlineData": { "mimeType": "image/png", "data": expected_data } })
        );
    }

    #[test]
    fn build_request_uri_content_uses_file_data() {
        let uc = UriContent {
            uri: "https://example.com/cat.png".into(),
            media_type: "image/png".into(),
        };
        let msg = Message::with_contents(Role::user(), vec![Content::Uri(uc)]);
        let body = build_request(&[msg], &ChatOptions::new());
        assert_eq!(
            body["contents"][0]["parts"][0],
            json!({ "fileData": { "mimeType": "image/png", "fileUri": "https://example.com/cat.png" } })
        );
    }

    #[test]
    fn build_request_generation_config_temperature_max_tokens_top_p_stop() {
        let mut options = ChatOptions::new()
            .with_temperature(0.5)
            .with_max_tokens(256);
        options.top_p = Some(0.9);
        options.stop = Some(vec!["STOP".into()]);
        let body = build_request(&[user("hi")], &options);
        assert_eq!(body["generationConfig"]["temperature"], json!(0.5_f32));
        assert_eq!(body["generationConfig"]["maxOutputTokens"], json!(256));
        assert_eq!(body["generationConfig"]["topP"], json!(0.9_f32));
        assert_eq!(body["generationConfig"]["stopSequences"], json!(["STOP"]));
    }

    #[test]
    fn build_request_no_options_omits_generation_config() {
        let body = build_request(&[user("hi")], &ChatOptions::new());
        assert!(body.get("generationConfig").is_none());
    }

    #[test]
    fn build_request_response_format_json_object() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::JsonObject);
        let body = build_request(&[user("hi")], &options);
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            json!("application/json")
        );
        assert!(body["generationConfig"].get("responseSchema").is_none());
    }

    #[test]
    fn build_request_response_format_json_schema_embeds_schema() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::json_schema(
            "Person",
            json!({ "type": "object", "properties": { "name": { "type": "string" } } }),
        ));
        let body = build_request(&[user("hi")], &options);
        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            json!("application/json")
        );
        assert_eq!(
            body["generationConfig"]["responseSchema"],
            json!({ "type": "object", "properties": { "name": { "type": "string" } } })
        );
    }

    #[test]
    fn build_request_tool_choice_modes() {
        let cases = [
            (
                ToolMode::Auto,
                json!({ "functionCallingConfig": { "mode": "AUTO" } }),
            ),
            (
                ToolMode::Required(None),
                json!({ "functionCallingConfig": { "mode": "ANY" } }),
            ),
            (
                ToolMode::Required(Some("get_weather".into())),
                json!({ "functionCallingConfig": { "mode": "ANY", "allowedFunctionNames": ["get_weather"] } }),
            ),
            (
                ToolMode::None,
                json!({ "functionCallingConfig": { "mode": "NONE" } }),
            ),
        ];
        for (mode, expected) in cases {
            let options = ChatOptions::new().with_tool_choice(mode);
            let body = build_request(&[user("hi")], &options);
            assert_eq!(body["toolConfig"], expected);
        }
    }

    fn make_tool(kind: ToolKind, name: &str, parameters: Value) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: "a tool".into(),
            parameters,
            kind,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        }
    }

    #[test]
    fn build_request_function_tool_becomes_function_declarations() {
        let tool = make_tool(
            ToolKind::Function,
            "get_weather",
            json!({ "type": "object", "properties": {} }),
        );
        let options = ChatOptions::new().with_tool(tool);
        let body = build_request(&[user("hi")], &options);
        assert_eq!(
            body["tools"],
            json!([{
                "functionDeclarations": [{
                    "name": "get_weather",
                    "description": "a tool",
                    "parameters": { "type": "object", "properties": {} }
                }]
            }])
        );
    }

    #[test]
    fn tools_to_gemini_multiple_function_tools_share_one_declarations_list() {
        let tools = vec![
            make_tool(ToolKind::Function, "a", json!({})),
            make_tool(ToolKind::Function, "b", json!({})),
        ];
        let out = tools_to_gemini(&tools);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0]["functionDeclarations"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn tools_to_gemini_web_search_and_code_interpreter() {
        let tools = vec![
            make_tool(ToolKind::HostedWebSearch, "web_search", json!({})),
            make_tool(ToolKind::HostedCodeInterpreter, "code", json!({})),
        ];
        let out = tools_to_gemini(&tools);
        assert!(out.contains(&json!({ "googleSearch": {} })));
        assert!(out.contains(&json!({ "codeExecution": {} })));
    }

    #[test]
    fn tools_to_gemini_unsupported_hosted_kinds_are_skipped() {
        let tools = vec![
            make_tool(
                ToolKind::HostedFileSearch { max_results: None },
                "fs",
                json!({}),
            ),
            make_tool(
                ToolKind::HostedMcp {
                    url: "https://example.com".into(),
                    allowed_tools: None,
                },
                "mcp",
                json!({}),
            ),
            make_tool(ToolKind::HostedImageGeneration, "img", json!({})),
        ];
        let out = tools_to_gemini(&tools);
        assert!(out.is_empty());
    }

    #[test]
    fn build_request_additional_properties_pass_through() {
        let mut options = ChatOptions::new();
        options
            .additional_properties
            .insert("cachedContent".into(), json!("cachedContents/abc"));
        let body = build_request(&[user("hi")], &options);
        assert_eq!(body["cachedContent"], json!("cachedContents/abc"));
    }

    // endregion

    // region: response parsing

    #[test]
    fn parse_response_text_and_usage() {
        let value = json!({
            "modelVersion": "gemini-x",
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "Hello!" }] },
                "finishReason": "STOP",
            }],
            "usageMetadata": { "promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15 },
        });
        let resp = parse_response(&value);
        assert_eq!(resp.model.as_deref(), Some("gemini-x"));
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(10));
        assert_eq!(usage.output_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(15));
    }

    #[test]
    fn parse_response_function_call_generates_call_id_and_upgrades_finish_reason() {
        let value = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "functionCall": { "name": "get_weather", "args": { "city": "Paris" } } }
                ] },
                "finishReason": "STOP",
            }],
        });
        let resp = parse_response(&value);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert!(
            !calls[0].call_id.is_empty(),
            "a call id must be synthesized"
        );
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &json!("Paris")
        );
        // STOP + a function call present -> upgraded to tool_calls.
        assert_eq!(resp.finish_reason, Some(FinishReason::tool_calls()));
    }

    #[test]
    fn parse_response_max_tokens_with_function_call_is_not_upgraded() {
        let value = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "functionCall": { "name": "f", "args": {} } }
                ] },
                "finishReason": "MAX_TOKENS",
            }],
        });
        let resp = parse_response(&value);
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::new(FinishReason::LENGTH))
        );
    }

    #[test]
    fn parse_response_thought_part_becomes_text_reasoning() {
        let value = json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "text": "thinking...", "thought": true },
                    { "text": "answer" }
                ] },
            }],
        });
        let resp = parse_response(&value);
        let msg = &resp.messages[0];
        assert!(matches!(msg.contents[0], Content::TextReasoning(_)));
        assert_eq!(msg.contents[0].as_text(), Some("thinking..."));
        assert!(matches!(msg.contents[1], Content::Text(_)));
    }

    #[test]
    fn parse_response_blocked_prompt_sets_content_filter_no_error() {
        let value = json!({
            "promptFeedback": { "blockReason": "SAFETY" },
        });
        let resp = parse_response(&value);
        assert_eq!(
            resp.finish_reason,
            Some(FinishReason::new(FinishReason::CONTENT_FILTER))
        );
        assert_eq!(resp.text(), "");
    }

    #[test]
    fn map_finish_reason_covers_documented_mapping() {
        assert_eq!(map_finish_reason("STOP"), FinishReason::stop());
        assert_eq!(
            map_finish_reason("MAX_TOKENS"),
            FinishReason::new(FinishReason::LENGTH)
        );
        for reason in [
            "SAFETY",
            "RECITATION",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
        ] {
            assert_eq!(
                map_finish_reason(reason),
                FinishReason::new(FinishReason::CONTENT_FILTER),
                "{reason}"
            );
        }
    }

    #[test]
    fn parse_usage_reads_thoughts_token_count() {
        let usage =
            json!({ "promptTokenCount": 1, "candidatesTokenCount": 2, "thoughtsTokenCount": 3 });
        let details = parse_usage(&usage);
        assert_eq!(details.reasoning_output_token_count, Some(3));
    }

    // endregion

    // region: streaming chunk parsing

    #[test]
    fn parse_stream_chunk_text_delta() {
        let value = json!({
            "candidates": [{ "content": { "role": "model", "parts": [{ "text": "Hel" }] } }],
        });
        let update = parse_stream_chunk(&value).unwrap();
        assert_eq!(update.text_content(), "Hel");
        assert_eq!(update.finish_reason, None);
    }

    #[test]
    fn parse_stream_chunk_final_carries_usage_and_finish_reason() {
        let value = json!({
            "candidates": [{ "content": { "role": "model", "parts": [] }, "finishReason": "STOP" }],
            "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10 },
        });
        let update = parse_stream_chunk(&value).unwrap();
        assert_eq!(update.finish_reason, Some(FinishReason::stop()));
        let usage_content = update
            .contents
            .iter()
            .find_map(|c| match c {
                Content::Usage(u) => Some(u.details.clone()),
                _ => None,
            })
            .unwrap();
        assert_eq!(usage_content.total_token_count, Some(10));
    }

    #[test]
    fn parse_stream_chunk_empty_returns_none() {
        assert!(parse_stream_chunk(&json!({})).is_none());
    }

    #[test]
    fn stream_chunks_aggregate_into_full_text_via_chat_response() {
        use agent_framework_core::types::ChatResponse;
        let chunks = [
            json!({ "candidates": [{ "content": { "parts": [{ "text": "Hel" }] } }] }),
            json!({ "candidates": [{ "content": { "parts": [{ "text": "lo!" }] }, "finishReason": "STOP" }],
                     "usageMetadata": { "promptTokenCount": 5, "candidatesTokenCount": 2 } }),
        ];
        let updates: Vec<_> = chunks.iter().filter_map(parse_stream_chunk).collect();
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(5));
        assert_eq!(usage.output_token_count, Some(2));
    }

    // endregion
}
