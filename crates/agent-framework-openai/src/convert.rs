//! Conversion between framework types and the OpenAI chat-completions wire
//! format.

use agent_framework_core::tools::ToolKind;
use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, Content, DataContent, FinishReason, FunctionArguments,
    FunctionCallContent, FunctionResultContent, Role, TextContent, ToolMode, UsageDetails,
};
use serde_json::{json, Map, Value};

/// Default filename used for file attachments when the content carries none.
pub(crate) const DEFAULT_FILENAME: &str = "file";

/// The lower-cased top-level media type (the part before `/`), mirroring
/// upstream `_has_top_level_media_type`.
pub(crate) fn top_level_media_type(media_type: &str) -> String {
    let span = match media_type.find('/') {
        Some(i) => &media_type[..i],
        None => media_type,
    };
    span.trim().to_ascii_lowercase()
}

/// The OpenAI audio `format` string for a media type, or `None` if unsupported.
pub(crate) fn audio_format(media_type: &str) -> Option<&'static str> {
    if media_type.contains("wav") {
        Some("wav")
    } else if media_type.contains("mp3") || media_type.contains("mpeg") {
        Some("mp3")
    } else {
        None
    }
}

/// Strip a leading `data:<media-type>;base64,` prefix, returning just the base64
/// payload; a non-`data:` URI is returned unchanged (mirrors upstream).
pub(crate) fn strip_data_uri_prefix(uri: &str) -> &str {
    if uri.starts_with("data:") {
        uri.split_once(',').map_or(uri, |(_, payload)| payload)
    } else {
        uri
    }
}

/// Resolve a [`DataContent`]'s media type, parsing it from the `data:` URI when
/// the explicit field is absent (mirrors upstream, which always populates it
/// from the URI on construction).
pub(crate) fn data_content_media_type(data: &DataContent) -> Option<String> {
    if let Some(mt) = &data.media_type {
        return Some(mt.clone());
    }
    let rest = data.uri.strip_prefix("data:")?;
    let end = rest.find([';', ','])?;
    Some(rest[..end].to_string())
}

/// Convert framework messages into the OpenAI `messages` array.
pub fn messages_to_openai(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg.role.as_str();
        // Collect concatenated text, ordered content parts (for multimodal
        // input), tool calls, and tool results.
        let mut text = String::new();
        let mut parts: Vec<Value> = Vec::new();
        let mut has_media_part = false;
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut tool_results: Vec<&FunctionResultContent> = Vec::new();

        for content in &msg.contents {
            match content {
                Content::Text(t) => {
                    text.push_str(&t.text);
                    parts.push(json!({ "type": "text", "text": t.text }));
                }
                Content::FunctionCall(fc) => tool_calls.push(function_call_to_openai(fc)),
                Content::FunctionResult(fr) => tool_results.push(fr),
                Content::Uri(u) => {
                    if let Some(part) = content_part_to_openai(&u.uri, Some(&u.media_type)) {
                        parts.push(part);
                        has_media_part = true;
                    }
                }
                Content::Data(d) => {
                    if let Some(part) =
                        content_part_to_openai(&d.uri, data_content_media_type(d).as_deref())
                    {
                        parts.push(part);
                        has_media_part = true;
                    }
                }
                _ => {}
            }
        }

        // A tool-role message maps each result to its own OpenAI `tool` message.
        if role == Role::TOOL {
            for fr in tool_results {
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": fr.call_id,
                    "content": result_to_string(fr),
                }));
            }
            continue;
        }

        let mut obj = Map::new();
        obj.insert("role".into(), json!(role));
        // When a non-text (image/audio/file) part is present, emit a typed
        // content-parts array with text folded in as `{"type":"text",...}`
        // parts; otherwise keep the plain-string form for wire stability.
        if has_media_part {
            obj.insert("content".into(), Value::Array(parts));
        } else if !text.is_empty() || tool_calls.is_empty() {
            obj.insert("content".into(), json!(text));
        }
        if let Some(name) = &msg.author_name {
            // OpenAI `name` must be a simple token; skip if it has spaces.
            if !name.contains(char::is_whitespace) {
                obj.insert("name".into(), json!(name));
            }
        }
        if !tool_calls.is_empty() {
            obj.insert("tool_calls".into(), json!(tool_calls));
        }
        out.push(Value::Object(obj));
    }
    out
}

/// Map a URI/data content item to a Chat Completions content part, or `None`
/// when it has no wire mapping (skipped, mirroring upstream
/// `_openai_content_parser`). Handles images (`image_url`), audio
/// (`input_audio`), and `application/*` data URIs (`file`).
fn content_part_to_openai(uri: &str, media_type: Option<&str>) -> Option<Value> {
    let media_type = media_type?;
    match top_level_media_type(media_type).as_str() {
        "image" => Some(json!({ "type": "image_url", "image_url": { "url": uri } })),
        "audio" => {
            let format = audio_format(media_type)?;
            Some(json!({
                "type": "input_audio",
                "input_audio": { "data": strip_data_uri_prefix(uri), "format": format },
            }))
        }
        "application" if uri.starts_with("data:") => Some(json!({
            "type": "file",
            "file": { "file_data": uri, "filename": DEFAULT_FILENAME },
        })),
        _ => None,
    }
}

/// Render a function/tool result as the plain-text/JSON string OpenAI wire
/// formats expect for tool output. Shared by Chat Completions and Responses.
pub(crate) fn result_to_string(fr: &FunctionResultContent) -> String {
    if let Some(exc) = &fr.exception {
        return format!("error: {exc}");
    }
    match &fr.result {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

fn function_call_to_openai(fc: &FunctionCallContent) -> Value {
    json!({
        "id": fc.call_id,
        "type": "function",
        "function": { "name": fc.name, "arguments": function_arguments_to_string(&fc.arguments) }
    })
}

/// Render function-call arguments as the JSON-encoded string the OpenAI wire
/// formats (Chat Completions and Responses alike) expect.
///
/// Public (but hidden) so `responses.rs` in this crate, and any
/// OpenAI-wire-compatible reuse outside it, can share the exact same
/// stringification instead of duplicating it.
#[doc(hidden)]
pub fn function_arguments_to_string(args: &Option<FunctionArguments>) -> String {
    match args {
        Some(FunctionArguments::Raw(s)) => s.clone(),
        Some(FunctionArguments::Object(m)) => {
            serde_json::to_string(m).unwrap_or_else(|_| "{}".into())
        }
        None => "{}".into(),
    }
}

/// Build the tools array and tool_choice for a request.
///
/// Only [`ToolKind::Function`] tools are emitted in `tools`. A hosted
/// web-search tool is surfaced separately as the top-level `web_search_options`
/// field (see [`apply_options`]); any other hosted kind is unsupported on Chat
/// Completions and skipped with a warning. Mirrors upstream
/// `_chat_client._chat_to_tool_spec`.
pub fn tools_to_openai(options: &ChatOptions) -> (Option<Value>, Option<Value>) {
    let tools: Vec<Value> = options
        .tools
        .iter()
        .filter_map(|t| match &t.kind {
            ToolKind::Function => Some(t.to_openai_spec()),
            // Surfaced as `web_search_options`, not a `tools` entry.
            ToolKind::HostedWebSearch => None,
            other => {
                tracing::warn!(
                    tool = %t.name,
                    kind = ?other,
                    "hosted tool is not supported on the Chat Completions API; skipping",
                );
                None
            }
        })
        .collect();
    if tools.is_empty() {
        return (None, None);
    }
    let choice = options.tool_choice.as_ref().map(|tc| match tc {
        ToolMode::Auto => json!("auto"),
        ToolMode::None => json!("none"),
        ToolMode::Required(Some(name)) => {
            json!({ "type": "function", "function": { "name": name } })
        }
        ToolMode::Required(None) => json!("required"),
    });
    (Some(json!(tools)), choice)
}

/// Apply the scalar chat options onto a request body map.
pub fn apply_options(body: &mut Map<String, Value>, options: &ChatOptions) {
    macro_rules! set {
        ($key:literal, $val:expr) => {
            if let Some(v) = $val {
                body.insert($key.into(), json!(v));
            }
        };
    }
    set!("temperature", options.temperature);
    set!("top_p", options.top_p);
    set!("max_tokens", options.max_tokens);
    set!("frequency_penalty", options.frequency_penalty);
    set!("presence_penalty", options.presence_penalty);
    set!("seed", options.seed);
    set!("user", options.user.clone());
    set!("store", options.store);
    if let Some(stop) = &options.stop {
        body.insert("stop".into(), json!(stop));
    }
    if let Some(fmt) = &options.response_format {
        // `ResponseFormat` serializes directly to the Chat Completions
        // `response_format` object for all three variants.
        body.insert("response_format".into(), json!(fmt));
    }
    if let Some(bias) = &options.logit_bias {
        body.insert("logit_bias".into(), json!(bias));
    }
    if let Some(metadata) = &options.metadata {
        body.insert("metadata".into(), json!(metadata));
    }
    // `parallel_tool_calls` is only valid alongside function tools; upstream
    // drops it otherwise (`_chat_client._prepare_options`).
    if let Some(allow) = options.allow_multiple_tool_calls {
        if options.tools.iter().any(|t| t.kind == ToolKind::Function) {
            body.insert("parallel_tool_calls".into(), json!(allow));
        }
    }
    // A hosted web-search tool is expressed as the top-level
    // `web_search_options` request field rather than an entry in `tools`
    // (`_chat_client._process_web_search_tool`).
    if let Some(tool) = options
        .tools
        .iter()
        .find(|t| t.kind == ToolKind::HostedWebSearch)
    {
        let mut ws = Map::new();
        if let Some(loc) = tool.parameters.get("user_location") {
            ws.insert(
                "user_location".into(),
                json!({ "type": "approximate", "approximate": loc }),
            );
        }
        body.insert("web_search_options".into(), Value::Object(ws));
    }
    for (k, v) in &options.additional_properties {
        body.entry(k.clone()).or_insert_with(|| v.clone());
    }
}

/// Parse a full (non-streaming) OpenAI chat-completion response.
pub fn parse_response(value: &Value) -> ChatResponse {
    let mut response = ChatResponse {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model_id: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    if let Some(choice) = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        let mut contents: Vec<Content> = Vec::new();
        if let Some(msg) = choice.get("message") {
            let mut has_text = false;
            if let Some(text) = msg.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    contents.push(Content::Text(TextContent::new(text)));
                    has_text = true;
                }
            }
            // A refusal is surfaced as plain text (mirrors upstream
            // `_parse_text_from_choice`), used only when there is no content.
            if !has_text {
                if let Some(refusal) = msg.get("refusal").and_then(Value::as_str) {
                    if !refusal.is_empty() {
                        contents.push(Content::Text(TextContent::new(refusal)));
                    }
                }
            }
            if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    if let Some(fc) = parse_tool_call(call) {
                        contents.push(Content::FunctionCall(fc));
                    }
                }
            }
        }
        let mut message = ChatMessage::with_contents(Role::assistant(), contents);
        message.message_id = response.response_id.clone();
        response.messages.push(message);

        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            response.finish_reason = Some(FinishReason::new(fr));
        }
    }

    if let Some(usage) = value.get("usage") {
        response.usage_details = Some(parse_usage(usage));
    }
    response
}

fn parse_tool_call(call: &Value) -> Option<FunctionCallContent> {
    let id = call.get("id").and_then(Value::as_str)?.to_string();
    let func = call.get("function")?;
    let name = func
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let args = func
        .get("arguments")
        .and_then(Value::as_str)
        .unwrap_or("{}")
        .to_string();
    Some(FunctionCallContent::new(
        id,
        name,
        Some(FunctionArguments::Raw(args)),
    ))
}

/// Parse an OpenAI-shaped `usage` object into [`UsageDetails`].
///
/// Detail breakdowns (`completion_tokens_details.*`, `prompt_tokens_details.*`)
/// are folded into [`UsageDetails::additional_counts`] under the same keys
/// upstream uses (`_chat_client._usage_details_from_openai`).
///
/// Public (but hidden) so `agent-framework-azure` can reuse it verbatim.
#[doc(hidden)]
pub fn parse_usage(usage: &Value) -> UsageDetails {
    let mut details = UsageDetails {
        input_token_count: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("completion_tokens").and_then(Value::as_u64),
        total_token_count: usage.get("total_tokens").and_then(Value::as_u64),
        additional_counts: Default::default(),
    };
    if let Some(ctd) = usage.get("completion_tokens_details") {
        add_usage_detail(
            &mut details,
            ctd,
            "accepted_prediction_tokens",
            "completion/accepted_prediction_tokens",
        );
        add_usage_detail(&mut details, ctd, "audio_tokens", "completion/audio_tokens");
        add_usage_detail(
            &mut details,
            ctd,
            "reasoning_tokens",
            "completion/reasoning_tokens",
        );
        add_usage_detail(
            &mut details,
            ctd,
            "rejected_prediction_tokens",
            "completion/rejected_prediction_tokens",
        );
    }
    if let Some(ptd) = usage.get("prompt_tokens_details") {
        add_usage_detail(&mut details, ptd, "audio_tokens", "prompt/audio_tokens");
        add_usage_detail(&mut details, ptd, "cached_tokens", "prompt/cached_tokens");
    }
    details
}

/// Copy a positive token count from `obj[src]` into `details.additional_counts`
/// under `dest`. Zero/absent counts are skipped, mirroring upstream's truthy
/// `if tokens := ...` guard.
fn add_usage_detail(details: &mut UsageDetails, obj: &Value, src: &str, dest: &str) {
    if let Some(v) = obj.get(src).and_then(Value::as_u64) {
        if v > 0 {
            details.additional_counts.insert(dest.to_string(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{
        hosted_code_interpreter, hosted_web_search, ApprovalMode, ToolDefinition,
    };
    use agent_framework_core::types::{TextReasoningContent, UriContent};
    use std::collections::HashMap;

    fn function_tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: "desc".into(),
            parameters: json!({ "type": "object", "properties": {} }),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        }
    }

    fn user_with(contents: Vec<Content>) -> ChatMessage {
        ChatMessage::with_contents(Role::user(), contents)
    }

    // region: multimodal input

    #[test]
    fn text_only_message_stays_a_content_string() {
        let out = messages_to_openai(&[ChatMessage::user("hi")]);
        assert_eq!(out[0], json!({ "role": "user", "content": "hi" }));
    }

    #[test]
    fn text_plus_reasoning_stays_string_reasoning_skipped() {
        // TextReasoning has no chat-completions wire mapping: it is skipped and
        // the message keeps the plain-string content form.
        let msg = user_with(vec![
            Content::Text(TextContent::new("hi")),
            Content::TextReasoning(TextReasoningContent {
                text: "secret".into(),
                annotations: None,
            }),
        ]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({ "role": "user", "content": "hi" })
        );
    }

    #[test]
    fn image_uri_becomes_content_parts_array() {
        let msg = user_with(vec![
            Content::Text(TextContent::new("look:")),
            Content::Uri(UriContent {
                uri: "https://example.com/cat.png".into(),
                media_type: "image/png".into(),
            }),
        ]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": "look:" },
                    { "type": "image_url", "image_url": { "url": "https://example.com/cat.png" } },
                ],
            })
        );
    }

    #[test]
    fn image_data_uri_infers_media_type_from_uri() {
        // media_type is None; it must be parsed from the data URI prefix.
        let msg = user_with(vec![Content::Data(DataContent {
            uri: "data:image/png;base64,AAAA".into(),
            media_type: None,
        })]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({
                "role": "user",
                "content": [
                    { "type": "image_url", "image_url": { "url": "data:image/png;base64,AAAA" } },
                ],
            })
        );
    }

    #[test]
    fn audio_data_becomes_input_audio_with_stripped_base64() {
        let msg = user_with(vec![Content::Data(DataContent {
            uri: "data:audio/wav;base64,QQQQ".into(),
            media_type: Some("audio/wav".into()),
        })]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({
                "role": "user",
                "content": [
                    { "type": "input_audio", "input_audio": { "data": "QQQQ", "format": "wav" } },
                ],
            })
        );
    }

    #[test]
    fn audio_mpeg_maps_to_mp3_format() {
        let msg = user_with(vec![Content::Data(DataContent {
            uri: "data:audio/mpeg;base64,SUQz".into(),
            media_type: Some("audio/mpeg".into()),
        })]);
        let out = messages_to_openai(&[msg]);
        assert_eq!(out[0]["content"][0]["input_audio"]["format"], json!("mp3"));
    }

    #[test]
    fn unsupported_audio_format_is_skipped() {
        // ogg has no OpenAI `input_audio` format: the part is dropped, leaving
        // a text-only string message.
        let msg = user_with(vec![
            Content::Text(TextContent::new("clip")),
            Content::Data(DataContent {
                uri: "data:audio/ogg;base64,T2dn".into(),
                media_type: Some("audio/ogg".into()),
            }),
        ]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({ "role": "user", "content": "clip" })
        );
    }

    #[test]
    fn application_data_becomes_file_part_with_default_filename() {
        let msg = user_with(vec![Content::Data(DataContent {
            uri: "data:application/pdf;base64,JVBERi0x".into(),
            media_type: Some("application/pdf".into()),
        })]);
        assert_eq!(
            messages_to_openai(&[msg])[0],
            json!({
                "role": "user",
                "content": [
                    { "type": "file", "file": {
                        "file_data": "data:application/pdf;base64,JVBERi0x",
                        "filename": "file",
                    } },
                ],
            })
        );
    }

    #[test]
    fn assistant_tool_call_only_still_omits_content() {
        let msg = ChatMessage::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "f",
                Some(FunctionArguments::Raw("{}".into())),
            ))],
        );
        let out = messages_to_openai(&[msg]);
        assert!(out[0].get("content").is_none());
        assert_eq!(out[0]["tool_calls"][0]["function"]["name"], json!("f"));
    }

    // endregion

    // region: tools & hosted tools

    #[test]
    fn only_function_tools_emitted_web_search_excluded() {
        let mut options = ChatOptions::new();
        options.tools = vec![function_tool("get_weather"), hosted_web_search()];
        let (tools, _choice) = tools_to_openai(&options);
        let tools = tools.unwrap();
        assert_eq!(tools.as_array().unwrap().len(), 1);
        assert_eq!(tools[0]["function"]["name"], json!("get_weather"));
    }

    #[test]
    fn other_hosted_tools_are_skipped() {
        let mut options = ChatOptions::new();
        options.tools = vec![hosted_code_interpreter()];
        let (tools, choice) = tools_to_openai(&options);
        assert!(tools.is_none());
        assert!(choice.is_none());
    }

    #[test]
    fn web_search_tool_sets_web_search_options_with_user_location() {
        let mut tool = hosted_web_search();
        tool.parameters = json!({ "user_location": { "city": "Seattle", "country": "US" } });
        let mut options = ChatOptions::new();
        options.tools = vec![tool];
        let mut body = Map::new();
        apply_options(&mut body, &options);
        assert_eq!(
            body["web_search_options"],
            json!({
                "user_location": {
                    "type": "approximate",
                    "approximate": { "city": "Seattle", "country": "US" },
                },
            })
        );
    }

    #[test]
    fn web_search_tool_without_location_sets_empty_options() {
        let mut options = ChatOptions::new();
        options.tools = vec![hosted_web_search()];
        let mut body = Map::new();
        apply_options(&mut body, &options);
        assert_eq!(body["web_search_options"], json!({}));
    }

    // endregion

    // region: request options

    #[test]
    fn logit_bias_metadata_and_parallel_tool_calls_are_sent() {
        let mut options = ChatOptions::new();
        options.logit_bias = Some(HashMap::from([("50256".to_string(), -100.0)]));
        options.metadata = Some(HashMap::from([("session".to_string(), "abc".to_string())]));
        options.allow_multiple_tool_calls = Some(true);
        options.tools = vec![function_tool("f")];
        let mut body = Map::new();
        apply_options(&mut body, &options);
        assert_eq!(body["logit_bias"], json!({ "50256": -100.0 }));
        assert_eq!(body["metadata"], json!({ "session": "abc" }));
        assert_eq!(body["parallel_tool_calls"], json!(true));
    }

    #[test]
    fn parallel_tool_calls_omitted_without_function_tools() {
        let mut options = ChatOptions::new();
        options.allow_multiple_tool_calls = Some(true);
        // Only a hosted tool: `parallel_tool_calls` would be an API error.
        options.tools = vec![hosted_web_search()];
        let mut body = Map::new();
        apply_options(&mut body, &options);
        assert!(body.get("parallel_tool_calls").is_none());
    }

    // endregion

    // region: response parsing

    #[test]
    fn refusal_is_parsed_as_text() {
        let value = json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{
                "message": { "role": "assistant", "refusal": "I can't help with that." },
                "finish_reason": "stop",
            }],
        });
        let resp = parse_response(&value);
        assert_eq!(resp.text(), "I can't help with that.");
    }

    #[test]
    fn content_takes_precedence_over_refusal() {
        let value = json!({
            "choices": [{
                "message": { "role": "assistant", "content": "answer", "refusal": "nope" },
            }],
        });
        let resp = parse_response(&value);
        assert_eq!(resp.text(), "answer");
    }

    #[test]
    fn usage_detail_breakdowns_are_folded_into_additional_counts() {
        let usage = json!({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "completion_tokens_details": {
                "reasoning_tokens": 20,
                "audio_tokens": 5,
                "accepted_prediction_tokens": 3,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": { "cached_tokens": 40, "audio_tokens": 2 },
        });
        let d = parse_usage(&usage);
        assert_eq!(d.input_token_count, Some(100));
        assert_eq!(d.output_token_count, Some(50));
        assert_eq!(
            d.additional_counts.get("completion/reasoning_tokens"),
            Some(&20)
        );
        assert_eq!(d.additional_counts.get("completion/audio_tokens"), Some(&5));
        assert_eq!(
            d.additional_counts
                .get("completion/accepted_prediction_tokens"),
            Some(&3)
        );
        // Zero-valued counts are skipped (truthy guard).
        assert!(!d
            .additional_counts
            .contains_key("completion/rejected_prediction_tokens"));
        assert_eq!(d.additional_counts.get("prompt/cached_tokens"), Some(&40));
        assert_eq!(d.additional_counts.get("prompt/audio_tokens"), Some(&2));
    }

    // endregion
}
