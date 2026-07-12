//! Conversion between this crate's message/response types and the Bedrock
//! Runtime [`Converse` API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html)
//! JSON shapes.
//!
//! Unlike the OpenAI-shaped Chat Completions wire format, Bedrock's Converse
//! request separates a top-level `system` array from `messages`, and each
//! message's `content` is a list of typed blocks: `{"text": ...}`,
//! `{"toolUse": {"toolUseId", "name", "input"}}`, or `{"toolResult":
//! {"toolUseId", "content", "status"}}`. There is no dedicated `system` or
//! `tool` role at the message level — a `system`-role
//! [`Message`] contributes to the top-level `system` array instead of a
//! turn, and a tool result is sent back as a `user`-role message carrying a
//! `toolResult` block, exactly like Anthropic's Messages API (Bedrock's
//! Converse API is deliberately provider-agnostic, but its shape mirrors
//! Anthropic's most closely of the providers this workspace supports).
//! Reference: <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html>.

use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatOptions, ChatResponse, Content, FinishReason, FunctionArguments, FunctionCallContent,
    FunctionResultContent, Message, Role, ToolMode, UsageDetails,
};
use serde_json::{json, Map, Value};

/// Build a full Bedrock `Converse` / `ConverseStream` request body (minus
/// the `modelId`, which the caller embeds in the URL path rather than the
/// body).
pub fn build_request(messages: &[Message], options: &ChatOptions) -> Value {
    let (system_from_messages, turns) = messages_to_bedrock(messages);
    let system =
        merge_instructions_into_system(system_from_messages, options.instructions.as_deref());

    let mut body = Map::new();
    body.insert("messages".into(), Value::Array(turns));
    if let Some(system) = system {
        body.insert("system".into(), system);
    }
    apply_inference_config(&mut body, options);
    if let Some(Value::Object(tool_config)) = tools_to_bedrock(options) {
        for (k, v) in tool_config {
            body.insert(k, v);
        }
    }
    for (k, v) in &options.additional_properties {
        body.entry(k.clone()).or_insert_with(|| v.clone());
    }
    Value::Object(body)
}

/// Split messages into Bedrock's top-level `system` blocks and its
/// `messages` turn array.
///
/// Every `system`-role [`Message`] (regardless of position — Bedrock has no
/// turn-taking concept of "system", just an out-of-band field) contributes a
/// `{"text": ...}` block to the returned system array. All other messages
/// become a `user`/`assistant` turn (`assistant` role maps to `"assistant"`;
/// everything else — `user`, `tool`, or any custom role — maps to
/// `"user"`), and consecutive same-role turns are merged (Converse, like
/// Anthropic's Messages API, requires alternating `user`/`assistant` turns
/// starting with `user`).
pub fn messages_to_bedrock(messages: &[Message]) -> (Option<Value>, Vec<Value>) {
    let mut system_blocks: Vec<Value> = Vec::new();
    let mut turns: Vec<Value> = Vec::new();

    for msg in messages {
        if msg.role == Role::system() {
            let text = msg.text();
            if !text.is_empty() {
                system_blocks.push(json!({ "text": text }));
            }
            continue;
        }

        let role = if msg.role == Role::assistant() {
            "assistant"
        } else {
            "user"
        };
        let blocks: Vec<Value> = msg.contents.iter().filter_map(content_to_block).collect();
        if blocks.is_empty() {
            // Converse rejects a turn with an empty `content` array.
            continue;
        }
        turns.push(json!({ "role": role, "content": blocks }));
    }

    let turns = normalize_role_alternation(turns);
    let system = if system_blocks.is_empty() {
        None
    } else {
        Some(Value::Array(system_blocks))
    };
    (system, turns)
}

/// Prepend [`ChatOptions::instructions`] (the request-level system prompt,
/// distinct from any `system`-role [`Message`]) as a leading system block,
/// ahead of whatever [`messages_to_bedrock`] already extracted from the
/// conversation's own `system`-role messages.
fn merge_instructions_into_system(
    system: Option<Value>,
    instructions: Option<&str>,
) -> Option<Value> {
    let instructions = instructions.filter(|s| !s.is_empty());
    match (instructions, system) {
        (None, system) => system,
        (Some(instr), None) => Some(json!([{ "text": instr }])),
        (Some(instr), Some(Value::Array(mut blocks))) => {
            blocks.insert(0, json!({ "text": instr }));
            Some(Value::Array(blocks))
        }
        // `system` is always built as `Value::Array` by `messages_to_bedrock`;
        // this arm only guards against a future change to that invariant.
        (Some(instr), Some(other)) => Some(json!([{ "text": instr }, other])),
    }
}

/// Enforce Converse's conversation-shape rule: turns must alternate between
/// `user` and `assistant`, starting with `user`. Consecutive same-role turns
/// (e.g. several tool-result messages produced by the local tool loop, or
/// several user turns from a group chat) are merged by concatenating their
/// content blocks; a leading assistant turn gets a minimal synthetic user
/// turn inserted before it so the greeting is preserved.
fn normalize_role_alternation(turns: Vec<Value>) -> Vec<Value> {
    let mut out: Vec<Value> = Vec::with_capacity(turns.len());
    for turn in turns {
        match out.last_mut() {
            Some(prev) if prev["role"] == turn["role"] => {
                if let (Some(prev_blocks), Some(new_blocks)) =
                    (prev["content"].as_array_mut(), turn["content"].as_array())
                {
                    prev_blocks.extend(new_blocks.iter().cloned());
                }
            }
            _ => out.push(turn),
        }
    }
    if out.first().map(|m| m["role"] == "assistant") == Some(true) {
        out.insert(
            0,
            json!({
                "role": "user",
                "content": [{ "text": "(continuing the conversation)" }],
            }),
        );
    }
    out
}

/// Map a single [`Content`] item to a Converse content block, or `None` for
/// a variant Converse has no representation for (skipped, mirroring the
/// other non-OpenAI-shaped providers in this workspace).
fn content_to_block(content: &Content) -> Option<Value> {
    match content {
        Content::Text(t) => Some(json!({ "text": t.text })),
        Content::FunctionCall(fc) => Some(tool_use_block(fc)),
        Content::FunctionResult(fr) => Some(tool_result_block(fr)),
        _ => None,
    }
}

/// Build a `{"toolUse": {...}}` block from a [`FunctionCallContent`].
/// `toolUseId` is required by Converse; an empty `call_id` (not expected in
/// practice, since providers always assign one) is given a synthesized one
/// rather than sent blank.
fn tool_use_block(fc: &FunctionCallContent) -> Value {
    let tool_use_id = if fc.call_id.is_empty() {
        format!("call_{}", uuid::Uuid::new_v4())
    } else {
        fc.call_id.clone()
    };
    let input = fc.parse_arguments().unwrap_or_default();
    json!({
        "toolUse": {
            "toolUseId": tool_use_id,
            "name": fc.name,
            "input": Value::Object(input.into_iter().collect()),
        }
    })
}

/// Build a `{"toolResult": {...}}` block from a [`FunctionResultContent`].
/// An exception maps to `status: "error"`; a successful result maps to
/// `status: "success"` with the value carried as a `json` content part when
/// it's a JSON object/array/number/bool, or a `text` part for a bare string
/// or a missing result (Converse requires `content` to be non-empty).
fn tool_result_block(fr: &FunctionResultContent) -> Value {
    let (content, status) = if let Some(exc) = &fr.exception {
        (vec![json!({ "text": exc })], "error")
    } else {
        match &fr.result {
            Some(Value::String(s)) => (vec![json!({ "text": s })], "success"),
            Some(v) => (vec![json!({ "json": v })], "success"),
            None => (vec![json!({ "text": "" })], "success"),
        }
    };
    json!({
        "toolResult": {
            "toolUseId": fr.call_id,
            "content": content,
            "status": status,
        }
    })
}

/// Build the top-level `toolConfig` field from `options.tools`, or `None`
/// when there are no (supported) tools to advertise.
///
/// Only [`ToolKind::Function`] tools map to Converse's `toolSpec`; Converse
/// has no universal hosted-tool mechanism analogous to Anthropic's
/// server-side web-search/code-execution tools or Gemini's `googleSearch` —
/// each Bedrock model family exposes hosted capabilities (if any)
/// differently, outside the tool-calling contract this client implements —
/// so other [`ToolKind`] variants are skipped with a warning rather than
/// guessed at.
pub fn tools_to_bedrock(options: &ChatOptions) -> Option<Value> {
    if options.tools.is_empty() {
        return None;
    }
    let specs: Vec<Value> = options.tools.iter().filter_map(tool_spec).collect();
    if specs.is_empty() {
        return None;
    }

    let mut tool_config = Map::new();
    tool_config.insert("tools".into(), Value::Array(specs));
    if let Some(mode) = &options.tool_choice {
        if let Some(choice) = tool_choice_to_bedrock(mode) {
            tool_config.insert("toolChoice".into(), choice);
        }
    }
    Some(json!({ "toolConfig": Value::Object(tool_config) }))
}

fn tool_spec(tool: &ToolDefinition) -> Option<Value> {
    match &tool.kind {
        ToolKind::Function => Some(json!({
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": { "json": tool.parameters },
            }
        })),
        other => {
            tracing::warn!(
                tool = %tool.name,
                kind = ?other,
                "Bedrock Converse: hosted tool kind is not supported by this client; skipping"
            );
            None
        }
    }
}

/// Map a [`ToolMode`] to Converse's `toolChoice` shape. `ToolMode::None`
/// (tools disabled) has no `toolChoice` representation — callers should omit
/// `tools`/`toolConfig` entirely in that case, which `tools_to_bedrock`'s
/// caller ([`build_request`]) never does since it is only invoked when
/// `options.tools` is non-empty; a `ToolMode::None` alongside a non-empty
/// tool list simply omits `toolChoice`, leaving Converse's own default
/// (`auto`) in effect.
fn tool_choice_to_bedrock(mode: &ToolMode) -> Option<Value> {
    match mode {
        ToolMode::Auto => Some(json!({ "auto": {} })),
        ToolMode::Required(None) => Some(json!({ "any": {} })),
        ToolMode::Required(Some(name)) => Some(json!({ "tool": { "name": name } })),
        ToolMode::None => None,
    }
}

/// Map [`ChatOptions`] into Converse's `inferenceConfig` object, inserting
/// it into `body` only when at least one supported field is set.
pub fn apply_inference_config(body: &mut Map<String, Value>, options: &ChatOptions) {
    let mut cfg = Map::new();
    if let Some(t) = options.temperature {
        cfg.insert("temperature".into(), json!(t));
    }
    if let Some(mt) = options.max_tokens {
        cfg.insert("maxTokens".into(), json!(mt));
    }
    if let Some(tp) = options.top_p {
        cfg.insert("topP".into(), json!(tp));
    }
    if let Some(stop) = &options.stop {
        if !stop.is_empty() {
            cfg.insert("stopSequences".into(), json!(stop));
        }
    }
    if !cfg.is_empty() {
        body.insert("inferenceConfig".into(), Value::Object(cfg));
    }
}

/// Parse a Bedrock `Converse` API response body into a [`ChatResponse`].
pub fn parse_response(value: &Value) -> ChatResponse {
    let mut response = ChatResponse::default();

    let message = value.get("output").and_then(|o| o.get("message"));
    if let Some(message) = message {
        let contents = message
            .get("content")
            .and_then(Value::as_array)
            .map(|blocks| parse_content_blocks(blocks))
            .unwrap_or_default();
        response
            .messages
            .push(Message::with_contents(Role::assistant(), contents));
    }

    if let Some(reason) = value.get("stopReason").and_then(Value::as_str) {
        response.finish_reason = Some(map_stop_reason(reason));
    }
    if let Some(usage) = value.get("usage") {
        response.usage_details = Some(parse_usage(usage));
    }
    response
}

fn parse_content_blocks(blocks: &[Value]) -> Vec<Content> {
    blocks.iter().filter_map(parse_content_block).collect()
}

fn parse_content_block(block: &Value) -> Option<Content> {
    if let Some(text) = block.get("text").and_then(Value::as_str) {
        return Some(Content::text(text));
    }
    if let Some(tool_use) = block.get("toolUse") {
        let call_id = tool_use
            .get("toolUseId")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let name = tool_use
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let arguments = tool_use.get("input").map(|input| match input {
            Value::Object(m) => FunctionArguments::Object(m.clone().into_iter().collect()),
            other => FunctionArguments::Raw(other.to_string()),
        });
        return Some(Content::FunctionCall(FunctionCallContent::new(
            call_id, name, arguments,
        )));
    }
    None
}

/// Map a Converse `stopReason` to a [`FinishReason`].
///
/// Reference: <https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html#API_runtime_Converse_ResponseSyntax>.
pub(crate) fn map_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" | "stop_sequence" => FinishReason::stop(),
        "max_tokens" => FinishReason::new(FinishReason::LENGTH),
        "tool_use" => FinishReason::tool_calls(),
        "content_filtered" | "guardrail_intervened" => {
            FinishReason::new(FinishReason::CONTENT_FILTER)
        }
        other => FinishReason::new(other.to_string()),
    }
}

/// Parse a Converse `usage` object (`{"inputTokens","outputTokens","totalTokens"}`)
/// into [`UsageDetails`].
pub(crate) fn parse_usage(usage: &Value) -> UsageDetails {
    UsageDetails {
        input_token_count: usage.get("inputTokens").and_then(Value::as_u64),
        output_token_count: usage.get("outputTokens").and_then(Value::as_u64),
        total_token_count: usage.get("totalTokens").and_then(Value::as_u64),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::ApprovalMode;
    use agent_framework_core::types::FunctionArguments;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn make_tool(kind: ToolKind, name: &str, parameters: Value) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: format!("{name} description"),
            parameters,
            kind,
            approval_mode: ApprovalMode::default(),
            executor: None as Option<Arc<dyn agent_framework_core::tools::Tool>>,
        }
    }

    // region: messages_to_bedrock — system extraction

    #[test]
    fn system_message_becomes_system_block_not_a_turn() {
        let messages = vec![Message::system("Be concise."), Message::user("hi")];
        let (system, turns) = messages_to_bedrock(&messages);
        assert_eq!(system, Some(json!([{ "text": "Be concise." }])));
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0]["role"], "user");
    }

    #[test]
    fn multiple_system_messages_all_contribute_regardless_of_position() {
        let messages = vec![
            Message::system("first"),
            Message::user("hi"),
            Message::system("second"),
        ];
        let (system, _turns) = messages_to_bedrock(&messages);
        assert_eq!(
            system,
            Some(json!([{ "text": "first" }, { "text": "second" }]))
        );
    }

    #[test]
    fn no_system_messages_yields_none() {
        let (system, _) = messages_to_bedrock(&[Message::user("hi")]);
        assert_eq!(system, None);
    }

    // endregion

    // region: messages_to_bedrock — role mapping and content blocks

    #[test]
    fn assistant_and_user_roles_map_directly() {
        let messages = vec![Message::user("hi"), Message::assistant("hello")];
        let (_, turns) = messages_to_bedrock(&messages);
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[0]["content"], json!([{ "text": "hi" }]));
        assert_eq!(turns[1]["role"], "assistant");
        assert_eq!(turns[1]["content"], json!([{ "text": "hello" }]));
    }

    #[test]
    fn tool_role_maps_to_user() {
        let messages = vec![
            Message::user("hi"),
            Message::assistant("calling a tool"),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent::new(
                    "call_1",
                    Some(json!("42")),
                ))],
            ),
        ];
        let (_, turns) = messages_to_bedrock(&messages);
        assert_eq!(turns[2]["role"], "user");
        assert!(turns[2]["content"][0].get("toolResult").is_some());
    }

    #[test]
    fn function_call_becomes_tool_use_block_with_parsed_input() {
        let call = FunctionCallContent::new(
            "call_1",
            "get_weather",
            Some(FunctionArguments::Raw(r#"{"city":"Paris"}"#.to_string())),
        );
        // A leading user turn avoids the synthetic-user-turn insertion
        // `normalize_role_alternation` performs for a leading assistant
        // turn, so the assistant turn under test stays at index 1.
        let messages = vec![
            Message::user("go"),
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]),
        ];
        let (_, turns) = messages_to_bedrock(&messages);
        let block = &turns[1]["content"][0]["toolUse"];
        assert_eq!(block["toolUseId"], "call_1");
        assert_eq!(block["name"], "get_weather");
        assert_eq!(block["input"], json!({ "city": "Paris" }));
    }

    #[test]
    fn function_call_with_empty_call_id_gets_synthesized_id() {
        let call = FunctionCallContent::new("", "f", None);
        let messages = vec![
            Message::user("go"),
            Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)]),
        ];
        let (_, turns) = messages_to_bedrock(&messages);
        let tool_use_id = turns[1]["content"][0]["toolUse"]["toolUseId"]
            .as_str()
            .unwrap();
        assert!(!tool_use_id.is_empty());
        assert!(tool_use_id.starts_with("call_"));
    }

    #[test]
    fn function_result_success_object_becomes_json_content_part() {
        let result = FunctionResultContent::new("call_1", Some(json!({ "temp": 72 })));
        let messages = vec![Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(result)],
        )];
        let (_, turns) = messages_to_bedrock(&messages);
        let block = &turns[0]["content"][0]["toolResult"];
        assert_eq!(block["toolUseId"], "call_1");
        assert_eq!(block["status"], "success");
        assert_eq!(block["content"], json!([{ "json": { "temp": 72 } }]));
    }

    #[test]
    fn function_result_string_becomes_text_content_part() {
        let result = FunctionResultContent::new("call_1", Some(json!("sunny")));
        let messages = vec![Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(result)],
        )];
        let (_, turns) = messages_to_bedrock(&messages);
        let block = &turns[0]["content"][0]["toolResult"];
        assert_eq!(block["content"], json!([{ "text": "sunny" }]));
        assert_eq!(block["status"], "success");
    }

    #[test]
    fn function_result_exception_becomes_error_status() {
        let mut result = FunctionResultContent::new("call_1", None);
        result.exception = Some("boom".to_string());
        let messages = vec![Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(result)],
        )];
        let (_, turns) = messages_to_bedrock(&messages);
        let block = &turns[0]["content"][0]["toolResult"];
        assert_eq!(block["status"], "error");
        assert_eq!(block["content"], json!([{ "text": "boom" }]));
    }

    #[test]
    fn message_with_no_convertible_content_is_dropped() {
        // A message whose only content is a variant Converse can't represent
        // (here, a bare Usage content item) must not produce an
        // empty-content turn.
        let messages = vec![
            Message::user("hi"),
            Message::with_contents(
                Role::assistant(),
                vec![Content::Usage(agent_framework_core::types::UsageContent {
                    details: UsageDetails::default(),
                })],
            ),
        ];
        let (_, turns) = messages_to_bedrock(&messages);
        assert_eq!(turns.len(), 1);
    }

    // endregion

    // region: role alternation normalization

    #[test]
    fn consecutive_same_role_messages_are_merged() {
        let messages = vec![Message::user("first"), Message::user("second")];
        let (_, turns) = messages_to_bedrock(&messages);
        assert_eq!(turns.len(), 1);
        assert_eq!(
            turns[0]["content"],
            json!([{ "text": "first" }, { "text": "second" }])
        );
    }

    #[test]
    fn leading_assistant_message_gets_synthetic_user_turn() {
        let messages = vec![Message::assistant("hello there")];
        let (_, turns) = messages_to_bedrock(&messages);
        assert_eq!(turns.len(), 2);
        assert_eq!(turns[0]["role"], "user");
        assert_eq!(turns[1]["role"], "assistant");
    }

    // endregion

    // region: tools_to_bedrock

    #[test]
    fn no_tools_yields_none() {
        assert_eq!(tools_to_bedrock(&ChatOptions::new()), None);
    }

    #[test]
    fn function_tool_becomes_tool_spec() {
        let tool = make_tool(
            ToolKind::Function,
            "get_weather",
            json!({ "type": "object", "properties": { "city": { "type": "string" } } }),
        );
        let options = ChatOptions::new().with_tool(tool);
        let out = tools_to_bedrock(&options).unwrap();
        assert_eq!(
            out,
            json!({
                "toolConfig": {
                    "tools": [{
                        "toolSpec": {
                            "name": "get_weather",
                            "description": "get_weather description",
                            "inputSchema": {
                                "json": { "type": "object", "properties": { "city": { "type": "string" } } }
                            },
                        }
                    }]
                }
            })
        );
    }

    #[test]
    fn hosted_tool_kinds_are_skipped() {
        let tool = make_tool(ToolKind::HostedWebSearch, "web_search", json!({}));
        let options = ChatOptions::new().with_tool(tool);
        assert_eq!(tools_to_bedrock(&options), None);
    }

    #[test]
    fn tool_choice_auto_maps_to_auto_object() {
        let tool = make_tool(ToolKind::Function, "f", json!({}));
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::auto());
        let out = tools_to_bedrock(&options).unwrap();
        assert_eq!(out["toolConfig"]["toolChoice"], json!({ "auto": {} }));
    }

    #[test]
    fn tool_choice_required_any_maps_to_any_object() {
        let tool = make_tool(ToolKind::Function, "f", json!({}));
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::required_any());
        let out = tools_to_bedrock(&options).unwrap();
        assert_eq!(out["toolConfig"]["toolChoice"], json!({ "any": {} }));
    }

    #[test]
    fn tool_choice_required_function_maps_to_named_tool() {
        let tool = make_tool(ToolKind::Function, "f", json!({}));
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::required_function("f"));
        let out = tools_to_bedrock(&options).unwrap();
        assert_eq!(
            out["toolConfig"]["toolChoice"],
            json!({ "tool": { "name": "f" } })
        );
    }

    // endregion

    // region: apply_inference_config

    #[test]
    fn apply_inference_config_maps_supported_fields() {
        let mut body = Map::new();
        let options = ChatOptions {
            temperature: Some(0.5),
            max_tokens: Some(256),
            top_p: Some(0.9),
            stop: Some(vec!["STOP".to_string()]),
            ..ChatOptions::new()
        };
        apply_inference_config(&mut body, &options);
        // `temperature`/`top_p` are `f32` on `ChatOptions`; compare against
        // `json!` of the same `f32` values rather than bare float literals
        // (which parse as `f64` and don't bit-for-bit match an `f32` widened
        // through `json!`, e.g. `0.9_f32` -> `0.8999999761581421_f64`).
        assert_eq!(
            body["inferenceConfig"],
            json!({
                "temperature": 0.5_f32,
                "maxTokens": 256,
                "topP": 0.9_f32,
                "stopSequences": ["STOP"],
            })
        );
    }

    #[test]
    fn apply_inference_config_omits_key_when_nothing_set() {
        let mut body = Map::new();
        apply_inference_config(&mut body, &ChatOptions::new());
        assert!(!body.contains_key("inferenceConfig"));
    }

    // endregion

    // region: parse_response

    #[test]
    fn parse_response_text_and_usage() {
        let value = json!({
            "output": { "message": { "role": "assistant", "content": [{ "text": "Hello!" }] } },
            "stopReason": "end_turn",
            "usage": { "inputTokens": 10, "outputTokens": 5, "totalTokens": 15 },
        });
        let resp = parse_response(&value);
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
            "output": { "message": { "role": "assistant", "content": [
                { "text": "Let me check." },
                { "toolUse": { "toolUseId": "call_1", "name": "get_weather", "input": { "city": "Paris" } } },
            ] } },
            "stopReason": "tool_use",
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
    fn map_stop_reason_covers_documented_mapping() {
        assert_eq!(map_stop_reason("end_turn"), FinishReason::stop());
        assert_eq!(map_stop_reason("stop_sequence"), FinishReason::stop());
        assert_eq!(
            map_stop_reason("max_tokens"),
            FinishReason::new(FinishReason::LENGTH)
        );
        assert_eq!(map_stop_reason("tool_use"), FinishReason::tool_calls());
        assert_eq!(
            map_stop_reason("content_filtered"),
            FinishReason::new(FinishReason::CONTENT_FILTER)
        );
        assert_eq!(
            map_stop_reason("guardrail_intervened"),
            FinishReason::new(FinishReason::CONTENT_FILTER)
        );
        assert_eq!(
            map_stop_reason("something_new"),
            FinishReason::new("something_new")
        );
    }

    #[test]
    fn parse_response_missing_fields_defaults_gracefully() {
        let resp = parse_response(&json!({}));
        assert!(resp.messages.is_empty());
        assert_eq!(resp.finish_reason, None);
        assert_eq!(resp.usage_details, None);
    }

    // endregion

    // region: build_request

    #[test]
    fn build_request_assembles_system_messages_inference_config_and_tools() {
        let tool = make_tool(ToolKind::Function, "f", json!({}));
        let options = ChatOptions::new()
            .with_instructions("Be terse.")
            .with_temperature(0.2)
            .with_tool(tool);
        let messages = vec![Message::user("hi")];
        let body = build_request(&messages, &options);
        assert_eq!(body["system"], json!([{ "text": "Be terse." }]));
        assert_eq!(body["messages"][0]["role"], "user");
        // `with_temperature` takes `f32`; compare against the same `f32`
        // widened through `json!`, not an `f64` literal (see the precision
        // note on `apply_inference_config_maps_supported_fields`).
        assert_eq!(body["inferenceConfig"]["temperature"], json!(0.2_f32));
        assert_eq!(body["toolConfig"]["tools"][0]["toolSpec"]["name"], "f");
    }

    #[test]
    fn build_request_instructions_prepend_explicit_system_messages() {
        let messages = vec![
            Message::system("From the conversation."),
            Message::user("hi"),
        ];
        let options = ChatOptions::new().with_instructions("From options.");
        let body = build_request(&messages, &options);
        assert_eq!(
            body["system"],
            json!([{ "text": "From options." }, { "text": "From the conversation." }])
        );
    }

    #[test]
    fn build_request_instructions_alone_populate_system() {
        let options = ChatOptions::new().with_instructions("Be terse.");
        let body = build_request(&[Message::user("hi")], &options);
        assert_eq!(body["system"], json!([{ "text": "Be terse." }]));
    }

    #[test]
    fn build_request_omits_absent_sections() {
        let body = build_request(&[Message::user("hi")], &ChatOptions::new());
        assert!(body.get("system").is_none());
        assert!(body.get("inferenceConfig").is_none());
        assert!(body.get("toolConfig").is_none());
    }

    #[test]
    fn build_request_merges_additional_properties() {
        let mut options = ChatOptions::new();
        let mut extra = HashMap::new();
        extra.insert(
            "guardrailConfig".to_string(),
            json!({ "guardrailIdentifier": "g1" }),
        );
        options.additional_properties = extra;
        let body = build_request(&[Message::user("hi")], &options);
        assert_eq!(
            body["guardrailConfig"],
            json!({ "guardrailIdentifier": "g1" })
        );
    }

    // endregion
}
