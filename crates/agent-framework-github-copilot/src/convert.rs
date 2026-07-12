//! Conversion between framework types and GitHub Copilot's OpenAI-compatible
//! `/chat/completions` wire format.
//!
//! Once authenticated with an exchanged Copilot API token, GitHub Copilot's
//! chat endpoint speaks the *exact* same JSON shapes as OpenAI's Chat
//! Completions API for everything this crate needs: `messages`,
//! `tools`/`tool_choice`, the scalar sampling options, the non-streaming
//! response envelope, `usage`, and the `chat.completion.chunk` SSE delta
//! shape. So request-body assembly and response/usage parsing are reused
//! verbatim from [`agent_framework_openai::convert`] rather than duplicated
//! (mirrors how `agent-framework-ollama` and `agent-framework-foundry-local`
//! reuse the same module, and for the same reason). Only the streaming-delta
//! parser is reimplemented locally, kept small on purpose, so it can be
//! unit-tested here over fixture JSON without going through a live
//! `reqwest::Response`.
//!
//! This module has nothing to do with the GitHub OAuth-token-to-Copilot-token
//! exchange (see the crate root for that) — it only converts chat
//! request/response bodies, after authentication has already produced a
//! usable Copilot bearer token.

use std::collections::HashMap;

use agent_framework_core::types::{
    ChatOptions, Content, FinishReason, FunctionArguments, FunctionCallContent, Message, Role,
    TextContent, UsageContent,
};
use serde_json::{json, Map, Value};

/// Build the `/chat/completions` request body.
///
/// `model` is the effective model id (already resolved from
/// `options.model` or the client default). Message/tool/option conversion is
/// delegated to [`agent_framework_openai::convert`] since GitHub Copilot's
/// chat endpoint accepts the identical shapes.
pub fn build_request(
    messages: &[Message],
    options: &ChatOptions,
    model: &str,
    stream: bool,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert(
        "messages".into(),
        json!(agent_framework_openai::convert::messages_to_openai(
            messages
        )),
    );
    agent_framework_openai::convert::apply_options(&mut body, options);
    let (tools, tool_choice) = agent_framework_openai::convert::tools_to_openai(options);
    if let Some(tools) = tools {
        body.insert("tools".into(), tools);
    }
    if let Some(choice) = tool_choice {
        body.insert("tool_choice".into(), choice);
    }
    if stream {
        body.insert("stream".into(), json!(true));
        // Copilot's chat endpoint honors `stream_options.include_usage` the
        // same way OpenAI does: the final chunk carries a top-level `usage`
        // object.
        body.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    Value::Object(body)
}

/// Parse one streamed `chat.completion.chunk` value into an update, resolving
/// tool-call ids from the index map (a streamed tool-call delta only carries
/// its `id` on the first chunk; later chunks reference it by `index`).
///
/// A near-identical shape to `agent-framework-openai`'s private `parse_delta`
/// (Copilot's SSE chunks are byte-for-byte the same shape) — reimplemented
/// locally, rather than reused, purely so it is directly unit-testable here
/// over fixture JSON without a live HTTP response.
pub fn parse_delta(
    value: &Value,
    tool_ids: &mut HashMap<i64, String>,
) -> Option<agent_framework_core::types::ChatResponseUpdate> {
    let mut update = agent_framework_core::types::ChatResponseUpdate {
        response_id: value.get("id").and_then(Value::as_str).map(String::from),
        model: value.get("model").and_then(Value::as_str).map(String::from),
        ..Default::default()
    };

    let mut contents: Vec<Content> = Vec::new();

    if let Some(choice) = value
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first())
    {
        if let Some(delta) = choice.get("delta") {
            if let Some(r) = delta.get("role").and_then(Value::as_str) {
                update.role = Some(Role::new(r));
            }
            if let Some(text) = delta.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    contents.push(Content::Text(TextContent::new(text)));
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    let index = call.get("index").and_then(Value::as_i64).unwrap_or(0);
                    let chunk_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
                    let id = if chunk_id.is_empty() {
                        tool_ids.get(&index).cloned().unwrap_or_default()
                    } else {
                        tool_ids.insert(index, chunk_id.to_string());
                        chunk_id.to_string()
                    };
                    let func = call.get("function");
                    let name = func
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let args = func
                        .and_then(|f| f.get("arguments"))
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    contents.push(Content::FunctionCall(FunctionCallContent::new(
                        id,
                        name,
                        Some(FunctionArguments::Raw(args)),
                    )));
                }
            }
        }
        if let Some(fr) = choice.get("finish_reason").and_then(Value::as_str) {
            update.finish_reason = Some(FinishReason::new(fr));
        }
    }

    // The final chunk (with `stream_options.include_usage`) carries top-level
    // `usage` and no choices.
    if let Some(usage) = value.get("usage").filter(|u| u.is_object()) {
        contents.push(Content::Usage(UsageContent {
            details: agent_framework_openai::convert::parse_usage(usage),
        }));
    }

    if update.role.is_none() {
        update.role = Some(Role::assistant());
    }
    update.contents = contents;
    Some(update)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
    use agent_framework_core::types::ToolMode;

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

    // region: request building

    #[test]
    fn build_request_includes_model_and_messages() {
        let body = build_request(&[Message::user("hi")], &ChatOptions::new(), "gpt-4o", false);
        assert_eq!(body["model"], json!("gpt-4o"));
        assert_eq!(
            body["messages"],
            json!([{ "role": "user", "content": "hi" }])
        );
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn build_request_applies_temperature_and_max_tokens() {
        let mut options = ChatOptions::new();
        options.temperature = Some(0.4);
        options.max_tokens = Some(256);
        let body = build_request(&[Message::user("hi")], &options, "gpt-4o", false);
        assert_eq!(body["temperature"], json!(0.4_f32));
        assert_eq!(body["max_tokens"], json!(256));
    }

    #[test]
    fn build_request_sets_stream_and_include_usage() {
        let body = build_request(&[Message::user("hi")], &ChatOptions::new(), "gpt-4o", true);
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({ "include_usage": true }));
    }

    #[test]
    fn build_request_includes_tools_and_tool_choice() {
        let mut options = ChatOptions::new();
        options.tools = vec![function_tool("get_weather")];
        options.tool_choice = Some(ToolMode::Auto);
        let body = build_request(&[Message::user("weather?")], &options, "gpt-4o", false);
        assert_eq!(body["tools"][0]["function"]["name"], json!("get_weather"));
        assert_eq!(body["tool_choice"], json!("auto"));
    }

    // endregion

    // region: response / usage parsing (reused from agent-framework-openai,
    // exercised here against Copilot-shaped fixtures)

    #[test]
    fn parses_copilot_chat_completion_response() {
        let value = json!({
            "id": "chatcmpl-123",
            "model": "gpt-4o",
            "choices": [{
                "message": { "role": "assistant", "content": "Hello there!" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16 },
        });
        let resp = agent_framework_openai::convert::parse_response(&value);
        assert_eq!(resp.text(), "Hello there!");
        assert_eq!(resp.response_id.as_deref(), Some("chatcmpl-123"));
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(12));
        assert_eq!(usage.output_token_count, Some(4));
        assert_eq!(usage.total_token_count, Some(16));
    }

    #[test]
    fn parses_copilot_tool_call_response() {
        let value = json!({
            "id": "chatcmpl-456",
            "model": "gpt-4o",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" },
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        });
        let resp = agent_framework_openai::convert::parse_response(&value);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(resp.finish_reason, Some(FinishReason::new("tool_calls")));
    }

    // endregion

    // region: streaming delta parsing

    #[test]
    fn parse_delta_accumulates_text() {
        let mut tool_ids = HashMap::new();
        let value = json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{ "delta": { "role": "assistant", "content": "Hi" }, "finish_reason": null }],
        });
        let update = parse_delta(&value, &mut tool_ids).unwrap();
        assert_eq!(update.response_id.as_deref(), Some("chatcmpl-1"));
        assert_eq!(update.role, Some(Role::assistant()));
        match &update.contents[0] {
            Content::Text(t) => assert_eq!(t.text, "Hi"),
            other => panic!("expected text content, got {other:?}"),
        }
    }

    #[test]
    fn parse_delta_resolves_tool_call_id_across_chunks() {
        let mut tool_ids = HashMap::new();
        let first = json!({
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "id": "call_1", "function": { "name": "get_weather", "arguments": "{\"ci" } }] },
            }],
        });
        let update1 = parse_delta(&first, &mut tool_ids).unwrap();
        let Content::FunctionCall(fc1) = &update1.contents[0] else {
            panic!("expected function call content");
        };
        assert_eq!(fc1.call_id, "call_1");
        assert_eq!(fc1.name, "get_weather");

        // Continuation chunk carries only the index, no id/name.
        let second = json!({
            "choices": [{
                "delta": { "tool_calls": [{ "index": 0, "function": { "arguments": "ty\":\"Paris\"}" } }] },
            }],
        });
        let update2 = parse_delta(&second, &mut tool_ids).unwrap();
        let Content::FunctionCall(fc2) = &update2.contents[0] else {
            panic!("expected function call content");
        };
        assert_eq!(fc2.call_id, "call_1");
        assert_eq!(fc2.name, "");
    }

    #[test]
    fn parse_delta_finish_reason_and_trailing_usage_chunk() {
        let mut tool_ids = HashMap::new();
        let value = json!({
            "choices": [{ "delta": {}, "finish_reason": "stop" }],
        });
        let update = parse_delta(&value, &mut tool_ids).unwrap();
        assert_eq!(update.finish_reason, Some(FinishReason::stop()));

        let usage_only = json!({
            "id": "chatcmpl-1",
            "usage": { "prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8 },
        });
        let update = parse_delta(&usage_only, &mut tool_ids).unwrap();
        match &update.contents[0] {
            Content::Usage(u) => {
                assert_eq!(u.details.input_token_count, Some(5));
                assert_eq!(u.details.output_token_count, Some(3));
            }
            other => panic!("expected usage content, got {other:?}"),
        }
    }

    #[test]
    fn parse_delta_defaults_role_to_assistant_when_absent() {
        let mut tool_ids = HashMap::new();
        let value = json!({ "choices": [{ "delta": { "content": "x" } }] });
        let update = parse_delta(&value, &mut tool_ids).unwrap();
        assert_eq!(update.role, Some(Role::assistant()));
    }

    // endregion
}
