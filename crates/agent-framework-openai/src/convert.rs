//! Conversion between framework types and the OpenAI chat-completions wire
//! format.

use agent_framework_core::types::{
    ChatMessage, ChatOptions, ChatResponse, Content, FinishReason, FunctionArguments,
    FunctionCallContent, FunctionResultContent, Role, TextContent, ToolMode, UsageDetails,
};
use serde_json::{json, Map, Value};

/// Convert framework messages into the OpenAI `messages` array.
pub fn messages_to_openai(messages: &[ChatMessage]) -> Vec<Value> {
    let mut out = Vec::with_capacity(messages.len());
    for msg in messages {
        let role = msg.role.as_str();
        // Collect text and any tool calls / results.
        let mut text = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        let mut tool_results: Vec<(&str, &FunctionResultContent)> = Vec::new();

        for content in &msg.contents {
            match content {
                Content::Text(t) => text.push_str(&t.text),
                Content::FunctionCall(fc) => tool_calls.push(function_call_to_openai(fc)),
                Content::FunctionResult(fr) => tool_results.push((&msg.role.0, fr)),
                _ => {}
            }
        }

        // A tool-role message maps each result to its own OpenAI `tool` message.
        if role == Role::TOOL {
            for (_, fr) in tool_results {
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
        if !text.is_empty() || tool_calls.is_empty() {
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

fn result_to_string(fr: &FunctionResultContent) -> String {
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
    let args = match &fc.arguments {
        Some(FunctionArguments::Raw(s)) => s.clone(),
        Some(FunctionArguments::Object(m)) => {
            serde_json::to_string(m).unwrap_or_else(|_| "{}".into())
        }
        None => "{}".into(),
    };
    json!({
        "id": fc.call_id,
        "type": "function",
        "function": { "name": fc.name, "arguments": args }
    })
}

/// Build the tools array and tool_choice for a request.
pub fn tools_to_openai(options: &ChatOptions) -> (Option<Value>, Option<Value>) {
    if options.tools.is_empty() {
        return (None, None);
    }
    let tools: Vec<Value> = options.tools.iter().map(|t| t.to_openai_spec()).collect();
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
        body.insert(
            "response_format".into(),
            json!({ "type": "json_schema", "json_schema": fmt }),
        );
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
            if let Some(text) = msg.get("content").and_then(Value::as_str) {
                if !text.is_empty() {
                    contents.push(Content::Text(TextContent::new(text)));
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

pub(crate) fn parse_usage(usage: &Value) -> UsageDetails {
    UsageDetails {
        input_token_count: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("completion_tokens").and_then(Value::as_u64),
        total_token_count: usage.get("total_tokens").and_then(Value::as_u64),
        additional_counts: Default::default(),
    }
}
