//! Conversion between framework types and the Azure AI Foundry (persistent
//! agents) wire format, plus parsing of run/message/step JSON.
//!
//! The Azure AI Agents data plane is Assistants-API-shaped: agents (a.k.a.
//! assistants), threads, messages, and runs, with tool calls surfaced as a
//! run's `required_action` and answered via `submit_tool_outputs`. These
//! helpers are pure so they can be unit-tested against fixture JSON without a
//! network.

use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    ChatMessage, ChatOptions, Content, FinishReason, FunctionArguments, FunctionCallContent, Role,
    ToolMode, UsageDetails,
};
use serde_json::{json, Map, Value};

/// The Entra ID scope (audience) for the Azure AI Foundry data plane.
pub const AI_FOUNDRY_SCOPE: &str = "https://ai.azure.com/.default";

// ---------------------------------------------------------------------------
// Tool-call id round-tripping
// ---------------------------------------------------------------------------

/// Encode a `(run_id, tool_call_id)` pair into the synthetic call id carried by
/// a [`FunctionCallContent`], mirroring the Python client's
/// `["<run_id>", "<tool_call_id>"]` JSON-array convention. The run id is needed
/// when the tool result is submitted back (`submit_tool_outputs` targets a run),
/// but only the bare `tool_call_id` is sent on the wire.
pub fn encode_call_id(run_id: &str, tool_call_id: &str) -> String {
    Value::Array(vec![json!(run_id), json!(tool_call_id)]).to_string()
}

/// Decode a synthetic call id produced by [`encode_call_id`] back into
/// `(run_id, tool_call_id)`. Returns `None` when the id is not the expected
/// two-element array of non-empty strings.
pub fn decode_call_id(call_id: &str) -> Option<(String, String)> {
    let v: Value = serde_json::from_str(call_id).ok()?;
    let arr = v.as_array()?;
    if arr.len() != 2 {
        return None;
    }
    let run = arr[0].as_str()?.to_string();
    let call = arr[1].as_str()?.to_string();
    if run.is_empty() || call.is_empty() {
        return None;
    }
    Some((run, call))
}

// ---------------------------------------------------------------------------
// Request-message preparation
// ---------------------------------------------------------------------------

/// The result of splitting a request's messages into the pieces the Azure AI
/// Agents API consumes separately.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct PreparedMessages {
    /// System/developer text folded into run instructions (Azure AI has no
    /// system message role).
    pub instructions: Option<String>,
    /// Thread messages to create before the run: `{"role","content":[blocks]}`.
    pub messages: Vec<Value>,
    /// Decoded tool outputs `(tool_call_id, output)` to submit to an active run.
    pub tool_outputs: Vec<(String, String)>,
    /// Decoded tool approvals `(tool_call_id, approved)` to submit.
    pub tool_approvals: Vec<(String, bool)>,
    /// The run id shared by the tool results, if any were present.
    pub run_id: Option<String>,
}

/// Serialize a tool result value the way `submit_tool_outputs` expects: a bare
/// string is sent verbatim, anything else is JSON-encoded.
fn result_to_output(result: &Option<Value>) -> String {
    match result {
        Some(Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
        None => String::new(),
    }
}

/// Split request messages into instructions, thread messages, and tool results.
///
/// * `system`/`developer` messages become instructions.
/// * `assistant` messages become `{"role":"assistant"}` thread messages;
///   everything else becomes `{"role":"user"}`.
/// * text becomes a `{"type":"text","text":...}` block; image data/URI content
///   becomes an `{"type":"image_url","image_url":{"url":...}}` block.
/// * function results / approval responses are pulled out as tool results
///   (decoded from their synthetic `[run_id, call_id]` call ids) rather than
///   added as messages.
pub fn prepare_messages(messages: &[ChatMessage]) -> PreparedMessages {
    let mut out = PreparedMessages::default();
    let mut instructions: Vec<String> = Vec::new();

    for msg in messages {
        let role = msg.role.as_str();
        if role == Role::SYSTEM || role == "developer" {
            for content in &msg.contents {
                if let Content::Text(t) = content {
                    instructions.push(t.text.clone());
                }
            }
            continue;
        }

        let mut blocks: Vec<Value> = Vec::new();
        for content in &msg.contents {
            match content {
                Content::Text(t) => blocks.push(json!({"type": "text", "text": t.text})),
                Content::Uri(u) if u.media_type.starts_with("image") => {
                    blocks.push(json!({"type": "image_url", "image_url": {"url": u.uri}}))
                }
                Content::Data(d) if is_image(d.media_type.as_deref()) => {
                    blocks.push(json!({"type": "image_url", "image_url": {"url": d.uri}}))
                }
                Content::FunctionResult(fr) => {
                    if let Some((run_id, call_id)) = decode_call_id(&fr.call_id) {
                        out.run_id.get_or_insert(run_id);
                        out.tool_outputs
                            .push((call_id, result_to_output(&fr.result)));
                    }
                }
                Content::FunctionApprovalResponse(ar) => {
                    if let Some((run_id, call_id)) = decode_call_id(&ar.id) {
                        out.run_id.get_or_insert(run_id);
                        out.tool_approvals.push((call_id, ar.approved));
                    }
                }
                _ => {}
            }
        }

        if !blocks.is_empty() {
            let wire_role = if role == Role::ASSISTANT {
                "assistant"
            } else {
                "user"
            };
            out.messages
                .push(json!({"role": wire_role, "content": blocks}));
        }
    }

    if !instructions.is_empty() {
        out.instructions = Some(instructions.join("\n"));
    }
    out
}

fn is_image(media_type: Option<&str>) -> bool {
    media_type.map(|m| m.starts_with("image")).unwrap_or(false)
}

// ---------------------------------------------------------------------------
// Tool definitions & tool_choice
// ---------------------------------------------------------------------------

/// Map framework [`ToolDefinition`]s to Azure AI tool payloads (Assistants
/// shape). Function tools become `{"type":"function","function":{…}}`; hosted
/// markers map to their service tool types.
pub fn tools_to_azure(tools: &[ToolDefinition]) -> Vec<Value> {
    tools.iter().map(tool_to_azure).collect()
}

fn tool_to_azure(tool: &ToolDefinition) -> Value {
    match &tool.kind {
        ToolKind::Function => json!({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
        }),
        ToolKind::HostedCodeInterpreter => json!({"type": "code_interpreter"}),
        ToolKind::HostedFileSearch { .. } => json!({"type": "file_search"}),
        ToolKind::HostedWebSearch => json!({"type": "bing_grounding"}),
        ToolKind::HostedMcp { url, allowed_tools } => {
            let mut mcp = Map::new();
            mcp.insert("type".into(), json!("mcp"));
            mcp.insert("server_label".into(), json!(tool.name.replace(' ', "_")));
            mcp.insert("server_url".into(), json!(url));
            if let Some(allowed) = allowed_tools {
                mcp.insert("allowed_tools".into(), json!(allowed));
            }
            Value::Object(mcp)
        }
    }
}

/// Map a [`ToolMode`] to an Azure AI `tool_choice` value: `"none"`, `"auto"`,
/// `"required"`, or a specific `{"type":"function","function":{"name":…}}`.
pub fn tool_choice_to_azure(mode: &ToolMode) -> Value {
    match mode {
        ToolMode::None => json!("none"),
        ToolMode::Auto => json!("auto"),
        ToolMode::Required(None) => json!("required"),
        ToolMode::Required(Some(name)) => {
            json!({"type": "function", "function": {"name": name}})
        }
    }
}

// ---------------------------------------------------------------------------
// Run / agent request bodies
// ---------------------------------------------------------------------------

/// Build the `POST …/threads/{id}/runs` body for `agent_id`, applying the
/// request `options` and `instructions`. `stream` toggles SSE.
pub fn build_run_body(
    agent_id: &str,
    model: Option<&str>,
    instructions: Option<&str>,
    options: &ChatOptions,
    stream: bool,
) -> Value {
    let mut body = Map::new();
    body.insert("assistant_id".into(), json!(agent_id));
    if let Some(m) = model {
        body.insert("model".into(), json!(m));
    }
    if let Some(instr) = instructions {
        body.insert("instructions".into(), json!(instr));
    }
    apply_common_run_fields(&mut body, options);

    // Tools + tool choice.
    let tool_choice = options.tool_choice.clone().unwrap_or(ToolMode::None);
    if tool_choice != ToolMode::None && !options.tools.is_empty() {
        body.insert("tools".into(), json!(tools_to_azure(&options.tools)));
    }
    if options.tool_choice.is_some() {
        body.insert("tool_choice".into(), tool_choice_to_azure(&tool_choice));
    }

    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

/// Build the `POST …/assistants` body to auto-create an agent.
pub fn build_agent_body(
    model: &str,
    name: Option<&str>,
    description: Option<&str>,
    instructions: Option<&str>,
    options: &ChatOptions,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    if let Some(n) = name {
        body.insert("name".into(), json!(n));
    }
    if let Some(d) = description {
        body.insert("description".into(), json!(d));
    }
    if let Some(instr) = instructions {
        body.insert("instructions".into(), json!(instr));
    }
    if !options.tools.is_empty() {
        body.insert("tools".into(), json!(tools_to_azure(&options.tools)));
    }
    if let Some(temp) = options.temperature {
        body.insert("temperature".into(), json!(temp));
    }
    if let Some(top_p) = options.top_p {
        body.insert("top_p".into(), json!(top_p));
    }
    if let Some(rf) = &options.response_format {
        body.insert("response_format".into(), response_format_to_azure(rf));
    }
    Value::Object(body)
}

fn apply_common_run_fields(body: &mut Map<String, Value>, options: &ChatOptions) {
    if let Some(t) = options.temperature {
        body.insert("temperature".into(), json!(t));
    }
    if let Some(p) = options.top_p {
        body.insert("top_p".into(), json!(p));
    }
    if let Some(m) = options.max_tokens {
        body.insert("max_completion_tokens".into(), json!(m));
    }
    if let Some(parallel) = options.allow_multiple_tool_calls {
        body.insert("parallel_tool_calls".into(), json!(parallel));
    }
    if let Some(rf) = &options.response_format {
        body.insert("response_format".into(), response_format_to_azure(rf));
    }
    if let Some(meta) = &options.metadata {
        body.insert("metadata".into(), json!(meta));
    }
}

fn response_format_to_azure(rf: &agent_framework_core::types::ResponseFormat) -> Value {
    // The core `ResponseFormat` already serializes to the OpenAI/Assistants
    // `response_format` object shape.
    serde_json::to_value(rf).unwrap_or(Value::Null)
}

/// Build the `submit_tool_outputs` body from decoded outputs/approvals.
pub fn build_submit_body(
    tool_outputs: &[(String, String)],
    tool_approvals: &[(String, bool)],
    stream: bool,
) -> Value {
    let mut body = Map::new();
    if !tool_outputs.is_empty() {
        let outputs: Vec<Value> = tool_outputs
            .iter()
            .map(|(id, output)| json!({"tool_call_id": id, "output": output}))
            .collect();
        body.insert("tool_outputs".into(), json!(outputs));
    }
    if !tool_approvals.is_empty() {
        let approvals: Vec<Value> = tool_approvals
            .iter()
            .map(|(id, approve)| json!({"tool_call_id": id, "approve": approve}))
            .collect();
        body.insert("tool_approvals".into(), json!(approvals));
    }
    if stream {
        body.insert("stream".into(), json!(true));
    }
    Value::Object(body)
}

// ---------------------------------------------------------------------------
// Response / run parsing
// ---------------------------------------------------------------------------

/// The `status` field of a run object (`"queued"`, `"in_progress"`,
/// `"requires_action"`, `"completed"`, `"failed"`, `"cancelled"`, `"expired"`).
pub fn run_status(run: &Value) -> Option<&str> {
    run.get("status").and_then(Value::as_str)
}

/// A run is terminal when it will not progress further without a new request.
pub fn is_terminal_status(status: &str) -> bool {
    matches!(
        status,
        "completed" | "failed" | "cancelled" | "expired" | "requires_action"
    )
}

/// Parse `usage: {prompt_tokens, completion_tokens, total_tokens}` into
/// [`UsageDetails`].
pub fn parse_usage(value: &Value) -> Option<UsageDetails> {
    let usage = value.get("usage").filter(|u| u.is_object())?;
    Some(UsageDetails {
        input_token_count: usage.get("prompt_tokens").and_then(Value::as_u64),
        output_token_count: usage.get("completion_tokens").and_then(Value::as_u64),
        total_token_count: usage.get("total_tokens").and_then(Value::as_u64),
        additional_counts: Default::default(),
    })
}

/// Build [`FunctionCallContent`]/approval contents from a run's
/// `required_action`, encoding each call id as `[run_id, tool_call_id]`.
pub fn required_action_contents(run: &Value, run_id: &str) -> Vec<Content> {
    let Some(required) = run.get("required_action") else {
        return Vec::new();
    };
    let action_type = required.get("type").and_then(Value::as_str).unwrap_or("");
    let mut out = Vec::new();

    if action_type == "submit_tool_outputs" {
        let calls = required
            .get("submit_tool_outputs")
            .and_then(|s| s.get("tool_calls"))
            .and_then(Value::as_array);
        for call in calls.into_iter().flatten() {
            if call.get("type").and_then(Value::as_str) != Some("function") {
                continue;
            }
            let tool_call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let func = call.get("function");
            let name = func
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            let args = func
                .and_then(|f| f.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("");
            out.push(Content::FunctionCall(FunctionCallContent::new(
                encode_call_id(run_id, tool_call_id),
                name,
                Some(FunctionArguments::Raw(args.to_string())),
            )));
        }
    } else if action_type == "submit_tool_approval" {
        let calls = required
            .get("submit_tool_approval")
            .and_then(|s| s.get("tool_calls"))
            .and_then(Value::as_array);
        for call in calls.into_iter().flatten() {
            let tool_call_id = call.get("id").and_then(Value::as_str).unwrap_or_default();
            let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
            let args = call.get("arguments").and_then(Value::as_str).unwrap_or("");
            let encoded = encode_call_id(run_id, tool_call_id);
            out.push(Content::FunctionApprovalRequest(
                agent_framework_core::types::FunctionApprovalRequestContent {
                    id: encoded.clone(),
                    function_call: FunctionCallContent::new(
                        encoded,
                        name,
                        Some(FunctionArguments::Raw(args.to_string())),
                    ),
                },
            ));
        }
    }
    out
}

/// The finish reason for a terminal run status.
pub fn finish_reason_for(status: &str) -> Option<FinishReason> {
    match status {
        "completed" => Some(FinishReason::stop()),
        "requires_action" => Some(FinishReason::tool_calls()),
        "cancelled" | "expired" => Some(FinishReason::new(status)),
        _ => None,
    }
}

/// Extract assistant text from a `…/threads/{id}/messages` list response,
/// concatenating the text of every content block of assistant-role messages
/// (the list is expected ordered oldest→newest).
pub fn assistant_text_from_messages(list: &Value) -> String {
    let Some(data) = list.get("data").and_then(Value::as_array) else {
        return String::new();
    };
    let mut out = String::new();
    for msg in data {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        for block in msg
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if block.get("type").and_then(Value::as_str) == Some("text") {
                if let Some(text) = block
                    .get("text")
                    .and_then(|t| t.get("value"))
                    .and_then(Value::as_str)
                {
                    out.push_str(text);
                }
            }
        }
    }
    out
}

/// The last-error message on a failed run.
pub fn last_error_message(run: &Value) -> String {
    run.get("last_error")
        .and_then(|e| e.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("run failed")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{hosted_code_interpreter, hosted_file_search, AiFunction};
    use agent_framework_core::types::FunctionResultContent;

    #[test]
    fn call_id_round_trips() {
        let encoded = encode_call_id("run_1", "call_9");
        assert_eq!(
            decode_call_id(&encoded),
            Some(("run_1".into(), "call_9".into()))
        );
    }

    #[test]
    fn decode_rejects_malformed_call_ids() {
        assert_eq!(decode_call_id("not-json"), None);
        assert_eq!(decode_call_id("[\"run\"]"), None);
        assert_eq!(decode_call_id("[\"\", \"call\"]"), None);
    }

    #[test]
    fn system_messages_become_instructions() {
        let prepared =
            prepare_messages(&[ChatMessage::system("be terse"), ChatMessage::user("hi")]);
        assert_eq!(prepared.instructions.as_deref(), Some("be terse"));
        assert_eq!(prepared.messages.len(), 1);
        assert_eq!(prepared.messages[0]["role"], json!("user"));
        assert_eq!(prepared.messages[0]["content"][0]["text"], json!("hi"));
    }

    #[test]
    fn function_results_are_pulled_out_as_tool_outputs() {
        let call_id = encode_call_id("run_7", "call_3");
        let msg = ChatMessage::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(json!("sunny")),
            ))],
        );
        let prepared = prepare_messages(&[msg]);
        assert!(prepared.messages.is_empty());
        assert_eq!(prepared.run_id.as_deref(), Some("run_7"));
        assert_eq!(
            prepared.tool_outputs,
            vec![("call_3".into(), "sunny".into())]
        );
    }

    #[test]
    fn run_body_includes_tools_only_when_choice_allows() {
        let tool = AiFunction::new(
            "get_weather",
            "Get weather",
            json!({"type": "object", "properties": {}}),
            |_| async { Ok(json!("ok")) },
        )
        .into_definition();
        let options = ChatOptions::new()
            .with_tool(tool)
            .with_tool_choice(ToolMode::Auto)
            .with_temperature(0.5);
        let body = build_run_body("asst_1", Some("gpt-4o"), Some("sys"), &options, true);
        assert_eq!(body["assistant_id"], json!("asst_1"));
        assert_eq!(body["model"], json!("gpt-4o"));
        assert_eq!(body["instructions"], json!("sys"));
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("get_weather"));
        assert_eq!(body["temperature"], json!(0.5));
    }

    #[test]
    fn run_body_omits_tools_when_choice_none() {
        let tool = hosted_code_interpreter();
        let options = ChatOptions::new().with_tool(tool); // tool_choice unset
        let body = build_run_body("asst_1", None, None, &options, false);
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn hosted_tools_map_to_service_types() {
        let tools = vec![hosted_code_interpreter(), hosted_file_search(Some(3))];
        let mapped = tools_to_azure(&tools);
        assert_eq!(mapped[0], json!({"type": "code_interpreter"}));
        assert_eq!(mapped[1], json!({"type": "file_search"}));
    }

    #[test]
    fn required_named_function_tool_choice() {
        let v = tool_choice_to_azure(&ToolMode::required_function("do_it"));
        assert_eq!(
            v,
            json!({"type": "function", "function": {"name": "do_it"}})
        );
    }

    #[test]
    fn required_action_builds_function_calls_with_encoded_ids() {
        let run = json!({
            "id": "run_42",
            "status": "requires_action",
            "required_action": {
                "type": "submit_tool_outputs",
                "submit_tool_outputs": {
                    "tool_calls": [
                        {"id": "call_a", "type": "function",
                         "function": {"name": "get_weather", "arguments": "{\"loc\":\"NYC\"}"}}
                    ]
                }
            }
        });
        let contents = required_action_contents(&run, "run_42");
        assert_eq!(contents.len(), 1);
        let Content::FunctionCall(fc) = &contents[0] else {
            panic!("expected function call");
        };
        assert_eq!(fc.name, "get_weather");
        assert_eq!(
            decode_call_id(&fc.call_id),
            Some(("run_42".into(), "call_a".into()))
        );
    }

    #[test]
    fn usage_and_message_text_parsing() {
        let run =
            json!({"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}});
        let usage = parse_usage(&run).unwrap();
        assert_eq!(usage.total_token_count, Some(14));

        let list = json!({"data": [
            {"role": "user", "content": [{"type": "text", "text": {"value": "hi"}}]},
            {"role": "assistant", "content": [{"type": "text", "text": {"value": "hello there"}}]},
        ]});
        assert_eq!(assistant_text_from_messages(&list), "hello there");
    }

    #[test]
    fn submit_body_shapes_outputs_and_approvals() {
        let body = build_submit_body(
            &[("call_1".into(), "42".into())],
            &[("call_2".into(), true)],
            true,
        );
        assert_eq!(
            body["tool_outputs"][0],
            json!({"tool_call_id": "call_1", "output": "42"})
        );
        assert_eq!(
            body["tool_approvals"][0],
            json!({"tool_call_id": "call_2", "approve": true})
        );
        assert_eq!(body["stream"], json!(true));
    }
}
