//! Conversion between framework types and the Mistral chat-completions wire
//! format.
//!
//! Mistral's `POST /v1/chat/completions` endpoint is wire-compatible with
//! OpenAI's Chat Completions API for the parts both providers share (message
//! shape, function-tool shape, tool-call/response shape, token usage shape),
//! so message conversion, tool conversion, and response/usage parsing are all
//! reused verbatim from [`agent_framework_openai::convert`] — the same
//! approach `agent-framework-azure` takes for Azure OpenAI. Request-option
//! handling is *not* fully shared, because Mistral's supported option set
//! differs from OpenAI's in real ways (see [`apply_options`]), so that part
//! is implemented here instead of reused.

use agent_framework_core::error::Error;
use agent_framework_core::types::{ChatOptions, ChatResponse};
use agent_framework_openai::convert as oai;
use serde_json::{json, Map, Value};

/// Build a full Mistral `POST /v1/chat/completions` request body.
pub fn build_request(
    messages: &[agent_framework_core::types::Message],
    options: &ChatOptions,
    model: &str,
    stream: bool,
) -> Value {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    // Message shape (text/image parts, tool_calls, tool results) is
    // identical to OpenAI's, so this is reused verbatim.
    body.insert("messages".into(), json!(oai::messages_to_openai(messages)));

    apply_options(&mut body, options);

    // Function-tool shape (`{"type":"function","function":{...}}`) and
    // `tool_choice` (`"auto"`/`"none"`/`"required"`/named) are identical to
    // OpenAI's; hosted-tool kinds (web search, code interpreter, ...) are not
    // supported by the Mistral Chat Completions API and are skipped with a
    // warning by `tools_to_openai` itself, which is exactly the "skip
    // unsupported gracefully" behavior wanted here too.
    let (tools, tool_choice) = oai::tools_to_openai(options);
    if let Some(tools) = tools {
        body.insert("tools".into(), tools);
    }
    if let Some(choice) = tool_choice {
        body.insert("tool_choice".into(), choice);
    }

    if stream {
        body.insert("stream".into(), json!(true));
        // Mirrors OpenAI's `stream_options.include_usage`: Mistral's
        // streaming Chat Completions API accepts the same field and, like
        // OpenAI, emits a final usage-only chunk when it is set.
        body.insert("stream_options".into(), json!({ "include_usage": true }));
    }
    Value::Object(body)
}

/// Apply the scalar [`ChatOptions`] fields Mistral's Chat Completions API
/// actually supports onto a request body map.
///
/// Deliberately narrower than [`agent_framework_openai::convert::apply_options`]:
/// * `temperature`, `top_p`, `max_tokens`, `stop`, `frequency_penalty`,
///   `presence_penalty`, `response_format`, and `tool_choice`-adjacent
///   `parallel_tool_calls` map straight across (documented Mistral request
///   fields).
/// * `seed` maps to Mistral's differently-named `random_seed` field rather
///   than OpenAI's `seed`.
/// * `store`, `metadata`, `logit_bias`, and `user` have no Mistral Chat
///   Completions equivalent and are skipped gracefully (never sent) instead
///   of being forwarded as unrecognized fields.
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
    // Mistral's random-seed parameter is named `random_seed`, not `seed`.
    set!("random_seed", options.seed);
    if let Some(stop) = &options.stop {
        body.insert("stop".into(), json!(stop));
    }
    if let Some(fmt) = &options.response_format {
        // `ResponseFormat` serializes to `{"type":"text"|"json_object"}` or
        // `{"type":"json_schema","json_schema":{...}}`, which is exactly the
        // shape Mistral's `response_format` field expects.
        body.insert("response_format".into(), json!(fmt));
    }
    // Only meaningful alongside function tools, exactly like OpenAI's.
    if let Some(allow) = options.allow_multiple_tool_calls {
        if options
            .tools
            .iter()
            .any(|t| t.kind == agent_framework_core::tools::ToolKind::Function)
        {
            body.insert("parallel_tool_calls".into(), json!(allow));
        }
    }
    // `store`, `metadata`, `logit_bias`, `user`: no Mistral Chat Completions
    // equivalent -- intentionally not forwarded.
    for (k, v) in &options.additional_properties {
        body.entry(k.clone()).or_insert_with(|| v.clone());
    }
}

/// Parse a full (non-streaming) Mistral chat-completion response.
///
/// The response shape (`id`, `model`, `choices[0].message.{content,tool_calls}`,
/// `choices[0].finish_reason`, `usage.{prompt_tokens,completion_tokens,total_tokens}`)
/// is identical to OpenAI's, so parsing is reused verbatim.
pub fn parse_response(value: &Value) -> ChatResponse {
    oai::parse_response(value)
}

/// Classify a non-success Mistral Chat Completions API HTTP response into a
/// granular [`Error`].
///
/// Mistral's documented error body is `{"object":"error","message":...,
/// "type":...,"param":...,"code":...}` (OpenAI-shaped but without a nested
/// `error` wrapper). This maps by HTTP status only:
///
/// * `401` / `403` -> [`Error::ServiceInvalidAuth`]
/// * `400` / `422` -> [`Error::ServiceInvalidRequest`]
/// * anything else — notably `408` / `429` / `5xx`, which the retry layer
///   depends on — -> [`Error::ServiceStatus`], unchanged
///
/// Unlike OpenAI, Mistral has no documented content-filter-specific error
/// marker to key off of (no `code: "content_filter"` convention), so this
/// never constructs [`Error::ServiceContentFilter`] -- don't invent one.
pub fn classify_mistral_error(
    status: u16,
    message: impl Into<String>,
    retry_after: Option<f64>,
) -> Error {
    let message = message.into();
    match status {
        401 | 403 => Error::service_invalid_auth(message),
        400 | 422 => Error::service_invalid_request(message),
        _ => Error::service_status(status, message, retry_after),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::tools::{ApprovalMode, ToolDefinition, ToolKind};
    use agent_framework_core::types::{Message, ResponseFormat, ToolMode};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    // region: request building

    #[test]
    fn build_request_simple_text() {
        let body = build_request(
            &[user("Hello there")],
            &ChatOptions::new(),
            "mistral-large-latest",
            false,
        );
        assert_eq!(
            body,
            json!({
                "model": "mistral-large-latest",
                "messages": [{ "role": "user", "content": "Hello there" }],
            })
        );
    }

    #[test]
    fn build_request_temperature_max_tokens_top_p() {
        let mut options = ChatOptions::new()
            .with_temperature(0.4)
            .with_max_tokens(256);
        options.top_p = Some(0.8);
        let body = build_request(&[user("hi")], &options, "mistral-small-latest", false);
        assert_eq!(body["temperature"], json!(0.4_f32));
        assert_eq!(body["max_tokens"], json!(256));
        assert_eq!(body["top_p"], json!(0.8_f32));
    }

    #[test]
    fn build_request_seed_maps_to_random_seed() {
        let mut options = ChatOptions::new();
        options.seed = Some(42);
        let body = build_request(&[user("hi")], &options, "mistral-small-latest", false);
        assert_eq!(body["random_seed"], json!(42));
        assert!(body.get("seed").is_none());
    }

    #[test]
    fn build_request_unsupported_options_are_skipped_gracefully() {
        let mut options = ChatOptions::new();
        options.store = Some(true);
        options.user = Some("user-123".into());
        options.metadata = Some(std::collections::HashMap::from([(
            "k".to_string(),
            "v".to_string(),
        )]));
        let body = build_request(&[user("hi")], &options, "mistral-small-latest", false);
        assert!(body.get("store").is_none());
        assert!(body.get("user").is_none());
        assert!(body.get("metadata").is_none());
    }

    #[test]
    fn build_request_stop_and_penalties() {
        let mut options = ChatOptions::new();
        options.stop = Some(vec!["STOP".into()]);
        options.frequency_penalty = Some(0.1);
        options.presence_penalty = Some(0.2);
        let body = build_request(&[user("hi")], &options, "mistral-small-latest", false);
        assert_eq!(body["stop"], json!(["STOP"]));
        assert_eq!(body["frequency_penalty"], json!(0.1_f32));
        assert_eq!(body["presence_penalty"], json!(0.2_f32));
    }

    #[test]
    fn build_request_stream_flag_includes_usage_option() {
        let body = build_request(
            &[user("hi")],
            &ChatOptions::new(),
            "mistral-small-latest",
            true,
        );
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["stream_options"], json!({ "include_usage": true }));
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
        let body = build_request(&[user("hi")], &options, "mistral-large-latest", false);
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tools"][0]["function"]["name"], json!("get_weather"));
        assert_eq!(
            body["tool_choice"],
            json!({ "type": "function", "function": { "name": "get_weather" } })
        );
    }

    #[test]
    fn build_request_hosted_tools_are_skipped_gracefully() {
        let tool = ToolDefinition {
            name: "web_search".into(),
            description: String::new(),
            parameters: json!({}),
            kind: ToolKind::HostedWebSearch,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        };
        let options = ChatOptions::new().with_tool(tool);
        let body = build_request(&[user("hi")], &options, "mistral-large-latest", false);
        // No Mistral wire equivalent for hosted web search: dropped rather
        // than sent as a bogus `tools[]` entry or erroring the request.
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn build_request_parallel_tool_calls_only_with_function_tools() {
        let mut options = ChatOptions::new();
        options.allow_multiple_tool_calls = Some(false);
        let body = build_request(&[user("hi")], &options, "mistral-large-latest", false);
        assert!(body.get("parallel_tool_calls").is_none());

        let tool = ToolDefinition {
            name: "get_weather".into(),
            description: String::new(),
            parameters: json!({}),
            kind: ToolKind::Function,
            approval_mode: ApprovalMode::NeverRequire,
            executor: None,
        };
        let mut options2 = ChatOptions::new().with_tool(tool);
        options2.allow_multiple_tool_calls = Some(false);
        let body2 = build_request(&[user("hi")], &options2, "mistral-large-latest", false);
        assert_eq!(body2["parallel_tool_calls"], json!(false));
    }

    #[test]
    fn build_request_response_format_json_object() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::JsonObject);
        let body = build_request(&[user("hi")], &options, "mistral-large-latest", false);
        assert_eq!(body["response_format"], json!({ "type": "json_object" }));
    }

    #[test]
    fn build_request_response_format_json_schema() {
        let mut options = ChatOptions::new();
        options.response_format = Some(ResponseFormat::json_schema(
            "Person",
            json!({ "type": "object", "properties": { "name": { "type": "string" } } }),
        ));
        let body = build_request(&[user("hi")], &options, "mistral-large-latest", false);
        assert_eq!(body["response_format"]["type"], json!("json_schema"));
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            json!("Person")
        );
    }

    // endregion

    // region: response parsing (reuses agent-framework-openai::convert verbatim)

    #[test]
    fn parse_response_text_and_usage() {
        let value = json!({
            "id": "cmpl-123",
            "model": "mistral-large-latest",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop",
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 },
        });
        let resp = parse_response(&value);
        assert_eq!(resp.response_id.as_deref(), Some("cmpl-123"));
        assert_eq!(resp.text(), "Hello!");
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::stop())
        );
        let usage = resp.usage_details.unwrap();
        assert_eq!(usage.input_token_count, Some(10));
        assert_eq!(usage.output_token_count, Some(5));
        assert_eq!(usage.total_token_count, Some(15));
    }

    #[test]
    fn parse_response_tool_call() {
        let value = json!({
            "id": "cmpl-124",
            "model": "mistral-large-latest",
            "choices": [{
                "index": 0,
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
        let resp = parse_response(&value);
        assert_eq!(
            resp.finish_reason,
            Some(agent_framework_core::types::FinishReason::tool_calls())
        );
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].call_id, "call_1");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            calls[0].parse_arguments().unwrap().get("city").unwrap(),
            &json!("Paris")
        );
    }

    // endregion

    // region: classify_mistral_error

    #[test]
    fn classifies_401_and_403_as_invalid_auth() {
        for status in [401, 403] {
            let err = classify_mistral_error(status, format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidAuth { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn classifies_400_and_422_as_invalid_request() {
        for status in [400, 422] {
            let err = classify_mistral_error(status, format!("err {status}"), None);
            assert!(
                matches!(err, Error::ServiceInvalidRequest { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    #[test]
    fn leaves_retryable_statuses_as_service_status() {
        for status in [408, 429, 500, 503] {
            let err = classify_mistral_error(status, format!("err {status}"), Some(2.0));
            assert_eq!(err.status(), Some(status), "{err:?}");
            assert_eq!(err.retry_after(), Some(2.0), "{err:?}");
        }
    }

    #[test]
    fn never_produces_content_filter() {
        // Mistral has no documented content-filter-specific HTTP error
        // marker, so this classification path must never invent one.
        for status in [400, 401, 403, 404, 422, 429, 500] {
            let err = classify_mistral_error(status, "err", None);
            assert!(
                !matches!(err, Error::ServiceContentFilter { .. }),
                "status {status}: {err:?}"
            );
        }
    }

    // endregion
}
