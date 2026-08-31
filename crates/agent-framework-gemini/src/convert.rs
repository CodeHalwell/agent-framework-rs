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
    // `call_id -> thoughtSignature`, accumulated as the conversation is
    // emitted. A replayed call always follows the turn that issued it, so a
    // single forward pass is enough to sign it — and building the map here,
    // from the same walk that applies the pairing rules, is what keeps the
    // fallback from contradicting them. See `message_contents_to_parts`.
    let mut thought_signatures: HashMap<String, String> = HashMap::new();
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
        let parts: Vec<Value> =
            message_contents_to_parts(&msg.contents, &call_names, &mut thought_signatures);
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

/// Convert one message's contents into Gemini `Part`s.
///
/// Reasoning content is **not** sent back as a part. Gemini 3 instead uses a
/// `thoughtSignature` that must be echoed when a function call is replayed, or
/// the follow-up turn is rejected. Two placements exist and both are handled:
///
/// * **On the function-call part itself** — Gemini 3's usual placement, carried
///   through on [`FunctionCallContent::protected_data`]. This wins.
/// * **On a preceding thought part** — carried on
///   [`TextReasoningContent::protected_data`] and used only to *backfill* a
///   call that has none of its own, mirroring upstream's "backfill only when
///   the raw Part lacks one".
///
/// A backfilled signature applies only to a function call *immediately*
/// following its reasoning content: any other **wire-visible** content in
/// between clears it, so a stale signature is never stamped onto an unrelated
/// call. Content that emits no Part at all does *not* clear it — an approval
/// request or response sitting between a carrier and its call is precisely the
/// case that must keep the pairing (upstream #7546).
///
/// When neither the call nor an adjacent carrier supplies one,
/// `thought_signatures` is the last resort: a `call_id -> signature` map of
/// every call this conversation has already signed. That is what makes a call
/// replayed through an approval round trip — arriving in a later message with
/// no carrier beside it — still reach the wire signed.
///
/// The map is threaded through and written here, rather than gathered by a
/// separate pre-pass, so that the rules above have exactly one implementation.
/// A pre-pass has to restate them, and a restatement that drifts is worse than
/// no fallback at all: the map is consulted *after* adjacency has deliberately
/// declined to sign a call, so a map built by laxer rules would silently
/// override that decision and re-sign calls this function just refused.
///
/// Mirrors upstream's `_convert_message_contents` (`_chat_client.py:660`).
fn message_contents_to_parts(
    contents: &[Content],
    call_names: &HashMap<String, String>,
    thought_signatures: &mut HashMap<String, String>,
) -> Vec<Value> {
    let mut parts = Vec::with_capacity(contents.len());
    let mut pending_signature: Option<&str> = None;
    for content in contents {
        if let Content::TextReasoning(t) = content {
            // Keep the last non-empty signature across a run of consecutive
            // reasoning parts: they are one thought block, and only some carry
            // a signature. Overwriting unconditionally let an unsigned part
            // that merely follows a signed one erase it. Non-reasoning content
            // still clears it below, so a signature never reaches an unrelated
            // call. This matters because `parse_response` does not coalesce
            // adjacent reasoning content the way streaming aggregation does.
            if let Some(signature) = t.protected_data.as_deref().filter(|s| !s.is_empty()) {
                pending_signature = Some(signature);
            }
            continue;
        }
        let Some(mut part) = content_to_part(content, call_names) else {
            // Emits no Part, so it is not "content in between" as far as the
            // wire is concerned: hold the signature rather than clearing it.
            // An approval response between a carrier and its call used to
            // drop the signature here, failing the replayed turn with a 400.
            continue;
        };
        let pending = pending_signature.take();
        if let Content::FunctionCall(fc) = content {
            // The call's own signature wins; the preceding reasoning content's
            // only backfills when the call carries none (mirroring upstream's
            // "backfill only when the raw Part lacks one"). Falling back to the
            // conversation-wide map last keeps a replayed call signed even when
            // its carrier is no longer beside it.
            let signature = fc
                .protected_data
                .as_deref()
                .filter(|s| !s.is_empty())
                .or(pending)
                .map(str::to_owned)
                .or_else(|| thought_signatures.get(&fc.call_id).cloned());
            if let Some(signature) = signature {
                if let Some(obj) = part.as_object_mut() {
                    obj.insert("thoughtSignature".into(), json!(signature));
                }
                // Remember it for a later replay of this same call, which
                // arrives with no carrier of its own. Recording only what was
                // actually signed here is what keeps the fallback in step with
                // the adjacency rules: a call this pass declined to sign
                // contributes nothing, so it cannot be signed later either.
                if !fc.call_id.is_empty() {
                    thought_signatures.insert(fc.call_id.clone(), signature);
                }
            }
        }
        parts.push(part);
    }
    parts
}

fn content_to_part(content: &Content, call_names: &HashMap<String, String>) -> Option<Value> {
    match content {
        Content::Text(t) => Some(json!({ "text": t.text })),
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
            out.push(Content::FunctionCall(
                FunctionCallContent::new(call_id, name, Some(FunctionArguments::Object(args)))
                    // Gemini 3's usual placement for `thoughtSignature` is the
                    // part carrying the call itself; a signature on a preceding
                    // thought part is the fallback, not the norm.
                    .with_protected_data(
                        part.get("thoughtSignature")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    ),
            ));
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
                    // Gemini 3 pairs a thought part with a `thoughtSignature`
                    // that must be echoed on the function call this reasoning
                    // produced; keep it on the reasoning content.
                    protected_data: part
                        .get("thoughtSignature")
                        .and_then(Value::as_str)
                        .map(str::to_string),
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
///
/// `FINISH_REASON_UNSPECIFIED` — the proto3 default, meaning the field was
/// never set — is treated exactly like an absent reason, so it takes the
/// `has_call` path rather than surfacing as a finish reason of its own.
fn finalize_finish_reason(raw: Option<&str>, contents: &[Content]) -> Option<FinishReason> {
    let has_call = contents
        .iter()
        .any(|c| matches!(c, Content::FunctionCall(_)));
    match raw.and_then(map_finish_reason) {
        Some(mapped) => {
            if has_call && mapped == FinishReason::stop() {
                Some(FinishReason::tool_calls())
            } else {
                Some(mapped)
            }
        }
        None => has_call.then(FinishReason::tool_calls),
    }
}

/// Map one raw Gemini `finishReason` name onto the shared [`FinishReason`].
///
/// The mapped names mirror upstream Python's `_FINISH_REASON_MAP`
/// (`gemini/_chat_client.py`), which covers 13 of the 18 members of
/// `google.genai.types.FinishReason`. Anything outside it — `OTHER`,
/// `TOO_MANY_TOOL_CALLS`, `NO_IMAGE`, `IMAGE_OTHER`, and whatever the API
/// adds next — passes through lowercased rather than being dropped, since
/// [`FinishReason`] is an open string enum and a caller learning the turn
/// ended abnormally beats it learning nothing.
///
/// Returns `None` only for `FINISH_REASON_UNSPECIFIED` (and the empty
/// string): both mean "no reason was reported", not "a reason named
/// unspecified".
pub(crate) fn map_finish_reason(reason: &str) -> Option<FinishReason> {
    let mapped = match reason {
        "" | "FINISH_REASON_UNSPECIFIED" => return None,
        "STOP" => FinishReason::stop(),
        "MAX_TOKENS" => FinishReason::new(FinishReason::LENGTH),
        "SAFETY"
        | "RECITATION"
        | "LANGUAGE"
        | "BLOCKLIST"
        | "PROHIBITED_CONTENT"
        | "SPII"
        | "IMAGE_SAFETY"
        | "IMAGE_PROHIBITED_CONTENT"
        | "IMAGE_RECITATION" => FinishReason::new(FinishReason::CONTENT_FILTER),
        "MALFORMED_FUNCTION_CALL" | "UNEXPECTED_TOOL_CALL" => FinishReason::tool_calls(),
        other => FinishReason::new(other.to_lowercase()),
    };
    Some(mapped)
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
    use agent_framework_core::types::FunctionApprovalResponseContent;

    // region: universal-rendering contract

    /// The canonical samples the core contract claims render on every
    /// provider. If this converter stops emitting one of them, the failure
    /// belongs here — next to the converter — not in compaction.
    fn universal_content_samples() -> Vec<Content> {
        use agent_framework_core::types::{
            DataContent, FunctionArguments, FunctionCallContent, FunctionResultContent,
        };
        vec![
            Content::text("hello"),
            Content::FunctionCall(FunctionCallContent::new(
                "contract_call_1",
                "get_weather",
                Some(FunctionArguments::Raw("{\"city\":\"SF\"}".into())),
            )),
            Content::FunctionResult(FunctionResultContent::new(
                "contract_call_1",
                Some(serde_json::json!("sunny")),
            )),
            Content::Data(DataContent::from_bytes(b"png-bytes", "image/png")),
            Content::Data(DataContent::from_bytes(b"jpeg-bytes", "image/jpeg")),
            Content::Data(DataContent::from_bytes(b"webp-bytes", "image/webp")),
            Content::Data(DataContent::from_bytes(b"gif-bytes", "image/gif")),
        ]
    }

    #[test]
    fn every_universal_content_produces_a_gemini_part() {
        for content in universal_content_samples() {
            assert!(content.renders_on_every_provider(), "sample not universal");
            let msg = Message::with_contents(Role::user(), vec![content.clone()]);
            let parts =
                message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
            assert!(
                !parts.is_empty(),
                "core claims this renders everywhere but Gemini emits nothing: {content:?}"
            );
        }
    }

    // region: Gemini 3 thought_signature replay (upstream #7095)

    #[test]
    fn parse_response_captures_thought_signature_on_reasoning() {
        let value = json!({
            "candidates": [{ "content": { "parts": [
                { "text": "thinking...", "thought": true, "thoughtSignature": "c2ln" },
                { "functionCall": { "name": "get_weather", "args": { "city": "SF" } } },
            ] } }]
        });
        let resp = parse_response(&value);
        let contents = &resp.messages[0].contents;
        match &contents[0] {
            Content::TextReasoning(t) => {
                assert_eq!(t.protected_data.as_deref(), Some("c2ln"));
            }
            other => panic!("expected reasoning content, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_captures_a_signature_on_the_function_call_part() {
        // Gemini 3's usual placement: the signature rides on the part carrying
        // the call, not on a preceding thought part.
        let value = json!({
            "candidates": [{ "content": { "parts": [
                {
                    "functionCall": { "name": "get_weather", "args": { "city": "SF" } },
                    "thoughtSignature": "c2ln"
                },
            ] } }]
        });
        let resp = parse_response(&value);
        match &resp.messages[0].contents[0] {
            Content::FunctionCall(fc) => {
                assert_eq!(fc.protected_data.as_deref(), Some("c2ln"));
            }
            other => panic!("expected a function call, got {other:?}"),
        }
    }

    #[test]
    fn a_call_replays_its_own_signature_without_any_reasoning_content() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(
                FunctionCallContent::new("call_1", "get_weather", None)
                    .with_protected_data(Some("c2ln".into())),
            )],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["thoughtSignature"], json!("c2ln"));
    }

    #[test]
    fn a_calls_own_signature_wins_over_the_preceding_reasoning_one() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("from-reasoning".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(
                    FunctionCallContent::new("call_1", "get_weather", None)
                        .with_protected_data(Some("from-call".into())),
                ),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["thoughtSignature"], json!("from-call"));
    }

    /// An approval request/response between a carrier and its call emits no
    /// Part, so it must not break the pairing (upstream #7546). Before the
    /// fix the signature was cleared by any intervening content, and the
    /// replayed call went to Gemini 3 unsigned — a 400.
    #[test]
    fn approval_content_between_reasoning_and_call_does_not_drop_the_signature() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::FunctionApprovalResponse(FunctionApprovalResponseContent {
                    approved: true,
                    id: "call_1".into(),
                    function_call: FunctionCallContent::new("call_1", "get_weather", None),
                }),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        // Only the call reaches the wire, and it is signed.
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["thoughtSignature"], json!("c2ln"));
    }

    /// The conversation-wide map signs a call replayed in a *later* message,
    /// with no carrier of its own anywhere near it — the shape an approval
    /// round trip actually produces.
    #[test]
    fn a_replayed_call_is_signed_from_the_conversation_wide_map() {
        let signed_turn = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        // The replay carries the call alone: no reasoning, no part-level
        // signature. Adjacency has nothing to work with here.
        let replay = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );

        let contents = messages_to_gemini(&[signed_turn, replay]);
        assert_eq!(contents.len(), 2);
        assert_eq!(contents[0]["parts"][0]["thoughtSignature"], json!("c2ln"));
        assert_eq!(contents[1]["parts"][0]["thoughtSignature"], json!("c2ln"));
    }

    /// One carrier signs one call. A self-signed call still consumes it, so a
    /// later unrelated call in the same message is not stamped with a stale
    /// signature — the same rule the emit side has always applied.
    #[test]
    fn a_carrier_does_not_sign_a_second_call() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("carrier".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(
                    FunctionCallContent::new("call_1", "get_weather", None)
                        .with_protected_data(Some("own".into())),
                ),
                Content::FunctionCall(FunctionCallContent::new("call_2", "get_time", None)),
            ],
        );
        let contents = messages_to_gemini(&[msg]);
        let parts = &contents[0]["parts"];
        assert_eq!(parts[0]["thoughtSignature"], json!("own"));
        assert_eq!(parts[1].get("thoughtSignature"), None);
    }

    /// Wire-visible content between a carrier and a call breaks the pairing,
    /// and the conversation-wide fallback must not undo that.
    ///
    /// The fallback is consulted only after adjacency has *declined* to sign a
    /// call, so a map built by laxer rules would silently re-sign exactly the
    /// calls this converter just refused. Caught in review on PR #15, when the
    /// map was still gathered by a separate pre-pass that restated the rules
    /// and omitted this clear.
    #[test]
    fn wire_visible_content_between_a_carrier_and_a_call_breaks_the_pairing() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("carrier".into()),
                    ..Default::default()
                }),
                Content::Text(TextContent::new("an unrelated remark")),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let contents = messages_to_gemini(&[msg]);
        let parts = &contents[0]["parts"];
        assert_eq!(parts[0]["text"], json!("an unrelated remark"));
        assert_eq!(parts[1].get("thoughtSignature"), None);
    }

    /// ...and a call the converter declined to sign contributes nothing to the
    /// map, so replaying it later cannot resurrect the signature either.
    #[test]
    fn an_unsigned_call_is_not_signed_when_replayed_later() {
        let broken_pairing = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("carrier".into()),
                    ..Default::default()
                }),
                Content::Text(TextContent::new("an unrelated remark")),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let replay = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let contents = messages_to_gemini(&[broken_pairing, replay]);
        assert_eq!(contents[1]["parts"][0].get("thoughtSignature"), None);
    }

    /// The map must not invent a signature for a call that never had one.
    #[test]
    fn an_unsigned_call_stays_unsigned() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let contents = messages_to_gemini(&[msg]);
        assert_eq!(contents[0]["parts"][0].get("thoughtSignature"), None);
    }

    #[test]
    fn a_round_trip_preserves_a_part_level_signature() {
        // The whole point: parse a Gemini 3 tool-call response and replay it.
        let value = json!({
            "candidates": [{ "content": { "parts": [
                {
                    "functionCall": { "name": "get_weather", "args": {} },
                    "thoughtSignature": "c2ln"
                },
            ] } }]
        });
        let resp = parse_response(&value);
        let parts = message_contents_to_parts(
            &resp.messages[0].contents,
            &HashMap::new(),
            &mut HashMap::new(),
        );
        assert_eq!(parts[0]["thoughtSignature"], json!("c2ln"));
    }

    #[test]
    fn reasoning_is_not_replayed_as_a_part_but_signs_the_next_call() {
        // Gemini does not accept thought parts back; the signature it carries
        // must instead ride on the function call that reasoning produced.
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 1, "reasoning must not be sent back as a part");
        assert_eq!(parts[0]["thoughtSignature"], json!("c2ln"));
        assert_eq!(parts[0]["functionCall"]["name"], json!("get_weather"));
    }

    #[test]
    fn a_signature_survives_a_later_unsigned_reasoning_part() {
        // Gemini can return several adjacent thought parts with the signature
        // on an earlier one. `parse_response` does not coalesce them, so an
        // unsigned part that merely follows a signed one used to erase it.
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "first".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::TextReasoning(TextReasoningContent {
                    text: "second".into(),
                    ..Default::default()
                }),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0]["thoughtSignature"], json!("c2ln"));
    }

    #[test]
    fn a_later_reasoning_signature_wins_over_an_earlier_one() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "first".into(),
                    protected_data: Some("older".into()),
                    ..Default::default()
                }),
                Content::TextReasoning(TextReasoningContent {
                    text: "second".into(),
                    protected_data: Some("newer".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts[0]["thoughtSignature"], json!("newer"));
    }

    #[test]
    fn a_signature_only_applies_to_an_immediately_following_call() {
        // Text between the reasoning and the call clears the pending signature,
        // so a stale signature is never stamped onto an unrelated call.
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(TextReasoningContent {
                    text: "thinking...".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::text("here goes"),
                Content::FunctionCall(FunctionCallContent::new("call_1", "get_weather", None)),
            ],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 2);
        assert!(parts[0].get("thoughtSignature").is_none());
        assert!(parts[1].get("thoughtSignature").is_none());
    }

    #[test]
    fn a_call_without_reasoning_carries_no_signature() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(FunctionCallContent::new(
                "call_1",
                "get_weather",
                None,
            ))],
        );
        let parts = message_contents_to_parts(&msg.contents, &HashMap::new(), &mut HashMap::new());
        assert_eq!(parts.len(), 1);
        assert!(parts[0].get("thoughtSignature").is_none());
    }

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
        assert_eq!(map_finish_reason("STOP"), Some(FinishReason::stop()));
        assert_eq!(
            map_finish_reason("MAX_TOKENS"),
            Some(FinishReason::new(FinishReason::LENGTH))
        );
        for reason in [
            "SAFETY",
            "RECITATION",
            "LANGUAGE",
            "BLOCKLIST",
            "PROHIBITED_CONTENT",
            "SPII",
            "IMAGE_SAFETY",
            "IMAGE_PROHIBITED_CONTENT",
            "IMAGE_RECITATION",
        ] {
            assert_eq!(
                map_finish_reason(reason),
                Some(FinishReason::new(FinishReason::CONTENT_FILTER)),
                "{reason}"
            );
        }
        for reason in ["MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL"] {
            assert_eq!(
                map_finish_reason(reason),
                Some(FinishReason::tool_calls()),
                "{reason}"
            );
        }
    }

    /// `FinishReason` is an open string enum, so a name the map does not
    /// cover — `OTHER`, `TOO_MANY_TOOL_CALLS`, `NO_IMAGE`, `IMAGE_OTHER`,
    /// or whatever the API adds next — must reach the caller rather than
    /// being dropped. Upstream's own `_FINISH_REASON_MAP.get(reason)`
    /// looked up without a default and lost exactly these (#7837); pinned
    /// here because a later "tidy the match into a lookup table" refactor
    /// is precisely how it would come back.
    #[test]
    fn map_finish_reason_passes_unmapped_values_through() {
        for reason in ["OTHER", "TOO_MANY_TOOL_CALLS", "NO_IMAGE", "IMAGE_OTHER"] {
            assert_eq!(
                map_finish_reason(reason),
                Some(FinishReason::new(reason.to_lowercase())),
                "{reason}"
            );
        }
    }

    /// `FINISH_REASON_UNSPECIFIED` is proto3's "field never set", not a
    /// reason in its own right: it must read as *absent*, so a response
    /// carrying it behaves exactly like one carrying no `finishReason` at
    /// all — including the `tool_calls` upgrade when the turn ends in a
    /// function call.
    #[test]
    fn unspecified_finish_reason_reads_as_absent() {
        assert_eq!(map_finish_reason("FINISH_REASON_UNSPECIFIED"), None);
        assert_eq!(map_finish_reason(""), None);

        let resp = parse_response(&json!({
            "candidates": [{
                "content": { "role": "model", "parts": [{ "text": "hi" }] },
                "finishReason": "FINISH_REASON_UNSPECIFIED",
            }],
        }));
        assert_eq!(resp.finish_reason, None);

        let with_call = parse_response(&json!({
            "candidates": [{
                "content": { "role": "model", "parts": [
                    { "functionCall": { "name": "f", "args": {} } }
                ]},
                "finishReason": "FINISH_REASON_UNSPECIFIED",
            }],
        }));
        assert_eq!(with_call.finish_reason, Some(FinishReason::tool_calls()));
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

    /// The second half of upstream #7837: Python attached a streamed
    /// chunk's usage only when the finish reason was truthy, so an unmapped
    /// reason dropping to `None` took the whole turn's token accounting
    /// down with it. Usage here is attached from `usageMetadata` alone,
    /// independent of the finish reason — pinned for both an unmapped
    /// reason and `FINISH_REASON_UNSPECIFIED`, the two values that made the
    /// cascade fire.
    #[test]
    fn parse_stream_chunk_carries_usage_regardless_of_finish_reason() {
        for (raw, expected) in [
            (
                "TOO_MANY_TOOL_CALLS",
                Some(FinishReason::new("too_many_tool_calls")),
            ),
            ("FINISH_REASON_UNSPECIFIED", None),
        ] {
            let value = json!({
                "candidates": [{ "content": { "role": "model", "parts": [] }, "finishReason": raw }],
                "usageMetadata": { "promptTokenCount": 7, "candidatesTokenCount": 3, "totalTokenCount": 10 },
            });
            let update = parse_stream_chunk(&value).unwrap();
            assert_eq!(update.finish_reason, expected, "{raw}");
            let usage = update
                .contents
                .iter()
                .find_map(|c| match c {
                    Content::Usage(u) => Some(u.details.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| panic!("{raw}: usage dropped"));
            assert_eq!(usage.total_token_count, Some(10), "{raw}");
        }
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
