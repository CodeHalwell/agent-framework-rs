//! Per-point codecs: framework values <-> AGENT-HOOKS wire JSON.
//!
//! Each codec owns both directions for its point: `*_to_wire` projects the
//! native value into the spec payload, and `*_write_back` converts the
//! (possibly transformed) wire target back. They share one rule: a wire value
//! the interceptors left untouched maps back to the untouched native value,
//! so only a genuine transform changes native state, and a transform that
//! cannot be translated is an error (the run fails closed) rather than being
//! dropped. Rich content is projected as content objects, never flattened to
//! text. Ports upstream's `_InputCodec`, `_ModelRequestCodec`,
//! `_ModelResponseCodec`, `_ToolArgumentsCodec`, `_ToolResultCodec` and
//! `_OutputCodec`.

use std::collections::HashSet;

use serde_json::{Map, Value};

use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::{
    AgentResponse, ChatResponse, Content, FinishReason, FunctionArguments, FunctionCallContent,
    Message, Role, UsageDetails,
};

fn write_back_error(point: &str, what: &str) -> Error {
    Error::middleware_failure(format!(
        "agent-hooks {point} transform could not be written back: {what}"
    ))
}

/// Map a framework role onto the spec's input role enum.
fn input_role(role: &Role) -> &'static str {
    match role.0.as_str() {
        Role::USER => "user",
        Role::SYSTEM => "system",
        _ => "external",
    }
}

/// Project one content item as the model will see it.
///
/// `FunctionResultContent::exception` serializes as a redaction marker for
/// persistence, but provider converters read the raw field and send that
/// text to the model. The projection carries the raw text instead, so an
/// interceptor judges what the model actually receives (and an untouched
/// entry still matches its original on write-back).
fn content_to_wire(content: &Content) -> Result<Value> {
    let mut value = serde_json::to_value(content)?;
    if let (Content::FunctionResult(result), Value::Object(map)) = (content, &mut value) {
        if let Some(exception) = &result.exception {
            map.insert("exception".into(), Value::String(exception.clone()));
        }
    }
    Ok(value)
}

/// Plain text as a string; anything else as a list of content objects.
pub(super) fn contents_to_wire(contents: &[Content]) -> Result<Value> {
    if let [Content::Text(text)] = contents {
        return Ok(Value::String(text.text.clone()));
    }
    contents
        .iter()
        .map(content_to_wire)
        .collect::<Result<Vec<_>>>()
        .map(Value::Array)
}

fn message_to_wire_with_role(message: &Message, role: &str) -> Result<Value> {
    Ok(serde_json::json!({ "role": role, "content": contents_to_wire(&message.contents)? }))
}

fn message_to_wire(message: &Message) -> Result<Value> {
    message_to_wire_with_role(message, &message.role.0)
}

/// Decode a transformed wire content value back into content items.
///
/// An item whose `type` this library does not know would deserialize to the
/// inert `Content::Unknown` and silently drop the interceptor's content, so it
/// is rejected instead.
fn wire_to_contents(value: &Value, point: &str) -> Result<Vec<Content>> {
    let items: Vec<&Value> = match value {
        Value::Null => return Ok(Vec::new()),
        Value::String(s) => return Ok(vec![Content::text(s.clone())]),
        Value::Object(_) => vec![value],
        Value::Array(items) => items.iter().collect(),
        _ => return Err(write_back_error(point, "unsupported content value")),
    };
    items
        .into_iter()
        .map(|item| match item {
            Value::String(s) => Ok(Content::text(s.clone())),
            Value::Object(map) if map.contains_key("type") => {
                match serde_json::from_value::<Content>(item.clone()) {
                    Ok(Content::Unknown) | Err(_) => {
                        Err(write_back_error(point, "undecodable content item"))
                    }
                    Ok(content) => Ok(content),
                }
            }
            _ => Err(write_back_error(point, "unsupported content item")),
        })
        .collect()
}

fn looks_like_message_list(value: &Value) -> bool {
    matches!(value, Value::Array(items)
        if !items.is_empty() && items.iter().all(|i| i.get("content").is_some()))
}

/// Convert a transformed wire message list back into messages.
///
/// The transformed list is authoritative. Entries are matched to originals by
/// projection identity, not position, so an insertion or removal in the
/// middle does not shift content onto the wrong message: an entry equal to an
/// unconsumed original's projection reuses that original untouched; a changed
/// entry rewrites the next unconsumed original in place only when that
/// original is not preserved later in the list and its role is unchanged;
/// anything else becomes a new message.
fn write_back_message_list(
    originals: Vec<Message>,
    before: &[Value],
    after: &Value,
    point: &str,
) -> Result<Vec<Message>> {
    let Value::Array(after_items) = after else {
        return Err(write_back_error(point, "expected a list of messages"));
    };
    // Every entry needs a string role and a content member: a missing or
    // non-string role is an unappliable transform, not an implicit `user`.
    if after_items
        .iter()
        .any(|i| !i.get("role").is_some_and(Value::is_string) || i.get("content").is_none())
    {
        return Err(write_back_error(point, "a message lacks role/content"));
    }
    let mut result = Vec::with_capacity(after_items.len());
    let mut cursor = 0;
    for (index, item) in after_items.iter().enumerate() {
        if let Some(found) = (cursor..originals.len()).find(|&p| before[p] == *item) {
            result.push(originals[found].clone());
            cursor = found + 1;
            continue;
        }
        let role = item["role"].as_str().unwrap_or_default();
        let content = wire_to_contents(&item["content"], point)?;
        if cursor < originals.len() {
            let candidate = &before[cursor];
            let preserved_later = after_items[index + 1..].iter().any(|l| l == candidate);
            if !preserved_later && candidate.get("role").and_then(Value::as_str) == Some(role) {
                let mut message = originals[cursor].clone();
                cursor += 1;
                message.contents = content;
                result.push(message);
                continue;
            }
        }
        result.push(Message::with_contents(role, content));
    }
    Ok(result)
}

// ---- input ---------------------------------------------------------------

fn input_message_to_wire(message: &Message) -> Result<Value> {
    message_to_wire_with_role(message, input_role(&message.role))
}

/// Project the run's input: one message as its content and mapped role;
/// several as a list of `{role, content}` objects under role `user`.
pub(super) fn input_to_wire(messages: &[Message]) -> Result<(Value, &'static str)> {
    if let [message] = messages {
        return Ok((
            contents_to_wire(&message.contents)?,
            input_role(&message.role),
        ));
    }
    let list = messages
        .iter()
        .map(input_message_to_wire)
        .collect::<Result<Vec<_>>>()?;
    Ok((Value::Array(list), "user"))
}

/// Write a transformed `input` target (`{content, role}`) back.
pub(super) fn input_write_back(
    messages: &mut Vec<Message>,
    before: &Value,
    after: &Value,
) -> Result<()> {
    if after == before {
        return Ok(());
    }
    let Value::Object(after_map) = after else {
        return Err(write_back_error("input", "expected an input object"));
    };
    let after_role = after_map.get("role");
    if after_role != before.get("role") {
        // The role is per-message only for single-message input; for several
        // messages the top-level role is synthetic.
        match (messages.len(), after_role.and_then(Value::as_str)) {
            (1, Some(role)) => messages[0].role = Role::new(role),
            _ => return Err(write_back_error("input", "the input role changed")),
        }
    }
    let after_content = after_map.get("content").unwrap_or(&Value::Null);
    if Some(after_content) == before.get("content") {
        return Ok(());
    }
    if messages.len() == 1 && !looks_like_message_list(after_content) {
        messages[0].contents = wire_to_contents(after_content, "input")?;
        return Ok(());
    }
    let before_list = messages
        .iter()
        .map(input_message_to_wire)
        .collect::<Result<Vec<_>>>()?;
    let originals = std::mem::take(messages);
    *messages = write_back_message_list(originals, &before_list, after_content, "input")?;
    Ok(())
}

// ---- pre_model_call ------------------------------------------------------

/// Project the outgoing request messages.
pub(super) fn request_to_wire(messages: &[Message]) -> Result<Vec<Value>> {
    messages.iter().map(message_to_wire).collect()
}

/// The transformed request, or the original messages when untouched.
pub(super) fn request_write_back(
    messages: Vec<Message>,
    before: &[Value],
    after: &Value,
) -> Result<Vec<Message>> {
    if matches!(after, Value::Array(items) if items.as_slice() == before) {
        return Ok(messages);
    }
    write_back_message_list(messages, before, after, "pre_model_call")
}

/// Project the call's tools for `pre_model_call` (`{name, description?}`),
/// or `None` (the optional field omitted) when the call offers none.
pub(super) fn tools_to_wire(tools: &[ToolDefinition]) -> Option<Vec<Value>> {
    if tools.is_empty() {
        return None;
    }
    Some(
        tools
            .iter()
            .map(|t| {
                let mut entry = Map::new();
                entry.insert("name".into(), Value::String(t.name.clone()));
                if !t.description.is_empty() {
                    entry.insert("description".into(), Value::String(t.description.clone()));
                }
                Value::Object(entry)
            })
            .collect(),
    )
}

// ---- post_model_call -----------------------------------------------------

/// Calls whose result is already in the same response were executed by the
/// provider; only the rest are host-executed (the ones the tool seam will
/// bracket).
fn provider_resolved_ids(messages: &[Message]) -> HashSet<String> {
    messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(Content::as_function_result)
        .map(|r| r.call_id.clone())
        .collect()
}

fn is_host_call<'a>(
    content: &'a Content,
    resolved: &HashSet<String>,
) -> Option<&'a FunctionCallContent> {
    match content {
        Content::FunctionCall(call) if !resolved.contains(&call.call_id) => Some(call),
        _ => None,
    }
}

/// Project a call's arguments as the spec's `args` object; unparseable raw
/// arguments ride as `{"raw_arguments": ...}`.
fn call_args_to_wire(call: &FunctionCallContent) -> Map<String, Value> {
    match call.parse_arguments() {
        Ok(map) => map.into_iter().collect(),
        Err(_) => {
            let raw = match &call.arguments {
                Some(FunctionArguments::Raw(raw)) => raw.clone(),
                _ => String::new(),
            };
            let mut map = Map::new();
            map.insert("raw_arguments".into(), Value::String(raw));
            map
        }
    }
}

/// The response's finish reason as a wire string (`stop` when absent).
pub(super) fn finish_reason_to_wire(reason: Option<&FinishReason>) -> String {
    reason.map_or_else(
        || FinishReason::STOP.to_string(),
        |r| r.as_str().to_string(),
    )
}

/// The spec's optional `usage` block, when the response reported any.
pub(super) fn usage_to_wire(usage: Option<&UsageDetails>) -> Option<Value> {
    let usage = usage?;
    let mut map = Map::new();
    if let Some(n) = usage.input_token_count {
        map.insert("prompt_tokens".into(), Value::from(n));
    }
    if let Some(n) = usage.output_token_count {
        map.insert("completion_tokens".into(), Value::from(n));
    }
    (!map.is_empty()).then_some(Value::Object(map))
}

/// Project the assembled response: everything except host-executed tool
/// calls as `content` (hosted-tool activity included, so it is interceptable
/// here even though the tool seam never sees it), those calls as
/// `tool_calls`, and the finish reason.
pub(super) fn response_to_wire(response: &ChatResponse) -> Result<Value> {
    let resolved = provider_resolved_ids(&response.messages);
    let mut parts = Vec::new();
    let mut calls = Vec::new();
    for message in &response.messages {
        let mut visible = Vec::new();
        for content in &message.contents {
            match is_host_call(content, &resolved) {
                Some(call) => calls.push(serde_json::json!({
                    "id": call.call_id,
                    "name": call.name,
                    "args": call_args_to_wire(call),
                })),
                None => visible.push(content.clone()),
            }
        }
        if !visible.is_empty() {
            parts.push(message_to_wire_with_role(
                &Message::with_contents(message.role.clone(), visible),
                &message.role.0,
            )?);
        }
    }
    let content = match parts.as_slice() {
        [] => Value::Null,
        [only] if only["content"].is_string() => only["content"].clone(),
        _ => Value::Array(parts),
    };
    Ok(serde_json::json!({
        "content": content,
        "tool_calls": calls,
        "finish_reason": finish_reason_to_wire(response.finish_reason.as_ref()),
    }))
}

/// Write a transformed `post_model_call` target back. Returns whether the
/// response changed; a changed response drops its parsed structured `value`,
/// which was derived from the pre-transform text.
pub(super) fn response_write_back(
    response: &mut ChatResponse,
    before: &Value,
    after: &Value,
) -> Result<bool> {
    const POINT: &str = "post_model_call";
    if after == before {
        return Ok(false);
    }
    let Value::Object(after_map) = after else {
        return Err(write_back_error(POINT, "expected a response object"));
    };
    let mut changed = false;
    let after_finish = after_map.get("finish_reason");
    if after_finish != before.get("finish_reason") {
        let Some(reason) = after_finish.and_then(Value::as_str) else {
            return Err(write_back_error(POINT, "finish_reason must be a string"));
        };
        response.finish_reason = Some(FinishReason::new(reason));
        changed = true;
    }
    let after_calls = after_map.get("tool_calls").unwrap_or(&Value::Null);
    if Some(after_calls) != before.get("tool_calls") {
        changed |= write_back_tool_calls(response, after_calls)?;
    }
    let after_content = after_map.get("content").unwrap_or(&Value::Null);
    if Some(after_content) != before.get("content") {
        write_back_content(response, after_content)?;
        changed = true;
    }
    if changed {
        response.value = None;
    }
    Ok(changed)
}

fn write_back_tool_calls(response: &mut ChatResponse, after_calls: &Value) -> Result<bool> {
    const POINT: &str = "post_model_call";
    let Value::Array(items) = after_calls else {
        return Err(write_back_error(POINT, "tool_calls must stay a list"));
    };
    let mut wire_calls: Vec<(String, &Map<String, Value>)> = Vec::new();
    for item in items {
        let (Some(map), Some(id)) = (item.as_object(), item.get("id").and_then(Value::as_str))
        else {
            return Err(write_back_error(POINT, "a tool call lacks an id"));
        };
        if !map
            .get("name")
            .is_some_and(|n| n.as_str().is_some_and(|n| !n.is_empty()))
        {
            return Err(write_back_error(POINT, "a tool call lacks a name"));
        }
        if !map.get("args").is_some_and(Value::is_object) {
            return Err(write_back_error(
                POINT,
                "a tool call's args must be an object",
            ));
        }
        wire_calls.push((id.to_string(), map));
    }
    // Reconcile occurrence-aware: providers may reuse a call id within one
    // response, so the k-th wire entry with an id pairs with the k-th native
    // call carrying it. The rebuilt calls follow the transformed order.
    let resolved = provider_resolved_ids(&response.messages);
    let mut originals: Vec<Option<FunctionCallContent>> = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter_map(|c| is_host_call(c, &resolved).cloned())
        .map(Some)
        .collect();
    let before: Vec<FunctionCallContent> = originals.iter().flatten().cloned().collect();
    let mut rebuilt = Vec::with_capacity(wire_calls.len());
    for (id, wire) in &wire_calls {
        let name = wire["name"].as_str().unwrap_or_default();
        let args = wire["args"].as_object().cloned().unwrap_or_default();
        let matched = originals
            .iter_mut()
            .find(|c| c.as_ref().is_some_and(|c| &c.call_id == id))
            .and_then(Option::take);
        let call = match matched {
            Some(mut call) => {
                if name != call.name {
                    call.name = name.to_string();
                }
                if call_args_to_wire(&call) != args {
                    call.arguments = Some(FunctionArguments::Object(args.into_iter().collect()));
                }
                call
            }
            None => FunctionCallContent::new(
                id.clone(),
                name,
                Some(FunctionArguments::Object(args.into_iter().collect())),
            ),
        };
        rebuilt.push(call);
    }
    if rebuilt == before {
        return Ok(false);
    }
    // Put the rebuilt calls where the first native host call was (or at the
    // end of the last assistant message when there was none).
    let mut anchor: Option<(usize, usize)> = None;
    for (mi, message) in response.messages.iter_mut().enumerate() {
        let mut kept = Vec::with_capacity(message.contents.len());
        for content in std::mem::take(&mut message.contents) {
            if is_host_call(&content, &resolved).is_some() {
                anchor.get_or_insert((mi, kept.len()));
            } else {
                kept.push(content);
            }
        }
        message.contents = kept;
    }
    let rebuilt: Vec<Content> = rebuilt.into_iter().map(Content::FunctionCall).collect();
    match anchor {
        Some((mi, at)) => {
            let contents = &mut response.messages[mi].contents;
            contents.splice(at..at, rebuilt);
        }
        None if rebuilt.is_empty() => {}
        None => match response
            .messages
            .iter_mut()
            .rev()
            .find(|m| m.role.0 == Role::ASSISTANT)
        {
            Some(target) => target.contents.extend(rebuilt),
            None => response
                .messages
                .push(Message::with_contents(Role::assistant(), rebuilt)),
        },
    }
    Ok(true)
}

/// Rebuild the response's visible content, keeping its host-executed tool
/// calls (already reconciled with `tool_calls`).
fn write_back_content(response: &mut ChatResponse, after_content: &Value) -> Result<()> {
    const POINT: &str = "post_model_call";
    let resolved = provider_resolved_ids(&response.messages);
    let calls: Vec<Content> = response
        .messages
        .iter()
        .flat_map(|m| m.contents.iter())
        .filter(|c| is_host_call(c, &resolved).is_some())
        .cloned()
        .collect();
    let mut base: Vec<Message> = match after_content {
        Value::Null => Vec::new(),
        Value::String(s) => vec![Message::assistant(s.clone())],
        Value::Array(items) => items
            .iter()
            .map(|item| {
                // A missing or non-string role is an unappliable transform,
                // not an implicit `assistant`.
                let role = match item.get("role").and_then(Value::as_str) {
                    Some(role) if item.get("content").is_some() => role,
                    _ => return Err(write_back_error(POINT, "content lacks role/content")),
                };
                Ok(Message::with_contents(
                    role,
                    wire_to_contents(&item["content"], POINT)?,
                ))
            })
            .collect::<Result<_>>()?,
        _ => return Err(write_back_error(POINT, "unsupported content")),
    };
    if !calls.is_empty() {
        match base.last_mut() {
            Some(last) if last.role.0 == Role::ASSISTANT => last.contents.extend(calls),
            _ => base.push(Message::with_contents(Role::assistant(), calls)),
        }
    }
    response.messages = base;
    Ok(())
}

// ---- pre_tool_call / post_tool_call --------------------------------------

/// Project tool arguments as the spec's `args` object.
pub(super) fn tool_args_to_wire(arguments: &Value) -> Map<String, Value> {
    match arguments {
        Value::Object(map) => map.clone(),
        Value::Null => Map::new(),
        other => {
            let mut map = Map::new();
            map.insert("raw_arguments".into(), other.clone());
            map
        }
    }
}

/// The transformed arguments, or `None` when untouched.
pub(super) fn tool_args_write_back(
    before: &Map<String, Value>,
    after: &Value,
) -> Result<Option<Map<String, Value>>> {
    let Value::Object(after) = after else {
        return Err(write_back_error(
            "pre_tool_call",
            "expected an arguments object",
        ));
    };
    Ok((after != before).then(|| after.clone()))
}

// ---- output --------------------------------------------------------------

/// Project the run output: a single plain-text message as a string, else a
/// list of `{role, content}` objects.
pub(super) fn output_to_wire(response: &AgentResponse) -> Result<Value> {
    let parts = response
        .messages
        .iter()
        .map(message_to_wire)
        .collect::<Result<Vec<_>>>()?;
    Ok(match parts.as_slice() {
        [only] if only["content"].is_string() => only["content"].clone(),
        _ => Value::Array(parts),
    })
}

/// Write a transformed `output` target (`{content}`) back. Returns whether
/// the response changed; a changed response drops its parsed structured
/// `value`, which was derived from the pre-transform text.
pub(super) fn output_write_back(
    response: &mut AgentResponse,
    before_content: &Value,
    after: &Value,
) -> Result<bool> {
    let Value::Object(after) = after else {
        return Err(write_back_error("output", "expected an output object"));
    };
    let after_content = after.get("content").unwrap_or(&Value::Null);
    if after_content == before_content {
        return Ok(false);
    }
    match after_content {
        Value::String(text) if response.messages.len() == 1 => {
            response.messages[0].contents = vec![Content::text(text.clone())];
        }
        Value::String(text) => response.messages = vec![Message::assistant(text.clone())],
        Value::Null => response.messages.clear(),
        _ => {
            let before_list = response
                .messages
                .iter()
                .map(message_to_wire)
                .collect::<Result<Vec<_>>>()?;
            let originals = std::mem::take(&mut response.messages);
            response.messages =
                write_back_message_list(originals, &before_list, after_content, "output")?;
        }
    }
    response.value = None;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn single_text_input_projects_as_string() {
        let messages = vec![Message::user("hello")];
        let (content, role) = input_to_wire(&messages).unwrap();
        assert_eq!(content, json!("hello"));
        assert_eq!(role, "user");
        let assistant = vec![Message::assistant("x")];
        assert_eq!(input_to_wire(&assistant).unwrap().1, "external");
    }

    #[test]
    fn rich_content_is_never_flattened() {
        let m = Message::with_contents(
            "user",
            vec![
                Content::text("look"),
                Content::Uri(crate::types::UriContent {
                    uri: "https://example.com/a.png".into(),
                    media_type: "image/png".into(),
                }),
            ],
        );
        let wire = contents_to_wire(&m.contents).unwrap();
        assert_eq!(wire[0]["type"], "text");
        assert_eq!(wire[1]["type"], "uri");
        let back = wire_to_contents(&wire, "input").unwrap();
        assert_eq!(back, m.contents);
    }

    #[test]
    fn unknown_content_types_fail_write_back() {
        let err = wire_to_contents(&json!([{"type": "from_the_future"}]), "output").unwrap_err();
        assert!(err.is_middleware_failure());
        assert!(wire_to_contents(&json!(42), "output").is_err());
    }

    #[test]
    fn input_write_back_rewrites_single_message_in_place() {
        let mut messages = vec![Message::user("my ssn is 123")];
        let (content, role) = input_to_wire(&messages).unwrap();
        let before = json!({"content": content, "role": role});
        input_write_back(
            &mut messages,
            &before,
            &json!({"content": "my ssn is [redacted]", "role": "user"}),
        )
        .unwrap();
        assert_eq!(messages[0].text(), "my ssn is [redacted]");
        // Untouched is a no-op.
        let mut same = vec![Message::user("x")];
        let before = json!({"content": "x", "role": "user"});
        input_write_back(&mut same, &before, &before).unwrap();
        assert_eq!(same[0].text(), "x");
    }

    #[test]
    fn message_list_write_back_matches_by_identity() {
        let originals = vec![
            Message::system("sys"),
            Message::user("a"),
            Message::user("b"),
        ];
        let before = request_to_wire(&originals).unwrap();
        // Remove the middle message and rewrite the last.
        let after = json!([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "B!"}
        ]);
        let out = request_write_back(originals.clone(), &before, &after).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text(), "sys");
        assert_eq!(out[1].text(), "B!");
        // Untouched returns the originals.
        let same =
            request_write_back(originals.clone(), &before, &Value::Array(before.clone())).unwrap();
        assert_eq!(same.len(), 3);
        assert!(request_write_back(originals, &before, &json!("nope")).is_err());
    }

    #[test]
    fn message_list_write_back_rejects_missing_or_non_string_roles() {
        let originals = vec![Message::user("a"), Message::user("b")];
        let before = request_to_wire(&originals).unwrap();
        for after in [
            json!([{"role": "user", "content": "a"}, {"content": "sanitized"}]),
            json!([{"role": "user", "content": "a"}, {"role": 7, "content": "b"}]),
            json!([{"role": "user", "content": "a"}, {"role": null, "content": "b"}]),
        ] {
            let err = request_write_back(originals.clone(), &before, &after).unwrap_err();
            assert!(err.is_middleware_failure(), "{after}");
        }
    }

    fn tool_call_response() -> ChatResponse {
        ChatResponse {
            messages: vec![Message::with_contents(
                "assistant",
                vec![
                    Content::text("calling"),
                    Content::FunctionCall(FunctionCallContent::new(
                        "c1",
                        "lookup",
                        Some(FunctionArguments::Raw("{\"q\":\"x\"}".into())),
                    )),
                ],
            )],
            finish_reason: Some(FinishReason::tool_calls()),
            ..Default::default()
        }
    }

    #[test]
    fn response_projection_splits_host_calls() {
        let wire = response_to_wire(&tool_call_response()).unwrap();
        assert_eq!(wire["content"], json!("calling"));
        assert_eq!(
            wire["tool_calls"],
            json!([{"id": "c1", "name": "lookup", "args": {"q": "x"}}])
        );
        assert_eq!(wire["finish_reason"], "tool_calls");
        let empty = response_to_wire(&ChatResponse::default()).unwrap();
        assert_eq!(empty["tool_calls"], json!([]));
        assert_eq!(empty["content"], Value::Null);
    }

    #[test]
    fn response_write_back_rewrites_args_and_content() {
        let mut response = tool_call_response();
        response.value = Some(json!({"stale": true}));
        let before = response_to_wire(&response).unwrap();
        let mut after = before.clone();
        after["tool_calls"][0]["args"]["q"] = json!("y");
        after["content"] = json!("calling (sanitized)");
        assert!(response_write_back(&mut response, &before, &after).unwrap());
        let wire = response_to_wire(&response).unwrap();
        assert_eq!(wire, after);
        assert_eq!(response.value, None);

        // Dropping every call removes them.
        let mut response = tool_call_response();
        let before = response_to_wire(&response).unwrap();
        let mut after = before.clone();
        after["tool_calls"] = json!([]);
        assert!(response_write_back(&mut response, &before, &after).unwrap());
        assert!(response.messages[0]
            .contents
            .iter()
            .all(|c| c.as_function_call().is_none()));

        let mut response = tool_call_response();
        let before = response_to_wire(&response).unwrap();
        let mut after = before.clone();
        after["tool_calls"] = json!([{"id": "c1"}]);
        assert!(response_write_back(&mut response, &before, &after).is_err());
    }

    #[test]
    fn response_content_write_back_rejects_missing_or_non_string_roles() {
        let original = ChatResponse {
            messages: vec![Message::assistant("a"), Message::assistant("b")],
            ..Default::default()
        };
        let before = response_to_wire(&original).unwrap();
        for content in [
            json!([{"role": "assistant", "content": "a"}, {"content": "sanitized"}]),
            json!([{"role": "assistant", "content": "a"}, {"role": 7, "content": "b"}]),
            json!([{"role": "assistant", "content": "a"}, {"role": null, "content": "b"}]),
        ] {
            let mut response = original.clone();
            let mut after = before.clone();
            after["content"] = content.clone();
            let err = response_write_back(&mut response, &before, &after).unwrap_err();
            assert!(err.is_middleware_failure(), "{content}");
        }
        // A well-formed list still applies.
        let mut response = original.clone();
        let mut after = before.clone();
        after["content"] = json!([{"role": "assistant", "content": "only"}]);
        assert!(response_write_back(&mut response, &before, &after).unwrap());
        assert_eq!(response.text(), "only");
    }

    #[test]
    fn output_codec_round_trip() {
        let mut response = AgentResponse {
            messages: vec![Message::assistant("secret")],
            value: Some(json!("secret")),
            ..Default::default()
        };
        let before = output_to_wire(&response).unwrap();
        assert_eq!(before, json!("secret"));
        assert!(!output_write_back(&mut response, &before, &json!({"content": "secret"})).unwrap());
        assert!(
            output_write_back(&mut response, &before, &json!({"content": "[redacted]"})).unwrap()
        );
        assert_eq!(response.text(), "[redacted]");
        assert_eq!(response.value, None);
        assert!(output_write_back(&mut response, &before, &json!("bare")).is_err());
    }

    #[test]
    fn tool_args_round_trip() {
        let args = json!({"path": "/etc/passwd"});
        let wire = tool_args_to_wire(&args);
        assert_eq!(
            tool_args_write_back(&wire, &Value::Object(wire.clone())).unwrap(),
            None
        );
        let changed = tool_args_write_back(&wire, &json!({"path": "/tmp/x"})).unwrap();
        assert_eq!(changed.unwrap()["path"], "/tmp/x");
        assert!(tool_args_write_back(&wire, &json!([1])).is_err());
    }
}
