//! The AG-UI chat client (feature `agui-client`).
//!
//! [`AgUiChatClient`] is a [`ChatClient`] for a remote AG-UI server, ported
//! from Python's `AGUIChatClient` (`agent_framework_ag_ui._client`) with its
//! helpers `AGUIEventConverter`, `AGUIHttpService` and
//! `agent_framework_messages_to_agui`. Each request is one AG-UI run: the
//! client posts a [`RunAgentInput`] and reads the `text/event-stream`
//! response, turning each event into a [`ChatResponseUpdate`].
//!
//! # Requests
//! - **Thread id.** Taken from `options.metadata["thread_id"]`, or a fresh
//!   `thread_<hex>` id. A fresh `run_<hex>` id is minted per request. The
//!   server's ids come back in the response's `additional_properties` under
//!   `thread_id` and `run_id`.
//! - **History.** The client sends exactly the messages it is given; it keeps
//!   no history of its own. An [`Agent`](agent_framework_core::agent::Agent)
//!   with a session sends the whole conversation each turn. A server that
//!   keeps history per thread can be driven with a fixed thread id instead.
//! - **Tools.** Every function tool in `options.tools` is declared to the
//!   server as `{name, description, parameters}` so the server's model can
//!   call it; hosted tools are not sent. When the server calls one, the call
//!   comes back as a [`FunctionCallContent`] with no result, and the
//!   function-invocation loop of the wrapping `Agent` runs it locally and
//!   sends the result on the next request. Tools the server runs itself come
//!   back as a call followed by its result (`TOOL_CALL_RESULT`), which the
//!   loop recognises as already resolved and does not run.
//! - **State.** A message built with [`state_carrier`] is not sent as a chat
//!   message; the newest one's JSON object becomes the run's `state`.
//! - **Instructions.** `options.instructions` is sent as a leading system
//!   message, as the Go provider does.
//! - **Other run fields.** `options.additional_properties` may carry
//!   `available_interrupts` (or `availableInterrupts`), `resume`,
//!   `forwarded_props` (or `forwardedProps`) and `context`. Interrupts and
//!   resume entries are normalised to the protocol's canonical shape the way
//!   upstream normalises them.
//!
//! # Events
//! | AG-UI event | Update |
//! |---|---|
//! | `RUN_STARTED` | `additional_properties` `thread_id`, `run_id` |
//! | `TEXT_MESSAGE_START` | assistant update carrying the `messageId` |
//! | `TEXT_MESSAGE_CONTENT`, `TEXT_MESSAGE_CHUNK` | text delta |
//! | `TEXT_MESSAGE_END`, `TOOL_CALL_END` | nothing |
//! | `TOOL_CALL_START` | a [`FunctionCallContent`] with empty arguments |
//! | `TOOL_CALL_ARGS` | an argument fragment of the call its `toolCallId` names (the latest open call when it has none) |
//! | `TOOL_CALL_CHUNK` | a call fragment: opens the call on its first chunk, later chunks without an id continue it |
//! | `TOOL_CALL_RESULT` | a tool-role [`FunctionResultContent`] carrying the event's `messageId` |
//! | `REASONING_MESSAGE_CONTENT`, `REASONING_MESSAGE_CHUNK` | reasoning delta |
//! | `STATE_SNAPSHOT`, `STATE_DELTA` | JSON [`DataContent`] (`application/json`, `application/json-patch+json`) |
//! | `MESSAGES_SNAPSHOT` | `additional_properties["ag_ui_messages_snapshot"]` |
//! | `RUN_FINISHED` | finish reason `stop`; `interrupt`, `outcome`, `interrupts`, `result` metadata |
//! | `RUN_ERROR` | an [`ErrorContent`] with code `RUN_ERROR` |
//! | `CUSTOM` | `additional_properties["ag_ui_custom_event"]`; an `annotations` event restores text annotations |
//!
//! A `RUN_ERROR` is in-band, as upstream has it: the request succeeds and the
//! response carries the error content. An HTTP error status fails the request.
//!
//! # Divergences from upstream
//! - **Server-side tools without a result.** Python wraps a call to a tool the
//!   client did not declare as a `server_function_call` and registers a
//!   placeholder on its own function-invocation layer, so the call is never
//!   run locally, and unwraps it afterwards. Here the function-invocation
//!   loop belongs to the wrapping `Agent` and a chat client cannot reach it.
//!   A server tool that comes with its `TOOL_CALL_RESULT` is still left alone
//!   (the loop skips resolved calls), and that is every tool the server
//!   executes. A server tool call with **no** result (a server-side approval
//!   pause) gets the loop's unknown-tool handling when the caller also has
//!   local tools.
//! - **State carriers** are marked on the [`Message`] (its
//!   `additional_properties`), because content items here carry no
//!   additional properties. The deprecated implicit carrier (a trailing
//!   unmarked JSON attachment) is not recognised.
//! - **Wire spelling.** Requests use the protocol's camelCase throughout
//!   (`threadId`, `toolCalls`, `toolCallId`); Python sends `thread_id` and
//!   `tool_calls`. Servers accept both.
//! - Function calls carry no `agui_thread_id` property: content items have no
//!   additional properties.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine as _;
use futures::stream::BoxStream;
use futures::StreamExt;
use reqwest::header::ACCEPT;
use serde_json::{json, Map, Value};

use agent_framework_core::client::{ChatClient, ChatStream};
use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::tools::{ToolDefinition, ToolKind};
use agent_framework_core::types::{
    Annotation, ChatOptions, ChatResponse, ChatResponseUpdate, Content, DataContent, ErrorContent,
    FinishReason, FunctionCallContent, FunctionResultContent, Message, Role, TextContent,
    TextReasoningContent, ToolMode,
};

use super::{arguments_delta, event_type, result_content, RunAgentInput};

/// The `Message::additional_properties` key that marks a request state
/// carrier (see [`state_carrier`]). Same key as upstream's `STATE_CARRIER_KEY`.
pub const STATE_CARRIER_KEY: &str = "__ag_ui_state_carrier__";

/// The metadata key a `CUSTOM` event is recorded under.
const CUSTOM_EVENT_KEY: &str = "ag_ui_custom_event";

/// Build a message that carries AG-UI shared state rather than chat content.
///
/// Put it anywhere in the history passed to [`AgUiChatClient`]: the client
/// removes every carrier from the messages it sends and posts the newest
/// carrier's object as the run's `state`. Ordinary `application/json`
/// attachments without the marker stay chat content.
///
/// Mirrors upstream's `state_carrier`, which returns a marked content item;
/// here the marker sits on the message (see the module docs).
///
/// ```
/// use agent_framework_core::types::Message;
/// use agent_framework_hosting::agui::state_carrier;
/// use serde_json::json;
///
/// let state = json!({ "selected_tab": "sales" });
/// let messages = vec![
///     Message::user("Update the dashboard"),
///     state_carrier(state.as_object().unwrap().clone()),
/// ];
/// # let _ = messages;
/// ```
pub fn state_carrier(state: Map<String, Value>) -> Message {
    let bytes = serde_json::to_vec(&Value::Object(state)).unwrap_or_else(|_| b"{}".to_vec());
    let mut message = Message::with_contents(
        Role::user(),
        vec![Content::Data(DataContent::from_bytes(
            &bytes,
            "application/json",
        ))],
    );
    message
        .additional_properties
        .insert(STATE_CARRIER_KEY.to_string(), Value::Bool(true));
    message
}

fn is_state_carrier(message: &Message) -> bool {
    message.contents.len() == 1
        && message.additional_properties.get(STATE_CARRIER_KEY) == Some(&Value::Bool(true))
}

/// Decode a base64 `application/json` data URI into a JSON object.
fn decode_json_state(content: &Content) -> Option<Map<String, Value>> {
    let Content::Data(data) = content else {
        return None;
    };
    let (header, payload) = data.uri.strip_prefix("data:")?.split_once(',')?;
    let mut parts = header.split(';');
    if parts.next() != Some("application/json") || !parts.any(|p| p == "base64") {
        return None;
    }
    let bytes = match base64::engine::general_purpose::STANDARD.decode(payload) {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::warn!("failed to decode AG-UI state carrier: {e}");
            return None;
        }
    };
    match serde_json::from_slice::<Value>(&bytes) {
        Ok(Value::Object(state)) => Some(state),
        Ok(_) => {
            tracing::warn!("AG-UI state carrier JSON must decode to an object");
            None
        }
        Err(e) => {
            tracing::warn!("failed to parse AG-UI state carrier: {e}");
            None
        }
    }
}

/// Split state carriers off the history: the messages to send, and the newest
/// valid carrier's state.
fn extract_state(messages: Vec<Message>) -> (Vec<Message>, Option<Map<String, Value>>) {
    let mut state = None;
    let mut to_send = Vec::with_capacity(messages.len());
    for message in messages {
        if is_state_carrier(&message) {
            if let Some(decoded) = decode_json_state(&message.contents[0]) {
                state = Some(decoded);
            }
            continue;
        }
        to_send.push(message);
    }
    (to_send, state)
}

// ---------------------------------------------------------------------------
// Messages: framework -> AG-UI
// ---------------------------------------------------------------------------

fn new_message_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// The AG-UI role for a framework role. `tool` is not here: a tool message is
/// emitted per function result. Anything else becomes `user`, as upstream's
/// `FRAMEWORK_TO_AGUI_ROLE.get(role, "user")` does.
fn agui_role(role: &Role) -> &'static str {
    match role.as_str() {
        Role::ASSISTANT => "assistant",
        Role::SYSTEM => "system",
        _ => "user",
    }
}

/// An AG-UI multimodal input part for a URI or data item.
fn media_part(uri: &str, media_type: Option<&str>, inline: bool) -> Value {
    let part_type = match media_type
        .and_then(|m| m.split('/').next())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("image") => "image",
        Some("audio") => "audio",
        Some("video") => "video",
        _ => "document",
    };
    let mut source = Map::new();
    let base64_payload = inline
        .then(|| {
            let (header, payload) = uri.split_once(',')?;
            header
                .split(';')
                .skip(1)
                .any(|p| p.eq_ignore_ascii_case("base64"))
                .then_some(payload)
        })
        .flatten();
    match base64_payload {
        Some(payload) => {
            source.insert("type".into(), json!("data"));
            source.insert("value".into(), json!(payload));
        }
        None => {
            source.insert("type".into(), json!("url"));
            source.insert("value".into(), json!(uri));
        }
    }
    if let Some(media_type) = media_type {
        source.insert("mimeType".into(), json!(media_type));
    }
    json!({ "type": part_type, "source": source })
}

/// Encode a run of text, media and call contents as an AG-UI message's
/// `content` and `toolCalls`. Media is kept only for user messages, which
/// then use the multimodal parts form.
fn encode_segment(contents: &[&Content], role: &str) -> (Value, Vec<Value>) {
    let mut text = String::new();
    let mut parts: Vec<Value> = Vec::new();
    let mut has_media = false;
    let mut tool_calls = Vec::new();
    for content in contents {
        match content {
            Content::Text(t) => {
                text.push_str(&t.text);
                if role == "user" {
                    parts.push(json!({ "type": "text", "text": t.text }));
                }
            }
            Content::Uri(u) if role == "user" => {
                parts.push(media_part(&u.uri, Some(&u.media_type), false));
                has_media = true;
            }
            Content::Data(d) if role == "user" => {
                parts.push(media_part(&d.uri, d.media_type.as_deref(), true));
                has_media = true;
            }
            Content::FunctionCall(fc) => tool_calls.push(tool_call_json(fc)),
            _ => {}
        }
    }
    let content = if has_media {
        Value::Array(parts)
    } else {
        Value::String(text)
    };
    (content, tool_calls)
}

fn tool_call_json(fc: &FunctionCallContent) -> Value {
    let arguments = arguments_delta(fc)
        .filter(|a| !a.trim().is_empty())
        .unwrap_or_else(|| "{}".to_string());
    json!({
        "id": fc.call_id,
        "type": "function",
        "function": { "name": fc.name, "arguments": arguments },
    })
}

fn is_segment_content(content: &Content, role: &str) -> bool {
    matches!(content, Content::Text(_) | Content::FunctionCall(_))
        || (role == "user" && matches!(content, Content::Uri(_) | Content::Data(_)))
}

/// Split a message that carries function results into ordered AG-UI
/// messages, keeping every call ahead of its result and no assistant message
/// between an open call and its result. A line-for-line port of upstream's
/// `_split_mixed_message_to_agui`, whose docstring gives the four rules.
fn split_mixed_message(
    msg: &Message,
    role: &str,
    unresolved: &mut HashSet<String>,
    out: &mut Vec<Value>,
) {
    struct Splitter<'a> {
        role: &'a str,
        source_id: Option<String>,
        segment: Vec<&'a Content>,
        segment_has_call: bool,
        segment_call_ids: HashSet<String>,
        queued: Vec<&'a FunctionResultContent>,
    }

    impl<'a> Splitter<'a> {
        fn next_id(&mut self) -> String {
            self.source_id.take().unwrap_or_else(new_message_id)
        }

        fn flush(&mut self, unresolved: &mut HashSet<String>, out: &mut Vec<Value>) {
            if self.segment.is_empty() {
                return;
            }
            let (content, tool_calls) = encode_segment(&self.segment, self.role);
            self.segment.clear();
            self.segment_has_call = false;
            self.segment_call_ids.clear();
            let empty_text = content.as_str().is_some_and(str::is_empty);
            if empty_text && tool_calls.is_empty() {
                return;
            }
            let mut message =
                json!({ "id": self.next_id(), "role": self.role, "content": content });
            if !tool_calls.is_empty() {
                for call in &tool_calls {
                    if let Some(id) = call["id"].as_str() {
                        unresolved.insert(id.to_string());
                    }
                }
                message["toolCalls"] = Value::Array(tool_calls);
            }
            out.push(message);
        }

        fn emit_result(
            &mut self,
            result: &FunctionResultContent,
            unresolved: &mut HashSet<String>,
            out: &mut Vec<Value>,
        ) {
            out.push(json!({
                "id": self.next_id(),
                "role": "tool",
                "content": result_content(result),
                "toolCallId": result.call_id,
            }));
            unresolved.remove(&result.call_id);
        }

        fn drain_queued(&mut self, unresolved: &mut HashSet<String>, out: &mut Vec<Value>) {
            if self.queued.is_empty() {
                return;
            }
            self.flush(unresolved, out);
            for result in std::mem::take(&mut self.queued) {
                self.emit_result(result, unresolved, out);
            }
        }
    }

    let mut splitter = Splitter {
        role,
        source_id: msg.message_id.clone().filter(|id| !id.is_empty()),
        segment: Vec::new(),
        segment_has_call: false,
        segment_call_ids: HashSet::new(),
        queued: Vec::new(),
    };

    for content in &msg.contents {
        if is_segment_content(content, role) {
            splitter.segment.push(content);
            if let Content::FunctionCall(fc) = content {
                splitter.segment_has_call = true;
                splitter.segment_call_ids.insert(fc.call_id.clone());
            }
        } else if let Content::FunctionResult(result) = content {
            if unresolved.contains(&result.call_id) {
                // Its call is already out and still open: the result can go now.
                splitter.emit_result(result, unresolved, out);
                if unresolved.is_empty() {
                    splitter.drain_queued(unresolved, out);
                }
            } else if splitter.segment_has_call && unresolved.is_empty() {
                // No older batch is open: flush the buffered calls first.
                splitter.flush(unresolved, out);
                splitter.emit_result(result, unresolved, out);
                splitter.drain_queued(unresolved, out);
            } else if splitter.segment_call_ids.contains(&result.call_id) {
                // Its call is buffered behind an open older batch: hold it.
                splitter.queued.push(result);
            } else {
                // Its call came from an earlier message: emit in place.
                splitter.emit_result(result, unresolved, out);
            }
        }
    }

    splitter.flush(unresolved, out);
    for result in std::mem::take(&mut splitter.queued) {
        splitter.emit_result(result, unresolved, out);
    }
}

/// Convert framework messages to AG-UI request messages.
///
/// Port of upstream's `agent_framework_messages_to_agui`:
/// - text becomes the `content` string; for a **user** message with URI or
///   data content, `content` is the ordered multimodal parts list instead
///   (`{type: "text", text}` and `{type: image|audio|video|document,
///   source: {type: "data"|"url", value, mimeType}}`);
/// - function calls become `toolCalls` (`{id, type: "function", function:
///   {name, arguments}}`, arguments as a JSON string);
/// - each function result becomes its own `tool` message (`{toolCallId,
///   content}`), ordered so that no result precedes its call and no assistant
///   message separates an open call from its result;
/// - every message gets an `id`: its own `message_id`, else a fresh UUID.
pub fn messages_to_agui(messages: &[Message]) -> Vec<Value> {
    let mut out = Vec::new();
    // Calls emitted so far whose results have not been, carried across
    // messages so a mixed message never slips a new assistant segment
    // between an earlier call and its result.
    let mut unresolved: HashSet<String> = HashSet::new();

    for msg in messages {
        let role = agui_role(&msg.role);
        if msg
            .contents
            .iter()
            .any(|c| matches!(c, Content::FunctionResult(_)))
        {
            split_mixed_message(msg, role, &mut unresolved, &mut out);
            continue;
        }

        let segment: Vec<&Content> = msg.contents.iter().collect();
        let (content, tool_calls) = encode_segment(&segment, role);
        let id = msg
            .message_id
            .clone()
            .filter(|id| !id.is_empty())
            .unwrap_or_else(new_message_id);
        let mut message = json!({ "id": id, "role": role, "content": content });
        // A non-tool message resets the open set to the calls it introduces.
        unresolved.clear();
        if !tool_calls.is_empty() {
            for call in &tool_calls {
                if let Some(id) = call["id"].as_str() {
                    unresolved.insert(id.to_string());
                }
            }
            message["toolCalls"] = Value::Array(tool_calls);
        }
        out.push(message);
    }
    out
}

/// Declare the request's function tools to the server (metadata only; the
/// implementations stay here). Hosted tools are not AG-UI tools and are left
/// out, as upstream leaves out anything that is not a function tool.
fn tools_to_agui(tools: &[ToolDefinition]) -> Vec<Value> {
    tools
        .iter()
        .filter(|t| t.kind == ToolKind::Function)
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Interrupt and resume options
// ---------------------------------------------------------------------------

/// Python truthiness for a JSON value.
fn truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// The snake_case spellings of interrupt fields upstream's model accepts by
/// name, with their protocol aliases.
const INTERRUPT_FIELD_ALIASES: &[(&str, &str)] = &[
    ("tool_call_id", "toolCallId"),
    ("response_schema", "responseSchema"),
    ("expires_at", "expiresAt"),
];

/// Normalise interrupt descriptors to the canonical protocol shape. An entry
/// without a `reason` derives one from its legacy `type` (`request_info` or
/// none becomes `input_required`), as upstream's
/// `_serialize_available_interrupts` does.
fn serialize_available_interrupts(value: &Value) -> Result<Value> {
    let Value::Array(entries) = value else {
        return Err(Error::Configuration(
            "AG-UI `available_interrupts` must be a list".to_string(),
        ));
    };
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let Value::Object(entry) = entry else {
            return Err(Error::Configuration(
                "each AG-UI interrupt must be an object".to_string(),
            ));
        };
        let mut entry = entry.clone();
        if !entry.contains_key("reason") {
            match entry.remove("type") {
                None | Some(Value::Null) => {
                    entry.insert("reason".into(), json!("input_required"));
                }
                Some(Value::String(t)) if t == "request_info" => {
                    entry.insert("reason".into(), json!("input_required"));
                }
                Some(Value::String(t)) => {
                    entry.insert("reason".into(), json!(t));
                }
                Some(_) => {}
            }
        }
        for (snake, camel) in INTERRUPT_FIELD_ALIASES {
            if let Some(v) = entry.remove(*snake) {
                entry.entry(camel.to_string()).or_insert(v);
            }
        }
        entry.retain(|_, v| !v.is_null());
        if !entry.contains_key("id") || !entry.contains_key("reason") {
            return Err(Error::Configuration(
                "each AG-UI interrupt needs an `id` and a `reason`".to_string(),
            ));
        }
        out.push(Value::Object(entry));
    }
    Ok(Value::Array(out))
}

/// One resume entry in canonical form (`{interruptId, status, payload?}`),
/// accepting upstream's legacy spellings (`id`, `interrupt_id`, `toolCallId`;
/// `value` or `response` for the payload).
fn serialize_resume_entry(entry: &Value) -> Result<Value> {
    let Value::Object(entry) = entry else {
        return Err(Error::Configuration(
            "each AG-UI resume entry must be an object".to_string(),
        ));
    };
    let interrupt_id = ["interruptId", "interrupt_id", "id", "toolCallId"]
        .iter()
        .filter_map(|k| entry.get(*k))
        .find(|v| truthy(v))
        .ok_or_else(|| {
            Error::Configuration("each AG-UI resume entry must include interruptId".to_string())
        })?;
    let interrupt_id = match interrupt_id {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    let status = entry
        .get("status")
        .filter(|v| truthy(v))
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
        .unwrap_or_else(|| "resolved".to_string());
    let payload = if let Some(p) = entry.get("payload") {
        p.clone()
    } else if let Some(v) = entry.get("value") {
        v.clone()
    } else if let Some(r) = entry.get("response") {
        r.clone()
    } else {
        Value::Object(
            entry
                .iter()
                .filter(|(k, _)| {
                    !matches!(
                        k.as_str(),
                        "id" | "interruptId" | "interrupt_id" | "toolCallId" | "type" | "status"
                    )
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    };
    let mut out = json!({ "interruptId": interrupt_id, "status": status });
    if (status != "cancelled" || truthy(&payload)) && !payload.is_null() {
        out["payload"] = payload;
    }
    Ok(out)
}

/// Normalise a resume payload as upstream's `_serialize_resume` does: a list
/// of entries, `{interrupts: [...]}` / `{interrupt: [...]}`, or one bare
/// entry all become a list of canonical entries; anything else passes
/// through.
fn serialize_resume(value: &Value) -> Result<Value> {
    match value {
        Value::Array(entries) => entries
            .iter()
            .map(serialize_resume_entry)
            .collect::<Result<Vec<_>>>()
            .map(Value::Array),
        Value::Object(map) => {
            for key in ["interrupts", "interrupt"] {
                if let Some(Value::Array(entries)) = map.get(key) {
                    return entries
                        .iter()
                        .map(serialize_resume_entry)
                        .collect::<Result<Vec<_>>>()
                        .map(Value::Array);
                }
            }
            if ["interruptId", "interrupt_id", "id", "toolCallId"]
                .iter()
                .any(|k| map.contains_key(*k))
            {
                return Ok(Value::Array(vec![serialize_resume_entry(value)?]));
            }
            Ok(value.clone())
        }
        other => Ok(other.clone()),
    }
}

fn first_property<'a>(options: &'a ChatOptions, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .filter_map(|k| options.additional_properties.get(*k))
        .find(|v| !v.is_null())
}

// ---------------------------------------------------------------------------
// Event conversion
// ---------------------------------------------------------------------------

/// Converts AG-UI events into [`ChatResponseUpdate`]s, one event at a time.
///
/// Port of upstream's `AGUIEventConverter`, plus the text-chunk, reasoning,
/// state and messages-snapshot events the Go provider handles. Stateful: it
/// tracks the open text message, every open tool call by `toolCallId` (so
/// argument deltas for parallel calls each land on their own call) and the
/// run's ids. Use one converter per run.
#[derive(Debug, Default, Clone)]
pub struct AgUiEventConverter {
    current_message_id: Option<String>,
    /// Open tool calls as `(toolCallId, name)`, oldest first. Several can be
    /// open at once when a server streams parallel calls.
    open_tool_calls: Vec<(String, String)>,
    /// The call the last `TOOL_CALL_CHUNK` named, so a later chunk that omits
    /// `toolCallId` continues it.
    last_chunk_tool_call_id: Option<String>,
    last_chunk_message_id: Option<String>,
    last_reasoning_message_id: Option<String>,
    thread_id: Option<String>,
    run_id: Option<String>,
}

fn str_field(event: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .filter_map(|k| event.get(*k))
        .find(|v| !v.is_null())
        .map(|v| match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        })
}

fn delta_of(event: &Value) -> String {
    event
        .get("delta")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn opt_value(v: &Option<String>) -> Value {
    v.as_ref().map_or(Value::Null, |s| Value::String(s.clone()))
}

fn assistant_update(contents: Vec<Content>) -> ChatResponseUpdate {
    ChatResponseUpdate {
        contents,
        role: Some(Role::assistant()),
        ..Default::default()
    }
}

/// JSON as a base64 data item, the way the Go provider surfaces state events.
fn tool_name_of(event: &Value) -> Option<String> {
    str_field(event, &["toolName", "toolCallName", "tool_call_name"])
}

/// One streamed fragment of call `id`; aggregation merges fragments by id.
fn call_fragment(id: String, name: String, arguments: String) -> Content {
    Content::FunctionCall(FunctionCallContent::new(
        id,
        name,
        Some(agent_framework_core::types::FunctionArguments::Raw(
            arguments,
        )),
    ))
}

fn json_data(value: &Value, media_type: &str) -> Content {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| b"null".to_vec());
    Content::Data(DataContent::from_bytes(&bytes, media_type))
}

impl AgUiEventConverter {
    /// A converter with no run in progress.
    pub fn new() -> Self {
        Self::default()
    }

    /// The thread id the server reported in `RUN_STARTED`, once seen.
    pub fn thread_id(&self) -> Option<&str> {
        self.thread_id.as_deref()
    }

    /// The run id the server reported in `RUN_STARTED`, once seen.
    pub fn run_id(&self) -> Option<&str> {
        self.run_id.as_deref()
    }

    fn run_ids(&self) -> HashMap<String, Value> {
        HashMap::from([
            ("thread_id".to_string(), opt_value(&self.thread_id)),
            ("run_id".to_string(), opt_value(&self.run_id)),
        ])
    }

    /// Convert one AG-UI event. Returns `None` for events that carry nothing
    /// a chat update can hold (`TEXT_MESSAGE_END`, `TOOL_CALL_END`, unknown
    /// types). The `type` match is case-insensitive.
    pub fn convert_event(&mut self, event: &Value) -> Option<ChatResponseUpdate> {
        let raw_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let kind = raw_type.to_ascii_uppercase();
        match kind.as_str() {
            event_type::RUN_STARTED => {
                self.thread_id = str_field(event, &["threadId", "thread_id"]);
                self.run_id = str_field(event, &["runId", "run_id"]);
                let mut update = assistant_update(Vec::new());
                update.additional_properties = self.run_ids();
                Some(update)
            }
            event_type::TEXT_MESSAGE_START => {
                self.current_message_id = str_field(event, &["messageId", "message_id"]);
                let mut update = assistant_update(Vec::new());
                update.message_id = self.current_message_id.clone();
                Some(update)
            }
            event_type::TEXT_MESSAGE_CONTENT => {
                let message_id = str_field(event, &["messageId", "message_id"]);
                if message_id != self.current_message_id {
                    self.current_message_id = message_id;
                }
                let mut update = assistant_update(vec![Content::text(delta_of(event))]);
                update.message_id = self.current_message_id.clone();
                Some(update)
            }
            event_type::TEXT_MESSAGE_CHUNK => {
                let delta = delta_of(event);
                if let Some(id) = str_field(event, &["messageId", "message_id"]) {
                    self.last_chunk_message_id = Some(id);
                }
                if delta.is_empty() {
                    return None;
                }
                let mut update = assistant_update(vec![Content::text(delta)]);
                update.message_id = self.last_chunk_message_id.clone();
                Some(update)
            }
            event_type::REASONING_MESSAGE_CONTENT | event_type::REASONING_MESSAGE_CHUNK => {
                let delta = delta_of(event);
                if let Some(id) = str_field(event, &["messageId", "message_id"]) {
                    self.last_reasoning_message_id = Some(id);
                }
                if delta.is_empty() {
                    return None;
                }
                let mut update =
                    assistant_update(vec![Content::TextReasoning(TextReasoningContent {
                        text: delta,
                        ..Default::default()
                    })]);
                update.message_id = self.last_reasoning_message_id.clone();
                Some(update)
            }
            event_type::TEXT_MESSAGE_END => None,
            event_type::TOOL_CALL_START => {
                let id = str_field(event, &["toolCallId", "tool_call_id"]).unwrap_or_default();
                let name = tool_name_of(event).unwrap_or_default();
                self.open_call(id.clone(), name.clone());
                Some(assistant_update(vec![call_fragment(
                    id,
                    name,
                    String::new(),
                )]))
            }
            event_type::TOOL_CALL_ARGS => {
                // Fragments are keyed by `toolCallId`; aggregation merges each
                // into the call with that id, so interleaved deltas for
                // parallel calls stay with their own call. A delta with no id
                // continues the most recently opened call.
                let id = str_field(event, &["toolCallId", "tool_call_id"])
                    .or_else(|| self.open_tool_calls.last().map(|(id, _)| id.clone()))
                    .unwrap_or_default();
                let name = self.open_call_name(&id);
                Some(assistant_update(vec![call_fragment(
                    id,
                    name,
                    delta_of(event),
                )]))
            }
            event_type::TOOL_CALL_CHUNK => {
                // Shorthand for start/args/end: the first chunk of a call
                // names it (`toolCallId`, `toolCallName`), later chunks may
                // omit the id and continue the last chunked call. There is no
                // explicit end; the call simply stops receiving deltas.
                let id = str_field(event, &["toolCallId", "tool_call_id"])
                    .or_else(|| self.last_chunk_tool_call_id.clone());
                let Some(id) = id else {
                    tracing::warn!("ignoring TOOL_CALL_CHUNK with no toolCallId and no open call");
                    return None;
                };
                let name = tool_name_of(event);
                if !self.open_tool_calls.iter().any(|(open, _)| *open == id) {
                    self.open_call(id.clone(), name.clone().unwrap_or_default());
                }
                self.last_chunk_tool_call_id = Some(id.clone());
                let name = self.open_call_name(&id);
                Some(assistant_update(vec![call_fragment(
                    id,
                    name,
                    delta_of(event),
                )]))
            }
            event_type::TOOL_CALL_END => {
                match str_field(event, &["toolCallId", "tool_call_id"]) {
                    Some(id) => self.open_tool_calls.retain(|(open, _)| *open != id),
                    None => {
                        self.open_tool_calls.pop();
                    }
                }
                None
            }
            event_type::TOOL_CALL_RESULT => {
                let call_id = str_field(event, &["toolCallId", "tool_call_id"]).unwrap_or_default();
                let result = event
                    .get("result")
                    .filter(|v| !v.is_null())
                    .or_else(|| event.get("content"))
                    .cloned()
                    .filter(|v| !v.is_null());
                Some(ChatResponseUpdate {
                    contents: vec![Content::FunctionResult(FunctionResultContent::new(
                        call_id, result,
                    ))],
                    role: Some(Role::tool()),
                    message_id: str_field(event, &["messageId", "message_id"]),
                    ..Default::default()
                })
            }
            event_type::STATE_SNAPSHOT => {
                let snapshot = event.get("snapshot").cloned().unwrap_or(Value::Null);
                Some(assistant_update(vec![json_data(
                    &snapshot,
                    "application/json",
                )]))
            }
            event_type::STATE_DELTA => {
                let delta = event.get("delta").cloned().unwrap_or(Value::Null);
                Some(assistant_update(vec![json_data(
                    &delta,
                    "application/json-patch+json",
                )]))
            }
            event_type::MESSAGES_SNAPSHOT => {
                let mut update = assistant_update(Vec::new());
                update.additional_properties = self.run_ids();
                update.additional_properties.insert(
                    "ag_ui_messages_snapshot".to_string(),
                    event.get("messages").cloned().unwrap_or(Value::Null),
                );
                Some(update)
            }
            event_type::RUN_FINISHED => {
                let mut props = self.run_ids();
                if let Some(interrupt) = event.get("interrupt") {
                    props.insert("interrupt".into(), interrupt.clone());
                }
                if let Some(outcome) = event.get("outcome") {
                    props.insert("outcome".into(), outcome.clone());
                    match outcome {
                        Value::Object(o) => {
                            if o.get("type").and_then(Value::as_str) == Some("interrupt") {
                                if let Some(list @ Value::Array(_)) = o.get("interrupts") {
                                    props.insert("interrupts".into(), list.clone());
                                }
                            }
                        }
                        _ => tracing::warn!(
                            "RUN_FINISHED outcome should be an object; preserving the raw outcome"
                        ),
                    }
                }
                if let Some(result) = event.get("result") {
                    props.insert("result".into(), result.clone());
                }
                let mut update = assistant_update(Vec::new());
                update.finish_reason = Some(FinishReason::new(FinishReason::STOP));
                update.additional_properties = props;
                Some(update)
            }
            event_type::RUN_ERROR => {
                let message = event
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown error")
                    .to_string();
                let mut update = assistant_update(vec![Content::Error(ErrorContent {
                    message: Some(message),
                    error_code: Some("RUN_ERROR".to_string()),
                    details: str_field(event, &["code"]),
                })]);
                update.additional_properties = self.run_ids();
                Some(update)
            }
            event_type::CUSTOM | "CUSTOM_EVENT" => Some(self.custom_event(event, raw_type)),
            _ => None,
        }
    }

    /// Record `id` as open, replacing an earlier call with the same id.
    fn open_call(&mut self, id: String, name: String) {
        self.open_tool_calls.retain(|(open, _)| *open != id);
        self.open_tool_calls.push((id, name));
    }

    /// The name the open call `id` started with, or empty when it is not open.
    fn open_call_name(&self, id: &str) -> String {
        self.open_tool_calls
            .iter()
            .rev()
            .find(|(open, _)| open == id)
            .map(|(_, name)| name.clone())
            .unwrap_or_default()
    }

    /// A `CUSTOM` event stays inspectable as metadata; an `annotations` event
    /// (`{messageId, annotations: [...]}`) also restores those annotations on
    /// the message it names.
    fn custom_event(&self, event: &Value, raw_type: &str) -> ChatResponseUpdate {
        let name = event.get("name").cloned().unwrap_or(Value::Null);
        let value = event.get("value").cloned().unwrap_or(Value::Null);
        let mut update = assistant_update(Vec::new());
        update.additional_properties = self.run_ids();
        update.additional_properties.insert(
            CUSTOM_EVENT_KEY.to_string(),
            json!({ "name": name, "value": value, "raw_type": raw_type }),
        );
        if name.as_str() == Some("annotations") {
            let message_id = value
                .get("messageId")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            let annotations = value
                .get("annotations")
                .and_then(Value::as_array)
                .and_then(|list| {
                    list.iter()
                        .map(|a| {
                            a.is_object()
                                .then(|| serde_json::from_value::<Annotation>(a.clone()).ok())
                                .flatten()
                        })
                        .collect::<Option<Vec<_>>>()
                });
            match (message_id, annotations) {
                (Some(message_id), Some(annotations)) => {
                    update.message_id = Some(message_id.to_string());
                    if !annotations.is_empty() {
                        let mut text = TextContent::new("");
                        text.annotations = Some(annotations);
                        update.contents = vec![Content::Text(text)];
                    }
                }
                _ => tracing::warn!(
                    "invalid annotations custom event: expected messageId and an annotations array"
                ),
            }
        }
        update
    }
}

/// Whether `update` is an annotation batch from [`AgUiEventConverter`].
fn annotation_batch(update: &ChatResponseUpdate) -> Option<(String, Vec<Annotation>)> {
    let custom = update.additional_properties.get(CUSTOM_EVENT_KEY)?;
    if custom.get("name").and_then(Value::as_str) != Some("annotations") {
        return None;
    }
    let message_id = update.message_id.clone()?;
    let annotations = update
        .contents
        .iter()
        .filter_map(|c| match c {
            Content::Text(t) => t.annotations.clone(),
            _ => None,
        })
        .flatten()
        .collect();
    Some((message_id, annotations))
}

/// Aggregate a run's updates into a response, attaching annotation batches to
/// the messages they name rather than letting them start new text (port of
/// upstream's `_finalize_agui_response`). Messages left with no content —
/// a `TEXT_MESSAGE_START` that never received text — are dropped.
fn finalize_response(updates: Vec<ChatResponseUpdate>) -> ChatResponse {
    let mut batches = Vec::new();
    let mut aggregatable = Vec::with_capacity(updates.len());
    let mut all_properties: Vec<HashMap<String, Value>> = Vec::with_capacity(updates.len());
    for update in updates {
        all_properties.push(update.additional_properties.clone());
        match annotation_batch(&update) {
            Some(batch) => batches.push(batch),
            None => aggregatable.push(update),
        }
    }
    let mut response = ChatResponse::from_updates(aggregatable);
    // Annotation events are kept out of the messages, but their metadata
    // still applies in stream order.
    response.additional_properties.clear();
    for props in all_properties {
        response.additional_properties.extend(props);
    }

    for (message_id, annotations) in batches {
        if annotations.is_empty() {
            continue;
        }
        let existing = response
            .messages
            .iter_mut()
            .find(|m| m.message_id.as_deref() == Some(message_id.as_str()));
        let Some(message) = existing else {
            let mut text = TextContent::new("");
            text.annotations = Some(annotations);
            let mut message = Message::with_contents(Role::assistant(), vec![Content::Text(text)]);
            message.message_id = Some(message_id);
            response.messages.push(message);
            continue;
        };
        match message.contents.iter_mut().find_map(|c| match c {
            Content::Text(t) => Some(t),
            _ => None,
        }) {
            Some(text) => text
                .annotations
                .get_or_insert_with(Vec::new)
                .extend(annotations),
            None => {
                let mut text = TextContent::new("");
                text.annotations = Some(annotations);
                message.contents.push(Content::Text(text));
            }
        }
    }
    response.messages.retain(|m| !m.contents.is_empty());
    response
}

// ---------------------------------------------------------------------------
// SSE
// ---------------------------------------------------------------------------

/// An incremental Server-Sent Events decoder: bytes in, the `data` payload of
/// each complete event out. Follows the SSE framing rules (`data` lines
/// joined with `\n`, one optional space after the colon, comments and other
/// fields ignored, LF or CRLF line ends) and is safe across chunk boundaries,
/// including multi-byte characters split between chunks.
#[derive(Debug, Default)]
struct SseDecoder {
    utf8: Utf8StreamDecoder,
    line: String,
    data: Option<String>,
    /// The last line ended in a CR at the end of a chunk; an LF opening the
    /// next chunk completes that CRLF rather than ending an empty line.
    skip_lf: bool,
}

impl SseDecoder {
    fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let text = self.utf8.push(bytes);
        let mut out = Vec::new();
        self.line.push_str(&text);
        loop {
            if self.skip_lf && !self.line.is_empty() {
                if self.line.starts_with('\n') {
                    self.line.remove(0);
                }
                self.skip_lf = false;
            }
            // SSE lines end in CRLF, LF or a bare CR.
            let Some(pos) = self.line.find(['\r', '\n']) else {
                break;
            };
            let mut line: String = self.line.drain(..=pos).collect();
            if line.pop() == Some('\r') {
                self.skip_lf = true;
            }
            self.process_line(&line, &mut out);
        }
        out
    }

    /// End of stream: a final event without its blank line still counts.
    fn finish(&mut self) -> Vec<String> {
        let tail = self.utf8.flush();
        let mut out = self.push(tail.as_bytes());
        let rest = std::mem::take(&mut self.line);
        if !rest.is_empty() {
            self.process_line(&rest, &mut out);
        }
        if let Some(data) = self.data.take() {
            out.push(data);
        }
        out
    }

    fn process_line(&mut self, line: &str, out: &mut Vec<String>) {
        if line.is_empty() {
            if let Some(data) = self.data.take() {
                out.push(data);
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        if field == "data" {
            match &mut self.data {
                Some(data) => {
                    data.push('\n');
                    data.push_str(value);
                }
                None => self.data = Some(value.to_string()),
            }
        }
    }
}

/// Folds the run-id-only update of `RUN_STARTED` (no content, no message id,
/// no finish reason, nothing but `thread_id` / `run_id`) into the next
/// update, so aggregation does not open an empty assistant message for it.
/// Any other metadata-only update (`CUSTOM`, `MESSAGES_SNAPSHOT`) is passed
/// on at once: folding those would let consecutive ones overwrite each other
/// under the same key and hold them back until the next content arrives.
#[derive(Debug, Default)]
struct MetadataFold {
    pending: Option<HashMap<String, Value>>,
}

impl MetadataFold {
    fn push(&mut self, mut update: ChatResponseUpdate) -> Option<ChatResponseUpdate> {
        let run_ids_only = update.contents.is_empty()
            && update.message_id.is_none()
            && update.finish_reason.is_none()
            && update
                .additional_properties
                .keys()
                .all(|k| k == "thread_id" || k == "run_id");
        if run_ids_only {
            self.pending
                .get_or_insert_with(HashMap::new)
                .extend(update.additional_properties);
            return None;
        }
        if let Some(earlier) = self.pending.take() {
            for (key, value) in earlier {
                update.additional_properties.entry(key).or_insert(value);
            }
        }
        Some(update)
    }

    fn finish(&mut self) -> Option<ChatResponseUpdate> {
        self.pending.take().map(|props| ChatResponseUpdate {
            role: Some(Role::assistant()),
            additional_properties: props,
            ..Default::default()
        })
    }
}

/// One run's open event stream, decoded into updates as bytes arrive.
struct RunStream {
    bytes: BoxStream<'static, reqwest::Result<bytes::Bytes>>,
    sse: SseDecoder,
    converter: AgUiEventConverter,
    fold: MetadataFold,
    ready: VecDeque<ChatResponseUpdate>,
    done: bool,
}

impl RunStream {
    fn handle_data(&mut self, data: &str) {
        let event = match serde_json::from_str::<Value>(data) {
            Ok(event @ Value::Object(_)) => event,
            Ok(_) => {
                tracing::warn!("ignoring AG-UI SSE data that is not a JSON object");
                return;
            }
            Err(e) => {
                tracing::warn!("failed to parse AG-UI SSE data: {e}");
                return;
            }
        };
        let kind = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        tracing::debug!(kind, "AG-UI event");
        if let Some(update) = self.converter.convert_event(&event) {
            if let Some(update) = self.fold.push(update) {
                self.ready.push_back(update);
            }
        }
    }

    async fn next_update(&mut self) -> Option<Result<ChatResponseUpdate>> {
        loop {
            if let Some(update) = self.ready.pop_front() {
                return Some(Ok(update));
            }
            if self.done {
                return None;
            }
            match self.bytes.next().await {
                Some(Ok(chunk)) => {
                    for data in self.sse.push(chunk.as_ref()) {
                        self.handle_data(&data);
                    }
                }
                Some(Err(e)) => {
                    self.done = true;
                    return Some(Err(Error::service(format!(
                        "AG-UI event stream failed: {e}"
                    ))));
                }
                None => {
                    self.done = true;
                    for data in self.sse.finish() {
                        self.handle_data(&data);
                    }
                    if let Some(update) = self.fold.finish() {
                        self.ready.push_back(update);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The client
// ---------------------------------------------------------------------------

/// A [`ChatClient`] for a remote AG-UI server (upstream `AGUIChatClient`).
///
/// Each request posts one [`RunAgentInput`] to the endpoint and reads the
/// server's event stream. See the [module docs](crate::agui::client) for how
/// requests are built and events mapped.
///
/// ```no_run
/// use agent_framework_core::agent::Agent;
/// use agent_framework_hosting::agui::AgUiChatClient;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let client = AgUiChatClient::new("http://localhost:8888/");
/// let agent = Agent::builder(client).name("remote").build();
/// let response = agent.run_once("Hello!").await?;
/// println!("{}", response.text());
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AgUiChatClient {
    endpoint: String,
    http: reqwest::Client,
}

/// Request ids plus the body for one run.
struct PreparedRun {
    input: RunAgentInput,
}

impl AgUiChatClient {
    /// How long the default HTTP client waits to connect, and between reads
    /// of the event stream. Matches upstream's 60-second default.
    pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

    /// A client for the AG-UI server at `endpoint`, with its own HTTP client
    /// using [`Self::DEFAULT_TIMEOUT`]. A trailing `/` is dropped, as upstream
    /// drops it. The HTTP client keeps no cookies between runs.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self::with_timeout(endpoint, Self::DEFAULT_TIMEOUT)
    }

    /// Like [`Self::new`], with `timeout` for connecting and for each read of
    /// the event stream. It does not bound the whole run, which streams for
    /// as long as the server keeps sending.
    pub fn with_timeout(endpoint: impl Into<String>, timeout: Duration) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(timeout)
            .read_timeout(timeout)
            .build()
            .unwrap_or_else(|e| {
                tracing::warn!("falling back to a default HTTP client: {e}");
                reqwest::Client::new()
            });
        Self::with_http_client(endpoint, http)
    }

    /// A client that sends its requests through `http`, for custom headers,
    /// authentication, proxies or TLS. The caller's client keeps whatever
    /// cookie behaviour it was built with; scope cookie-based authentication
    /// to one principal rather than sharing it across users.
    pub fn with_http_client(endpoint: impl Into<String>, http: reqwest::Client) -> Self {
        let endpoint = endpoint.into();
        let endpoint = endpoint.trim_end_matches('/').to_string();
        Self { endpoint, http }
    }

    /// The server endpoint requests are posted to.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    fn prepare(&self, messages: Vec<Message>, options: &ChatOptions) -> Result<PreparedRun> {
        let (mut to_send, state) = extract_state(messages);
        if let Some(instructions) = options.instructions.as_deref().filter(|s| !s.is_empty()) {
            to_send.insert(0, Message::system(instructions));
        }

        let thread_id = options
            .metadata
            .as_ref()
            .and_then(|m| m.get("thread_id"))
            .filter(|id| !id.is_empty())
            .cloned()
            .unwrap_or_else(|| format!("thread_{}", uuid::Uuid::new_v4().simple()));
        let run_id = format!("run_{}", uuid::Uuid::new_v4().simple());

        let available_interrupts =
            first_property(options, &["available_interrupts", "availableInterrupts"])
                .map(serialize_available_interrupts)
                .transpose()?;
        let resume = first_property(options, &["resume"])
            .map(serialize_resume)
            .transpose()?;
        let forwarded_props = first_property(
            options,
            &["forwarded_props", "forwardedProps", "forward_props"],
        )
        .cloned();
        let context = match first_property(options, &["context"]) {
            Some(Value::Array(items)) => items.clone(),
            Some(_) => {
                return Err(Error::Configuration(
                    "AG-UI `context` must be a list of {description, value} objects".to_string(),
                ))
            }
            None => Vec::new(),
        };

        let input = RunAgentInput {
            thread_id: Some(thread_id),
            run_id: Some(run_id),
            messages: messages_to_agui(&to_send),
            // The function-invocation loop's final request disables tools
            // with `ToolMode::None` but leaves `options.tools` in place;
            // declaring them anyway would let the server call them again.
            tools: if matches!(options.tool_choice, Some(ToolMode::None)) {
                Vec::new()
            } else {
                tools_to_agui(&options.tools)
            },
            state: state.map(Value::Object),
            context,
            forwarded_props,
            available_interrupts,
            resume,
        };
        Ok(PreparedRun { input })
    }

    async fn open_run(&self, run: PreparedRun) -> Result<RunStream> {
        tracing::debug!(
            endpoint = %self.endpoint,
            thread_id = ?run.input.thread_id,
            run_id = ?run.input.run_id,
            messages = run.input.messages.len(),
            tools = run.input.tools.len(),
            has_state = run.input.state.is_some(),
            "posting AG-UI run"
        );
        let response = self
            .http
            .post(&self.endpoint)
            .header(ACCEPT, "text/event-stream")
            .json(&run.input)
            .send()
            .await
            .map_err(|e| Error::service(format!("AG-UI request failed: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::service_status(
                status.as_u16(),
                format!("AG-UI server returned {status}: {body}"),
                None,
            ));
        }
        Ok(RunStream {
            bytes: response.bytes_stream().boxed(),
            sse: SseDecoder::default(),
            converter: AgUiEventConverter::new(),
            fold: MetadataFold::default(),
            ready: VecDeque::new(),
            done: false,
        })
    }
}

#[async_trait]
impl ChatClient for AgUiChatClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        let run = self.prepare(messages, &options)?;
        let mut stream = self.open_run(run).await?;
        let mut updates = Vec::new();
        while let Some(update) = stream.next_update().await {
            updates.push(update?);
        }
        Ok(finalize_response(updates))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let run = self.prepare(messages, &options)?;
        let stream = self.open_run(run).await?;
        Ok(futures::stream::unfold(stream, |mut stream| async move {
            let item = stream.next_update().await?;
            Some((item, stream))
        })
        .boxed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{FunctionArguments, UriContent};

    fn convert_all(events: &[Value]) -> Vec<ChatResponseUpdate> {
        let mut converter = AgUiEventConverter::new();
        events
            .iter()
            .filter_map(|e| converter.convert_event(e))
            .collect()
    }

    fn calls(update: &ChatResponseUpdate) -> Vec<&FunctionCallContent> {
        update
            .contents
            .iter()
            .filter_map(Content::as_function_call)
            .collect()
    }

    fn raw_args(fc: &FunctionCallContent) -> &str {
        match &fc.arguments {
            Some(FunctionArguments::Raw(s)) => s,
            _ => panic!("expected raw arguments"),
        }
    }

    // --- event conversion (ports of test_event_converters.py) -------------

    #[test]
    fn run_started_records_ids() {
        let mut c = AgUiEventConverter::new();
        let u = c
            .convert_event(&json!({"type": "RUN_STARTED", "threadId": "t1", "runId": "r1"}))
            .unwrap();
        assert_eq!(u.role, Some(Role::assistant()));
        assert!(u.contents.is_empty());
        assert_eq!(u.additional_properties["thread_id"], "t1");
        assert_eq!(u.additional_properties["run_id"], "r1");
        assert_eq!(c.thread_id(), Some("t1"));
        assert_eq!(c.run_id(), Some("r1"));
    }

    #[test]
    fn text_message_start_content_end() {
        let mut c = AgUiEventConverter::new();
        let start = c
            .convert_event(
                &json!({"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"}),
            )
            .unwrap();
        assert_eq!(start.message_id.as_deref(), Some("m1"));
        assert!(start.contents.is_empty());
        let a = c
            .convert_event(
                &json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Hello"}),
            )
            .unwrap();
        let b = c
            .convert_event(
                &json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": " world"}),
            )
            .unwrap();
        assert_eq!(a.message_id.as_deref(), Some("m1"));
        assert_eq!(a.contents, vec![Content::text("Hello")]);
        assert_eq!(b.contents, vec![Content::text(" world")]);
        assert!(c
            .convert_event(&json!({"type": "TEXT_MESSAGE_END", "messageId": "m1"}))
            .is_none());
    }

    #[test]
    fn tool_call_start_accepts_every_name_spelling() {
        for key in ["toolName", "toolCallName", "tool_call_name"] {
            let mut c = AgUiEventConverter::new();
            let u = c
                .convert_event(
                    &json!({"type": "TOOL_CALL_START", "toolCallId": "c1", key: "search"}),
                )
                .unwrap();
            let fc = calls(&u)[0];
            assert_eq!(fc.call_id, "c1");
            assert_eq!(fc.name, "search");
            assert_eq!(raw_args(fc), "");
        }
    }

    #[test]
    fn tool_call_args_stream_into_one_call() {
        let updates = convert_all(&[
            json!({"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "search"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": "{\"q\":"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": "\"rust\"}"}),
            json!({"type": "TOOL_CALL_END", "toolCallId": "c1"}),
        ]);
        assert_eq!(updates.len(), 3);
        assert_eq!(raw_args(calls(&updates[1])[0]), "{\"q\":");
        assert_eq!(calls(&updates[2])[0].name, "search");
        let response = ChatResponse::from_updates(updates);
        let fc = response.function_calls()[0].clone();
        assert_eq!(fc.call_id, "c1");
        assert_eq!(fc.parse_arguments().unwrap()["q"], "rust");
    }

    #[test]
    fn parallel_tool_call_args_land_on_their_own_call() {
        let mut c = AgUiEventConverter::new();
        c.convert_event(
            &json!({"type": "TOOL_CALL_START", "toolCallId": "safe", "toolName": "safe_tool"}),
        );
        c.convert_event(
            &json!({"type": "TOOL_CALL_START", "toolCallId": "danger", "toolName": "danger_tool"}),
        );
        // Args for the earlier, still-open call go to that call, not the
        // most recent one.
        let u = c
            .convert_event(&json!({"type": "TOOL_CALL_ARGS", "toolCallId": "safe", "delta": "{\"amount\": 100}"}))
            .unwrap();
        assert_eq!(calls(&u)[0].call_id, "safe");
        assert_eq!(calls(&u)[0].name, "safe_tool");
        c.convert_event(&json!({"type": "TOOL_CALL_END", "toolCallId": "safe"}));
        // Ending one call leaves the other open.
        let u = c
            .convert_event(
                &json!({"type": "TOOL_CALL_ARGS", "toolCallId": "danger", "delta": "{}"}),
            )
            .unwrap();
        assert_eq!(calls(&u)[0].call_id, "danger");
        assert_eq!(calls(&u)[0].name, "danger_tool");
        c.convert_event(&json!({"type": "TOOL_CALL_END", "toolCallId": "danger"}));
        assert!(c.open_tool_calls.is_empty());
    }

    #[test]
    fn interleaved_parallel_args_aggregate_per_call() {
        let updates = convert_all(&[
            json!({"type": "TOOL_CALL_START", "toolCallId": "a", "toolCallName": "first"}),
            json!({"type": "TOOL_CALL_START", "toolCallId": "b", "toolCallName": "second"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "a", "delta": "{\"x\":"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "b", "delta": "{\"y\":"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "a", "delta": "1}"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "b", "delta": "2}"}),
            json!({"type": "TOOL_CALL_END", "toolCallId": "a"}),
            json!({"type": "TOOL_CALL_END", "toolCallId": "b"}),
        ]);
        let response = ChatResponse::from_updates(updates);
        let calls = response.function_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].call_id.as_str(), calls[0].name.as_str()),
            ("a", "first")
        );
        assert_eq!(calls[0].parse_arguments().unwrap()["x"], 1);
        assert_eq!(
            (calls[1].call_id.as_str(), calls[1].name.as_str()),
            ("b", "second")
        );
        assert_eq!(calls[1].parse_arguments().unwrap()["y"], 2);
    }

    #[test]
    fn tool_call_args_without_an_id_continue_the_latest_open_call() {
        let mut c = AgUiEventConverter::new();
        c.convert_event(
            &json!({"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "f"}),
        );
        let u = c
            .convert_event(&json!({"type": "TOOL_CALL_ARGS", "delta": "{}"}))
            .unwrap();
        assert_eq!(calls(&u)[0].call_id, "c1");
        assert_eq!(calls(&u)[0].name, "f");
    }

    #[test]
    fn tool_call_chunks_decode_into_calls() {
        let updates = convert_all(&[
            json!({"type": "TOOL_CALL_CHUNK", "toolCallId": "c1", "toolCallName": "search", "parentMessageId": "m1", "delta": "{\"q\":"}),
            // A follow-up chunk may omit the id and name.
            json!({"type": "TOOL_CALL_CHUNK", "delta": "\"rust\"}"}),
            json!({"type": "TOOL_CALL_CHUNK", "toolCallId": "c2", "toolCallName": "lookup"}),
            json!({"type": "TOOL_CALL_CHUNK", "toolCallId": "c2", "delta": "{}"}),
        ]);
        assert_eq!(updates.len(), 4);
        assert_eq!(calls(&updates[1])[0].call_id, "c1");
        assert_eq!(calls(&updates[1])[0].name, "search");
        assert_eq!(calls(&updates[3])[0].name, "lookup");
        let response = ChatResponse::from_updates(updates);
        let calls = response.function_calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            (calls[0].call_id.as_str(), calls[0].name.as_str()),
            ("c1", "search")
        );
        assert_eq!(calls[0].parse_arguments().unwrap()["q"], "rust");
        assert_eq!(
            (calls[1].call_id.as_str(), calls[1].name.as_str()),
            ("c2", "lookup")
        );
        assert_eq!(raw_args(calls[1]), "{}");
    }

    #[test]
    fn tool_call_chunk_without_any_call_is_ignored() {
        let mut c = AgUiEventConverter::new();
        assert!(c
            .convert_event(&json!({"type": "TOOL_CALL_CHUNK", "delta": "{}"}))
            .is_none());
    }

    #[test]
    fn tool_call_result_carries_its_message_id() {
        for key in ["messageId", "message_id"] {
            let mut c = AgUiEventConverter::new();
            let u = c
                .convert_event(&json!({"type": "TOOL_CALL_RESULT", "toolCallId": "c1", "content": "42", key: "tool-msg-1"}))
                .unwrap();
            assert_eq!(u.message_id.as_deref(), Some("tool-msg-1"));
        }
        let mut c = AgUiEventConverter::new();
        let u = c
            .convert_event(
                &json!({"type": "TOOL_CALL_RESULT", "toolCallId": "c1", "content": "42"}),
            )
            .unwrap();
        assert!(u.message_id.is_none());
    }

    #[test]
    fn tool_call_result_accepts_both_id_spellings() {
        for key in ["toolCallId", "tool_call_id"] {
            let mut c = AgUiEventConverter::new();
            let u = c
                .convert_event(&json!({"type": "TOOL_CALL_RESULT", key: "c1", "content": "42", "messageId": "x"}))
                .unwrap();
            assert_eq!(u.role, Some(Role::tool()));
            let Content::FunctionResult(fr) = &u.contents[0] else {
                panic!("expected a result")
            };
            assert_eq!(fr.call_id, "c1");
            assert_eq!(fr.result, Some(json!("42")));
        }
    }

    #[test]
    fn run_finished_carries_interrupts_outcome_and_result() {
        let mut c = AgUiEventConverter::new();
        c.convert_event(&json!({"type": "RUN_STARTED", "threadId": "t1", "runId": "r1"}));
        let u = c
            .convert_event(&json!({
                "type": "RUN_FINISHED", "threadId": "t1", "runId": "r1",
                "interrupt": [{"id": "i1"}],
                "outcome": {"type": "interrupt", "interrupts": [{"id": "i1", "reason": "tool_call"}]},
                "result": {"ok": true},
            }))
            .unwrap();
        assert_eq!(u.finish_reason, Some(FinishReason::new(FinishReason::STOP)));
        let p = &u.additional_properties;
        assert_eq!(p["thread_id"], "t1");
        assert_eq!(p["interrupt"], json!([{"id": "i1"}]));
        assert_eq!(
            p["interrupts"],
            json!([{"id": "i1", "reason": "tool_call"}])
        );
        assert_eq!(p["result"], json!({"ok": true}));

        // A success outcome is kept but adds no interrupts; a non-object one
        // is preserved as-is.
        let ok = c
            .convert_event(&json!({"type": "RUN_FINISHED", "outcome": {"type": "success"}}))
            .unwrap();
        assert!(!ok.additional_properties.contains_key("interrupts"));
        let odd = c
            .convert_event(&json!({"type": "RUN_FINISHED", "outcome": "done"}))
            .unwrap();
        assert_eq!(odd.additional_properties["outcome"], "done");
    }

    #[test]
    fn run_error_becomes_error_content() {
        let mut c = AgUiEventConverter::new();
        let u = c
            .convert_event(&json!({"type": "RUN_ERROR", "message": "boom", "code": "E1"}))
            .unwrap();
        let Content::Error(e) = &u.contents[0] else {
            panic!("expected error content")
        };
        assert_eq!(e.message.as_deref(), Some("boom"));
        assert_eq!(e.error_code.as_deref(), Some("RUN_ERROR"));
        assert_eq!(e.details.as_deref(), Some("E1"));
    }

    #[test]
    fn unknown_events_are_ignored_and_types_are_case_insensitive() {
        let mut c = AgUiEventConverter::new();
        assert!(c.convert_event(&json!({"type": "SOMETHING_NEW"})).is_none());
        assert!(c.convert_event(&json!({})).is_none());
        let u = c
            .convert_event(&json!({"type": "text_message_content", "messageId": "m", "delta": "x"}))
            .unwrap();
        assert_eq!(u.contents, vec![Content::text("x")]);
    }

    #[test]
    fn custom_events_are_kept_as_metadata() {
        for kind in ["CUSTOM", "CUSTOM_EVENT"] {
            let mut c = AgUiEventConverter::new();
            let u = c
                .convert_event(&json!({"type": kind, "name": "progress", "value": {"pct": 50}}))
                .unwrap();
            assert!(u.contents.is_empty());
            assert_eq!(
                u.additional_properties[CUSTOM_EVENT_KEY],
                json!({"name": "progress", "value": {"pct": 50}, "raw_type": kind})
            );
        }
    }

    #[test]
    fn annotations_custom_event_restores_annotations() {
        let annotations = json!([{
            "type": "citation",
            "title": "Document",
            "url": "https://example.com/doc.pdf",
        }]);
        let mut c = AgUiEventConverter::new();
        c.current_message_id = Some("another".into());
        let u = c
            .convert_event(&json!({
                "type": "CUSTOM", "name": "annotations",
                "value": {"messageId": "msg_citations", "annotations": annotations},
            }))
            .unwrap();
        assert_eq!(u.message_id.as_deref(), Some("msg_citations"));
        let Content::Text(t) = &u.contents[0] else {
            panic!("expected text")
        };
        assert_eq!(t.text, "");
        let restored = serde_json::to_value(t.annotations.as_ref().unwrap()).unwrap();
        assert_eq!(restored, annotations);
    }

    #[test]
    fn malformed_annotations_keep_metadata_only() {
        for value in [
            Value::Null,
            json!({"annotations": []}),
            json!({"messageId": 123, "annotations": []}),
            json!({"messageId": "m", "annotations": "invalid"}),
            json!({"messageId": "m", "annotations": ["invalid"]}),
        ] {
            let mut c = AgUiEventConverter::new();
            let u = c
                .convert_event(&json!({"type": "CUSTOM", "name": "annotations", "value": value}))
                .unwrap();
            assert!(u.contents.is_empty());
            assert_eq!(u.additional_properties[CUSTOM_EVENT_KEY]["value"], value);
        }
    }

    #[test]
    fn late_annotations_attach_to_their_message() {
        let updates = convert_all(&[
            json!({"type": "TEXT_MESSAGE_START", "messageId": "m1"}),
            json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "First"}),
            json!({"type": "TEXT_MESSAGE_START", "messageId": "m2"}),
            json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m2", "delta": "Second"}),
            json!({"type": "TOOL_CALL_RESULT", "toolCallId": "call-1", "result": "done"}),
            json!({"type": "CUSTOM", "name": "annotations",
                   "value": {"messageId": "m1", "annotations": [{"type": "citation", "url": "https://example.com/first"}]}}),
        ]);
        let response = finalize_response(updates);
        let ids: Vec<_> = response
            .messages
            .iter()
            .map(|m| m.message_id.clone())
            .collect();
        assert_eq!(ids.iter().filter(|i| i.as_deref() == Some("m1")).count(), 1);
        assert_eq!(ids.iter().filter(|i| i.is_none()).count(), 1);
        let first = &response.messages[0];
        let Content::Text(t) = &first.contents[0] else {
            panic!("expected text")
        };
        assert_eq!(t.text, "First");
        assert_eq!(
            t.annotations.as_ref().unwrap()[0].url.as_deref(),
            Some("https://example.com/first")
        );
        assert_eq!(
            response.additional_properties[CUSTOM_EVENT_KEY]["name"],
            "annotations"
        );
    }

    #[test]
    fn streamed_annotations_survive_plain_aggregation() {
        // What a caller of `get_streaming_response` sees when it aggregates
        // the updates itself, without the client's `finalize_response`.
        let mut c = AgUiEventConverter::new();
        let mut fold = MetadataFold::default();
        let updates: Vec<_> = [
            json!({"type": "RUN_STARTED", "threadId": "t", "runId": "r"}),
            json!({"type": "TEXT_MESSAGE_START", "messageId": "m1"}),
            json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Cited"}),
            json!({"type": "CUSTOM", "name": "annotations",
                   "value": {"messageId": "m1", "annotations": [{"type": "citation", "url": "https://example.com"}]}}),
            json!({"type": "RUN_FINISHED", "threadId": "t", "runId": "r"}),
        ]
        .iter()
        .filter_map(|e| c.convert_event(e))
        .filter_map(|u| fold.push(u))
        .collect();
        let response = ChatResponse::from_updates(updates);
        let message = response
            .messages
            .iter()
            .find(|m| m.message_id.as_deref() == Some("m1"))
            .unwrap();
        let Content::Text(t) = &message.contents[0] else {
            panic!("expected text")
        };
        assert_eq!(t.text, "Cited");
        assert_eq!(
            t.annotations.as_ref().unwrap()[0].url.as_deref(),
            Some("https://example.com")
        );
    }

    #[test]
    fn full_conversation_flow_aggregates() {
        let updates = convert_all(&[
            json!({"type": "RUN_STARTED", "threadId": "t1", "runId": "r1"}),
            json!({"type": "TEXT_MESSAGE_START", "messageId": "m1", "role": "assistant"}),
            json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "Let me "}),
            json!({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m1", "delta": "check."}),
            json!({"type": "TEXT_MESSAGE_END", "messageId": "m1"}),
            json!({"type": "TOOL_CALL_START", "toolCallId": "c1", "toolCallName": "a"}),
            json!({"type": "TOOL_CALL_ARGS", "toolCallId": "c1", "delta": "{}"}),
            json!({"type": "TOOL_CALL_END", "toolCallId": "c1"}),
            json!({"type": "TOOL_CALL_START", "toolCallId": "c2", "toolCallName": "b"}),
            json!({"type": "TOOL_CALL_END", "toolCallId": "c2"}),
            json!({"type": "RUN_FINISHED", "threadId": "t1", "runId": "r1"}),
        ]);
        let response = finalize_response(updates);
        assert_eq!(response.text(), "Let me check.");
        let names: Vec<_> = response
            .function_calls()
            .iter()
            .map(|c| c.name.clone())
            .collect();
        assert_eq!(names, ["a", "b"]);
        assert_eq!(response.additional_properties["thread_id"], "t1");
        assert_eq!(
            response.finish_reason,
            Some(FinishReason::new(FinishReason::STOP))
        );
    }

    #[test]
    fn state_and_snapshot_events() {
        let updates = convert_all(&[
            json!({"type": "STATE_SNAPSHOT", "snapshot": {"count": 1}}),
            json!({"type": "STATE_DELTA", "delta": [{"op": "replace", "path": "/count", "value": 2}]}),
            json!({"type": "MESSAGES_SNAPSHOT", "messages": [{"id": "1", "role": "user", "content": "hi"}]}),
            json!({"type": "REASONING_MESSAGE_CONTENT", "messageId": "r", "delta": "thinking"}),
            json!({"type": "TEXT_MESSAGE_CHUNK", "messageId": "c", "delta": "a"}),
            json!({"type": "TEXT_MESSAGE_CHUNK", "delta": "b"}),
        ]);
        let Content::Data(snapshot) = &updates[0].contents[0] else {
            panic!("expected data")
        };
        assert_eq!(snapshot.media_type.as_deref(), Some("application/json"));
        assert_eq!(
            decode_json_state(&updates[0].contents[0]).unwrap()["count"],
            1
        );
        let Content::Data(delta) = &updates[1].contents[0] else {
            panic!("expected data")
        };
        assert_eq!(
            delta.media_type.as_deref(),
            Some("application/json-patch+json")
        );
        assert_eq!(
            updates[2].additional_properties["ag_ui_messages_snapshot"][0]["content"],
            "hi"
        );
        assert!(
            matches!(&updates[3].contents[0], Content::TextReasoning(r) if r.text == "thinking")
        );
        // A chunk without an id continues the previous chunk's message.
        assert_eq!(updates[5].message_id.as_deref(), Some("c"));
    }

    // --- request building (ports of test_ag_ui_client.py) ----------------

    #[test]
    fn converts_messages_to_agui() {
        let mut assistant = Message::assistant("Let me check.");
        assistant.message_id = Some("msg_123".into());
        let out = messages_to_agui(&[Message::user("What is the weather?"), assistant]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"], "What is the weather?");
        assert!(out[0]["id"].as_str().is_some_and(|s| !s.is_empty()));
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[1]["content"], "Let me check.");
        assert_eq!(out[1]["id"], "msg_123");
    }

    #[test]
    fn user_media_becomes_ordered_parts_and_assistant_media_is_dropped() {
        let mut message = Message::with_contents(
            Role::user(),
            vec![
                Content::text("describe this"),
                Content::Uri(UriContent {
                    uri: "https://example.com/cat.png".into(),
                    media_type: "image/png".into(),
                }),
                Content::Data(DataContent::from_bytes(b"abc", "image/png")),
                Content::Data(DataContent::from_bytes(b"%PDF", "application/pdf")),
            ],
        );
        message.message_id = Some("msg-request".into());
        let out = messages_to_agui(&[message]);
        assert_eq!(
            out[0],
            json!({
                "id": "msg-request",
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image", "source": {"type": "url", "value": "https://example.com/cat.png", "mimeType": "image/png"}},
                    {"type": "image", "source": {"type": "data", "value": "YWJj", "mimeType": "image/png"}},
                    {"type": "document", "source": {"type": "data", "value": "JVBERg==", "mimeType": "application/pdf"}},
                ],
            })
        );

        let assistant = Message::with_contents(
            Role::assistant(),
            vec![
                Content::text("see"),
                Content::Data(DataContent::from_bytes(b"abc", "image/png")),
            ],
        );
        assert_eq!(messages_to_agui(&[assistant])[0]["content"], "see");
    }

    #[test]
    fn tool_calls_and_results_round_trip_in_order() {
        let call = FunctionCallContent::new(
            "call_a",
            "get_weather",
            Some(FunctionArguments::Raw("{\"city\":\"Seattle\"}".into())),
        );
        let no_args = FunctionCallContent::new("call_b", "now", None);
        let assistant = Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(call), Content::FunctionCall(no_args)],
        );
        let tool = Message::with_contents(
            Role::tool(),
            vec![
                Content::FunctionResult(FunctionResultContent::new("call_a", Some(json!("Sunny")))),
                Content::FunctionResult(FunctionResultContent::new(
                    "call_b",
                    Some(json!({"h": 9})),
                )),
            ],
        );
        let out = messages_to_agui(&[assistant, tool]);
        let roles: Vec<_> = out.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(roles, ["assistant", "tool", "tool"]);
        assert_eq!(
            out[0]["toolCalls"],
            json!([
                {"id": "call_a", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Seattle\"}"}},
                {"id": "call_b", "type": "function", "function": {"name": "now", "arguments": "{}"}},
            ])
        );
        assert_eq!(out[1]["toolCallId"], "call_a");
        assert_eq!(out[1]["content"], "Sunny");
        assert_eq!(out[2]["content"], "{\"h\":9}");
        assert_ne!(out[1]["id"], out[2]["id"]);
    }

    fn call(id: &str) -> Content {
        Content::FunctionCall(FunctionCallContent::new(
            id,
            format!("f{id}"),
            Some(FunctionArguments::Raw("{}".into())),
        ))
    }

    fn result(id: &str) -> Content {
        Content::FunctionResult(FunctionResultContent::new(
            id,
            Some(json!(format!("r{id}"))),
        ))
    }

    fn mixed(id: &str, contents: Vec<Content>) -> Message {
        let mut m = Message::with_contents(Role::assistant(), contents);
        m.message_id = Some(id.into());
        m
    }

    fn shape(out: &[Value]) -> Vec<String> {
        out.iter()
            .map(|m| match m["role"].as_str().unwrap() {
                "tool" => format!("tool({})", m["toolCallId"].as_str().unwrap()),
                role => {
                    let calls: Vec<_> = m["toolCalls"]
                        .as_array()
                        .map(|c| c.iter().map(|c| c["id"].as_str().unwrap()).collect())
                        .unwrap_or_default();
                    if calls.is_empty() {
                        format!("{role}:{}", m["content"].as_str().unwrap_or_default())
                    } else {
                        format!("{role}({})", calls.join(","))
                    }
                }
            })
            .collect()
    }

    #[test]
    fn mixed_message_keeps_call_before_result_and_defers_text() {
        let out = messages_to_agui(&[mixed("m", vec![call("a"), result("a")])]);
        assert_eq!(shape(&out), ["assistant(a)", "tool(a)"]);
        assert_eq!(out[0]["id"], "m");
        assert_ne!(out[1]["id"], "m");

        let out = messages_to_agui(&[mixed(
            "m",
            vec![call("a"), result("a"), Content::text("It is sunny.")],
        )]);
        assert_eq!(
            shape(&out),
            ["assistant(a)", "tool(a)", "assistant:It is sunny."]
        );
        let ids: HashSet<_> = out.iter().map(|m| m["id"].to_string()).collect();
        assert_eq!(ids.len(), 3);

        // Text ahead of a result goes after it.
        let out = messages_to_agui(&[mixed("m", vec![Content::text("Here."), result("w")])]);
        assert_eq!(shape(&out), ["tool(w)", "assistant:Here."]);
    }

    #[test]
    fn interleaved_parallel_batch_keeps_calls_adjacent_to_results() {
        let out = messages_to_agui(&[mixed(
            "i",
            vec![
                call("a"),
                call("b"),
                result("a"),
                call("c"),
                result("b"),
                result("c"),
            ],
        )]);
        assert_eq!(
            shape(&out),
            [
                "assistant(a,b)",
                "tool(a)",
                "tool(b)",
                "assistant(c)",
                "tool(c)"
            ]
        );
        assert_eq!(out[0]["id"], "i");
    }

    #[test]
    fn pending_call_carries_across_messages() {
        let out = messages_to_agui(&[
            mixed("m1", vec![call("a")]),
            mixed("m2", vec![call("c"), result("a"), result("c")]),
        ]);
        assert_eq!(
            shape(&out),
            ["assistant(a)", "tool(a)", "assistant(c)", "tool(c)"]
        );
        let out = messages_to_agui(&[
            mixed("m1", vec![call("a")]),
            mixed("m2", vec![Content::text("Let me check."), result("a")]),
        ]);
        assert_eq!(
            shape(&out),
            ["assistant(a)", "tool(a)", "assistant:Let me check."]
        );
    }

    #[test]
    fn result_for_buffered_call_waits_for_its_call() {
        let out = messages_to_agui(&[mixed(
            "q",
            vec![
                call("a"),
                call("b"),
                result("a"),
                call("c"),
                result("c"),
                result("b"),
            ],
        )]);
        assert_eq!(
            shape(&out),
            [
                "assistant(a,b)",
                "tool(a)",
                "tool(b)",
                "assistant(c)",
                "tool(c)"
            ]
        );
    }

    #[test]
    fn none_result_becomes_empty_content() {
        let msg = Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                "c", None,
            ))],
        );
        assert_eq!(messages_to_agui(&[msg])[0]["content"], "");
    }

    #[test]
    fn state_carriers_are_extracted_and_latest_wins() {
        let mut first = Map::new();
        first.insert("v".into(), json!(1));
        let mut second = Map::new();
        second.insert("v".into(), json!(2));
        let messages = vec![
            state_carrier(first),
            Message::user("Hello"),
            state_carrier(second),
        ];
        let (rest, state) = extract_state(messages);
        assert_eq!(rest.len(), 1);
        assert_eq!(state.unwrap()["v"], 2);

        // An unmarked JSON attachment stays a chat message.
        let plain = Message::with_contents(
            Role::user(),
            vec![Content::Data(DataContent::from_bytes(
                b"{\"a\":1}",
                "application/json",
            ))],
        );
        let (rest, state) = extract_state(vec![plain]);
        assert_eq!(rest.len(), 1);
        assert!(state.is_none());

        // A carrier with bad JSON is dropped without supplying state.
        let mut bad = state_carrier(Map::new());
        bad.contents = vec![Content::Data(DataContent::from_bytes(
            b"not json",
            "application/json",
        ))];
        let (rest, state) = extract_state(vec![bad]);
        assert!(rest.is_empty());
        assert!(state.is_none());
        assert!(extract_state(Vec::new()).1.is_none());
    }

    #[test]
    fn legacy_interrupts_and_resume_are_normalised() {
        assert_eq!(
            serialize_available_interrupts(&json!([{"id": "req_1", "type": "request_info"}]))
                .unwrap(),
            json!([{"id": "req_1", "reason": "input_required"}])
        );
        assert_eq!(
            serialize_available_interrupts(&json!([{
                "id": "approval_1", "reason": "tool_call", "tool_call_id": "call_1",
                "response_schema": {"type": "object"}
            }]))
            .unwrap(),
            json!([{"id": "approval_1", "reason": "tool_call", "toolCallId": "call_1", "responseSchema": {"type": "object"}}])
        );
        assert_eq!(
            serialize_resume(&json!({"interrupts": [{"id": "req_1", "value": "approved"}]}))
                .unwrap(),
            json!([{"interruptId": "req_1", "status": "resolved", "payload": "approved"}])
        );
        assert_eq!(
            serialize_resume(&json!({"interruptId": "a", "approved": true})).unwrap(),
            json!([{"interruptId": "a", "status": "resolved", "payload": {"approved": true}}])
        );
        assert_eq!(
            serialize_resume(&json!([{"interruptId": "a", "status": "cancelled"}])).unwrap(),
            json!([{"interruptId": "a", "status": "cancelled"}])
        );
        assert!(serialize_resume(&json!([{"status": "resolved"}])).is_err());
        assert!(serialize_resume(&json!(["x"])).is_err());
        assert_eq!(serialize_resume(&json!("opaque")).unwrap(), json!("opaque"));
    }

    #[test]
    fn thread_id_comes_from_metadata_or_is_generated() {
        let client = AgUiChatClient::new("http://localhost:8888/");
        assert_eq!(client.endpoint(), "http://localhost:8888");
        let options = ChatOptions {
            metadata: Some(HashMap::from([(
                "thread_id".to_string(),
                "t-9".to_string(),
            )])),
            ..Default::default()
        };
        let run = client.prepare(vec![Message::user("hi")], &options).unwrap();
        assert_eq!(run.input.thread_id.as_deref(), Some("t-9"));
        assert!(run.input.run_id.unwrap().starts_with("run_"));

        let run = client
            .prepare(vec![Message::user("hi")], &ChatOptions::default())
            .unwrap();
        assert!(run.input.thread_id.unwrap().starts_with("thread_"));
    }

    // --- SSE --------------------------------------------------------------

    #[test]
    fn sse_decoder_handles_split_chunks_multiline_data_and_comments() {
        let mut d = SseDecoder::default();
        let mut out = d.push(b": keep-alive\r\ndata: {\"a\":");
        assert!(out.is_empty());
        out.extend(d.push(b"1}\r\n\r\nevent: x\ndata:one\ndata: two\n\n"));
        let euro = "data: \"€\"\n\n".as_bytes();
        out.extend(d.push(&euro[..8]));
        out.extend(d.push(&euro[8..]));
        out.extend(d.push(b"data: tail"));
        out.extend(d.finish());
        assert_eq!(out, ["{\"a\":1}", "one\ntwo", "\"€\"", "tail"]);
    }

    #[test]
    fn sse_decoder_accepts_bare_cr_line_ends() {
        let mut d = SseDecoder::default();
        let out = d.push(b"data: 1\r\rdata: 2\r\r");
        assert_eq!(out, ["1", "2"]);
        assert!(d.finish().is_empty());
    }

    #[test]
    fn sse_decoder_joins_crlf_split_across_chunks() {
        let mut d = SseDecoder::default();
        // The CR ends one chunk and its LF opens the next: one line end, not
        // two, so the event is not dispatched before its second data line.
        let mut out = d.push(b"data: a\r");
        out.extend(d.push(b"\ndata: b\r"));
        assert!(out.is_empty());
        // The leading LF completes the split CRLF; the CR after it is the
        // blank line that dispatches the event.
        out.extend(d.push(b"\n\r"));
        assert_eq!(out, ["a\nb"]);
        // And the LF completing that CRLF is not a second blank line that
        // would swallow the next event's first field.
        out = d.push(b"\ndata: z\n\n");
        assert_eq!(out, ["z"]);
        // Mixed line ends in one stream.
        out = d.push(b"data: c\n\ndata: d\r\n\r\ndata: e\r\r");
        assert_eq!(out, ["c", "d", "e"]);
    }

    #[test]
    fn metadata_only_updates_fold_into_the_next() {
        let mut fold = MetadataFold::default();
        let mut started = assistant_update(Vec::new());
        started
            .additional_properties
            .insert("thread_id".into(), json!("t"));
        assert!(fold.push(started).is_none());
        let text = fold
            .push(assistant_update(vec![Content::text("hi")]))
            .unwrap();
        assert_eq!(text.additional_properties["thread_id"], "t");
        assert!(fold.finish().is_none());
    }

    #[test]
    fn consecutive_custom_and_snapshot_events_are_each_delivered() {
        let mut c = AgUiEventConverter::new();
        let mut fold = MetadataFold::default();
        let events = [
            json!({"type": "RUN_STARTED", "threadId": "t", "runId": "r"}),
            json!({"type": "CUSTOM", "name": "progress", "value": 1}),
            json!({"type": "CUSTOM", "name": "progress", "value": 2}),
            json!({"type": "MESSAGES_SNAPSHOT", "messages": [{"id": "a"}]}),
            json!({"type": "MESSAGES_SNAPSHOT", "messages": [{"id": "b"}]}),
        ];
        let out: Vec<_> = events
            .iter()
            .filter_map(|e| c.convert_event(e))
            .filter_map(|u| fold.push(u))
            .collect();
        assert_eq!(out.len(), 4, "each event is delivered at once");
        assert_eq!(out[0].additional_properties[CUSTOM_EVENT_KEY]["value"], 1);
        assert_eq!(out[1].additional_properties[CUSTOM_EVENT_KEY]["value"], 2);
        assert_eq!(
            out[2].additional_properties["ag_ui_messages_snapshot"][0]["id"],
            "a"
        );
        assert_eq!(
            out[3].additional_properties["ag_ui_messages_snapshot"][0]["id"],
            "b"
        );
        assert!(out.iter().all(|u| u.additional_properties["run_id"] == "r"));
        assert!(fold.finish().is_none());
    }

    #[test]
    fn tool_mode_none_declares_no_tools() {
        let client = AgUiChatClient::new("http://localhost:8888/");
        let tool = ToolDefinition {
            name: "lookup".into(),
            description: "d".into(),
            parameters: json!({"type": "object"}),
            kind: ToolKind::Function,
            approval_mode: Default::default(),
            executor: None,
        };
        let mut options = ChatOptions {
            tools: vec![tool],
            ..Default::default()
        };
        for (choice, declared) in [
            (None, 1),
            (Some(ToolMode::Auto), 1),
            (Some(ToolMode::None), 0),
        ] {
            options.tool_choice = choice;
            let run = client.prepare(vec![Message::user("hi")], &options).unwrap();
            assert_eq!(run.input.tools.len(), declared);
        }
    }
}
