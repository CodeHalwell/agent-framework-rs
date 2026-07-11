//! Response and streaming-update types for chat clients and agents.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::content::{Content, FunctionApprovalRequestContent, FunctionCallContent, UsageDetails};
use super::message::{ChatMessage, Role};
use super::options::ResponseFormat;
use crate::error::{Error, Result};

/// Reason a chat response finished. Open value wrapper, like `FinishReason`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FinishReason(pub String);

impl FinishReason {
    pub const CONTENT_FILTER: &'static str = "content_filter";
    pub const LENGTH: &'static str = "length";
    pub const STOP: &'static str = "stop";
    pub const TOOL_CALLS: &'static str = "tool_calls";

    pub fn new(v: impl Into<String>) -> Self {
        FinishReason(v.into())
    }
    pub fn stop() -> Self {
        FinishReason(Self::STOP.into())
    }
    pub fn tool_calls() -> Self {
        FinishReason(Self::TOOL_CALLS.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// A full (non-streaming) response from a chat client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponse {
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finish_reason: Option<FinishReason>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage_details: Option<UsageDetails>,
    /// Parsed structured-output value, when a response format was requested.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value: Option<Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
}

impl ChatResponse {
    /// Build a response from a single assistant text message.
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            messages: vec![ChatMessage::assistant(text)],
            ..Default::default()
        }
    }

    /// The concatenated text of all messages (newline-joined, trimmed).
    pub fn text(&self) -> String {
        self.messages
            .iter()
            .map(ChatMessage::text)
            .collect::<Vec<_>>()
            .join("\n")
            .trim()
            .to_string()
    }

    /// All function-call content items across the response's messages.
    pub fn function_calls(&self) -> Vec<&FunctionCallContent> {
        self.messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_call)
            .collect()
    }

    /// All pending user-input (function-approval) requests across the messages.
    ///
    /// Mirrors Python `ChatResponse.user_input_requests`.
    pub fn user_input_requests(&self) -> Vec<&FunctionApprovalRequestContent> {
        self.messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_approval_request)
            .collect()
    }

    /// Parse the response's concatenated text into a structured value.
    ///
    /// Mirrors Python's `response.value`: the message text is treated as JSON
    /// and deserialized into `T`.
    pub fn parse_json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.text())
            .map_err(|e| Error::Serialization(format!("failed to parse structured output: {e}")))
    }

    /// Aggregate a stream of updates into a full response.
    pub fn from_updates(updates: Vec<ChatResponseUpdate>) -> Self {
        Self::from_updates_with_format(updates, None)
    }

    /// Aggregate a stream of updates into a full response, then auto-populate
    /// [`ChatResponse::value`] from the aggregated text when a structured
    /// `response_format` was requested.
    ///
    /// Mirrors Python `ChatResponse.from_chat_response_updates(...,
    /// output_format_type=...)`, whose final step calls `try_parse_value`.
    /// This is the streaming counterpart of the auto-fill that
    /// [`crate::client::FunctionInvokingChatClient`] performs for
    /// non-streaming responses.
    pub fn from_updates_with_format(
        updates: Vec<ChatResponseUpdate>,
        response_format: Option<&ResponseFormat>,
    ) -> Self {
        let mut resp = ChatResponse::default();
        for u in updates {
            resp.absorb_update(u);
        }
        resp.finalize();
        resp.try_parse_value(response_format);
        resp
    }

    /// Alias for [`ChatResponse::from_updates`], matching Python's
    /// `ChatResponse.from_chat_response_updates`.
    pub fn from_chat_response_updates(updates: Vec<ChatResponseUpdate>) -> Self {
        Self::from_updates(updates)
    }

    /// Auto-populate [`ChatResponse::value`] from the response text when a
    /// structured `response_format` (JSON object / JSON schema) was requested
    /// and `value` is not already set.
    ///
    /// Mirrors Python's `try_parse_value` (`_types.py:2551-2557`): the
    /// concatenated response text is parsed as JSON. A parse failure is *not*
    /// an error — it is logged at `debug` and `value` is left as `None` (the
    /// model may still be mid-way to producing valid JSON, or the format was
    /// advisory).
    pub fn try_parse_value(&mut self, response_format: Option<&ResponseFormat>) {
        if self.value.is_some() {
            return;
        }
        let wants_json = matches!(
            response_format,
            Some(ResponseFormat::JsonObject) | Some(ResponseFormat::JsonSchema { .. })
        );
        if !wants_json {
            return;
        }
        let text = self.text();
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => self.value = Some(v),
            Err(e) => tracing::debug!("failed to parse structured-output value from text: {e}"),
        }
    }

    /// Merge a single streaming update into this response in place.
    ///
    /// Mirrors `_process_update`: updates are coalesced into a single
    /// assistant message keyed by `message_id`, function-call fragments are
    /// merged, and usage is accumulated.
    pub fn absorb_update(&mut self, update: ChatResponseUpdate) {
        if let Some(r) = &update.response_id {
            self.response_id.get_or_insert_with(|| r.clone());
        }
        if let Some(c) = &update.conversation_id {
            self.conversation_id.get_or_insert_with(|| c.clone());
        }
        if let Some(m) = &update.model_id {
            self.model_id.get_or_insert_with(|| m.clone());
        }
        if let Some(cr) = &update.created_at {
            self.created_at.get_or_insert_with(|| cr.clone());
        }
        if update.finish_reason.is_some() {
            self.finish_reason = update.finish_reason.clone();
        }

        let role = update.role.clone().unwrap_or_else(Role::assistant);
        // Find or create the target message by message_id.
        let idx = match &update.message_id {
            Some(mid) => self
                .messages
                .iter()
                .position(|m| m.message_id.as_deref() == Some(mid.as_str())),
            None => self.messages.iter().rposition(|m| m.role == role),
        };
        let msg_idx = match idx {
            Some(i) => i,
            None => {
                let mut m = ChatMessage::with_contents(role, Vec::new());
                m.message_id = update.message_id.clone();
                m.author_name = update.author_name.clone();
                self.messages.push(m);
                self.messages.len() - 1
            }
        };

        // Incorporate the update's identity fields into the (possibly
        // pre-existing) target message when later updates carry them, mirroring
        // Python's `_process_update` (`_types.py:2185-2190`).
        if let Some(author) = &update.author_name {
            self.messages[msg_idx].author_name = Some(author.clone());
        }
        if let Some(r) = &update.role {
            self.messages[msg_idx].role = r.clone();
        }
        if let Some(mid) = &update.message_id {
            self.messages[msg_idx].message_id = Some(mid.clone());
        }

        // Merge the update's `additional_properties` onto the response
        // (`_types.py:2218-2221`).
        for (k, v) in &update.additional_properties {
            self.additional_properties.insert(k.clone(), v.clone());
        }

        for content in update.contents {
            match content {
                Content::Usage(u) => {
                    let entry = self.usage_details.get_or_insert_with(UsageDetails::new);
                    entry.add_assign(&u.details);
                }
                Content::FunctionCall(fc) => {
                    // Merge with an existing partial call of the same call_id.
                    let existing = self.messages[msg_idx].contents.iter_mut().find_map(|c| {
                        if let Content::FunctionCall(e) = c {
                            if e.call_id == fc.call_id || fc.call_id.is_empty() {
                                return Some(e);
                            }
                        }
                        None
                    });
                    match existing {
                        Some(e) => {
                            let _ = e.merge(&fc);
                        }
                        None => self.messages[msg_idx]
                            .contents
                            .push(Content::FunctionCall(fc)),
                    }
                }
                other => self.messages[msg_idx].contents.push(other),
            }
        }
    }

    /// Coalesce adjacent text fragments in every message.
    pub fn finalize(&mut self) {
        for msg in &mut self.messages {
            coalesce_text(&mut msg.contents);
        }
    }
}

fn coalesce_text(contents: &mut Vec<Content>) {
    let mut out: Vec<Content> = Vec::with_capacity(contents.len());
    for c in contents.drain(..) {
        match (out.last_mut(), &c) {
            (Some(Content::Text(prev)), Content::Text(cur)) => prev.text.push_str(&cur.text),
            (Some(Content::TextReasoning(prev)), Content::TextReasoning(cur)) => {
                prev.text.push_str(&cur.text)
            }
            _ => out.push(c),
        }
    }
    *contents = out;
}

/// A single streaming chunk from a chat client.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatResponseUpdate {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub author_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finish_reason: Option<FinishReason>,
    /// Provider-specific metadata for this update. Merged onto
    /// [`ChatResponse::additional_properties`] during aggregation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
    /// The raw provider payload this update was decoded from, if retained.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw_representation: Option<Value>,
}

impl ChatResponseUpdate {
    /// A text-only update from the assistant.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            contents: vec![Content::text(text)],
            role: Some(Role::assistant()),
            ..Default::default()
        }
    }

    /// The concatenated text of this update.
    pub fn text_content(&self) -> String {
        self.contents
            .iter()
            .filter_map(Content::as_text)
            .collect::<String>()
    }
}

/// A full response from an agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRunResponse {
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub response_id: Option<String>,
    /// Service-side conversation id, when the backing service manages the
    /// conversation (e.g. Responses API `previous_response_id`, Azure AI
    /// thread id). `ChatAgent` persists it onto the [`AgentThread`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub conversation_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub usage_details: Option<UsageDetails>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub value: Option<Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
}

impl AgentRunResponse {
    /// The concatenated text of all messages (no separator), matching Python.
    pub fn text(&self) -> String {
        self.messages
            .iter()
            .map(ChatMessage::text)
            .collect::<String>()
    }

    /// All pending user-input (function-approval) requests across the messages.
    ///
    /// Mirrors Python `AgentRunResponse.user_input_requests`; use this to detect
    /// when a run paused awaiting human approval of a tool call.
    pub fn user_input_requests(&self) -> Vec<&FunctionApprovalRequestContent> {
        self.messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_approval_request)
            .collect()
    }

    /// Parse the run's concatenated text into a structured value.
    ///
    /// Mirrors Python's `response.value`: the message text is treated as JSON
    /// and deserialized into `T`.
    pub fn parse_json<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.text())
            .map_err(|e| Error::Serialization(format!("failed to parse structured output: {e}")))
    }

    /// Build from a chat response, mapping fields across.
    pub fn from_chat_response(resp: ChatResponse) -> Self {
        Self {
            messages: resp.messages,
            response_id: resp.response_id,
            conversation_id: resp.conversation_id,
            created_at: resp.created_at,
            usage_details: resp.usage_details,
            value: resp.value,
            additional_properties: resp.additional_properties,
        }
    }

    /// Aggregate a stream of agent updates into a full response.
    pub fn from_updates(updates: Vec<AgentRunResponseUpdate>) -> Self {
        Self::from_updates_with_format(updates, None)
    }

    /// Aggregate a stream of agent updates into a full response, auto-populating
    /// [`AgentRunResponse::value`] from the aggregated text when a structured
    /// `response_format` was requested (mirrors Python's
    /// `output_format_type` argument to `from_agent_run_response_updates`).
    pub fn from_updates_with_format(
        updates: Vec<AgentRunResponseUpdate>,
        response_format: Option<&ResponseFormat>,
    ) -> Self {
        let chat_updates: Vec<ChatResponseUpdate> = updates
            .into_iter()
            .map(AgentRunResponseUpdate::into_chat_update)
            .collect();
        Self::from_chat_response(ChatResponse::from_updates_with_format(
            chat_updates,
            response_format,
        ))
    }

    /// Alias for [`AgentRunResponse::from_updates`], matching Python's
    /// `AgentRunResponse.from_agent_run_response_updates`.
    pub fn from_agent_run_response_updates(updates: Vec<AgentRunResponseUpdate>) -> Self {
        Self::from_updates(updates)
    }
}

/// A single streaming chunk from an agent run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentRunResponseUpdate {
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub role: Option<Role>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub author_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub created_at: Option<String>,
    /// Provider-specific metadata for this update. Merged onto
    /// [`AgentRunResponse::additional_properties`] during aggregation.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, Value>,
    /// The raw provider payload this update was decoded from, if retained.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw_representation: Option<Value>,
}

impl AgentRunResponseUpdate {
    /// The concatenated text of this update.
    pub fn text(&self) -> String {
        self.contents
            .iter()
            .filter_map(Content::as_text)
            .collect::<String>()
    }

    /// The user-input (function-approval) requests carried by this update.
    ///
    /// Mirrors Python `AgentRunResponseUpdate.user_input_requests`.
    pub fn user_input_requests(&self) -> Vec<&FunctionApprovalRequestContent> {
        self.contents
            .iter()
            .filter_map(Content::as_function_approval_request)
            .collect()
    }

    /// Wrap a chat update as an agent update.
    pub fn from_chat_update(u: &ChatResponseUpdate) -> Self {
        Self {
            contents: u.contents.clone(),
            role: u.role.clone(),
            author_name: u.author_name.clone(),
            response_id: u.response_id.clone(),
            message_id: u.message_id.clone(),
            created_at: u.created_at.clone(),
            additional_properties: u.additional_properties.clone(),
            raw_representation: u.raw_representation.clone(),
        }
    }

    fn into_chat_update(self) -> ChatResponseUpdate {
        ChatResponseUpdate {
            contents: self.contents,
            role: self.role,
            author_name: self.author_name,
            response_id: self.response_id,
            message_id: self.message_id,
            created_at: self.created_at,
            additional_properties: self.additional_properties,
            raw_representation: self.raw_representation,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResponseFormat;
    use serde_json::json;

    fn text_update(text: &str) -> ChatResponseUpdate {
        ChatResponseUpdate {
            contents: vec![Content::text(text)],
            role: Some(Role::assistant()),
            ..Default::default()
        }
    }

    // region: task 3 — structured-output value auto-fill on streaming aggregation

    #[test]
    fn from_updates_with_format_fills_value_for_json_object() {
        let updates = vec![text_update("{\"a\":"), text_update(" 1}")];
        let resp =
            ChatResponse::from_updates_with_format(updates, Some(&ResponseFormat::JsonObject));
        assert_eq!(resp.value, Some(json!({"a": 1})));
    }

    #[test]
    fn from_updates_with_format_fills_value_for_json_schema() {
        let fmt = ResponseFormat::json_schema("S", json!({"type": "object"}));
        let updates = vec![text_update("{\"ok\": true}")];
        let resp = ChatResponse::from_updates_with_format(updates, Some(&fmt));
        assert_eq!(resp.value, Some(json!({"ok": true})));
    }

    #[test]
    fn from_updates_without_format_leaves_value_none() {
        let resp = ChatResponse::from_updates(vec![text_update("{\"a\": 1}")]);
        assert_eq!(resp.value, None);
    }

    #[test]
    fn try_parse_value_is_failure_tolerant() {
        // Non-JSON text with a JSON format requested must NOT error the call.
        let updates = vec![text_update("not json at all")];
        let resp =
            ChatResponse::from_updates_with_format(updates, Some(&ResponseFormat::JsonObject));
        assert_eq!(
            resp.value, None,
            "parse failure leaves value None, no panic/err"
        );
    }

    #[test]
    fn agent_run_response_from_updates_with_format_propagates_value() {
        let updates = vec![AgentRunResponseUpdate {
            contents: vec![Content::text("{\"n\": 42}")],
            role: Some(Role::assistant()),
            ..Default::default()
        }];
        let resp =
            AgentRunResponse::from_updates_with_format(updates, Some(&ResponseFormat::JsonObject));
        assert_eq!(resp.value, Some(json!({"n": 42})));
    }

    // region: task 8 — streaming update metadata

    #[test]
    fn absorb_update_merges_additional_properties_onto_response() {
        let mut u1 = text_update("hi");
        u1.additional_properties
            .insert("provider".into(), json!("openai"));
        let mut u2 = text_update(" there");
        u2.additional_properties
            .insert("region".into(), json!("us"));

        let resp = ChatResponse::from_updates(vec![u1, u2]);
        assert_eq!(
            resp.additional_properties.get("provider"),
            Some(&json!("openai"))
        );
        assert_eq!(resp.additional_properties.get("region"), Some(&json!("us")));
    }

    #[test]
    fn later_updates_update_author_name_on_in_progress_message() {
        // First update has no author; a later same-message update supplies one.
        let mut first = ChatResponseUpdate {
            contents: vec![Content::text("a")],
            role: Some(Role::assistant()),
            message_id: Some("m1".into()),
            ..Default::default()
        };
        first.author_name = None;
        let second = ChatResponseUpdate {
            contents: vec![Content::text("b")],
            role: Some(Role::assistant()),
            message_id: Some("m1".into()),
            author_name: Some("assistant-42".into()),
            ..Default::default()
        };
        let resp = ChatResponse::from_updates(vec![first, second]);
        assert_eq!(resp.messages.len(), 1);
        assert_eq!(
            resp.messages[0].author_name.as_deref(),
            Some("assistant-42")
        );
        assert_eq!(resp.messages[0].text(), "ab");
    }

    #[test]
    fn update_new_fields_roundtrip_and_are_back_compat() {
        // Round-trip with the new fields set.
        let mut u = text_update("x");
        u.additional_properties.insert("k".into(), json!(1));
        u.raw_representation = Some(json!({"raw": true}));
        let s = serde_json::to_string(&u).unwrap();
        let back: ChatResponseUpdate = serde_json::from_str(&s).unwrap();
        assert_eq!(back.additional_properties.get("k"), Some(&json!(1)));
        assert_eq!(back.raw_representation, Some(json!({"raw": true})));

        // Back-compat: an old payload without the new fields still deserializes,
        // and empty fields are skipped on serialize.
        let old: ChatResponseUpdate =
            serde_json::from_str(r#"{"contents":[],"role":"assistant"}"#).unwrap();
        assert!(old.additional_properties.is_empty());
        assert_eq!(old.raw_representation, None);
        let reserialized = serde_json::to_string(&text_update("y")).unwrap();
        assert!(!reserialized.contains("additional_properties"));
        assert!(!reserialized.contains("raw_representation"));
    }
}
