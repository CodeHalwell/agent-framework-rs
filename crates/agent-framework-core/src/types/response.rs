//! Response and streaming-update types for chat clients and agents.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::content::{Content, FunctionApprovalRequestContent, FunctionCallContent, UsageDetails};
use super::message::{ChatMessage, Role};
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
        let mut resp = ChatResponse::default();
        for u in updates {
            resp.absorb_update(u);
        }
        resp.finalize();
        resp
    }

    /// Alias for [`ChatResponse::from_updates`], matching Python's
    /// `ChatResponse.from_chat_response_updates`.
    pub fn from_chat_response_updates(updates: Vec<ChatResponseUpdate>) -> Self {
        Self::from_updates(updates)
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
            created_at: resp.created_at,
            usage_details: resp.usage_details,
            value: resp.value,
            additional_properties: resp.additional_properties,
        }
    }

    /// Aggregate a stream of agent updates into a full response.
    pub fn from_updates(updates: Vec<AgentRunResponseUpdate>) -> Self {
        let chat_updates: Vec<ChatResponseUpdate> = updates
            .into_iter()
            .map(AgentRunResponseUpdate::into_chat_update)
            .collect();
        Self::from_chat_response(ChatResponse::from_updates(chat_updates))
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
            ..Default::default()
        }
    }
}
