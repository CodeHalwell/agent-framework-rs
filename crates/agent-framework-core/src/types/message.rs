//! Chat messages and author roles.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use super::content::Content;

/// The role of a message author.
///
/// Like the Python `Role`, this is an *open* value wrapper: the well-known
/// constants are provided, but any string value is permitted.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Role(pub String);

impl Role {
    pub const SYSTEM: &'static str = "system";
    pub const USER: &'static str = "user";
    pub const ASSISTANT: &'static str = "assistant";
    pub const TOOL: &'static str = "tool";

    pub fn new(value: impl Into<String>) -> Self {
        Role(value.into())
    }
    pub fn system() -> Self {
        Role(Self::SYSTEM.into())
    }
    pub fn user() -> Self {
        Role(Self::USER.into())
    }
    pub fn assistant() -> Self {
        Role(Self::ASSISTANT.into())
    }
    pub fn tool() -> Self {
        Role(Self::TOOL.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for Role {
    fn from(s: &str) -> Self {
        Role(s.to_string())
    }
}
impl From<String> for Role {
    fn from(s: String) -> Self {
        Role(s)
    }
}

/// A single chat message: an author role plus an ordered list of content items.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub author_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, serde_json::Value>,
}

impl Message {
    /// Create a message with a single text content item.
    pub fn new(role: impl Into<Role>, text: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            contents: vec![Content::text(text)],
            author_name: None,
            message_id: None,
            additional_properties: HashMap::new(),
        }
    }

    /// Create a message with a single text content item, from an explicit role.
    ///
    /// Parity alias for the Python `Message(role=…, text=…)` constructor.
    pub fn from_text(role: impl Into<Role>, text: impl Into<String>) -> Self {
        Self::new(role, text)
    }

    /// Create a message from a role and explicit content items.
    pub fn with_contents(role: impl Into<Role>, contents: Vec<Content>) -> Self {
        Self {
            role: role.into(),
            contents,
            author_name: None,
            message_id: None,
            additional_properties: HashMap::new(),
        }
    }

    /// Convenience: a `user` message.
    pub fn user(text: impl Into<String>) -> Self {
        Self::new(Role::user(), text)
    }
    /// Convenience: a `system` message.
    pub fn system(text: impl Into<String>) -> Self {
        Self::new(Role::system(), text)
    }
    /// Convenience: an `assistant` message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new(Role::assistant(), text)
    }

    /// Set the author name (builder style).
    pub fn with_author(mut self, name: impl Into<String>) -> Self {
        self.author_name = Some(name.into());
        self
    }

    /// The concatenated text of all text content items (space-joined).
    ///
    /// Returns an **empty string** when the message carries a provider
    /// refusal (see [`TextContent::refusal`](super::content::TextContent::refusal)).
    /// A refusal is the model declining to answer; handing its text back here
    /// would present it as the answer, which is how a caller ends up parsing
    /// "I can't help with that" as its requested JSON. Reach for
    /// [`Self::refusal_text`] to read it deliberately, or [`Self::has_refusal`]
    /// to branch. Mirrors upstream's `Message.text` (#7992).
    pub fn text(&self) -> String {
        if self.has_refusal() {
            return String::new();
        }
        self.contents
            .iter()
            .filter_map(Content::as_text)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Whether any content item is a provider refusal.
    pub fn has_refusal(&self) -> bool {
        self.contents
            .iter()
            .any(|c| matches!(c, Content::Text(t) if t.refusal))
    }

    /// The refusal text, when the model declined — `None` otherwise.
    ///
    /// Several refusal items (which streaming can produce) are space-joined,
    /// matching [`Self::text`]'s convention for ordinary text.
    pub fn refusal_text(&self) -> Option<String> {
        let parts: Vec<&str> = self
            .contents
            .iter()
            .filter_map(|c| match c {
                Content::Text(t) if t.refusal => Some(t.text.as_str()),
                _ => None,
            })
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" "))
        }
    }

    /// The function-call content items in this message.
    pub fn function_calls(&self) -> Vec<&super::content::FunctionCallContent> {
        self.contents
            .iter()
            .filter_map(Content::as_function_call)
            .collect()
    }

    /// The function-result content items in this message.
    pub fn function_results(&self) -> Vec<&super::content::FunctionResultContent> {
        self.contents
            .iter()
            .filter_map(Content::as_function_result)
            .collect()
    }

    /// The function-approval request content items in this message.
    pub fn user_input_requests(&self) -> Vec<&super::content::FunctionApprovalRequestContent> {
        self.contents
            .iter()
            .filter_map(Content::as_function_approval_request)
            .collect()
    }
}

/// Normalize loosely-typed input into a list of chat messages, optionally
/// prepending a system instruction. Mirrors `prepare_messages`.
pub fn prepare_messages(messages: Vec<Message>, system_instructions: Option<&str>) -> Vec<Message> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(instr) = system_instructions {
        if !instr.is_empty() {
            out.push(Message::system(instr));
        }
    }
    out.extend(messages);
    out
}

/// Trait for values that can be turned into a list of chat messages, so the
/// public API can accept `&str`, `String`, `Message`, or vectors thereof —
/// mirroring the Python `str | Message | list[...]` unions.
pub trait IntoMessages {
    fn into_messages(self) -> Vec<Message>;
}

impl IntoMessages for Vec<Message> {
    fn into_messages(self) -> Vec<Message> {
        self
    }
}
impl IntoMessages for Message {
    fn into_messages(self) -> Vec<Message> {
        vec![self]
    }
}
impl IntoMessages for &str {
    fn into_messages(self) -> Vec<Message> {
        vec![Message::user(self)]
    }
}
impl IntoMessages for String {
    fn into_messages(self) -> Vec<Message> {
        vec![Message::user(self)]
    }
}
impl IntoMessages for Vec<String> {
    fn into_messages(self) -> Vec<Message> {
        self.into_iter().map(Message::user).collect()
    }
}
impl IntoMessages for Vec<&str> {
    fn into_messages(self) -> Vec<Message> {
        self.into_iter().map(Message::user).collect()
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::*;
    use crate::types::content::TextContent;

    fn refusal_message() -> Message {
        Message {
            contents: vec![
                Content::Text(TextContent::new("here is half an answer")),
                Content::Text(TextContent::refusal("I can't do the rest")),
            ],
            ..Message::new(Role::assistant(), "")
        }
    }

    #[test]
    fn text_is_withheld_when_a_refusal_is_present() {
        // The whole point: a caller reading `.text()` must not receive
        // something that reads like the answer when the model declined.
        assert_eq!(refusal_message().text(), "");
    }

    #[test]
    fn refusal_text_reads_the_decline_deliberately() {
        assert_eq!(
            refusal_message().refusal_text().as_deref(),
            Some("I can't do the rest")
        );
        assert!(refusal_message().has_refusal());
    }

    #[test]
    fn several_refusal_items_join_like_ordinary_text() {
        // Streaming can split one refusal across chunks.
        let m = Message {
            contents: vec![
                Content::Text(TextContent::refusal("I can't")),
                Content::Text(TextContent::refusal("help with that")),
            ],
            ..Message::new(Role::assistant(), "")
        };
        assert_eq!(m.refusal_text().as_deref(), Some("I can't help with that"));
    }

    #[test]
    fn an_ordinary_message_is_unaffected() {
        let m = Message::assistant("the answer");
        assert!(!m.has_refusal());
        assert_eq!(m.refusal_text(), None);
        assert_eq!(m.text(), "the answer");
    }

    #[test]
    fn the_refusal_flag_round_trips_through_serde() {
        let m = refusal_message();
        let restored: Message = serde_json::from_value(serde_json::to_value(&m).unwrap()).unwrap();
        assert!(restored.has_refusal());
        assert_eq!(restored.text(), "");

        // And an ordinary message serializes without the field at all, so the
        // wire shape is unchanged for everything that is not a refusal.
        let plain = serde_json::to_value(Message::assistant("hi")).unwrap();
        assert!(plain["contents"][0].get("refusal").is_none());
    }
}
