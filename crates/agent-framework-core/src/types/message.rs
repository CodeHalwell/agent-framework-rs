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
pub struct ChatMessage {
    pub role: Role,
    pub contents: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub author_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub message_id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_properties: HashMap<String, serde_json::Value>,
}

impl ChatMessage {
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
    pub fn text(&self) -> String {
        self.contents
            .iter()
            .filter_map(Content::as_text)
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Normalize loosely-typed input into a list of chat messages, optionally
/// prepending a system instruction. Mirrors `prepare_messages`.
pub fn prepare_messages(
    messages: Vec<ChatMessage>,
    system_instructions: Option<&str>,
) -> Vec<ChatMessage> {
    let mut out = Vec::with_capacity(messages.len() + 1);
    if let Some(instr) = system_instructions {
        if !instr.is_empty() {
            out.push(ChatMessage::system(instr));
        }
    }
    out.extend(messages);
    out
}

/// Trait for values that can be turned into a list of chat messages, so the
/// public API can accept `&str`, `String`, `ChatMessage`, or vectors thereof —
/// mirroring the Python `str | ChatMessage | list[...]` unions.
pub trait IntoMessages {
    fn into_messages(self) -> Vec<ChatMessage>;
}

impl IntoMessages for Vec<ChatMessage> {
    fn into_messages(self) -> Vec<ChatMessage> {
        self
    }
}
impl IntoMessages for ChatMessage {
    fn into_messages(self) -> Vec<ChatMessage> {
        vec![self]
    }
}
impl IntoMessages for &str {
    fn into_messages(self) -> Vec<ChatMessage> {
        vec![ChatMessage::user(self)]
    }
}
impl IntoMessages for String {
    fn into_messages(self) -> Vec<ChatMessage> {
        vec![ChatMessage::user(self)]
    }
}
impl IntoMessages for Vec<String> {
    fn into_messages(self) -> Vec<ChatMessage> {
        self.into_iter().map(ChatMessage::user).collect()
    }
}
impl IntoMessages for Vec<&str> {
    fn into_messages(self) -> Vec<ChatMessage> {
        self.into_iter().map(ChatMessage::user).collect()
    }
}
