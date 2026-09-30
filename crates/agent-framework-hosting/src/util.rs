//! Small shared helpers: timestamps and id generation.

use agent_framework_core::types::{Content, FunctionArguments, FunctionCallContent, Message};

use std::time::{SystemTime, UNIX_EPOCH};

use uuid::Uuid;

/// Unix time in fractional seconds (OpenAI `created_at` convention).
pub(crate) fn now_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// A short hex fragment for synthesized ids.
pub(crate) fn short_hex() -> String {
    Uuid::new_v4().simple().to_string()[..8].to_string()
}

/// A `resp_…` id (OpenAI response id convention).
pub(crate) fn resp_id() -> String {
    format!("resp_{}", short_hex())
}

/// A `msg_…` id (OpenAI message-item id convention).
pub(crate) fn msg_id() -> String {
    format!("msg_{}", short_hex())
}

/// The function calls declared in `contents`, in order.
///
/// A declaration-only call is one the *caller* is expected to execute: core
/// deliberately leaves [`FunctionCallContent`] intact rather than resolving
/// it, so a host that serializes only text drops the id, name and arguments
/// the client needs and leaves it nothing to act on.
pub(crate) fn function_calls(contents: &[Content]) -> Vec<&FunctionCallContent> {
    contents
        .iter()
        .filter_map(|c| match c {
            Content::FunctionCall(call) => Some(call),
            _ => None,
        })
        .collect()
}

/// The same, across every message of a response.
pub(crate) fn function_calls_of(messages: &[Message]) -> Vec<&FunctionCallContent> {
    messages
        .iter()
        .flat_map(|m| function_calls(&m.contents))
        .collect()
}

/// A call's arguments as the **JSON string** both OpenAI surfaces expect.
///
/// `FunctionArguments` is either a raw string — which is what a provider
/// streams, possibly a fragment — or a parsed object, which has to be
/// re-serialized. Absent arguments become `{}` rather than `null`, since the
/// wire field is typed as a string and clients `JSON.parse` it.
pub(crate) fn arguments_string(call: &FunctionCallContent) -> String {
    match &call.arguments {
        Some(FunctionArguments::Raw(raw)) => raw.clone(),
        Some(FunctionArguments::Object(map)) => {
            serde_json::to_string(map).unwrap_or_else(|_| "{}".to_string())
        }
        None => "{}".to_string(),
    }
}
