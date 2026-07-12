//! Direct-to-Engine wire activities: the (Bot Framework) `Activity` shape
//! Copilot Studio speaks over Direct-to-Engine, reduced to just the fields
//! this port reads or writes.
//!
//! # Fidelity
//!
//! The outgoing shape (`{"activity": {"type": "message", "text": ...,
//! "conversation": {"id": ...}}}`) and field names (`type`, `id`, `from`,
//! `conversation.id`, `text`) are a faithful port of
//! `microsoft_agents.copilotstudio.client.execute_turn_request.ExecuteTurnRequest`
//! and the `microsoft_agents.activity.Activity` / `ConversationAccount` /
//! `ChannelAccount` pydantic models (camelCase-aliased, with `from_property`
//! aliased to the wire key `"from"`) — verified against that package's actual
//! source (see the crate docs for provenance), not guessed.
//!
//! The **framing** of the response body (SSE `event: activity` / `data:
//! {...}` pairs from `CopilotClient.post_request`, vs. a bare JSON array) is
//! handled defensively: see [`parse_activities`].

use serde::Deserialize;
use serde_json::Value;

/// One Direct-to-Engine activity, as received in a response. Only the fields
/// this port consumes are modeled; anything else in the payload is ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct WireActivity {
    #[serde(rename = "type")]
    pub activity_type: String,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(rename = "from", default)]
    pub from: Option<WireChannelAccount>,
    #[serde(default)]
    pub conversation: Option<WireConversationAccount>,
}

/// `ChannelAccount`, reduced to the `name` this port surfaces as
/// [`Message::author_name`](agent_framework_core::types::Message::author_name).
#[derive(Debug, Clone, Deserialize)]
pub struct WireChannelAccount {
    #[serde(default)]
    pub name: Option<String>,
}

/// `ConversationAccount`, reduced to the `id` this port needs for
/// conversation continuity.
#[derive(Debug, Clone, Deserialize)]
pub struct WireConversationAccount {
    pub id: String,
}

/// Build the `{"activity": {"type": "message", ...}}` execute-turn request
/// body. Mirrors `ExecuteTurnRequest(activity=Activity(type="message",
/// text=question, conversation=ConversationAccount(id=conversation_id)))`
/// (`CopilotClient.ask_question`), with unset optional activity fields
/// (`id`, `from`) simply absent — matching Python's
/// `model_dump(..., exclude_unset=True)`.
pub fn build_message_activity_body(text: &str, conversation_id: &str) -> Value {
    serde_json::json!({
        "activity": {
            "type": "message",
            "text": text,
            "conversation": { "id": conversation_id }
        }
    })
}

/// Build the `start_conversation` request body. Mirrors
/// `CopilotClient.start_conversation`'s `{"emitStartConversationEvent":
/// emit_start_conversation_event}`.
pub fn build_start_conversation_body(emit_start_conversation_event: bool) -> Value {
    serde_json::json!({ "emitStartConversationEvent": emit_start_conversation_event })
}

/// Parse a Direct-to-Engine response body into its constituent
/// [`WireActivity`]s.
///
/// Handles two shapes, per this work package's spec:
///
/// - **Server-Sent Events**: `event: activity` / `data: {...}` frames
///   separated by a blank line — the real wire format emitted by
///   `CopilotClient.post_request`. Any blank-line-delimited block containing
///   at least one `data:` line is treated as one activity (multiple `data:`
///   lines within a block are joined with `\n`, per the SSE spec); this is
///   intentionally more lenient than the reference client, which also
///   requires a preceding `event: activity` line — real Direct-to-Engine
///   responses only ever emit `activity` events on this endpoint, so the
///   extra gate has no practical effect and dropping it keeps this parser
///   simpler.
/// - **A bare JSON array** of activity objects — a defensive fallback for
///   transports/tests that hand back a whole batch as one JSON document
///   rather than an SSE stream (chosen by sniffing whether the
///   (whitespace-trimmed) body starts with `[`).
///
/// A block/entry that fails to parse as a [`WireActivity`] is skipped rather
/// than failing the whole parse, mirroring the reference client's per-line
/// handling (which simply never yields for a line it can't associate with an
/// `activity` event).
pub fn parse_activities(body: &str) -> Vec<WireActivity> {
    let trimmed = body.trim_start();
    if trimmed.starts_with('[') {
        return serde_json::from_str::<Vec<WireActivity>>(trimmed).unwrap_or_default();
    }
    parse_sse_activities(body)
}

fn parse_sse_activities(body: &str) -> Vec<WireActivity> {
    let normalized = body.replace("\r\n", "\n");
    let mut out = Vec::new();
    for block in normalized.split("\n\n") {
        let mut data_lines = Vec::new();
        for line in block.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                data_lines.push(rest.strip_prefix(' ').unwrap_or(rest));
            }
        }
        if data_lines.is_empty() {
            continue;
        }
        let payload = data_lines.join("\n");
        if let Ok(activity) = serde_json::from_str::<WireActivity>(&payload) {
            out.push(activity);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- outgoing body builders -------------------------------------------

    #[test]
    fn build_message_activity_body_shape() {
        let body = build_message_activity_body("hello there", "conv-1");
        assert_eq!(
            body,
            serde_json::json!({
                "activity": {
                    "type": "message",
                    "text": "hello there",
                    "conversation": { "id": "conv-1" }
                }
            })
        );
    }

    #[test]
    fn build_start_conversation_body_shape() {
        let body = build_start_conversation_body(true);
        assert_eq!(
            body,
            serde_json::json!({ "emitStartConversationEvent": true })
        );
    }

    // -- parse_activities: SSE ---------------------------------------------

    #[test]
    fn parse_activities_single_message_sse_frame() {
        let body = "event: activity\ndata: {\"type\":\"message\",\"id\":\"a1\",\"text\":\"hi\",\"from\":{\"name\":\"Bot\"},\"conversation\":{\"id\":\"conv-1\"}}\n\n";
        let activities = parse_activities(body);
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].activity_type, "message");
        assert_eq!(activities[0].text.as_deref(), Some("hi"));
        assert_eq!(activities[0].id.as_deref(), Some("a1"));
        assert_eq!(
            activities[0].from.as_ref().and_then(|f| f.name.as_deref()),
            Some("Bot")
        );
        assert_eq!(
            activities[0].conversation.as_ref().map(|c| c.id.as_str()),
            Some("conv-1")
        );
    }

    #[test]
    fn parse_activities_mixed_typing_and_message_frames() {
        let body = "event: activity\ndata: {\"type\":\"typing\",\"id\":\"t1\"}\n\n\
                    event: activity\ndata: {\"type\":\"message\",\"id\":\"m1\",\"text\":\"final answer\"}\n\n";
        let activities = parse_activities(body);
        assert_eq!(activities.len(), 2);
        assert_eq!(activities[0].activity_type, "typing");
        assert_eq!(activities[1].activity_type, "message");
        assert_eq!(activities[1].text.as_deref(), Some("final answer"));
    }

    #[test]
    fn parse_activities_multiple_data_lines_in_one_frame_are_joined() {
        // Not something the real server does for JSON payloads, but the SSE
        // spec allows a block to carry several `data:` lines that are joined
        // with `\n`, and the parser must not silently drop the second line.
        let body = "data: {\"type\":\"message\",\ndata: \"id\":\"m1\",\"text\":\"joined\"}\n\n";
        let activities = parse_activities(body);
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].id.as_deref(), Some("m1"));
        assert_eq!(activities[0].text.as_deref(), Some("joined"));
    }

    #[test]
    fn parse_activities_skips_unparseable_frames() {
        let body = "data: not json at all\n\ndata: {\"type\":\"message\",\"text\":\"ok\"}\n\n";
        let activities = parse_activities(body);
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].text.as_deref(), Some("ok"));
    }

    #[test]
    fn parse_activities_empty_body_yields_no_activities() {
        assert!(parse_activities("").is_empty());
    }

    // -- parse_activities: JSON array fallback -----------------------------

    #[test]
    fn parse_activities_json_array_fallback() {
        let body = serde_json::json!([
            {"type": "message", "id": "m1", "text": "one"},
            {"type": "message", "id": "m2", "text": "two"},
        ])
        .to_string();
        let activities = parse_activities(&body);
        assert_eq!(activities.len(), 2);
        assert_eq!(activities[0].text.as_deref(), Some("one"));
        assert_eq!(activities[1].text.as_deref(), Some("two"));
    }

    #[test]
    fn parse_activities_json_array_with_leading_whitespace() {
        let body = "  \n  [{\"type\":\"message\",\"text\":\"padded\"}]";
        let activities = parse_activities(body);
        assert_eq!(activities.len(), 1);
        assert_eq!(activities[0].text.as_deref(), Some("padded"));
    }

    #[test]
    fn parse_activities_malformed_json_array_yields_empty() {
        assert!(parse_activities("[not valid json").is_empty());
    }
}
