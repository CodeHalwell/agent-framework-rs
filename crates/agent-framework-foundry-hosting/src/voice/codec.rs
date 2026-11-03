//! The context-free codec for the Voice Live Bridge text/control profile
//! (upstream `voice/_codec.py`).
//!
//! [`decode_inbound_message`] validates one Bridge-to-agent text frame and
//! returns its typed message, `None` for an unknown future type (forward
//! compatibility), or a [`VoiceProtocolError`] carrying upstream's message
//! and close code. [`encode_outbound_message`] validates and encodes one
//! agent-to-Bridge message as compact JSON. Neither keeps any state between
//! frames.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};

use super::models::*;
use super::VoiceError;

/// The largest frame, in UTF-8 bytes, either direction (1 MiB).
pub const MAX_FRAME_BYTES: usize = 1_048_576;
/// The largest identifier, in UTF-8 bytes.
pub const MAX_IDENTIFIER_BYTES: usize = 256;
/// The most digits a JSON integer may have.
pub const MAX_INTEGER_DIGITS: usize = 128;
/// The deepest JSON nesting a frame may have.
pub const MAX_JSON_DEPTH: usize = 32;
/// The most JSON values a frame may have.
pub const MAX_JSON_NODES: usize = 8_192;
/// The longest proactive admission timeout.
pub const MAX_ADMISSION_TIMEOUT_MS: u64 = 60_000;

/// The protocol-error close code.
pub const CLOSE_PROTOCOL_ERROR: u16 = 1002;
/// The unsupported-data close code.
pub const CLOSE_UNSUPPORTED_DATA: u16 = 1003;
/// The message-too-big close code.
pub const CLOSE_MESSAGE_TOO_BIG: u16 = 1009;

const KNOWN_UNSUPPORTED_INBOUND: &[&str] = &[
    "conversation.item.create",
    "conversation.item.delete",
    "dtmf",
    "dtmf.collect.cancelled",
    "dtmf.collect.rejected",
    "handoff.failed",
];
const OUTBOUND_TYPES: &[&str] = &[
    "end_call",
    "error",
    "response.cancel",
    "response.created",
    "response.done",
    "response.none",
    "response.output_data.done",
    "response.output_text.delta",
    "response.output_text.done",
    "session.ready",
    "session.rejected",
];

/// One frame violated the profile (upstream `VoiceProtocolError`, and its
/// subclass `VoiceUnsupportedMessageError` when [`is_unsupported`](Self::is_unsupported)).
/// The message is content-free.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceProtocolError {
    message: String,
    close_code: u16,
    unsupported: bool,
}

impl VoiceProtocolError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            close_code: CLOSE_PROTOCOL_ERROR,
            unsupported: false,
        }
    }

    fn with_code(message: impl Into<String>, close_code: u16) -> Self {
        Self {
            close_code,
            ..Self::new(message)
        }
    }

    fn unsupported(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            close_code: CLOSE_UNSUPPORTED_DATA,
            unsupported: true,
        }
    }

    /// The diagnostic.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The RFC 6455 close code for the violation (1002, 1003 or 1009).
    pub fn close_code(&self) -> u16 {
        self.close_code
    }

    /// Whether the frame is a known Protocol 1.0 family this profile
    /// excludes (upstream `VoiceUnsupportedMessageError`, close code 1003).
    pub fn is_unsupported(&self) -> bool {
        self.unsupported
    }
}

impl fmt::Display for VoiceProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VoiceProtocolError {}

impl From<String> for VoiceProtocolError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

type PResult<T> = std::result::Result<T, VoiceProtocolError>;
type Checked<T> = std::result::Result<T, String>;

// ---------------------------------------------------------------------------
// Decoding.

/// Decode one Bridge-to-agent text frame. `Ok(None)` is an unknown future
/// message type (or a `user.message` with no supported part), which the
/// host ignores.
pub fn decode_inbound_message(frame: &str) -> PResult<Option<InboundVoiceMessage>> {
    let payload = decode_json_object(frame)?;
    let message_type = required_string(&payload, "type", true)?;
    let id = identifier(&payload, "id")?;
    let ts = match payload.get("ts") {
        Some(Value::String(ts)) => {
            check_timestamp(ts)?;
            ts.clone()
        }
        _ => return Err(VoiceProtocolError::new("ts must be a string")),
    };

    if KNOWN_UNSUPPORTED_INBOUND.contains(&message_type.as_str()) {
        return Err(VoiceProtocolError::unsupported(
            "Voice message type is not supported by this profile",
        ));
    }
    if OUTBOUND_TYPES.contains(&message_type.as_str()) {
        return Err(VoiceProtocolError::new(
            "Agent-to-Bridge message received in the inbound direction",
        ));
    }

    let p = &payload;
    Ok(Some(match message_type.as_str() {
        "session.start" => InboundVoiceMessage::SessionStart(parse_session_start(p, id, ts)?),
        "user.message" => match parse_user_message(p, id, ts)? {
            Some(message) => InboundVoiceMessage::UserMessage(message),
            None => return Ok(None),
        },
        "user.no_input" => InboundVoiceMessage::UserNoInput(UserNoInput {
            id,
            ts,
            item_id: prefixed_identifier(p, "item_id", "in_")?,
            count: positive_integer(p, "count")?,
        }),
        "user.speech_started" => {
            InboundVoiceMessage::UserSpeechStarted(UserSpeechStarted { id, ts })
        }
        "user.barge_in" => InboundVoiceMessage::UserBargeIn(UserBargeIn {
            id,
            ts,
            response_id: prefixed_identifier(p, "response_id", "r_")?,
            item_id: optional_prefixed_identifier(p, "item_id", "it_")?,
            heard_text: required_string(p, "heard_text", false)?,
        }),
        "response.accepted" => InboundVoiceMessage::ResponseAccepted(ResponseAccepted {
            id,
            ts,
            response_id: prefixed_identifier(p, "response_id", "r_")?,
        }),
        "response.dropped" => InboundVoiceMessage::ResponseDropped(ResponseDropped {
            id,
            ts,
            response_id: prefixed_identifier(p, "response_id", "r_")?,
            reason: required_string(p, "reason", true)?,
        }),
        "response.cancelled" => InboundVoiceMessage::ResponseCancelled(ResponseCancelled {
            id,
            ts,
            response_id: prefixed_identifier(p, "response_id", "r_")?,
            item_id: optional_prefixed_identifier(p, "item_id", "it_")?,
            heard_text: required_string(p, "heard_text", false)?,
        }),
        "response.timeout" => {
            InboundVoiceMessage::ResponseTimeout(parse_response_timeout(p, id, ts)?)
        }
        "session.end" => InboundVoiceMessage::SessionEnd(SessionEnd {
            id,
            ts,
            reason: required_string(p, "reason", true)?,
        }),
        _ => return Ok(None),
    }))
}

fn parse_session_start(p: &JsonObject, id: String, ts: String) -> PResult<SessionStart> {
    let timeouts = required_mapping(p, "response_timeouts")?;
    let Some(Value::Bool(reconnect)) = p.get("reconnect") else {
        return Err(VoiceProtocolError::new(
            "session.start reconnect must be a boolean",
        ));
    };
    let reconnect = *reconnect;
    if reconnect && p.contains_key("greeting") {
        return Err(VoiceProtocolError::new(
            "session.start greeting must be absent on reconnect",
        ));
    }
    let greeting = optional_string(p, "greeting")?;
    let no_input_timeout_ms = if p.contains_key("no_input_timeout_ms") {
        Some(positive_integer(p, "no_input_timeout_ms")?)
    } else {
        None
    };
    let caller = if p.contains_key("caller") {
        let caller = required_mapping(p, "caller")?;
        reject_caller_credentials(caller)?;
        Some(caller.clone())
    } else {
        None
    };
    Ok(SessionStart {
        id,
        ts,
        protocol_version: required_string(p, "protocol_version", true)?,
        reconnect,
        response_timeouts: ResponseTimeouts {
            first_output_ms: positive_integer(timeouts, "first_output_ms")?,
            idle_ms: positive_integer(timeouts, "idle_ms")?,
            max_duration_ms: positive_integer(timeouts, "max_duration_ms")?,
        },
        greeting,
        no_input_timeout_ms,
        caller,
    })
}

fn parse_user_message(p: &JsonObject, id: String, ts: String) -> PResult<Option<UserMessage>> {
    let raw_content = match p.get("content") {
        Some(Value::Array(parts)) if !parts.is_empty() => parts,
        _ => {
            return Err(VoiceProtocolError::new(
                "user.message content must be a non-empty array",
            ))
        }
    };
    let mut content = Vec::new();
    for raw_part in raw_content {
        let Value::Object(part) = raw_part else {
            return Err(VoiceProtocolError::new(
                "user.message content parts must be objects",
            ));
        };
        match part.get("type") {
            Some(Value::String(t)) if t == "input_text" => {
                content.push(InputContentPart::InputText {
                    text: required_string(part, "text", false)?,
                })
            }
            Some(Value::String(t)) if t == "input_data" => {
                content.push(InputContentPart::InputData {
                    data: required_mapping(part, "data")?.clone(),
                })
            }
            Some(Value::String(t)) if t == "input_image" => {
                return Err(VoiceProtocolError::unsupported(
                    "Image content is not supported by this profile",
                ))
            }
            Some(Value::String(_)) => {}
            _ => {
                return Err(VoiceProtocolError::new(
                    "user.message content part type must be a string",
                ))
            }
        }
    }
    if content.is_empty() {
        return Ok(None);
    }
    Ok(Some(UserMessage {
        id,
        ts,
        item_id: prefixed_identifier(p, "item_id", "in_")?,
        content,
    }))
}

fn parse_response_timeout(p: &JsonObject, id: String, ts: String) -> PResult<ResponseTimeout> {
    let has_response_id = p.contains_key("response_id");
    if has_response_id == p.contains_key("item_ids") {
        return Err(VoiceProtocolError::new(
            "response.timeout requires exactly one target",
        ));
    }
    let (response_id, item_ids) = if has_response_id {
        (Some(prefixed_identifier(p, "response_id", "r_")?), None)
    } else {
        let raw = match p.get("item_ids") {
            Some(Value::Array(items)) if !items.is_empty() => items,
            _ => {
                return Err(VoiceProtocolError::new(
                    "response.timeout item_ids must be a non-empty array",
                ))
            }
        };
        let mut item_ids = Vec::with_capacity(raw.len());
        for item in raw {
            let Value::String(item) = item else {
                return Err(VoiceProtocolError::new("item_ids must be a string"));
            };
            check_prefixed(item, "item_ids", "in_")?;
            item_ids.push(item.clone());
        }
        if item_ids.iter().collect::<BTreeSet<_>>().len() != item_ids.len() {
            return Err(VoiceProtocolError::new(
                "response.timeout item_ids must be unique",
            ));
        }
        (None, Some(item_ids))
    };
    Ok(ResponseTimeout {
        id,
        ts,
        stage: required_string(p, "stage", true)?,
        response_id,
        item_ids,
    })
}

/// A JSON value that rejects duplicate object keys while parsing (upstream
/// `object_pairs_hook=_object_without_duplicate_keys`).
struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(StrictVisitor).map(StrictValue)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E>(self, v: bool) -> Result<Value, E> {
        Ok(Value::Bool(v))
    }

    fn visit_i64<E>(self, v: i64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_u64<E>(self, v: u64) -> Result<Value, E> {
        Ok(Value::Number(v.into()))
    }

    fn visit_f64<E: de::Error>(self, v: f64) -> Result<Value, E> {
        Number::from_f64(v)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite number"))
    }

    fn visit_str<E>(self, v: &str) -> Result<Value, E> {
        Ok(Value::String(v.to_string()))
    }

    fn visit_string<E>(self, v: String) -> Result<Value, E> {
        Ok(Value::String(v))
    }

    fn visit_unit<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(StrictValue(item)) = seq.next_element()? {
            items.push(item);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut object = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if object.contains_key(&key) {
                return Err(de::Error::custom("duplicate JSON object key"));
            }
            let StrictValue(value) = map.next_value()?;
            object.insert(key, value);
        }
        Ok(Value::Object(object))
    }
}

/// Whether the text has an integer literal of more than
/// [`MAX_INTEGER_DIGITS`] digits outside strings (upstream `parse_int`).
fn has_oversized_integer(frame: &str) -> bool {
    let bytes = frame.as_bytes();
    let (mut i, mut in_string) = (0, false);
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            match b {
                b'\\' => i += 1,
                b'"' => in_string = false,
                _ => {}
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            i += 1;
            continue;
        }
        if b == b'-' || b.is_ascii_digit() {
            let start = i;
            while i < bytes.len()
                && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
            {
                i += 1;
            }
            let token = &frame[start..i];
            let digits = token.strip_prefix('-').unwrap_or(token);
            if digits.bytes().all(|d| d.is_ascii_digit()) && digits.len() > MAX_INTEGER_DIGITS {
                return true;
            }
            continue;
        }
        i += 1;
    }
    false
}

fn decode_json_object(frame: &str) -> PResult<JsonObject> {
    if frame.len() > MAX_FRAME_BYTES {
        return Err(VoiceProtocolError::with_code(
            "Voice frame exceeds the maximum size",
            CLOSE_MESSAGE_TOO_BIG,
        ));
    }
    if has_oversized_integer(frame) {
        return Err(VoiceProtocolError::new("Voice frame is not valid JSON"));
    }
    let value = match serde_json::from_str::<StrictValue>(frame) {
        Ok(StrictValue(value)) => value,
        Err(e) => {
            let detail = e.to_string();
            // Python decodes lone surrogates and then fails to encode them;
            // serde_json refuses them while parsing.
            if detail.contains("surrogate")
                || detail.contains("unexpected end of hex escape")
                || detail.contains("invalid unicode code point")
            {
                return Err(VoiceProtocolError::new(
                    "Voice frame contains invalid Unicode",
                ));
            }
            // Python reads an overflowing float as infinity.
            if detail.contains("number out of range") || detail.contains("non-finite number") {
                return Err(VoiceProtocolError::new(
                    "Voice frame contains a non-finite number",
                ));
            }
            return Err(VoiceProtocolError::new("Voice frame is not valid JSON"));
        }
    };
    let Value::Object(object) = value else {
        return Err(VoiceProtocolError::new("Voice frame must be a JSON object"));
    };
    validate_json_tree(&object)?;
    Ok(object)
}

/// Node-count and depth limits (upstream `_validate_json_tree`, same
/// traversal order).
fn validate_json_tree(root: &JsonObject) -> PResult<()> {
    let mut pending: Vec<(Option<&Value>, usize)> = vec![(None, 1)];
    let mut nodes = 0;
    while let Some((current, depth)) = pending.pop() {
        nodes += 1;
        if nodes > MAX_JSON_NODES {
            return Err(VoiceProtocolError::new(
                "Voice frame exceeds the maximum JSON node count",
            ));
        }
        if depth > MAX_JSON_DEPTH {
            return Err(VoiceProtocolError::new(
                "Voice frame exceeds the maximum JSON depth",
            ));
        }
        match current {
            None => pending.extend(root.values().map(|v| (Some(v), depth + 1))),
            Some(Value::Object(map)) => pending.extend(map.values().map(|v| (Some(v), depth + 1))),
            Some(Value::Array(items)) => pending.extend(items.iter().map(|v| (Some(v), depth + 1))),
            Some(_) => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Field validation, shared by both directions.

/// The RFC 3339 profile: `YYYY-MM-DDTHH:MM:SS[.f{1,9}](Z|±HH:MM)`, ASCII
/// digits only, and a real calendar date.
fn check_timestamp(value: &str) -> Checked<()> {
    let profile = || "ts must match the RFC 3339 profile".to_string();
    let b = value.as_bytes();
    let digit = |i: usize| b.get(i).is_some_and(u8::is_ascii_digit);
    let digits = |range: std::ops::Range<usize>| range.into_iter().all(digit);
    let hour = |i: usize| match (b.get(i), b.get(i + 1)) {
        (Some(b'0' | b'1'), Some(d)) => d.is_ascii_digit(),
        (Some(b'2'), Some(b'0'..=b'3')) => true,
        _ => false,
    };
    let minute = |i: usize| matches!(b.get(i), Some(b'0'..=b'5')) && digit(i + 1);
    let date_ok = digits(0..4)
        && b.get(4) == Some(&b'-')
        && digits(5..7)
        && b.get(7) == Some(&b'-')
        && digits(8..10)
        && b.get(10) == Some(&b'T')
        && hour(11)
        && b.get(13) == Some(&b':')
        && minute(14)
        && b.get(16) == Some(&b':')
        && minute(17);
    if !date_ok {
        return Err(profile());
    }
    let mut i = 19;
    if b.get(i) == Some(&b'.') {
        let start = i + 1;
        i = start;
        while digit(i) {
            i += 1;
        }
        if !(1..=9).contains(&(i - start)) {
            return Err(profile());
        }
    }
    match b.get(i) {
        Some(b'Z') if i + 1 == b.len() => {}
        Some(b'+' | b'-')
            if i + 6 == b.len() && hour(i + 1) && b.get(i + 3) == Some(&b':') && minute(i + 4) => {}
        _ => return Err(profile()),
    }
    let number = |range: std::ops::Range<usize>| value[range].parse::<u32>().unwrap_or(0);
    let (year, month, day) = (number(0..4), number(5..7), number(8..10));
    let leap = (year % 4 == 0 && year % 100 != 0) || year % 400 == 0;
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        2 => 28,
        _ => 0,
    };
    if year == 0 || day == 0 || day > days_in_month {
        return Err("ts must contain a valid calendar timestamp".into());
    }
    Ok(())
}

fn check_string(value: &str, name: &str, non_empty: bool) -> Checked<()> {
    if non_empty && value.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    Ok(())
}

fn check_identifier(value: &str, name: &str) -> Checked<()> {
    check_string(value, name, true)?;
    if value.len() > MAX_IDENTIFIER_BYTES {
        return Err(format!("{name} exceeds the maximum identifier size"));
    }
    Ok(())
}

fn check_prefixed(value: &str, name: &str, prefix: &str) -> Checked<()> {
    check_identifier(value, name)?;
    if !value.starts_with(prefix) || value.len() == prefix.len() {
        return Err(format!("{name} must start with {prefix}"));
    }
    Ok(())
}

/// Validate a prefixed identifier (shared with target-turn completion).
pub(crate) fn validate_prefixed_identifier(value: &str, name: &str, prefix: &str) -> Checked<()> {
    check_prefixed(value, name, prefix)
}

fn required_mapping<'a>(p: &'a JsonObject, name: &str) -> PResult<&'a JsonObject> {
    match p.get(name) {
        Some(Value::Object(map)) => Ok(map),
        _ => Err(VoiceProtocolError::new(format!("{name} must be an object"))),
    }
}

fn required_string(p: &JsonObject, name: &str, non_empty: bool) -> PResult<String> {
    match p.get(name) {
        None => Err(VoiceProtocolError::new(format!("{name} is required"))),
        Some(Value::String(s)) => {
            check_string(s, name, non_empty)?;
            Ok(s.clone())
        }
        Some(_) => Err(VoiceProtocolError::new(format!("{name} must be a string"))),
    }
}

fn optional_string(p: &JsonObject, name: &str) -> PResult<Option<String>> {
    if !p.contains_key(name) {
        return Ok(None);
    }
    required_string(p, name, false).map(Some)
}

fn positive_integer(p: &JsonObject, name: &str) -> PResult<u64> {
    let value = p
        .get(name)
        .ok_or_else(|| VoiceProtocolError::new(format!("{name} is required")))?;
    match value {
        Value::Number(n) if n.is_u64() => match n.as_u64() {
            Some(0) | None => Err(VoiceProtocolError::new(format!("{name} must be positive"))),
            Some(v) => Ok(v),
        },
        Value::Number(n) if n.is_i64() => {
            Err(VoiceProtocolError::new(format!("{name} must be positive")))
        }
        _ => Err(VoiceProtocolError::new(format!(
            "{name} must be an integer"
        ))),
    }
}

fn identifier(p: &JsonObject, name: &str) -> PResult<String> {
    let value = required_string(p, name, true)?;
    check_identifier(&value, name)?;
    Ok(value)
}

fn prefixed_identifier(p: &JsonObject, name: &str, prefix: &str) -> PResult<String> {
    let value = identifier(p, name)?;
    if !value.starts_with(prefix) || value.len() == prefix.len() {
        return Err(VoiceProtocolError::new(format!(
            "{name} must start with {prefix}"
        )));
    }
    Ok(value)
}

fn optional_prefixed_identifier(
    p: &JsonObject,
    name: &str,
    prefix: &str,
) -> PResult<Option<String>> {
    if !p.contains_key(name) {
        return Ok(None);
    }
    prefixed_identifier(p, name, prefix).map(Some)
}

// ---------------------------------------------------------------------------
// Caller credential screening (upstream `_reject_caller_credentials`).

static CREDENTIAL_FIELD: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"(?:^|_)(?:authorization(?:_header)?|credentials?|password|passwd|pwd|secret(?:_value)?|",
        r"api_(?:key|token)|auth_token|bearer_token|access_(?:key|token)|refresh_token|id_token|",
        r"client_(?:assertion|secret)|private_key|connection_string|sas(?:_token|_url)?|account_key|",
        r"subscription_key|shared_access_(?:key|signature))(?:_|$)"
    ))
    .expect("valid regex")
});
static ACRONYM_BOUNDARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"([A-Z]+)([A-Z][a-z])").expect("valid regex"));

const COMPACT_CREDENTIAL_FIELDS: &[&str] = &[
    "accesskey",
    "accesstoken",
    "accountkey",
    "apikey",
    "apitoken",
    "authorization",
    "authorizationheader",
    "authtoken",
    "azurepwd",
    "azuresas",
    "bearertoken",
    "clientassertion",
    "clientsecret",
    "connectionstring",
    "credential",
    "credentials",
    "idtoken",
    "password",
    "passwd",
    "privatekey",
    "pwd",
    "refreshtoken",
    "sas",
    "sastoken",
    "sasurl",
    "secret",
    "secretvalue",
    "sharedaccesskey",
    "sharedaccesssignature",
    "subscriptionkey",
];

/// `snake_case` a field name: split acronyms and camel humps, lower-case,
/// and collapse everything else to `_` (upstream `_normalize_field_name`).
fn normalize_field_name(name: &str) -> String {
    let split = ACRONYM_BOUNDARY.replace_all(name, "${1}_${2}");
    let mut humped = String::with_capacity(split.len() + 4);
    let mut previous: Option<char> = None;
    for c in split.chars() {
        if c.is_ascii_uppercase()
            && previous.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            humped.push('_');
        }
        humped.push(c);
        previous = Some(c);
    }
    let lowered = humped.to_lowercase();
    let mut out = String::with_capacity(lowered.len());
    let mut in_gap = false;
    for c in lowered.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            out.push(c);
            in_gap = false;
        } else if !in_gap {
            out.push('_');
            in_gap = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn is_credential_field(name: &str) -> bool {
    let normalized = normalize_field_name(name);
    let compact = normalized.replace('_', "");
    let suffixes = COMPACT_CREDENTIAL_FIELDS
        .iter()
        .filter(|f| !matches!(**f, "azurepwd" | "azuresas" | "pwd" | "sas"));
    CREDENTIAL_FIELD.is_match(&normalized)
        || COMPACT_CREDENTIAL_FIELDS.contains(&compact.as_str())
        || suffixes.clone().any(|s| compact.ends_with(s))
        || suffixes
            .filter(|s| **s != "secret")
            .any(|m| compact.contains(m))
}

fn reject_caller_credentials(caller: &JsonObject) -> PResult<()> {
    let check = |map: &JsonObject| -> PResult<()> {
        if map.keys().any(|key| is_credential_field(key)) {
            return Err(VoiceProtocolError::new(
                "session.start caller must not contain credentials",
            ));
        }
        Ok(())
    };
    check(caller)?;
    let mut pending: Vec<&Value> = caller.values().collect();
    while let Some(current) = pending.pop() {
        match current {
            Value::Object(map) => {
                check(map)?;
                pending.extend(map.values());
            }
            Value::Array(items) => pending.extend(items.iter()),
            _ => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Encoding.

const VOICE_TYPES: &[&str] = &[
    "openai",
    "azure-standard",
    "azure-custom",
    "azure-personal",
    "avatar-voice-sync",
    "azure-realtime-native",
];
const VOICE_STRING_FIELDS: &[&str] = &["endpoint_id", "model", "name"];
const VOICE_NON_EMPTY_STRING_FIELDS: &[&str] = &["endpoint_id", "name"];
const VOICE_NULLABLE_STRING_FIELDS: &[&str] = &[
    "custom_lexicon_url",
    "custom_text_normalization_url",
    "locale",
    "multi_talker_speaker_name",
    "pitch",
    "rate",
    "style",
    "volume",
];

fn invalid(message: impl Into<String>) -> VoiceError {
    VoiceError::Invalid(message.into())
}

/// Validate and encode one agent-to-Bridge message as one compact JSON text
/// frame. Validation is context-free: it never consults earlier messages.
pub fn encode_outbound_message(
    message: &OutboundVoiceMessage,
) -> std::result::Result<String, VoiceError> {
    use OutboundVoiceMessage as M;
    let (id, ts) = match message {
        M::SessionReady(m) => (&m.id, &m.ts),
        M::SessionRejected(m) => (&m.id, &m.ts),
        M::ResponseCreated(m) => (&m.id, &m.ts),
        M::ResponseNone(m) => (&m.id, &m.ts),
        M::ResponseOutputDataDone(m) => (&m.id, &m.ts),
        M::ResponseOutputTextDelta(m) => (&m.id, &m.ts),
        M::ResponseOutputTextDone(m) => (&m.id, &m.ts),
        M::ResponseDone(m) => (&m.id, &m.ts),
        M::ResponseCancel(m) => (&m.id, &m.ts),
        M::EndCall(m) => (&m.id, &m.ts),
        M::Error(m) => (&m.id, &m.ts),
    };
    check_identifier(id, "id").map_err(invalid)?;
    check_timestamp(ts).map_err(invalid)?;

    let mut normalized = None;
    match message {
        M::SessionReady(_) => {}
        M::SessionRejected(m) => check_string(&m.code, "code", true).map_err(invalid)?,
        M::ResponseCreated(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?;
            if let Some(in_reply_to) = &m.in_reply_to {
                check_input_prefix(in_reply_to)?;
                if m.admission_timeout_ms.is_some() || m.supersede_key.is_some() {
                    return Err(invalid(
                        "reply response.created cannot contain proactive admission controls",
                    ));
                }
            } else {
                if let Some(timeout) = m.admission_timeout_ms {
                    if timeout == 0 {
                        return Err(invalid("admission_timeout_ms must be positive"));
                    }
                    if timeout > MAX_ADMISSION_TIMEOUT_MS {
                        return Err(invalid("admission_timeout_ms must be at most 60000"));
                    }
                }
                if let Some(key) = &m.supersede_key {
                    check_string(key, "supersede_key", true).map_err(invalid)?;
                }
            }
        }
        M::ResponseNone(m) => check_input_prefix(&m.in_reply_to)?,
        M::ResponseOutputTextDelta(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?;
            check_prefixed(&m.item_id, "item_id", "it_").map_err(invalid)?;
            check_string(&m.delta, "delta", true).map_err(invalid)?;
            if let Some(voice) = &m.voice {
                let mut m = m.clone();
                m.voice = Some(normalize_voice(voice)?);
                normalized = Some(M::ResponseOutputTextDelta(m));
            }
        }
        M::ResponseOutputTextDone(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?;
            check_prefixed(&m.item_id, "item_id", "it_").map_err(invalid)?;
            if let Some(voice) = &m.voice {
                let mut m = m.clone();
                m.voice = Some(normalize_voice(voice)?);
                normalized = Some(M::ResponseOutputTextDone(m));
            }
        }
        M::ResponseOutputDataDone(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?;
            check_prefixed(&m.item_id, "item_id", "it_").map_err(invalid)?;
        }
        M::ResponseDone(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?
        }
        M::ResponseCancel(m) => {
            check_prefixed(&m.response_id, "response_id", "r_").map_err(invalid)?
        }
        M::EndCall(m) => check_string(&m.reason, "reason", true).map_err(invalid)?,
        M::Error(m) => {
            check_string(&m.code, "code", true).map_err(invalid)?;
            if let Some(response_id) = &m.response_id {
                check_prefixed(response_id, "response_id", "r_").map_err(invalid)?;
            }
            if let Some(item_id) = &m.item_id {
                if m.response_id.is_none() {
                    return Err(invalid("item_id requires response_id"));
                }
                check_prefixed(item_id, "item_id", "it_").map_err(invalid)?;
            }
        }
    }

    let frame = serde_json::to_string(normalized.as_ref().unwrap_or(message))
        .map_err(|_| invalid("Voice message contains a value that cannot be encoded"))?;
    if frame.len() > MAX_FRAME_BYTES {
        return Err(invalid(
            "Voice message exceeds the maximum encoded frame size",
        ));
    }
    Ok(frame)
}

fn check_input_prefix(values: &[String]) -> std::result::Result<(), VoiceError> {
    if values.is_empty() {
        return Err(invalid("in_reply_to must be a non-empty tuple"));
    }
    for value in values {
        check_prefixed(value, "in_reply_to", "in_").map_err(invalid)?;
    }
    if values.iter().collect::<BTreeSet<_>>().len() != values.len() {
        return Err(invalid("in_reply_to must contain unique identifiers"));
    }
    Ok(())
}

/// Validate a Voice Live voice merge patch and normalize its `type` alias
/// (upstream `_normalize_voice`); unknown fields pass through.
fn normalize_voice(voice: &JsonObject) -> std::result::Result<JsonObject, VoiceError> {
    if voice.is_empty() {
        return Err(invalid("voice must be a non-empty mapping"));
    }
    validate_json_tree(voice).map_err(|e| invalid(e.message))?;
    let mut voice = voice.clone();
    if let Some(voice_type) = voice.get("type") {
        let Value::String(voice_type) = voice_type else {
            return Err(invalid("voice.type must be a string"));
        };
        let voice_type = match voice_type.as_str() {
            "azure-platform" => "azure-standard",
            "custom" => "azure-custom",
            other => other,
        };
        if !VOICE_TYPES.contains(&voice_type) {
            return Err(invalid("voice.type is not a supported Voice Live variant"));
        }
        voice.insert("type".into(), Value::String(voice_type.to_string()));
    }
    for field in VOICE_STRING_FIELDS {
        match voice.get(*field) {
            None => {}
            Some(Value::String(s)) => check_string(
                s,
                &format!("voice.{field}"),
                VOICE_NON_EMPTY_STRING_FIELDS.contains(field),
            )
            .map_err(invalid)?,
            Some(_) => return Err(invalid(format!("voice.{field} must be a string"))),
        }
    }
    for field in VOICE_NULLABLE_STRING_FIELDS {
        if !matches!(
            voice.get(*field),
            None | Some(Value::Null | Value::String(_))
        ) {
            return Err(invalid(format!("voice.{field} must be a string or null")));
        }
    }
    match voice.get("temperature") {
        None | Some(Value::Null) => {}
        Some(Value::Number(n)) => {
            let t = n.as_f64().unwrap_or(f64::NAN);
            if !(0.0..=1.0).contains(&t) {
                return Err(invalid("voice.temperature must be between 0.0 and 1.0"));
            }
        }
        Some(_) => return Err(invalid("voice.temperature must be a number or null")),
    }
    match voice.get("prefer_locales") {
        None | Some(Value::Null) => {}
        Some(Value::Array(items)) if items.iter().all(Value::is_string) => {}
        Some(_) => {
            return Err(invalid(
                "voice.prefer_locales must be an array of strings or null",
            ))
        }
    }
    Ok(voice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn field_names_normalize_like_upstream() {
        assert_eq!(normalize_field_name("APIKey"), "api_key");
        assert_eq!(normalize_field_name("azureApiKey"), "azure_api_key");
        assert_eq!(normalize_field_name("SASUrl"), "sas_url");
        assert_eq!(normalize_field_name("client-secret"), "client_secret");
        assert_eq!(normalize_field_name("passwordCount"), "password_count");
        assert_eq!(normalize_field_name("__x__"), "x");
    }

    #[test]
    fn oversized_integers_are_found_outside_strings_only() {
        let big = "1".repeat(129);
        assert!(has_oversized_integer(&format!(r#"{{"a":{big}}}"#)));
        assert!(has_oversized_integer(&format!(r#"{{"a":-{big}}}"#)));
        assert!(!has_oversized_integer(&format!(r#"{{"a":"{big}"}}"#)));
        assert!(!has_oversized_integer(&format!(r#"{{"a":{big}.5}}"#)));
        assert!(!has_oversized_integer(&format!(
            r#"{{"a":{}}}"#,
            "1".repeat(128)
        )));
    }

    #[test]
    fn timestamps_follow_the_profile() {
        for ok in [
            "2026-08-12T00:00:00Z",
            "2026-08-12T23:59:59.123456789Z",
            "2024-02-29T00:00:00+05:30",
            "2026-08-12T00:00:00-23:59",
        ] {
            assert_eq!(check_timestamp(ok), Ok(()), "{ok}");
        }
        for bad in [
            "2026-08-12 00:00:00Z",
            "2026-08-12T24:00:00Z",
            "2026-08-12T00:00:00",
            "2026-08-12T00:00:00.Z",
            "2026-08-12T00:00:00.1234567890Z",
            "2026-08-12T00:00:00z",
            "2026-08-12T00:00:00Z\n",
            "٢٠٢٦-08-12T00:00:00Z",
        ] {
            assert_eq!(
                check_timestamp(bad),
                Err("ts must match the RFC 3339 profile".into()),
                "{bad}"
            );
        }
        for bad in [
            "2026-02-29T00:00:00Z",
            "2026-13-01T00:00:00Z",
            "0000-01-01T00:00:00Z",
        ] {
            assert_eq!(
                check_timestamp(bad),
                Err("ts must contain a valid calendar timestamp".into()),
                "{bad}"
            );
        }
    }
}
