//! The Voice Live Bridge text/control profile messages (upstream
//! `voice/_models.py`).
//!
//! Inbound messages come from the Bridge and are built by the codec
//! ([`decode_inbound_message`](super::decode_inbound_message)), which
//! validates them first. Outbound messages are built by the agent; each
//! constructor generates the message's own `id` (`m_…`) and `ts` (UTC
//! RFC 3339, millisecond precision), as upstream's defaults do. Field names
//! are the wire names.
//!
//! `Debug` output redacts payload fields (text, caller context, data,
//! voice patches, messages) and is bounded to 1024 characters, as
//! upstream's `repr`; identifiers, codes and counts are shown.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Map, Value};

/// A JSON object (application data, caller context, voice patches).
pub type JsonObject = Map<String, Value>;

/// A new Bridge message id (`m_` and 32 hex digits).
pub fn new_message_id() -> String {
    format!("m_{}", uuid::Uuid::new_v4().simple())
}

/// A new agent response id (`r_` and 32 hex digits).
pub fn new_response_id() -> String {
    format!("r_{}", uuid::Uuid::new_v4().simple())
}

/// A new agent output-item id (`it_` and 32 hex digits).
pub fn new_item_id() -> String {
    format!("it_{}", uuid::Uuid::new_v4().simple())
}

/// The current UTC time as RFC 3339 with millisecond precision
/// (`2026-08-12T00:00:00.000Z`).
pub fn new_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs() as i64;
    let (year, month, day) = civil_from_days(secs.div_euclid(86_400));
    let of_day = secs.rem_euclid(86_400);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z",
        of_day / 3600,
        of_day % 3600 / 60,
        of_day % 60,
        now.subsec_millis()
    )
}

/// Days since 1970-01-01 to a proleptic Gregorian date.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

// ---------------------------------------------------------------------------
// Bounded, redacting Debug (upstream `_voice_model_repr`).

const MAX_MODEL_DEBUG_LENGTH: usize = 1024;
const MAX_FIELD_DEBUG_LENGTH: usize = 128;
const REDACTED: &str = "<redacted>";
const SAFE_DEBUG_FIELDS: &[&str] = &[
    "admission_timeout_ms",
    "code",
    "count",
    "first_output_ms",
    "id",
    "idle_ms",
    "in_reply_to",
    "item_id",
    "item_ids",
    "max_duration_ms",
    "mode",
    "no_input_timeout_ms",
    "protocol_version",
    "reconnect",
    "response_id",
    "response_timeouts",
    "retriable",
    "stage",
    "ts",
    "type",
];

fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let keep: String = text.chars().take(max.saturating_sub(3)).collect();
    format!("{keep}...")
}

#[doc(hidden)]
pub fn debug_field<T: Serialize + ?Sized>(name: &str, value: &T) -> String {
    let rendered = serde_json::to_value(value).unwrap_or(Value::Null);
    let shown = if !rendered.is_null() && !SAFE_DEBUG_FIELDS.contains(&name) {
        REDACTED.to_string()
    } else {
        truncate(&rendered.to_string(), MAX_FIELD_DEBUG_LENGTH)
    };
    format!("{name}={shown}")
}

#[doc(hidden)]
pub fn debug_finish(name: &str, fields: &[String], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let rendered = format!("{name}({})", fields.join(", "));
    if rendered.chars().count() <= MAX_MODEL_DEBUG_LENGTH {
        return f.write_str(&rendered);
    }
    let keep: String = rendered.chars().take(MAX_MODEL_DEBUG_LENGTH - 4).collect();
    write!(f, "{keep}...)")
}

macro_rules! redacted_debug {
    ($ty:ident { $($field:ident),* $(,)? }) => {
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let fields = [$(debug_field(stringify!($field), &self.$field)),*];
                debug_finish(stringify!($ty), &fields, f)
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Shared value types.

/// Response deadlines selected by the Bridge.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseTimeouts {
    /// Maximum milliseconds until first output.
    pub first_output_ms: u64,
    /// Maximum milliseconds between output progress messages.
    pub idle_ms: u64,
    /// Absolute maximum response duration in milliseconds.
    pub max_duration_ms: u64,
}
redacted_debug!(ResponseTimeouts {
    first_output_ms,
    idle_ms,
    max_duration_ms
});

/// One part of an ordered `user.message` (wire `content[]`).
#[derive(Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InputContentPart {
    /// `input_text`: final recognized or application-supplied text.
    #[serde(rename = "input_text")]
    InputText {
        /// The text.
        text: String,
    },
    /// `input_data`: one application-defined JSON object.
    #[serde(rename = "input_data")]
    InputData {
        /// The object.
        data: JsonObject,
    },
}

impl fmt::Debug for InputContentPart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputText { text } => debug_finish(
                "InputTextPart",
                &[debug_field("text", text), "type=\"input_text\"".into()],
                f,
            ),
            Self::InputData { data } => debug_finish(
                "InputDataPart",
                &[debug_field("data", data), "type=\"input_data\"".into()],
                f,
            ),
        }
    }
}

/// The local transport-disconnect event (not a Bridge wire message): the
/// peer or a proxy closed the WebSocket, or a send found it gone.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SessionDisconnected {
    /// The RFC 6455 close code the transport reported.
    pub code: u16,
    /// The close reason the transport reported, if any.
    pub reason: Option<String>,
}
redacted_debug!(SessionDisconnected { code, reason });

// ---------------------------------------------------------------------------
// Inbound (Bridge to agent).

/// `session.start`: the Bridge application-start event.
#[derive(Clone, PartialEq, Serialize)]
pub struct SessionStart {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The exact Bridge protocol version (see
    /// [`PROTOCOL_VERSION`](super::PROTOCOL_VERSION)).
    pub protocol_version: String,
    /// Whether this transport reattaches the logical session.
    pub reconnect: bool,
    /// Effective response deadlines.
    pub response_timeouts: ResponseTimeouts,
    /// A Bridge-owned greeting (never present on reconnect).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub greeting: Option<String>,
    /// The user-silence threshold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_input_timeout_ms: Option<u64>,
    /// Caller context (never contains credential-named fields).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<JsonObject>,
}
redacted_debug!(SessionStart {
    id,
    ts,
    protocol_version,
    reconnect,
    response_timeouts,
    greeting,
    no_input_timeout_ms,
    caller
});

/// `user.message`: a completed user turn of ordered text and data parts.
#[derive(Clone, PartialEq, Serialize)]
pub struct UserMessage {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The Bridge-allocated input item id (`in_…`).
    pub item_id: String,
    /// The supported parts, in wire order (unknown part types are skipped).
    pub content: Vec<InputContentPart>,
}
redacted_debug!(UserMessage {
    id,
    ts,
    item_id,
    content
});

/// `user.no_input`: a Bridge-generated user-silence turn.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct UserNoInput {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The Bridge-allocated input item id (`in_…`).
    pub item_id: String,
    /// The consecutive no-input count.
    pub count: u64,
}
redacted_debug!(UserNoInput {
    id,
    ts,
    item_id,
    count
});

/// `user.speech_started`: caller speech began while no response was open
/// (advisory).
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct UserSpeechStarted {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
}
redacted_debug!(UserSpeechStarted { id, ts });

/// `user.barge_in`: the caller interrupted, with the playback snapshot.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct UserBargeIn {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The interrupted response (`r_…`).
    pub response_id: String,
    /// The text played before the interruption.
    pub heard_text: String,
    /// The output item playing at the interruption (`it_…`), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}
redacted_debug!(UserBargeIn {
    id,
    ts,
    response_id,
    heard_text,
    item_id
});

/// `response.accepted`: the Bridge admitted a proactive response.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseAccepted {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The accepted proactive response (`r_…`).
    pub response_id: String,
}
redacted_debug!(ResponseAccepted {
    id,
    ts,
    response_id
});

/// `response.dropped`: the Bridge rejected or expired a proactive request.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseDropped {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The dropped proactive response (`r_…`).
    pub response_id: String,
    /// The open-enum drop reason.
    pub reason: String,
}
redacted_debug!(ResponseDropped {
    id,
    ts,
    response_id,
    reason
});

/// `response.cancelled`: the winning self-cancel playback outcome.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseCancelled {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The cancelled response (`r_…`).
    pub response_id: String,
    /// The text played before cancellation completed.
    pub heard_text: String,
    /// The output item playing at cancellation (`it_…`), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}
redacted_debug!(ResponseCancelled {
    id,
    ts,
    response_id,
    heard_text,
    item_id
});

/// `response.timeout`: a response or pending-input timeout. Exactly one of
/// `response_id` and `item_ids` is set.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseTimeout {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The open-enum timeout stage.
    pub stage: String,
    /// The timed-out response, when one was open.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The timed-out inputs, before a response was created.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_ids: Option<Vec<String>>,
}
redacted_debug!(ResponseTimeout {
    id,
    ts,
    stage,
    response_id,
    item_ids
});

/// `session.end`: Bridge-initiated session termination.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SessionEnd {
    /// Wire message id.
    pub id: String,
    /// Sender timestamp.
    pub ts: String,
    /// The open-enum termination reason.
    pub reason: String,
}
redacted_debug!(SessionEnd { id, ts, reason });

/// Every inbound message the profile selects, tagged by its wire `type`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum InboundVoiceMessage {
    /// `session.start`
    #[serde(rename = "session.start")]
    SessionStart(SessionStart),
    /// `user.message`
    #[serde(rename = "user.message")]
    UserMessage(UserMessage),
    /// `user.no_input`
    #[serde(rename = "user.no_input")]
    UserNoInput(UserNoInput),
    /// `user.speech_started`
    #[serde(rename = "user.speech_started")]
    UserSpeechStarted(UserSpeechStarted),
    /// `user.barge_in`
    #[serde(rename = "user.barge_in")]
    UserBargeIn(UserBargeIn),
    /// `response.accepted`
    #[serde(rename = "response.accepted")]
    ResponseAccepted(ResponseAccepted),
    /// `response.dropped`
    #[serde(rename = "response.dropped")]
    ResponseDropped(ResponseDropped),
    /// `response.cancelled`
    #[serde(rename = "response.cancelled")]
    ResponseCancelled(ResponseCancelled),
    /// `response.timeout`
    #[serde(rename = "response.timeout")]
    ResponseTimeout(ResponseTimeout),
    /// `session.end`
    #[serde(rename = "session.end")]
    SessionEnd(SessionEnd),
}

impl InboundVoiceMessage {
    /// The wire `type`.
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::SessionStart(_) => "session.start",
            Self::UserMessage(_) => "user.message",
            Self::UserNoInput(_) => "user.no_input",
            Self::UserSpeechStarted(_) => "user.speech_started",
            Self::UserBargeIn(_) => "user.barge_in",
            Self::ResponseAccepted(_) => "response.accepted",
            Self::ResponseDropped(_) => "response.dropped",
            Self::ResponseCancelled(_) => "response.cancelled",
            Self::ResponseTimeout(_) => "response.timeout",
            Self::SessionEnd(_) => "session.end",
        }
    }

    /// The wire message id.
    pub fn id(&self) -> &str {
        match self {
            Self::SessionStart(m) => &m.id,
            Self::UserMessage(m) => &m.id,
            Self::UserNoInput(m) => &m.id,
            Self::UserSpeechStarted(m) => &m.id,
            Self::UserBargeIn(m) => &m.id,
            Self::ResponseAccepted(m) => &m.id,
            Self::ResponseDropped(m) => &m.id,
            Self::ResponseCancelled(m) => &m.id,
            Self::ResponseTimeout(m) => &m.id,
            Self::SessionEnd(m) => &m.id,
        }
    }
}

// ---------------------------------------------------------------------------
// Outbound (agent to Bridge).

/// `session.ready`: explicit positive readiness acknowledgement.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SessionReady {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
}
redacted_debug!(SessionReady { id, ts });

impl SessionReady {
    /// A new acknowledgement.
    pub fn new() -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
        }
    }
}

impl Default for SessionReady {
    fn default() -> Self {
        Self::new()
    }
}

/// `session.rejected`: explicit negative readiness acknowledgement.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct SessionRejected {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// Open-enum rejection code (for example `protocol_mismatch`).
    pub code: String,
    /// Whether the Bridge may retry activation.
    pub retriable: bool,
    /// Sanitized diagnostic detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}
redacted_debug!(SessionRejected {
    id,
    ts,
    code,
    retriable,
    message
});

impl SessionRejected {
    /// A rejection with `code`.
    pub fn new(code: impl Into<String>, retriable: bool) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            code: code.into(),
            retriable,
            message: None,
        }
    }

    /// Add a sanitized diagnostic.
    pub fn with_message(mut self, message: impl Into<String>) -> Self {
        self.message = Some(message.into());
        self
    }
}

/// `response.created`: open a reply response (`in_reply_to` set) or request
/// proactive admission (`in_reply_to` absent).
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseCreated {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The agent-allocated response id (`r_…`).
    pub response_id: String,
    /// The ordered input prefix a reply answers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_reply_to: Option<Vec<String>>,
    /// Proactive admission timeout (at most 60000).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub admission_timeout_ms: Option<u64>,
    /// Proactive supersession key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersede_key: Option<String>,
}
redacted_debug!(ResponseCreated {
    id,
    ts,
    response_id,
    in_reply_to,
    admission_timeout_ms,
    supersede_key
});

impl ResponseCreated {
    /// A reply to the ordered inputs `in_reply_to`.
    pub fn reply<I, S>(response_id: impl Into<String>, in_reply_to: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            in_reply_to: Some(in_reply_to.into_iter().map(Into::into).collect()),
            ..Self::proactive(response_id)
        }
    }

    /// A proactive admission request.
    pub fn proactive(response_id: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
            in_reply_to: None,
            admission_timeout_ms: None,
            supersede_key: None,
        }
    }

    /// Set the proactive admission timeout.
    pub fn with_admission_timeout_ms(mut self, timeout: u64) -> Self {
        self.admission_timeout_ms = Some(timeout);
        self
    }

    /// Set the proactive supersession key.
    pub fn with_supersede_key(mut self, key: impl Into<String>) -> Self {
        self.supersede_key = Some(key.into());
        self
    }
}

/// `response.none`: explicitly decline an ordered input prefix.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseNone {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The non-empty ordered input prefix being declined.
    pub in_reply_to: Vec<String>,
    /// Open-enum decline reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
redacted_debug!(ResponseNone {
    id,
    ts,
    in_reply_to,
    reason
});

impl ResponseNone {
    /// Decline `in_reply_to`.
    pub fn new<I, S>(in_reply_to: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            in_reply_to: in_reply_to.into_iter().map(Into::into).collect(),
            reason: None,
        }
    }

    /// Add the decline reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// `response.output_text.delta`: one streaming text increment.
#[derive(Clone, PartialEq, Serialize)]
pub struct ResponseOutputTextDelta {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The owning response (`r_…`).
    pub response_id: String,
    /// The output item (`it_…`).
    pub item_id: String,
    /// The non-empty text increment.
    pub delta: String,
    /// A Voice Live synthesis merge patch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<JsonObject>,
}
redacted_debug!(ResponseOutputTextDelta {
    id,
    ts,
    response_id,
    item_id,
    delta,
    voice
});

impl ResponseOutputTextDelta {
    /// A text increment.
    pub fn new(
        response_id: impl Into<String>,
        item_id: impl Into<String>,
        delta: impl Into<String>,
    ) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
            item_id: item_id.into(),
            delta: delta.into(),
            voice: None,
        }
    }

    /// Add a voice merge patch.
    pub fn with_voice(mut self, voice: JsonObject) -> Self {
        self.voice = Some(voice);
        self
    }
}

/// `response.output_text.done`: complete one output item's text.
#[derive(Clone, PartialEq, Serialize)]
pub struct ResponseOutputTextDone {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The owning response (`r_…`).
    pub response_id: String,
    /// The output item (`it_…`).
    pub item_id: String,
    /// The complete item text.
    pub text: String,
    /// A Voice Live synthesis merge patch.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<JsonObject>,
}
redacted_debug!(ResponseOutputTextDone {
    id,
    ts,
    response_id,
    item_id,
    text,
    voice
});

impl ResponseOutputTextDone {
    /// A completed text item.
    pub fn new(
        response_id: impl Into<String>,
        item_id: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
            item_id: item_id.into(),
            text: text.into(),
            voice: None,
        }
    }

    /// Add a voice merge patch.
    pub fn with_voice(mut self, voice: JsonObject) -> Self {
        self.voice = Some(voice);
        self
    }
}

/// `response.output_data.done`: complete one application-data item.
#[derive(Clone, PartialEq, Serialize)]
pub struct ResponseOutputDataDone {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The owning response (`r_…`).
    pub response_id: String,
    /// The output item (`it_…`).
    pub item_id: String,
    /// The application-defined object.
    pub data: JsonObject,
}
redacted_debug!(ResponseOutputDataDone {
    id,
    ts,
    response_id,
    item_id,
    data
});

impl ResponseOutputDataDone {
    /// A completed data item.
    pub fn new(
        response_id: impl Into<String>,
        item_id: impl Into<String>,
        data: JsonObject,
    ) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
            item_id: item_id.into(),
            data,
        }
    }
}

/// `response.done`: the normal response-level terminator.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseDone {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The completed response (`r_…`).
    pub response_id: String,
}
redacted_debug!(ResponseDone {
    id,
    ts,
    response_id
});

impl ResponseDone {
    /// Complete `response_id`.
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
        }
    }
}

/// `response.cancel`: cancel an open or pending proactive response.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct ResponseCancel {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The response or pending proactive request (`r_…`).
    pub response_id: String,
    /// Open-enum cancellation reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}
redacted_debug!(ResponseCancel {
    id,
    ts,
    response_id,
    reason
});

impl ResponseCancel {
    /// Cancel `response_id`.
    pub fn new(response_id: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            response_id: response_id.into(),
            reason: None,
        }
    }

    /// Add the cancellation reason.
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
}

/// How `end_call` ends the call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EndCallMode {
    /// Finish queued playback first (the default).
    #[default]
    Drain,
    /// End now.
    Immediate,
}

impl EndCallMode {
    /// The wire value.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Drain => "drain",
            Self::Immediate => "immediate",
        }
    }
}

/// `end_call`: ask the Bridge to end the call.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct EndCall {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// The non-empty call-ending reason.
    pub reason: String,
    /// The call-ending mode.
    pub mode: EndCallMode,
}
redacted_debug!(EndCall {
    id,
    ts,
    reason,
    mode
});

impl EndCall {
    /// End the call (mode `drain`).
    pub fn new(reason: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            reason: reason.into(),
            mode: EndCallMode::Drain,
        }
    }

    /// Set the mode.
    pub fn with_mode(mut self, mode: EndCallMode) -> Self {
        self.mode = mode;
        self
    }
}

/// `error`: an explicit response- or session-scoped agent failure
/// (upstream `AgentError`).
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct AgentError {
    /// Generated wire message id.
    pub id: String,
    /// Generated sender timestamp.
    pub ts: String,
    /// Open-enum error code.
    pub code: String,
    /// Sanitized diagnostic detail.
    pub message: String,
    /// The related response, when response-scoped (`r_…`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_id: Option<String>,
    /// The related output item (`it_…`; requires `response_id`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}
redacted_debug!(AgentError {
    id,
    ts,
    code,
    message,
    response_id,
    item_id
});

impl AgentError {
    /// A session-scoped error.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            id: new_message_id(),
            ts: new_timestamp(),
            code: code.into(),
            message: message.into(),
            response_id: None,
            item_id: None,
        }
    }

    /// Scope the error to a response.
    pub fn with_response_id(mut self, response_id: impl Into<String>) -> Self {
        self.response_id = Some(response_id.into());
        self
    }

    /// Name the related output item.
    pub fn with_item_id(mut self, item_id: impl Into<String>) -> Self {
        self.item_id = Some(item_id.into());
        self
    }
}

/// Every outbound message the profile selects, tagged by its wire `type`.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum OutboundVoiceMessage {
    /// `session.ready`
    #[serde(rename = "session.ready")]
    SessionReady(SessionReady),
    /// `session.rejected`
    #[serde(rename = "session.rejected")]
    SessionRejected(SessionRejected),
    /// `response.created`
    #[serde(rename = "response.created")]
    ResponseCreated(ResponseCreated),
    /// `response.none`
    #[serde(rename = "response.none")]
    ResponseNone(ResponseNone),
    /// `response.output_data.done`
    #[serde(rename = "response.output_data.done")]
    ResponseOutputDataDone(ResponseOutputDataDone),
    /// `response.output_text.delta`
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta(ResponseOutputTextDelta),
    /// `response.output_text.done`
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone(ResponseOutputTextDone),
    /// `response.done`
    #[serde(rename = "response.done")]
    ResponseDone(ResponseDone),
    /// `response.cancel`
    #[serde(rename = "response.cancel")]
    ResponseCancel(ResponseCancel),
    /// `end_call`
    #[serde(rename = "end_call")]
    EndCall(EndCall),
    /// `error`
    #[serde(rename = "error")]
    Error(AgentError),
}

impl OutboundVoiceMessage {
    /// The wire `type`.
    pub fn message_type(&self) -> &'static str {
        match self {
            Self::SessionReady(_) => "session.ready",
            Self::SessionRejected(_) => "session.rejected",
            Self::ResponseCreated(_) => "response.created",
            Self::ResponseNone(_) => "response.none",
            Self::ResponseOutputDataDone(_) => "response.output_data.done",
            Self::ResponseOutputTextDelta(_) => "response.output_text.delta",
            Self::ResponseOutputTextDone(_) => "response.output_text.done",
            Self::ResponseDone(_) => "response.done",
            Self::ResponseCancel(_) => "response.cancel",
            Self::EndCall(_) => "end_call",
            Self::Error(_) => "error",
        }
    }
}

macro_rules! outbound_from {
    ($($variant:ident($ty:ident)),* $(,)?) => {
        $(impl From<$ty> for OutboundVoiceMessage {
            fn from(message: $ty) -> Self {
                Self::$variant(message)
            }
        })*
    };
}

outbound_from!(
    SessionReady(SessionReady),
    SessionRejected(SessionRejected),
    ResponseCreated(ResponseCreated),
    ResponseNone(ResponseNone),
    ResponseOutputDataDone(ResponseOutputDataDone),
    ResponseOutputTextDelta(ResponseOutputTextDelta),
    ResponseOutputTextDone(ResponseOutputTextDone),
    ResponseDone(ResponseDone),
    ResponseCancel(ResponseCancel),
    EndCall(EndCall),
    Error(AgentError),
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_proleptic_gregorian() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_723), (2024, 1, 1));
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }
}
