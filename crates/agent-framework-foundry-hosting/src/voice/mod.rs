//! Voice invocations (feature `voice`; upstream's experimental
//! `azure.ai.agentserver.invocations.voice`): a typed Voice Live Bridge
//! event relay over WebSocket, on `/voice_conversation_engine` and on the
//! legacy `/invocations_ws`.
//!
//! Upstream: "The host performs only per-frame decoding and static callback
//! dispatch. Agent code owns IDs, application tasks, response lifecycle,
//! terminal-event correlation, cancellation, history, and reconnect
//! restoration."
//!
//! - [`VoiceAgentServerHost`] serves a [`VoiceHandler`]: one async method
//!   per inbound event, each a no-op by default (an event with no handler
//!   is ignored, as upstream with no registered callback). Events are
//!   dispatched in order, one at a time; the next frame is read only when
//!   the callback returns. Frames are read on a separate task and output is
//!   written by the connection loop, so a callback that is busy (or a
//!   handler task that never reads) cannot stall output.
//! - [`Session`] is the send-only connection context: [`Session::send`]
//!   encodes and writes one outbound event, serialized with every other
//!   send, and fails once the session is terminating or the client is gone.
//! - The codec ([`decode_inbound_message`], [`encode_outbound_message`])
//!   validates every frame with upstream's rules and messages. A frame that
//!   violates the profile closes the connection with its close code: 1002
//!   (malformed), 1003 (a binary frame, or a known but excluded message
//!   family or `input_image` content), or 1009 (over 1 MiB). Unknown
//!   message types are ignored for forward compatibility.
//! - Close codes: `session.end` (its callback succeeding) closes with 1000;
//!   a failing or panicking callback closes with 1011 ("Internal server
//!   error"); a peer close is not answered with a new close code, and its
//!   code is the connection's. [`Session::termination`] is the first
//!   terminal fact ([`SessionTermination`]).
//! - The handshake: the Bridge sends `session.start` with its
//!   `protocol_version`; the agent answers `session.ready`, or
//!   `session.rejected` (upstream's sample rejects anything but
//!   [`PROTOCOL_VERSION`] with code `protocol_mismatch`). As upstream, the
//!   host itself does not check the version.
//! - Tracing: each connection runs in an `agentserver.connection` span
//!   (server kind) that joins the caller's W3C trace (`traceparent`, a valid
//!   `tracestate`, and an allow-list of baggage ids, with the `otel`
//!   feature); each callback runs in a `voice.callback` child span; and
//!   [`Session::start_target_turn`] opens an application-owned
//!   `invoke_agent` span ([`TargetTurn`]). With `otel`, the connection and
//!   target durations and propagation failures are also recorded as
//!   metrics.
//!
//! ```no_run
//! use agent_framework_core::error::Result;
//! use agent_framework_foundry_hosting::voice::*;
//!
//! struct Echo;
//!
//! #[async_trait::async_trait]
//! impl VoiceHandler for Echo {
//!     async fn on_session_start(&self, session: &Session, event: SessionStart) -> Result<()> {
//!         if event.protocol_version != PROTOCOL_VERSION {
//!             session.send(SessionRejected::new("protocol_mismatch", false)).await?;
//!             return Ok(());
//!         }
//!         session.send(SessionReady::new()).await?;
//!         Ok(())
//!     }
//!
//!     async fn on_user_message(&self, session: &Session, event: UserMessage) -> Result<()> {
//!         let response_id = new_response_id();
//!         session.send(ResponseCreated::reply(&response_id, [event.item_id])).await?;
//!         session.send(ResponseOutputTextDone::new(&response_id, new_item_id(), "Hello.")).await?;
//!         session.send(ResponseDone::new(&response_id)).await?;
//!         Ok(())
//!     }
//! }
//!
//! # async fn demo() {
//! let router = VoiceAgentServerHost::new(Echo).into_router();
//! # let _ = router;
//! # }
//! ```

mod codec;
mod host;
mod models;
mod session;
mod telemetry;
mod turn;

use std::fmt;

pub use codec::{
    decode_inbound_message, encode_outbound_message, VoiceProtocolError, MAX_ADMISSION_TIMEOUT_MS,
    MAX_FRAME_BYTES, MAX_IDENTIFIER_BYTES, MAX_INTEGER_DIGITS, MAX_JSON_DEPTH, MAX_JSON_NODES,
};
pub use host::{
    classify_close_code, VoiceAgentServerHost, VoiceHandler, VOICE_CONVERSATION_ENGINE_PATH,
};
pub use models::*;
pub use session::{Session, SessionTermination, CLOSE_TIMEOUT};
pub use turn::{TargetTurn, TargetTurnOrigin, TargetTurnOutcome};

/// The Bridge protocol version this profile implements.
pub const PROTOCOL_VERSION: &str = "1.0";

/// Why a [`Session::send`] (or a target-turn operation) failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoiceError {
    /// The message or argument is invalid (upstream `TypeError` /
    /// `ValueError`, with upstream's message).
    Invalid(String),
    /// The session is terminating: its connection ended, or is ending.
    Terminating,
    /// The client is gone; the frame was not written.
    Transport(String),
}

impl fmt::Display for VoiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) | Self::Transport(message) => f.write_str(message),
            Self::Terminating => f.write_str("Voice Session is terminating"),
        }
    }
}

impl std::error::Error for VoiceError {}

impl From<VoiceError> for agent_framework_core::error::Error {
    fn from(error: VoiceError) -> Self {
        match error {
            VoiceError::Invalid(message) => Self::Content(message),
            other => Self::Other(other.to_string()),
        }
    }
}
