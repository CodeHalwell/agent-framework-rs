//! The thin, send-only connection context (upstream `voice/_session.py`).

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};

use super::codec::encode_outbound_message;
use super::models::{OutboundVoiceMessage, SessionDisconnected};
use super::turn::{TargetTurn, TargetTurnOrigin};
use super::VoiceError;
use crate::scope::FoundryRequestContext;

/// How long the host waits for its close frame to be written.
pub const CLOSE_TIMEOUT: Duration = Duration::from_secs(5);

/// The first source-aware terminal fact for one physical connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SessionTermination {
    /// `session.end` was handled, or the peer closed normally (1000/1001).
    Completed,
    /// The connection task was cancelled.
    Cancelled,
    /// A frame violated the protocol, or the peer closed with a protocol
    /// close code (1002, 1003, 1007-1010).
    ProtocolError,
    /// The WebSocket upgrade failed.
    AcceptError,
    /// An event callback failed or panicked.
    CallbackError,
    /// The host failed outside a callback.
    InternalError,
    /// The transport failed, or the peer closed abnormally.
    TransportError,
}

impl SessionTermination {
    /// The wire / telemetry value (`completed`, `protocol_error`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::ProtocolError => "protocol_error",
            Self::AcceptError => "accept_error",
            Self::CallbackError => "callback_error",
            Self::InternalError => "internal_error",
            Self::TransportError => "transport_error",
        }
    }
}

/// One frame for the connection loop to write; the loop answers on `ack`.
pub(crate) struct Outgoing {
    pub(crate) frame: String,
    pub(crate) ack: oneshot::Sender<Result<(), VoiceError>>,
}

#[derive(Default)]
struct State {
    terminal: bool,
    termination: Option<SessionTermination>,
    close_code: Option<u16>,
    disconnect: Option<SessionDisconnected>,
    /// The connection span, released once terminal (so a handle the
    /// application keeps does not keep the span open).
    connection_span: Option<tracing::Span>,
}

struct Inner {
    outgoing: mpsc::Sender<Outgoing>,
    state: Mutex<State>,
    context: FoundryRequestContext,
}

/// The send-only context for one accepted Voice connection. Cloning gives
/// another handle to the same connection (to send from an application task).
///
/// It keeps only write serialization and a terminal write gate: the agent
/// owns response ids, pending work, generation tasks, terminal-event
/// correlation, cancellation, history, and reconnect restoration.
#[derive(Clone)]
pub struct Session {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("termination", &self.termination())
            .finish_non_exhaustive()
    }
}

impl Session {
    pub(crate) fn new(
        outgoing: mpsc::Sender<Outgoing>,
        context: FoundryRequestContext,
        connection_span: tracing::Span,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                outgoing,
                state: Mutex::new(State {
                    connection_span: Some(connection_span),
                    ..State::default()
                }),
                context,
            }),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Whether two handles are the same connection.
    pub fn same_connection(&self, other: &Session) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// The first terminal fact of the physical connection, once committed.
    pub fn termination(&self) -> Option<SessionTermination> {
        self.state().termination
    }

    /// The connection's request context (platform headers and session id;
    /// upstream sets it as the ambient request context).
    pub fn context(&self) -> &FoundryRequestContext {
        &self.inner.context
    }

    /// Encode and send one agent-to-Bridge event. Concurrent sends are
    /// written in order, one at a time; this returns once the frame is
    /// written. It does not retry, infer commitment, or track protocol
    /// state. Fails with [`VoiceError::Terminating`] once the session is
    /// terminating, [`VoiceError::Transport`] when the client is gone, and
    /// [`VoiceError::Invalid`] for a message the codec rejects.
    pub async fn send(&self, message: impl Into<OutboundVoiceMessage>) -> Result<(), VoiceError> {
        self.ensure_writable()?;
        let frame = encode_outbound_message(&message.into())?;
        let (ack, done) = oneshot::channel();
        if self
            .inner
            .outgoing
            .send(Outgoing { frame, ack })
            .await
            .is_err()
        {
            return Err(self.gone());
        }
        done.await.unwrap_or_else(|_| Err(self.gone()))
    }

    fn gone(&self) -> VoiceError {
        if self.state().terminal {
            VoiceError::Terminating
        } else {
            VoiceError::Transport("Voice WebSocket send failed: the connection is closed".into())
        }
    }

    fn ensure_writable(&self) -> Result<(), VoiceError> {
        if self.state().terminal {
            return Err(VoiceError::Terminating);
        }
        Ok(())
    }

    /// Start one application-owned target-agent decision trace (an
    /// `invoke_agent` span under the connection span). `input_count` is the
    /// number of inputs the decision consumes: at least 1, except 0 for
    /// [`TargetTurnOrigin::Proactive`] (which requires 0) and
    /// [`TargetTurnOrigin::Other`].
    pub fn start_target_turn(
        &self,
        origin: TargetTurnOrigin,
        input_count: u64,
    ) -> Result<TargetTurn, VoiceError> {
        let span = self.live_span()?;
        TargetTurn::create(&span, origin, input_count)
    }

    fn live_span(&self) -> Result<tracing::Span, VoiceError> {
        let state = self.state();
        match (&state.connection_span, state.terminal) {
            (Some(span), false) => Ok(span.clone()),
            _ => Err(VoiceError::Terminating),
        }
    }

    /// [`start_target_turn`](Self::start_target_turn), linked to an ended
    /// trigger span (feature `otel`). The link drops the trigger's
    /// `tracestate`.
    #[cfg(feature = "otel")]
    #[cfg_attr(docsrs, doc(cfg(feature = "otel")))]
    pub fn start_target_turn_linked(
        &self,
        origin: TargetTurnOrigin,
        input_count: u64,
        trigger: opentelemetry::trace::SpanContext,
    ) -> Result<TargetTurn, VoiceError> {
        let span = self.live_span()?;
        let turn = TargetTurn::create(&span, origin, input_count)?;
        turn.link(trigger)?;
        Ok(turn)
    }

    // -- host side ----------------------------------------------------------

    pub(crate) fn is_terminal(&self) -> bool {
        self.state().terminal
    }

    /// Close the write gate, committing `termination` if none is yet.
    pub(crate) fn begin_termination(&self, termination: Option<SessionTermination>) {
        let mut state = self.state();
        state.terminal = true;
        state.connection_span = None;
        if state.termination.is_none() {
            state.termination = termination;
        }
    }

    /// Select the close code, unless one was selected first.
    pub(crate) fn select_close_code(&self, code: u16) {
        self.state().close_code.get_or_insert(code);
    }

    pub(crate) fn selected_close_code(&self) -> Option<u16> {
        self.state().close_code
    }

    /// The peer (or the transport) ended the connection: the first such
    /// fact selects the close code, the termination, and the disconnect
    /// event.
    pub(crate) fn observe_disconnect(
        &self,
        code: u16,
        reason: Option<String>,
        termination: SessionTermination,
    ) {
        let mut state = self.state();
        state.terminal = true;
        state.connection_span = None;
        state.close_code.get_or_insert(code);
        if state.termination.is_none() {
            state.termination = Some(termination);
        }
        if state.disconnect.is_none() {
            state.disconnect = Some(SessionDisconnected { code, reason });
        }
    }

    pub(crate) fn peek_disconnect(&self) -> Option<SessionDisconnected> {
        self.state().disconnect.clone()
    }

    pub(crate) fn take_disconnect(&self) -> Option<SessionDisconnected> {
        self.state().disconnect.take()
    }
}
