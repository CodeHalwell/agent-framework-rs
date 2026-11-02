//! The typed Voice relay over WebSocket (upstream `voice/_voice_host.py`).

use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework_core::error::Result;
use async_trait::async_trait;
use axum::extract::ws::{CloseFrame, Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use futures::{FutureExt, SinkExt, StreamExt};
use tokio::sync::{mpsc, oneshot};
use tracing::Instrument;

use super::codec::{
    decode_inbound_message, CLOSE_MESSAGE_TOO_BIG, CLOSE_UNSUPPORTED_DATA, MAX_FRAME_BYTES,
};
use super::models::*;
use super::session::{Outgoing, Session, SessionTermination, CLOSE_TIMEOUT};
use super::telemetry;
use super::VoiceError;
use crate::config::FoundryHostConfig;
use crate::invocations_ws::{CLOSE_INTERNAL_ERROR, CLOSE_NORMAL, INVOCATIONS_WS_PATH};
use crate::scope::{headers, FoundryRequestContext};

/// The standalone conversation-engine route.
pub const VOICE_CONVERSATION_ENGINE_PATH: &str = "/voice_conversation_engine";

/// Classify a close code: 1000/1001 completed, 1002/1003/1007-1010 a
/// protocol error, anything else a transport error.
pub fn classify_close_code(code: u16) -> SessionTermination {
    match code {
        1000 | 1001 => SessionTermination::Completed,
        1002 | 1003 | 1007 | 1008 | 1009 | 1010 => SessionTermination::ProtocolError,
        _ => SessionTermination::TransportError,
    }
}

/// The agent's Voice callbacks: one async method per inbound event (each a
/// no-op by default), the peer-disconnect callback, and the synchronous
/// connection-terminating signal. A callback that returns an error (or
/// panics) ends the connection with 1011.
#[async_trait]
pub trait VoiceHandler: Send + Sync + 'static {
    /// `session.start`: restore state when `reconnect`, then answer
    /// `session.ready` or `session.rejected`.
    async fn on_session_start(&self, _session: &Session, _event: SessionStart) -> Result<()> {
        Ok(())
    }

    /// `user.message`: a completed user turn.
    async fn on_user_message(&self, _session: &Session, _event: UserMessage) -> Result<()> {
        Ok(())
    }

    /// `user.no_input`: a user-silence turn.
    async fn on_user_no_input(&self, _session: &Session, _event: UserNoInput) -> Result<()> {
        Ok(())
    }

    /// `user.speech_started`: advisory.
    async fn on_user_speech_started(
        &self,
        _session: &Session,
        _event: UserSpeechStarted,
    ) -> Result<()> {
        Ok(())
    }

    /// `user.barge_in`: stop the interrupted response's work.
    async fn on_user_barge_in(&self, _session: &Session, _event: UserBargeIn) -> Result<()> {
        Ok(())
    }

    /// `response.accepted`: a proactive response was admitted.
    async fn on_response_accepted(
        &self,
        _session: &Session,
        _event: ResponseAccepted,
    ) -> Result<()> {
        Ok(())
    }

    /// `response.dropped`: a proactive request was rejected or expired.
    async fn on_response_dropped(&self, _session: &Session, _event: ResponseDropped) -> Result<()> {
        Ok(())
    }

    /// `response.cancelled`: the outcome of a self-cancel.
    async fn on_response_cancelled(
        &self,
        _session: &Session,
        _event: ResponseCancelled,
    ) -> Result<()> {
        Ok(())
    }

    /// `response.timeout`: stop the timed-out response's or inputs' work.
    async fn on_response_timeout(&self, _session: &Session, _event: ResponseTimeout) -> Result<()> {
        Ok(())
    }

    /// `session.end`: the last event dispatched; when it returns `Ok` the
    /// session is `Completed` and the host closes with 1000.
    async fn on_session_end(&self, _session: &Session, _event: SessionEnd) -> Result<()> {
        Ok(())
    }

    /// The peer (or the transport) closed the connection. Runs after the
    /// connection ends; sends fail.
    async fn on_disconnect(&self, _session: &Session, _event: SessionDisconnected) -> Result<()> {
        Ok(())
    }

    /// The connection handler is exiting (called once per connection). It
    /// must return promptly, be idempotent, and not send; use it to cancel
    /// connection-owned tasks. The host does not wait for them.
    fn on_connection_terminating(&self, _session: &Session) -> Result<()> {
        Ok(())
    }
}

/// Serves a [`VoiceHandler`] on `/voice_conversation_engine` and
/// `/invocations_ws` (upstream `VoiceAgentServerHost`). See the
/// [module docs](super).
///
/// Merge [`into_router`](Self::into_router) into the host's router; it
/// owns `/invocations_ws`, so it does not combine with
/// [`InvocationsWs`](crate::invocations_ws::InvocationsWs) (axum refuses
/// the overlapping route, as upstream refuses a conflicting route).
#[derive(Clone)]
pub struct VoiceAgentServerHost {
    handler: Arc<dyn VoiceHandler>,
    ping_interval: Option<Duration>,
    config: FoundryHostConfig,
}

impl std::fmt::Debug for VoiceAgentServerHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VoiceAgentServerHost")
            .field("ping_interval", &self.ping_interval)
            .finish_non_exhaustive()
    }
}

impl VoiceAgentServerHost {
    /// Serve `handler`, without keepalive pings.
    pub fn new(handler: impl VoiceHandler) -> Self {
        Self::from_arc(Arc::new(handler))
    }

    /// Serve a shared handler.
    pub fn from_arc(handler: Arc<dyn VoiceHandler>) -> Self {
        Self {
            handler,
            ping_interval: None,
            config: FoundryHostConfig::local(),
        }
    }

    /// Ping the client every `interval` (zero disables; see
    /// [`InvocationsWs::ping_interval_from_env`](crate::invocations_ws::InvocationsWs::ping_interval_from_env)).
    pub fn ping_interval(mut self, interval: Duration) -> Self {
        self.ping_interval = (!interval.is_zero()).then_some(interval);
        self
    }

    /// The host configuration (its session id names each connection's
    /// session; default: a new id per connection).
    pub fn with_config(mut self, config: FoundryHostConfig) -> Self {
        self.config = config;
        self
    }

    /// The `/voice_conversation_engine` and `/invocations_ws` routes.
    pub fn into_router(self) -> Router {
        Router::new()
            .route(INVOCATIONS_WS_PATH, any(upgrade))
            .route(VOICE_CONVERSATION_ENGINE_PATH, any(upgrade))
            .with_state(Arc::new(self))
    }
}

async fn upgrade(
    State(server): State<Arc<VoiceAgentServerHost>>,
    request_headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let started = Instant::now();
    let session_id = Some(server.config.session_id.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    let context = FoundryRequestContext::from_headers(&request_headers, Some(session_id.clone()));
    // The span goes to whichever runs: axum keeps the failure callback
    // alive for the whole connection, so it must not hold its own clone.
    let span = Arc::new(std::sync::Mutex::new(Some(telemetry::connection_span(
        &request_headers,
        &session_id,
    ))));
    let failed_span = span.clone();
    let failed_session = session_id.clone();
    let mut response = ws
        .max_message_size(TRANSPORT_FRAME_LIMIT)
        .max_frame_size(TRANSPORT_FRAME_LIMIT)
        .on_failed_upgrade(move |_error| {
            tracing::error!(target: "azure.ai.agentserver", "Voice WebSocket accept failed");
            let span = take_span(&failed_span);
            close_event(
                &span,
                &failed_session,
                CLOSE_INTERNAL_ERROR,
                Some("accept_failed"),
                SessionTermination::AcceptError.as_str(),
                started,
            );
        })
        .on_upgrade(move |socket| {
            let span = take_span(&span);
            connection(server, socket, context, span, started)
        });
    if let Ok(version) = HeaderValue::from_str(&crate::host::server_version()) {
        response
            .headers_mut()
            .insert(HeaderName::from_static(headers::SERVER_VERSION), version);
    }
    response
}

/// The transport's own frame limit. Frames up to it are read whole, so the
/// codec rejects one over [`MAX_FRAME_BYTES`] with a clean 1009 close; a
/// frame over it fails in the transport (also closed with 1009, but its
/// unread remainder may reset the connection).
const TRANSPORT_FRAME_LIMIT: usize = 4 * MAX_FRAME_BYTES;

fn take_span(slot: &std::sync::Mutex<Option<tracing::Span>>) -> tracing::Span {
    slot.lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take()
        .unwrap_or_else(tracing::Span::none)
}

/// What the reader hands the dispatcher.
enum Inbound {
    Frame(Message),
    /// The peer or transport ended the connection (already recorded on the
    /// session).
    Closed,
    /// A frame over [`MAX_FRAME_BYTES`].
    TooBig,
}

/// How dispatch ended.
enum Dispatch {
    /// `session.end` was handled.
    Completed,
    /// The peer or transport ended the connection.
    Disconnected,
    /// A frame violated the protocol: close with this code.
    Protocol(u16),
    /// A callback failed or panicked.
    CallbackFailed,
}

fn is_too_big(error: &axum::Error) -> bool {
    // tungstenite's capacity error ("Space limit exceeded: Message too
    // long"), raised by the configured `max_message_size`.
    let detail = error.to_string();
    detail.contains("Space limit exceeded") || detail.contains("too long")
}

async fn connection(
    server: Arc<VoiceAgentServerHost>,
    socket: WebSocket,
    context: FoundryRequestContext,
    span: tracing::Span,
    started: Instant,
) {
    let session_id = context.session_id.clone().unwrap_or_default();
    let (mut sink, mut stream) = socket.split();
    let (in_tx, in_rx) = mpsc::channel::<Inbound>(64);
    let (out_tx, mut out_rx) = mpsc::channel::<Outgoing>(64);
    let session = Session::new(out_tx, context, span.clone());

    // The reader delivers frames on its own task, so output keeps flowing
    // while a callback runs (or the inbound queue is full).
    let (gone_tx, mut gone_rx) = oneshot::channel::<()>();
    let reader_session = session.clone();
    // Set once the host starts its own close: the peer's reply is not a
    // disconnect.
    let closing = Arc::new(AtomicBool::new(false));
    let reader_closing = closing.clone();
    let mut reader = tokio::spawn(async move {
        let mut too_big = false;
        loop {
            let next = stream.next().await;
            if reader_closing.load(Ordering::SeqCst)
                && matches!(next, None | Some(Err(_)) | Some(Ok(Message::Close(_))))
            {
                return;
            }
            let item = match next {
                Some(Ok(Message::Ping(_) | Message::Pong(_))) => continue,
                Some(Ok(Message::Close(frame))) => {
                    let (code, reason) = match frame {
                        Some(frame) => (
                            frame.code,
                            Some(frame.reason.to_string()).filter(|r| !r.is_empty()),
                        ),
                        None => (1005, None),
                    };
                    reader_session.observe_disconnect(code, reason, classify_close_code(code));
                    Inbound::Closed
                }
                Some(Ok(message)) => {
                    // A dispatcher that has ended no longer reads: keep
                    // reading (discarding) to notice the close.
                    let _ = in_tx.send(Inbound::Frame(message)).await;
                    continue;
                }
                Some(Err(error)) if !too_big && is_too_big(&error) => {
                    // Keep reading (discarding) until the peer answers the
                    // 1009 close, so the connection is not reset under it.
                    too_big = true;
                    let _ = in_tx.send(Inbound::TooBig).await;
                    continue;
                }
                Some(Err(_)) | None if too_big => return,
                Some(Err(_)) | None => {
                    reader_session.observe_disconnect(
                        1006,
                        None,
                        SessionTermination::TransportError,
                    );
                    Inbound::Closed
                }
            };
            let _ = gone_tx.send(());
            let _ = in_tx.send(item).await;
            return;
        }
    });

    let mut dispatcher = tokio::spawn(dispatch(
        server.handler.clone(),
        session.clone(),
        in_rx,
        span.clone(),
    ));
    let mut ping = server.ping_interval.map(|interval| {
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker
    });
    let mut client_open = true;
    // Set once the outbound channel is closed and drained: `recv` would
    // then return `None` at once on every iteration, spinning the loop for
    // as long as a callback keeps running.
    let mut outgoing_open = true;
    // Once the client is gone, the outbound channel closes: a later send
    // fails at once, and queued ones are answered with an error.
    fn client_gone(client_open: &mut bool, out_rx: &mut mpsc::Receiver<Outgoing>) {
        *client_open = false;
        out_rx.close();
    }
    let transport_lost = |session: &Session| {
        session.observe_disconnect(1006, None, SessionTermination::TransportError)
    };
    let outcome = loop {
        tokio::select! {
            outcome = &mut dispatcher => break outcome,
            _ = &mut gone_rx, if client_open => client_gone(&mut client_open, &mut out_rx),
            outgoing = out_rx.recv(), if outgoing_open => {
                let Some(Outgoing { frame, ack }) = outgoing else {
                    outgoing_open = false;
                    continue;
                };
                let result = if session.is_terminal() {
                    Err(VoiceError::Terminating)
                } else if !client_open {
                    Err(VoiceError::Transport("Voice WebSocket send failed: the connection is closed".into()))
                } else if sink.send(Message::Text(frame.into())).await.is_err() {
                    transport_lost(&session);
                    client_gone(&mut client_open, &mut out_rx);
                    Err(VoiceError::Transport("Voice WebSocket send failed: the connection is closed".into()))
                } else {
                    Ok(())
                };
                let _ = ack.send(result);
            }
            _ = async { ping.as_mut().expect("guarded").tick().await }, if ping.is_some() && client_open => {
                if sink.send(Message::Ping(Default::default())).await.is_err() {
                    transport_lost(&session);
                    client_gone(&mut client_open, &mut out_rx);
                }
            }
        }
    };
    // The write gate is closed: anything still queued is refused.
    session.begin_termination(None);
    out_rx.close();
    while let Ok(Outgoing { ack, .. }) = out_rx.try_recv() {
        let _ = ack.send(Err(VoiceError::Terminating));
    }

    let (computed, mut error_code) = match outcome {
        Ok(Dispatch::Completed) => (CLOSE_NORMAL, None),
        Ok(Dispatch::Disconnected) => (session.peek_disconnect().map_or(1006, |d| d.code), None),
        Ok(Dispatch::Protocol(code)) => (code, None),
        Ok(Dispatch::CallbackFailed) => (CLOSE_INTERNAL_ERROR, Some("internal_error")),
        Err(_) => {
            session.begin_termination(Some(SessionTermination::InternalError));
            (CLOSE_INTERNAL_ERROR, Some("internal_error"))
        }
    };
    if error_code.is_some() {
        tracing::error!(target: "azure.ai.agentserver", "Voice WebSocket handler failed");
    }
    let close_code = session.selected_close_code().unwrap_or(computed);
    let disconnect = session.peek_disconnect();
    if session.termination().is_none() {
        session.begin_termination(Some(match (&error_code, &disconnect) {
            (Some(_), _) => SessionTermination::InternalError,
            (None, Some(event)) => classify_close_code(event.code),
            (None, None) => classify_close_code(close_code),
        }));
    }
    if client_open && disconnect.is_none() && !matches!(close_code, 1005 | 1006 | 1015) {
        let reason = if close_code == CLOSE_INTERNAL_ERROR {
            "Internal server error"
        } else {
            ""
        };
        closing.store(true, Ordering::SeqCst);
        let close = async {
            sink.send(Message::Close(Some(CloseFrame {
                code: close_code,
                reason: reason.into(),
            })))
            .await?;
            sink.close().await
        };
        if !matches!(tokio::time::timeout(CLOSE_TIMEOUT, close).await, Ok(Ok(()))) {
            tracing::debug!(target: "azure.ai.agentserver", "Voice WebSocket close failed");
        }
    }

    let terminating = std::panic::catch_unwind(AssertUnwindSafe(|| {
        server.handler.on_connection_terminating(&session)
    }));
    let termination_failed = !matches!(terminating, Ok(Ok(())));
    let mut disconnect_failed = false;
    if let Some(event) = session.take_disconnect() {
        let callback_span = telemetry::callback_span(&span, "disconnect");
        let result = AssertUnwindSafe(server.handler.on_disconnect(&session, event))
            .catch_unwind()
            .instrument(callback_span.clone())
            .await;
        if !matches!(result, Ok(Ok(()))) {
            telemetry::record_error(&callback_span, "callback_error");
            disconnect_failed = true;
        }
    }
    if termination_failed {
        tracing::error!(target: "azure.ai.agentserver", "Voice connection termination callback failed");
    }
    if disconnect_failed {
        tracing::error!(target: "azure.ai.agentserver", "Voice disconnect callback failed");
    }
    if termination_failed || disconnect_failed {
        error_code = Some("internal_error");
    }
    let outcome = session
        .termination()
        .unwrap_or(SessionTermination::InternalError)
        .as_str();
    close_event(&span, &session_id, close_code, error_code, outcome, started);
    // The connection span ends here, not after the linger below.
    drop(span);
    drop(session);
    if closing.load(Ordering::SeqCst) {
        // Linger for the peer's close reply (the reader discards anything
        // before it), so the connection is not reset under the close.
        let _ = tokio::time::timeout(CLOSE_TIMEOUT, &mut reader).await;
    }
    reader.abort();
}

/// Complete the connection span and log the close event (target
/// `azure.ai.agentserver`, upstream's `invocations_ws` structured fields).
fn close_event(
    span: &tracing::Span,
    session_id: &str,
    close_code: u16,
    error_code: Option<&str>,
    outcome: &str,
    started: Instant,
) {
    telemetry::complete_connection(span, outcome, close_code, started.elapsed());
    let session_id =
        telemetry::is_valid_correlation_id(telemetry::SESSION_ID_BAGGAGE_KEY, session_id)
            .then_some(session_id);
    tracing::event!(
        target: "azure.ai.agentserver",
        tracing::Level::INFO,
        azure.ai.agentserver.invocations_ws.close_code = close_code,
        azure.ai.agentserver.invocations_ws.duration_ms = started.elapsed().as_millis() as u64,
        azure.ai.agentserver.invocations_ws.session_id = session_id,
        azure.ai.agentserver.invocations_ws.error.code = error_code,
        "Voice connection closed"
    );
}

fn protocol_close(session: &Session, code: u16) -> Dispatch {
    session.select_close_code(code);
    session.begin_termination(Some(SessionTermination::ProtocolError));
    Dispatch::Protocol(code)
}

async fn invoke(
    handler: &dyn VoiceHandler,
    session: &Session,
    event: InboundVoiceMessage,
) -> Result<()> {
    match event {
        InboundVoiceMessage::SessionStart(e) => handler.on_session_start(session, e).await,
        InboundVoiceMessage::UserMessage(e) => handler.on_user_message(session, e).await,
        InboundVoiceMessage::UserNoInput(e) => handler.on_user_no_input(session, e).await,
        InboundVoiceMessage::UserSpeechStarted(e) => {
            handler.on_user_speech_started(session, e).await
        }
        InboundVoiceMessage::UserBargeIn(e) => handler.on_user_barge_in(session, e).await,
        InboundVoiceMessage::ResponseAccepted(e) => handler.on_response_accepted(session, e).await,
        InboundVoiceMessage::ResponseDropped(e) => handler.on_response_dropped(session, e).await,
        InboundVoiceMessage::ResponseCancelled(e) => {
            handler.on_response_cancelled(session, e).await
        }
        InboundVoiceMessage::ResponseTimeout(e) => handler.on_response_timeout(session, e).await,
        InboundVoiceMessage::SessionEnd(e) => handler.on_session_end(session, e).await,
    }
}

/// Decode each frame and dispatch it to its callback, in order, one at a
/// time (upstream `_handle_voice_connection`).
async fn dispatch(
    handler: Arc<dyn VoiceHandler>,
    session: Session,
    mut inbound: mpsc::Receiver<Inbound>,
    span: tracing::Span,
) -> Dispatch {
    loop {
        let text = match inbound.recv().await {
            Some(Inbound::Frame(Message::Text(text))) => text,
            Some(Inbound::Frame(Message::Binary(_))) => {
                return protocol_close(&session, CLOSE_UNSUPPORTED_DATA)
            }
            Some(Inbound::Frame(_)) => continue,
            Some(Inbound::TooBig) => return protocol_close(&session, CLOSE_MESSAGE_TOO_BIG),
            Some(Inbound::Closed) | None => {
                if session.peek_disconnect().is_none() {
                    session.observe_disconnect(1006, None, SessionTermination::TransportError);
                }
                return Dispatch::Disconnected;
            }
        };
        let event = match decode_inbound_message(text.as_str()) {
            Ok(Some(event)) => event,
            Ok(None) => continue,
            Err(error) => return protocol_close(&session, error.close_code()),
        };
        if let InboundVoiceMessage::SessionStart(start) = &event {
            span.record(
                "azure.ai.agentserver.invocations_ws.reconnect",
                start.reconnect,
            );
            if telemetry::is_safe_protocol_version(&start.protocol_version) {
                span.record(
                    "azure.ai.agentserver.invocations_ws.protocol_version",
                    start.protocol_version.as_str(),
                );
            }
        }
        let is_end = matches!(event, InboundVoiceMessage::SessionEnd(_));
        let callback_span = telemetry::callback_span(&span, event.message_type());
        let result = AssertUnwindSafe(invoke(&*handler, &session, event))
            .catch_unwind()
            .instrument(callback_span.clone())
            .await;
        match result {
            Ok(Ok(())) if is_end => {
                session.begin_termination(Some(SessionTermination::Completed));
                return Dispatch::Completed;
            }
            Ok(Ok(())) => {}
            failure => {
                if let Ok(Err(error)) = &failure {
                    tracing::debug!(target: "azure.ai.agentserver", error = %error, "Voice callback failed");
                }
                session.begin_termination(Some(SessionTermination::CallbackError));
                let termination = session.termination();
                let error_type = match termination {
                    Some(SessionTermination::TransportError) => "transport_error",
                    Some(SessionTermination::CallbackError) => "callback_error",
                    _ => "internal_error",
                };
                telemetry::record_error(&callback_span, error_type);
                // A callback that failed because the peer had gone (its
                // send failed) ends like the disconnect, not as a failure.
                if termination != Some(SessionTermination::CallbackError)
                    && session.peek_disconnect().is_some()
                {
                    return Dispatch::Disconnected;
                }
                return Dispatch::CallbackFailed;
            }
        }
    }
}
