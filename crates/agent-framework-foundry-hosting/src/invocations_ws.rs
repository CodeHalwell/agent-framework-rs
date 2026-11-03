//! The agentserver's `/invocations_ws` WebSocket route (feature
//! `websocket`; upstream `InvocationAgentServerHost.ws_handler`).
//!
//! [`InvocationsWs`] serves one handler. For each connection the host
//! accepts the upgrade, then runs the handler with a [`WsSession`]: the
//! client's data frames ([`WsSession::recv`], `None` once the client
//! closes), a sender ([`WsSession::send`]), and the request's
//! [`FoundryRequestContext`] (call id, user id, session id and original
//! subject from the platform headers; a malformed `x-ms-original-subject`
//! closes the accepted connection with 1011 before the handler runs). When the handler returns the host closes the
//! connection with `1000` (normal), or `1011` ("Internal server error") when
//! it fails or panics, and logs a close event (target
//! `azure.ai.agentserver`) with the session id, close code and duration. Protocol pings and pongs are handled by the host: with a
//! ping interval (`WS_KEEPALIVE_INTERVAL` seconds in hosted containers, see
//! [`InvocationsWs::ping_interval_from_env`]) it pings the client so idle
//! connections survive proxy timeouts.
//!
//! Merge [`InvocationsWs::into_router`] into the host's router (for example
//! an [`InvocationsHostServer`](crate::invocations) router) to serve both.

use std::sync::Arc;
use std::time::{Duration, Instant};

use agent_framework_core::error::{Error, Result};
use axum::extract::ws::{CloseFrame, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;
use axum::routing::any;
use axum::Router;
use futures::future::BoxFuture;
use futures::{FutureExt, SinkExt, StreamExt};
use tokio::sync::mpsc;

pub use axum::extract::ws::Message;

use crate::config::FoundryHostConfig;
use crate::scope::FoundryRequestContext;

/// The route path.
pub const INVOCATIONS_WS_PATH: &str = "/invocations_ws";
/// The keepalive interval variable (seconds; `0` or unset disables).
pub const WS_KEEPALIVE_INTERVAL: &str = "WS_KEEPALIVE_INTERVAL";
/// The close code for a handler that returned.
pub const CLOSE_NORMAL: u16 = 1000;
/// The close code for a handler that failed.
pub const CLOSE_INTERNAL_ERROR: u16 = 1011;

type Handler = Arc<dyn Fn(WsSession) -> BoxFuture<'static, Result<()>> + Send + Sync>;

/// One connection, as the handler sees it. See the [module docs](self).
#[derive(Debug)]
pub struct WsSession {
    incoming: mpsc::Receiver<Message>,
    outgoing: mpsc::Sender<Message>,
    context: FoundryRequestContext,
}

impl WsSession {
    /// The client's next data frame (text or binary); `None` once the
    /// client has closed the connection.
    pub async fn recv(&mut self) -> Option<Message> {
        self.incoming.recv().await
    }

    /// Send a frame to the client.
    pub async fn send(&self, message: Message) -> Result<()> {
        self.outgoing
            .send(message)
            .await
            .map_err(|_| Error::Other("the WebSocket connection is closed".into()))
    }

    /// The connection's request context (platform headers and session id).
    pub fn context(&self) -> &FoundryRequestContext {
        &self.context
    }
}

/// Serves a handler on `/invocations_ws`. See the [module docs](self).
#[derive(Clone)]
pub struct InvocationsWs {
    handler: Handler,
    ping_interval: Option<Duration>,
    config: FoundryHostConfig,
}

impl std::fmt::Debug for InvocationsWs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InvocationsWs")
            .field("ping_interval", &self.ping_interval)
            .finish_non_exhaustive()
    }
}

impl InvocationsWs {
    /// Serve `handler`, without keepalive pings.
    pub fn new<F, Fut>(handler: F) -> Self
    where
        F: Fn(WsSession) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<()>> + Send + 'static,
    {
        Self {
            handler: Arc::new(move |session| handler(session).boxed()),
            ping_interval: None,
            config: FoundryHostConfig::local(),
        }
    }

    /// Ping the client every `interval` (a zero interval disables pings).
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

    /// The keepalive interval from `WS_KEEPALIVE_INTERVAL`.
    pub fn ping_interval_from_env() -> Result<Option<Duration>> {
        Self::ping_interval_from(|name| std::env::var(name).ok())
    }

    /// The keepalive interval from `lookup`'s `WS_KEEPALIVE_INTERVAL`
    /// (seconds, fractional allowed; unset, empty or `0` disables). A value
    /// that is not a non-negative finite number is an error, as upstream.
    pub fn ping_interval_from(lookup: impl Fn(&str) -> Option<String>) -> Result<Option<Duration>> {
        let Some(raw) = lookup(WS_KEEPALIVE_INTERVAL).filter(|r| !r.is_empty()) else {
            return Ok(None);
        };
        let seconds = raw
            .trim()
            .parse::<f64>()
            .ok()
            .filter(|s| s.is_finite() && *s >= 0.0)
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "Invalid value for {WS_KEEPALIVE_INTERVAL}: {raw:?} (expected a non-negative finite number)"
                ))
            })?;
        Ok((seconds > 0.0).then(|| Duration::from_secs_f64(seconds)))
    }

    /// The `/invocations_ws` route.
    pub fn into_router(self) -> Router {
        Router::new()
            .route(INVOCATIONS_WS_PATH, any(upgrade))
            .with_state(Arc::new(self))
    }
}

async fn upgrade(
    State(server): State<Arc<InvocationsWs>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let session_id = Some(server.config.session_id.clone())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
    // The platform binds one invocation to the connection: the subject
    // comes from the upgrade headers, never from message payloads.
    match FoundryRequestContext::from_headers(&headers, Some(session_id.clone())) {
        Ok(context) => ws.on_upgrade(move |socket| connection(server, socket, context)),
        Err(_) => {
            let started = Instant::now();
            ws.on_upgrade(move |socket| reject_invalid_subject(socket, session_id, started))
        }
    }
}

/// Accept, then close with 1011 without running the handler (upstream:
/// a malformed original-subject header).
async fn reject_invalid_subject(mut socket: WebSocket, session_id: String, started: Instant) {
    tracing::warn!("WebSocket rejected: invalid original-subject platform header.");
    let _ = socket
        .send(Message::Close(Some(CloseFrame {
            code: CLOSE_INTERNAL_ERROR,
            reason: "Internal server error".into(),
        })))
        .await;
    let _ = socket.close().await;
    tracing::info!(
        target: "azure.ai.agentserver",
        session_id = %session_id,
        close_code = CLOSE_INTERNAL_ERROR,
        duration_ms = started.elapsed().as_millis() as u64,
        error_code = "invalid_original_subject",
        "invocations_ws connection closed"
    );
}

async fn connection(server: Arc<InvocationsWs>, socket: WebSocket, context: FoundryRequestContext) {
    let started = Instant::now();
    let session_id = context.session_id.clone().unwrap_or_default();
    let (mut sink, mut stream) = socket.split();
    let (in_tx, in_rx) = mpsc::channel::<Message>(64);
    let (out_tx, mut out_rx) = mpsc::channel::<Message>(64);
    let session = WsSession {
        incoming: in_rx,
        outgoing: out_tx,
        context,
    };
    let mut handler = tokio::spawn((server.handler)(session));
    // The reader delivers the client's frames on its own task: when the
    // handler is not reading and its inbound queue is full, only the reader
    // waits (backpressure reaches the client), while the loop below keeps
    // sending the handler's output, keepalive pings and the close.
    let (closed_tx, mut closed_rx) = tokio::sync::oneshot::channel::<()>();
    let reader = tokio::spawn(async move {
        let mut in_tx = Some(in_tx);
        while let Some(frame) = stream.next().await {
            match frame {
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(Message::Ping(_) | Message::Pong(_)) => {}
                Ok(message) => {
                    // A handler that dropped its session no longer reads:
                    // keep reading (discarding) to notice the close.
                    if let Some(tx) = &in_tx {
                        if tx.send(message).await.is_err() {
                            in_tx = None;
                        }
                    }
                }
            }
        }
        // The handler's `recv` now yields `None`.
        drop(in_tx);
        let _ = closed_tx.send(());
    });
    let mut ping = server.ping_interval.map(|interval| {
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        ticker
    });
    let mut client_open = true;
    let mut outgoing_open = true;
    // Once the client is gone (it closed, or a write failed), the handler's
    // outbound channel is closed so its next `send` fails: a handler that
    // only produces output, never reading, would otherwise keep generating
    // it, with its side effects, for no one.
    fn client_gone(client_open: &mut bool, out_rx: &mut mpsc::Receiver<Message>) {
        *client_open = false;
        out_rx.close();
    }
    let outcome = loop {
        tokio::select! {
            outcome = &mut handler => break outcome,
            _ = &mut closed_rx, if client_open => client_gone(&mut client_open, &mut out_rx),
            message = out_rx.recv(), if outgoing_open && client_open => match message {
                Some(message) => {
                    if sink.send(message).await.is_err() {
                        client_gone(&mut client_open, &mut out_rx);
                    }
                }
                None => outgoing_open = false,
            },
            _ = async { ping.as_mut().expect("guarded").tick().await }, if ping.is_some() && client_open => {
                if sink.send(Message::Ping(Default::default())).await.is_err() {
                    client_gone(&mut client_open, &mut out_rx);
                }
            }
        }
    };
    reader.abort();
    // Frames the handler sent before returning go out before the close.
    while let Ok(message) = out_rx.try_recv() {
        if !client_open {
            break;
        }
        if sink.send(message).await.is_err() {
            break;
        }
    }
    let code = match &outcome {
        Ok(Ok(())) => CLOSE_NORMAL,
        Ok(Err(e)) => {
            tracing::error!(session_id = %session_id, error = %e, "invocations_ws handler failed");
            CLOSE_INTERNAL_ERROR
        }
        Err(e) => {
            tracing::error!(session_id = %session_id, error = %e, "invocations_ws handler panicked");
            CLOSE_INTERNAL_ERROR
        }
    };
    let reason = if code == CLOSE_INTERNAL_ERROR {
        "Internal server error"
    } else {
        ""
    };
    let _ = sink
        .send(Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        })))
        .await;
    let _ = sink.close().await;
    tracing::info!(
        target: "azure.ai.agentserver",
        session_id = %session_id,
        close_code = code,
        duration_ms = started.elapsed().as_millis() as u64,
        "invocations_ws connection closed"
    );
}
