//! WebSocket transport: connects to an MCP server over a WebSocket using the
//! "mcp" subprotocol and frames each JSON-RPC message as one text frame.
//!
//! Structurally this mirrors [`super::stdio::McpStdioTransport`]: a background
//! task reads frames off the socket and routes responses back to their
//! waiting caller by request id, while notifications and server-initiated
//! requests are logged and otherwise ignored (no sampling/roots support).

use std::collections::HashMap;
use std::sync::Mutex as StdMutex;

use async_trait::async_trait;
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::net::TcpStream;
use tokio::sync::{oneshot, Mutex as AsyncMutex};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::http::Uri;
use tokio_tungstenite::tungstenite::{ClientRequestBuilder, Message};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use agent_framework_core::error::{Error, Result};

use crate::protocol::{self, IdGenerator, IncomingMessage, RpcError};
use crate::transport::McpTransport;

/// The WebSocket subprotocol MCP servers expect during the handshake.
const MCP_SUBPROTOCOL: &str = "mcp";

type WsConn = WebSocketStream<MaybeTlsStream<TcpStream>>;
type WsSink = SplitSink<WsConn, Message>;
type WsSource = SplitStream<WsConn>;
type PendingMap = StdMutex<HashMap<i64, oneshot::Sender<std::result::Result<Value, RpcError>>>>;

/// An MCP transport backed by a WebSocket connection.
///
/// Connects with the `"mcp"` subprotocol advertised via `Sec-WebSocket-Protocol`
/// and, optionally, custom headers on the upgrade request (e.g. `Authorization`).
///
/// Note: `wss://` (TLS) connections require a process-wide `rustls`
/// `CryptoProvider` to be installed (e.g.
/// `rustls::crypto::ring::default_provider().install_default()`, called once
/// at process startup) — this crate does not install one itself so as not to
/// override a choice made elsewhere in the process. Plain `ws://` connections
/// (used by this crate's own tests) are unaffected.
pub struct McpWebsocketTransport {
    inner: std::sync::Arc<WsInner>,
}

struct WsInner {
    sink: AsyncMutex<WsSink>,
    pending: PendingMap,
    next_id: IdGenerator,
    reader_task: StdMutex<Option<JoinHandle<()>>>,
}

impl McpWebsocketTransport {
    /// Connect to the MCP server at `url` (`ws://` or `wss://`), negotiating
    /// the `"mcp"` subprotocol and sending `headers` (e.g. `Authorization`) on
    /// the upgrade request.
    pub async fn connect(url: &str, headers: &[(String, String)]) -> Result<Self> {
        let uri: Uri = url
            .parse()
            .map_err(|e| Error::Configuration(format!("invalid MCP websocket URL '{url}': {e}")))?;

        let mut builder = ClientRequestBuilder::new(uri).with_sub_protocol(MCP_SUBPROTOCOL);
        for (k, v) in headers {
            builder = builder.with_header(k.clone(), v.clone());
        }
        let request = builder
            .into_client_request()
            .map_err(|e| Error::Configuration(format!("invalid MCP websocket request: {e}")))?;

        let (stream, _response) = tokio_tungstenite::connect_async(request)
            .await
            .map_err(|e| {
                Error::service(format!("failed to connect to MCP websocket server: {e}"))
            })?;

        let (sink, source) = stream.split();
        let inner = std::sync::Arc::new(WsInner {
            sink: AsyncMutex::new(sink),
            pending: StdMutex::new(HashMap::new()),
            next_id: IdGenerator::new(),
            reader_task: StdMutex::new(None),
        });

        let reader_task = spawn_reader(source, inner.clone());
        *inner.reader_task.lock().unwrap() = Some(reader_task);

        Ok(Self { inner })
    }

    async fn write_text(&self, message: &Value) -> Result<()> {
        let text = serde_json::to_string(message)
            .map_err(|e| Error::service(format!("failed to encode MCP message: {e}")))?;
        let mut sink = self.inner.sink.lock().await;
        sink.send(Message::text(text))
            .await
            .map_err(|e| Error::service(format!("failed to write to MCP websocket: {e}")))
    }
}

#[async_trait]
impl McpTransport for McpWebsocketTransport {
    async fn call(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.inner.next_id.next();
        let (tx, rx) = oneshot::channel();
        self.inner.pending.lock().unwrap().insert(id, tx);

        let request = protocol::build_request(id, method, params);
        if let Err(e) = self.write_text(&request).await {
            self.inner.pending.lock().unwrap().remove(&id);
            return Err(e);
        }

        match rx.await {
            Ok(Ok(value)) => Ok(value),
            Ok(Err(rpc_err)) => Err(Error::service(rpc_err.to_string())),
            Err(_) => Err(Error::service(
                "MCP server closed the connection before responding",
            )),
        }
    }

    async fn notify(&self, method: &str, params: Value) -> Result<()> {
        let notification = protocol::build_notification(method, params);
        self.write_text(&notification).await
    }

    async fn close(&self) -> Result<()> {
        let mut sink = self.inner.sink.lock().await;
        // Best effort: a server that already dropped the connection makes
        // this fail, which is not itself an error worth surfacing.
        let _ = sink.send(Message::Close(None)).await;
        Ok(())
    }
}

impl Drop for WsInner {
    fn drop(&mut self) {
        if let Ok(mut task) = self.reader_task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        // Fail any requests still waiting on a response so callers don't hang.
        if let Ok(mut pending) = self.pending.lock() {
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err(RpcError {
                    code: -1,
                    message: "MCP websocket transport dropped".to_string(),
                    data: None,
                }));
            }
        }
    }
}

/// Spawn the background task that reads text frames from the websocket and
/// routes them: responses go to their waiting caller by id, notifications and
/// server-initiated requests are logged and otherwise ignored.
fn spawn_reader(mut source: WsSource, inner: std::sync::Arc<WsInner>) -> JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            match source.next().await {
                Some(Ok(Message::Text(text))) => {
                    let trimmed = text.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<Value>(trimmed) {
                        Ok(value) => route_incoming(&inner, value),
                        Err(e) => {
                            tracing::warn!(error = %e, text = %trimmed, "MCP: non-JSON text frame from websocket server");
                        }
                    }
                }
                Some(Ok(Message::Close(_))) => {
                    tracing::debug!("MCP websocket server closed the connection");
                    break;
                }
                // Pings/pongs are answered by tokio-tungstenite itself; binary
                // and raw frames are not part of MCP's websocket framing.
                Some(Ok(
                    Message::Ping(_) | Message::Pong(_) | Message::Binary(_) | Message::Frame(_),
                )) => {}
                Some(Err(e)) => {
                    tracing::warn!(error = %e, "MCP: error reading from websocket");
                    break;
                }
                None => {
                    tracing::debug!("MCP websocket stream ended");
                    break;
                }
            }
        }
        // The server is gone; unblock anyone still waiting on a response.
        let mut pending = inner.pending.lock().unwrap();
        for (_, tx) in pending.drain() {
            let _ = tx.send(Err(RpcError {
                code: -1,
                message: "MCP websocket connection closed".to_string(),
                data: None,
            }));
        }
    })
}

/// Identical routing logic to the stdio transport's `route_incoming`: classify
/// the message and either resolve a pending response or log-and-ignore.
fn route_incoming(inner: &std::sync::Arc<WsInner>, value: Value) {
    match protocol::parse_incoming(value) {
        IncomingMessage::Response { id, result } => {
            if let Some(tx) = inner.pending.lock().unwrap().remove(&id) {
                let _ = tx.send(result);
            } else {
                tracing::debug!(id, "MCP: response for unknown/already-resolved request id");
            }
        }
        IncomingMessage::Notification { method, params } => {
            tracing::debug!(method = %method, params = %params, "MCP server notification");
        }
        IncomingMessage::ServerRequest { id, method, params } => {
            tracing::warn!(
                id = %id,
                method = %method,
                params = %params,
                "MCP server sent a server-initiated request; sampling/roots are not supported, ignoring"
            );
        }
        IncomingMessage::Malformed(v) => {
            tracing::warn!(raw = %v, "MCP: unrecognized JSON-RPC message shape");
        }
    }
}
