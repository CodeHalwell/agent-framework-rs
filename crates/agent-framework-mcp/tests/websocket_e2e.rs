//! Hermetic end-to-end test of the websocket transport against an in-process
//! fake MCP server built directly on `tokio-tungstenite`'s server side (no
//! external process, no network). Mirrors `stdio_e2e.rs`'s script of
//! initialize/tools/list/tools/call, and additionally checks that the client
//! negotiates the `"mcp"` subprotocol and that `close()` performs a real
//! WebSocket close handshake.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::error::Error;
use agent_framework_mcp::{McpClient, McpTransport as _, McpWebsocketTool, McpWebsocketTransport};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;

/// Handle one JSON-RPC request-shaped value the same way the stdio fake
/// server's Python script does, returning the response (or notification+
/// response pair) to send back.
fn handle_message(value: &Value) -> Vec<Value> {
    let method = value.get("method").and_then(Value::as_str).unwrap_or("");
    let id = value.get("id").cloned();

    match method {
        "initialize" => vec![
            // A stray notification sent right before the response, proving
            // the reader routes it away from response correlation instead of
            // misinterpreting it as the reply (routing-skips-notifications).
            json!({
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "info", "data": "starting up"},
            }),
            json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-ws-mcp-server", "version": "0.0.1"},
                },
            }),
        ],
        "notifications/initialized" => vec![],
        "ping" => vec![json!({"jsonrpc": "2.0", "id": id, "result": {}})],
        "tools/list" => vec![json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo the input text back.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    },
                ],
            },
        })],
        "tools/call" => {
            let params = value.get("params").cloned().unwrap_or_default();
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_default();
            if name == "echo" {
                let text = args.get("text").and_then(Value::as_str).unwrap_or("");
                vec![json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {"content": [{"type": "text", "text": text}], "isError": false},
                })]
            } else {
                vec![json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{"type": "text", "text": format!("unknown tool: {name}")}],
                        "isError": true,
                    },
                })]
            }
        }
        _ if id.is_some() => vec![json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {"code": -32601, "message": format!("Method not found: {method}")},
        })],
        _ => vec![],
    }
}

/// Whether the client's upgrade request advertised the `"mcp"` subprotocol.
fn client_offered_mcp_subprotocol(req: &Request) -> bool {
    req.headers()
        .get("Sec-WebSocket-Protocol")
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(',').any(|p| p.trim() == "mcp"))
        .unwrap_or(false)
}

/// Echo `"mcp"` back as the negotiated subprotocol, as a compliant server
/// would. `tokio-tungstenite`'s client rejects the handshake if it asked for
/// a subprotocol and the server doesn't confirm one.
fn echo_mcp_subprotocol(mut response: Response) -> Response {
    response
        .headers_mut()
        .insert("Sec-WebSocket-Protocol", "mcp".parse().unwrap());
    response
}

/// Spawn a minimal MCP-over-websocket server on an ephemeral loopback port.
/// `subprotocol_seen` is set once the server observes the client's upgrade
/// request advertising the `"mcp"` `Sec-WebSocket-Protocol`.
async fn spawn_fake_server(
    subprotocol_seen: Arc<AtomicBool>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");
    let url = format!("ws://{addr}/");

    let handle = tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };

        // `Callback::on_request`'s `Err` type (`ErrorResponse`) is dictated by
        // tungstenite and can't be shrunk from here; this closure never
        // actually returns it.
        #[allow(clippy::result_large_err)]
        let callback = move |req: &Request, response: Response| {
            subprotocol_seen.store(client_offered_mcp_subprotocol(req), Ordering::SeqCst);
            Ok(echo_mcp_subprotocol(response))
        };

        let Ok(ws_stream) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
            return;
        };
        let (mut sink, mut source) = ws_stream.split();

        while let Some(msg) = source.next().await {
            let Ok(msg) = msg else { break };
            match msg {
                Message::Text(text) => {
                    let value: Value = match serde_json::from_str(&text) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    for reply in handle_message(&value) {
                        if sink.send(Message::text(reply.to_string())).await.is_err() {
                            return;
                        }
                    }
                }
                Message::Close(_) => {
                    let _ = sink.send(Message::Close(None)).await;
                    break;
                }
                _ => {}
            }
        }
    });

    (url, handle)
}

#[tokio::test]
async fn websocket_client_initialize_list_and_call() {
    let subprotocol_seen = Arc::new(AtomicBool::new(false));
    let (url, server) = spawn_fake_server(subprotocol_seen.clone()).await;

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let transport = McpWebsocketTransport::connect(&url, &[])
            .await
            .expect("connect to fake MCP websocket server");
        let client = McpClient::new(Arc::new(transport));

        let init = client
            .initialize("test-client", "0.0.1")
            .await
            .expect("initialize handshake");
        assert_eq!(init.server_info.name, "fake-ws-mcp-server");
        assert_eq!(init.protocol_version, "2025-06-18");
        assert!(client.is_initialized().await);

        client.ping().await.expect("ping");

        let tools = client.list_tools().await.expect("list_tools");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"echo"),
            "expected an 'echo' tool: {names:?}"
        );

        let echoed = client
            .call_tool_value("echo", json!({"text": "hello websocket"}))
            .await
            .expect("call echo");
        assert_eq!(echoed, json!("hello websocket"));

        let err = client
            .call_tool_value("does-not-exist", json!({}))
            .await
            .expect_err("unknown tool should report isError");
        match err {
            Error::Tool(msg) => assert!(msg.contains("unknown tool"), "unexpected message: {msg}"),
            other => panic!("expected Error::Tool, got {other:?}"),
        }

        client.close().await.expect("close");
    })
    .await;

    outcome.expect("websocket_client_initialize_list_and_call timed out");
    let _ = server.await;
    assert!(
        subprotocol_seen.load(Ordering::SeqCst),
        "client should advertise the 'mcp' Sec-WebSocket-Protocol on connect"
    );
}

#[tokio::test]
async fn websocket_tool_high_level_api_connects_and_filters_allowed_tools() {
    let subprotocol_seen = Arc::new(AtomicBool::new(false));
    let (url, server) = spawn_fake_server(subprotocol_seen).await;

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let tool = McpWebsocketTool::new("fake-ws", url)
            .description("A fake MCP websocket server for tests")
            .allowed_tools(["echo"]);

        // connect() must be idempotent and safe to call more than once.
        tool.connect().await.expect("connect");
        tool.connect().await.expect("connect again");

        let defs = tool.tool_definitions().await.expect("tool_definitions");
        assert_eq!(defs.len(), 1, "allowed_tools should pass through 'echo'");
        assert_eq!(defs[0].name, "echo");
        assert!(defs[0].is_executable());

        let executor = defs[0].executor.clone().expect("executable tool");
        let value = executor
            .invoke(json!({"text": "hi"}))
            .await
            .expect("invoke echo through ToolDefinition");
        assert_eq!(value, json!("hi"));

        tool.close().await.expect("close");
    })
    .await;

    outcome.expect("websocket_tool_high_level_api_connects_and_filters_allowed_tools timed out");
    let _ = server.await;
}

#[tokio::test]
async fn websocket_close_performs_close_handshake() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");
    let url = format!("ws://{addr}/");

    let (close_tx, close_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        #[allow(clippy::result_large_err)]
        let callback = |_req: &Request, response: Response| Ok(echo_mcp_subprotocol(response));
        let Ok(ws_stream) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
            return;
        };
        let (mut sink, mut source) = ws_stream.split();
        while let Some(msg) = source.next().await {
            if let Ok(Message::Close(frame)) = msg {
                let _ = sink.send(Message::Close(frame)).await;
                let _ = close_tx.send(());
                break;
            }
        }
    });

    let outcome = tokio::time::timeout(Duration::from_secs(10), async {
        let transport = McpWebsocketTransport::connect(&url, &[])
            .await
            .expect("connect");
        transport.close().await.expect("close() should succeed");

        close_rx
            .await
            .expect("server should observe a Close frame from the client");
    })
    .await;

    outcome.expect("websocket_close_performs_close_handshake timed out");
    let _ = server.await;
}
