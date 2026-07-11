//! Hermetic end-to-end test of the websocket transport against an in-process
//! fake MCP server built directly on `tokio-tungstenite`'s server side (no
//! external process, no network). Mirrors `stdio_e2e.rs`'s script of
//! initialize/tools/list/tools/call, and additionally checks that the client
//! negotiates the `"mcp"` subprotocol and that `close()` performs a real
//! WebSocket close handshake.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::error::Error;
use agent_framework_core::tools::ToolSource;
use agent_framework_mcp::{
    CreateMessageParams, CreateMessageResult, McpClient, McpTransport as _, McpWebsocketTool,
    McpWebsocketTransport, SamplingHandler,
};
use futures::stream::{SplitSink, SplitStream};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex as AsyncMutex;
use tokio_tungstenite::tungstenite::handshake::server::{Request, Response};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::WebSocketStream;

type WsSink = SplitSink<WebSocketStream<TcpStream>, Message>;
type WsSource = SplitStream<WebSocketStream<TcpStream>>;

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
                    "capabilities": {"tools": {}, "prompts": {}},
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
                    {
                        "name": "ask_llm",
                        "description": "Ask the client's LLM (via MCP sampling) a question.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"question": {"type": "string"}},
                            "required": ["question"],
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
        "prompts/list" => vec![json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "prompts": [
                    {
                        "name": "greet",
                        "description": "A friendly greeting prompt.",
                        "arguments": [
                            {"name": "name", "description": "Who to greet", "required": true},
                        ],
                    },
                ],
            },
        })],
        "prompts/get" => {
            let params = value.get("params").cloned().unwrap_or_default();
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or_default();
            if name == "greet" {
                let who = args.get("name").and_then(Value::as_str).unwrap_or("there");
                vec![json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "description": "A friendly greeting prompt.",
                        "messages": [
                            {"role": "user", "content": {"type": "text", "text": format!("Say hello to {who}")}},
                        ],
                    },
                })]
            } else {
                vec![json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": {"code": -32602, "message": format!("unknown prompt: {name}")},
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
                    let method = value.get("method").and_then(Value::as_str).unwrap_or("");
                    let params = value.get("params").cloned().unwrap_or_default();
                    let is_ask_llm = method == "tools/call"
                        && params.get("name").and_then(Value::as_str) == Some("ask_llm");
                    if is_ask_llm {
                        // Exercises server-request routing over the websocket
                        // transport: ask the client to sample a completion
                        // (a server-initiated request), then block reading
                        // the very next frame for the correlated response —
                        // safe because the test that uses this drives exactly
                        // one `tools/call` and awaits it, so nothing else is
                        // ever in flight in this window.
                        if !send_ask_llm_via_sampling(&mut sink, &mut source, &value).await {
                            return;
                        }
                        continue;
                    }
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

/// Handle one `tools/call("ask_llm", {"question": ...})` request by sending
/// the client a server-initiated `sampling/createMessage` request and
/// relaying its answer back as the tool result. Returns `false` if the
/// socket closed underneath us (caller should stop the server loop).
async fn send_ask_llm_via_sampling(
    sink: &mut WsSink,
    source: &mut WsSource,
    value: &Value,
) -> bool {
    let id = value.get("id").cloned();
    let params = value.get("params").cloned().unwrap_or_default();
    let args = params.get("arguments").cloned().unwrap_or_default();
    let question = args.get("question").and_then(Value::as_str).unwrap_or("");

    let sampling_request = json!({
        "jsonrpc": "2.0",
        "id": "srv-samp-1",
        "method": "sampling/createMessage",
        "params": {
            "messages": [{"role": "user", "content": {"type": "text", "text": question}}],
            "maxTokens": 50,
        },
    });
    if sink
        .send(Message::text(sampling_request.to_string()))
        .await
        .is_err()
    {
        return false;
    }

    let Some(Ok(Message::Text(reply_text))) = source.next().await else {
        return false;
    };
    let reply: Value = serde_json::from_str(&reply_text).unwrap_or_default();
    let answer_text = reply
        .get("result")
        .and_then(|r| r.get("content"))
        .and_then(|c| c.get("text"))
        .and_then(Value::as_str)
        .unwrap_or("");

    let result = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"content": [{"type": "text", "text": answer_text}], "isError": false},
    });
    sink.send(Message::text(result.to_string())).await.is_ok()
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

#[tokio::test]
async fn websocket_client_prompts_list_and_get() {
    let subprotocol_seen = Arc::new(AtomicBool::new(false));
    let (url, server) = spawn_fake_server(subprotocol_seen).await;

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let transport = McpWebsocketTransport::connect(&url, &[])
            .await
            .expect("connect to fake MCP websocket server");
        let client = McpClient::new(Arc::new(transport));
        client
            .initialize("test-client", "0.0.1")
            .await
            .expect("initialize handshake");

        assert!(client.supports_prompts().await);
        let prompts = client.list_prompts().await.expect("list_prompts");
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "greet");

        let result = client
            .get_prompt("greet", json!({"name": "Grace"}))
            .await
            .expect("get_prompt");
        assert_eq!(result.messages.len(), 1);
        assert_eq!(
            result.messages[0].content_block(),
            agent_framework_mcp::ContentBlock::Text("Say hello to Grace".to_string())
        );

        client.close().await.expect("close");
    })
    .await;

    outcome.expect("websocket_client_prompts_list_and_get timed out");
    let _ = server.await;
}

/// The websocket counterpart of `stdio_e2e.rs`'s headline sampling test:
/// the fake server answers `tools/call("ask_llm", ...)` by sending a
/// server-initiated `sampling/createMessage` request back over the same
/// socket, which the client must route to the registered [`SamplingHandler`]
/// and answer — proving server-request routing works over the websocket
/// transport too, not just stdio.
#[tokio::test]
async fn websocket_sampling_round_trip_via_server_initiated_request() {
    let subprotocol_seen = Arc::new(AtomicBool::new(false));
    let (url, server) = spawn_fake_server(subprotocol_seen).await;

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let received_question: Arc<AsyncMutex<Option<String>>> = Arc::new(AsyncMutex::new(None));
        let received_question_for_handler = received_question.clone();
        let handler: SamplingHandler = Arc::new(move |params: CreateMessageParams| {
            let received_question = received_question_for_handler.clone();
            Box::pin(async move {
                assert_eq!(params.max_tokens, 50);
                let question = params
                    .messages
                    .first()
                    .and_then(|m| m.content.get("text"))
                    .and_then(|t| t.as_str())
                    .unwrap_or_default()
                    .to_string();
                *received_question.lock().await = Some(question);
                Ok(CreateMessageResult::text("assistant", "42", "test-model"))
            })
        });

        let tool = McpWebsocketTool::new("fake-ws", url).sampling_handler(handler);
        tool.connect().await.expect("connect");

        let defs = tool.tool_definitions().await.expect("tool_definitions");
        let ask_llm = defs
            .iter()
            .find(|d| d.name == "ask_llm")
            .expect("ask_llm tool present");
        let executor = ask_llm.executor.clone().expect("executable tool");

        let value = executor
            .invoke(json!({"question": "What is 6 times 7?"}))
            .await
            .expect("invoke ask_llm (drives the sampling round trip)");
        assert_eq!(value, json!(42));
        assert_eq!(
            received_question.lock().await.clone(),
            Some("What is 6 times 7?".to_string())
        );

        tool.close().await.expect("close");
    })
    .await;

    outcome.expect("websocket_sampling_round_trip_via_server_initiated_request timed out");
    let _ = server.await;
}

/// [`McpWebsocketTool`] as a [`ToolSource`]: `resolve_tools` must lazily
/// connect (there's no prior `.connect()`/`.tool_definitions()` call here)
/// and return the server's tools, same as `tool_definitions()`. A second
/// `resolve_tools` call must also succeed — it's served from
/// `McpClient::list_tools_cached`'s cache; the cache-hit-vs-live-refetch
/// distinction itself, and `list_changed` invalidation, are covered
/// deterministically (no socket, no timing) by `client.rs`'s
/// `list_tools_cached_reuses_result_until_invalidated` test.
#[tokio::test]
async fn websocket_tool_source_resolve_tools_lazily_connects_and_returns_tools() {
    let subprotocol_seen = Arc::new(AtomicBool::new(false));
    let (url, server) = spawn_fake_server(subprotocol_seen).await;

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let tool = McpWebsocketTool::new("fake-ws", url);

        let resolved = ToolSource::resolve_tools(&tool)
            .await
            .expect("resolve_tools should lazily connect and list tools");
        let names: HashSet<&str> = resolved.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains("echo"), "expected an 'echo' tool: {names:?}");
        assert!(
            names.contains("ask_llm"),
            "expected an 'ask_llm' tool: {names:?}"
        );

        let resolved_again = ToolSource::resolve_tools(&tool)
            .await
            .expect("a second resolve_tools call should also succeed");
        assert_eq!(resolved_again.len(), resolved.len());

        tool.close().await.expect("close");
    })
    .await;

    outcome
        .expect("websocket_tool_source_resolve_tools_lazily_connects_and_returns_tools timed out");
    let _ = server.await;
}

/// [`McpWebsocketTransport::with_request_timeout`] must actually cut off a
/// call that never gets a response, not just be a stored, unused config
/// value. The fake server here accepts the connection and then never writes
/// anything back, so a request sent to it hangs forever without the timeout.
///
/// The client explicitly [`McpWebsocketTransport::close`]s at the end (as
/// opposed to just dropping the transport) so the fake server's loop below
/// observes a `Close` frame and exits on its own, rather than this test
/// leaking a task that blocks forever on a connection nobody will ever
/// close.
#[tokio::test]
async fn websocket_request_timeout_cuts_off_a_call_that_never_responds() {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback listener");
    let addr = listener.local_addr().expect("listener local addr");
    let url = format!("ws://{addr}/");

    let server = tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        #[allow(clippy::result_large_err)]
        let callback = |_req: &Request, response: Response| Ok(echo_mcp_subprotocol(response));
        let Ok(ws_stream) = tokio_tungstenite::accept_hdr_async(stream, callback).await else {
            return;
        };
        // Accept the connection and the handshake, but never respond to
        // anything sent over it -- until the client closes.
        let (mut sink, mut source) = ws_stream.split();
        while let Some(Ok(msg)) = source.next().await {
            if let Message::Close(frame) = msg {
                let _ = sink.send(Message::Close(frame)).await;
                break;
            }
        }
    });

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let transport = McpWebsocketTransport::connect(&url, &[])
            .await
            .expect("connect")
            .with_request_timeout(Duration::from_millis(150));

        let started = std::time::Instant::now();
        let err = transport
            .call("initialize", json!({}))
            .await
            .expect_err("a request to a server that never responds must time out");
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "timeout should fire close to the configured 150ms, not fall back to hanging"
        );
        assert!(
            err.to_string().contains("timed out"),
            "unexpected error message: {err}"
        );

        transport.close().await.expect("close");
    })
    .await;

    outcome.expect("websocket_request_timeout_cuts_off_a_call_that_never_responds timed out");
    let _ = server.await;
}
