//! MCP over HTTP and WebSocket: `McpStreamableHttpTool` and
//! `McpWebsocketTool`, the two remote transports beside the stdio one every
//! other `mcp_*` example uses.
//!
//! Which transport to reach for:
//!
//! | Transport | Server lives | Use when |
//! | --- | --- | --- |
//! | `McpStdioTool` | a child process you spawn | local tools, dev loops |
//! | `McpStreamableHttpTool` | behind a URL | a shared/remote server, or one behind a gateway |
//! | `McpWebsocketTool` | behind a `ws://`/`wss://` URL | the server needs a persistent duplex socket |
//!
//! All three produce the same `ToolDefinition`s and satisfy the same
//! `ToolSource` trait, so everything above them -- agents, approval modes,
//! prompts, sampling, roots -- is transport-agnostic. Swapping one for
//! another is a one-line change.
//!
//! What is transport-specific: `headers(..)` for auth (the usual reason to
//! pick HTTP over stdio -- an `Authorization` header a gateway can check),
//! `timeout(..)` per request, and the `Mcp-Session-Id` the streamable-HTTP
//! transport captures from the first response and echoes on every later one.
//!
//! Offline and self-terminating: this example serves a **real MCP server**
//! in-process on an ephemeral port (about 60 lines of axum below, speaking
//! JSON-RPC), connects the actual `McpStreamableHttpTool` to it over a real
//! socket, and runs an agent against its tools. No `npx`, no network.
//!
//! ```bash
//! cargo run -p agent-framework-examples --example mcp_http_transport
//! ```

use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use agent_framework::mcp::{McpStreamableHttpTool, McpWebsocketTool};
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

/// Counts requests, so the example can show the transport reusing one
/// session rather than re-handshaking per call.
#[derive(Clone, Default)]
struct ServerState {
    requests: Arc<AtomicUsize>,
}

/// A minimal but genuine MCP server: JSON-RPC 2.0 over a single POST
/// endpoint, handling `initialize`, `tools/list`, and `tools/call`.
///
/// A real one would be a separate service; the point here is that the client
/// side below is the framework's actual HTTP transport, talking to it over a
/// real socket, not a mock.
async fn mcp_endpoint(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let n = state.requests.fetch_add(1, Ordering::SeqCst) + 1;
    let method = request["method"].as_str().unwrap_or_default().to_string();
    let session = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("(none yet)");
    println!("    [server] #{n} {method:<28} session={session}");

    // A notification (no `id`) expects no JSON-RPC response body, only a 2xx.
    let Some(id) = request.get("id").cloned() else {
        return ([("mcp-session-id", "sess-demo-1")], Json(json!({}))).into_response();
    };

    let result = match method.as_str() {
        "initialize" => json!({
            "protocolVersion": "2025-06-18",
            "capabilities": { "tools": { "listChanged": false } },
            "serverInfo": { "name": "in-process-demo", "version": "1.0.0" },
        }),
        "tools/list" => json!({
            "tools": [
                {
                    "name": "weather",
                    "description": "Current conditions for a city.",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "city": { "type": "string" } },
                        "required": ["city"]
                    }
                },
                {
                    "name": "tide_times",
                    "description": "Today's high and low tides for a coastal town.",
                    "inputSchema": {
                        "type": "object",
                        "properties": { "town": { "type": "string" } },
                        "required": ["town"]
                    }
                }
            ]
        }),
        "tools/call" => {
            let name = request["params"]["name"].as_str().unwrap_or_default();
            let args = &request["params"]["arguments"];
            let text = match name {
                "weather" => format!(
                    "{}: 14 degrees, overcast, light drizzle.",
                    args["city"].as_str().unwrap_or("unknown")
                ),
                "tide_times" => format!(
                    "{}: high 06:12 and 18:40, low 12:25.",
                    args["town"].as_str().unwrap_or("unknown")
                ),
                other => format!("no such tool: {other}"),
            };
            json!({ "content": [{ "type": "text", "text": text }], "isError": false })
        }
        "ping" => json!({}),
        other => {
            return Json(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("method not found: {other}") }
            }))
            .into_response()
        }
    };

    // Handing back an `Mcp-Session-Id` is what makes the transport echo it on
    // every subsequent request -- how a stateful server correlates a session.
    (
        [("mcp-session-id", "sess-demo-1")],
        Json(json!({ "jsonrpc": "2.0", "id": id, "result": result })),
    )
        .into_response()
}

/// A canned model, so the example needs no API key. It calls `weather` once
/// and then answers.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatResponse> {
        if let Some(result) =
            messages
                .iter()
                .flat_map(|m| m.contents.iter())
                .find_map(|c| match c {
                    Content::FunctionResult(r) => r.result.clone(),
                    _ => None,
                })
        {
            let text = result
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| result.to_string());
            return Ok(ChatResponse::from_text(format!(
                "The MCP server says: {text}"
            )));
        }
        // Only call the tool if the MCP server actually advertised it.
        if options.tools.iter().any(|t| t.name == "weather") {
            return Ok(ChatResponse {
                messages: vec![Message::with_contents(
                    Role::assistant(),
                    vec![Content::FunctionCall(FunctionCallContent::new(
                        "call-1",
                        "weather",
                        Some(FunctionArguments::Raw(
                            json!({"city": "Manchester"}).to_string(),
                        )),
                    ))],
                )],
                finish_reason: Some(FinishReason::tool_calls()),
                ..Default::default()
            });
        }
        Ok(ChatResponse::from_text("I have no tools to work with."))
    }

    async fn get_streaming_response(
        &self,
        _m: Vec<Message>,
        _o: ChatOptions,
    ) -> Result<ChatStream> {
        Ok(Box::pin(futures::stream::empty()))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // --- serve the MCP server in-process -------------------------------
    let state = ServerState::default();
    let app = Router::new()
        .route("/mcp", post(mcp_endpoint))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| Error::Configuration(format!("bind failed: {e}")))?;
    let addr: SocketAddr = listener
        .local_addr()
        .map_err(|e| Error::Configuration(e.to_string()))?;
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let url = format!("http://{addr}/mcp");
    println!("MCP server listening at {url}\n");

    // --- the client side: the framework's real HTTP transport ----------
    println!("== connecting McpStreamableHttpTool ==\n");

    let mcp = McpStreamableHttpTool::new("demo", &url)
        .description("An in-process MCP server over streamable HTTP.")
        // The usual reason to prefer HTTP over stdio: the server is behind a
        // gateway that wants a credential. Headers go on every request.
        .headers([("Authorization", "Bearer demo-token")])
        // Bound each round trip, so an unresponsive server cannot wedge a run.
        .timeout(Duration::from_secs(10));

    // `tool_definitions()` performs the `initialize` handshake (if it has not
    // already) and lists the server's tools as ordinary `ToolDefinition`s.
    let tools = mcp.tool_definitions().await?;
    println!("\n  discovered {} tool(s):", tools.len());
    for tool in &tools {
        println!("    - {}: {}", tool.name, tool.description);
    }

    println!("\n== restricting what the model sees ==\n");

    // `allowed_tools` filters the list without the server needing to change:
    // hand the model the two calls this agent should make, not the server's
    // whole surface.
    let restricted = McpStreamableHttpTool::new("demo", &url).allowed_tools(["weather"]);
    let names: Vec<String> = restricted
        .tool_definitions()
        .await?
        .into_iter()
        .map(|t| t.name)
        .collect();
    println!("\n  with allowed_tools([\"weather\"]): {names:?}");

    println!("\n== running an agent against them ==\n");

    let agent = Agent::builder(CannedClient)
        .name("assistant")
        .instructions("Use the MCP tools when they help.")
        .tools(tools)
        .build();
    let response = agent.run_once("What's the weather in Manchester?").await?;
    println!("\n  {}", response.text());

    // The `session=` column in the trace above shows the session id the
    // server handed back on the first response being echoed on every request
    // since -- the transport captured it with no help from this code.
    // (`McpStreamableHttpTransport::session_id()` exposes it, if you build the
    // transport yourself rather than going through the tool.)
    println!(
        "  total server requests: {}",
        state.requests.load(Ordering::SeqCst)
    );

    // `close()` best-effort DELETEs the session. A stateful server frees its
    // resources here; a stateless one ignores it.
    mcp.close().await?;

    println!("\n== the WebSocket transport ==\n");
    println!(
        "  Identical API, different scheme -- the only change is the constructor:\n\n    \
         let mcp = McpWebsocketTool::new(\"demo\", \"wss://mcp.example.com/ws\")\n        \
         .headers([(\"Authorization\", \"Bearer …\")])\n        \
         .allowed_tools([\"weather\"]);\n    \
         let tools = mcp.tool_definitions().await?;\n\n  \
         Prefer it when the server needs a persistent duplex socket -- \n  \
         server-initiated sampling and `list_changed` notifications arrive\n  \
         without the client holding a request open. Note the streamable-HTTP\n  \
         transport in this workspace only sees a server notification embedded\n  \
         in the response to an active call; standalone GET-based SSE listening\n  \
         is not implemented, so WebSocket is the better fit for a chatty server."
    );

    // Constructed but not connected -- there is no ws:// server here to dial.
    let ws = McpWebsocketTool::new("demo-ws", "wss://mcp.example.com/ws")
        .description("Same tools, over a duplex socket.")
        .allowed_tools(["weather"]);
    println!("\n  built (not connected): {}", ws.name());

    println!(
        "\nsee also: `mcp_tools` (stdio), `mcp_first_class_tools` (ToolSource\n\
         re-resolution), `mcp_prompts`, `mcp_sampling`, `mcp_roots` -- every one\n\
         of those works over this transport too, unchanged."
    );

    Ok(())
}
