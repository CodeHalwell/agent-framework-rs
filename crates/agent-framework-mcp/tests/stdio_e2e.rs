//! Hermetic end-to-end test of the stdio transport against a hand-rolled
//! fake MCP server written in plain python3 (stdlib only — the `mcp` pip
//! package is not assumed to be installed, and this test must not touch the
//! network).

use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::error::Error;
use agent_framework_mcp::{McpClient, McpStdioTool, McpStdioTransport};
use serde_json::json;

/// A minimal MCP server: handles `initialize`, `notifications/initialized`,
/// `ping`, `tools/list` (two tools: `echo`, `add`), and `tools/call`. Emits a
/// stray `notifications/message` right before the `initialize` response, so
/// the test proves the client's reader routes notifications away from
/// response correlation instead of misinterpreting one as the reply.
const FAKE_SERVER_PY: &str = r#"
import sys, json

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def main():
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            send({"jsonrpc": "2.0", "method": "notifications/message",
                  "params": {"level": "info", "data": "starting up"}})
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "serverInfo": {"name": "fake-mcp-server", "version": "0.0.1"},
                },
            })
        elif method == "notifications/initialized":
            pass
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        elif method == "tools/list":
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
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
                            "name": "add",
                            "description": "Add two numbers.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                                "required": ["a", "b"],
                            },
                        },
                    ],
                },
            })
        elif method == "tools/call":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "echo":
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": args.get("text", "")}], "isError": False},
                })
            elif name == "add":
                try:
                    total = args["a"] + args["b"]
                    send({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": json.dumps(total)}], "isError": False},
                    })
                except Exception as exc:
                    send({
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {"content": [{"type": "text", "text": str(exc)}], "isError": True},
                    })
            else:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": "unknown tool: " + str(name)}], "isError": True},
                })
        else:
            if msg_id is not None:
                send({"jsonrpc": "2.0", "id": msg_id,
                      "error": {"code": -32601, "message": "Method not found: " + str(method)}})

if __name__ == "__main__":
    main()
"#;

/// Write the fake server script to a fresh temp file and return its path.
fn write_fake_server() -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!("af_mcp_fake_server_{}.py", uuid::Uuid::new_v4()));
    let mut file = std::fs::File::create(&path).expect("create fake MCP server script");
    file.write_all(FAKE_SERVER_PY.as_bytes())
        .expect("write fake MCP server script");
    path
}

#[tokio::test]
async fn stdio_client_initialize_list_and_call() {
    let script = write_fake_server();

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let transport = McpStdioTransport::spawn(
            "python3",
            &[script.to_string_lossy().to_string()],
            None,
            None,
        )
        .await
        .expect("spawn fake MCP server");
        let client = McpClient::new(Arc::new(transport));

        let init = client
            .initialize("test-client", "0.0.1")
            .await
            .expect("initialize handshake");
        assert_eq!(init.server_info.name, "fake-mcp-server");
        assert_eq!(init.protocol_version, "2025-06-18");
        assert_eq!(client.server_info().await.unwrap().name, "fake-mcp-server");
        assert!(client.is_initialized().await);

        // Idempotent: a second call must not hang or re-send the handshake.
        client
            .initialize("test-client", "0.0.1")
            .await
            .expect("re-initialize");

        client.ping().await.expect("ping");

        let tools = client.list_tools().await.expect("list_tools");
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"echo"),
            "expected an 'echo' tool, got {names:?}"
        );
        assert!(
            names.contains(&"add"),
            "expected an 'add' tool, got {names:?}"
        );

        // A single non-JSON text block maps to a plain string.
        let echoed = client
            .call_tool_value("echo", json!({"text": "hello world"}))
            .await
            .expect("call echo");
        assert_eq!(echoed, json!("hello world"));

        // A single text block that happens to be valid JSON is parsed.
        let summed = client
            .call_tool_value("add", json!({"a": 40, "b": 2}))
            .await
            .expect("call add");
        assert_eq!(summed, json!(42));

        // isError: true maps to Err(Error::Tool(..)).
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

    let _ = std::fs::remove_file(&script);
    outcome.expect("stdio_client_initialize_list_and_call timed out");
}

#[tokio::test]
async fn stdio_tool_high_level_api_connects_and_filters_allowed_tools() {
    let script = write_fake_server();

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let tool = McpStdioTool::new("fake", "python3")
            .args([script.to_string_lossy().to_string()])
            .description("A fake MCP server for tests")
            .allowed_tools(["echo"]);

        // connect() must be idempotent and safe to call more than once.
        tool.connect().await.expect("connect");
        tool.connect().await.expect("connect again");

        let defs = tool.tool_definitions().await.expect("tool_definitions");
        assert_eq!(defs.len(), 1, "allowed_tools should filter out 'add'");
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

    let _ = std::fs::remove_file(&script);
    outcome.expect("stdio_tool_high_level_api_connects_and_filters_allowed_tools timed out");
}
