//! Hermetic end-to-end test of the stdio transport against a hand-rolled
//! fake MCP server written in plain python3 (stdlib only — the `mcp` pip
//! package is not assumed to be installed, and this test must not touch the
//! network).

use std::collections::HashSet;
use std::io::Write as _;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::error::Error;
use agent_framework_core::tools::ToolSource;
use agent_framework_mcp::{
    CreateMessageParams, CreateMessageResult, McpClient, McpStdioTool, McpStdioTransport,
    McpTransport as _, SamplingHandler,
};
use serde_json::json;
use tokio::sync::Mutex as AsyncMutex;

/// A minimal MCP server: handles `initialize`, `notifications/initialized`,
/// `ping`, `tools/list` (`echo`, `add`, `ask_llm`), `tools/call`,
/// `prompts/list`, and `prompts/get`. Emits a stray `notifications/message`
/// right before the `initialize` response, so the test proves the client's
/// reader routes notifications away from response correlation instead of
/// misinterpreting one as the reply.
///
/// `ask_llm` exercises server-initiated request routing over stdio: handling
/// it sends the client a server-initiated `sampling/createMessage` request,
/// then blocks reading the very next stdin line for the correlated
/// response. That ordering is safe (not a race) because the test issues
/// exactly one `tools/call` and awaits it before doing anything else — the
/// only two things the client ever writes to this process's stdin in that
/// window are the original `tools/call` request (already sent) and the
/// client's answer to this server-initiated request, in that order.
const FAKE_SERVER_PY: &str = r#"
import sys, json

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def read_message():
    line = sys.stdin.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return read_message()
    try:
        return json.loads(line)
    except Exception:
        return read_message()

def main():
    while True:
        msg = read_message()
        if msg is None:
            break
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
                    "capabilities": {"tools": {}, "prompts": {}},
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
            elif name == "ask_llm":
                question = args.get("question", "")
                send({
                    "jsonrpc": "2.0",
                    "id": "srv-samp-1",
                    "method": "sampling/createMessage",
                    "params": {
                        "messages": [{"role": "user", "content": {"type": "text", "text": question}}],
                        "maxTokens": 50,
                    },
                })
                reply = read_message() or {}
                result = reply.get("result") or {}
                content = result.get("content") or {}
                answer_text = content.get("text", "")
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": answer_text}], "isError": False},
                })
            else:
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": "unknown tool: " + str(name)}], "isError": True},
                })
        elif method == "prompts/list":
            send({
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "prompts": [
                        {
                            "name": "greet",
                            "description": "A friendly greeting prompt.",
                            "arguments": [
                                {"name": "name", "description": "Who to greet", "required": True},
                            ],
                        },
                    ],
                },
            })
        elif method == "prompts/get":
            params = msg.get("params") or {}
            name = params.get("name")
            args = params.get("arguments") or {}
            if name == "greet":
                who = args.get("name", "there")
                send({
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "description": "A friendly greeting prompt.",
                        "messages": [
                            {"role": "user", "content": {"type": "text", "text": "Say hello to " + who}},
                        ],
                    },
                })
            else:
                send({"jsonrpc": "2.0", "id": msg_id,
                      "error": {"code": -32602, "message": "unknown prompt: " + str(name)}})
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

#[tokio::test]
async fn stdio_tool_prompts_list_and_get() {
    let script = write_fake_server();

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let tool =
            McpStdioTool::new("fake", "python3").args([script.to_string_lossy().to_string()]);
        tool.connect().await.expect("connect");

        let prompts = tool.prompts().await.expect("prompts");
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts[0].name, "greet");
        assert_eq!(
            prompts[0].description.as_deref(),
            Some("A friendly greeting prompt.")
        );
        let args = prompts[0].arguments.as_ref().expect("greet has arguments");
        assert_eq!(args.len(), 1);
        assert_eq!(args[0].name, "name");
        assert_eq!(args[0].required, Some(true));

        let messages = tool
            .get_prompt("greet", json!({"name": "Ada"}))
            .await
            .expect("get_prompt");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].text(), "Say hello to Ada");

        tool.close().await.expect("close");
    })
    .await;

    let _ = std::fs::remove_file(&script);
    outcome.expect("stdio_tool_prompts_list_and_get timed out");
}

/// The headline test for server-request routing over stdio: the fake server
/// answers `tools/call("ask_llm", ...)` by sending a server-initiated
/// `sampling/createMessage` request back to the client, which must route it
/// to the registered [`SamplingHandler`], get an answer, and write the
/// JSON-RPC response back over stdin — all without the test ever touching
/// the transport directly. See the `FAKE_SERVER_PY` doc comment for why the
/// server's blocking read of "the next line" is safe here, not a race.
#[tokio::test]
async fn stdio_tool_sampling_round_trip_via_server_initiated_request() {
    let script = write_fake_server();

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

        let tool = McpStdioTool::new("fake", "python3")
            .args([script.to_string_lossy().to_string()])
            .sampling_handler(handler);
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
        // The server echoes the client's sampling answer back as the tool
        // result; "42" happens to parse as JSON, so `to_value()` yields the
        // number (matching the crate's established single-text-block rule).
        assert_eq!(value, json!(42));

        assert_eq!(
            received_question.lock().await.clone(),
            Some("What is 6 times 7?".to_string()),
            "the server's sampling request should have carried the tool's question through"
        );

        tool.close().await.expect("close");
    })
    .await;

    let _ = std::fs::remove_file(&script);
    outcome.expect("stdio_tool_sampling_round_trip_via_server_initiated_request timed out");
}

/// [`McpStdioTool`] as a [`ToolSource`]: `resolve_tools` must lazily connect
/// (there's no prior `.connect()`/`.tool_definitions()` call here) and
/// return the same tools `tool_definitions()` would, filtered by
/// `allowed_tools`. A second `resolve_tools` call must also succeed and
/// return the same tools — it's served from `McpClient::list_tools_cached`'s
/// cache; the cache-hit-vs-live-refetch distinction itself, and
/// `list_changed` invalidation, are covered deterministically (no process
/// spawn, no timing) by `client.rs`'s
/// `list_tools_cached_reuses_result_until_invalidated` test.
#[tokio::test]
async fn stdio_tool_source_resolve_tools_lazily_connects_and_returns_tools() {
    let script = write_fake_server();

    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let tool = McpStdioTool::new("fake", "python3")
            .args([script.to_string_lossy().to_string()])
            .allowed_tools(["echo", "add"]);

        let resolved = ToolSource::resolve_tools(&tool)
            .await
            .expect("resolve_tools should lazily connect and list tools");
        let names: HashSet<&str> = resolved.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(
            names,
            HashSet::from(["echo", "add"]),
            "allowed_tools filter should still apply via the ToolSource path"
        );

        let resolved_again = ToolSource::resolve_tools(&tool)
            .await
            .expect("a second resolve_tools call should also succeed");
        assert_eq!(resolved_again.len(), 2);

        tool.close().await.expect("close");
    })
    .await;

    let _ = std::fs::remove_file(&script);
    outcome.expect("stdio_tool_source_resolve_tools_lazily_connects_and_returns_tools timed out");
}

/// [`McpStdioTransport::with_request_timeout`] must actually cut off a call
/// that never gets a response, not just be a stored, unused config value.
/// `sleep` never reads its stdin or writes to stdout, so a request sent to
/// it as if it were an MCP server hangs forever without the timeout.
#[tokio::test]
async fn stdio_request_timeout_cuts_off_a_call_that_never_responds() {
    let outcome = tokio::time::timeout(Duration::from_secs(20), async {
        let transport = McpStdioTransport::spawn("sleep", &["100".to_string()], None, None)
            .await
            .expect("spawn sleep as a stand-in for a hung MCP server")
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

    outcome.expect("stdio_request_timeout_cuts_off_a_call_that_never_responds timed out");
}
