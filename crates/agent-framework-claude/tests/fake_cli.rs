//! End-to-end tests against a scripted stand-in for the `claude` CLI that
//! speaks the stream-json control protocol. Unix only (the fake is a
//! `python3` script).

#![cfg(unix)]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use agent_framework_claude::{
    format_prompt, ClaudeAgent, ClaudeAgentOptions, FunctionApprovalHandler, HookCallback,
    HookMatcher, PermissionResult, ToolPermissionHandler, ToolPermissionRequest,
    STRUCTURED_OUTPUT_KEY,
};
use agent_framework_core::agent::{AgentRunOptions, SupportsAgentRun};
use agent_framework_core::error::Result;
use agent_framework_core::tools::{ApprovalMode, FunctionTool};
use agent_framework_core::types::{ChatOptions, Content, FunctionCallContent, Message, Role};
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

const FAKE: &str = r#"#!/usr/bin/env python3
import json, os, sys

log = open(os.environ["FAKE_LOG"], "a")
def record(kind, value):
    log.write(json.dumps({"kind": kind, "value": value}) + "\n"); log.flush()
record("argv", sys.argv[1:])
record("env", {"entrypoint": os.environ.get("CLAUDE_CODE_ENTRYPOINT"), "claudecode": os.environ.get("CLAUDECODE")})
scenario = os.environ.get("FAKE_SCENARIO", "text")

def out(msg):
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()

pending = []
def read():
    if pending:
        return pending.pop(0)
    line = sys.stdin.readline()
    return json.loads(line) if line else None

hook_ids = []
counter = [0]
def ask(request):
    counter[0] += 1
    rid = "cli_%d" % counter[0]
    out({"type": "control_request", "request_id": rid, "request": request})
    while True:
        msg = sys.stdin.readline()
        if not msg:
            sys.exit(5)
        msg = json.loads(msg)
        if msg.get("type") == "control_response" and msg["response"].get("request_id") == rid:
            record("answer", msg["response"])
            return msg["response"]
        pending.append(msg)

def delta(kind, key, text):
    out({"type": "stream_event", "uuid": "u", "session_id": "sess-1",
         "event": {"type": "content_block_delta", "index": 0, "delta": {"type": kind, key: text}}})

def result(**extra):
    msg = {"type": "result", "subtype": "success", "duration_ms": 5, "duration_api_ms": 3,
           "is_error": False, "num_turns": 1, "session_id": "sess-1", "stop_reason": "end_turn",
           "total_cost_usd": 0.25,
           "usage": {"input_tokens": 10, "output_tokens": 4, "cache_read_input_tokens": 2}}
    msg.update(extra)
    out(msg)

while True:
    msg = read()
    if msg is None:
        break
    if msg.get("type") == "control_request":
        req = msg["request"]
        record("control", req)
        if req["subtype"] == "initialize":
            for matchers in (req.get("hooks") or {}).values():
                for m in matchers:
                    hook_ids.extend(m["hookCallbackIds"])
        out({"type": "control_response", "response": {"subtype": "success", "request_id": msg["request_id"], "response": {}}})
        continue
    if msg.get("type") != "user":
        continue
    record("user", msg)
    out({"type": "system", "subtype": "init", "session_id": "sess-1"})
    if scenario == "hang":
        record("pid", os.getpid())
        import time
        while True:
            time.sleep(1)
    if scenario == "crash":
        sys.stderr.write("fatal: no credentials\n"); sys.stderr.flush()
        sys.exit(3)
    if scenario == "api_error":
        out({"type": "assistant", "error": "rate_limit", "message": {"model": "m", "content": [{"type": "text", "text": "slow down"}]}})
        continue
    if scenario == "result_error":
        result(is_error=True, result="boom")
        continue
    text = "Hello"
    if scenario == "tool":
        mcp = lambda m: ask({"subtype": "mcp_message", "server_name": "_agent_framework_tools", "message": m})
        mcp({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}})
        listed = mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        names = [t["name"] for t in listed["response"]["mcp_response"]["result"]["tools"]]
        texts = []
        for name in names:
            called = mcp({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": {"text": "hi"}}})
            texts.append(called["response"]["mcp_response"]["result"]["content"][0]["text"])
        text = " | ".join(texts)
    if scenario == "permission":
        answer = ask({"subtype": "can_use_tool", "tool_name": "Bash", "input": {"command": "ls"}, "tool_use_id": "t1"})
        text = answer["response"]["behavior"]
    if scenario == "hook":
        answer = ask({"subtype": "hook_callback", "callback_id": hook_ids[0], "input": {"tool_name": "Bash"}, "tool_use_id": "t1"})
        text = json.dumps(answer["response"])
    delta("thinking_delta", "thinking", "pondering")
    for part in [text[:2], text[2:]]:
        delta("text_delta", "text", part)
    out({"type": "assistant", "message": {"model": "m", "content": [{"type": "text", "text": text}]}})
    if scenario == "structured":
        result(structured_output={"key": "value"})
    else:
        result()
"#;

/// A fake CLI in a fresh directory, and the log it writes.
struct Fake {
    dir: PathBuf,
}

impl Fake {
    fn new() -> Self {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("af-claude-{}", uuid_like()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = dir.join("claude");
        std::fs::write(&script, FAKE).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        Self { dir }
    }

    fn options(&self, scenario: &str) -> ClaudeAgentOptions {
        let mut env = HashMap::new();
        env.insert("FAKE_SCENARIO".to_string(), scenario.to_string());
        env.insert(
            "FAKE_LOG".to_string(),
            self.dir.join("log.jsonl").display().to_string(),
        );
        ClaudeAgentOptions {
            cli_path: Some(self.dir.join("claude")),
            env,
            ..Default::default()
        }
    }

    fn log(&self) -> Vec<(String, Value)> {
        std::fs::read_to_string(self.dir.join("log.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| {
                let v: Value = serde_json::from_str(l).unwrap();
                (v["kind"].as_str().unwrap().to_string(), v["value"].clone())
            })
            .collect()
    }

    fn argv(&self, run: usize) -> Vec<String> {
        let argv = self
            .log()
            .into_iter()
            .filter(|(k, _)| k == "argv")
            .nth(run)
            .unwrap()
            .1;
        serde_json::from_value(argv).unwrap()
    }

    fn entries(&self, kind: &str) -> Vec<Value> {
        self.log()
            .into_iter()
            .filter(|(k, _)| k == kind)
            .map(|(_, v)| v)
            .collect()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn uuid_like() -> String {
    static N: AtomicUsize = AtomicUsize::new(0);
    format!(
        "{}-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::SeqCst),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    )
}

#[tokio::test]
async fn a_run_streams_text_and_reports_usage_and_the_session() {
    let fake = Fake::new();
    let agent = ClaudeAgent::builder()
        .name("claude")
        .instructions("Be brief.")
        .builtin_tool("Read")
        .builtin_tool("Glob")
        .options(ClaudeAgentOptions {
            model: Some("sonnet".into()),
            max_turns: Some(3),
            permission_mode: Some("plan".into()),
            ..fake.options("text")
        })
        .build()
        .unwrap();
    let mut session = agent.create_session();
    let response = agent
        .run(vec![Message::user("hi")], Some(&mut session))
        .await
        .unwrap();
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.messages[0].author_name.as_deref(), Some("claude"));
    assert!(response.messages[0]
        .contents
        .iter()
        .any(|c| matches!(c, Content::TextReasoning(r) if r.text == "pondering")));
    let usage = response.usage_details.clone().unwrap();
    assert_eq!(
        (
            usage.input_token_count,
            usage.output_token_count,
            usage.total_token_count,
            usage.cache_read_input_token_count
        ),
        (Some(10), Some(4), Some(14), Some(2))
    );
    assert_eq!(response.finish_reason.as_ref().unwrap().0, "stop");
    assert_eq!(response.conversation_id.as_deref(), Some("sess-1"));
    assert_eq!(session.service_session_id(), Some("sess-1"));

    let argv = fake.argv(0);
    let pair = |flag: &str| {
        argv.iter()
            .position(|a| a == flag)
            .map(|i| argv[i + 1].clone())
    };
    assert_eq!(&argv[..3], ["--output-format", "stream-json", "--verbose"]);
    assert_eq!(pair("--system-prompt").as_deref(), Some("Be brief."));
    assert_eq!(pair("--tools").as_deref(), Some("Read,Glob"));
    assert_eq!(pair("--model").as_deref(), Some("sonnet"));
    assert_eq!(pair("--max-turns").as_deref(), Some("3"));
    assert_eq!(pair("--permission-mode").as_deref(), Some("plan"));
    assert!(argv.contains(&"--include-partial-messages".to_string()));
    assert_eq!(&argv[argv.len() - 2..], ["--input-format", "stream-json"]);
    assert!(!argv.iter().any(|a| a.starts_with("--resume")));
    let env = &fake.entries("env")[0];
    assert_eq!(env["entrypoint"], "sdk-rs");
    assert!(env["claudecode"].is_null());
    let init = &fake.entries("control")[0];
    assert_eq!(init["subtype"], "initialize");
    let user = &fake.entries("user")[0];
    assert_eq!(user["message"], json!({"role": "user", "content": "hi"}));

    // The second run resumes the CLI session.
    agent
        .run(vec![Message::user("again")], Some(&mut session))
        .await
        .unwrap();
    assert!(fake.argv(1).contains(&"--resume=sess-1".to_string()));
}

#[tokio::test]
async fn run_stream_yields_deltas_and_a_final_update() {
    let fake = Fake::new();
    let agent = ClaudeAgent::new(fake.options("structured")).unwrap();
    let updates: Vec<_> = agent
        .run_stream(vec![Message::user("hi")], None, None)
        .await
        .unwrap()
        .collect()
        .await;
    let updates: Vec<_> = updates.into_iter().map(Result::unwrap).collect();
    let texts: Vec<String> = updates.iter().map(|u| u.text()).collect();
    assert_eq!(texts, ["", "He", "llo", ""], "thinking, two deltas, final");
    let last = updates.last().unwrap();
    assert_eq!(last.conversation_id.as_deref(), Some("sess-1"));
    assert_eq!(
        last.additional_properties[STRUCTURED_OUTPUT_KEY],
        json!({"key": "value"})
    );
    assert_eq!(last.additional_properties["total_cost_usd"], json!(0.25));
    // A non-streaming run surfaces it as the response value.
    let response = agent.run(vec![Message::user("hi")], None).await.unwrap();
    assert_eq!(response.value, Some(json!({"key": "value"})));
}

#[tokio::test]
async fn function_tools_are_served_over_the_in_process_mcp_server() {
    let fake = Fake::new();
    let approvals = Arc::new(AtomicUsize::new(0));
    let seen = approvals.clone();
    let approve: Arc<dyn FunctionApprovalHandler> = Arc::new(move |call: FunctionCallContent| {
        let seen = seen.clone();
        Box::pin(async move {
            seen.fetch_add(1, Ordering::SeqCst);
            call.call_id == "af-claude-approval::guarded"
        }) as futures::future::BoxFuture<'static, bool>
    });
    let echo = FunctionTool::new(
        "echo",
        "Echo the text",
        json!({"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}),
        |args: Value| async move { Ok(json!(format!("echo:{}", args["text"].as_str().unwrap()))) },
    )
    .into_definition();
    let guarded = FunctionTool::new("guarded", "Needs approval", json!({}), |_| async {
        Ok(json!("ran"))
    })
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();
    let agent = ClaudeAgent::builder()
        .tool(echo)
        .tool(guarded)
        .options(ClaudeAgentOptions {
            on_function_approval: Some(approve),
            allowed_tools: vec!["Read".into()],
            ..fake.options("tool")
        })
        .build()
        .unwrap();
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "echo:hi | ran");
    assert_eq!(approvals.load(Ordering::SeqCst), 1);

    let argv = fake.argv(0);
    let pair = |flag: &str| argv[argv.iter().position(|a| a == flag).unwrap() + 1].clone();
    assert_eq!(
        pair("--allowedTools"),
        "Read,mcp___agent_framework_tools__echo,mcp___agent_framework_tools__guarded"
    );
    let config: Value = serde_json::from_str(&pair("--mcp-config")).unwrap();
    assert_eq!(
        config["mcpServers"]["_agent_framework_tools"],
        json!({"type": "sdk", "name": "_agent_framework_tools"})
    );
    let answers = fake.entries("answer");
    let init = &answers[0]["response"]["mcp_response"]["result"];
    assert_eq!(init["protocolVersion"], "2025-06-18");
    let tools = &answers[1]["response"]["mcp_response"]["result"]["tools"];
    assert_eq!(tools[0]["inputSchema"]["required"], json!(["text"]));
    assert_eq!(
        tools[1]["inputSchema"],
        json!({"type": "object", "properties": {}})
    );
}

#[tokio::test]
async fn tools_requiring_approval_are_denied_without_a_handler() {
    let fake = Fake::new();
    let guarded = FunctionTool::new("guarded", "Needs approval", json!({}), |_| async {
        Ok(json!("ran"))
    })
    .with_approval_mode(ApprovalMode::AlwaysRequire)
    .into_definition();
    let agent = ClaudeAgent::builder()
        .tool(guarded)
        .options(fake.options("tool"))
        .build()
        .unwrap();
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert!(
        response.text().contains("no on_function_approval callback"),
        "{}",
        response.text()
    );
}

struct DenyBash;

#[async_trait]
impl ToolPermissionHandler for DenyBash {
    async fn can_use_tool(&self, request: ToolPermissionRequest) -> Result<PermissionResult> {
        assert_eq!(request.tool_name, "Bash");
        assert_eq!(request.input, json!({"command": "ls"}));
        assert_eq!(request.tool_use_id.as_deref(), Some("t1"));
        Ok(PermissionResult::deny("not here"))
    }
}

#[tokio::test]
async fn permission_prompts_reach_the_handler() {
    let fake = Fake::new();
    let agent = ClaudeAgent::new(ClaudeAgentOptions {
        can_use_tool: Some(Arc::new(DenyBash)),
        ..fake.options("permission")
    })
    .unwrap();
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    assert_eq!(response.text(), "deny");
    let argv = fake.argv(0);
    let at = argv
        .iter()
        .position(|a| a == "--permission-prompt-tool")
        .unwrap();
    assert_eq!(argv[at + 1], "stdio");
    let answer = &fake.entries("answer")[0];
    assert_eq!(
        answer["response"],
        json!({"behavior": "deny", "message": "not here"})
    );
}

#[tokio::test]
async fn hooks_are_registered_and_called_back() {
    let fake = Fake::new();
    let hook: HookCallback = Arc::new(|input: Value, tool_use_id: Option<String>| {
        Box::pin(async move {
            Ok(json!({"continue": true, "tool": input["tool_name"], "id": tool_use_id}))
        })
    });
    let mut hooks = HashMap::new();
    hooks.insert(
        "PreToolUse".to_string(),
        vec![HookMatcher {
            matcher: Some("Bash".into()),
            hooks: vec![hook],
            timeout: Some(5.0),
        }],
    );
    let agent = ClaudeAgent::new(ClaudeAgentOptions {
        hooks,
        ..fake.options("hook")
    })
    .unwrap();
    let response = agent.run(vec![Message::user("go")], None).await.unwrap();
    let echoed: Value = serde_json::from_str(&response.text()).unwrap();
    assert_eq!(
        echoed,
        json!({"continue": true, "tool": "Bash", "id": "t1"})
    );
    let init = &fake.entries("control")[0];
    assert_eq!(
        init["hooks"],
        json!({"PreToolUse": [{"matcher": "Bash", "hookCallbackIds": ["hook_0"], "timeout": 5.0}]})
    );
}

#[tokio::test]
async fn per_run_model_and_permission_mode_are_applied() {
    let fake = Fake::new();
    let agent = ClaudeAgent::new(fake.options("text")).unwrap();
    let mut chat = ChatOptions::new();
    chat.model = Some("opus".into());
    chat.additional_properties
        .insert("permission_mode".into(), json!("acceptEdits"));
    agent
        .run_with_options(
            vec![Message::user("hi")],
            None,
            AgentRunOptions::new().with_chat_options(chat),
        )
        .await
        .unwrap();
    let controls = fake.entries("control");
    assert_eq!(
        controls[1],
        json!({"subtype": "set_model", "model": "opus"})
    );
    assert_eq!(
        controls[2],
        json!({"subtype": "set_permission_mode", "mode": "acceptEdits"})
    );
}

#[tokio::test]
async fn errors_fail_the_run() {
    let fake = Fake::new();
    let cases = [
        ("api_error", "Rate limit exceeded for Claude API: slow down"),
        ("result_error", "Claude API error: boom"),
        ("crash", "fatal: no credentials"),
    ];
    for (scenario, expected) in cases {
        let agent = ClaudeAgent::new(fake.options(scenario)).unwrap();
        let err = agent
            .run(vec![Message::user("hi")], None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains(expected), "{scenario}: {err}");
    }
    let missing = ClaudeAgent::new(ClaudeAgentOptions {
        cli_path: Some(fake.dir.join("missing")),
        ..Default::default()
    })
    .unwrap();
    assert!(missing.run(vec![Message::user("hi")], None).await.is_err());
}

#[test]
fn prompts_follow_upstream_formatting() {
    assert_eq!(format_prompt(&[]), "");
    assert_eq!(format_prompt(&[Message::user("hello")]), "hello");
    let messages = vec![
        Message::user("Hi"),
        Message::assistant("Hello!"),
        Message::user("café \"x\""),
    ];
    assert_eq!(
        format_prompt(&messages),
        "The following messages were supplied to this agent in conversation order.\n\
         Each JSON record contains the original speaker's role and message content.\n\
         Use these messages as context. If the final message is a user request, respond to it \
         while following your instructions.\n\
         {\"role\": \"user\", \"content\": \"Hi\"}\n\
         {\"role\": \"assistant\", \"content\": \"Hello!\"}\n\
         {\"role\": \"user\", \"content\": \"caf\\u00e9 \\\"x\\\"\"}"
    );
    // A single assistant message is still framed.
    assert!(format_prompt(&[Message::new(Role::assistant(), "x")]).starts_with("The following"));
}

#[test]
fn hosted_mcp_tools_are_refused() {
    let mcp = agent_framework_core::tools::hosted_mcp("docs", "https://example.com/mcp", None);
    let err = ClaudeAgent::builder().tool(mcp).build().unwrap_err();
    assert!(err.to_string().contains("mcp_servers"), "{err}");
}

#[test]
fn settings_come_from_the_environment_unless_set() {
    let agent = ClaudeAgent::new(ClaudeAgentOptions {
        model: Some("haiku".into()),
        ..Default::default()
    })
    .unwrap();
    assert_eq!(agent.options().model.as_deref(), Some("haiku"));
    let args = agent.cli_args(Some("abc"));
    assert!(args.contains(&"--resume=abc".to_string()));
    assert_eq!(
        args[args.iter().position(|a| a == "--system-prompt").unwrap() + 1],
        ""
    );
}

/// A real CLI round trip, when `AF_CLAUDE_CLI_TEST` is set (and the CLI is
/// installed and authenticated).
#[tokio::test]
async fn live_cli_round_trip() {
    if std::env::var("AF_CLAUDE_CLI_TEST").is_err() {
        return;
    }
    let agent = ClaudeAgent::new(ClaudeAgentOptions {
        max_turns: Some(1),
        ..Default::default()
    })
    .unwrap();
    let response = agent
        .run(vec![Message::user("Reply with the word pong.")], None)
        .await
        .unwrap();
    assert!(response.text().to_lowercase().contains("pong"));
    assert!(response.conversation_id.is_some());
}

#[tokio::test]
async fn dropping_the_stream_kills_a_silent_cli() {
    let fake = Fake::new();
    let agent = ClaudeAgent::new(fake.options("hang")).unwrap();
    let stream = agent
        .run_stream(vec![Message::user("hi")], None, None)
        .await
        .unwrap();
    let mut pid = None;
    for _ in 0..200 {
        if let Some(p) = fake.entries("pid").first() {
            pid = p.as_u64();
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    let pid = pid.expect("the fake CLI started").to_string();
    let alive = || {
        std::process::Command::new("kill")
            .args(["-0", &pid])
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    };
    assert!(alive());
    drop(stream);
    let mut gone = false;
    for _ in 0..200 {
        if !alive() {
            gone = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    assert!(gone, "the CLI is killed when its run is dropped");
}
