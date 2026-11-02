//! The harness file-access provider (`experimental-harness`).
//!
//! Ported from the `FileAccessProvider` tests in upstream
//! `test_harness_file_access.py`.
#![cfg(feature = "experimental-harness")]

use std::sync::Arc;

use agent_framework_core::harness::*;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::prelude::*;
use agent_framework_core::tools::ApprovalMode;
use agent_framework_core::types::FunctionCallContent;
use serde_json::{json, Value};

async fn prepare(provider: &FileAccessProvider, session_id: Option<&str>) -> SessionContext {
    let mut ctx = SessionContext::new(vec![Message::user("hi")]);
    ctx.session_id = session_id.map(str::to_string);
    provider.before_run(&mut ctx).await.unwrap();
    ctx
}

async fn call(ctx: &SessionContext, name: &str, args: Value) -> Value {
    ctx.tools
        .iter()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("no tool {name}"))
        .executor
        .as_ref()
        .unwrap()
        .invoke(args)
        .await
        .unwrap()
}

fn names(ctx: &SessionContext) -> Vec<&str> {
    ctx.tools.iter().map(|t| t.name.as_str()).collect()
}

fn shared() -> (Arc<InMemoryAgentFileStore>, FileAccessProvider) {
    let store = Arc::new(InMemoryAgentFileStore::new());
    (store.clone(), FileAccessProvider::new(store))
}

#[tokio::test]
async fn advertises_eight_tools_requiring_approval_and_its_instructions() {
    let (_, provider) = shared();
    let ctx = prepare(&provider, None).await;
    assert_eq!(
        names(&ctx),
        [
            "file_access_read",
            "file_access_read_lines",
            "file_access_ls",
            "file_access_grep",
            "file_access_write",
            "file_access_delete",
            "file_access_replace",
            "file_access_replace_lines",
        ]
    );
    assert!(ctx
        .tools
        .iter()
        .all(|t| t.approval_mode == ApprovalMode::AlwaysRequire));
    let instructions = ctx.instructions.clone().unwrap_or_default();
    assert!(instructions.starts_with("## File Access"));
    assert!(!instructions.contains("isolated to the current session"));
}

#[tokio::test]
async fn approval_switches_and_disabling_write_tools() {
    let (store, _) = shared();
    let provider = FileAccessProvider::new(store.clone()).disable_readonly_tool_approval();
    let ctx = prepare(&provider, None).await;
    for t in &ctx.tools {
        let read_only = FILE_ACCESS_READ_ONLY_TOOL_NAMES.contains(&t.name.as_str());
        let expected = if read_only {
            ApprovalMode::NeverRequire
        } else {
            ApprovalMode::AlwaysRequire
        };
        assert_eq!(t.approval_mode, expected, "{}", t.name);
    }
    let provider = FileAccessProvider::new(store.clone()).disable_write_tool_approval();
    let ctx = prepare(&provider, None).await;
    let write = ctx
        .tools
        .iter()
        .find(|t| t.name == "file_access_write")
        .unwrap();
    assert_eq!(write.approval_mode, ApprovalMode::NeverRequire);

    let provider = FileAccessProvider::new(store).disable_write_tools();
    let ctx = prepare(&provider, None).await;
    assert_eq!(
        names(&ctx),
        [
            "file_access_read",
            "file_access_read_lines",
            "file_access_ls",
            "file_access_grep"
        ]
    );
}

#[tokio::test]
async fn write_refuses_to_overwrite_unless_asked() {
    let (store, provider) = shared();
    let ctx = prepare(&provider, None).await;
    let out = call(
        &ctx,
        "file_access_write",
        json!({"file_name": "notes/a.md", "content": "one\n"}),
    )
    .await;
    assert_eq!(out, "File 'notes/a.md' written.");
    let out = call(
        &ctx,
        "file_access_write",
        json!({"file_name": "notes/a.md", "content": "two\n"}),
    )
    .await;
    assert_eq!(
        out,
        "File 'notes/a.md' already exists. To replace it, write again with overwrite set to true."
    );
    assert_eq!(
        store.read("notes/a.md").await.unwrap().as_deref(),
        Some("one\n")
    );
    call(
        &ctx,
        "file_access_write",
        json!({"file_name": "notes/a.md", "content": "two\n", "overwrite": true}),
    )
    .await;
    assert_eq!(
        call(&ctx, "file_access_read", json!({"file_name": "notes/a.md"})).await,
        "two\n"
    );
    let out = call(
        &ctx,
        "file_access_write",
        json!({"file_name": "../escape.md", "content": "x"}),
    )
    .await;
    assert!(
        out.as_str()
            .unwrap()
            .starts_with("Could not write file '../escape.md': "),
        "{out}"
    );
}

#[tokio::test]
async fn read_lines_numbers_each_line_and_validates_the_range() {
    let (store, provider) = shared();
    store.write("f.txt", "a\nb\nc\n").await.unwrap();
    let ctx = prepare(&provider, None).await;
    let args =
        |s: i64, e: Option<i64>| json!({"file_name": "f.txt", "start_line": s, "end_line": e});
    assert_eq!(
        call(&ctx, "file_access_read_lines", args(2, Some(99))).await,
        "2\tb\n3\tc\n4\t"
    );
    assert_eq!(
        call(&ctx, "file_access_read_lines", args(1, Some(1))).await,
        "1\ta\n"
    );
    for (s, e, msg) in [
        (0, None, "start_line must be a positive integer, got 0."),
        (
            3,
            Some(2),
            "end_line (2) must not be less than start_line (3).",
        ),
        (9, None, "start_line 9 is out of range (file has 4 lines)."),
    ] {
        let out = call(&ctx, "file_access_read_lines", args(s, e)).await;
        assert_eq!(
            out,
            format!("Could not read lines from file 'f.txt': {msg}")
        );
    }
    assert_eq!(
        call(
            &ctx,
            "file_access_read_lines",
            json!({"file_name": "missing.txt", "start_line": 1})
        )
        .await,
        "File 'missing.txt' not found."
    );
}

#[tokio::test]
async fn ls_grep_replace_and_delete() {
    let (store, provider) = shared();
    store.write("top.md", "alpha\n").await.unwrap();
    store
        .write("reports/2024/q1.md", "Alpha beta\nGamma\n")
        .await
        .unwrap();
    let ctx = prepare(&provider, None).await;

    assert_eq!(
        call(&ctx, "file_access_ls", json!({})).await,
        json!([{"name": "reports", "type": "directory"}, {"name": "top.md", "type": "file"}])
    );
    assert_eq!(
        call(
            &ctx,
            "file_access_ls",
            json!({"directory": "reports/2024", "glob_pattern": "*.md"})
        )
        .await,
        json!([{"name": "q1.md", "type": "file"}])
    );

    // Recursive from the root, and re-rooted when a directory is given.
    let all = call(&ctx, "file_access_grep", json!({"regex_pattern": "ALPHA"})).await;
    let mut files: Vec<&str> = all
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["file_name"].as_str().unwrap())
        .collect();
    files.sort();
    assert_eq!(files, ["reports/2024/q1.md", "top.md"]);
    let scoped = call(
        &ctx,
        "file_access_grep",
        json!({"regex_pattern": "gamma", "directory": "reports"}),
    )
    .await;
    assert_eq!(scoped[0]["file_name"], "reports/2024/q1.md");
    assert_eq!(scoped[0]["matching_lines"][0]["line_number"], 2);

    assert_eq!(
        call(
            &ctx,
            "file_access_replace",
            json!({"file_name": "top.md", "old_string": "alpha", "new_string": "omega"})
        )
        .await,
        "Replaced 1 occurrence(s) in 'top.md'."
    );
    assert_eq!(
        call(
            &ctx,
            "file_access_replace_lines",
            json!({"file_name": "reports/2024/q1.md", "edits": [{"line_number": 2, "new_line": "Delta\n", "expected_line": "Gamma"}]})
        )
        .await,
        "Replaced 1 line(s) in 'reports/2024/q1.md'."
    );
    assert_eq!(
        store.read("reports/2024/q1.md").await.unwrap().as_deref(),
        Some("Alpha beta\nDelta\n")
    );
    assert_eq!(
        call(&ctx, "file_access_delete", json!({"file_name": "top.md"})).await,
        "File 'top.md' deleted."
    );
    assert_eq!(
        call(&ctx, "file_access_delete", json!({"file_name": "top.md"})).await,
        "File 'top.md' not found."
    );
}

#[tokio::test]
async fn session_scoped_workspaces_are_isolated() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    store.write("shared.md", "root file\n").await.unwrap();
    let provider = FileAccessProvider::new(store.clone()).session_scoped();

    let one = prepare(&provider, Some("s1")).await;
    assert!(one
        .instructions
        .clone()
        .unwrap_or_default()
        .contains("isolated to the current session"));
    call(
        &one,
        "file_access_write",
        json!({"file_name": "a.md", "content": "mine\n"}),
    )
    .await;
    assert_eq!(
        store.read("~access-/s1/a.md").await.unwrap().as_deref(),
        Some("mine\n")
    );
    assert_eq!(
        call(&one, "file_access_ls", json!({})).await,
        json!([{"name": "a.md", "type": "file"}])
    );
    let found = call(
        &one,
        "file_access_grep",
        json!({"regex_pattern": "file|mine"}),
    )
    .await;
    assert_eq!(found.as_array().unwrap().len(), 1);
    assert_eq!(found[0]["file_name"], "a.md");

    let two = prepare(&provider, Some("s2")).await;
    assert_eq!(
        call(&two, "file_access_read", json!({"file_name": "a.md"})).await,
        "File 'a.md' not found."
    );

    // Scoped mode without any scope refuses to run rather than fall back to
    // the shared root.
    let mut ctx = SessionContext::new(vec![]);
    assert!(provider.before_run(&mut ctx).await.is_err());
}

#[tokio::test]
async fn prefixed_tools_and_auto_approval_rules() {
    let (store, _) = shared();
    let provider = FileAccessProvider::new(store).with_tool_name_prefix("docs");
    let ctx = prepare(&provider, None).await;
    assert!(names(&ctx)
        .iter()
        .all(|n| n.starts_with("docs_file_access_")));
    assert!(ctx
        .instructions
        .clone()
        .unwrap_or_default()
        .contains("`docs_file_access_grep`"));

    let call = |name: &str| FunctionCallContent {
        call_id: "c".into(),
        name: name.into(),
        arguments: None,
        protected_data: None,
        id: None,
    };
    assert!(FileAccessProvider::read_only_tools_auto_approval_rule(
        &call("file_access_ls")
    ));
    assert!(!FileAccessProvider::read_only_tools_auto_approval_rule(
        &call("file_access_write")
    ));
    assert!(FileAccessProvider::all_tools_auto_approval_rule(&call(
        "file_access_write"
    )));
    assert!(!FileAccessProvider::all_tools_auto_approval_rule(&call(
        "shell"
    )));
    let ro = provider.read_only_tools_approval_rule();
    assert!(ro(&call("docs_file_access_read")));
    assert!(!ro(&call("file_access_read")));
    assert!(!ro(&call("docs_file_access_delete")));
    let all = provider.all_tools_approval_rule();
    assert!(all(&call("docs_file_access_delete")));
}

#[tokio::test]
async fn file_access_and_file_memory_coexist_on_one_store() {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let access = FileAccessProvider::new(store.clone()).session_scoped();
    let memory = FileMemoryProvider::new(store.clone());
    let session = AgentSession::new();
    let a = prepare(&access, Some(session.session_id())).await;
    let mut m = SessionContext::new(vec![]);
    m.session = Some(session);
    memory.before_run(&mut m).await.unwrap();
    call(
        &a,
        "file_access_write",
        json!({"file_name": "x.md", "content": "access\n"}),
    )
    .await;
    let mem_write = m
        .tools
        .iter()
        .find(|t| t.name == "file_memory_write")
        .unwrap();
    mem_write
        .executor
        .as_ref()
        .unwrap()
        .invoke(json!({"file_name": "x.md", "content": "memory\n"}))
        .await
        .unwrap();
    assert_eq!(
        call(&a, "file_access_read", json!({"file_name": "x.md"})).await,
        "access\n"
    );
}
