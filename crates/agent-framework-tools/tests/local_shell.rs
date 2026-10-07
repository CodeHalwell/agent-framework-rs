//! LocalShellTool end to end against the real platform shell (POSIX only:
//! these run `sh`/`bash`, which every Linux CI image has). No network, no
//! Docker.
#![cfg(unix)]

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::tools::ApprovalMode;
use agent_framework_tools::shell::{
    LocalShellTool, LocalShellToolBuilder, ShellError, ShellExecutor, ShellMode, ShellPolicy,
};
use common::{eventually, process_alive, TempDir};
use serde_json::json;

fn unsafe_builder(mode: ShellMode) -> LocalShellToolBuilder {
    LocalShellTool::builder()
        .mode(mode)
        .approval_mode(ApprovalMode::NeverRequire)
        .acknowledge_unsafe(true)
}

fn stateless() -> LocalShellToolBuilder {
    unsafe_builder(ShellMode::Stateless)
}

fn persistent() -> LocalShellToolBuilder {
    unsafe_builder(ShellMode::Persistent)
}

// ---------------------------------------------------------------- security defaults

#[test]
fn defaults_are_persistent_and_approval_required() {
    let tool = LocalShellTool::builder().build().unwrap();
    assert_eq!(tool.mode(), ShellMode::Persistent);
    assert_eq!(tool.approval_mode(), ApprovalMode::AlwaysRequire);
    assert_eq!(tool.timeout(), Some(Duration::from_secs(30)));
    assert_eq!(tool.max_output_bytes(), 64 * 1024);
    let function = tool.as_function();
    assert_eq!(function.name, "run_shell");
    assert!(function.requires_approval());
    assert!(function.is_executable());
    assert_eq!(function.parameters["required"], json!(["command"]));
}

#[test]
fn disabling_approval_requires_acknowledgement() {
    let err = LocalShellTool::builder()
        .approval_mode(ApprovalMode::NeverRequire)
        .build()
        .unwrap_err();
    assert!(matches!(err, ShellError::Config(_)));
    assert!(err.to_string().contains("acknowledge_unsafe"));
    let tool = LocalShellTool::builder()
        .approval_mode(ApprovalMode::NeverRequire)
        .acknowledge_unsafe(true)
        .build()
        .unwrap();
    assert!(!tool.as_function().requires_approval());
}

#[test]
fn zero_output_limit_is_rejected() {
    assert!(LocalShellTool::builder()
        .max_output_bytes(0)
        .build()
        .is_err());
}

#[test]
fn as_function_with_custom_name_and_description() {
    let tool = LocalShellTool::builder().build().unwrap();
    let function = tool.as_function_with("shell_exec", Some("custom shell"));
    assert_eq!(function.name, "shell_exec");
    assert_eq!(function.description, "custom shell");
    assert!(function.requires_approval());
    let stateless = LocalShellTool::builder()
        .mode(ShellMode::Stateless)
        .build()
        .unwrap();
    assert!(stateless
        .as_function()
        .description
        .contains("fresh subprocess"));
    assert!(tool
        .as_function()
        .description
        .contains("persistent session"));
}

#[test]
fn environment_inherit_merge_and_clean() {
    std::env::set_var("AF_TOOLS_INHERITED", "yes");
    let inherited = stateless().build().unwrap();
    assert!(
        inherited.environment().is_none(),
        "no env given: inherit unchanged"
    );

    let merged = stateless().env("EXTRA", "1").build().unwrap();
    let env = merged.environment().unwrap();
    assert_eq!(env["AF_TOOLS_INHERITED"], "yes");
    assert_eq!(env["EXTRA"], "1");

    let clean = stateless()
        .env("ONLY", "2")
        .clean_env(true)
        .build()
        .unwrap();
    assert_eq!(clean.environment().unwrap().len(), 1);
    assert_eq!(clean.environment().unwrap()["ONLY"], "2");
}

// ---------------------------------------------------------------- stateless

#[tokio::test]
async fn stateless_echo() {
    let tool = stateless().build().unwrap();
    let result = tool.run("echo hello", None).await.unwrap();
    assert_eq!(result.stdout, "hello\n");
    assert_eq!(result.exit_code, 0);
    assert!(!result.timed_out);
    assert!(!result.truncated);
}

#[tokio::test]
async fn stateless_exit_code_and_stderr_propagate() {
    let tool = stateless().build().unwrap();
    let result = tool
        .run("echo oops >&2; sh -c 'exit 7'", None)
        .await
        .unwrap();
    assert_eq!(result.exit_code, 7);
    assert_eq!(result.stderr, "oops\n");
}

#[tokio::test]
async fn stateless_start_and_close_are_noops() {
    let tool = stateless().build().unwrap();
    tool.start().await.unwrap();
    tool.close().await.unwrap();
    assert_eq!(
        tool.run("echo still", None).await.unwrap().stdout,
        "still\n"
    );
}

#[tokio::test]
async fn stateless_runs_in_workdir_with_clean_env() {
    let dir = TempDir::new("stateless-wd");
    let tool = stateless()
        .workdir(dir.path())
        .env("AF_MARKER", "xyz")
        .clean_env(true)
        .build()
        .unwrap();
    let result = tool
        .run("pwd; echo \"$AF_MARKER\"; echo \"${HOME:-unset}\"", None)
        .await
        .unwrap();
    let lines: Vec<&str> = result.stdout.lines().collect();
    assert_eq!(std::fs::canonicalize(lines[0]).unwrap(), dir.path());
    assert_eq!(lines[1], "xyz");
    assert_eq!(
        lines[2], "unset",
        "clean_env must not leak the parent environment"
    );
}

#[tokio::test]
async fn stateless_does_not_keep_state() {
    let tool = stateless().build().unwrap();
    tool.run("export AF_GONE=1; cd /", None).await.unwrap();
    let result = tool.run("echo \"${AF_GONE:-none}\"", None).await.unwrap();
    assert_eq!(result.stdout, "none\n");
}

#[tokio::test]
async fn stateless_timeout_kills_long_command() {
    let tool = stateless()
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    let started = std::time::Instant::now();
    let result = tool.run("sleep 5", None).await.unwrap();
    assert!(result.timed_out);
    assert!(started.elapsed() < Duration::from_secs(4));
    assert!(result.format_for_model().contains("[command timed out]"));
}

#[tokio::test]
async fn per_call_timeout_overrides_default() {
    let tool = stateless().timeout(None).build().unwrap();
    let result = tool
        .run("sleep 5", Some(Duration::from_millis(200)))
        .await
        .unwrap();
    assert!(result.timed_out);
}

/// The timeout must take the command's children with it, not just the shell.
#[tokio::test]
async fn stateless_timeout_kills_the_process_group() {
    let tool = stateless()
        .timeout(Some(Duration::from_millis(500)))
        .build()
        .unwrap();
    let result = tool.run("sleep 30 & echo $!; wait", None).await.unwrap();
    assert!(result.timed_out);
    let pid: u32 = result.stdout.trim().parse().expect("background pid");
    assert!(
        eventually(|| !process_alive(pid)).await,
        "background child {pid} survived the timeout"
    );
}

/// Dropping the run future (cancellation) must not leak the command.
#[tokio::test]
async fn cancelled_stateless_run_kills_the_process_group() {
    let dir = TempDir::new("cancel");
    let pidfile = dir.path().join("pid");
    let tool = stateless().timeout(None).build().unwrap();
    let command = format!("sleep 30 & echo $! > '{}'; wait", pidfile.display());
    let outcome = tokio::time::timeout(Duration::from_millis(500), tool.run(&command, None)).await;
    assert!(outcome.is_err(), "the run should still have been going");
    let pid: u32 = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(
        eventually(|| !process_alive(pid)).await,
        "child {pid} survived cancellation"
    );
}

/// A background job left behind by a shell that exited normally must not
/// hold the pipes open (losing the output) or outlive the command.
#[tokio::test]
async fn stateless_background_job_is_killed_and_output_kept() {
    let tool = stateless().timeout(None).build().unwrap();
    let result = tool.run("sleep 30 & echo $!; echo ok", None).await.unwrap();
    assert!(!result.timed_out);
    assert_eq!(result.exit_code, 0);
    let mut lines = result.stdout.lines();
    let pid: u32 = lines.next().expect("pid").trim().parse().unwrap();
    assert_eq!(lines.next(), Some("ok"), "{:?}", result.stdout);
    assert!(
        result.duration < Duration::from_secs(1),
        "waited on the background job: {:?}",
        result.duration
    );
    assert!(
        eventually(|| !process_alive(pid)).await,
        "background job {pid} outlived the command"
    );
}

#[tokio::test]
async fn stateless_output_is_truncated_head_and_tail() {
    let tool = stateless().max_output_bytes(64).build().unwrap();
    let result = tool
        .run(
            "echo START; head -c 100000 /dev/zero | tr '\\0' x; echo; echo END",
            None,
        )
        .await
        .unwrap();
    assert!(result.truncated);
    assert!(result.stdout.starts_with("START"));
    assert!(result.stdout.trim_end().ends_with("END"));
    assert!(result.stdout.contains("[... truncated "));
    assert!(result.stdout.len() < 200);
}

#[tokio::test]
async fn missing_shell_is_an_execution_error() {
    let tool = stateless()
        .shell(["/definitely/not/a/shell"])
        .build()
        .unwrap();
    let err = tool.run("echo hi", None).await.unwrap_err();
    assert!(matches!(err, ShellError::Execution(_)), "{err:?}");
}

// ---------------------------------------------------------------- policy and audit

#[tokio::test]
async fn policy_denies_before_execution() {
    let dir = TempDir::new("policy");
    let marker = dir.path().join("ran");
    let policy = ShellPolicy::new().with_denylist([r"\btouch\b"]).unwrap();
    let tool = stateless().policy(policy).build().unwrap();
    let err = tool
        .run(&format!("touch '{}'", marker.display()), None)
        .await
        .unwrap_err();
    assert!(matches!(err, ShellError::Rejected(_)));
    assert!(err
        .to_string()
        .starts_with("Command rejected by policy: matches denylist pattern"));
    assert!(!marker.exists(), "a denied command must not run");
}

#[tokio::test]
async fn allowlist_narrows_to_approved_commands() {
    let policy = ShellPolicy::new().with_allowlist([r"^echo\b"]).unwrap();
    let tool = stateless().policy(policy).build().unwrap();
    assert_eq!(tool.run("echo ok", None).await.unwrap().stdout, "ok\n");
    assert!(matches!(
        tool.run("ls -la", None).await,
        Err(ShellError::Rejected(_))
    ));
}

#[tokio::test]
async fn empty_command_is_rejected() {
    let tool = stateless().build().unwrap();
    assert!(matches!(
        tool.run("   ", None).await,
        Err(ShellError::Rejected(_))
    ));
}

#[tokio::test]
async fn audit_hook_fires_for_allowed_commands_only() {
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let sink = seen.clone();
    let policy = ShellPolicy::new().with_denylist(["forbidden"]).unwrap();
    let tool = stateless()
        .policy(policy)
        .on_command(move |cmd| sink.lock().unwrap().push(cmd.to_string()))
        .build()
        .unwrap();
    tool.run("echo hi", None).await.unwrap();
    let _ = tool.run("echo forbidden", None).await;
    assert_eq!(*seen.lock().unwrap(), ["echo hi"]);
}

#[tokio::test]
async fn panicking_audit_hook_does_not_stop_the_command() {
    let tool = stateless()
        .on_command(|_| panic!("audit sink down"))
        .build()
        .unwrap();
    assert_eq!(
        tool.run("echo survived", None).await.unwrap().stdout,
        "survived\n"
    );
}

// ---------------------------------------------------------------- the function tool

#[tokio::test]
async fn function_returns_model_text_and_policy_errors() {
    let policy = ShellPolicy::new().with_denylist(["blocked"]).unwrap();
    let tool = stateless().policy(policy).build().unwrap();
    let function = tool.as_function();
    let executor = function.executor.clone().unwrap();

    let ok = executor
        .invoke(json!({"command": "echo hi; echo err >&2; exit 3"}))
        .await
        .unwrap();
    assert_eq!(ok, json!("hi\n\nstderr: err\n\nexit_code: 3"));

    let rejected = executor
        .invoke(json!({"command": "echo blocked"}))
        .await
        .unwrap();
    assert_eq!(
        rejected,
        json!("Command rejected by policy: matches denylist pattern: blocked")
    );

    let bad_args = executor.invoke(json!({"cmd": "echo"})).await.unwrap_err();
    assert!(bad_args.to_string().contains("command"));
}

#[tokio::test]
async fn function_surfaces_execution_failures_as_errors() {
    let tool = stateless()
        .shell(["/definitely/not/a/shell"])
        .build()
        .unwrap();
    let executor = tool.as_function().executor.unwrap();
    assert!(executor
        .invoke(json!({"command": "echo hi"}))
        .await
        .is_err());
}

// ---------------------------------------------------------------- persistent

#[tokio::test]
async fn persistent_preserves_cwd_and_exports_across_calls() {
    let dir = TempDir::new("persist");
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let tool = persistent()
        .workdir(dir.path())
        .confine_workdir(false)
        .build()
        .unwrap();
    tool.run("export AF_TEST_MARKER=xyz", None).await.unwrap();
    assert_eq!(
        tool.run("echo $AF_TEST_MARKER", None).await.unwrap().stdout,
        "xyz"
    );
    tool.run(&format!("cd '{}'", sub.display()), None)
        .await
        .unwrap();
    let pwd = tool.run("pwd", None).await.unwrap();
    assert_eq!(std::fs::canonicalize(pwd.stdout.trim()).unwrap(), sub);
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_confines_workdir_by_default() {
    let dir = TempDir::new("confine");
    let sub = dir.path().join("sub");
    std::fs::create_dir(&sub).unwrap();
    let tool = persistent().workdir(dir.path()).build().unwrap();
    tool.run(&format!("cd '{}'", sub.display()), None)
        .await
        .unwrap();
    let pwd = tool.run("pwd", None).await.unwrap();
    assert_eq!(
        std::fs::canonicalize(pwd.stdout.trim()).unwrap(),
        dir.path()
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_reports_exit_codes_and_stderr() {
    let tool = persistent().build().unwrap();
    let failed = tool
        .run("echo out; echo err >&2; (exit 5)", None)
        .await
        .unwrap();
    assert_eq!(failed.exit_code, 5);
    assert_eq!(failed.stdout, "out");
    assert_eq!(failed.stderr, "err\n");
    let ok = tool.run("true", None).await.unwrap();
    assert_eq!(ok.exit_code, 0);
    assert_eq!(ok.stderr, "", "stderr of an earlier command must not leak");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_set_e_survives_and_does_not_kill_the_session() {
    let tool = persistent().build().unwrap();
    tool.run("set -e", None).await.unwrap();
    let failed = tool.run("false", None).await.unwrap();
    assert_eq!(failed.exit_code, 1);
    // The wrapper turns errexit off around each command and restores it
    // afterwards, so the shell is still there for the next one.
    let next = tool.run("echo alive", None).await.unwrap();
    assert_eq!(next.stdout, "alive");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn concurrent_first_calls_share_one_shell() {
    let tool = persistent().build().unwrap();
    let (a, b) = tokio::join!(tool.run("echo $$", None), tool.run("echo $$", None));
    assert_eq!(
        a.unwrap().stdout,
        b.unwrap().stdout,
        "two shells were spawned"
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn sentinel_lookalike_does_not_corrupt_the_session() {
    let tool = persistent().build().unwrap();
    let result = tool
        .run("echo '__AF_END_fakebutscary___1234'", None)
        .await
        .unwrap();
    assert!(result.stdout.contains("__AF_END_fakebutscary__"));
    assert_eq!(result.exit_code, 0);
    let follow = tool.run("echo still-alive", None).await.unwrap();
    assert_eq!(follow.stdout, "still-alive");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_timeout_returns_and_next_command_works() {
    let tool = persistent()
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    let result = tool.run("sleep 10", None).await.unwrap();
    assert!(result.timed_out);
    let next = tool.run("echo recovered", None).await.unwrap();
    assert_eq!(next.stdout, "recovered");
    assert!(!next.timed_out);
    tool.close().await.unwrap();
}

/// Large output is not an error in persistent mode: the command runs to
/// completion and its output is head/tail truncated, as in stateless mode.
#[tokio::test]
async fn persistent_large_output_completes_with_truncated_output() {
    let tool = persistent()
        .max_output_bytes(1024)
        .timeout(Some(Duration::from_secs(20)))
        .build()
        .unwrap();
    // ~200x the limit, well past what used to count as runaway output.
    let result = tool
        .run(
            "echo FIRST; i=0; while [ $i -lt 2000 ]; do printf '%0100d\\n' $i; i=$((i+1)); done; echo LAST; export AF_KEPT=1; exit_code_check() { return 3; }; exit_code_check",
            None,
        )
        .await
        .unwrap();
    assert!(!result.timed_out);
    assert_eq!(result.exit_code, 3, "{result:?}");
    assert!(result.truncated);
    assert!(result.stdout.starts_with("FIRST\n"), "{}", result.stdout);
    assert!(result.stdout.ends_with("\nLAST"), "{}", result.stdout);
    assert!(
        result.stdout.contains("[... truncated "),
        "{}",
        result.stdout
    );
    assert!(result.stdout.len() < 1024 + 64, "{}", result.stdout.len());
    // The same shell carries on: state from the command is still there.
    assert_eq!(tool.run("echo $AF_KEPT", None).await.unwrap().stdout, "1");
    tool.close().await.unwrap();
}

/// A command that never stops printing is ended by the timeout, with
/// bounded output, and the next command still works.
#[tokio::test]
async fn persistent_endless_output_is_bounded_and_ends_at_the_timeout() {
    let tool = persistent()
        .max_output_bytes(1024)
        .timeout(Some(Duration::from_millis(500)))
        .build()
        .unwrap();
    let result = tool.run("yes x", None).await.unwrap();
    assert!(result.timed_out);
    assert!(result.truncated);
    assert!(result.stdout.len() < 1024 + 64, "{}", result.stdout.len());
    assert_eq!(tool.run("echo fresh", None).await.unwrap().stdout, "fresh");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_close_then_run_starts_a_new_shell() {
    let tool = persistent().build().unwrap();
    let first = tool.run("echo $$", None).await.unwrap().stdout;
    tool.close().await.unwrap();
    tool.close().await.unwrap(); // idempotent
    let second = tool.run("echo $$", None).await.unwrap().stdout;
    assert_ne!(first, second);
    let pid: u32 = first.trim().parse().unwrap();
    assert!(
        eventually(|| !process_alive(pid)).await,
        "old shell {pid} survived close"
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn cancelled_persistent_run_replaces_the_shell() {
    let tool = persistent().timeout(None).build().unwrap();
    let shell_pid = tool.run("echo $$", None).await.unwrap().stdout;
    let outcome =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(outcome.is_err());
    let next = tool.run("echo $$; echo after", None).await.unwrap();
    assert!(next.stdout.ends_with("after"));
    assert_ne!(next.stdout.lines().next().unwrap(), shell_pid.trim());
    let pid: u32 = shell_pid.trim().parse().unwrap();
    assert!(eventually(|| !process_alive(pid)).await);
    tool.close().await.unwrap();
}

/// Cancelling a persistent run stops the command at once, not only at the
/// next call or when the tool is dropped.
#[tokio::test]
async fn cancelled_persistent_run_kills_the_command_immediately() {
    let dir = TempDir::new("cancel-persistent");
    let pidfile = dir.path().join("pid");
    let tool = persistent()
        .confine_workdir(false)
        .timeout(None)
        .build()
        .unwrap();
    let command = format!("sleep 30 & echo $! > '{}'; wait", pidfile.display());
    let outcome = tokio::time::timeout(Duration::from_millis(500), tool.run(&command, None)).await;
    assert!(outcome.is_err(), "the run should still have been going");
    let pid: u32 = std::fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    // No further call, no close, and the tool is still alive.
    assert!(
        eventually(|| !process_alive(pid)).await,
        "cancelled command {pid} kept running"
    );
    assert_eq!(tool.run("echo next", None).await.unwrap().stdout, "next");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_env_is_applied() {
    let tool = persistent().env("AF_SEEDED", "seed").build().unwrap();
    assert_eq!(
        tool.run("echo $AF_SEEDED", None).await.unwrap().stdout,
        "seed"
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn clones_share_the_persistent_shell() {
    let tool = persistent().confine_workdir(false).build().unwrap();
    let clone = tool.clone();
    tool.run("export AF_SHARED=1", None).await.unwrap();
    assert_eq!(
        clone.run("echo $AF_SHARED", None).await.unwrap().stdout,
        "1"
    );
    tool.close().await.unwrap();
}
