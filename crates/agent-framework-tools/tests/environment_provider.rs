//! ShellEnvironmentProvider against a scripted executor (no real shell).

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_tools::shell::{
    default_instructions_formatter, ShellEnvironmentProvider, ShellEnvironmentProviderOptions,
    ShellEnvironmentSnapshot, ShellError, ShellExecutor, ShellFamily, ShellResult,
};
use async_trait::async_trait;

#[derive(Clone)]
enum Reply {
    Ok {
        stdout: &'static str,
        stderr: &'static str,
        exit_code: i32,
    },
    TimedOut,
    Rejected,
    Execution,
    Other,
}

fn ok(stdout: &'static str) -> Reply {
    Reply::Ok {
        stdout,
        stderr: "",
        exit_code: 0,
    }
}

#[derive(Default)]
struct FakeExecutor {
    replies: Mutex<HashMap<String, Reply>>,
    commands: Mutex<Vec<String>>,
    starts: AtomicUsize,
    delay: Option<Duration>,
    /// Fail every command with `ShellError::Other` until cleared.
    fail_other: Mutex<bool>,
}

impl FakeExecutor {
    fn with(replies: &[(&str, Reply)]) -> Self {
        Self {
            replies: Mutex::new(
                replies
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect(),
            ),
            ..Default::default()
        }
    }

    fn commands(&self) -> Vec<String> {
        self.commands.lock().unwrap().clone()
    }
}

#[async_trait]
impl ShellExecutor for FakeExecutor {
    async fn start(&self) -> Result<(), ShellError> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn close(&self) -> Result<(), ShellError> {
        Ok(())
    }

    async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        assert_eq!(
            timeout,
            Some(Duration::from_secs(5)),
            "probe timeout is passed through"
        );
        self.commands.lock().unwrap().push(command.to_string());
        if let Some(delay) = self.delay {
            tokio::time::sleep(delay).await;
        }
        if *self.fail_other.lock().unwrap() {
            return Err(ShellError::Other("executor bug".into()));
        }
        let key = if command.starts_with("echo \"VERSION=")
            || command.starts_with("Write-Output (\"VERSION=")
        {
            "shell".to_string()
        } else {
            command.to_string()
        };
        let reply = self
            .replies
            .lock()
            .unwrap()
            .get(&key)
            .cloned()
            .unwrap_or(Reply::Ok {
                stdout: "",
                stderr: "not found",
                exit_code: 127,
            });
        match reply {
            Reply::Ok {
                stdout,
                stderr,
                exit_code,
            } => Ok(ShellResult {
                stdout: stdout.into(),
                stderr: stderr.into(),
                exit_code,
                ..Default::default()
            }),
            Reply::TimedOut => Ok(ShellResult {
                timed_out: true,
                exit_code: 124,
                ..Default::default()
            }),
            Reply::Rejected => Err(ShellError::Rejected("blocked".into())),
            Reply::Execution => Err(ShellError::Execution("spawn failed".into())),
            Reply::Other => Err(ShellError::Other("bug".into())),
        }
    }
}

fn posix_options(tools: &[&str]) -> ShellEnvironmentProviderOptions {
    ShellEnvironmentProviderOptions {
        probe_tools: tools.iter().map(|t| t.to_string()).collect(),
        override_family: Some(ShellFamily::Posix),
        ..Default::default()
    }
}

#[tokio::test]
async fn probe_collects_shell_version_cwd_and_tools() {
    let exec = Arc::new(FakeExecutor::with(&[
        ("shell", ok("VERSION=5.2.15(1)-release\nCWD=/work\n")),
        ("git --version", ok("git version 2.43.0\n")),
    ]));
    let provider =
        ShellEnvironmentProvider::new(exec.clone(), Some(posix_options(&["git", "node"])));
    let snapshot = provider.refresh().await.unwrap();
    assert_eq!(snapshot.family, ShellFamily::Posix);
    assert_eq!(snapshot.shell_version.as_deref(), Some("5.2.15(1)-release"));
    assert_eq!(snapshot.working_directory, "/work");
    assert_eq!(
        snapshot.tool_versions["git"].as_deref(),
        Some("git version 2.43.0")
    );
    assert_eq!(snapshot.tool_versions["node"], None);
    assert_eq!(exec.starts.load(Ordering::SeqCst), 1);
    assert_eq!(provider.current_snapshot(), Some(snapshot));
}

#[tokio::test]
async fn unknown_shell_version_is_none() {
    let exec = Arc::new(FakeExecutor::with(&[(
        "shell",
        ok("VERSION=unknown\nCWD=/w\n"),
    )]));
    let snapshot = ShellEnvironmentProvider::new(exec, Some(posix_options(&[])))
        .refresh()
        .await
        .unwrap();
    assert_eq!(snapshot.shell_version, None);
}

#[tokio::test]
async fn probe_falls_back_to_stderr_for_version() {
    let exec = Arc::new(FakeExecutor::with(&[
        ("shell", ok("VERSION=5\nCWD=/\n")),
        (
            "java --version",
            Reply::Ok {
                stdout: "\n",
                stderr: "openjdk 21\n",
                exit_code: 0,
            },
        ),
    ]));
    let snapshot = ShellEnvironmentProvider::new(exec, Some(posix_options(&["java"])))
        .refresh()
        .await
        .unwrap();
    assert_eq!(
        snapshot.tool_versions["java"].as_deref(),
        Some("openjdk 21")
    );
}

#[tokio::test]
async fn expected_probe_failures_become_missing_values() {
    let exec = Arc::new(FakeExecutor::with(&[
        ("shell", Reply::TimedOut),
        ("git --version", Reply::Rejected),
        ("node --version", Reply::Execution),
        ("python --version", Reply::TimedOut),
    ]));
    let snapshot =
        ShellEnvironmentProvider::new(exec, Some(posix_options(&["git", "node", "python"])))
            .refresh()
            .await
            .unwrap();
    assert_eq!(snapshot.shell_version, None);
    assert_eq!(snapshot.working_directory, "");
    assert!(snapshot.tool_versions.values().all(Option::is_none));
}

#[tokio::test]
async fn unexpected_errors_propagate() {
    let exec = Arc::new(FakeExecutor::with(&[("shell", Reply::Other)]));
    let provider = ShellEnvironmentProvider::new(exec, Some(posix_options(&[])));
    assert!(matches!(
        provider.refresh().await,
        Err(ShellError::Other(_))
    ));
    let mut ctx = SessionContext::new(vec![]);
    assert!(provider.before_run(&mut ctx).await.is_err());
}

#[tokio::test]
async fn invalid_tool_names_are_never_spliced_into_a_command() {
    let exec = Arc::new(FakeExecutor::with(&[("shell", ok("VERSION=5\nCWD=/\n"))]));
    let provider = ShellEnvironmentProvider::new(
        exec.clone(),
        Some(posix_options(&[
            "git; rm -rf /",
            "$(id)",
            "a b",
            "",
            "ok-tool_1.2",
        ])),
    );
    let snapshot = provider.refresh().await.unwrap();
    let commands = exec.commands();
    assert_eq!(commands.len(), 2, "{commands:?}");
    assert_eq!(commands[1], "ok-tool_1.2 --version");
    assert_eq!(snapshot.tool_versions["git; rm -rf /"], None);
}

#[tokio::test]
async fn duplicate_tools_are_probed_once_case_insensitively() {
    let exec = Arc::new(FakeExecutor::with(&[("shell", ok("VERSION=5\nCWD=/\n"))]));
    let provider =
        ShellEnvironmentProvider::new(exec.clone(), Some(posix_options(&["git", "GIT", "Git"])));
    let snapshot = provider.refresh().await.unwrap();
    assert_eq!(
        exec.commands()
            .iter()
            .filter(|c| c.ends_with("--version"))
            .count(),
        1
    );
    assert_eq!(snapshot.tool_versions.len(), 1);
}

#[tokio::test]
async fn failed_probe_does_not_poison_later_runs() {
    let exec = Arc::new(FakeExecutor::with(&[("shell", ok("VERSION=5\nCWD=/w\n"))]));
    *exec.fail_other.lock().unwrap() = true;
    let provider = ShellEnvironmentProvider::new(exec.clone(), Some(posix_options(&[])));
    let mut ctx = SessionContext::new(vec![]);
    assert!(provider.before_run(&mut ctx).await.is_err());
    assert!(provider.current_snapshot().is_none());

    *exec.fail_other.lock().unwrap() = false;
    provider.before_run(&mut ctx).await.unwrap();
    assert!(ctx.instructions.unwrap().contains("Working directory: /w"));
}

#[tokio::test]
async fn concurrent_first_callers_share_one_probe() {
    let exec = Arc::new(FakeExecutor {
        delay: Some(Duration::from_millis(50)),
        ..FakeExecutor::with(&[("shell", ok("VERSION=5\nCWD=/\n"))])
    });
    let provider = Arc::new(ShellEnvironmentProvider::new(
        exec.clone(),
        Some(posix_options(&[])),
    ));
    let mut handles = Vec::new();
    for _ in 0..5 {
        let provider = provider.clone();
        handles.push(tokio::spawn(async move {
            let mut ctx = SessionContext::new(vec![]);
            provider.before_run(&mut ctx).await.unwrap();
            ctx.instructions.unwrap()
        }));
    }
    for handle in handles {
        assert!(handle.await.unwrap().starts_with("## Shell environment"));
    }
    assert_eq!(exec.commands().len(), 1, "the probe ran more than once");
}

#[tokio::test]
async fn before_run_appends_the_default_block() {
    let exec = Arc::new(FakeExecutor::with(&[
        ("shell", ok("VERSION=5.2\nCWD=/repo\n")),
        ("git --version", ok("git version 2.43.0")),
    ]));
    let provider = ShellEnvironmentProvider::new(exec, Some(posix_options(&["git", "docker"])));
    let mut ctx = SessionContext::new(vec![]);
    ctx.add_instructions("existing");
    provider.before_run(&mut ctx).await.unwrap();
    let text = ctx.instructions.unwrap();
    assert!(text.starts_with("existing\n## Shell environment\n"));
    assert!(text.contains("You are operating a POSIX shell 5.2 session on "));
    assert!(text.contains("export NAME=value"));
    assert!(text.contains("Working directory: /repo"));
    assert!(text.contains("Available CLIs: git (git version 2.43.0)"));
    assert!(text.contains("Not installed: docker"));
}

#[test]
fn default_formatter_powershell_block_uses_pwsh_idioms() {
    let snapshot = ShellEnvironmentSnapshot {
        family: ShellFamily::PowerShell,
        os_description: "windows x86_64".into(),
        shell_version: Some("7.4.1".into()),
        working_directory: "C:\\repo".into(),
        tool_versions: Default::default(),
    };
    let text = default_instructions_formatter(&snapshot);
    assert!(text.contains("PowerShell 7.4.1 session on windows x86_64"));
    assert!(text.contains("$env:NAME = 'value'"));
    assert!(text.contains("Out-Null"));
    assert!(!text.contains("export NAME"));
}

#[tokio::test]
async fn custom_formatter_is_used() {
    let exec = Arc::new(FakeExecutor::with(&[("shell", ok("VERSION=5\nCWD=/x\n"))]));
    let options = posix_options(&[]).with_instructions_formatter(|s: &ShellEnvironmentSnapshot| {
        format!("custom:{}", s.working_directory)
    });
    let provider = ShellEnvironmentProvider::new(exec, Some(options));
    let mut ctx = SessionContext::new(vec![]);
    provider.before_run(&mut ctx).await.unwrap();
    assert_eq!(ctx.instructions.as_deref(), Some("custom:/x"));
}

/// The real local tool behind the provider, end to end.
#[cfg(unix)]
#[tokio::test]
async fn probes_a_real_local_shell() {
    use agent_framework_core::tools::ApprovalMode;
    use agent_framework_tools::shell::{LocalShellTool, ShellMode};

    let tool = LocalShellTool::builder()
        .mode(ShellMode::Stateless)
        .approval_mode(ApprovalMode::NeverRequire)
        .acknowledge_unsafe(true)
        .build()
        .unwrap();
    let options = ShellEnvironmentProviderOptions {
        probe_tools: vec!["sh-definitely-missing-af".into()],
        override_family: Some(ShellFamily::Posix),
        ..Default::default()
    };
    let provider = ShellEnvironmentProvider::new(Arc::new(tool), Some(options));
    let snapshot = provider.refresh().await.unwrap();
    assert!(!snapshot.working_directory.is_empty());
    assert_eq!(snapshot.tool_versions["sh-definitely-missing-af"], None);
}
