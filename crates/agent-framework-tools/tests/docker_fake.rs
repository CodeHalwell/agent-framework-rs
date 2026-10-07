//! DockerShellTool driven through a fake `docker` CLI: a small `sh` script
//! that logs its argv and runs commands on the host. This exercises the
//! container lifecycle, the exact argv handed to the runtime and the timeout
//! reaper without Docker being installed.
#![cfg(unix)]

mod common;

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use agent_framework_tools::shell::{
    is_docker_available, DockerShellTool, ShellError, ShellExecutor, ShellMode, ShellPolicy,
};
use common::TempDir;
use serde_json::json;

struct FakeDocker {
    _dir: TempDir,
    binary: PathBuf,
    log: PathBuf,
}

impl FakeDocker {
    /// `kill_rc` is what `docker kill` exits with; `run_rc` what
    /// `docker run -d` exits with.
    fn new(kill_rc: i32, run_rc: i32) -> Self {
        Self::with_reap_rc(kill_rc, run_rc, 0)
    }

    /// `reap_rc` is what a non-interactive `docker exec` (the in-container
    /// process reaper) exits with.
    fn with_reap_rc(kill_rc: i32, run_rc: i32, reap_rc: i32) -> Self {
        let dir = TempDir::new("fake-docker");
        let log = dir.path().join("calls.log");
        let binary = dir.path().join("docker");
        let script = format!(
            r#"#!/bin/sh
printf '%s\n' "$*" >> '{log}'
case "$1" in
  run)
    if [ "$2" = "-d" ]; then
      if [ {run_rc} -ne 0 ]; then echo "no such image" >&2; exit {run_rc}; fi
      echo abcdef1234567890; exit 0
    fi
    for last; do :; done
    exec /bin/sh -c "$last" ;;
  exec)
    if [ "$2" = "-i" ]; then exec /bin/sh; fi
    exit {reap_rc} ;;
  kill) exit {kill_rc} ;;
  rm) exit 0 ;;
  version) echo 27.0.0 ;;
esac
"#,
            log = log.display()
        );
        std::fs::write(&binary, script).unwrap();
        std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
        // A process forked by another test thread while the script was open
        // for writing briefly inherits that descriptor, and exec then fails
        // with ETXTBSY. Wait until the script can be executed.
        for _ in 0..200 {
            match std::process::Command::new(&binary).arg("version").output() {
                Err(err) if err.raw_os_error() == Some(26) => {
                    std::thread::sleep(std::time::Duration::from_millis(10))
                }
                _ => break,
            }
        }
        let _ = std::fs::remove_file(&log);
        Self {
            _dir: dir,
            binary,
            log,
        }
    }

    /// Poll the call log until `pred` matches a line or two seconds pass.
    async fn wait_for_call(&self, pred: impl Fn(&str) -> bool) -> Vec<String> {
        for _ in 0..100 {
            let calls = self.calls();
            if calls.iter().any(|c| pred(c)) {
                return calls;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        self.calls()
    }

    fn calls(&self) -> Vec<String> {
        std::fs::read_to_string(&self.log)
            .unwrap_or_default()
            .lines()
            .map(str::to_string)
            .collect()
    }

    fn binary(&self) -> &Path {
        &self.binary
    }
}

fn builder(fake: &FakeDocker) -> agent_framework_tools::shell::DockerShellToolBuilder {
    DockerShellTool::builder()
        .docker_binary(fake.binary().to_string_lossy())
        .image("alpine:3")
        .shell("sh")
}

#[tokio::test]
async fn availability_check_uses_the_binary() {
    let fake = FakeDocker::new(0, 0);
    assert!(is_docker_available(&fake.binary().to_string_lossy()).await);
    assert!(!is_docker_available("/definitely/not/docker").await);
}

#[tokio::test]
async fn stateless_runs_each_command_in_a_fresh_container() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .mode(ShellMode::Stateless)
        .host_workdir("/repo")
        .mount_readonly(false)
        .env("AF_TEST", "1")
        .build()
        .unwrap();
    tool.start().await.unwrap();
    let result = tool
        .run("echo hello; echo warn >&2; exit 3", None)
        .await
        .unwrap();
    tool.close().await.unwrap();
    assert_eq!(result.stdout, "hello\n");
    assert_eq!(result.stderr, "warn\n");
    assert_eq!(result.exit_code, 3);
    assert!(!result.timed_out);

    let calls = fake.calls();
    assert_eq!(
        calls.len(),
        1,
        "start/close must not touch docker in stateless mode: {calls:?}"
    );
    let call = &calls[0];
    assert!(call.starts_with("run --rm -i --name af-shell-"));
    assert!(call.contains("--network none"));
    assert!(call.contains("--cap-drop ALL"));
    assert!(call.contains("-v /repo:/workspace:rw"));
    assert!(call.contains("-e AF_TEST=1"));
    assert!(call.ends_with("alpine:3 sh -c echo hello; echo warn >&2; exit 3"));
}

#[tokio::test]
async fn stateless_timeout_kills_then_removes_the_container() {
    let fake = FakeDocker::new(1, 0);
    let tool = builder(&fake)
        .mode(ShellMode::Stateless)
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    let result = tool.run("sleep 30", None).await.unwrap();
    assert!(result.timed_out);

    let calls = fake.calls();
    let name = container_name_of(&calls[0]);
    assert!(
        calls.contains(&format!("kill --signal KILL {name}")),
        "{calls:?}"
    );
    assert!(
        calls.contains(&format!("rm -f {name}")),
        "a failed kill must fall back to rm -f: {calls:?}"
    );
}

#[tokio::test]
async fn stateless_timeout_skips_rm_when_kill_succeeds() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .mode(ShellMode::Stateless)
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    assert!(tool.run("sleep 30", None).await.unwrap().timed_out);
    let calls = fake.calls();
    assert!(calls
        .iter()
        .any(|c| c.starts_with("kill --signal KILL af-shell-")));
    assert!(!calls.iter().any(|c| c.starts_with("rm -f")), "{calls:?}");
}

#[tokio::test]
async fn persistent_starts_one_container_and_removes_it_on_close() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .container_name("af-test-box")
        .build()
        .unwrap();
    tool.run("export AF_STATE=kept", None).await.unwrap();
    let result = tool.run("echo $AF_STATE", None).await.unwrap();
    assert_eq!(result.stdout, "kept");
    tool.close().await.unwrap();
    tool.close().await.unwrap(); // idempotent

    let calls = fake.calls();
    assert_eq!(calls.len(), 3, "{calls:?}");
    assert!(calls[0].starts_with("run -d --rm --name af-test-box --user 65534:65534"));
    assert!(calls[0].ends_with("alpine:3 sleep infinity"));
    assert_eq!(calls[1], "exec -i af-test-box sh");
    assert_eq!(calls[2], "rm -f af-test-box");
}

#[tokio::test]
async fn container_start_failure_is_reported() {
    let fake = FakeDocker::new(0, 125);
    let tool = builder(&fake).build().unwrap();
    let err = tool.run("echo hi", None).await.unwrap_err();
    assert!(matches!(err, ShellError::Execution(_)));
    assert!(err.to_string().contains("no such image"), "{err}");
}

#[tokio::test]
async fn policy_rejects_before_any_container_starts() {
    let fake = FakeDocker::new(0, 0);
    let policy = ShellPolicy::new().with_denylist(["curl"]).unwrap();
    let tool = builder(&fake).policy(policy).build().unwrap();
    let function = tool.as_function();
    let out = function
        .executor
        .unwrap()
        .invoke(json!({"command": "curl example.com"}))
        .await
        .unwrap();
    assert_eq!(
        out,
        json!("Command rejected by policy: matches denylist pattern: curl")
    );
    assert!(fake.calls().is_empty());
}

fn container_name_of(call: &str) -> String {
    call.split_whitespace()
        .skip_while(|t| *t != "--name")
        .nth(1)
        .unwrap()
        .to_string()
}

#[tokio::test]
async fn cancelled_stateless_run_removes_the_container() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .mode(ShellMode::Stateless)
        .timeout(None)
        .build()
        .unwrap();
    let dropped =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(dropped.is_err(), "the run should still be going");
    let calls = fake.calls();
    let name = container_name_of(&calls[0]);
    let calls = fake.wait_for_call(|c| c == format!("rm -f {name}")).await;
    assert!(calls.contains(&format!("rm -f {name}")), "{calls:?}");
}

#[tokio::test]
async fn finished_stateless_run_does_not_remove_by_name() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake).mode(ShellMode::Stateless).build().unwrap();
    tool.run("echo hi", None).await.unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    let calls = fake.calls();
    assert!(!calls.iter().any(|c| c.starts_with("rm -f")), "{calls:?}");
}

#[tokio::test]
async fn persistent_timeout_reaps_processes_in_the_container() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .container_name("af-reap-box")
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    assert!(tool.run("sleep 30", None).await.unwrap().timed_out);
    let calls = fake.calls();
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("exec af-reap-box sh -c kill -KILL -1")),
        "{calls:?}"
    );
    // The container is kept; the next call gets a fresh shell in it.
    assert_eq!(tool.run("echo again", None).await.unwrap().stdout, "again");
    let calls = fake.calls();
    assert_eq!(
        calls.iter().filter(|c| c.starts_with("run -d")).count(),
        1,
        "{calls:?}"
    );
    assert_eq!(
        calls
            .iter()
            .filter(|c| *c == "exec -i af-reap-box sh")
            .count(),
        2,
        "{calls:?}"
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn persistent_reap_failure_recreates_the_container() {
    let fake = FakeDocker::with_reap_rc(0, 0, 1);
    let tool = builder(&fake)
        .container_name("af-recreate-box")
        .timeout(Some(Duration::from_millis(300)))
        .build()
        .unwrap();
    assert!(tool.run("sleep 30", None).await.unwrap().timed_out);
    assert!(fake.calls().contains(&"rm -f af-recreate-box".to_string()));
    assert_eq!(tool.run("echo again", None).await.unwrap().stdout, "again");
    let calls = fake.calls();
    assert_eq!(
        calls.iter().filter(|c| c.starts_with("run -d")).count(),
        2,
        "{calls:?}"
    );
    tool.close().await.unwrap();
}

#[tokio::test]
async fn cancelled_persistent_command_is_reaped_before_the_next_call() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .container_name("af-cancel-box")
        .timeout(None)
        .build()
        .unwrap();
    tool.start().await.unwrap();
    let dropped =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(dropped.is_err());
    assert_eq!(tool.run("echo next", None).await.unwrap().stdout, "next");
    let calls = fake.calls();
    let reap = calls
        .iter()
        .position(|c| c.starts_with("exec af-cancel-box sh -c kill -KILL -1"))
        .unwrap_or_else(|| panic!("no reap: {calls:?}"));
    let second_shell = calls
        .iter()
        .enumerate()
        .filter(|(_, c)| *c == "exec -i af-cancel-box sh")
        .nth(1)
        .map(|(i, _)| i)
        .unwrap_or_else(|| panic!("no second shell: {calls:?}"));
    assert!(reap < second_shell, "{calls:?}");
    tool.close().await.unwrap();
}

/// Cancelling a persistent call kills the command inside the container
/// right away, not only when (or if) another call comes along: killing the
/// local `docker exec` does not stop it, since the container's init is
/// `sleep infinity`.
#[tokio::test]
async fn cancelled_persistent_command_is_reaped_immediately() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .container_name("af-now-box")
        .timeout(None)
        .build()
        .unwrap();
    tool.start().await.unwrap();
    let dropped =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(dropped.is_err());
    // No further call: the reap must come from the cancellation itself.
    let reap = "exec af-now-box sh -c kill -KILL -1 2>/dev/null; exit 0";
    let calls = fake.wait_for_call(|c| c == reap).await;
    assert!(calls.iter().any(|c| c == reap), "{calls:?}");
    // A caller-chosen container is kept, and the next call still works.
    assert!(!calls.iter().any(|c| c.starts_with("rm")), "{calls:?}");
    assert_eq!(tool.run("echo next", None).await.unwrap().stdout, "next");
    tool.close().await.unwrap();
}

/// When the immediate reap fails, a container whose name the tool generated
/// is removed at once, and the next call starts a new one.
#[tokio::test]
async fn failed_cancel_reap_removes_a_generated_container() {
    let fake = FakeDocker::with_reap_rc(0, 0, 1);
    let tool = builder(&fake).timeout(None).build().unwrap();
    let name = tool.container_name().to_string();
    tool.start().await.unwrap();
    let dropped =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(dropped.is_err());
    let rm = format!("rm -f {name}");
    let calls = fake.wait_for_call(|c| c == rm).await;
    assert!(calls.contains(&rm), "{calls:?}");
    assert_eq!(tool.run("echo next", None).await.unwrap().stdout, "next");
    let calls = fake.calls();
    assert_eq!(
        calls.iter().filter(|c| c.starts_with("run -d")).count(),
        2,
        "the next call must recreate the removed container: {calls:?}"
    );
    tool.close().await.unwrap();
}

/// ... but a caller-chosen name is never removed from the drop path.
#[tokio::test]
async fn failed_cancel_reap_keeps_an_explicit_container() {
    let fake = FakeDocker::with_reap_rc(0, 0, 1);
    let tool = builder(&fake)
        .container_name("af-explicit-box")
        .timeout(None)
        .build()
        .unwrap();
    tool.start().await.unwrap();
    let dropped =
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)).await;
    assert!(dropped.is_err());
    let calls = fake
        .wait_for_call(|c| c.starts_with("exec af-explicit-box sh -c kill -KILL -1"))
        .await;
    assert!(
        calls
            .iter()
            .any(|c| c.starts_with("exec af-explicit-box sh -c kill -KILL -1")),
        "{calls:?}"
    );
    tokio::time::sleep(Duration::from_millis(300)).await;
    let calls = fake.calls();
    assert!(!calls.iter().any(|c| c.starts_with("rm")), "{calls:?}");
}

/// A fake CLI whose `docker run` hangs, so a start can be cancelled before
/// it reports anything (such as a name conflict).
fn slow_docker() -> (TempDir, PathBuf, PathBuf) {
    let dir = TempDir::new("slow-docker");
    let log = dir.path().join("calls.log");
    let binary = dir.path().join("docker");
    std::fs::write(
        &binary,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\ncase \"$1\" in run) exec sleep 30 ;; esac\n",
            log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o755)).unwrap();
    for _ in 0..200 {
        match std::process::Command::new(&binary).arg("version").output() {
            Err(err) if err.raw_os_error() == Some(26) => {
                std::thread::sleep(Duration::from_millis(10))
            }
            _ => break,
        }
    }
    let _ = std::fs::remove_file(&log);
    (dir, binary, log)
}

fn read_log(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

#[tokio::test]
async fn cancelled_container_start_removes_the_generated_container() {
    let (_dir, binary, log) = slow_docker();
    let tool = DockerShellTool::builder()
        .docker_binary(binary.to_string_lossy())
        .build()
        .unwrap();
    let rm = format!("rm -f {}", tool.container_name());
    let dropped = tokio::time::timeout(Duration::from_millis(300), tool.start()).await;
    assert!(dropped.is_err());
    let mut calls = Vec::new();
    for _ in 0..100 {
        calls = read_log(&log);
        if calls.contains(&rm) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(calls.contains(&rm), "{calls:?}");
}

/// A caller-chosen name may already belong to another container; a start
/// cancelled before `docker run` reports the conflict must not remove it.
#[tokio::test]
async fn cancelled_container_start_never_removes_an_explicit_name() {
    let (_dir, binary, log) = slow_docker();
    let tool = DockerShellTool::builder()
        .docker_binary(binary.to_string_lossy())
        .container_name("someone-elses-box")
        .build()
        .unwrap();
    let dropped = tokio::time::timeout(Duration::from_millis(300), tool.start()).await;
    assert!(dropped.is_err());
    // Give a (wrongly) spawned cleanup thread time to show up in the log.
    tokio::time::sleep(Duration::from_millis(500)).await;
    let calls = read_log(&log);
    assert!(calls.iter().any(|c| c.starts_with("run -d")), "{calls:?}");
    assert!(!calls.iter().any(|c| c.starts_with("rm")), "{calls:?}");
}

/// A call queued behind one that is then cancelled must not slip past the
/// container reap: the poison check and recovery sit inside the same
/// serialisation as execution.
#[tokio::test]
async fn queued_call_behind_a_cancelled_one_reaps_first() {
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake)
        .container_name("af-queue-box")
        .timeout(None)
        .build()
        .unwrap();
    tool.start().await.unwrap();
    let (first, second) = tokio::join!(
        tokio::time::timeout(Duration::from_millis(300), tool.run("sleep 30", None)),
        async {
            // Queue up while the first call is still running.
            tokio::time::sleep(Duration::from_millis(50)).await;
            tool.run("echo second", None).await
        }
    );
    assert!(first.is_err(), "the first call should have been cancelled");
    assert_eq!(second.unwrap().stdout, "second");
    let calls = fake.calls();
    let reap = calls
        .iter()
        .position(|c| c.starts_with("exec af-queue-box sh -c kill -KILL -1"))
        .unwrap_or_else(|| panic!("the queued call ran without a reap: {calls:?}"));
    let second_shell = calls
        .iter()
        .enumerate()
        .filter(|(_, c)| *c == "exec -i af-queue-box sh")
        .nth(1)
        .map(|(i, _)| i)
        .unwrap_or_else(|| panic!("no second shell: {calls:?}"));
    assert!(reap < second_shell, "{calls:?}");
    tool.close().await.unwrap();
}

#[tokio::test]
async fn docker_reports_its_own_shell_family_and_system() {
    use agent_framework_tools::shell::ShellFamily;
    let fake = FakeDocker::new(0, 0);
    let tool = builder(&fake).build().unwrap();
    assert_eq!(tool.shell_family(), Some(ShellFamily::Posix));
    assert_eq!(
        tool.os_description().as_deref(),
        Some("a Linux container (alpine:3)")
    );
    let pwsh = builder(&fake).shell("pwsh").build().unwrap();
    assert_eq!(pwsh.shell_family(), Some(ShellFamily::PowerShell));
}
