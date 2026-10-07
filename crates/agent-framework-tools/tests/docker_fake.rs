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
  exec) exec /bin/sh ;;
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
    let name = calls[0]
        .split_whitespace()
        .skip_while(|t| *t != "--name")
        .nth(1)
        .unwrap()
        .to_string();
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
