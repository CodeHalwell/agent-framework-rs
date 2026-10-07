//! Persistent shell session.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use base64::Engine as _;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;

use super::process::{apply_env, isolate_process_group, kill_process_tree, KILL_GRACE};
use super::resolve::is_powershell;
use super::truncate::{truncate_head_tail, truncate_text_head_tail};
use super::types::{ShellError, ShellResult};

const READ_CHUNK: usize = 64 * 1024;
/// How long to let late stderr arrive after the sentinel.
const STDERR_QUIESCENCE: Duration = Duration::from_millis(50);
/// Exit code reported when a timed-out command could not be recovered.
const TIMEOUT_EXIT_CODE: i32 = 124;

#[derive(Default)]
struct Buffers {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    stdout_closed: bool,
}

#[derive(Default)]
struct Shared {
    buffers: StdMutex<Buffers>,
    stdout_changed: Notify,
}

impl Shared {
    fn lock(&self) -> std::sync::MutexGuard<'_, Buffers> {
        // A reader that panicked mid-push leaves valid bytes behind; keep going.
        self.buffers.lock().unwrap_or_else(|e| e.into_inner())
    }
}

struct Live {
    child: Child,
    stdin: ChildStdin,
    shared: Arc<Shared>,
    readers: Vec<JoinHandle<()>>,
}

impl Drop for Live {
    fn drop(&mut self) {
        for reader in &self.readers {
            reader.abort();
        }
        // `kill_on_drop` reaches only the shell; take its group with it.
        #[cfg(unix)]
        if matches!(self.child.try_wait(), Ok(None)) {
            if let Some(pid) = self.child.id() {
                super::process::signal_group(pid, libc::SIGKILL);
            }
        }
    }
}

enum WaitError {
    /// The shell closed stdout before printing the sentinel.
    Closed,
    /// More than the hard cap arrived without a sentinel.
    Overflow,
}

/// A [`ShellSession`] runs one long-lived shell and feeds it commands on
/// stdin, each followed by a **sentinel** line that reports the exit status.
/// Reading stdout until the sentinel appears marks where one command's output
/// ends without job control or a PTY, so the same code works for bash, sh and
/// PowerShell.
///
/// # Single owner
///
/// A session belongs to one conversation, i.e. one user. Its shell carries
/// state (working directory, exported variables, history, background jobs)
/// that every later command can see, and one stdin/stdout pipe serialises
/// every call. Nothing isolates one caller from another: create one session
/// per agent session, and close it when that session ends.
///
/// # Protocol notes (ported from upstream)
///
/// * The sentinel carries a per-session random tag plus a per-command random
///   suffix, so a command that prints a look-alike cannot end its own output
///   early.
/// * POSIX: the command runs in a brace group so its status is captured even
///   under `set -e`; the caller's `errexit` setting is saved and restored.
/// * PowerShell reads a whole script before running it, so the command is
///   base64-encoded and run through `Invoke-Expression` on one line. Output is
///   formatted inside the `try` so table output is not overtaken by the
///   sentinel, and the exit code combines `$LASTEXITCODE` (snapshotted, not
///   cleared) with `$?` and caught exceptions.
/// * Two reader tasks drain stdout and stderr for the life of the session.
///   Each command reads forward from offsets taken before it was written, and
///   waits briefly after the sentinel so late stderr is not blamed on the
///   next command.
pub struct ShellSession {
    argv: Vec<String>,
    workdir: Option<PathBuf>,
    env: Option<HashMap<String, String>>,
    max_output_bytes: usize,
    tag: String,
    is_pwsh: bool,
    live: Mutex<Option<Live>>,
    run_lock: Mutex<()>,
    /// Set when a `run` future was dropped mid-command: the shell may still
    /// be running it, so its next output cannot be trusted.
    poisoned: AtomicBool,
}

/// Marks the session poisoned unless the command it guards completed.
struct InFlight<'a> {
    poisoned: &'a AtomicBool,
    done: bool,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.poisoned.store(true, Ordering::SeqCst);
        }
    }
}

impl std::fmt::Debug for ShellSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShellSession")
            .field("argv", &self.argv)
            .field("workdir", &self.workdir)
            .field("max_output_bytes", &self.max_output_bytes)
            .finish_non_exhaustive()
    }
}

fn random_hex(len: usize) -> String {
    let mut s = uuid::Uuid::new_v4().simple().to_string();
    s.truncate(len);
    s
}

impl ShellSession {
    /// A session that will run `argv` (a shell reading commands from stdin)
    /// once started. `env` of `None` inherits the parent's environment.
    pub fn new(
        argv: Vec<String>,
        workdir: Option<PathBuf>,
        env: Option<HashMap<String, String>>,
        max_output_bytes: usize,
    ) -> Self {
        let is_pwsh = is_powershell(&argv);
        Self {
            argv,
            workdir,
            env,
            max_output_bytes: max_output_bytes.max(1),
            tag: random_hex(16),
            is_pwsh,
            live: Mutex::new(None),
            run_lock: Mutex::new(()),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Start the shell if it is not running. A session that was closed, or
    /// whose shell exited, starts a fresh one.
    pub async fn start(&self) -> Result<(), ShellError> {
        let mut live = self.live.lock().await;
        // A cancelled command may still be running: replace the shell rather
        // than attribute its late output to the next command.
        let poisoned = self.poisoned.swap(false, Ordering::SeqCst);
        if let Some(current) = live.as_mut() {
            if !poisoned && matches!(current.child.try_wait(), Ok(None)) {
                return Ok(());
            }
        }
        // Dropping the old shell kills its process group.
        *live = None;

        let (program, args) = self
            .argv
            .split_first()
            .ok_or_else(|| ShellError::Config("shell argv is empty".into()))?;
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        apply_env(&mut cmd, self.workdir.as_ref(), self.env.as_ref());
        isolate_process_group(&mut cmd);
        let mut child = cmd
            .spawn()
            .map_err(|e| ShellError::io("failed to start shell session", e))?;

        let stdin = child.stdin.take().expect("stdin piped");
        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");
        let shared = Arc::new(Shared::default());
        let readers = vec![
            spawn_reader(stdout, shared.clone(), true),
            spawn_reader(stderr, shared.clone(), false),
        ];
        let mut started = Live {
            child,
            stdin,
            shared,
            readers,
        };
        if self.is_pwsh {
            // Make PowerShell emit UTF-8 and turn cmdlet errors into
            // exceptions the command wrapper can catch.
            write_all(
                &mut started.stdin,
                "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false);$ErrorActionPreference = 'Stop'\n",
            )
            .await?;
        }
        *live = Some(started);
        Ok(())
    }

    /// Stop the shell: ask it to `exit`, then kill its process tree if it
    /// has not gone within the grace period. Idempotent.
    pub async fn close(&self) {
        let taken = self.live.lock().await.take();
        if let Some(mut live) = taken {
            if matches!(live.child.try_wait(), Ok(None)) {
                let _ = live.stdin.write_all(b"exit\n").await;
                let _ = live.stdin.flush().await;
                let _ = live.stdin.shutdown().await;
                if tokio::time::timeout(KILL_GRACE, live.child.wait())
                    .await
                    .is_err()
                {
                    kill_process_tree(&mut live.child, KILL_GRACE).await;
                }
            }
            // Dropping `live` aborts the readers.
        }
    }

    /// Run `command` and return its result. Calls are serialised.
    ///
    /// On timeout the running command is interrupted (`SIGINT` to the
    /// session's process group on Unix). If the shell recovers within the
    /// grace period the result is returned with `timed_out` set; otherwise
    /// the session is torn down, so the next call starts a fresh shell, and
    /// the result reports exit code 124.
    pub async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        let _serial = self.run_lock.lock().await;
        self.start().await?;
        let mut in_flight = InFlight {
            poisoned: &self.poisoned,
            done: false,
        };
        let result = self.run_started(command, timeout).await;
        in_flight.done = true;
        result
    }

    async fn run_started(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        let sentinel = format!("__AF_END_{}_{}__", self.tag, random_hex(8));
        let script = self.build_script(command, &sentinel);

        let (shared, pid) = {
            let mut guard = self.live.lock().await;
            let live = guard
                .as_mut()
                .ok_or_else(|| ShellError::Execution("shell session is not running".into()))?;
            let shared = live.shared.clone();
            let pid = live.child.id();
            // Only output produced after the command is written belongs to it.
            {
                let mut bufs = shared.lock();
                bufs.stdout.clear();
                bufs.stderr.clear();
            }
            if let Err(err) = write_all(&mut live.stdin, &script).await {
                drop(guard);
                self.close().await;
                return Err(ShellError::Execution(format!(
                    "persistent shell session is no longer alive: {err}"
                )));
            }
            (shared, pid)
        };

        let started = Instant::now();
        let needle = sentinel.into_bytes();
        let hard_cap = self.max_output_bytes.saturating_mul(4);

        let first = match timeout {
            Some(limit) => {
                tokio::time::timeout(limit, wait_for_sentinel(&shared, &needle, hard_cap))
                    .await
                    .ok()
            }
            None => Some(wait_for_sentinel(&shared, &needle, hard_cap).await),
        };

        let (found, timed_out) = match first {
            Some(Ok(found)) => (found, false),
            Some(Err(WaitError::Overflow)) => {
                // Runaway output with no sentinel: interrupt and restart.
                interrupt(pid);
                self.close().await;
                let bufs = shared.lock();
                let end = bufs.stdout.len().min(hard_cap);
                let (stdout, _) = truncate_head_tail(&bufs.stdout[..end], self.max_output_bytes);
                let (stderr, _) = truncate_head_tail(&bufs.stderr, self.max_output_bytes);
                return Ok(ShellResult {
                    stdout,
                    stderr,
                    exit_code: -1,
                    duration: started.elapsed(),
                    truncated: true,
                    timed_out: false,
                });
            }
            Some(Err(WaitError::Closed)) => {
                self.close().await;
                return Err(ShellError::Execution(
                    "shell closed stdout before emitting sentinel".into(),
                ));
            }
            None => {
                interrupt(pid);
                match tokio::time::timeout(
                    KILL_GRACE,
                    wait_for_sentinel(&shared, &needle, hard_cap),
                )
                .await
                {
                    Ok(Ok(found)) => (found, true),
                    _ => {
                        // Unrecoverable: tear down so the next call gets a
                        // fresh shell.
                        self.close().await;
                        let bufs = shared.lock();
                        let (stdout, out_t) =
                            truncate_head_tail(&bufs.stdout, self.max_output_bytes);
                        let (stderr, err_t) =
                            truncate_head_tail(&bufs.stderr, self.max_output_bytes);
                        return Ok(ShellResult {
                            stdout,
                            stderr,
                            exit_code: TIMEOUT_EXIT_CODE,
                            duration: started.elapsed(),
                            truncated: out_t || err_t,
                            timed_out: true,
                        });
                    }
                }
            }
        };

        tokio::time::sleep(STDERR_QUIESCENCE).await;
        let duration = started.elapsed();
        let (sentinel_idx, exit_code) = found;
        let mut bufs = shared.lock();
        let stdout_text = String::from_utf8_lossy(&bufs.stdout[..sentinel_idx]);
        let stdout_text = stdout_text.trim_end_matches(['\r', '\n']);
        let stderr_text = String::from_utf8_lossy(&bufs.stderr);
        let (stdout, out_t) = truncate_text_head_tail(stdout_text, self.max_output_bytes);
        let (stderr, err_t) = truncate_text_head_tail(&stderr_text, self.max_output_bytes);
        // Everything needed has been copied; keep memory bounded across
        // many commands.
        bufs.stdout.clear();
        bufs.stderr.clear();

        Ok(ShellResult {
            stdout,
            stderr,
            exit_code,
            duration,
            truncated: out_t || err_t,
            timed_out,
        })
    }

    fn build_script(&self, command: &str, sentinel: &str) -> String {
        if self.is_pwsh {
            let encoded = base64::engine::general_purpose::STANDARD.encode(command.as_bytes());
            return format!(
                "& {{ $__af_rc = 0; $__af_last = $LASTEXITCODE; try {{   \
$__af_cmd = [System.Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{encoded}'));   \
Invoke-Expression $__af_cmd | Out-Default;   $__af_ok = $?;   \
if ($LASTEXITCODE -ne $__af_last) {{ $__af_rc = $LASTEXITCODE }}   \
elseif (-not $__af_ok) {{ $__af_rc = 1 }} }} catch {{   \
[Console]::Error.WriteLine($_.ToString());   $__af_rc = 1 }} finally {{   \
[Console]::WriteLine('{sentinel}_' + $__af_rc);   [Console]::Out.Flush() }} }}\n"
            );
        }
        format!(
            "__af_e=$-; set +e; {{ {command}\n}}; __af_rc=$?; case \"$__af_e\" in *e*) set -e;; esac; printf '\\n{sentinel}_%s\\n' \"$__af_rc\"\n"
        )
    }
}

async fn write_all(stdin: &mut ChildStdin, text: &str) -> Result<(), ShellError> {
    stdin
        .write_all(text.as_bytes())
        .await
        .map_err(|e| ShellError::io("failed to write to shell", e))?;
    stdin
        .flush()
        .await
        .map_err(|e| ShellError::io("failed to write to shell", e))
}

fn spawn_reader<R>(mut stream: R, shared: Arc<Shared>, is_stdout: bool) -> JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut chunk = vec![0u8; READ_CHUNK];
        loop {
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => {
                    if is_stdout {
                        shared.lock().stdout_closed = true;
                        shared.stdout_changed.notify_waiters();
                    }
                    return;
                }
                Ok(n) => {
                    {
                        let mut bufs = shared.lock();
                        if is_stdout {
                            bufs.stdout.extend_from_slice(&chunk[..n]);
                        } else {
                            bufs.stderr.extend_from_slice(&chunk[..n]);
                        }
                    }
                    if is_stdout {
                        shared.stdout_changed.notify_waiters();
                    }
                }
            }
        }
    })
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Wait until the sentinel and its exit-code line have arrived. Returns the
/// sentinel's offset in stdout and the parsed exit code.
async fn wait_for_sentinel(
    shared: &Shared,
    needle: &[u8],
    hard_cap: usize,
) -> Result<(usize, i32), WaitError> {
    let mut found_at: Option<usize> = None;
    let mut tail_deadline: Option<Instant> = None;
    loop {
        // Register interest before checking, so a write between the check
        // and the wait is not missed.
        let changed = shared.stdout_changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        {
            let bufs = shared.lock();
            if found_at.is_none() {
                found_at = find(&bufs.stdout, needle);
            }
            if let Some(idx) = found_at {
                let after = &bufs.stdout[idx + needle.len()..];
                let deadline =
                    *tail_deadline.get_or_insert_with(|| Instant::now() + Duration::from_secs(1));
                // Wait briefly for the exit-code digits and their newline.
                if after.contains(&b'\n') || bufs.stdout_closed || Instant::now() >= deadline {
                    return Ok((idx, parse_rc(after)));
                }
            } else {
                if bufs.stdout_closed {
                    return Err(WaitError::Closed);
                }
                if bufs.stdout.len() > hard_cap {
                    return Err(WaitError::Overflow);
                }
            }
        }
        let _ = tokio::time::timeout(Duration::from_millis(100), changed).await;
    }
}

/// Interrupt the command running in the session's process group.
fn interrupt(pid: Option<u32>) {
    #[cfg(unix)]
    if let Some(pid) = pid {
        super::process::signal_group(pid, libc::SIGINT);
    }
    // Windows: upstream sends CTRL_BREAK_EVENT, which needs a console API
    // binding this crate does not take on. Without it the grace wait expires
    // and the session is torn down, which also stops the command.
    #[cfg(not(unix))]
    let _ = pid;
}

/// Parse the `_<digits>` that follows the sentinel; `-1` when absent or
/// malformed.
pub(crate) fn parse_rc(after: &[u8]) -> i32 {
    let Some(rest) = after.strip_prefix(b"_") else {
        return -1;
    };
    let digits: Vec<u8> = rest
        .iter()
        .copied()
        .take_while(|b| b.is_ascii_digit() || *b == b'-')
        .collect();
    std::str::from_utf8(&digits)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rc_cases() {
        assert_eq!(parse_rc(b"_0\n"), 0);
        assert_eq!(parse_rc(b"_127\n"), 127);
        assert_eq!(parse_rc(b"_-1\n"), -1);
        assert_eq!(parse_rc(b"_42\r\n"), 42);
        assert_eq!(parse_rc(b"_5"), 5);
        assert_eq!(parse_rc(b"42\n"), -1);
        assert_eq!(parse_rc(b""), -1);
        assert_eq!(parse_rc(b"_\n"), -1);
        assert_eq!(parse_rc(b"_abc\n"), -1);
        assert_eq!(parse_rc(b"_7 extra junk\n"), 7);
        assert_eq!(parse_rc(b"_12x34\n"), 12);
    }

    #[test]
    fn posix_script_wraps_command_and_restores_errexit() {
        let session = ShellSession::new(vec!["/bin/bash".into()], None, None, 1024);
        let script = session.build_script("echo hi", "__AF_END_x__");
        assert_eq!(
            script,
            "__af_e=$-; set +e; { echo hi\n}; __af_rc=$?; case \"$__af_e\" in *e*) set -e;; esac; printf '\\n__AF_END_x___%s\\n' \"$__af_rc\"\n"
        );
    }

    #[test]
    fn powershell_script_base64_encodes_the_command() {
        let session = ShellSession::new(vec!["pwsh".into()], None, None, 1024);
        let script = session.build_script("Write-Output 'café'", "__AF_END_x__");
        let encoded = base64::engine::general_purpose::STANDARD.encode("Write-Output 'café'");
        assert!(script.contains(&format!("FromBase64String('{encoded}')")));
        assert!(!script.contains("café"));
        assert!(script.contains("[Console]::WriteLine('__AF_END_x___' + $__af_rc)"));
        assert!(script.ends_with("}\n"));
        assert_eq!(script.lines().count(), 1);
    }

    #[test]
    fn sentinels_are_unique_per_session() {
        let a = ShellSession::new(vec!["sh".into()], None, None, 1);
        let b = ShellSession::new(vec!["sh".into()], None, None, 1);
        assert_ne!(a.tag, b.tag);
        assert_eq!(a.tag.len(), 16);
    }
}
