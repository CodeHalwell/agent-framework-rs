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

use super::process::{
    apply_env, isolate_process_group, kill_process_tree, kill_tree_now, KILL_GRACE,
};
use super::resolve::is_powershell;
use super::truncate::{truncate_text_head_tail, HeadTailBuffer};
use super::types::{ShellError, ShellResult};

const READ_CHUNK: usize = 64 * 1024;
/// How long to let late stderr arrive after the sentinel.
const STDERR_QUIESCENCE: Duration = Duration::from_millis(50);
/// Exit code reported when a timed-out command could not be recovered.
const TIMEOUT_EXIT_CODE: i32 = 124;
/// Extra stdout kept beyond the tail budget, so a sentinel split across two
/// reads is still found after older bytes were dropped.
const WINDOW_SLACK: usize = 1024;
/// Bytes kept after the sentinel: enough for its `_<exit code>` line.
const RC_SLACK: usize = 64;

/// stdout of the current command in bounded storage: the first
/// `keep / 2` bytes, then a rolling window of the most recent bytes, with
/// whatever fell between them only counted. The sentinel is looked for as
/// bytes arrive, before anything is dropped, so a command may print any
/// amount and still finish normally with head/tail-truncated output.
struct StdoutCapture {
    /// `max_output_bytes`.
    keep: usize,
    buf: Vec<u8>,
    /// Bytes dropped from between the head and the window.
    elided: usize,
    needle: Option<Vec<u8>>,
    /// Offset of the sentinel in `buf`, once seen. Storage stops shortly
    /// after it.
    found: Option<usize>,
}

impl StdoutCapture {
    fn new(keep: usize) -> Self {
        Self {
            keep: keep.max(1),
            buf: Vec::new(),
            elided: 0,
            needle: None,
            found: None,
        }
    }

    fn head_cap(&self) -> usize {
        self.keep / 2
    }

    fn window_cap(&self) -> usize {
        self.keep - self.keep / 2 + WINDOW_SLACK
    }

    /// Forget everything and look for `needle` from now on.
    fn reset(&mut self, needle: Option<Vec<u8>>) {
        self.buf = Vec::new();
        self.elided = 0;
        self.needle = needle;
        self.found = None;
    }

    fn push(&mut self, chunk: &[u8]) {
        let needle_len = self.needle.as_ref().map_or(0, Vec::len);
        if let Some(idx) = self.found {
            // Past the sentinel only its exit-code line matters; anything
            // else (a background job) is drained, not stored.
            let limit = idx + needle_len + RC_SLACK;
            if self.buf.len() < limit {
                let take = (limit - self.buf.len()).min(chunk.len());
                self.buf.extend_from_slice(&chunk[..take]);
            }
            return;
        }
        // A sentinel may straddle the previous read and this one.
        let from = self.buf.len().saturating_sub(needle_len.saturating_sub(1));
        self.buf.extend_from_slice(chunk);
        if let Some(needle) = &self.needle {
            if let Some(pos) = find(&self.buf[from..], needle) {
                let idx = from + pos;
                self.found = Some(idx);
                self.buf.truncate(idx + needle_len + RC_SLACK);
                return;
            }
        }
        let max = self.head_cap() + self.window_cap();
        if self.buf.len() > max {
            let excess = self.buf.len() - max;
            let head = self.head_cap();
            self.buf.drain(head..head + excess);
            self.elided += excess;
        }
    }

    /// The text of `buf[..end]` (optionally without trailing newlines),
    /// head/tail truncated to `keep` bytes.
    fn render(&self, end: usize, trim: bool) -> (String, bool) {
        let mut data = &self.buf[..end.min(self.buf.len())];
        if trim {
            while let [rest @ .., b'\r' | b'\n'] = data {
                data = rest;
            }
        }
        if self.elided == 0 {
            return truncate_text_head_tail(&String::from_utf8_lossy(data), self.keep);
        }
        let head_cap = self.head_cap().min(data.len());
        let tail_cap = self.keep - self.keep / 2;
        let tail_start = data.len().saturating_sub(tail_cap).max(head_cap);
        let kept = head_cap + (data.len() - tail_start);
        let dropped = data.len() + self.elided - kept;
        (
            format!(
                "{}\n[... truncated {dropped} bytes ...]\n{}",
                String::from_utf8_lossy(&data[..head_cap]),
                String::from_utf8_lossy(&data[tail_start..])
            ),
            true,
        )
    }
}

/// Output captured since the current command was written. Both streams are
/// bounded, so neither a command printing without end nor a background job
/// writing between commands can grow memory without limit.
struct Buffers {
    stdout: StdoutCapture,
    /// stderr in head/tail storage of `max_output_bytes`.
    stderr: HeadTailBuffer,
    stdout_closed: bool,
}

struct Shared {
    buffers: StdMutex<Buffers>,
    stdout_changed: Notify,
    stderr_cap: usize,
}

impl Shared {
    fn new(max_output_bytes: usize) -> Self {
        Self {
            buffers: StdMutex::new(Buffers {
                stdout: StdoutCapture::new(max_output_bytes),
                stderr: HeadTailBuffer::new(max_output_bytes),
                stdout_closed: false,
            }),
            stdout_changed: Notify::new(),
            stderr_cap: max_output_bytes,
        }
    }

    /// Drop everything captured so far; stdout looks for `needle` next.
    fn clear(&self, bufs: &mut Buffers, needle: Option<Vec<u8>>) {
        bufs.stdout.reset(needle);
        bufs.stderr = HeadTailBuffer::new(self.stderr_cap);
    }

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
    /// The shell's pid (and, on Unix, its process group id), kept because
    /// `child.id()` is `None` once the shell has been reaped.
    pid: Option<u32>,
}

impl Drop for Live {
    fn drop(&mut self) {
        for reader in &self.readers {
            reader.abort();
        }
        // `kill_on_drop` reaches only the shell; take its process tree with
        // it.
        let Some(pid) = self.pid else {
            return;
        };
        // On Unix the shell leads its own process group, which outlives it:
        // a background job (`sleep 3600 &`) is still in the group after the
        // shell exited normally, so the group is killed either way.
        #[cfg(unix)]
        kill_tree_now(pid);
        // Elsewhere the tree is found from the shell, which must be alive to
        // anchor it.
        #[cfg(not(unix))]
        if matches!(self.child.try_wait(), Ok(None)) {
            kill_tree_now(pid);
        }
    }
}

/// The shell closed stdout before printing the sentinel.
struct Closed;

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

/// Guards a command in flight. If the `run` future is dropped before the
/// command completed, the session is marked poisoned and its shell is torn
/// down on the spot: otherwise the cancelled command (say `sleep 3600`)
/// would keep running until the next call or until the tool is dropped.
struct InFlight<'a> {
    poisoned: &'a AtomicBool,
    live: &'a Mutex<Option<Live>>,
    /// The shell's pid, for when `live` is briefly held elsewhere.
    pid: Option<u32>,
    done: bool,
}

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        // Stays set after the shell is gone, so a wrapper (Docker) knows to
        // reap what the command left behind outside this process tree.
        self.poisoned.store(true, Ordering::SeqCst);
        match self.live.try_lock() {
            // Dropping `Live` kills the shell's process tree and aborts the
            // readers.
            Ok(mut live) => drop(live.take()),
            Err(_) => {
                if let Some(pid) = self.pid {
                    kill_tree_now(pid);
                }
            }
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
        let shared = Arc::new(Shared::new(self.max_output_bytes));
        let readers = vec![
            spawn_reader(stdout, shared.clone(), true),
            spawn_reader(stderr, shared.clone(), false),
        ];
        let mut started = Live {
            pid: child.id(),
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

    /// Whether a shell is currently running. `false` after [`close`](Self::close)
    /// and after an unrecoverable timeout tore the shell down.
    pub(crate) async fn is_live(&self) -> bool {
        match self.live.lock().await.as_mut() {
            Some(live) => matches!(live.child.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// Whether a `run` was cancelled mid-command, so the shell may still be
    /// running it.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::SeqCst)
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
            // Dropping `live` aborts the readers and, on Unix, kills what is
            // left of the shell's process group (background jobs survive a
            // normal `exit`).
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
        let pid = self
            .live
            .lock()
            .await
            .as_ref()
            .and_then(|live| live.child.id());
        let mut in_flight = InFlight {
            poisoned: &self.poisoned,
            live: &self.live,
            pid,
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
        let needle = sentinel.into_bytes();

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
                shared.clear(&mut bufs, Some(needle));
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

        // Output is bounded however much the command prints, so there is no
        // separate output limit: a command that never stops printing is
        // ended by the timeout like any other long-running command.
        let first = match timeout {
            Some(limit) => tokio::time::timeout(limit, wait_for_sentinel(&shared))
                .await
                .ok(),
            None => Some(wait_for_sentinel(&shared).await),
        };

        let (found, timed_out) = match first {
            Some(Ok(found)) => (found, false),
            Some(Err(Closed)) => {
                self.close().await;
                return Err(ShellError::Execution(
                    "shell closed stdout before emitting sentinel".into(),
                ));
            }
            None => {
                interrupt(pid);
                match tokio::time::timeout(KILL_GRACE, wait_for_sentinel(&shared)).await {
                    Ok(Ok(found)) => (found, true),
                    _ => {
                        // Unrecoverable: tear down so the next call gets a
                        // fresh shell.
                        self.close().await;
                        let mut bufs = shared.lock();
                        let (stdout, out_t) = bufs.stdout.render(usize::MAX, false);
                        let (stderr, err_t) = take_stderr(&shared, &mut bufs);
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
        let (stdout, out_t) = bufs.stdout.render(sentinel_idx, true);
        let (stderr, err_t) = take_stderr(&shared, &mut bufs);
        // Everything needed has been copied; release it.
        shared.clear(&mut bufs, None);

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
                            bufs.stdout.push(&chunk[..n]);
                        } else {
                            bufs.stderr.push(&chunk[..n]);
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

/// Take the captured stderr as head/tail-truncated text, leaving an empty
/// buffer behind.
fn take_stderr(shared: &Shared, bufs: &mut Buffers) -> (String, bool) {
    std::mem::replace(&mut bufs.stderr, HeadTailBuffer::new(shared.stderr_cap)).finish()
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Wait until the sentinel and its exit-code line have arrived. Returns the
/// sentinel's offset in the captured stdout and the parsed exit code.
async fn wait_for_sentinel(shared: &Shared) -> Result<(usize, i32), Closed> {
    let mut tail_deadline: Option<Instant> = None;
    loop {
        // Register interest before checking, so a write between the check
        // and the wait is not missed.
        let changed = shared.stdout_changed.notified();
        tokio::pin!(changed);
        changed.as_mut().enable();
        {
            let bufs = shared.lock();
            let capture = &bufs.stdout;
            if let Some(idx) = capture.found {
                let needle_len = capture.needle.as_ref().map_or(0, Vec::len);
                let after = &capture.buf[idx + needle_len..];
                let deadline =
                    *tail_deadline.get_or_insert_with(|| Instant::now() + Duration::from_secs(1));
                // Wait briefly for the exit-code digits and their newline.
                if after.contains(&b'\n') || bufs.stdout_closed || Instant::now() >= deadline {
                    return Ok((idx, parse_rc(after)));
                }
            } else if bufs.stdout_closed {
                return Err(Closed);
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

    #[tokio::test]
    async fn readers_bound_both_streams() {
        use tokio::io::AsyncReadExt as _;
        let shared = Arc::new(Shared::new(1024));
        let flood = 10 * 1024 * 1024;
        let out = spawn_reader(tokio::io::repeat(b'o').take(flood), shared.clone(), true);
        let err = spawn_reader(tokio::io::repeat(b'e').take(flood), shared.clone(), false);
        out.await.unwrap();
        err.await.unwrap();
        let mut bufs = shared.lock();
        assert!(
            bufs.stdout.buf.len() <= 1024 + WINDOW_SLACK,
            "{}",
            bufs.stdout.buf.len()
        );
        let (stdout, truncated) = bufs.stdout.render(usize::MAX, false);
        assert!(truncated);
        assert!(stdout.contains(&format!("truncated {} bytes", flood - 1024)));
        let (stderr, truncated) = take_stderr(&shared, &mut bufs);
        assert!(truncated);
        assert!(stderr.len() < 1024 + 64, "{}", stderr.len());
        assert!(stderr.contains(&format!("truncated {} bytes", flood - 1024)));
    }

    /// However much a command prints, the sentinel is found (even split
    /// across reads after older bytes were dropped) and the output matches
    /// one-shot head/tail truncation of everything before it.
    #[test]
    fn capture_finds_the_sentinel_after_dropping_output() {
        let needle = b"__AF_END_tag_1234__".to_vec();
        let body: Vec<u8> = (0..200_000u32).map(|i| b'a' + (i % 26) as u8).collect();
        let mut stream = body.clone();
        stream.extend_from_slice(b"\n");
        stream.extend_from_slice(&needle);
        stream.extend_from_slice(b"_7\nlate background output");
        for keep in [1usize, 2, 7, 100, 4096] {
            for chunk in [1usize, 5, 13, 4096, 65536] {
                let mut capture = StdoutCapture::new(keep);
                capture.reset(Some(needle.clone()));
                for piece in stream.chunks(chunk) {
                    capture.push(piece);
                }
                assert!(capture.buf.len() <= keep + WINDOW_SLACK + chunk + needle.len() + RC_SLACK);
                let idx = capture.found.expect("sentinel found");
                assert_eq!(parse_rc(&capture.buf[idx + needle.len()..]), 7);
                assert_eq!(
                    capture.render(idx, true),
                    super::super::truncate::truncate_head_tail(&body, keep),
                    "keep {keep} chunk {chunk}"
                );
            }
        }
    }

    #[test]
    fn sentinels_are_unique_per_session() {
        let a = ShellSession::new(vec!["sh".into()], None, None, 1);
        let b = ShellSession::new(vec!["sh".into()], None, None, 1);
        assert_ne!(a.tag, b.tag);
        assert_eq!(a.tag.len(), 16);
    }
}
