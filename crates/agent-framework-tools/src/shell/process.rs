//! Process plumbing: process-group isolation, tree termination and the
//! stateless (one process per command) runner.
//!
//! Every shell is started in its own process group (a new session group on
//! Unix, `CREATE_NEW_PROCESS_GROUP` on Windows) so a timeout can terminate
//! the command *and* everything it spawned (`make`, watchers, background
//! jobs). Leaving those running would defeat the timeout.
//!
//! On Unix the group gets `SIGTERM`, a grace period, then `SIGKILL`. A
//! descendant that moved itself into another process group (`setsid`) escapes
//! this; upstream's `psutil` walk of the process tree catches that case and
//! this port does not. On Windows `taskkill /T /F` is run by absolute path
//! (so a modified `PATH` cannot substitute another binary), then the shell
//! itself is killed. A stateless command on Windows also runs in a Job
//! Object, which plays the role of the Unix process group once the shell
//! has exited and `taskkill` can no longer walk its tree.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::{ExitStatus, Stdio};
use std::time::{Duration, Instant};

use tokio::io::AsyncReadExt;
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

use super::truncate::HeadTailBuffer;
use super::types::{ShellError, ShellResult};

/// Grace period between asking a process tree to stop and killing it.
pub(crate) const KILL_GRACE: Duration = Duration::from_secs(2);
const READ_CHUNK: usize = 64 * 1024;

/// Put the child in its own process group.
pub(crate) fn isolate_process_group(cmd: &mut Command) {
    #[cfg(unix)]
    {
        cmd.process_group(0);
    }
    #[cfg(windows)]
    {
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        cmd.creation_flags(CREATE_NEW_PROCESS_GROUP);
    }
}

/// Apply the working directory and environment to `cmd`.
///
/// `env` of `None` inherits the parent's environment; `Some` replaces it.
pub(crate) fn apply_env(
    cmd: &mut Command,
    workdir: Option<&PathBuf>,
    env: Option<&HashMap<String, String>>,
) {
    if let Some(dir) = workdir {
        cmd.current_dir(dir);
    }
    if let Some(env) = env {
        cmd.env_clear();
        cmd.envs(env);
    }
}

/// Map an exit status the way Python's `returncode` does: the exit code, or
/// the negated signal number for a process killed by a signal.
pub(crate) fn exit_code(status: ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    -1
}

/// Send `signal` to the process group led by `pid`. Best effort.
#[cfg(unix)]
pub(crate) fn signal_group(pid: u32, signal: libc::c_int) {
    let Ok(pid) = libc::pid_t::try_from(pid) else {
        return;
    };
    if pid <= 0 {
        // Never let a bad pid turn into kill(0)/kill(-1), which would hit
        // our own group or every process we may signal.
        return;
    }
    // SAFETY: killpg only sends a signal; with a positive pid it targets
    // exactly that process group and touches no memory.
    unsafe {
        libc::killpg(pid, signal);
    }
}

#[cfg(windows)]
fn taskkill_path() -> PathBuf {
    let root = std::env::var_os("SystemRoot")
        .or_else(|| std::env::var_os("SYSTEMROOT"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    let candidate = root.join("System32").join("taskkill.exe");
    if candidate.is_file() {
        candidate
    } else {
        PathBuf::from("taskkill")
    }
}

/// Terminate `child` and its process group. Best effort, never fails.
pub(crate) async fn kill_process_tree(child: &mut Child, grace: Duration) {
    if matches!(child.try_wait(), Ok(Some(_))) {
        return;
    }
    let Some(pid) = child.id() else {
        return;
    };
    #[cfg(unix)]
    {
        signal_group(pid, libc::SIGTERM);
        if tokio::time::timeout(grace, child.wait()).await.is_ok() {
            // The leader is gone; make sure no straggler in its group is
            // left behind before returning.
            signal_group(pid, libc::SIGKILL);
            return;
        }
        signal_group(pid, libc::SIGKILL);
        let _ = child.start_kill();
        let _ = tokio::time::timeout(grace, child.wait()).await;
    }
    #[cfg(windows)]
    {
        if let Ok(mut killer) = Command::new(taskkill_path())
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
        {
            let _ = tokio::time::timeout(grace, killer.wait()).await;
        }
        let _ = child.start_kill();
        let _ = tokio::time::timeout(grace, child.wait()).await;
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = pid;
        let _ = child.start_kill();
        let _ = tokio::time::timeout(grace, child.wait()).await;
    }
}

/// Kill `pid` and everything it spawned, synchronously, for drop paths that
/// cannot await. Unix: `SIGKILL` to its process group. Windows: a blocking
/// `taskkill /T /F`, which walks the tree from `pid`, so it must run while
/// that process is still alive.
pub(crate) fn kill_tree_now(pid: u32) {
    #[cfg(unix)]
    signal_group(pid, libc::SIGKILL);
    #[cfg(windows)]
    {
        let _ = std::process::Command::new(taskkill_path())
            .args(["/T", "/F", "/PID", &pid.to_string()])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    #[cfg(not(any(unix, windows)))]
    let _ = pid;
}

/// A Windows Job Object holding a stateless command's shell and everything
/// it starts, so the whole tree can be killed even after the shell itself
/// has exited (when `taskkill /T` no longer finds it to walk from).
/// `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` also kills the tree if the handle is
/// closed without an explicit [`terminate`](Self::terminate).
///
/// The shell is assigned right after it is spawned; a process it starts in
/// that instant is outside the job and is left to `taskkill /T`.
#[cfg(windows)]
pub(crate) struct Job(windows_sys::Win32::Foundation::HANDLE);

// SAFETY: a job handle is a kernel object handle, usable from any thread.
#[cfg(windows)]
unsafe impl Send for Job {}
// SAFETY: as above; every use is a single kernel call on the handle.
#[cfg(windows)]
unsafe impl Sync for Job {}

#[cfg(windows)]
impl Job {
    /// Put `child` in a new kill-on-close job. `None` if any step fails, in
    /// which case `taskkill /T` remains the only tree kill.
    pub(crate) fn for_child(child: &Child) -> Option<Self> {
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };
        let process = child.raw_handle()? as HANDLE;
        // SAFETY: null attributes and name create an unnamed job with
        // default security; the returned handle is owned by `Job` below.
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() {
            return None;
        }
        let job = Job(handle);
        // SAFETY: the struct is plain data, for which all-zero is valid.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `info` is a live JOBOBJECT_EXTENDED_LIMIT_INFORMATION and
        // the length passed is its size; `process` is the child's handle,
        // kept open by `child` for the duration of the call.
        let ok = unsafe {
            SetInformationJobObject(
                job.0,
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) != 0
                && AssignProcessToJobObject(job.0, process) != 0
        };
        ok.then_some(job)
    }

    /// Kill every process still in the job.
    pub(crate) fn terminate(&self) {
        // SAFETY: `self.0` is a job handle owned by `self`.
        unsafe {
            windows_sys::Win32::System::JobObjects::TerminateJobObject(self.0, 1);
        }
    }
}

#[cfg(windows)]
impl Drop for Job {
    fn drop(&mut self) {
        // SAFETY: `self.0` is owned by `self` and closed exactly once.
        // Closing the last handle kills what is left (KILL_ON_JOB_CLOSE).
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// Kills the process tree of a running stateless command if the future
/// running it is dropped (cancelled) before it finished. `kill_on_drop` alone
/// would only reach the shell, not what it spawned.
pub(crate) struct GroupGuard {
    pid: Option<u32>,
    #[cfg(windows)]
    job: Option<Job>,
}

impl GroupGuard {
    pub(crate) fn new(child: &Child) -> Self {
        Self {
            pid: child.id(),
            #[cfg(windows)]
            job: Job::for_child(child),
        }
    }

    /// Kill whatever the command left running after its shell exited (a
    /// background job), then disarm. Unix: the process group outlives its
    /// leader while any member is alive, so its id cannot have been reused.
    /// Windows: the job holds the leftovers even though `taskkill /T` can no
    /// longer find them from the exited shell.
    pub(crate) fn kill_leftovers_and_disarm(&mut self) {
        #[cfg(unix)]
        if let Some(pid) = self.pid {
            signal_group(pid, libc::SIGKILL);
        }
        #[cfg(windows)]
        if let Some(job) = self.job.take() {
            job.terminate();
        }
        self.pid = None;
    }
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        // Declared after the child in `run_to_completion`, so this runs
        // before `kill_on_drop` takes the shell down.
        if let Some(pid) = self.pid {
            kill_tree_now(pid);
        }
        // Windows: dropping `job` closes it, killing anything `taskkill`
        // missed.
    }
}

fn spawn_reader<R>(mut stream: R, cap: usize) -> JoinHandle<HeadTailBuffer>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut buf = HeadTailBuffer::new(cap);
        let mut chunk = vec![0u8; READ_CHUNK];
        loop {
            match stream.read(&mut chunk).await {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.push(&chunk[..n]),
            }
        }
        buf
    })
}

async fn collect(
    handle: JoinHandle<HeadTailBuffer>,
    cap: usize,
    deadline: Duration,
) -> HeadTailBuffer {
    let abort = handle.abort_handle();
    match tokio::time::timeout(deadline, handle).await {
        Ok(Ok(buf)) => buf,
        Ok(Err(_)) => HeadTailBuffer::new(cap),
        Err(_) => {
            // Something outside the killed group still holds the pipe open.
            abort.abort();
            HeadTailBuffer::new(cap)
        }
    }
}

/// What to do when a stateless command times out, after which the process
/// tree of the local child is killed regardless.
pub(crate) type OnTimeout<'a> = Box<
    dyn FnOnce() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send + 'a>>
        + Send
        + 'a,
>;

/// Spawn `cmd` (with stdin closed and stdout/stderr piped), collect bounded
/// output, and enforce `timeout` by killing the process tree.
pub(crate) async fn run_to_completion(
    mut cmd: Command,
    timeout: Option<Duration>,
    max_output_bytes: usize,
    on_timeout: Option<OnTimeout<'_>>,
) -> Result<ShellResult, ShellError> {
    cmd.stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    isolate_process_group(&mut cmd);

    let started = Instant::now();
    let mut child = cmd
        .spawn()
        .map_err(|e| ShellError::io("failed to start shell", e))?;
    let mut guard = GroupGuard::new(&child);
    let stdout = spawn_reader(child.stdout.take().expect("stdout piped"), max_output_bytes);
    let stderr = spawn_reader(child.stderr.take().expect("stderr piped"), max_output_bytes);

    let waited = match timeout {
        Some(limit) => tokio::time::timeout(limit, child.wait()).await.ok(),
        None => Some(child.wait().await),
    };
    let (status, timed_out) = match waited {
        Some(status) => (
            Some(status.map_err(|e| ShellError::io("failed to wait for shell", e))?),
            false,
        ),
        None => {
            if let Some(hook) = on_timeout {
                hook().await;
            }
            kill_process_tree(&mut child, KILL_GRACE).await;
            (child.try_wait().ok().flatten(), true)
        }
    };
    // The shell has exited, but a background job it left behind (`sleep
    // 3600 & echo ok`) still holds the pipes open and would outlive the
    // command: kill it now, so the readers see EOF and keep what was already
    // written. (Should that fail, the readers' bounded wait below still
    // returns.)
    guard.kill_leftovers_and_disarm();

    // After a normal exit the pipes close promptly; after a kill, give the
    // readers a bounded moment to drain what was already written.
    let (stdout, stderr) = tokio::join!(
        collect(stdout, max_output_bytes, KILL_GRACE),
        collect(stderr, max_output_bytes, KILL_GRACE),
    );
    let (stdout, out_truncated) = stdout.finish();
    let (stderr, err_truncated) = stderr.finish();

    Ok(ShellResult {
        stdout,
        stderr,
        exit_code: status.map(exit_code).unwrap_or(-1),
        duration: started.elapsed(),
        truncated: out_truncated || err_truncated,
        timed_out,
    })
}

/// Run `command` once with the stateless `argv` (which ends in `-c` or
/// `-Command`), appending the command as a single argument.
pub(crate) async fn run_stateless(
    argv: &[String],
    command: &str,
    workdir: Option<&PathBuf>,
    env: Option<&HashMap<String, String>>,
    timeout: Option<Duration>,
    max_output_bytes: usize,
) -> Result<ShellResult, ShellError> {
    let command = if super::resolve::is_powershell(argv) {
        // Windows PowerShell defaults to a legacy code page; force UTF-8 so
        // non-ASCII output is not mangled.
        format!(
            "$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); {command}"
        )
    } else {
        command.to_string()
    };
    let (program, args) = argv
        .split_first()
        .ok_or_else(|| ShellError::Config("shell argv is empty".into()))?;
    let mut cmd = Command::new(program);
    cmd.args(args).arg(command);
    apply_env(&mut cmd, workdir, env);
    run_to_completion(cmd, timeout, max_output_bytes, None).await
}

#[cfg(all(test, windows))]
mod windows_tests {
    use super::*;

    /// A background process left by a stateless command is killed once the
    /// shell exits (through the job), rather than holding the output pipe
    /// open until the readers give up after `KILL_GRACE`.
    #[tokio::test]
    async fn background_child_is_killed_after_the_shell_exits() {
        let argv = vec!["cmd.exe".to_string(), "/C".to_string()];
        let result = run_stateless(
            &argv,
            "start /B ping -n 30 127.0.0.1 & echo ok",
            None,
            None,
            Some(Duration::from_secs(20)),
            64 * 1024,
        )
        .await
        .unwrap();
        assert!(result.stdout.contains("ok"), "{result:?}");
        assert!(!result.timed_out);
        assert!(
            result.duration < KILL_GRACE,
            "the background ping outlived the shell: {result:?}"
        );
    }
}
