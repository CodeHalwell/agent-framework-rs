//! [`DockerShellTool`]: run model-written commands inside a container.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use agent_framework_core::tools::{ApprovalMode, ToolDefinition};
use async_trait::async_trait;
use tokio::process::Command;
use tokio::sync::Mutex;

use super::environment::ShellFamily;
use super::executor::ShellExecutor;
use super::policy::ShellPolicy;
use super::process::{run_to_completion, OnTimeout};
use super::resolve::is_powershell;
use super::session::ShellSession;
use super::tool::{
    admit, shell_function, CommandHook, DEFAULT_MAX_OUTPUT_BYTES, DEFAULT_TIMEOUT,
    DEFAULT_TOOL_NAME,
};
use super::types::{ShellError, ShellMode, ShellResult};

/// Default image: a small Microsoft-maintained base with `bash`.
pub const DEFAULT_IMAGE: &str = "mcr.microsoft.com/azurelinux/base/core:3.0";
/// Default container user, `nobody:nogroup` on most distributions.
pub const DEFAULT_CONTAINER_USER: &str = "65534:65534";
/// Default network mode: none.
pub const DEFAULT_NETWORK: &str = "none";
/// Default memory limit.
pub const DEFAULT_MEMORY: &str = "512m";
/// Default process limit.
pub const DEFAULT_PIDS_LIMIT: u32 = 256;
/// Default working directory inside the container.
pub const DEFAULT_WORKDIR: &str = "/workspace";
const TMPFS: &str = "/tmp:rw,nosuid,nodev,size=64m";
/// Run inside the persistent container after a timeout, runaway output or
/// a cancelled command: kills every process the container user may signal
/// except the container's init (`sleep infinity`) and this shell itself.
const REAP_SCRIPT: &str = "kill -KILL -1 2>/dev/null; exit 0";

const PERSISTENT_DESCRIPTION: &str =
    "Execute a single shell command inside an isolated Docker container \
and return its stdout, stderr, and exit code. Commands run in a \
persistent session so `cd` and environment variables from previous \
calls are preserved within the container.";

const STATELESS_DESCRIPTION: &str =
    "Execute a single shell command inside an isolated Docker container \
and return its stdout, stderr, and exit code. Each command runs in a \
fresh container, so `cd` and environment variables do not persist \
between calls.";

/// `docker run` flags that would dismantle the isolation defaults if passed
/// through `extra_run_args`.
pub(crate) const BLOCKED_EXTRA_RUN_FLAGS: &[&str] = &[
    "--privileged",
    "--cap-add",
    "--security-opt",
    "--network",
    "--net",
    "--volume",
    "-v",
    "--mount",
    "--device",
    "--device-cgroup-rule",
    "--pid",
    "--ipc",
    "--userns",
    "--user",
    "-u",
    "--cgroupns",
    "--add-host",
    "--gpus",
    "--read-only",
    "--tmpfs",
    "--memory",
    "-m",
    "--memory-swap",
    "--pids-limit",
    // Binds the Docker API socket into the container: full daemon access.
    "--use-api-socket",
    // Imports another container's mounts, which may be host-backed.
    "--volumes-from",
    // A detached or renamed container escapes the timeout and cancellation
    // cleanup, which addresses the container by its generated name.
    "--detach",
    "-d",
    "--name",
];

/// `docker run` long options that take no value. Any other long option is
/// assumed to take the next token, so a detached value that starts with a
/// dash (`--env-file -vars.env`) is not misread as an option.
pub(crate) const VALUELESS_LONG_FLAGS: &[&str] = &[
    "--detach",
    "--disable-content-trust",
    "--help",
    "--init",
    "--interactive",
    "--no-healthcheck",
    "--oom-kill-disable",
    "--privileged",
    "--publish-all",
    "--quiet",
    "--read-only",
    "--rm",
    "--sig-proxy",
    "--tty",
    "--use-api-socket",
];

/// Boolean `docker run` shorthands (may be clustered: `-it`).
pub(crate) const BOOLEAN_SHORT_FLAGS: &str = "diPqt";
/// Value-taking `docker run` shorthands (the rest of the token is the value:
/// `-v/:/host`).
pub(crate) const VALUE_SHORT_FLAGS: &str = "acehlmpuvw";

/// Expand a single-dash token into the shorthands it sets. Scanning stops at
/// the first value-taking shorthand, because the rest is its value, and at
/// any unknown character.
fn short_flags_in_token(token: &str) -> Vec<String> {
    let mut flags = Vec::new();
    for c in token.chars().skip(1) {
        if BOOLEAN_SHORT_FLAGS.contains(c) {
            flags.push(format!("-{c}"));
            continue;
        }
        if VALUE_SHORT_FLAGS.contains(c) {
            flags.push(format!("-{c}"));
        }
        break;
    }
    flags
}

fn is_blocked_token(raw: &str) -> bool {
    let flag = raw.split('=').next().unwrap_or(raw);
    if flag.starts_with("--") {
        return BLOCKED_EXTRA_RUN_FLAGS.contains(&flag);
    }
    short_flags_in_token(flag)
        .iter()
        .any(|s| BLOCKED_EXTRA_RUN_FLAGS.contains(&s.as_str()))
}

/// Whether `raw` takes the following token as its value.
pub(crate) fn consumes_next_token(raw: &str) -> bool {
    let (name, attached) = match raw.split_once('=') {
        Some((name, _)) => (name, true),
        None => (raw, false),
    };
    if attached {
        return false;
    }
    if name.starts_with("--") {
        return !VALUELESS_LONG_FLAGS.contains(&name);
    }
    let shorts = short_flags_in_token(name);
    let Some(last) = shorts.last() else {
        return false;
    };
    if name.chars().count() - 1 > shorts.len() {
        // Trailing text is the value: `-v/:/host:rw`, `-m0`.
        return false;
    }
    last.chars()
        .nth(1)
        .is_some_and(|c| VALUE_SHORT_FLAGS.contains(c))
}

/// Reject `extra_run_args` that would break the isolation contract,
/// including attached (`-v/:/host`) and clustered (`-itv/:/host`) forms.
pub(crate) fn validate_extra_run_args(args: &[String]) -> Result<(), ShellError> {
    let mut bad = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let raw = &args[i];
        i += 1;
        // "-" is a positional and "--" only ends option parsing; arguments
        // after "--" are still checked so nothing hides behind it.
        if raw == "-" || raw == "--" || !raw.starts_with('-') {
            continue;
        }
        if is_blocked_token(raw) {
            bad.push(raw.clone());
        }
        if consumes_next_token(raw) {
            i += 1;
        }
    }
    if bad.is_empty() {
        return Ok(());
    }
    Err(ShellError::Config(format!(
        "extra_run_args contains flags that would dismantle DockerShellTool's isolation \
         defaults: {bad:?}. Use the dedicated builder methods (network, host_workdir, \
         mount_readonly, read_only_root, memory, pids_limit, user) instead."
    )))
}

/// Whether `binary` is on `PATH` and its daemon answers `version`.
pub async fn is_docker_available(binary: &str) -> bool {
    if super::resolve::which(binary).is_none() && !std::path::Path::new(binary).is_file() {
        return false;
    }
    let mut cmd = Command::new(binary);
    cmd.args(["version", "--format", "{{.Server.Version}}"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true);
    match tokio::time::timeout(Duration::from_secs(5), cmd.output()).await {
        Ok(Ok(out)) => out.status.success() && !out.stdout.trim_ascii().is_empty(),
        _ => false,
    }
}

/// Builder for [`DockerShellTool`]. Every setting has upstream's default.
#[derive(Clone)]
pub struct DockerShellToolBuilder {
    image: String,
    container_name: Option<String>,
    mode: ShellMode,
    host_workdir: Option<PathBuf>,
    workdir: String,
    mount_readonly: bool,
    network: String,
    memory: String,
    pids_limit: u32,
    user: String,
    read_only_root: bool,
    extra_run_args: Vec<String>,
    env: Vec<(String, String)>,
    policy: ShellPolicy,
    timeout: Option<Duration>,
    max_output_bytes: usize,
    approval_mode: ApprovalMode,
    acknowledge_unsafe: bool,
    on_command: Option<CommandHook>,
    docker_binary: String,
    shell: String,
}

impl Default for DockerShellToolBuilder {
    fn default() -> Self {
        Self {
            image: DEFAULT_IMAGE.into(),
            container_name: None,
            mode: ShellMode::Persistent,
            host_workdir: None,
            workdir: DEFAULT_WORKDIR.into(),
            mount_readonly: true,
            network: DEFAULT_NETWORK.into(),
            memory: DEFAULT_MEMORY.into(),
            pids_limit: DEFAULT_PIDS_LIMIT,
            user: DEFAULT_CONTAINER_USER.into(),
            read_only_root: true,
            extra_run_args: Vec::new(),
            env: Vec::new(),
            policy: ShellPolicy::new(),
            timeout: Some(DEFAULT_TIMEOUT),
            max_output_bytes: DEFAULT_MAX_OUTPUT_BYTES,
            approval_mode: ApprovalMode::AlwaysRequire,
            acknowledge_unsafe: false,
            on_command: None,
            docker_binary: "docker".into(),
            shell: "bash".into(),
        }
    }
}

impl std::fmt::Debug for DockerShellToolBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DockerShellToolBuilder")
            .field("image", &self.image)
            .field("mode", &self.mode)
            .field("network", &self.network)
            .field("user", &self.user)
            .field("approval_mode", &self.approval_mode)
            .finish_non_exhaustive()
    }
}

impl DockerShellToolBuilder {
    /// The image to run; it must contain the shell (and `sleep` for
    /// persistent mode). Default [`DEFAULT_IMAGE`].
    pub fn image(mut self, image: impl Into<String>) -> Self {
        self.image = image.into();
        self
    }

    /// An explicit container name; by default a unique one is generated.
    pub fn container_name(mut self, name: impl Into<String>) -> Self {
        self.container_name = Some(name.into());
        self
    }

    /// [`ShellMode::Persistent`] (default): one container for the tool's
    /// life. [`ShellMode::Stateless`]: a fresh container per command.
    pub fn mode(mut self, mode: ShellMode) -> Self {
        self.mode = mode;
        self
    }

    /// A host directory to mount at the working directory. Read-only unless
    /// [`mount_readonly(false)`](Self::mount_readonly).
    pub fn host_workdir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.host_workdir = Some(dir.into());
        self
    }

    /// The working directory inside the container. Default `/workspace`.
    pub fn workdir(mut self, dir: impl Into<String>) -> Self {
        self.workdir = dir.into();
        self
    }

    /// Mount the host directory read-only. Default `true`.
    pub fn mount_readonly(mut self, readonly: bool) -> Self {
        self.mount_readonly = readonly;
        self
    }

    /// The Docker network mode. Default `none`.
    pub fn network(mut self, network: impl Into<String>) -> Self {
        self.network = network.into();
        self
    }

    /// The memory limit (`512m`, `2g`). Default `512m`.
    pub fn memory(mut self, memory: impl Into<String>) -> Self {
        self.memory = memory.into();
        self
    }

    /// The process limit inside the container. Default 256.
    pub fn pids_limit(mut self, limit: u32) -> Self {
        self.pids_limit = limit;
        self
    }

    /// The `UID:GID` to run as. Default `65534:65534`.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = user.into();
        self
    }

    /// Mount the root filesystem read-only. Default `true`.
    pub fn read_only_root(mut self, read_only: bool) -> Self {
        self.read_only_root = read_only;
        self
    }

    /// Extra `docker run` arguments, appended after the isolation defaults.
    /// Flags that would undo those defaults are rejected by
    /// [`build`](Self::build); use the dedicated methods instead.
    pub fn extra_run_args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extra_run_args = args.into_iter().map(Into::into).collect();
        self
    }

    /// Set an environment variable inside the container (`-e KEY=VALUE`).
    /// The value appears on the `docker` command line, so it is visible in
    /// the host's process list: do not pass secrets this way.
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.push((key.into(), value.into()));
        self
    }

    /// The policy applied before execution. Default: empty.
    pub fn policy(mut self, policy: ShellPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The per-command timeout; `None` disables it. Default 30 seconds.
    pub fn timeout(mut self, timeout: Option<Duration>) -> Self {
        self.timeout = timeout;
        self
    }

    /// The byte limit per output stream. Default 64 KiB. Must be positive.
    pub fn max_output_bytes(mut self, max: usize) -> Self {
        self.max_output_bytes = max;
        self
    }

    /// The approval mode of [`DockerShellTool::as_function`]. Default
    /// [`ApprovalMode::AlwaysRequire`].
    ///
    /// With the isolation defaults kept, [`ApprovalMode::NeverRequire`] needs
    /// no acknowledgement, because the container (with a trusted runtime) is
    /// the intended boundary. If the configuration weakens that boundary
    /// (root user, a writable host mount, a writable root filesystem, or a
    /// network other than `none`), [`build`](Self::build) fails unless
    /// [`acknowledge_unsafe`](Self::acknowledge_unsafe) is also set.
    pub fn approval_mode(mut self, mode: ApprovalMode) -> Self {
        self.approval_mode = mode;
        self
    }

    /// Confirm that running without approval is intended even though the
    /// isolation defaults were weakened (see
    /// [`approval_mode`](Self::approval_mode)).
    pub fn acknowledge_unsafe(mut self, acknowledge: bool) -> Self {
        self.acknowledge_unsafe = acknowledge;
        self
    }

    /// The isolation defaults this configuration gives up, if any.
    fn weakened_isolation(&self) -> Vec<&'static str> {
        let mut weakened = Vec::new();
        let uid = self.user.split(':').next().unwrap_or("").trim();
        if uid.is_empty() || uid == "0" || uid.eq_ignore_ascii_case("root") {
            weakened.push("the container runs as root");
        }
        if self.host_workdir.is_some() && !self.mount_readonly {
            weakened.push("the host directory is mounted writable");
        }
        if !self.read_only_root {
            weakened.push("the root filesystem is writable");
        }
        if self.network != DEFAULT_NETWORK {
            weakened.push("the container has network access");
        }
        weakened
    }

    /// An audit hook called with every command that passes the policy.
    pub fn on_command<F>(mut self, hook: F) -> Self
    where
        F: Fn(&str) + Send + Sync + 'static,
    {
        self.on_command = Some(Arc::new(hook));
        self
    }

    /// The container CLI. Default `docker`; `podman` also works.
    pub fn docker_binary(mut self, binary: impl Into<String>) -> Self {
        self.docker_binary = binary.into();
        self
    }

    /// The shell inside the container. Default `bash`; use `sh` for images
    /// without bash.
    pub fn shell(mut self, shell: impl Into<String>) -> Self {
        self.shell = shell.into();
        self
    }

    /// Validate the configuration and build the tool. Nothing is started
    /// until the first command or [`start`](ShellExecutor::start).
    pub fn build(self) -> Result<DockerShellTool, ShellError> {
        validate_extra_run_args(&self.extra_run_args)?;
        if self.approval_mode == ApprovalMode::NeverRequire && !self.acknowledge_unsafe {
            let weakened = self.weakened_isolation();
            if !weakened.is_empty() {
                return Err(ShellError::Config(format!(
                    "Setting approval mode to NeverRequire relies on the container as the \
                     security boundary, but this configuration weakens it ({}). Restore the \
                     isolation defaults, keep approval on, or call acknowledge_unsafe(true) if \
                     unapproved model commands with these privileges are intended.",
                    weakened.join("; ")
                )));
            }
        }
        if self.max_output_bytes == 0 {
            return Err(ShellError::Config(
                "max_output_bytes must be positive".into(),
            ));
        }
        let container_name = self
            .container_name
            .clone()
            .unwrap_or_else(generate_container_name);
        Ok(DockerShellTool {
            inner: Arc::new(DockerInner {
                container_name,
                config: self,
                state: Mutex::new(PersistentState::default()),
            }),
        })
    }
}

/// Removes a container by name when dropped while armed, i.e. when the
/// future that created it was cancelled and nothing else will clean it up.
/// Killing the local CLI (`kill_on_drop`) does not stop the container, and
/// `--rm` reaps it only once it exits on its own.
///
/// `Drop` cannot await, so `docker rm -f` is spawned synchronously and
/// waited on (with a short retry, in case the daemon is still creating the
/// container) on a detached thread.
struct ContainerCleanup {
    binary: String,
    name: String,
    armed: bool,
}

impl ContainerCleanup {
    fn new(binary: &str, name: &str) -> Self {
        Self {
            binary: binary.to_string(),
            name: name.to_string(),
            armed: true,
        }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for ContainerCleanup {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let binary = std::mem::take(&mut self.binary);
        let name = std::mem::take(&mut self.name);
        let spawn = move |binary: &str, name: &str| {
            std::process::Command::new(binary)
                .args(["rm", "-f", name])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
        };
        let first = match spawn(&binary, &name) {
            Ok(child) => child,
            Err(err) => {
                tracing::error!(container = %name, error = %err, "could not run docker rm -f after cancellation; the container may be leaked");
                return;
            }
        };
        let reaper = std::thread::Builder::new()
            .name("af-shell-docker-rm".into())
            .spawn(move || {
                let mut child = first;
                for attempt in 1..=3 {
                    if child.wait().is_ok_and(|status| status.success()) {
                        return;
                    }
                    if attempt == 3 {
                        break;
                    }
                    std::thread::sleep(Duration::from_secs(1));
                    child = match spawn(&binary, &name) {
                        Ok(child) => child,
                        Err(_) => break,
                    };
                }
                tracing::warn!(container = %name, "docker rm -f after cancellation did not succeed; the container may need manual cleanup");
            });
        if let Err(err) = reaper {
            tracing::warn!(error = %err, "could not start the docker rm -f reaper thread");
        }
    }
}

fn generate_container_name() -> String {
    let id = uuid::Uuid::new_v4().simple().to_string();
    format!("af-shell-{}", &id[..12])
}

#[derive(Default)]
struct PersistentState {
    container_started: bool,
    session: Option<Arc<ShellSession>>,
}

struct DockerInner {
    container_name: String,
    config: DockerShellToolBuilder,
    state: Mutex<PersistentState>,
}

/// A shell tool that runs commands inside a Docker (or compatible)
/// container.
///
/// A thin wrapper over the `docker` CLI (or a compatible one such as
/// `podman`): persistent mode starts one long-lived container with
/// `docker run -d ... sleep infinity` and drives a shell in it through
/// `docker exec -i`, reusing the persistent [`ShellSession`] protocol;
/// stateless mode runs every command in a throwaway `docker run --rm`.
///
/// The container is the intended security boundary. Its effectiveness
/// depends on the runtime, the image and the flags it is started with. The
/// defaults are:
///
/// * `--network none`: no network.
/// * `--user 65534:65534`: `nobody:nogroup`.
/// * `--read-only` root filesystem; only the optional host mount is
///   writable, and only with [`mount_readonly(false)`](DockerShellToolBuilder::mount_readonly).
/// * `--memory 512m` and `--pids-limit 256`.
/// * `--cap-drop ALL` and `--security-opt no-new-privileges`.
/// * `--tmpfs /tmp` for scratch space that does not escape the container.
///
/// [`extra_run_args`](DockerShellToolBuilder::extra_run_args) that would undo
/// any of these are rejected when the tool is built.
///
/// Every `docker` invocation is a direct exec with an argv; no host shell is
/// involved. The command itself is interpreted by the shell **inside** the
/// container.
///
/// # Single-session ownership
///
/// In persistent mode the tool owns a long-lived container whose
/// filesystem, environment and working directory every later command sees.
/// It belongs to one conversation: create one per session and
/// [`close`](ShellExecutor::close) it when the session ends, which removes
/// the container. Use [`ShellMode::Stateless`] if one instance must be
/// shared.
#[derive(Clone)]
pub struct DockerShellTool {
    inner: Arc<DockerInner>,
}

impl std::fmt::Debug for DockerShellTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DockerShellTool")
            .field("container_name", &self.inner.container_name)
            .field("config", &self.inner.config)
            .finish_non_exhaustive()
    }
}

impl DockerShellTool {
    /// A builder with upstream's defaults.
    pub fn builder() -> DockerShellToolBuilder {
        DockerShellToolBuilder::default()
    }

    /// The persistent container's name.
    pub fn container_name(&self) -> &str {
        &self.inner.container_name
    }

    /// The tool's mode.
    pub fn mode(&self) -> ShellMode {
        self.inner.config.mode
    }

    /// The approval mode [`as_function`](Self::as_function) applies.
    pub fn approval_mode(&self) -> ApprovalMode {
        self.inner.config.approval_mode
    }

    /// The function to hand an agent, named `run_shell`.
    pub fn as_function(&self) -> ToolDefinition {
        self.as_function_with(DEFAULT_TOOL_NAME, None)
    }

    /// [`as_function`](Self::as_function) with a custom name and description.
    /// Pick a name no other tool uses, so approval rules for another tool
    /// cannot match it.
    pub fn as_function_with(&self, name: &str, description: Option<&str>) -> ToolDefinition {
        let default = match self.inner.config.mode {
            ShellMode::Persistent => PERSISTENT_DESCRIPTION,
            ShellMode::Stateless => STATELESS_DESCRIPTION,
        };
        shell_function(
            self.clone(),
            name,
            description.unwrap_or(default),
            self.inner.config.approval_mode,
        )
    }

    /// The `docker run -d` argv that starts the persistent container.
    pub fn run_argv(&self) -> Vec<String> {
        let c = &self.inner.config;
        let mut argv = vec![
            c.docker_binary.clone(),
            "run".into(),
            "-d".into(),
            "--rm".into(),
        ];
        argv.extend(["--name".into(), self.inner.container_name.clone()]);
        self.push_isolation(&mut argv);
        argv.extend([c.image.clone(), "sleep".into(), "infinity".into()]);
        argv
    }

    /// The `docker exec -i` argv for the persistent shell (`interactive`) or
    /// the prefix a command is appended to.
    pub fn exec_argv(&self, interactive: bool) -> Vec<String> {
        let c = &self.inner.config;
        let mut argv = vec![
            c.docker_binary.clone(),
            "exec".into(),
            "-i".into(),
            self.inner.container_name.clone(),
            c.shell.clone(),
        ];
        if !interactive {
            argv.push("-c".into());
        } else if c.shell == "bash" {
            argv.extend(["--noprofile".into(), "--norc".into()]);
        }
        argv
    }

    /// The `docker run --rm -i` argv that runs `command` in a fresh container
    /// named `name`.
    pub fn stateless_argv(&self, name: &str, command: &str) -> Vec<String> {
        let c = &self.inner.config;
        let mut argv = vec![
            c.docker_binary.clone(),
            "run".into(),
            "--rm".into(),
            "-i".into(),
            "--name".into(),
            name.into(),
        ];
        self.push_isolation(&mut argv);
        argv.extend([
            c.image.clone(),
            c.shell.clone(),
            "-c".into(),
            command.into(),
        ]);
        argv
    }

    /// The isolation defaults, then the host mount, env and extra args. Extra
    /// args come last so docker's last-flag-wins never lets a default
    /// override them, and validation keeps them from overriding a default.
    fn push_isolation(&self, argv: &mut Vec<String>) {
        let c = &self.inner.config;
        argv.extend([
            "--user".into(),
            c.user.clone(),
            "--network".into(),
            c.network.clone(),
            "--memory".into(),
            c.memory.clone(),
            "--pids-limit".into(),
            c.pids_limit.to_string(),
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            "--tmpfs".into(),
            TMPFS.into(),
            "--workdir".into(),
            c.workdir.clone(),
        ]);
        if c.read_only_root {
            argv.push("--read-only".into());
        }
        if let Some(host) = &c.host_workdir {
            let mode = if c.mount_readonly { "ro" } else { "rw" };
            argv.extend([
                "-v".into(),
                format!("{}:{}:{mode}", host.to_string_lossy(), c.workdir),
            ]);
        }
        for (k, v) in &c.env {
            argv.extend(["-e".into(), format!("{k}={v}")]);
        }
        argv.extend(c.extra_run_args.iter().cloned());
    }

    async fn start_container(&self) -> Result<(), ShellError> {
        let argv = self.run_argv();
        // If this future is dropped mid-start the daemon may still create the
        // container, and nothing has recorded it yet: remove it by name.
        let mut cleanup =
            ContainerCleanup::new(&self.inner.config.docker_binary, &self.inner.container_name);
        let out = Command::new(&argv[0])
            .args(&argv[1..])
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await;
        // A refused start is not cleaned up by name: with an explicit
        // `container_name` the name may belong to someone else's container.
        cleanup.disarm();
        let out = out.map_err(|e| ShellError::io("failed to run the container CLI", e))?;
        if !out.status.success() {
            return Err(ShellError::Execution(format!(
                "Failed to start container ({}): {}",
                out.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        let id = String::from_utf8_lossy(&out.stdout);
        tracing::info!(
            container = %self.inner.container_name,
            id = %id.trim().chars().take(12).collect::<String>(),
            "started docker container"
        );
        Ok(())
    }

    /// `docker rm -f`, retried once. A failure is logged, not returned, so
    /// an operator can clean up by name.
    async fn stop_container(&self) {
        let binary = &self.inner.config.docker_binary;
        let name = &self.inner.container_name;
        for (attempt, limit) in [(1, Duration::from_secs(10)), (2, Duration::from_secs(5))] {
            match docker_quiet(binary, &["rm", "-f", name], limit).await {
                Ok(()) => return,
                Err(err) if attempt == 1 => {
                    tracing::warn!(container = %name, error = %err, "docker rm -f failed; retrying")
                }
                Err(err) => tracing::error!(
                    container = %name,
                    error = %err,
                    "docker rm -f failed after retry; the container may need manual cleanup"
                ),
            }
        }
    }

    async fn run_stateless(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        let name = generate_container_name();
        let argv = self.stateless_argv(&name, command);
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..]);
        let binary = self.inner.config.docker_binary.clone();
        let mut cleanup = ContainerCleanup::new(&binary, &name);
        let on_timeout: OnTimeout<'_> = Box::new(move || {
            Box::pin(async move {
                // Killing the local CLI does not stop the container: kill it
                // by name. If that fails, `--rm` never reaps it, so remove it.
                if let Err(err) = docker_quiet(
                    &binary,
                    &["kill", "--signal", "KILL", &name],
                    Duration::from_secs(5),
                )
                .await
                {
                    tracing::warn!(container = %name, error = %err, "docker kill failed; trying docker rm -f");
                    if let Err(err) =
                        docker_quiet(&binary, &["rm", "-f", &name], Duration::from_secs(5)).await
                    {
                        tracing::error!(container = %name, error = %err, "docker rm -f failed; container may be leaked");
                    }
                }
            })
        });
        let result = run_to_completion(
            cmd,
            timeout,
            self.inner.config.max_output_bytes,
            Some(on_timeout),
        )
        .await;
        // Finished, or timed out and already killed by name.
        cleanup.disarm();
        result
    }

    /// Stop whatever the persistent container is still running after a
    /// timeout, runaway output or a cancelled command. Killing the local
    /// `docker exec` CLI does not stop the shell or the command inside the
    /// container, so they would keep running and overlap the next call.
    /// Kills every process except the container's init; if that fails, the
    /// container is removed so the next call recreates it.
    async fn reap_container(&self, state: &mut PersistentState) {
        if !state.container_started {
            return;
        }
        let c = &self.inner.config;
        let name = &self.inner.container_name;
        match docker_quiet(
            &c.docker_binary,
            &["exec", name, &c.shell, "-c", REAP_SCRIPT],
            Duration::from_secs(10),
        )
        .await
        {
            Ok(()) => {
                tracing::info!(container = %name, "killed leftover processes in the container")
            }
            Err(err) => {
                tracing::warn!(container = %name, error = %err, "could not kill leftover processes in the container; recreating it");
                if let Some(session) = state.session.take() {
                    session.close().await;
                }
                self.stop_container().await;
                state.container_started = false;
            }
        }
    }
}

/// Run a container-CLI housekeeping command with a time limit.
async fn docker_quiet(binary: &str, args: &[&str], limit: Duration) -> Result<(), String> {
    let mut cmd = Command::new(binary);
    cmd.args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    match tokio::time::timeout(limit, cmd.output()).await {
        Err(_) => Err(format!("{} timed out", args.join(" "))),
        Ok(Err(err)) => Err(err.to_string()),
        Ok(Ok(out)) if out.status.success() => Ok(()),
        Ok(Ok(out)) => Err(format!(
            "exit {}: {}",
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stderr).trim()
        )),
    }
}

#[async_trait]
impl ShellExecutor for DockerShellTool {
    /// Start the container and its shell. A no-op in stateless mode.
    async fn start(&self) -> Result<(), ShellError> {
        if self.inner.config.mode == ShellMode::Stateless {
            return Ok(());
        }
        let mut state = self.inner.state.lock().await;
        if !state.container_started {
            self.start_container().await?;
            state.container_started = true;
            state.session = Some(Arc::new(ShellSession::new(
                self.exec_argv(true),
                None,
                None,
                self.inner.config.max_output_bytes,
            )));
        }
        if let Some(session) = &state.session {
            session.start().await?;
        }
        Ok(())
    }

    /// Stop the shell and remove the container. A no-op in stateless mode.
    async fn close(&self) -> Result<(), ShellError> {
        if self.inner.config.mode == ShellMode::Stateless {
            return Ok(());
        }
        let mut state = self.inner.state.lock().await;
        if let Some(session) = state.session.take() {
            session.close().await;
        }
        if state.container_started {
            self.stop_container().await;
            state.container_started = false;
        }
        Ok(())
    }

    async fn run(
        &self,
        command: &str,
        timeout: Option<Duration>,
    ) -> Result<ShellResult, ShellError> {
        let c = &self.inner.config;
        admit(
            &c.policy,
            c.on_command.as_ref(),
            command,
            Some(PathBuf::from(&c.workdir)),
        )?;
        let timeout = timeout.or(c.timeout);
        if c.mode == ShellMode::Stateless {
            return self.run_stateless(command, timeout).await;
        }
        {
            // A previous call was cancelled mid-command: its command may still
            // be running in the container. Reap it before the session is
            // replaced.
            let mut state = self.inner.state.lock().await;
            if state.session.as_ref().is_some_and(|s| s.is_poisoned()) {
                self.reap_container(&mut state).await;
            }
        }
        self.start().await?;
        let session = self
            .inner
            .state
            .lock()
            .await
            .session
            .clone()
            .ok_or_else(|| {
                ShellError::Execution("DockerShellTool session failed to start".into())
            })?;
        let result = session.run(command, timeout).await;
        let timed_out = matches!(&result, Ok(r) if r.timed_out);
        if timed_out || !session.is_live().await {
            // The session interrupted (or gave up on) the local CLI only.
            // Start the next call from a fresh shell with nothing left over.
            session.close().await;
            let mut state = self.inner.state.lock().await;
            self.reap_container(&mut state).await;
        }
        result
    }

    fn shell_family(&self) -> Option<ShellFamily> {
        Some(
            if is_powershell(std::slice::from_ref(&self.inner.config.shell)) {
                ShellFamily::PowerShell
            } else {
                ShellFamily::Posix
            },
        )
    }

    fn os_description(&self) -> Option<String> {
        Some(format!("a Linux container ({})", self.inner.config.image))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    fn rejects(extra: &[&str]) -> bool {
        DockerShellTool::builder()
            .extra_run_args(extra.iter().copied())
            .build()
            .is_err()
    }

    #[test]
    fn run_argv_minimal_defaults() {
        let tool = DockerShellTool::builder()
            .image("ubuntu:24.04")
            .container_name("af-shell-test")
            .build()
            .unwrap();
        let argv = tool.run_argv();
        let after = |flag: &str| argv[argv.iter().position(|a| a == flag).unwrap() + 1].clone();
        assert_eq!(argv[..4], s(&["docker", "run", "-d", "--rm"]));
        assert_eq!(after("--name"), "af-shell-test");
        assert_eq!(after("--network"), "none");
        assert_eq!(after("--user"), "65534:65534");
        assert_eq!(after("--cap-drop"), "ALL");
        assert_eq!(after("--security-opt"), "no-new-privileges");
        assert_eq!(after("--memory"), "512m");
        assert_eq!(after("--pids-limit"), "256");
        assert!(argv.contains(&"--read-only".to_string()));
        assert!(!argv.contains(&"-v".to_string()));
        assert_eq!(
            argv[argv.len() - 3..],
            s(&["ubuntu:24.04", "sleep", "infinity"])
        );
    }

    #[test]
    fn run_argv_with_host_workdir_readonly_and_writable() {
        let ro = DockerShellTool::builder()
            .host_workdir("/tmp/host")
            .workdir("/work")
            .build()
            .unwrap()
            .run_argv();
        assert_eq!(
            ro[ro.iter().position(|a| a == "-v").unwrap() + 1],
            "/tmp/host:/work:ro"
        );

        let rw = DockerShellTool::builder()
            .host_workdir("/data")
            .workdir("/work")
            .mount_readonly(false)
            .read_only_root(false)
            .build()
            .unwrap()
            .run_argv();
        assert_eq!(
            rw[rw.iter().position(|a| a == "-v").unwrap() + 1],
            "/data:/work:rw"
        );
        assert!(!rw.contains(&"--read-only".to_string()));
    }

    #[test]
    fn run_argv_passes_env_and_extra_args_after_isolation_defaults() {
        let argv = DockerShellTool::builder()
            .docker_binary("podman")
            .image("alpine")
            .env("FOO", "bar")
            .env("X", "y z")
            .extra_run_args(["--label", "team=af"])
            .build()
            .unwrap()
            .run_argv();
        assert_eq!(argv[0], "podman");
        let env: Vec<&String> = argv
            .iter()
            .enumerate()
            .filter(|(_, a)| *a == "-e")
            .map(|(i, _)| &argv[i + 1])
            .collect();
        assert_eq!(env, ["FOO=bar", "X=y z"]);
        let pos = |flag: &str| argv.iter().position(|a| a == flag).unwrap();
        let label = pos("--label");
        assert!(label < pos("alpine"));
        for flag in [
            "--user",
            "--network",
            "--memory",
            "--pids-limit",
            "--cap-drop",
            "--security-opt",
            "--read-only",
        ] {
            assert!(label > pos(flag), "{flag}");
        }
    }

    #[test]
    fn exec_argv_variants() {
        let tool = DockerShellTool::builder()
            .container_name("c")
            .build()
            .unwrap();
        assert_eq!(
            tool.exec_argv(true),
            s(&["docker", "exec", "-i", "c", "bash", "--noprofile", "--norc"])
        );
        assert_eq!(
            tool.exec_argv(false),
            s(&["docker", "exec", "-i", "c", "bash", "-c"])
        );
        let sh = DockerShellTool::builder()
            .container_name("c")
            .shell("sh")
            .build()
            .unwrap();
        assert_eq!(sh.exec_argv(true), s(&["docker", "exec", "-i", "c", "sh"]));
    }

    #[test]
    fn stateless_argv_ends_with_shell_and_single_command_argument() {
        let tool = DockerShellTool::builder()
            .mode(ShellMode::Stateless)
            .docker_binary("podman")
            .image("alpine:3")
            .shell("sh")
            .host_workdir("/repo")
            .mount_readonly(false)
            .env("AF_TEST", "1")
            .build()
            .unwrap();
        let argv = tool.stateless_argv("n", "echo hi; rm -rf /tmp/x");
        assert_eq!(argv[..4], s(&["podman", "run", "--rm", "-i"]));
        assert!(argv.contains(&"/repo:/workspace:rw".to_string()));
        assert!(argv.contains(&"AF_TEST=1".to_string()));
        assert_eq!(
            argv[argv.len() - 4..],
            s(&["alpine:3", "sh", "-c", "echo hi; rm -rf /tmp/x"])
        );
    }

    #[test]
    fn rejects_isolation_breaking_extra_run_args() {
        let cases: &[&[&str]] = &[
            &["--privileged"],
            &["--network=host"],
            &["--network", "host"],
            &["--net=host"],
            &["-v", "/:/host:rw"],
            &["--volume=/etc:/etc"],
            &["--cap-add=ALL"],
            &["--cap-add", "SYS_ADMIN"],
            &["--security-opt", "seccomp=unconfined"],
            &["--device", "/dev/kvm"],
            &["--pid=host"],
            &["--ipc=host"],
            &["--userns=host"],
            &["--user=0:0"],
            &["--read-only=false"],
            &["--tmpfs", "/var:rw"],
            &["--gpus", "all"],
            &["--add-host", "evil:1.2.3.4"],
            &["--label", "x=1", "--privileged"],
            &["-v/:/host:rw"],
            &["-v/var/run/docker.sock:/var/run/docker.sock"],
            &["-v=/etc:/etc"],
            &["-u0:0"],
            &["-u", "0:0"],
            &["-u=0:0"],
            &["-itv/:/host:rw"],
            &["-itu0:0"],
            &["-m0"],
            &["-m", "0"],
            &["--memory=0"],
            &["--memory", "0"],
            &["--memory-swap=-1"],
            &["--pids-limit=-1"],
            &["--pids-limit", "-1"],
            &["-m0", "--pids-limit=-1"],
            &["--", "-u0:0"],
            &["-it", "-u0:0"],
            &["--privileged", "-v/:/host:rw"],
            &["--use-api-socket"],
            &["--volumes-from", "other"],
            &["--volumes-from=other"],
            &["--detach"],
            &["-d"],
            &["-dit"],
            &["-itd"],
            &["--name", "other"],
            &["--name=other"],
        ];
        for extra in cases {
            assert!(rejects(extra), "{extra:?} should be rejected");
        }
    }

    #[test]
    fn rejection_message_reports_the_raw_token() {
        let err = DockerShellTool::builder()
            .extra_run_args(["-v/:/host:rw"])
            .build()
            .unwrap_err();
        assert!(err.to_string().contains("-v/:/host:rw"));
        assert!(err.to_string().contains("isolation defaults"));
    }

    #[test]
    fn accepts_extra_run_args_that_are_not_blocked() {
        let cases: &[&[&str]] = &[
            &["--label", "team=af", "--name-suffix", "x"],
            &["--userland-proxy=false"],
            &["--network-alias", "svc"],
            &["-l", "team=af"],
            &["-lversion=1"],
            &["-e", "USER=root"],
            &["-eversion=1"],
            &["-w/workspace"],
            &["-it"],
            &["--"],
            &["-"],
            &["--env-file", "-variables.env"],
            &["--label", "-upper"],
            &["--label", "-v/:/host:rw"],
            &["--entrypoint", "-v"],
            &["-e", "-value"],
            &["--hostname", "-unusual"],
            &["--env-file=-variables.env"],
        ];
        for extra in cases {
            assert!(!rejects(extra), "{extra:?} should be accepted");
        }
    }

    /// `docker run --help` short options, so the tables can be audited
    /// without a daemon.
    const DOCKER_RUN_SHORT_ALIASES: &[(&str, &str)] = &[
        ("--attach", "-a"),
        ("--cpu-shares", "-c"),
        ("--detach", "-d"),
        ("--env", "-e"),
        ("--hostname", "-h"),
        ("--interactive", "-i"),
        ("--label", "-l"),
        ("--memory", "-m"),
        ("--publish", "-p"),
        ("--publish-all", "-P"),
        ("--quiet", "-q"),
        ("--tty", "-t"),
        ("--user", "-u"),
        ("--volume", "-v"),
        ("--workdir", "-w"),
    ];

    #[test]
    fn blocked_flags_cover_every_docker_short_alias() {
        for (long, short) in DOCKER_RUN_SHORT_ALIASES {
            if BLOCKED_EXTRA_RUN_FLAGS.contains(long) {
                assert!(
                    BLOCKED_EXTRA_RUN_FLAGS.contains(short),
                    "{long} blocked without {short}"
                );
            }
        }
    }

    #[test]
    fn short_flag_tables_match_docker_run_options() {
        let mut modelled: Vec<String> = BOOLEAN_SHORT_FLAGS
            .chars()
            .chain(VALUE_SHORT_FLAGS.chars())
            .map(|c| format!("-{c}"))
            .collect();
        modelled.sort();
        let mut docker: Vec<String> = DOCKER_RUN_SHORT_ALIASES
            .iter()
            .map(|(_, s)| s.to_string())
            .collect();
        docker.sort();
        assert_eq!(modelled, docker);
        assert!(!BOOLEAN_SHORT_FLAGS
            .chars()
            .any(|c| VALUE_SHORT_FLAGS.contains(c)));
    }

    #[test]
    fn valueless_long_flags_cover_every_docker_boolean() {
        for flag in [
            "--detach",
            "--help",
            "--init",
            "--interactive",
            "--no-healthcheck",
            "--oom-kill-disable",
            "--privileged",
            "--publish-all",
            "--quiet",
            "--read-only",
            "--rm",
            "--sig-proxy",
            "--tty",
            "--use-api-socket",
        ] {
            assert!(VALUELESS_LONG_FLAGS.contains(&flag), "{flag}");
        }
    }

    #[test]
    fn consumes_next_token_cases() {
        for (token, consumes) in [
            ("--env-file", true),
            ("--env-file=vars.env", false),
            ("--privileged", false),
            ("--read-only", false),
            ("--network", true),
            ("-v", true),
            ("-v/:/host:rw", false),
            ("-m", true),
            ("-m0", false),
            ("-it", false),
            ("-itv", true),
            ("-e", true),
        ] {
            assert_eq!(consumes_next_token(token), consumes, "{token}");
        }
    }

    #[test]
    fn approval_defaults_and_never_require_needs_no_acknowledgement() {
        let tool = DockerShellTool::builder().build().unwrap();
        assert_eq!(tool.approval_mode(), ApprovalMode::AlwaysRequire);
        assert!(tool.as_function().requires_approval());
        let open = DockerShellTool::builder()
            .approval_mode(ApprovalMode::NeverRequire)
            .host_workdir("/repo")
            .build()
            .unwrap();
        assert!(!open.as_function().requires_approval());
    }

    #[test]
    fn never_require_with_weakened_isolation_needs_acknowledgement() {
        type Weaken = fn(DockerShellToolBuilder) -> DockerShellToolBuilder;
        let cases: &[(Weaken, &str)] = &[
            (|b| b.user("0:0"), "root"),
            (|b| b.user("root"), "root"),
            (|b| b.user("0"), "root"),
            (
                |b| b.host_workdir("/").mount_readonly(false),
                "mounted writable",
            ),
            (|b| b.read_only_root(false), "root filesystem is writable"),
            (|b| b.network("bridge"), "network access"),
        ];
        for (weaken, reason) in cases {
            let never =
                || weaken(DockerShellTool::builder()).approval_mode(ApprovalMode::NeverRequire);
            let err = never().build().unwrap_err().to_string();
            assert!(err.contains("acknowledge_unsafe"), "{err}");
            assert!(err.contains(reason), "{err}");
            assert!(never().acknowledge_unsafe(true).build().is_ok());
            // With approval on, weakening needs no acknowledgement.
            assert!(weaken(DockerShellTool::builder()).build().is_ok());
        }
    }

    #[test]
    fn generates_unique_container_names() {
        let a = DockerShellTool::builder().build().unwrap();
        let b = DockerShellTool::builder().build().unwrap();
        assert_ne!(a.container_name(), b.container_name());
        assert!(a.container_name().starts_with("af-shell-"));
    }

    #[test]
    fn rejects_zero_output_limit() {
        assert!(DockerShellTool::builder()
            .max_output_bytes(0)
            .build()
            .is_err());
    }
}
