//! Shell discovery: which shell binary runs the commands, and with what flags.

use std::path::{Path, PathBuf};

use super::types::ShellError;

/// The environment variable that overrides the default shell when no shell
/// is configured explicitly.
pub const SHELL_ENV_OVERRIDE: &str = "AGENT_FRAMEWORK_SHELL";

/// How the caller named the shell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellSpec {
    /// A command line, split with POSIX shell-word rules (`"bash --norc"`).
    Line(String),
    /// An argv used verbatim.
    Argv(Vec<String>),
}

impl From<&str> for ShellSpec {
    fn from(value: &str) -> Self {
        ShellSpec::Line(value.to_string())
    }
}

impl From<String> for ShellSpec {
    fn from(value: String) -> Self {
        ShellSpec::Line(value)
    }
}

impl From<Vec<String>> for ShellSpec {
    fn from(value: Vec<String>) -> Self {
        ShellSpec::Argv(value)
    }
}

impl From<Vec<&str>> for ShellSpec {
    fn from(value: Vec<&str>) -> Self {
        ShellSpec::Argv(value.into_iter().map(str::to_string).collect())
    }
}

impl<const N: usize> From<[&str; N]> for ShellSpec {
    fn from(value: [&str; N]) -> Self {
        ShellSpec::Argv(value.iter().map(|s| s.to_string()).collect())
    }
}

fn split_line(line: &str) -> Result<Vec<String>, ShellError> {
    let parts = shlex::split(line)
        .ok_or_else(|| ShellError::Config(format!("cannot parse shell override {line:?}")))?;
    if parts.is_empty() {
        return Err(ShellError::Config(
            "shell override must not be empty".to_string(),
        ));
    }
    Ok(parts)
}

/// Resolve the shell argv.
///
/// Priority: the explicit `shell`, then the [`SHELL_ENV_OVERRIDE`]
/// environment variable, then the platform default (`pwsh` or `powershell`
/// on Windows; `/bin/bash`, `/usr/bin/bash`, `/bin/sh`, `/usr/bin/sh`, then
/// `sh` on `PATH` elsewhere).
///
/// With `interactive` (persistent mode) the argv starts a shell that reads
/// commands from stdin. Without it (stateless mode) the argv ends with `-c`
/// (or `-Command` for PowerShell) so a command string can be appended as one
/// argument; an override that lacks the flag has it added.
pub fn resolve_shell(
    shell: Option<&ShellSpec>,
    interactive: bool,
) -> Result<Vec<String>, ShellError> {
    let finish = |parts: Vec<String>| {
        if interactive {
            parts
        } else {
            ensure_command_flag(parts)
        }
    };
    if let Some(spec) = shell {
        let parts = match spec {
            ShellSpec::Line(line) => split_line(line)?,
            ShellSpec::Argv(argv) if argv.is_empty() => {
                return Err(ShellError::Config(
                    "shell override must not be empty".to_string(),
                ))
            }
            ShellSpec::Argv(argv) => argv.clone(),
        };
        return Ok(finish(parts));
    }

    if let Ok(line) = std::env::var(SHELL_ENV_OVERRIDE) {
        if let Some(parts) = shlex::split(&line).filter(|p| !p.is_empty()) {
            return Ok(finish(parts));
        }
    }

    platform_default(interactive)
}

#[cfg(windows)]
fn platform_default(interactive: bool) -> Result<Vec<String>, ShellError> {
    let binary = which("pwsh").or_else(|| which("powershell")).ok_or_else(|| {
        ShellError::Execution(format!(
            "Neither 'pwsh' nor 'powershell' was found on PATH. Install PowerShell 7+ or set {SHELL_ENV_OVERRIDE}."
        ))
    })?;
    let mut argv = vec![
        binary.to_string_lossy().into_owned(),
        "-NoLogo".into(),
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-Command".into(),
    ];
    if interactive {
        // A persistent session reads its script from stdin via '-'.
        argv.push("-".into());
    }
    Ok(argv)
}

#[cfg(not(windows))]
fn platform_default(interactive: bool) -> Result<Vec<String>, ShellError> {
    for candidate in ["/bin/bash", "/usr/bin/bash", "/bin/sh", "/usr/bin/sh"] {
        if Path::new(candidate).exists() {
            let mut argv = vec![candidate.to_string()];
            if !interactive {
                argv.push("-c".into());
            } else if candidate.ends_with("bash") {
                argv.extend(["--noprofile".into(), "--norc".into()]);
            }
            return Ok(argv);
        }
    }
    let sh = which("sh").ok_or_else(|| {
        ShellError::Execution(format!(
            "No POSIX shell found on PATH. Set {SHELL_ENV_OVERRIDE} to override."
        ))
    })?;
    let mut argv = vec![sh.to_string_lossy().into_owned()];
    if !interactive {
        argv.push("-c".into());
    }
    Ok(argv)
}

/// Find `name` on `PATH` (with the `PATHEXT` extensions on Windows).
pub(crate) fn which(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    let exts: Vec<String> = if cfg!(windows) {
        std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT;.COM".into())
            .split(';')
            .map(str::to_string)
            .chain(std::iter::once(String::new()))
            .collect()
    } else {
        vec![String::new()]
    };
    for dir in std::env::split_paths(&path) {
        for ext in &exts {
            let candidate = dir.join(format!("{name}{ext}"));
            if is_executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Whether `argv[0]` names PowerShell.
pub fn is_powershell(argv: &[String]) -> bool {
    let Some(first) = argv.first() else {
        return false;
    };
    // Split on both separators so a Windows path is recognised on any host.
    let name = first
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(first)
        .to_ascii_lowercase();
    matches!(
        name.as_str(),
        "pwsh" | "pwsh.exe" | "powershell" | "powershell.exe"
    )
}

/// Make sure a stateless argv ends with the flag that takes a command
/// string. Without it a POSIX shell would read the command as a script path.
pub(crate) fn ensure_command_flag(mut argv: Vec<String>) -> Vec<String> {
    let Some(last) = argv.last().map(|s| s.to_ascii_lowercase()) else {
        return argv;
    };
    if is_powershell(&argv) {
        if last != "-command" && last != "-c" {
            argv.push("-Command".into());
        }
    } else if last != "-c" {
        argv.push("-c".into());
    }
    argv
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn empty_string_override_rejected() {
        let err = resolve_shell(Some(&"".into()), true).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn whitespace_string_override_rejected() {
        let err = resolve_shell(Some(&"   ".into()), false).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn empty_argv_override_rejected() {
        let err = resolve_shell(Some(&ShellSpec::Argv(vec![])), true).unwrap_err();
        assert!(err.to_string().contains("must not be empty"));
    }

    #[test]
    fn unbalanced_quotes_rejected() {
        assert!(resolve_shell(Some(&"bash '-c".into()), true).is_err());
    }

    #[test]
    fn stateless_appends_dash_c_for_posix_shell_without_flag() {
        assert_eq!(
            resolve_shell(Some(&"/bin/bash".into()), false).unwrap(),
            s(&["/bin/bash", "-c"])
        );
    }

    #[test]
    fn stateless_appends_command_for_pwsh_without_flag() {
        let argv = resolve_shell(Some(&"/usr/bin/pwsh -NoProfile".into()), false).unwrap();
        assert_eq!(argv, s(&["/usr/bin/pwsh", "-NoProfile", "-Command"]));
    }

    #[test]
    fn stateless_preserves_existing_dash_c_flag() {
        assert_eq!(
            resolve_shell(Some(&"/bin/bash -c".into()), false).unwrap(),
            s(&["/bin/bash", "-c"])
        );
    }

    #[test]
    fn stateless_preserves_existing_pwsh_command_flag() {
        let argv = resolve_shell(Some(&"pwsh -NoProfile -Command".into()), false).unwrap();
        assert_eq!(argv.iter().filter(|a| *a == "-Command").count(), 1);
        assert_eq!(argv.last().unwrap(), "-Command");
    }

    #[test]
    fn interactive_does_not_append_command_flag() {
        let argv = resolve_shell(Some(&"/bin/bash --noprofile".into()), true).unwrap();
        assert_eq!(argv, s(&["/bin/bash", "--noprofile"]));
    }

    #[test]
    fn argv_override_is_used_verbatim() {
        let argv = resolve_shell(Some(&["my shell", "--flag"].into()), true).unwrap();
        assert_eq!(argv, s(&["my shell", "--flag"]));
    }

    #[test]
    fn is_powershell_and_command_flag_helpers() {
        assert!(!is_powershell(&[]));
        assert!(ensure_command_flag(vec![]).is_empty());
        assert!(is_powershell(&s(&[
            "C:\\Program Files\\PowerShell\\7\\pwsh.exe"
        ])));
        assert!(is_powershell(&s(&["/usr/bin/PWSH"])));
        assert!(!is_powershell(&s(&["/bin/bash"])));
    }

    #[cfg(unix)]
    #[test]
    fn posix_default_ends_with_dash_c_when_stateless() {
        // Only meaningful when no override is set in the test environment.
        if std::env::var_os(SHELL_ENV_OVERRIDE).is_some() {
            return;
        }
        let argv = resolve_shell(None, false).unwrap();
        assert_eq!(argv.last().unwrap(), "-c");
        let interactive = resolve_shell(None, true).unwrap();
        assert!(!interactive.contains(&"-c".to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn which_finds_sh() {
        assert!(which("sh").is_some());
        assert!(which("definitely-not-a-real-binary-af").is_none());
    }
}
