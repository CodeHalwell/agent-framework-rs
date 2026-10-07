//! Shared types: [`ShellResult`], [`ShellMode`] and [`ShellError`].

use std::time::Duration;

use agent_framework_core::types::{Content, ShellCommandOutputContent, ShellToolResultContent};

/// Whether a shell tool keeps one long-lived shell or spawns one per command.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShellMode {
    /// One long-lived shell process: `cd` and exported variables carry over
    /// from one command to the next. The default, as upstream.
    #[default]
    Persistent,
    /// A fresh process per command: nothing carries over.
    Stateless,
}

/// The outcome of one shell command.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ShellResult {
    /// Captured standard output, possibly truncated.
    pub stdout: String,
    /// Captured standard error, possibly truncated.
    pub stderr: String,
    /// The exit status reported by the shell. A process killed by a signal
    /// reports the negated signal number, as Python's `returncode` does; a
    /// status that could not be determined is `-1`.
    pub exit_code: i32,
    /// How long the command took.
    pub duration: Duration,
    /// `true` when stdout or stderr was cut down to the output limit.
    pub truncated: bool,
    /// `true` when the command was killed for exceeding its timeout.
    pub timed_out: bool,
}

impl ShellResult {
    /// The result as one text block for the model: stdout, then
    /// `stderr: ...`, the truncation and timeout markers, and the exit code.
    pub fn format_for_model(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.stdout.is_empty() {
            parts.push(self.stdout.clone());
        }
        if !self.stderr.is_empty() {
            parts.push(format!("stderr: {}", self.stderr));
        }
        if self.truncated {
            parts.push("[output truncated]".to_string());
        }
        if self.timed_out {
            parts.push("[command timed out]".to_string());
        }
        parts.push(format!("exit_code: {}", self.exit_code));
        parts.join("\n")
    }

    /// This result as a hosted-shell [`ShellCommandOutputContent`].
    pub fn to_command_output(&self) -> ShellCommandOutputContent {
        ShellCommandOutputContent {
            stdout: Some(self.stdout.clone()),
            stderr: Some(self.stderr.clone()),
            exit_code: Some(i64::from(self.exit_code)),
            timed_out: Some(self.timed_out),
        }
    }

    /// This result as a hosted-shell [`ShellToolResultContent`] answering
    /// `call_id`, with one [`ShellCommandOutputContent`] in `outputs`.
    ///
    /// `max_output_length` is the byte limit the output was held to, so a
    /// consumer can tell a truncated result from a complete one.
    pub fn to_tool_result(
        &self,
        call_id: impl Into<String>,
        max_output_length: Option<usize>,
    ) -> ShellToolResultContent {
        ShellToolResultContent {
            call_id: Some(call_id.into()),
            outputs: Some(vec![Content::ShellCommandOutput(self.to_command_output())]),
            max_output_length: max_output_length.and_then(|n| i64::try_from(n).ok()),
        }
    }
}

/// A shell-tool failure.
#[derive(Debug, thiserror::Error)]
pub enum ShellError {
    /// The [`ShellPolicy`](super::ShellPolicy) rejected the command. Upstream's
    /// `ShellCommandError`.
    #[error("Command rejected by policy: {0}")]
    Rejected(String),
    /// The shell could not be started, or the session broke. Upstream's
    /// `ShellExecutionError`.
    #[error("shell execution failed: {0}")]
    Execution(String),
    /// A command ran past its timeout and no result could be recovered.
    /// Upstream's `ShellTimeoutError`. The built-in executors report a timeout
    /// as a [`ShellResult`] with `timed_out` set instead, so this comes only
    /// from custom executors.
    #[error("shell command timed out: {0}")]
    Timeout(String),
    /// The tool was configured in a way that cannot work or is unsafe.
    #[error("invalid shell tool configuration: {0}")]
    Config(String),
    /// Any other failure from a custom [`ShellExecutor`](super::ShellExecutor).
    /// Unlike the variants above, which
    /// [`ShellEnvironmentProvider`](super::ShellEnvironmentProvider) records as
    /// a missing value, this one propagates, so a bug is not swallowed.
    #[error(transparent)]
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl ShellError {
    pub(crate) fn io(context: &str, err: std::io::Error) -> Self {
        ShellError::Execution(format!("{context}: {err}"))
    }
}

impl From<ShellError> for agent_framework_core::error::Error {
    fn from(err: ShellError) -> Self {
        agent_framework_core::error::Error::tool(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make(
        stdout: &str,
        stderr: &str,
        exit_code: i32,
        truncated: bool,
        timed_out: bool,
    ) -> ShellResult {
        ShellResult {
            stdout: stdout.into(),
            stderr: stderr.into(),
            exit_code,
            duration: Duration::from_millis(1),
            truncated,
            timed_out,
        }
    }

    #[test]
    fn format_stdout_only() {
        assert_eq!(
            make("hello", "", 0, false, false).format_for_model(),
            "hello\nexit_code: 0"
        );
    }

    #[test]
    fn format_stdout_truncated_appends_marker() {
        let text = make("part", "", 0, true, false).format_for_model();
        assert!(text.contains("[output truncated]"));
        assert!(text.starts_with("part"));
    }

    #[test]
    fn format_stderr_only_truncated_marker() {
        let text = make("", "boom", 1, true, false).format_for_model();
        assert!(text.contains("[output truncated]"));
        assert!(text.contains("stderr: boom"));
    }

    #[test]
    fn format_truncated_with_empty_streams() {
        let text = make("", "", 0, true, false).format_for_model();
        assert!(text.contains("[output truncated]"));
        assert!(text.contains("exit_code: 0"));
    }

    #[test]
    fn format_stderr_prefixed() {
        let text = make("", "boom", 1, false, false).format_for_model();
        assert!(text.contains("stderr: boom"));
        assert!(text.contains("exit_code: 1"));
    }

    #[test]
    fn format_timed_out_marker() {
        let text = make("", "", 124, false, true).format_for_model();
        assert!(text.contains("[command timed out]"));
        assert!(text.contains("exit_code: 124"));
    }

    #[test]
    fn format_empty_streams_still_reports_exit_code() {
        assert_eq!(
            make("", "", 0, false, false).format_for_model(),
            "exit_code: 0"
        );
    }

    #[test]
    fn format_combines_all_signals_in_order() {
        let text = make("out", "err", 2, true, true).format_for_model();
        let lines: Vec<&str> = text.split('\n').collect();
        assert_eq!(
            lines,
            [
                "out",
                "stderr: err",
                "[output truncated]",
                "[command timed out]",
                "exit_code: 2"
            ]
        );
    }

    #[test]
    fn converts_to_hosted_shell_content() {
        let result = make("out", "err", 3, false, true);
        let output = result.to_command_output();
        assert_eq!(output.stdout.as_deref(), Some("out"));
        assert_eq!(output.stderr.as_deref(), Some("err"));
        assert_eq!(output.exit_code, Some(3));
        assert_eq!(output.timed_out, Some(true));

        let tool_result = result.to_tool_result("call-1", Some(1024));
        assert_eq!(tool_result.call_id.as_deref(), Some("call-1"));
        assert_eq!(tool_result.max_output_length, Some(1024));
        assert_eq!(
            tool_result.outputs,
            Some(vec![Content::ShellCommandOutput(output)])
        );
    }
}
