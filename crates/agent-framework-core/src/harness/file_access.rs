//! File access: a [`ContextProvider`] that gives an agent tools to read,
//! write, search and edit files in an [`AgentFileStore`].
//!
//! Port of the tool half of upstream `_harness/_file_access.py`
//! (`FileAccessProvider`; .NET `FileAccessProvider`). The provider adds its
//! instructions and eight tools to every run:
//!
//! | Tool | Does | Kind |
//! |---|---|---|
//! | `file_access_read` | Read a file. | read-only |
//! | `file_access_read_lines` | Read a 1-based inclusive line range, each line prefixed `"{n}\t"`. | read-only |
//! | `file_access_ls` | List a directory's direct children. | read-only |
//! | `file_access_grep` | Recursively search contents with a case-insensitive regex. | read-only |
//! | `file_access_write` | Write a file; refuses to overwrite unless `overwrite` is true. | write |
//! | `file_access_delete` | Delete a file. | write |
//! | `file_access_replace` | Replace a substring in a file. | write |
//! | `file_access_replace_lines` | Replace whole lines in a file. | write |
//!
//! Unlike [`FileMemoryProvider`](super::FileMemoryProvider), the store is
//! shared: by default every session and agent sees the same files. With
//! [`FileAccessProvider::session_scoped`] (or a
//! [`scope`](FileAccessProvider::with_scope)) the tools are confined to a
//! workspace folder under `~access-/`, derived from the scope or the session
//! id with [`storage_key_segment`].
//!
//! # Approval
//!
//! Every tool requires approval by default
//! ([`ApprovalMode::AlwaysRequire`]), so run the agent behind a
//! [`ToolApprovalAgent`](super::ToolApprovalAgent) (or answer the approval
//! requests yourself). [`FileAccessProvider::disable_readonly_tool_approval`]
//! and [`FileAccessProvider::disable_write_tool_approval`] register the
//! affected tools with [`ApprovalMode::NeverRequire`] instead, and
//! [`FileAccessProvider::read_only_tools_auto_approval_rule`] /
//! [`FileAccessProvider::all_tools_auto_approval_rule`] are auto-approval
//! rules for the default names. With a
//! [`tool_name_prefix`](FileAccessProvider::with_tool_name_prefix), use the
//! instance's [`read_only_tools_approval_rule`](FileAccessProvider::read_only_tools_approval_rule)
//! / [`all_tools_approval_rule`](FileAccessProvider::all_tools_approval_rule).
//!
//! The rules approve by tool name, so make sure no other tool is registered
//! under one of these names.
//!
//! # Divergences
//!
//! - **Hosted calls.** Upstream's rules refuse a call marked `hosted`. This
//!   port's `FunctionCallContent` has no hosted marker (hosted tool calls are
//!   separate content variants and never reach the rules), so the rules
//!   compare the name only.
//! - **Write lock key.** Edits hold a lock per store instance and folder,
//!   shared with [`FileMemoryProvider`](super::FileMemoryProvider). Upstream
//!   folds the folder to NFC before lowercasing it; this port lowercases
//!   only, so two Unicode spellings of one folder get separate locks.
//! - **Error text.** A store error's own message is reported after
//!   `Could not … file '…':`. Upstream maps "is a directory" and "parent is a
//!   file" to fixed sentences; the Rust stores' messages say the same thing
//!   in their own words.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::storage_keys::storage_key_segment;
use crate::tools::{ApprovalMode, FunctionTool, ToolDefinition};
use crate::types::FunctionCallContent;

use super::file_store::{
    apply_replace, apply_replace_lines, compile_glob, normalize_relative_path, slice_lines,
    store_write_lock, AgentFileStore, FileLineEdit,
};

/// The default `source_id` of a [`FileAccessProvider`].
pub const DEFAULT_FILE_ACCESS_SOURCE_ID: &str = "file_access";

/// The default instructions a [`FileAccessProvider`] adds to every run.
pub const DEFAULT_FILE_ACCESS_INSTRUCTIONS: &str = "## File Access\n\
You have access to a shared file storage area via the `file_access_*` tools \
for reading, writing, and managing files.\n\
These files persist beyond the current session and may be shared across \
sessions or agents.\n\
Use these tools to read input data provided by the user, write output \
artifacts, and manage any files the user has asked you to work with.\n\n\
- Never delete or overwrite existing files unless the user has explicitly \
asked you to do so.\n\
- Files may be organized into subdirectories. Use `file_access_ls` \
to explore the tree level by level, \
or `file_access_grep` to search file contents recursively across \
the whole store.\n\
- To change part of a file, find the line numbers with `file_access_grep`, \
read the range around them with `file_access_read_lines`, then edit with \
`file_access_replace_lines`. Reading the whole file first is rarely necessary.";

/// Appended to the instructions in session-scoped mode.
const SESSION_SCOPED_INSTRUCTIONS_SUFFIX: &str = "\n- Your file workspace is isolated to the current session or configured scope: files written \
here are not visible outside that workspace.";

/// Prefix for an encoded session or scope folder name.
const ENCODED_SESSION_PREFIX: &str = "~access-";
/// The namespace folder every scoped workspace lives in, so a workspace never
/// shares a folder with a [`FileMemoryProvider`](super::FileMemoryProvider)
/// working folder on the same store.
const WORKSPACE_NAMESPACE: &str = "~access-";

/// Name of the tool that writes a file.
pub const FILE_ACCESS_WRITE_TOOL_NAME: &str = "file_access_write";
/// Name of the tool that reads a file.
pub const FILE_ACCESS_READ_TOOL_NAME: &str = "file_access_read";
/// Name of the tool that reads a range of lines.
pub const FILE_ACCESS_READ_LINES_TOOL_NAME: &str = "file_access_read_lines";
/// Name of the tool that deletes a file.
pub const FILE_ACCESS_DELETE_TOOL_NAME: &str = "file_access_delete";
/// Name of the tool that lists a directory.
pub const FILE_ACCESS_LS_TOOL_NAME: &str = "file_access_ls";
/// Name of the tool that searches file contents.
pub const FILE_ACCESS_GREP_TOOL_NAME: &str = "file_access_grep";
/// Name of the tool that replaces a substring.
pub const FILE_ACCESS_REPLACE_TOOL_NAME: &str = "file_access_replace";
/// Name of the tool that replaces whole lines.
pub const FILE_ACCESS_REPLACE_LINES_TOOL_NAME: &str = "file_access_replace_lines";

/// The tools that only read the store.
pub const FILE_ACCESS_READ_ONLY_TOOL_NAMES: [&str; 4] = [
    FILE_ACCESS_READ_TOOL_NAME,
    FILE_ACCESS_READ_LINES_TOOL_NAME,
    FILE_ACCESS_LS_TOOL_NAME,
    FILE_ACCESS_GREP_TOOL_NAME,
];

/// The tools that modify the store.
pub const FILE_ACCESS_WRITE_TOOL_NAMES: [&str; 4] = [
    FILE_ACCESS_WRITE_TOOL_NAME,
    FILE_ACCESS_DELETE_TOOL_NAME,
    FILE_ACCESS_REPLACE_TOOL_NAME,
    FILE_ACCESS_REPLACE_LINES_TOOL_NAME,
];

/// A [`ContextProvider`] that gives an agent access to a shared file store.
/// See the [module docs](self).
#[derive(Clone)]
pub struct FileAccessProvider {
    store: Arc<dyn AgentFileStore>,
    source_id: String,
    instructions: Option<String>,
    disable_write_tools: bool,
    disable_readonly_tool_approval: bool,
    disable_write_tool_approval: bool,
    session_scoped: bool,
    scope: Option<String>,
    tool_name_prefix: Option<String>,
}

impl std::fmt::Debug for FileAccessProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileAccessProvider")
            .field("source_id", &self.source_id)
            .field("disable_write_tools", &self.disable_write_tools)
            .field("session_scoped", &self.session_scoped)
            .field("scope", &self.scope)
            .field("tool_name_prefix", &self.tool_name_prefix)
            .finish_non_exhaustive()
    }
}

impl FileAccessProvider {
    /// A provider over `store`, which should already be scoped to the folder
    /// or storage location the agent may use.
    pub fn new(store: Arc<dyn AgentFileStore>) -> Self {
        Self {
            store,
            source_id: DEFAULT_FILE_ACCESS_SOURCE_ID.to_string(),
            instructions: None,
            disable_write_tools: false,
            disable_readonly_tool_approval: false,
            disable_write_tool_approval: false,
            session_scoped: false,
            scope: None,
            tool_name_prefix: None,
        }
    }

    /// Set the provider's `source_id`.
    pub fn with_source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// Replace the default instructions.
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into()).filter(|s| !s.is_empty());
        self
    }

    /// Advertise only the read-only tools.
    pub fn disable_write_tools(mut self) -> Self {
        self.disable_write_tools = true;
        self
    }

    /// Let the read-only tools run without approval.
    pub fn disable_readonly_tool_approval(mut self) -> Self {
        self.disable_readonly_tool_approval = true;
        self
    }

    /// Let the write tools run without approval.
    pub fn disable_write_tool_approval(mut self) -> Self {
        self.disable_write_tool_approval = true;
        self
    }

    /// Confine the tools to a workspace derived from the session id (or the
    /// [`scope`](Self::with_scope), when set).
    pub fn session_scoped(mut self) -> Self {
        self.session_scoped = true;
        self
    }

    /// Confine the tools to a workspace for `scope` (for example a user or
    /// tenant id), shared by every session with that scope. A non-empty
    /// scope enables scoped mode by itself. The scope is an opaque key, not
    /// a path.
    pub fn with_scope(mut self, scope: impl Into<String>) -> Self {
        self.scope = Some(scope.into()).filter(|s| !s.is_empty());
        self
    }

    /// Prefix every tool name with `"{prefix}_"`, to avoid collisions when an
    /// agent has more than one file-access provider. The default
    /// instructions are rewritten to the prefixed names.
    pub fn with_tool_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tool_name_prefix = Some(prefix.into()).filter(|s| !s.is_empty());
        self
    }

    /// The provider's `source_id`.
    pub fn source_id(&self) -> &str {
        &self.source_id
    }

    /// `name` with the configured prefix.
    pub fn tool_name(&self, name: &str) -> String {
        match &self.tool_name_prefix {
            Some(prefix) => format!("{prefix}_{name}"),
            None => name.to_string(),
        }
    }

    /// The instructions this provider adds, before any scoped-mode suffix.
    pub fn instructions(&self) -> String {
        match (&self.instructions, &self.tool_name_prefix) {
            (Some(custom), _) => custom.clone(),
            (None, Some(prefix)) => DEFAULT_FILE_ACCESS_INSTRUCTIONS
                .replace("file_access_", &format!("{prefix}_file_access_")),
            (None, None) => DEFAULT_FILE_ACCESS_INSTRUCTIONS.to_string(),
        }
    }

    /// An auto-approval rule for the read-only tools under their default
    /// names. Pass it to
    /// [`ToolApprovalAgent::with_auto_approval_rule`](super::ToolApprovalAgent::with_auto_approval_rule).
    pub fn read_only_tools_auto_approval_rule(call: &FunctionCallContent) -> bool {
        FILE_ACCESS_READ_ONLY_TOOL_NAMES.contains(&call.name.as_str())
    }

    /// An auto-approval rule for every tool under its default name.
    pub fn all_tools_auto_approval_rule(call: &FunctionCallContent) -> bool {
        FILE_ACCESS_READ_ONLY_TOOL_NAMES.contains(&call.name.as_str())
            || FILE_ACCESS_WRITE_TOOL_NAMES.contains(&call.name.as_str())
    }

    /// An auto-approval rule for this provider's read-only tool names,
    /// prefix included.
    pub fn read_only_tools_approval_rule(
        &self,
    ) -> impl Fn(&FunctionCallContent) -> bool + Send + Sync + 'static {
        let names: Vec<String> = FILE_ACCESS_READ_ONLY_TOOL_NAMES
            .iter()
            .map(|n| self.tool_name(n))
            .collect();
        move |call| names.contains(&call.name)
    }

    /// An auto-approval rule for all of this provider's tool names, prefix
    /// included.
    pub fn all_tools_approval_rule(
        &self,
    ) -> impl Fn(&FunctionCallContent) -> bool + Send + Sync + 'static {
        let names: Vec<String> = FILE_ACCESS_READ_ONLY_TOOL_NAMES
            .iter()
            .chain(FILE_ACCESS_WRITE_TOOL_NAMES.iter())
            .map(|n| self.tool_name(n))
            .collect();
        move |call| names.contains(&call.name)
    }

    /// The workspace folder for a run, or `""` when the provider is not
    /// scoped.
    ///
    /// Fails in scoped mode when there is neither a scope nor a session id,
    /// because falling back to the store root would expose every other
    /// session's files.
    pub fn workspace(&self, session_id: Option<&str>) -> Result<String> {
        if !self.session_scoped && self.scope.is_none() {
            return Ok(String::new());
        }
        let raw = self
            .scope
            .as_deref()
            .or(session_id)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::Configuration(
                    "FileAccessProvider session-scoped mode requires a scope: set one with with_scope or run \
                     with a session that has a session id. Without one, files cannot be isolated from other sessions."
                        .into(),
                )
            })?;
        let segment = storage_key_segment(raw, ENCODED_SESSION_PREFIX);
        Ok(combine(WORKSPACE_NAMESPACE, &segment))
    }

    /// The tools for a run in `workspace` (`""` for the store root): the
    /// read-only ones, then the write ones unless they are disabled.
    pub fn tools(&self, workspace: &str) -> Vec<ToolDefinition> {
        let ctx = ToolCtx {
            store: self.store.clone(),
            workspace: workspace.to_string(),
        };
        let readonly = if self.disable_readonly_tool_approval {
            ApprovalMode::NeverRequire
        } else {
            ApprovalMode::AlwaysRequire
        };
        let write = if self.disable_write_tool_approval {
            ApprovalMode::NeverRequire
        } else {
            ApprovalMode::AlwaysRequire
        };
        let mut tools = vec![
            self.read_tool(&ctx, readonly),
            self.read_lines_tool(&ctx, readonly),
            self.ls_tool(&ctx, readonly),
            self.grep_tool(&ctx, readonly),
        ];
        if !self.disable_write_tools {
            tools.extend([
                self.write_tool(&ctx, write),
                self.delete_tool(&ctx, write),
                self.replace_tool(&ctx, write),
                self.replace_lines_tool(&ctx, write),
            ]);
        }
        tools
    }

    fn write_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to write.
            file_name: String,
            /// Full text content to write to the file.
            content: String,
            /// When true, replace an existing file; otherwise writing fails if it exists.
            #[serde(default)]
            overwrite: bool,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_WRITE_TOOL_NAME),
            "Write a file with the given name and content. By default, does not overwrite an existing file \
             unless overwrite is set to true.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let path = match ctx.path(name) {
                        Ok(path) => path,
                        Err(e) => return Ok(text(format!("Could not write file '{name}': {}", detail(&e)))),
                    };
                    let lock = store_write_lock(&ctx.store, parent_folder(&path));
                    let _guard = lock.lock().await;
                    let outcome = if args.overwrite {
                        ctx.store.write(&path, &args.content).await
                    } else {
                        ctx.store.create_new(&path, &args.content).await
                    };
                    Ok(text(match outcome {
                        Ok(()) => format!("File '{name}' written."),
                        Err(e) if !args.overwrite && detail(&e).contains("already exists") => format!(
                            "File '{name}' already exists. To replace it, write again with overwrite set to true."
                        ),
                        Err(e) => format!("Could not write file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn read_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to read.
            file_name: String,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_READ_TOOL_NAME),
            "Read the content of a file by name. Returns the file content or a message indicating the file could \
             not be read. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each line \
             keeps its own terminator, and content ending in a newline has a final empty line.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let outcome = async { ctx.store.read(&ctx.path(name)?).await }.await;
                    Ok(text(match outcome {
                        Ok(Some(content)) => content,
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not read file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn read_lines_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to read.
            file_name: String,
            /// 1-based line number to read from, inclusive.
            start_line: i64,
            /// 1-based line number to read to, inclusive. Omit to read to the end of the file; a value past the
            /// last line is clamped to it.
            #[serde(default)]
            end_line: Option<i64>,
        }
        let ctx = ctx.clone();
        let replace_lines = self.tool_name(FILE_ACCESS_REPLACE_LINES_TOOL_NAME);
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_READ_LINES_TOOL_NAME),
            format!(
                "Read part of a file by 1-based inclusive line number; omit end_line to read to the end of the \
                 file, and an end_line past the last line is clamped. Each line is prefixed with its number and a \
                 tab; everything after that tab is verbatim, including the line's own terminator, so it can be \
                 reused as a `new_line` value by `{replace_lines}`. Line numbers count lines split on \\n only: a \
                 lone \\r never starts a new line, each line keeps its own terminator, and content ending in a \
                 newline has a final empty line."
            ),
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let outcome = async {
                        let Some(content) = ctx.store.read(&ctx.path(name)?).await? else {
                            return Ok(None);
                        };
                        let start = positive(args.start_line, "start_line")?;
                        let end = args.end_line.map(|e| positive(e, "end_line")).transpose()?;
                        let lines = slice_lines(&content, start, end)?;
                        // Each line keeps its terminator, so it doubles as
                        // the row separator.
                        Ok::<_, Error>(Some(
                            lines
                                .iter()
                                .enumerate()
                                .map(|(i, line)| format!("{}\t{line}", start + i))
                                .collect::<String>(),
                        ))
                    }
                    .await;
                    Ok(text(match outcome {
                        Ok(Some(rows)) => rows,
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not read lines from file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn delete_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to delete.
            file_name: String,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_DELETE_TOOL_NAME),
            "Delete a file by name.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let outcome = async {
                        let path = ctx.path(name)?;
                        let lock = store_write_lock(&ctx.store, parent_folder(&path));
                        let _guard = lock.lock().await;
                        ctx.store.delete(&path).await
                    }
                    .await;
                    Ok(text(match outcome {
                        Ok(true) => format!("File '{name}' deleted."),
                        Ok(false) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not delete file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn ls_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Relative directory to list; omit or pass empty to list the root.
            #[serde(default)]
            directory: Option<String>,
            /// Optional glob (e.g. "*.md") matched against entry names to filter the listing.
            #[serde(default)]
            glob_pattern: Option<String>,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_LS_TOOL_NAME),
            "List the direct child files and subdirectories of a directory. Omit `directory` (or pass an empty \
             string) to list the root. To enumerate a subdirectory, pass its relative path, for example \
             \"reports\" or \"reports/2024\". Optionally filter entries with a `glob_pattern` (e.g. \"*.md\"). \
             Subdirectories are listed before files, and each entry is {\"name\": <name>, \"type\": \
             \"file\"|\"directory\"}.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let shown = args.directory.clone().unwrap_or_default();
                    let outcome = async {
                        let (target, _) = ctx.directory(args.directory.as_deref())?;
                        let glob = compile_glob(args.glob_pattern.as_deref())?;
                        let entries = ctx.store.list_children(&target).await?;
                        Ok::<_, Error>(
                            entries
                                .into_iter()
                                .filter(|e| glob.matches(&e.name))
                                .map(|e| json!({"name": e.name, "type": e.entry_type}))
                                .collect::<Vec<_>>(),
                        )
                    }
                    .await;
                    Ok(match outcome {
                        Ok(entries) => Value::Array(entries),
                        Err(e) => text(format!("Could not list directory '{shown}': {}", detail(&e))),
                    })
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn grep_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Case-insensitive regex matched against file contents; 256 characters or fewer.
            regex_pattern: String,
            /// Optional glob to filter which files are searched (e.g. "*.md", "reports/*").
            #[serde(default)]
            glob_pattern: Option<String>,
            /// Optional relative directory to search; omit or pass empty for the root.
            #[serde(default)]
            directory: Option<String>,
        }
        let ctx = ctx.clone();
        let (read, replace_lines, read_lines) = (
            self.tool_name(FILE_ACCESS_READ_TOOL_NAME),
            self.tool_name(FILE_ACCESS_REPLACE_LINES_TOOL_NAME),
            self.tool_name(FILE_ACCESS_READ_LINES_TOOL_NAME),
        );
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_GREP_TOOL_NAME),
            format!(
                "Search the contents of files in the store using a case-insensitive regular expression. The \
                 search runs recursively across all subdirectories. Optionally restrict the search to a \
                 `directory` (relative path), and filter which files to search using a glob `glob_pattern` \
                 matched against each file's path relative to that directory. The glob's `*` matches any \
                 characters including `/`: use \"*.md\" to match markdown files at any depth, or \"reports/*\" to \
                 restrict the search to the `reports` subtree. Leave empty or omit to search all files. Returns \
                 matching results whose file_name values are paths relative to the store root (directly usable \
                 with `{read}`), along with snippets and matching lines with line numbers. Each matching line is \
                 reported verbatim, including its own line terminator, so it can normally be reused as a \
                 `new_line` value by `{replace_lines}`; prefer `{read_lines}` when the exact text matters. Line \
                 numbers count lines split on \\n only, with a final empty line when content ends in a newline. \
                 The regex_pattern must be 256 characters or fewer."
            ),
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let glob = args.glob_pattern.as_deref().filter(|g| !g.trim().is_empty());
                    let outcome = async {
                        let (target, relative) = ctx.directory(args.directory.as_deref())?;
                        let results = ctx.store.search(&target, &args.regex_pattern, glob, true).await?;
                        // The store reports names relative to the searched
                        // directory; re-root them so they compose with read.
                        let prefix = relative.trim_matches('/');
                        Ok::<_, Error>(
                            results
                                .into_iter()
                                .map(|mut r| {
                                    if !prefix.is_empty() {
                                        r.file_name = format!("{prefix}/{}", r.file_name);
                                    }
                                    r
                                })
                                .collect::<Vec<_>>(),
                        )
                    }
                    .await;
                    Ok(match outcome {
                        Ok(results) => serde_json::to_value(results)?,
                        Err(e) => text(format!("Could not search files: {}", detail(&e))),
                    })
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn replace_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to modify.
            file_name: String,
            /// Substring to find and replace.
            old_string: String,
            /// Replacement text.
            new_string: String,
            /// When true, replace every occurrence; when false, fail unless exactly one occurrence exists.
            #[serde(default)]
            replace_all: bool,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_REPLACE_TOOL_NAME),
            "Replace occurrences of old_string with new_string in a file. Fails if old_string is not found, or if \
             it occurs more than once and replace_all is false. Returns the number of occurrences replaced.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let outcome = async {
                        let path = ctx.path(name)?;
                        let lock = store_write_lock(&ctx.store, parent_folder(&path));
                        let _guard = lock.lock().await;
                        let Some(content) = ctx.store.read(&path).await? else {
                            return Ok(None);
                        };
                        let (updated, count) =
                            apply_replace(&content, &args.old_string, &args.new_string, args.replace_all)?;
                        ctx.store.write(&path, &updated).await?;
                        Ok::<_, Error>(Some(count))
                    }
                    .await;
                    Ok(text(match outcome {
                        Ok(Some(count)) => format!("Replaced {count} occurrence(s) in '{name}'."),
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not replace in file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }

    fn replace_lines_tool(&self, ctx: &ToolCtx, mode: ApprovalMode) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name (relative path) of the file to modify.
            file_name: String,
            /// List of 1-based line numbers and their literal replacement text.
            edits: Vec<FileLineEdit>,
        }
        let ctx = ctx.clone();
        FunctionTool::typed(
            self.tool_name(FILE_ACCESS_REPLACE_LINES_TOOL_NAME),
            "Replace lines in a file. Provide a list of edits, each with a 1-based line_number and a literal \
             new_line (include your own trailing newline); an empty new_line deletes the line, including its line \
             break. Fails on out-of-range or duplicate line numbers. Line numbers count lines split on \\n only: a \
             lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline \
             has a final empty line.",
            move |args: Args| {
                let ctx = ctx.clone();
                async move {
                    let name = &args.file_name;
                    let outcome = async {
                        let path = ctx.path(name)?;
                        let lock = store_write_lock(&ctx.store, parent_folder(&path));
                        let _guard = lock.lock().await;
                        let Some(content) = ctx.store.read(&path).await? else {
                            return Ok(false);
                        };
                        let updated = apply_replace_lines(&content, &args.edits)?;
                        ctx.store.write(&path, &updated).await?;
                        Ok::<_, Error>(true)
                    }
                    .await;
                    Ok(text(match outcome {
                        Ok(true) => format!("Replaced {} line(s) in '{name}'.", args.edits.len()),
                        Ok(false) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not edit file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .with_approval_mode(mode)
        .into_definition()
    }
}

/// What every tool closure needs: the store and the run's workspace.
#[derive(Clone)]
struct ToolCtx {
    store: Arc<dyn AgentFileStore>,
    workspace: String,
}

impl ToolCtx {
    /// The store path of file `name`.
    fn path(&self, name: &str) -> Result<String> {
        Ok(combine(
            &self.workspace,
            &normalize_relative_path(name, false)?,
        ))
    }

    /// The store path of `directory` and its normalized relative form.
    fn directory(&self, directory: Option<&str>) -> Result<(String, String)> {
        let raw = directory.map(str::trim).unwrap_or_default();
        let relative = if raw.is_empty() {
            String::new()
        } else {
            normalize_relative_path(raw, true)?
        };
        Ok((combine(&self.workspace, &relative), relative))
    }
}

#[async_trait::async_trait]
impl ContextProvider for FileAccessProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session_id = ctx
            .session_id
            .clone()
            .or_else(|| ctx.session.as_ref().map(|s| s.session_id().to_string()));
        let workspace = self.workspace(session_id.as_deref())?;
        let mut instructions = self.instructions();
        if !workspace.is_empty() {
            tracing::debug!(workspace = %workspace, "session-scoped file access");
            self.store.create_directory(&workspace).await?;
            instructions.push_str(SESSION_SCOPED_INSTRUCTIONS_SUFFIX);
        }
        ctx.add_instructions(instructions);
        ctx.tools.extend(self.tools(&workspace));
        Ok(())
    }
}

fn positive(value: i64, field: &str) -> Result<usize> {
    if value < 1 {
        return Err(Error::tool(format!(
            "{field} must be a positive integer, got {value}."
        )));
    }
    Ok(value as usize)
}

/// The folder of a normalized store path (`""` at the root).
fn parent_folder(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(parent, _)| parent)
}

fn combine(folder: &str, name: &str) -> String {
    match (folder.is_empty(), name.is_empty()) {
        (true, _) => name.to_string(),
        (false, true) => folder.to_string(),
        (false, false) => format!(
            "{}/{}",
            folder.trim_end_matches('/'),
            name.trim_start_matches('/')
        ),
    }
}

fn text(s: String) -> Value {
    Value::String(s)
}

/// An error's message without the variant's prefix, for a tool result.
fn detail(e: &Error) -> String {
    match e {
        Error::Tool(m) | Error::Other(m) | Error::Configuration(m) => m.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixed_instructions_and_names() {
        let p = FileAccessProvider::new(Arc::new(super::super::InMemoryAgentFileStore::new()))
            .with_tool_name_prefix("docs");
        assert_eq!(p.tool_name("file_access_read"), "docs_file_access_read");
        assert!(p.instructions().contains("`docs_file_access_ls`"));
        assert!(!p.instructions().contains(" `file_access_ls`"));
    }

    #[test]
    fn workspace_needs_a_scope_only_when_scoped() {
        let store: Arc<dyn AgentFileStore> = Arc::new(super::super::InMemoryAgentFileStore::new());
        let shared = FileAccessProvider::new(store.clone());
        assert_eq!(shared.workspace(None).unwrap(), "");
        let scoped = FileAccessProvider::new(store.clone()).session_scoped();
        assert!(scoped.workspace(None).is_err());
        assert_eq!(scoped.workspace(Some("s1")).unwrap(), "~access-/s1");
        let by_scope = FileAccessProvider::new(store).with_scope("tenants/alice");
        let ws = by_scope.workspace(Some("s1")).unwrap();
        assert!(ws.starts_with("~access-/~access-"), "{ws}");
        assert_eq!(ws.matches('/').count(), 1, "{ws}");
    }

    #[test]
    fn parent_folders() {
        assert_eq!(parent_folder("a.md"), "");
        assert_eq!(parent_folder("x/y/a.md"), "x/y");
    }
}
