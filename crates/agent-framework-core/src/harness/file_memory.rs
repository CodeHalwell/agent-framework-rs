//! File-based memory: a [`ContextProvider`] that gives an agent tools to
//! keep notes as files in an [`AgentFileStore`].
//!
//! Port of upstream `_harness/_file_memory.py` (.NET `FileMemoryProvider`,
//! Go `agent/harness/filememory`). [`FileMemoryProvider`] contributes seven
//! tools, its usage instructions, and a `user` message carrying the memory
//! index:
//!
//! | Tool | Does |
//! |---|---|
//! | `file_memory_write` | Write a file, with an optional description. |
//! | `file_memory_read` | Read a file. |
//! | `file_memory_delete` | Delete a file and its description. |
//! | `file_memory_ls` | List files with their descriptions. |
//! | `file_memory_grep` | Search file contents with a regex. |
//! | `file_memory_replace` | Replace a substring in a file. |
//! | `file_memory_replace_lines` | Replace whole lines in a file. |
//!
//! Memories live in one flat working folder per scope. A description is
//! kept in a companion file (`notes.md` → `notes.md_description.md`), and a
//! `memories.md` index of up to 50 files is rebuilt on every write and
//! delete. Both are hidden from the agent's listings and searches.
//!
//! # Scope
//!
//! By default each session gets its own working folder, derived from its
//! session id. [`FileMemoryProvider::with_scope`] groups memories some
//! other way, for example per user. The scope is an opaque key, not a path:
//! [`storage_key_segment`] maps it to exactly one folder name, so two
//! different scopes never share a folder and `"tenants/alice"` is one folder,
//! not two.
//!
//! # Divergences
//!
//! - **Scope encoding.** Upstream encodes a scope that is not usable as a
//!   folder name in base32, and past a length limit falls back to a SHA-256
//!   digest. This port uses the crate's hex [`storage_key_segment`] and
//!   refuses a scope whose folder name would pass 255 bytes, so the mapping
//!   stays injective rather than collision-resistant. Folder names for such
//!   scopes therefore differ from upstream's.
//! - **Write lock.** Mutating tools hold a lock for their whole
//!   read-modify-write and index rebuild. Upstream Python shares that lock
//!   per store and folder across every provider in the process; here it
//!   belongs to the provider (as in .NET), which is enough while this is the
//!   only provider writing through a store.

use std::sync::Arc;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::storage_keys::storage_key_segment;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::Message;

use super::file_store::{
    apply_replace, apply_replace_lines, matches_glob, normalize_relative_path, AgentFileStore,
    FileLineEdit,
};

/// The default `source_id` of a [`FileMemoryProvider`].
pub const DEFAULT_FILE_MEMORY_SOURCE_ID: &str = "file_memory";

/// The default instructions a [`FileMemoryProvider`] adds to every run.
pub const DEFAULT_FILE_MEMORY_INSTRUCTIONS: &str = "## File Based Memory\n\
You have access to a session-scoped, file-based memory system via the `file_memory_*` tools \
for storing and retrieving information across interactions. \
These files act as your working memory for the current session and are isolated from other sessions. \
Use these tools to store plans, memories, processing results, or downloaded data.\n\n\
- Use descriptive file names (e.g., \"projectarchitecture.md\", \"userpreferences.md\").\n\
- Include a description when writing a file to help with future discovery.\n\
- Before starting new tasks, use file_memory_ls and file_memory_grep to check for \
relevant existing memories to avoid duplicate work.\n\
- Keep memories up-to-date by overwriting files when information changes.\n\
- When you receive large amounts of data (e.g., downloaded web pages, API responses, research results), \
write them to files if they will be required later, so that they are not lost when older context is \
compacted or truncated. This ensures important data remains accessible across long-running sessions.";

const DESCRIPTION_SUFFIX: &str = "_description.md";
const MEMORY_INDEX_FILE_NAME: &str = "memories.md";
const MAX_INDEX_ENTRIES: usize = 50;
const ENCODED_SCOPE_PREFIX: &str = "~scope-";
const MAX_FOLDER_NAME_BYTES: usize = 255;
/// The common file-name limit, which a description file's name must fit.
const MAX_FILE_NAME_BYTES: usize = 255;
const INDEX_INTRODUCTION: &str =
    "The following is your memory index \u{2014} a list of files you have previously written. \
You can read any of these files using the file_memory_read tool.\n\n";

/// A [`ContextProvider`] that gives an agent file-based memory. See the
/// [module docs](self).
#[derive(Clone)]
pub struct FileMemoryProvider {
    inner: Arc<Inner>,
}

struct Inner {
    store: Arc<dyn AgentFileStore>,
    source_id: String,
    scope: Option<String>,
    instructions: String,
    write_lock: tokio::sync::Mutex<()>,
}

impl std::fmt::Debug for FileMemoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileMemoryProvider")
            .field("source_id", &self.inner.source_id)
            .field("scope", &self.inner.scope)
            .finish_non_exhaustive()
    }
}

impl FileMemoryProvider {
    /// A provider keeping memories in `store`, one folder per session.
    pub fn new(store: Arc<dyn AgentFileStore>) -> Self {
        Self {
            inner: Arc::new(Inner {
                store,
                source_id: DEFAULT_FILE_MEMORY_SOURCE_ID.to_string(),
                scope: None,
                instructions: DEFAULT_FILE_MEMORY_INSTRUCTIONS.to_string(),
                write_lock: tokio::sync::Mutex::new(()),
            }),
        }
    }

    fn update(mut self, f: impl FnOnce(&mut Inner)) -> Self {
        let inner = Arc::get_mut(&mut self.inner).expect("builder methods run before sharing");
        f(inner);
        self
    }

    /// Set the provider's `source_id`.
    pub fn with_source_id(self, source_id: impl Into<String>) -> Self {
        let source_id = source_id.into();
        self.update(|i| i.source_id = source_id)
    }

    /// Keep memories under `scope` (for example a user id) instead of the
    /// session id, so they are shared by every session with that scope.
    pub fn with_scope(self, scope: impl Into<String>) -> Self {
        let scope = scope.into();
        self.update(|i| i.scope = Some(scope).filter(|s| !s.is_empty()))
    }

    /// Replace the default instructions.
    pub fn with_instructions(self, instructions: impl Into<String>) -> Self {
        let instructions = instructions.into();
        self.update(|i| i.instructions = instructions)
    }

    /// The provider's `source_id`.
    pub fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// The working folder for `session`: the configured scope or else the
    /// session id, as one folder name.
    ///
    /// Fails when there is neither, because falling back to the store root
    /// would expose every other scope's memories, and when the folder name
    /// would be longer than a file system allows.
    pub fn working_folder(&self, session: Option<&AgentSession>) -> Result<String> {
        let raw = self
            .inner
            .scope
            .as_deref()
            .or_else(|| session.map(AgentSession::session_id))
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                Error::Configuration(
                    "FileMemoryProvider requires a memory scope: set one with with_scope or run with a session. \
                     Without one, memories cannot be isolated from other scopes."
                        .into(),
                )
            })?;
        let folder = storage_key_segment(raw, ENCODED_SCOPE_PREFIX);
        if folder.len() > MAX_FOLDER_NAME_BYTES {
            return Err(Error::Configuration(format!(
                "The memory scope is too long to use as a folder name ({} bytes encoded, at most {MAX_FOLDER_NAME_BYTES}).",
                folder.len()
            )));
        }
        Ok(folder)
    }

    /// The seven tools, bound to `working_folder`.
    pub fn tools(&self, working_folder: &str) -> Vec<ToolDefinition> {
        vec![
            self.write_tool(working_folder),
            self.read_tool(working_folder),
            self.delete_tool(working_folder),
            self.ls_tool(working_folder),
            self.grep_tool(working_folder),
            self.replace_tool(working_folder),
            self.replace_lines_tool(working_folder),
        ]
    }

    fn write_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Flat file name to write under; must not contain path separators.
            file_name: String,
            /// Full text content to write to the file.
            content: String,
            /// Optional summary used to aid future discovery; recommended for large files.
            #[serde(default)]
            description: Option<String>,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_write",
            "Write a memory file with the given name and content. Overwrites the file if it already exists. \
             Include a description for large files to provide a summary that helps with future discovery.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let name = &args.file_name;
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(text(format!("Could not write file '{name}': {}", detail(&e)))),
                    };
                    if normalized.contains('/') {
                        return Ok(text(format!(
                            "Could not write file '{name}': memory files must not be written into a subdirectory. \
                             Please choose a flat file name without path separators."
                        )));
                    }
                    if is_internal_file(&normalized) {
                        return Ok(text(format!(
                            "Could not write file '{name}': the file name is reserved for internal use. \
                             Please choose a different file name."
                        )));
                    }
                    // The description file's name is longer than the file's,
                    // so check it fits before writing anything, rather than
                    // leaving the file written and its description failed.
                    let desc_name = description_file_name(&normalized);
                    if desc_name.len() > MAX_FILE_NAME_BYTES {
                        return Ok(text(format!(
                            "Could not write file '{name}': the file name is too long. Please keep it to {} bytes or fewer.",
                            MAX_FILE_NAME_BYTES - DESCRIPTION_SUFFIX.len()
                        )));
                    }
                    let described = args.description.as_deref().is_some_and(|d| !d.trim().is_empty());
                    let _guard = inner.write_lock.lock().await;
                    let outcome = async {
                        inner.store.write(&combine(&folder, &normalized), &args.content).await?;
                        let desc_path = combine(&folder, &desc_name);
                        match args.description.as_deref().filter(|_| described) {
                            Some(description) => inner.store.write(&desc_path, description).await?,
                            None => {
                                inner.store.delete(&desc_path).await?;
                            }
                        }
                        inner.rebuild_index(&folder).await
                    }
                    .await;
                    Ok(text(match outcome {
                        Err(e) => format!("Could not write file '{name}': {}", detail(&e)),
                        Ok(()) if described => format!("File '{name}' written with description."),
                        Ok(()) => format!("File '{name}' written."),
                    }))
                }
            },
        )
        .into_definition()
    }

    fn read_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name of the memory file to read.
            file_name: String,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_read",
            "Read the content of a memory file by name. Returns the file content or a message indicating the file \
             was not found. Line numbers count lines split on \\n only: a lone \\r never starts a new line, each \
             line keeps its own terminator, and content ending in a newline has a final empty line.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let name = &args.file_name;
                    let normalized = match normalize_relative_path(name, false) {
                        Ok(n) => n,
                        Err(e) => return Ok(text(format!("Could not read file '{name}': {}", detail(&e)))),
                    };
                    if normalized.contains('/') {
                        return Ok(text(format!("File '{name}' not found.")));
                    }
                    Ok(text(match inner.store.read(&combine(&folder, &normalized)).await {
                        Ok(Some(content)) => content,
                        Ok(None) => format!("File '{name}' not found."),
                        Err(e) => format!("Could not read file '{name}': {}", detail(&e)),
                    }))
                }
            },
        )
        .into_definition()
    }

    fn delete_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name of the memory file to delete.
            file_name: String,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_delete",
            "Delete a memory file by name. Also removes its companion description file if one exists.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let name = &args.file_name;
                    // The index and description files are not the agent's to
                    // delete, any more than to edit.
                    let path = match inner.editable_path(&folder, name, "delete") {
                        Ok(path) => path,
                        Err(message) => return Ok(text(message)),
                    };
                    let _guard = inner.write_lock.lock().await;
                    let outcome = async {
                        let deleted = inner.store.delete(&path).await?;
                        inner.store.delete(&description_file_name(&path)).await?;
                        inner.rebuild_index(&folder).await?;
                        Ok::<_, Error>(deleted)
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
        .into_definition()
    }

    fn ls_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Optional glob (e.g. "*.md") matched against file names to filter the listing.
            #[serde(default)]
            glob_pattern: Option<String>,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_ls",
            "List all memory files with their descriptions (if available). Optionally filter file names with a \
             glob_pattern (e.g. \"*.md\"). Internal files (description sidecars and the memory index) are not \
             shown. Each entry is {\"name\": <name>, \"type\": \"file\", \"description\": <desc-or-null>}.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let entries = match inner.store.list_children(&folder).await {
                        Ok(entries) => entries,
                        Err(e) => return Ok(text(format!("Could not list memory files: {}", detail(&e)))),
                    };
                    let names: Vec<String> =
                        entries.into_iter().filter(|e| e.is_file()).map(|e| e.name).collect();
                    let mut listed = Vec::new();
                    for name in &names {
                        if is_internal_file(name) || !matches_glob(name, args.glob_pattern.as_deref()) {
                            continue;
                        }
                        // Read through the store rather than matching listed
                        // names, so a sidecar whose letter case differs from
                        // the file's is still found where the store ignores case.
                        let description = inner
                            .store
                            .read(&combine(&folder, &description_file_name(name)))
                            .await
                            .ok()
                            .flatten();
                        listed.push(json!({"name": name, "type": "file", "description": description}));
                    }
                    Ok(Value::Array(listed))
                }
            },
        )
        .into_definition()
    }

    fn grep_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Case-insensitive regex matched against file contents; 256 characters or fewer.
            regex_pattern: String,
            /// Optional glob to filter which files are searched (e.g. "*.md", "research*").
            #[serde(default)]
            glob_pattern: Option<String>,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_grep",
            "Search memory file contents using a case-insensitive regular expression. Optionally filter which \
             files to search using a glob pattern (e.g., \"*.md\", \"research*\"). Returns matching file names, \
             content snippets, and matching lines with line numbers. The regex_pattern must be 256 characters \
             or fewer.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let glob = args.glob_pattern.as_deref().filter(|g| !g.trim().is_empty());
                    match inner.store.search(&folder, &args.regex_pattern, glob, false).await {
                        // The index and description files are the provider's
                        // own bookkeeping, so matches in them are dropped.
                        Ok(results) => Ok(serde_json::to_value(
                            results
                                .into_iter()
                                .filter(|r| !is_internal_file(&r.file_name))
                                .collect::<Vec<_>>(),
                        )?),
                        Err(e) => Ok(text(format!("Could not search memory files: {}", detail(&e)))),
                    }
                }
            },
        )
        .into_definition()
    }

    fn replace_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name of the memory file to modify.
            file_name: String,
            /// Substring to find and replace.
            old_string: String,
            /// Replacement text.
            new_string: String,
            /// When true, replace every occurrence; when false, fail unless exactly one occurrence exists.
            #[serde(default)]
            replace_all: bool,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_replace",
            "Replace occurrences of old_string with new_string in a memory file. Fails if old_string is not found, \
             or if it occurs more than once and replace_all is false. Returns the number of occurrences replaced.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let name = &args.file_name;
                    let path = match inner.editable_path(&folder, name, "replace in") {
                        Ok(path) => path,
                        Err(message) => return Ok(text(message)),
                    };
                    let _guard = inner.write_lock.lock().await;
                    let outcome = async {
                        let Some(content) = inner.store.read(&path).await? else {
                            return Ok(None);
                        };
                        let (updated, count) =
                            apply_replace(&content, &args.old_string, &args.new_string, args.replace_all)?;
                        inner.store.write(&path, &updated).await?;
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
        .into_definition()
    }

    fn replace_lines_tool(&self, folder: &str) -> ToolDefinition {
        #[derive(Deserialize, schemars::JsonSchema)]
        struct Args {
            /// Name of the memory file to modify.
            file_name: String,
            /// List of 1-based line numbers and their literal replacement text.
            edits: Vec<FileLineEdit>,
        }
        let (inner, folder) = (self.inner.clone(), folder.to_string());
        FunctionTool::typed(
            "file_memory_replace_lines",
            "Replace lines in a memory file. Provide a list of edits, each with a 1-based line_number and a literal \
             new_line (include your own trailing newline); an empty new_line deletes the line, including its line \
             break. Fails on out-of-range or duplicate line numbers. Line numbers count lines split on \\n only: a \
             lone \\r never starts a new line, each line keeps its own terminator, and content ending in a newline \
             has a final empty line.",
            move |args: Args| {
                let (inner, folder) = (inner.clone(), folder.clone());
                async move {
                    let name = &args.file_name;
                    let path = match inner.editable_path(&folder, name, "edit") {
                        Ok(path) => path,
                        Err(message) => return Ok(text(message)),
                    };
                    let _guard = inner.write_lock.lock().await;
                    let outcome = async {
                        let Some(content) = inner.store.read(&path).await? else {
                            return Ok(false);
                        };
                        let updated = apply_replace_lines(&content, &args.edits)?;
                        inner.store.write(&path, &updated).await?;
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
        .into_definition()
    }
}

impl Inner {
    /// The store path of a file the agent may change or delete, or the
    /// message to return instead.
    fn editable_path(
        &self,
        folder: &str,
        name: &str,
        verb: &str,
    ) -> std::result::Result<String, String> {
        let normalized = normalize_relative_path(name, false)
            .map_err(|e| format!("Could not {verb} file '{name}': {}", detail(&e)))?;
        if normalized.contains('/') {
            return Err(format!("File '{name}' not found."));
        }
        if is_internal_file(&normalized) {
            return Err(format!(
                "Could not {verb} file '{name}': the file name is reserved for internal use. \
                 Please choose a different file name."
            ));
        }
        Ok(combine(folder, &normalized))
    }

    /// Rewrite `memories.md`: the visible files in name order (ignoring
    /// case), at most 50, each with its description.
    async fn rebuild_index(&self, folder: &str) -> Result<()> {
        let mut names: Vec<String> = self
            .store
            .list_children(folder)
            .await?
            .into_iter()
            .filter(|e| e.is_file() && !is_internal_file(&e.name))
            .map(|e| e.name)
            .collect();
        names.sort_by_key(|n| n.to_lowercase());
        let mut lines = vec!["# Memory Index".to_string(), String::new()];
        for name in names.iter().take(MAX_INDEX_ENTRIES) {
            let description = self
                .store
                .read(&combine(folder, &description_file_name(name)))
                .await?;
            match description
                .as_deref()
                .map(str::trim)
                .filter(|d| !d.is_empty())
            {
                Some(d) => lines.push(format!("- **{name}**: {d}")),
                None => lines.push(format!("- **{name}**")),
            }
        }
        let index = lines.join("\n") + "\n";
        self.store
            .write(&combine(folder, MEMORY_INDEX_FILE_NAME), &index)
            .await
    }
}

#[async_trait::async_trait]
impl ContextProvider for FileMemoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let folder = self.working_folder(ctx.session.as_ref())?;
        self.inner.store.create_directory(&folder).await?;
        ctx.add_instructions(self.inner.instructions.clone());
        ctx.tools.extend(self.tools(&folder));
        // An unreadable index must not block the run; the next write or
        // delete rebuilds it.
        match self
            .inner
            .store
            .read(&combine(&folder, MEMORY_INDEX_FILE_NAME))
            .await
        {
            Ok(Some(index)) if !index.trim().is_empty() => {
                ctx.messages
                    .push(Message::user(format!("{INDEX_INTRODUCTION}{index}")));
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(error = %e, "could not read the memory index; skipping it for this run");
            }
        }
        Ok(())
    }
}

/// `notes.md` → `notes.md_description.md`.
///
/// The whole name is kept, as in Go's port, so `notes.md` and `notes.txt`
/// get separate descriptions. Python and .NET drop the extension
/// (`notes_description.md`), which makes those two share one.
fn description_file_name(name: &str) -> String {
    format!("{name}{DESCRIPTION_SUFFIX}")
}

/// Whether `name` is a description file or the index, hidden from the agent.
fn is_internal_file(name: &str) -> bool {
    let lowered = name.to_lowercase();
    lowered.ends_with(DESCRIPTION_SUFFIX) || lowered == MEMORY_INDEX_FILE_NAME
}

fn combine(folder: &str, name: &str) -> String {
    if folder.is_empty() {
        name.to_string()
    } else {
        format!(
            "{}/{}",
            folder.trim_end_matches('/'),
            name.trim_start_matches('/')
        )
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
    fn description_and_internal_names() {
        assert_eq!(description_file_name("notes.md"), "notes.md_description.md");
        assert_eq!(
            description_file_name("notes.txt"),
            "notes.txt_description.md"
        );
        assert_eq!(description_file_name("notes"), "notes_description.md");
        assert!(is_internal_file("Memories.MD"));
        assert!(is_internal_file("x_DESCRIPTION.md"));
        assert!(!is_internal_file("notes.md"));
    }
}
