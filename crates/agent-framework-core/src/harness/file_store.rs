//! The file store behind the harness file tools: an [`AgentFileStore`]
//! trait, an in-memory and a local-disk implementation, and the path, glob,
//! search and line-editing rules they share.
//!
//! Port of the store half of upstream `_harness/_file_access.py` (.NET
//! `Harness/FileStore`). [`FileMemoryProvider`](super::FileMemoryProvider)
//! is built on it; any other backend (blob storage, a database) can
//! implement [`AgentFileStore`] and be used in its place.
//!
//! # Paths
//!
//! Paths are relative to the store's root, use `/` as the separator and may
//! not escape the root: [`normalize_relative_path`] refuses rooted paths,
//! drive letters and `.`/`..` segments, and every built-in store runs each
//! path through it.
//!
//! # Lines
//!
//! [`split_lines`] is the one definition of a line for the whole surface:
//! every [`FileSearchMatch::line_number`] and every [`FileLineEdit`] is a
//! 1-based coordinate in it. Lines split on `\n` only and keep their
//! terminator, so a lone `\r` stays on its line, content ending in `\n` has a
//! final empty line, and joining the lines gives the content back exactly.
//! This follows upstream Python; .NET also splits on a lone `\r`. The rule
//! only has to be consistent within one SDK, because a line number never
//! crosses runtimes.
//!
//! # Divergences
//!
//! - **No regex deadline.** Upstream bounds every match with a timeout,
//!   because a model-supplied pattern such as `(a|a)*$` can backtrack
//!   catastrophically in Python's and .NET's engines. The `regex` crate
//!   matches in linear time, so no pattern can do that here. The 256-character
//!   cap on patterns is kept, and the whole search is still bounded by
//!   [`SEARCH_TIMEOUT`] to cover a slow store or a very large file; each
//!   file is scanned on a blocking thread so the timeout can always fire.
//! - **No `O_NOFOLLOW`.** [`FileSystemAgentFileStore`] refuses a symbolic
//!   link anywhere on a path before it opens it, as upstream does, but does
//!   not also ask the kernel to refuse a link that appears between that check
//!   and the open. Like upstream's store, it is meant for a root that no
//!   hostile process shares.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use async_trait::async_trait;
use regex::{Regex, RegexBuilder};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The longest search pattern a store accepts, in characters.
pub const MAX_SEARCH_PATTERN_LENGTH: usize = 256;

/// The wall-clock budget for one [`AgentFileStore::search`], covering every
/// file read and every line scanned.
pub const SEARCH_TIMEOUT: Duration = Duration::from_secs(10);

/// How many characters of context a search snippet keeps on each side of the
/// first match.
const SEARCH_SNIPPET_RADIUS: usize = 50;

/// Whether a [`FileStoreEntry`] is a file or a directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStoreEntryType {
    /// A regular file.
    File,
    /// A subdirectory.
    Directory,
}

/// One direct child of a listed directory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileStoreEntry {
    /// The entry's name: one path segment, not a full path.
    pub name: String,
    /// Whether it is a file or a directory.
    #[serde(rename = "type")]
    pub entry_type: FileStoreEntryType,
}

impl FileStoreEntry {
    /// A file entry.
    pub fn file(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entry_type: FileStoreEntryType::File,
        }
    }

    /// A directory entry.
    pub fn directory(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entry_type: FileStoreEntryType::Directory,
        }
    }

    /// Whether this entry is a file.
    pub fn is_file(&self) -> bool {
        self.entry_type == FileStoreEntryType::File
    }
}

/// One line of a file that matched a search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchMatch {
    /// The 1-based line number, a coordinate in [`split_lines`].
    pub line_number: usize,
    /// The line verbatim, terminator included, so it can be reused as a
    /// [`FileLineEdit::new_line`].
    pub line: String,
}

/// The matches found in one file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileSearchResult {
    /// The file's path relative to the searched directory.
    pub file_name: String,
    /// Up to 50 characters either side of the first match.
    pub snippet: String,
    /// Every matching line.
    pub matching_lines: Vec<FileSearchMatch>,
}

/// One whole-line replacement for [`apply_replace_lines`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct FileLineEdit {
    /// 1-based line number to replace.
    pub line_number: usize,
    /// Literal replacement text for the line, including any trailing newline
    /// you want to keep (the editor does not add one). Set to an empty string
    /// to delete the line entirely, including its line break.
    pub new_line: String,
    /// Optional: the text you believe is currently on that line, as reported
    /// by grep. When supplied, the edit is rejected unless it matches, which
    /// catches an out-of-date line number or a file that changed since you
    /// looked. The trailing newline is ignored in the comparison.
    #[serde(default)]
    pub expected_line: Option<String>,
}

impl FileLineEdit {
    /// An edit that replaces `line_number` with `new_line`.
    pub fn new(line_number: usize, new_line: impl Into<String>) -> Self {
        Self {
            line_number,
            new_line: new_line.into(),
            expected_line: None,
        }
    }

    /// Refuse the edit unless the line currently reads `expected`.
    pub fn expecting(mut self, expected: impl Into<String>) -> Self {
        self.expected_line = Some(expected.into());
        self
    }
}

/// File storage for the harness file tools.
///
/// All paths are relative to an implementation-defined root, use `/` as the
/// separator and must not escape the root; an implementation enforces that,
/// normally with [`normalize_relative_path`].
#[async_trait]
pub trait AgentFileStore: Send + Sync {
    /// Write `content` to `path`, replacing any existing file.
    async fn write(&self, path: &str, content: &str) -> Result<()>;

    /// Create `path` with `content`, failing if a file already exists there.
    /// The check and the write must be one atomic step.
    async fn create_new(&self, path: &str, content: &str) -> Result<()>;

    /// The content of `path`, or `None` if there is no such file.
    async fn read(&self, path: &str) -> Result<Option<String>>;

    /// Delete `path`, returning whether a file was removed.
    async fn delete(&self, path: &str) -> Result<bool>;

    /// The direct children of `directory` (`""` for the root), names only.
    /// Implementations should list subdirectories before files.
    async fn list_children(&self, directory: &str) -> Result<Vec<FileStoreEntry>>;

    /// Whether a file exists at `path`.
    async fn file_exists(&self, path: &str) -> Result<bool>;

    /// Make sure `path` exists as a directory.
    async fn create_directory(&self, path: &str) -> Result<()>;

    /// The files that *may* satisfy a search, as paths relative to
    /// `directory`.
    ///
    /// This is the hook a store with an index uses to narrow a search.
    /// Returning extra files is harmless, because [`search`](Self::search)
    /// re-applies the glob and scans every candidate, but leaving one out
    /// loses its matches. `regex_pattern` and `glob_pattern` both match
    /// case-insensitively, so an index that cannot must widen rather than
    /// narrow. The default walks [`list_children`](Self::list_children) and
    /// returns every file in scope.
    async fn find_matching_files(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        let _ = regex_pattern;
        let glob = compile_glob(glob_pattern)?;
        let mut names = Vec::new();
        let mut pending = vec![String::new()];
        while let Some(relative_dir) = pending.pop() {
            for entry in self
                .list_children(&combine_search_path(directory, &relative_dir))
                .await?
            {
                let child = combine_search_path(&relative_dir, &entry.name);
                if !entry.is_file() {
                    if recursive {
                        pending.push(child);
                    }
                } else if glob.matches(&child) {
                    names.push(child);
                }
            }
        }
        Ok(names)
    }

    /// Search the files in `directory` for lines matching `regex_pattern`
    /// (case-insensitive), optionally only those whose path relative to
    /// `directory` matches `glob_pattern`, and with `recursive` all
    /// descendants rather than direct children.
    ///
    /// The default asks [`find_matching_files`](Self::find_matching_files)
    /// for candidates, reads each and numbers matches with [`scan_content`],
    /// so a line number from it always addresses the same line in
    /// [`apply_replace_lines`]. A file that cannot be read is skipped. An
    /// override owns that numbering: it must report coordinates in
    /// [`split_lines`] of the content [`read`](Self::read) returns.
    async fn search(
        &self,
        directory: &str,
        regex_pattern: &str,
        glob_pattern: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<FileSearchResult>> {
        let regex = compile_search_regex(regex_pattern)?;
        let scan = async {
            let glob = compile_glob(glob_pattern)?;
            let names = self
                .find_matching_files(directory, regex_pattern, glob_pattern, recursive)
                .await?;
            let mut results = Vec::new();
            for name in names {
                // Re-applied because find_matching_files may over-return.
                if !glob.matches(&name) || (!recursive && name.contains('/')) {
                    continue;
                }
                match self.read(&combine_search_path(directory, &name)).await {
                    Ok(Some(content)) => {
                        // Off the executor, so a long scan cannot hold a
                        // worker and keep the timeout below from firing.
                        let regex = regex.clone();
                        let scanned =
                            blocking(move || Ok(scan_content(&name, &content, &regex))).await?;
                        if let Some(result) = scanned {
                            results.push(result);
                        }
                    }
                    Ok(None) => {} // Deleted since it was listed.
                    Err(e) => {
                        tracing::warn!(file = %name, error = %e, "skipping unreadable file during search");
                    }
                }
            }
            Ok(results)
        };
        match tokio::time::timeout(SEARCH_TIMEOUT, scan).await {
            Ok(results) => results,
            Err(_) => Err(Error::tool(format!(
                "Search did not complete within {} seconds. Narrow the pattern, or search a smaller directory.",
                SEARCH_TIMEOUT.as_secs()
            ))),
        }
    }
}

/// Normalize and validate a relative store path.
///
/// Trims surrounding whitespace, turns `\` into `/` and collapses repeated
/// separators. Refuses rooted paths, drive letters and `.`/`..` segments.
/// With `is_directory`, an empty path is the root; otherwise an empty path,
/// or one ending in a separator, is refused.
///
/// This is for paths given as tool arguments and is deliberately lossy
/// (`a//b` and `a/b` are one path). Never use it to turn an identifier that
/// draws an isolation boundary, such as a session id, into a folder: use
/// [`storage_key_segment`](crate::storage_keys::storage_key_segment).
pub fn normalize_relative_path(path: &str, is_directory: bool) -> Result<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        if is_directory {
            return Ok(String::new());
        }
        return Err(Error::tool(
            "A file path must not be empty or whitespace-only.",
        ));
    }
    let converted = trimmed.replace('\\', "/");
    if !is_directory && converted.ends_with('/') {
        return Err(Error::tool(format!(
            "Invalid path: '{trimmed}'. A file path must not end with a path separator."
        )));
    }
    let stripped = converted.trim_matches('/');
    let mut chars = stripped.chars();
    let drive = matches!((chars.next(), chars.next()), (Some(c), Some(':')) if c.is_alphabetic());
    if converted.starts_with('/') || drive {
        return Err(Error::tool(format!(
            "Invalid path: '{trimmed}'. Paths must be relative and must not start with '/', '\\', or a drive root."
        )));
    }
    // Elsewhere in a path, a colon names an NTFS alternate data stream
    // (`notes.md:hidden`), which a listing never shows.
    if stripped.contains(':') {
        return Err(Error::tool(format!(
            "Invalid path: '{trimmed}'. Paths must not contain ':'."
        )));
    }
    let mut segments = Vec::new();
    for segment in stripped.split('/') {
        match segment {
            "" => continue,
            "." | ".." => {
                return Err(Error::tool(format!(
                    "Invalid path: '{trimmed}'. Paths must not contain '.' or '..' segments."
                )))
            }
            _ => segments.push(segment),
        }
    }
    let result = segments.join("/");
    if !is_directory && result.is_empty() {
        return Err(Error::tool(format!(
            "Invalid path: '{trimmed}'. A file path must not be empty."
        )));
    }
    Ok(result)
}

/// Join a directory and a path relative to it; either may be empty.
pub(crate) fn combine_search_path(directory: &str, relative: &str) -> String {
    let base = directory.trim_matches('/');
    let tail = relative.trim_matches('/');
    match (base.is_empty(), tail.is_empty()) {
        (true, _) => tail.to_string(),
        (false, true) => base.to_string(),
        (false, false) => format!("{base}/{tail}"),
    }
}

/// A compiled glob, or none (which matches everything).
pub(crate) struct Glob(Option<Regex>);

impl Glob {
    pub(crate) fn matches(&self, name: &str) -> bool {
        self.0.as_ref().is_none_or(|re| re.is_match(name))
    }
}

/// Compile an optional glob. A blank pattern matches everything.
pub(crate) fn compile_glob(pattern: Option<&str>) -> Result<Glob> {
    Ok(Glob(
        pattern
            .filter(|p| !p.trim().is_empty())
            .map(glob_regex)
            .transpose()?,
    ))
}

/// Whether `file_name` matches `glob_pattern`, case-insensitively.
///
/// `file_name` is a `/`-separated path relative to the searched directory.
/// The syntax is Python's `fnmatch`: `*` matches any run of characters,
/// including `/`, so `*.md` matches markdown files at any depth; `?` matches
/// one character; `[abc]`, `[a-z]` and `[!abc]` match a set. `None` or a blank
/// pattern matches everything. A pattern over
/// [`MAX_SEARCH_PATTERN_LENGTH`] characters is an error, as for a search
/// regex.
pub fn matches_glob(file_name: &str, glob_pattern: Option<&str>) -> Result<bool> {
    Ok(compile_glob(glob_pattern)?.matches(file_name))
}

/// Translate an `fnmatch` pattern into an anchored, case-insensitive regex.
fn glob_regex(pattern: &str) -> Result<Regex> {
    let chars: Vec<char> = pattern.chars().collect();
    // Unbounded, a model-supplied glob could pass the regex size limit.
    if chars.len() > MAX_SEARCH_PATTERN_LENGTH {
        return Err(Error::tool(format!(
            "glob_pattern must be {MAX_SEARCH_PATTERN_LENGTH} characters or fewer."
        )));
    }
    let mut out = String::from("(?si)^");
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            '*' => {
                out.push_str(".*");
                while chars.get(i + 1) == Some(&'*') {
                    i += 1;
                }
            }
            '?' => out.push('.'),
            '[' => match glob_class(&chars, i) {
                Some((class, end)) => {
                    out.push_str(&class);
                    i = end;
                }
                None => out.push_str("\\["),
            },
            c => out.push_str(&regex::escape(&c.to_string())),
        }
        i += 1;
    }
    out.push('$');
    // Every piece is escaped or a well-formed class; a reversed range such
    // as `[z-a]` is the only way to fail, and then the pattern is matched
    // literally.
    Regex::new(&out)
        .or_else(|_| Regex::new(&format!("(?si)^{}$", regex::escape(pattern))))
        .map_err(|e| Error::tool(format!("Invalid glob_pattern: {e}")))
}

/// The regex class for the `[` at `start`, and the index of its `]`, or
/// `None` when the bracket is unclosed (and so literal).
fn glob_class(chars: &[char], start: usize) -> Option<(String, usize)> {
    let mut j = start + 1;
    let negated = chars.get(j) == Some(&'!');
    if negated {
        j += 1;
    }
    // A `]` straight after the opening bracket is a member, not the end.
    if chars.get(j) == Some(&']') {
        j += 1;
    }
    while j < chars.len() && chars[j] != ']' {
        j += 1;
    }
    if j >= chars.len() {
        return None;
    }
    let body_start = start + 1 + usize::from(negated);
    let mut class = String::from(if negated { "[^" } else { "[" });
    for (k, &c) in chars[body_start..j].iter().enumerate() {
        match c {
            '\\' | '[' | ']' | '&' | '~' => {
                class.push('\\');
                class.push(c);
            }
            '^' if k == 0 => class.push_str("\\^"),
            _ => class.push(c),
        }
    }
    class.push(']');
    Some((class, j))
}

/// Compile a model-supplied search pattern: case-insensitive, at most
/// [`MAX_SEARCH_PATTERN_LENGTH`] characters.
pub fn compile_search_regex(pattern: &str) -> Result<Regex> {
    let length = pattern.chars().count();
    if length > MAX_SEARCH_PATTERN_LENGTH {
        return Err(Error::tool(format!(
            "Regex pattern is too long ({length} characters). Maximum supported length is {MAX_SEARCH_PATTERN_LENGTH} characters."
        )));
    }
    RegexBuilder::new(pattern)
        .case_insensitive(true)
        .build()
        .map_err(|e| Error::tool(format!("Invalid regex pattern: {e}")))
}

/// Split `content` into the lines line numbers address: on `\n` only, each
/// line keeping its terminator. See the [module docs](self#lines).
pub fn split_lines(content: &str) -> Vec<&str> {
    content
        .split_inclusive('\n')
        .chain(
            // split_inclusive drops the empty piece after a final `\n`, which is
            // a line here (and empty content is one empty line).
            (content.is_empty() || content.ends_with('\n')).then_some(""),
        )
        .collect()
}

/// `line` without a trailing `\r\n` or `\n`. A lone `\r` is content.
fn strip_line_terminator(line: &str) -> &str {
    line.strip_suffix("\r\n")
        .or_else(|| line.strip_suffix('\n'))
        .unwrap_or(line)
}

/// Every line of `content` matching `regex`, numbered by [`split_lines`],
/// or `None` when nothing matches.
///
/// Lines are reported verbatim; the pattern is matched against each line
/// without its terminator, so `$` anchors to the end of the text on CRLF
/// content too. This is the numbering primitive the default
/// [`AgentFileStore::search`] uses, for a store that implements its own.
pub fn scan_content(file_name: &str, content: &str, regex: &Regex) -> Option<FileSearchResult> {
    let mut matching_lines = Vec::new();
    let mut snippet = None;
    let mut offset = 0;
    for (index, line) in split_lines(content).into_iter().enumerate() {
        if let Some(found) = regex.find(strip_line_terminator(line)) {
            matching_lines.push(FileSearchMatch {
                line_number: index + 1,
                line: line.to_string(),
            });
            if snippet.is_none() {
                let start = back_chars(content, offset + found.start(), SEARCH_SNIPPET_RADIUS);
                let end = forward_chars(content, offset + found.end(), SEARCH_SNIPPET_RADIUS);
                snippet = Some(content[start..end].to_string());
            }
        }
        offset += line.len();
    }
    (!matching_lines.is_empty()).then(|| FileSearchResult {
        file_name: file_name.to_string(),
        snippet: snippet.unwrap_or_default(),
        matching_lines,
    })
}

/// The byte index `count` (at least 1) characters before `index`, or 0.
fn back_chars(s: &str, index: usize, count: usize) -> usize {
    s[..index]
        .char_indices()
        .rev()
        .nth(count - 1)
        .map_or(0, |(i, _)| i)
}

/// The byte index `count` characters after `index` (or the end).
fn forward_chars(s: &str, index: usize, count: usize) -> usize {
    s[index..]
        .char_indices()
        .nth(count)
        .map_or(s.len(), |(i, _)| index + i)
}

/// Replace `old` with `new` in `content`, returning the new content and the
/// number of replacements. Fails when `old` is empty or absent, or occurs
/// more than once without `replace_all`.
pub fn apply_replace(
    content: &str,
    old: &str,
    new: &str,
    replace_all: bool,
) -> Result<(String, usize)> {
    if old.is_empty() {
        return Err(Error::tool("old_string must not be empty."));
    }
    let count = content.matches(old).count();
    if count == 0 {
        return Err(Error::tool(format!("old_string not found: {old:?}.")));
    }
    if count > 1 && !replace_all {
        return Err(Error::tool(format!(
            "old_string occurs {count} times; pass replace_all=true to replace all, or provide a more specific old_string."
        )));
    }
    Ok((content.replace(old, new), count))
}

/// Apply whole-line replacements to `content`.
///
/// Each `new_line` is written verbatim in place of its line, so it carries
/// its own terminator; an empty one deletes the line. An edit with an
/// `expected_line` is refused unless the line currently matches it (both
/// without their terminator). Fails with no edits, a line number out of
/// range or used twice, or a mismatch, and then changes nothing.
pub fn apply_replace_lines(content: &str, edits: &[FileLineEdit]) -> Result<String> {
    if edits.is_empty() {
        return Err(Error::tool("At least one line edit must be provided."));
    }
    let mut lines: Vec<&str> = split_lines(content);
    let mut seen = std::collections::HashSet::new();
    for edit in edits {
        if !seen.insert(edit.line_number) {
            return Err(Error::tool(format!(
                "Duplicate line_number {} in edits.",
                edit.line_number
            )));
        }
        if edit.line_number < 1 || edit.line_number > lines.len() {
            return Err(Error::tool(format!(
                "line_number {} is out of range (file has {} lines).",
                edit.line_number,
                lines.len()
            )));
        }
        if let Some(expected) = &edit.expected_line {
            if strip_line_terminator(lines[edit.line_number - 1]) != strip_line_terminator(expected)
            {
                return Err(Error::tool(format!(
                    "line_number {} does not match the expected text. Re-read the file to get current line numbers.",
                    edit.line_number
                )));
            }
        }
    }
    for edit in edits {
        lines[edit.line_number - 1] = &edit.new_line;
    }
    Ok(lines.concat())
}

/// An [`AgentFileStore`] held in memory.
///
/// Paths are compared case-insensitively, and listings and search results
/// keep the casing a file was written with. Directories exist implicitly,
/// as prefixes of file paths, and a path is never both a file and a
/// directory: writes and `create_directory` are refused where a file
/// system would refuse them.
#[derive(Default)]
pub struct InMemoryAgentFileStore {
    state: Mutex<InMemoryState>,
}

#[derive(Default)]
struct InMemoryState {
    /// Lowercased path to (path as written, content).
    files: HashMap<String, (String, String)>,
    /// Directories made with `create_directory`, lowercased to as written,
    /// so an empty one still exists. Others are implied by file paths.
    directories: HashMap<String, String>,
}

impl fmt::Debug for InMemoryAgentFileStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InMemoryAgentFileStore")
            .field("files", &self.state.lock().unwrap().files.len())
            .finish()
    }
}

impl InMemoryAgentFileStore {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

/// Refuse a path whose ancestor is a file, as a file system would.
fn check_ancestors(files: &HashMap<String, (String, String)>, key: &str) -> Result<()> {
    for (i, _) in key.match_indices('/') {
        if let Some((display, _)) = files.get(&key[..i]) {
            return Err(Error::tool(format!(
                "Parent path is not a directory: {display}"
            )));
        }
    }
    Ok(())
}

/// Refuse writing a file where a directory is, as a file system would.
fn check_file_path(state: &InMemoryState, key: &str, path: &str) -> Result<()> {
    check_ancestors(&state.files, key)?;
    let prefix = format!("{key}/");
    let is_directory = state.directories.contains_key(key)
        || state
            .files
            .keys()
            .chain(state.directories.keys())
            .any(|k| k.starts_with(&prefix));
    if is_directory {
        return Err(Error::tool(format!("Path is a directory: {path}")));
    }
    Ok(())
}

#[async_trait]
impl AgentFileStore for InMemoryAgentFileStore {
    async fn write(&self, path: &str, content: &str) -> Result<()> {
        let display = normalize_relative_path(path, false)?;
        let mut state = self.state.lock().unwrap();
        let key = display.to_lowercase();
        check_file_path(&state, &key, path)?;
        state.files.insert(key, (display, content.to_string()));
        Ok(())
    }

    async fn create_new(&self, path: &str, content: &str) -> Result<()> {
        let display = normalize_relative_path(path, false)?;
        let mut state = self.state.lock().unwrap();
        let key = display.to_lowercase();
        if state.files.contains_key(&key) {
            return Err(Error::tool(format!("File already exists: '{path}'")));
        }
        check_file_path(&state, &key, path)?;
        state.files.insert(key, (display, content.to_string()));
        Ok(())
    }

    async fn read(&self, path: &str) -> Result<Option<String>> {
        let key = normalize_relative_path(path, false)?.to_lowercase();
        Ok(self
            .state
            .lock()
            .unwrap()
            .files
            .get(&key)
            .map(|(_, c)| c.clone()))
    }

    async fn delete(&self, path: &str) -> Result<bool> {
        let key = normalize_relative_path(path, false)?.to_lowercase();
        Ok(self.state.lock().unwrap().files.remove(&key).is_some())
    }

    async fn list_children(&self, directory: &str) -> Result<Vec<FileStoreEntry>> {
        let mut prefix = normalize_relative_path(directory, true)?.to_lowercase();
        if !prefix.is_empty() {
            prefix.push('/');
        }
        let depth = prefix.matches('/').count();
        let mut directories = Vec::new();
        let mut files = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let state = self.state.lock().unwrap();
        let file_paths = state.files.iter().map(|(key, (display, _))| (key, display));
        // A created directory is listed by its parent like a directory a
        // file implies, so give it a trailing segment to cut at.
        let created: Vec<(String, String)> = state
            .directories
            .iter()
            .map(|(key, display)| (format!("{key}/"), format!("{display}/")))
            .collect();
        let dir_paths = created.iter().map(|(key, display)| (key, display));
        for (key, display) in file_paths.chain(dir_paths) {
            if !key.starts_with(&prefix) {
                continue;
            }
            // Lowercasing can change byte lengths, so the display path is cut
            // by segments rather than by the key's offsets.
            let remainder = display.splitn(depth + 1, '/').last().unwrap_or_default();
            match remainder.split_once('/') {
                // The directory being listed, when it was created itself.
                None if remainder.is_empty() => {}
                None => files.push(FileStoreEntry::file(remainder)),
                Some((segment, _)) if !segment.is_empty() => {
                    if seen.insert(segment.to_lowercase()) {
                        directories.push(FileStoreEntry::directory(segment));
                    }
                }
                Some(_) => {}
            }
        }
        directories.sort_by(|a, b| a.name.cmp(&b.name));
        files.sort_by(|a, b| a.name.cmp(&b.name));
        directories.extend(files);
        Ok(directories)
    }

    async fn file_exists(&self, path: &str) -> Result<bool> {
        let key = normalize_relative_path(path, false)?.to_lowercase();
        Ok(self.state.lock().unwrap().files.contains_key(&key))
    }

    async fn create_directory(&self, path: &str) -> Result<()> {
        let display = normalize_relative_path(path, true)?;
        let key = display.to_lowercase();
        let mut state = self.state.lock().unwrap();
        if key.is_empty() {
            return Ok(());
        }
        check_ancestors(&state.files, &key)?;
        if state.files.contains_key(&key) {
            return Err(Error::tool(format!("Path is a file: {path}")));
        }
        state.directories.entry(key).or_insert(display);
        Ok(())
    }
}

/// An [`AgentFileStore`] on local disk, under a root directory.
///
/// The root is created on the first write, so constructing a store touches
/// nothing. A symbolic link anywhere between the root and a path is refused
/// on every operation, and listings and searches skip links, so the store
/// never reads or writes outside its root through one. Files are UTF-8 text;
/// reading anything else is an error, and a search skips such files.
#[derive(Debug, Clone)]
pub struct FileSystemAgentFileStore {
    root: PathBuf,
}

impl FileSystemAgentFileStore {
    /// A store rooted at `root`, which need not exist yet.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let raw = root.as_ref();
        if raw.as_os_str().is_empty() || raw.to_string_lossy().trim().is_empty() {
            return Err(Error::Configuration(
                "root_directory must not be empty or whitespace-only.".into(),
            ));
        }
        let absolute = std::path::absolute(raw)
            .map_err(|e| Error::Configuration(format!("Invalid root directory: {e}")))?;
        // Resolved once, here, so a root given through a link means its
        // target; any link that appears at the root later is refused.
        let root = std::fs::canonicalize(&absolute).unwrap_or(absolute);
        Ok(Self { root })
    }

    /// The root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn check_root(&self) -> Result<()> {
        match std::fs::symlink_metadata(&self.root) {
            Ok(meta) if meta.file_type().is_symlink() => Err(symlink_error()),
            _ => Ok(()),
        }
    }

    /// Refuse a link on any existing segment below the root.
    fn check_no_symlinks(&self, relative: &str) -> Result<()> {
        let mut current = self.root.clone();
        for segment in relative.split('/').filter(|s| !s.is_empty()) {
            current.push(segment);
            match std::fs::symlink_metadata(&current) {
                Ok(meta) if meta.file_type().is_symlink() => return Err(symlink_error()),
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(e) if e.kind() == std::io::ErrorKind::NotADirectory => {
                    let parent = current.parent().unwrap_or(&self.root);
                    return Err(Error::tool(format!(
                        "Parent path is not a directory: {}",
                        parent.display()
                    )));
                }
                // Fail closed: a segment that cannot be inspected is refused.
                Err(e) => {
                    return Err(Error::tool(format!(
                        "Invalid path: unable to verify whether '{relative}' contains a symbolic link: {e}"
                    )))
                }
            }
        }
        Ok(())
    }

    fn resolve_file(&self, path: &str) -> Result<PathBuf> {
        let normalized = normalize_relative_path(path, false)?;
        self.check_root()?;
        self.check_no_symlinks(&normalized)?;
        Ok(self.root.join(normalized))
    }

    fn resolve_directory(&self, path: &str) -> Result<PathBuf> {
        let normalized = normalize_relative_path(path, true)?;
        self.check_root()?;
        self.check_no_symlinks(&normalized)?;
        Ok(if normalized.is_empty() {
            self.root.clone()
        } else {
            self.root.join(normalized)
        })
    }

    fn write_file(full: &Path, content: &str, overwrite: bool) -> Result<()> {
        if let Some(parent) = full.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                let blocked = parent
                    .ancestors()
                    .find(|a| a.exists() && !a.is_dir())
                    .map(|a| format!("Parent path is not a directory: {}", a.display()));
                Error::tool(blocked.unwrap_or_else(|| io_message(&e)))
            })?;
        }
        if full.is_dir() {
            return Err(Error::tool(format!(
                "Path is a directory: {}",
                full.display()
            )));
        }
        let mut options = std::fs::OpenOptions::new();
        options.write(true);
        if overwrite {
            options.create(true).truncate(true);
        } else {
            options.create_new(true);
        }
        let mut file = options.open(full).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                Error::tool(format!("File already exists: '{}'", full.display()))
            } else {
                Error::tool(io_message(&e))
            }
        })?;
        std::io::Write::write_all(&mut file, content.as_bytes())
            .map_err(|e| Error::tool(io_message(&e)))
    }

    fn read_file(full: &Path) -> Result<Option<String>> {
        if !full.is_file() {
            return Ok(None);
        }
        let bytes = match std::fs::read(full) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(Error::tool(io_message(&e))),
        };
        String::from_utf8(bytes).map(Some).map_err(|_| {
            let name = full
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            Error::tool(format!(
                "File '{name}' is not UTF-8 text and cannot be read."
            ))
        })
    }

    fn list_dir(full: &Path) -> Result<Vec<FileStoreEntry>> {
        if !full.is_dir() {
            return Ok(Vec::new());
        }
        let mut directories = Vec::new();
        let mut files = Vec::new();
        for entry in std::fs::read_dir(full).map_err(|e| Error::tool(io_message(&e)))? {
            // Fail closed: entries that cannot be inspected are left out.
            let Ok(entry) = entry else { continue };
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if file_type.is_symlink() {
                continue;
            }
            if file_type.is_dir() {
                directories.push(FileStoreEntry::directory(name));
            } else if file_type.is_file() {
                files.push(FileStoreEntry::file(name));
            }
        }
        directories.sort_by(|a, b| a.name.cmp(&b.name));
        files.sort_by(|a, b| a.name.cmp(&b.name));
        directories.extend(files);
        Ok(directories)
    }
}

fn symlink_error() -> Error {
    Error::tool("Invalid path: the resolved path contains a symbolic link or reparse point.")
}

fn io_message(e: &std::io::Error) -> String {
    e.to_string()
}

/// Run blocking filesystem work off the async runtime.
async fn blocking<T, F>(f: F) -> Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| Error::Other(format!("file store task failed: {e}")))?
}

#[async_trait]
impl AgentFileStore for FileSystemAgentFileStore {
    async fn write(&self, path: &str, content: &str) -> Result<()> {
        let full = self.resolve_file(path)?;
        let content = content.to_string();
        blocking(move || Self::write_file(&full, &content, true)).await
    }

    async fn create_new(&self, path: &str, content: &str) -> Result<()> {
        let full = self.resolve_file(path)?;
        let content = content.to_string();
        blocking(move || Self::write_file(&full, &content, false)).await
    }

    async fn read(&self, path: &str) -> Result<Option<String>> {
        let full = self.resolve_file(path)?;
        blocking(move || Self::read_file(&full)).await
    }

    async fn delete(&self, path: &str) -> Result<bool> {
        let full = self.resolve_file(path)?;
        blocking(move || {
            if !full.is_file() {
                return Ok(false);
            }
            match std::fs::remove_file(&full) {
                Ok(()) => Ok(true),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
                Err(e) => Err(Error::tool(io_message(&e))),
            }
        })
        .await
    }

    async fn list_children(&self, directory: &str) -> Result<Vec<FileStoreEntry>> {
        let full = self.resolve_directory(directory)?;
        blocking(move || Self::list_dir(&full)).await
    }

    async fn file_exists(&self, path: &str) -> Result<bool> {
        let full = self.resolve_file(path)?;
        blocking(move || Ok(full.is_file())).await
    }

    async fn create_directory(&self, path: &str) -> Result<()> {
        let full = self.resolve_directory(path)?;
        blocking(move || std::fs::create_dir_all(&full).map_err(|e| Error::tool(io_message(&e))))
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_normalize_or_are_refused() {
        assert_eq!(
            normalize_relative_path(" a\\b//c.md ", false).unwrap(),
            "a/b/c.md"
        );
        assert_eq!(normalize_relative_path("", true).unwrap(), "");
        assert_eq!(normalize_relative_path("dir/", true).unwrap(), "dir");
        for bad in [
            "", "  ", "/abs", "\\abs", "C:/x", "c:x", "a/../b", "./a", "a/", "a\\",
        ] {
            assert!(normalize_relative_path(bad, false).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn globs_follow_fnmatch() {
        assert!(matches_glob("notes.MD", Some("*.md")).unwrap());
        assert!(matches_glob("deep/notes.md", Some("*.md")).unwrap());
        assert!(matches_glob("research1", Some("research?")).unwrap());
        assert!(matches_glob("b.txt", Some("[abc].txt")).unwrap());
        assert!(!matches_glob("d.txt", Some("[abc].txt")).unwrap());
        assert!(matches_glob("d.txt", Some("[!abc].txt")).unwrap());
        assert!(matches_glob("[x", Some("[x")).unwrap());
        assert!(matches_glob("a.b", Some("a.b")).unwrap());
        assert!(!matches_glob("axb", Some("a.b")).unwrap());
        assert!(matches_glob("anything", None).unwrap());
        assert!(matches_glob("anything", Some("  ")).unwrap());
    }

    #[test]
    fn lines_split_on_newline_and_keep_terminators() {
        assert_eq!(split_lines(""), [""]);
        assert_eq!(split_lines("a"), ["a"]);
        assert_eq!(split_lines("a\n"), ["a\n", ""]);
        assert_eq!(split_lines("a\r\nb\rc\n"), ["a\r\n", "b\rc\n", ""]);
        assert_eq!(split_lines("a\r\nb\rc\n").concat(), "a\r\nb\rc\n");
    }

    #[test]
    fn scan_reports_verbatim_lines_and_a_snippet() {
        let regex = compile_search_regex("beta$").unwrap();
        let result = scan_content("f.md", "alpha\r\nBETA\r\ngamma", &regex).unwrap();
        assert_eq!(
            result.matching_lines,
            [FileSearchMatch {
                line_number: 2,
                line: "BETA\r\n".into()
            }]
        );
        assert_eq!(result.snippet, "alpha\r\nBETA\r\ngamma");
        assert!(scan_content("f.md", "nothing", &regex).is_none());

        let long = format!("{}needle{}", "é".repeat(60), "ü".repeat(60));
        let snippet = scan_content("f", &long, &compile_search_regex("needle").unwrap())
            .unwrap()
            .snippet;
        assert_eq!(
            snippet,
            format!("{}needle{}", "é".repeat(50), "ü".repeat(50))
        );
    }

    #[test]
    fn long_or_invalid_patterns_are_refused() {
        assert!(compile_search_regex(&"a".repeat(257)).is_err());
        assert!(compile_search_regex("(").is_err());
        assert!(compile_search_regex(&"a".repeat(256)).is_ok());
    }

    #[test]
    fn replace_requires_a_unique_match_unless_told_otherwise() {
        assert_eq!(
            apply_replace("a b a", "b", "c", false).unwrap(),
            ("a c a".into(), 1)
        );
        assert!(apply_replace("a b a", "a", "c", false).is_err());
        assert_eq!(
            apply_replace("a b a", "a", "c", true).unwrap(),
            ("c b c".into(), 2)
        );
        assert!(apply_replace("abc", "x", "y", false).is_err());
        assert!(apply_replace("abc", "", "y", false).is_err());
    }

    #[test]
    fn line_edits_are_literal_and_checked() {
        let content = "one\ntwo\r\nthree";
        assert_eq!(
            apply_replace_lines(content, &[FileLineEdit::new(2, "TWO\r\n")]).unwrap(),
            "one\nTWO\r\nthree"
        );
        assert_eq!(
            apply_replace_lines(content, &[FileLineEdit::new(1, "")]).unwrap(),
            "two\r\nthree"
        );
        assert_eq!(
            apply_replace_lines(content, &[FileLineEdit::new(2, "x\n").expecting("two")]).unwrap(),
            "one\nx\nthree"
        );
        assert!(
            apply_replace_lines(content, &[FileLineEdit::new(2, "x").expecting("one")]).is_err()
        );
        assert!(apply_replace_lines(content, &[FileLineEdit::new(4, "x")]).is_err());
        assert!(apply_replace_lines(content, &[FileLineEdit::new(0, "x")]).is_err());
        assert!(apply_replace_lines(
            content,
            &[FileLineEdit::new(1, "x"), FileLineEdit::new(1, "y")]
        )
        .is_err());
        assert!(apply_replace_lines(content, &[]).is_err());
        // The final empty line after a trailing newline is editable.
        assert_eq!(
            apply_replace_lines("a\n", &[FileLineEdit::new(2, "b\n")]).unwrap(),
            "a\nb\n"
        );
    }
}
