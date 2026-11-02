//! Durable memory: a [`ContextProvider`] that keeps topic memory files,
//! a `MEMORY.md` index and raw transcripts, and extracts and consolidates
//! memories with a chat model.
//!
//! Port of upstream `_harness/_memory.py` (`MemoryContextProvider`,
//! `MemoryStore`, `MemoryFileStore`, `MemoryTopicRecord`,
//! `MemoryIndexEntry`). Every run:
//!
//! - **before** the model call, the provider rebuilds `MEMORY.md` (one
//!   pointer line per topic), auto-loads up to `selection_limit` topic files
//!   whose topic or summary shares words with the input, optionally replays
//!   the last `recent_turns` turns of this session's transcript, and adds six
//!   tools (`list_memory_topics`, `read_memory_topic`, `write_memory`,
//!   `delete_memory_topic`, `search_memory_transcripts`,
//!   `consolidate_memories`) with their instructions;
//! - **after** a successful run, it appends the run to the session's
//!   transcript, records the session for consolidation, asks the chat client
//!   to extract up to `max_extractions` durable memories into topic files, and
//!   consolidates every topic once `consolidation_min_sessions` sessions have
//!   passed and `consolidation_interval` has elapsed.
//!
//! [`MemoryFileStore`] keeps everything under
//! `{base}/{source}/{owner}/{kind}/`: `MEMORY.md`, `topics/{slug}.md`,
//! `transcripts/{session}.jsonl` and `state.json`. The owner comes from a
//! session-state key, so memory is shared by every session of one owner.
//!
//! # Divergences
//!
//! - **The chat client is passed in.** Upstream uses the running agent's
//!   client; a Rust context provider does not see its agent, so extraction
//!   needs [`MemoryContextProviderBuilder::chat_client`]. Without one, no
//!   memories are extracted and consolidation only de-duplicates.
//! - **The store owns transcripts.** Upstream's store exposes a transcript
//!   directory that a `FileHistoryProvider` writes; [`MemoryStore`] has
//!   [`append_transcript`](MemoryStore::append_transcript) and
//!   [`load_transcript`](MemoryStore::load_transcript) instead, so a store
//!   need not be file based. The JSONL format matches. Upstream's
//!   `export_provider_state` / `import_provider_state` exist to rebuild a
//!   session from history-provider state; both hooks here receive the
//!   session itself, so they are not needed.
//! - **Origin attribution.** Upstream tags the injected memory message with
//!   the other sessions that contributed to the loaded topics
//!   (`origin_session_ids`). The Rust context has no such field yet.
//! - **Every client error is treated as transient** during extraction and
//!   consolidation (logged and skipped); upstream re-raises programming
//!   errors.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use regex::Regex;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::client::ChatClient;
use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::storage_keys::storage_key_segment;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::{ChatOptions, Content, Message, Role};

use super::atomic_write_text;

/// The default `source_id` of a [`MemoryContextProvider`].
pub const DEFAULT_MEMORY_SOURCE_ID: &str = "memory";
/// The default heading of the injected memory message.
pub const DEFAULT_MEMORY_CONTEXT_PROMPT: &str =
    "## Memory\nUse MEMORY.md and the loaded topic files when they are relevant.";
/// The default index file name.
pub const DEFAULT_MEMORY_INDEX_FILE_NAME: &str = "MEMORY.md";
/// The default topics directory name.
pub const DEFAULT_MEMORY_TOPICS_DIRECTORY_NAME: &str = "topics";
/// The default transcripts directory name.
pub const DEFAULT_MEMORY_TRANSCRIPTS_DIRECTORY_NAME: &str = "transcripts";
/// The default maintenance state file name.
pub const DEFAULT_MEMORY_STATE_FILE_NAME: &str = "state.json";
/// The default maximum number of pointer lines in `MEMORY.md`.
pub const DEFAULT_MEMORY_INDEX_LINE_LIMIT: usize = 200;
/// The default maximum length of one pointer line.
pub const DEFAULT_MEMORY_INDEX_LINE_LENGTH: usize = 150;
/// The default number of topic files auto-loaded per run.
pub const DEFAULT_MEMORY_SELECTION_LIMIT: usize = 3;
/// The default number of sessions before consolidation runs.
pub const DEFAULT_MEMORY_CONSOLIDATION_MIN_SESSIONS: usize = 5;
/// The default maximum number of memories extracted per run.
pub const DEFAULT_MEMORY_MAX_EXTRACTIONS: usize = 5;
/// The default minimum time between consolidations.
pub const DEFAULT_MEMORY_CONSOLIDATION_INTERVAL: Duration = Duration::from_secs(24 * 60 * 60);
/// The first line of `MEMORY.md`.
pub const DEFAULT_MEMORY_INDEX_HEADER: &str = "# MEMORY";
/// The index line written when there are no topics.
pub const DEFAULT_MEMORY_NO_TOPICS_TEXT: &str = "- none yet";
/// The default extraction prompt.
pub const DEFAULT_MEMORY_EXTRACTION_PROMPT: &str = r#"You extract durable memory candidates from an agent transcript delta.

Return only JSON with this exact shape:
{"memories":[{"topic":"short topic name","memory":"durable fact"}]}

Rules:
- include only durable facts, preferences, decisions, or patterns worth remembering later
- do not include transient tasks, temporary reminders, one-off outputs, or tool chatter
- keep topic names short and stable
- keep each memory item to one sentence
- return at most 5 memory items
- return {"memories": []} when nothing should be remembered
"#;
/// The default consolidation prompt.
pub const DEFAULT_MEMORY_CONSOLIDATION_PROMPT: &str = r#"You consolidate one topic memory file into a tighter durable form.

Return only JSON with this exact shape:
{"summary":"short summary","memories":["memory 1","memory 2"]}

Rules:
- preserve durable facts, preferences, and decisions
- remove duplicates and overlaps
- drop stale or obviously transient items
- keep the summary concise
- keep the memory list short and high-signal
"#;

const MEMORY_SEGMENT_PREFIX: &str = "~memory-";
const TRANSCRIPT_SESSION_PREFIX: &str = "~session-";
const NO_TOPICS_ITEM: &str = "none yet";

const MEMORY_INSTRUCTIONS: [&str; 8] = [
    "Use MEMORY.md as the always-loaded table of contents for durable memory.",
    "Use the loaded topic files when they are relevant, but do not assume every topic file was loaded.",
    "Use loaded recent transcript turns for short-term continuity when they are present, and use topic files for durable memory.",
    "Recent transcript loading can omit grouped tool-call turns, so use transcript search only when exact raw tool chatter is needed.",
    "Use write_memory to save durable facts, decisions, or preferences into a topic file.",
    "Use read_memory_topic to inspect or correct a specific topic file before editing it.",
    "Use search_memory_transcripts only when raw historical detail is necessary, because it searches the raw archive.",
    "Use consolidate_memories only when the user asks to rebuild or clean up memory explicitly.",
];

// ---------------------------------------------------------------------------
// Text helpers
// ---------------------------------------------------------------------------

fn collapse_whitespace(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_topic(topic: &str) -> Result<String> {
    let normalized = collapse_whitespace(topic);
    if normalized.is_empty() {
        return Err(Error::tool("topic must not be empty."));
    }
    Ok(normalized)
}

fn normalize_memory_text(memory: &str) -> Result<String> {
    let normalized = collapse_whitespace(memory);
    if normalized.is_empty() {
        return Err(Error::tool("memory must not be empty."));
    }
    Ok(normalized)
}

/// Escape a line that would read back as a heading.
fn escape_markdown_line(line: &str) -> String {
    let body = line.trim_start();
    if body.starts_with('#') || body.starts_with("\\#") {
        format!("\\{line}")
    } else {
        line.to_string()
    }
}

fn unescape_markdown_line(line: &str) -> String {
    let lead = line.len() - line.trim_start().len();
    let body = &line[lead..];
    match body.strip_prefix('\\') {
        Some(rest) if rest.starts_with('#') => format!("{}{rest}", &line[..lead]),
        _ => line.to_string(),
    }
}

/// The stable file stem of a topic: lowercase ASCII letters and digits
/// joined by `-`.
pub fn slugify_topic(topic: &str) -> Result<String> {
    let lowered = normalize_topic(topic)?.to_lowercase();
    let mut slug = String::new();
    let mut gap = false;
    for c in lowered.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() {
            if gap && !slug.is_empty() {
                slug.push('-');
            }
            gap = false;
            slug.push(c);
        } else {
            gap = true;
        }
    }
    Ok(if slug.is_empty() {
        "memory-topic".into()
    } else {
        slug
    })
}

fn dedupe_strings<S: AsRef<str>>(values: &[S]) -> Vec<String> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for value in values {
        let v = value.as_ref().trim();
        if !v.is_empty() && seen.insert(v.to_lowercase()) {
            out.push(v.to_string());
        }
    }
    out
}

fn coerce_summary(summary: &str, memories: &[String]) -> String {
    let normalized = collapse_whitespace(summary);
    if !normalized.is_empty() {
        return normalized;
    }
    match memories {
        [] => "No summary yet.".into(),
        [one] => one.clone(),
        [a, b, ..] => format!("{a} {b}").trim().to_string(),
    }
}

fn trim_pointer_line(line: &str, max_length: usize) -> String {
    if line.chars().count() <= max_length {
        return line.to_string();
    }
    if max_length <= 3 {
        return line.chars().take(max_length).collect();
    }
    let head: String = line.chars().take(max_length - 3).collect();
    format!("{}...", head.trim_end())
}

/// The JSON inside a fenced code block, or the trimmed text.
fn extract_json_text(text: &str) -> &str {
    let stripped = text.trim();
    if stripped.starts_with("```") && stripped.ends_with("```") {
        let lines: Vec<&str> = stripped.lines().collect();
        if lines.len() >= 3 {
            let start = stripped.find('\n').map_or(0, |i| i + 1);
            let end = stripped.rfind('\n').unwrap_or(stripped.len());
            return stripped[start..end].trim();
        }
    }
    stripped
}

fn preview(text: &str) -> String {
    let flat = collapse_whitespace(text);
    if flat.chars().count() > 120 {
        format!("{}…", flat.chars().take(120).collect::<String>())
    } else {
        flat
    }
}

fn word_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // A Unicode letter or digit, then word characters or `-`: so CJK and
    // Cyrillic text yields keywords too.
    RE.get_or_init(|| Regex::new(r"[^\W_][\w-]+").expect("valid regex"))
}

fn keywords(text: &str) -> HashSet<String> {
    word_pattern()
        .find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect()
}

// ---------------------------------------------------------------------------
// Timestamps (UTC, second precision, `+00:00`, as Python's isoformat)
// ---------------------------------------------------------------------------

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// Format seconds since the Unix epoch as `YYYY-MM-DDTHH:MM:SS+00:00`.
pub fn format_memory_timestamp(secs: i64) -> String {
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}+00:00",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Parse an ISO 8601 timestamp (`YYYY-MM-DD[T ]HH:MM[:SS[.fff]]` with an
/// optional `Z` or `±HH:MM` offset, UTC when absent) to Unix seconds.
pub fn parse_memory_timestamp(s: &str) -> Option<i64> {
    let s = s.trim();
    let (date, rest) = (s.get(..10)?, s.get(10..).unwrap_or(""));
    let mut parts = date.split('-');
    let y: i64 = parts.next()?.parse().ok()?;
    let m: u32 = parts.next()?.parse().ok()?;
    let d: u32 = parts.next()?.parse().ok()?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, m, d) * 86_400;
    if rest.is_empty() {
        return Some(secs);
    }
    let rest = rest.strip_prefix('T').or_else(|| rest.strip_prefix(' '))?;
    let (time, offset) = match rest.find(['Z', 'z', '+', '-']) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let time = time.split('.').next()?;
    let mut hms = time.split(':');
    let h: i64 = hms.next()?.parse().ok()?;
    let mi: i64 = hms.next()?.parse().ok()?;
    let se: i64 = hms.next().map_or(Some(0), |v| v.parse().ok())?;
    secs += h * 3600 + mi * 60 + se;
    match offset {
        "" | "Z" | "z" => {}
        o => {
            let sign = if o.starts_with('-') { -1 } else { 1 };
            let mut hm = o[1..].split(':');
            let oh: i64 = hm.next()?.parse().ok()?;
            let om: i64 = hm.next().map_or(Some(0), |v| v.parse().ok())?;
            secs -= sign * (oh * 3600 + om * 60);
        }
    }
    Some(secs)
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

/// One pointer line of `MEMORY.md`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Deserialize)]
pub struct MemoryIndexEntry {
    /// The topic name.
    pub topic: String,
    /// The topic file's stem.
    pub slug: String,
    /// The short summary shown in the index.
    pub summary: String,
    /// When the topic was last updated.
    pub updated_at: String,
}

impl MemoryIndexEntry {
    /// The index entry for `record`.
    pub fn from_topic_record(record: &MemoryTopicRecord) -> Self {
        Self {
            topic: record.topic.clone(),
            slug: record.slug.clone(),
            summary: coerce_summary(&record.summary, &record.memories),
            updated_at: record.updated_at.clone(),
        }
    }

    /// `- [topic](topics/slug.md): summary`, cut to `max_length` characters.
    pub fn to_pointer_line(&self, max_length: usize) -> String {
        trim_pointer_line(
            &format!(
                "- [{}](topics/{}.md): {}",
                self.topic, self.slug, self.summary
            ),
            max_length,
        )
    }
}

/// One topic memory file.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, Deserialize)]
pub struct MemoryTopicRecord {
    /// The topic name.
    pub topic: String,
    /// The file stem, derived from the topic unless given.
    pub slug: String,
    /// A short summary.
    pub summary: String,
    /// The memory bullets, de-duplicated ignoring case.
    pub memories: Vec<String>,
    /// When the topic was last updated.
    pub updated_at: String,
    /// The sessions that contributed to the topic.
    pub session_ids: Vec<String>,
}

impl MemoryTopicRecord {
    /// A normalized record: whitespace collapsed in the topic, the slug
    /// derived from `slug` or the topic, memories and session ids
    /// de-duplicated, and an empty summary filled from the memories.
    pub fn new(
        topic: &str,
        slug: Option<&str>,
        summary: &str,
        memories: &[String],
        updated_at: impl Into<String>,
        session_ids: &[String],
    ) -> Result<Self> {
        let topic = normalize_topic(topic)?;
        let slug = slugify_topic(slug.unwrap_or(&topic))?;
        let memories = dedupe_strings(memories);
        Ok(Self {
            summary: coerce_summary(summary, &memories),
            topic,
            slug,
            memories,
            updated_at: updated_at.into(),
            session_ids: dedupe_strings(session_ids),
        })
    }

    /// The canonical markdown file.
    pub fn to_markdown(&self) -> String {
        let sessions = if self.session_ids.is_empty() {
            "-".to_string()
        } else {
            self.session_ids.join(", ")
        };
        let summary: Vec<String> = self.summary.lines().map(escape_markdown_line).collect();
        let memories: Vec<String> = if self.memories.is_empty() {
            vec![format!("- {NO_TOPICS_ITEM}")]
        } else {
            self.memories
                .iter()
                .map(|m| format!("- {}", escape_markdown_line(m)))
                .collect()
        };
        let mut lines = vec![
            format!("# {}", self.topic),
            String::new(),
            format!("Updated: {}", self.updated_at),
            format!("Sessions: {sessions}"),
            String::new(),
            "## Summary".into(),
            summary.join("\n"),
            String::new(),
            "## Memories".into(),
        ];
        lines.extend(memories);
        lines.join("\n").trim_end().to_string()
    }

    /// Parse the canonical markdown, using `fallback_topic` when the file has
    /// no `# topic` heading.
    pub fn from_markdown(markdown: &str, fallback_topic: Option<&str>) -> Result<Self> {
        let mut topic = fallback_topic.map(str::to_string);
        let mut updated_at = format_memory_timestamp(now_secs());
        let mut session_ids: Vec<String> = Vec::new();
        let mut summary: Vec<String> = Vec::new();
        let mut memories: Vec<String> = Vec::new();
        #[derive(PartialEq)]
        enum Section {
            None,
            Summary,
            Memories,
        }
        let mut section = Section::None;
        for raw in markdown.lines() {
            let stripped = raw.trim();
            if let Some(t) = stripped.strip_prefix("# ") {
                topic = Some(t.trim().to_string());
                section = Section::None;
                continue;
            }
            if section == Section::None {
                if let Some(u) = stripped.strip_prefix("Updated: ") {
                    updated_at = u.trim().to_string();
                    continue;
                }
                if let Some(s) = stripped.strip_prefix("Sessions: ") {
                    let s = s.trim();
                    session_ids = if s.is_empty() || s == "-" {
                        Vec::new()
                    } else {
                        s.split(',')
                            .map(str::trim)
                            .filter(|i| !i.is_empty())
                            .map(str::to_string)
                            .collect()
                    };
                    continue;
                }
            }
            match stripped {
                "## Summary" => section = Section::Summary,
                "## Memories" => section = Section::Memories,
                _ => match section {
                    Section::Summary if !stripped.is_empty() => {
                        summary.push(unescape_markdown_line(stripped))
                    }
                    Section::Memories => {
                        if let Some(item) = stripped.strip_prefix("- ") {
                            let text = unescape_markdown_line(item.trim());
                            if !text.is_empty() && text != NO_TOPICS_ITEM {
                                memories.push(text);
                            }
                        }
                    }
                    _ => {}
                },
            }
        }
        let topic = topic.ok_or_else(|| {
            Error::Serialization("Memory topic markdown is missing a '# <topic>' heading.".into())
        })?;
        Self::new(
            &topic,
            None,
            summary.join("\n").trim(),
            &memories,
            updated_at,
            &session_ids,
        )
    }
}

/// The maintenance state a [`MemoryStore`] keeps per owner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MemoryMaintenanceState {
    /// When consolidation last succeeded.
    pub last_consolidated_at: Option<String>,
    /// The sessions seen since then.
    pub sessions_since_consolidation: Vec<String>,
    /// Any other keys, kept as they were.
    pub extra: Map<String, Value>,
}

impl MemoryMaintenanceState {
    fn from_value(value: Value) -> Result<Self> {
        let Value::Object(mut map) = value else {
            return Err(Error::Serialization(
                "Memory state file must contain a JSON object.".into(),
            ));
        };
        let last = match map.remove("last_consolidated_at") {
            Some(Value::String(s)) => Some(s),
            _ => None,
        };
        let sessions = match map.remove("sessions_since_consolidation") {
            Some(Value::Array(items)) => items
                .into_iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            last_consolidated_at: last,
            sessions_since_consolidation: sessions,
            extra: map,
        })
    }

    fn to_value(&self) -> Value {
        let mut map = self.extra.clone();
        map.insert(
            "last_consolidated_at".into(),
            json!(self.last_consolidated_at),
        );
        map.insert(
            "sessions_since_consolidation".into(),
            json!(self.sessions_since_consolidation),
        );
        Value::Object(map)
    }
}

/// One transcript line matching a search.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TranscriptSearchResult {
    /// The session the transcript belongs to, when it can be told.
    pub session_id: Option<String>,
    /// The 1-based line of the transcript file.
    pub line_number: usize,
    /// The message's role.
    pub role: String,
    /// The message's text.
    pub text: String,
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// The backing store of a [`MemoryContextProvider`].
#[async_trait]
pub trait MemoryStore: Send + Sync {
    /// The logical owner of `session`'s memory, if the store has owners.
    fn owner_id(&self, session: &AgentSession) -> Result<Option<String>> {
        let _ = session;
        Ok(None)
    }

    /// Every topic visible to `session`'s owner, ordered by topic ignoring
    /// case.
    async fn list_topics(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<Vec<MemoryTopicRecord>>;

    /// One topic by name or slug, or `None`.
    async fn get_topic(
        &self,
        session: &AgentSession,
        source_id: &str,
        topic: &str,
    ) -> Result<Option<MemoryTopicRecord>>;

    /// Write a topic file.
    async fn write_topic(
        &self,
        session: &AgentSession,
        record: &MemoryTopicRecord,
        source_id: &str,
    ) -> Result<()>;

    /// Delete a topic, returning whether it existed.
    async fn delete_topic(
        &self,
        session: &AgentSession,
        source_id: &str,
        topic: &str,
    ) -> Result<bool>;

    /// Rewrite `MEMORY.md` from the topics and return its entries (at most
    /// `line_limit`).
    async fn rebuild_index(
        &self,
        session: &AgentSession,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
    ) -> Result<Vec<MemoryIndexEntry>>;

    /// The text of `MEMORY.md`.
    async fn index_text(&self, session: &AgentSession, source_id: &str) -> Result<String>;

    /// The owner's maintenance state.
    async fn read_state(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<MemoryMaintenanceState>;

    /// Persist the owner's maintenance state.
    async fn write_state(
        &self,
        session: &AgentSession,
        state: &MemoryMaintenanceState,
        source_id: &str,
    ) -> Result<()>;

    /// Append `messages` to `session`'s transcript.
    async fn append_transcript(
        &self,
        session: &AgentSession,
        source_id: &str,
        messages: &[Message],
    ) -> Result<()>;

    /// `session`'s transcript, oldest first.
    async fn load_transcript(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<Vec<Message>>;

    /// Transcript messages whose text contains `query` (ignoring case),
    /// across the owner's sessions or only `session_id`'s, at most `limit`.
    async fn search_transcripts(
        &self,
        session: &AgentSession,
        source_id: &str,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TranscriptSearchResult>>;
}

/// A [`MemoryStore`] on the local file system. See the [module docs](self).
///
/// The base directory must not be writable by untrusted users, who could
/// replace the files behind this store's back.
#[derive(Debug, Clone)]
pub struct MemoryFileStore {
    base_path: PathBuf,
    kind: String,
    owner_prefix: String,
    owner_state_key: String,
    index_file_name: String,
    topics_directory_name: String,
    transcripts_directory_name: String,
    state_file_name: String,
}

impl MemoryFileStore {
    /// A store under `base_path` whose owner id is read from
    /// `session.state[owner_state_key]`.
    pub fn new(base_path: impl Into<PathBuf>, owner_state_key: impl Into<String>) -> Self {
        Self {
            base_path: base_path.into(),
            kind: "memory".into(),
            owner_prefix: String::new(),
            owner_state_key: owner_state_key.into(),
            index_file_name: DEFAULT_MEMORY_INDEX_FILE_NAME.into(),
            topics_directory_name: DEFAULT_MEMORY_TOPICS_DIRECTORY_NAME.into(),
            transcripts_directory_name: DEFAULT_MEMORY_TRANSCRIPTS_DIRECTORY_NAME.into(),
            state_file_name: DEFAULT_MEMORY_STATE_FILE_NAME.into(),
        }
    }

    /// The bucket name under each owner (default `memory`).
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = kind.into();
        self
    }

    /// A prefix applied to the owner id.
    pub fn with_owner_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.owner_prefix = prefix.into();
        self
    }

    /// Override the index, topics, transcripts and state names.
    pub fn with_file_names(
        mut self,
        index_file_name: impl Into<String>,
        topics_directory_name: impl Into<String>,
        transcripts_directory_name: impl Into<String>,
        state_file_name: impl Into<String>,
    ) -> Self {
        self.index_file_name = index_file_name.into();
        self.topics_directory_name = topics_directory_name.into();
        self.transcripts_directory_name = transcripts_directory_name.into();
        self.state_file_name = state_file_name.into();
        self
    }

    fn owner(&self, session: &AgentSession) -> Result<String> {
        let value = session
            .state
            .get(&self.owner_state_key)
            .filter(|v| !v.is_null())
            .ok_or_else(|| {
                Error::Configuration(format!(
                    "MemoryFileStore requires session.state[{:?}] to be set for file-backed storage.",
                    self.owner_state_key
                ))
            })?;
        let owner = match value {
            Value::String(s) => s,
            other => other.to_string(),
        };
        let path = Path::new(&owner);
        if path.is_absolute() || owner.split(['/', '\\']).any(|p| p == "..") {
            return Err(Error::Configuration(
                "Memory owner ID must not contain path traversal segments.".into(),
            ));
        }
        Ok(owner)
    }

    /// The directory holding `session`'s owner's memory for `source_id`.
    pub fn memory_root(&self, session: &AgentSession, source_id: &str) -> Result<PathBuf> {
        let owner = self.owner(session)?;
        Ok(self
            .base_path
            .join(storage_key_segment(source_id, MEMORY_SEGMENT_PREFIX))
            .join(storage_key_segment(
                &format!("{}{owner}", self.owner_prefix),
                MEMORY_SEGMENT_PREFIX,
            ))
            .join(&self.kind))
    }

    fn topic_path(&self, session: &AgentSession, source_id: &str, topic: &str) -> Result<PathBuf> {
        Ok(self
            .memory_root(session, source_id)?
            .join(&self.topics_directory_name)
            .join(format!("{}.md", slugify_topic(topic)?)))
    }

    fn transcripts_dir(&self, session: &AgentSession, source_id: &str) -> Result<PathBuf> {
        Ok(self
            .memory_root(session, source_id)?
            .join(&self.transcripts_directory_name))
    }

    fn transcript_stem(session_id: &str) -> String {
        storage_key_segment(session_id, TRANSCRIPT_SESSION_PREFIX)
    }

    /// The session id a transcript file stem names, when it can be decoded.
    fn decode_transcript_stem(stem: &str) -> Option<String> {
        match stem.strip_prefix(TRANSCRIPT_SESSION_PREFIX) {
            None => Some(stem.to_string()),
            Some(encoded) => decode_hex(encoded),
        }
    }

    async fn write_index(&self, session: &AgentSession, source_id: &str, text: &str) -> Result<()> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.index_file_name);
        if tokio::fs::read_to_string(&path).await.ok().as_deref() != Some(text) {
            atomic_write_text(&path, text).await?;
        }
        Ok(())
    }
}

/// The UTF-8 string `hex` encodes, as [`storage_key_segment`] writes it.
fn decode_hex(hex: &str) -> Option<String> {
    if !hex.len().is_multiple_of(2) {
        return None;
    }
    let bytes: Option<Vec<u8>> = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(hex.get(i..i + 2)?, 16).ok())
        .collect();
    String::from_utf8(bytes?).ok()
}

fn io_err(path: &Path, e: std::io::Error) -> Error {
    Error::Other(format!("{}: {e}", path.display()))
}

async fn read_optional(path: &Path) -> Result<Option<String>> {
    match tokio::fs::read_to_string(path).await {
        Ok(text) => Ok(Some(text)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io_err(path, e)),
    }
}

fn index_text_for(entries: &[MemoryIndexEntry], line_length: usize) -> String {
    let mut lines = vec![DEFAULT_MEMORY_INDEX_HEADER.to_string(), String::new()];
    if entries.is_empty() {
        lines.push(DEFAULT_MEMORY_NO_TOPICS_TEXT.into());
    } else {
        lines.extend(entries.iter().map(|e| e.to_pointer_line(line_length)));
    }
    format!("{}\n", lines.join("\n").trim_end())
}

#[async_trait]
impl MemoryStore for MemoryFileStore {
    fn owner_id(&self, session: &AgentSession) -> Result<Option<String>> {
        self.owner(session).map(Some)
    }

    async fn list_topics(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<Vec<MemoryTopicRecord>> {
        let dir = self
            .memory_root(session, source_id)?
            .join(&self.topics_directory_name);
        let mut reader = match tokio::fs::read_dir(&dir).await {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err(&dir, e)),
        };
        let mut paths = Vec::new();
        while let Some(entry) = reader.next_entry().await.map_err(|e| io_err(&dir, e))? {
            let path = entry.path();
            let is_file = entry
                .file_type()
                .await
                .map(|t| t.is_file())
                .unwrap_or(false);
            if is_file && path.extension().is_some_and(|e| e == "md") {
                paths.push(path);
            }
        }
        paths.sort();
        let mut topics = Vec::new();
        for path in paths {
            let text = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| io_err(&path, e))?;
            let fallback = path
                .file_stem()
                .map(|s| s.to_string_lossy().replace('-', " "));
            topics.push(MemoryTopicRecord::from_markdown(
                &text,
                fallback.as_deref(),
            )?);
        }
        topics.sort_by(|a, b| {
            (a.topic.to_lowercase(), &a.updated_at).cmp(&(b.topic.to_lowercase(), &b.updated_at))
        });
        Ok(topics)
    }

    async fn get_topic(
        &self,
        session: &AgentSession,
        source_id: &str,
        topic: &str,
    ) -> Result<Option<MemoryTopicRecord>> {
        let path = self.topic_path(session, source_id, topic)?;
        match read_optional(&path).await? {
            Some(text) => Ok(Some(MemoryTopicRecord::from_markdown(&text, Some(topic))?)),
            None => Ok(None),
        }
    }

    async fn write_topic(
        &self,
        session: &AgentSession,
        record: &MemoryTopicRecord,
        source_id: &str,
    ) -> Result<()> {
        let path = self.topic_path(session, source_id, &record.slug)?;
        atomic_write_text(&path, &format!("{}\n", record.to_markdown())).await
    }

    async fn delete_topic(
        &self,
        session: &AgentSession,
        source_id: &str,
        topic: &str,
    ) -> Result<bool> {
        let path = self.topic_path(session, source_id, topic)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(io_err(&path, e)),
        }
    }

    async fn rebuild_index(
        &self,
        session: &AgentSession,
        source_id: &str,
        line_limit: usize,
        line_length: usize,
    ) -> Result<Vec<MemoryIndexEntry>> {
        let mut entries: Vec<MemoryIndexEntry> = self
            .list_topics(session, source_id)
            .await?
            .iter()
            .map(MemoryIndexEntry::from_topic_record)
            .collect();
        entries.truncate(line_limit);
        self.write_index(session, source_id, &index_text_for(&entries, line_length))
            .await?;
        Ok(entries)
    }

    async fn index_text(&self, session: &AgentSession, source_id: &str) -> Result<String> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.index_file_name);
        Ok(read_optional(&path)
            .await?
            .unwrap_or_else(|| index_text_for(&[], DEFAULT_MEMORY_INDEX_LINE_LENGTH))
            .trim()
            .to_string())
    }

    async fn read_state(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<MemoryMaintenanceState> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.state_file_name);
        match read_optional(&path).await? {
            None => Ok(MemoryMaintenanceState::default()),
            Some(text) => {
                MemoryMaintenanceState::from_value(serde_json::from_str(&text).map_err(|e| {
                    Error::Serialization(format!("{} is not valid JSON: {e}", path.display()))
                })?)
            }
        }
    }

    async fn write_state(
        &self,
        session: &AgentSession,
        state: &MemoryMaintenanceState,
        source_id: &str,
    ) -> Result<()> {
        let path = self
            .memory_root(session, source_id)?
            .join(&self.state_file_name);
        atomic_write_text(&path, &format!("{}\n", state.to_value())).await
    }

    async fn append_transcript(
        &self,
        session: &AgentSession,
        source_id: &str,
        messages: &[Message],
    ) -> Result<()> {
        use tokio::io::AsyncWriteExt;
        let dir = self.transcripts_dir(session, source_id)?;
        tokio::fs::create_dir_all(&dir)
            .await
            .map_err(|e| io_err(&dir, e))?;
        let path = dir.join(format!(
            "{}.jsonl",
            Self::transcript_stem(session.session_id())
        ));
        let mut payload = String::new();
        for message in messages {
            payload.push_str(&serde_json::to_string(message)?);
            payload.push('\n');
        }
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .await
            .map_err(|e| io_err(&path, e))?;
        file.write_all(payload.as_bytes())
            .await
            .map_err(|e| io_err(&path, e))?;
        file.flush().await.map_err(|e| io_err(&path, e))
    }

    async fn load_transcript(
        &self,
        session: &AgentSession,
        source_id: &str,
    ) -> Result<Vec<Message>> {
        let path = self.transcripts_dir(session, source_id)?.join(format!(
            "{}.jsonl",
            Self::transcript_stem(session.session_id())
        ));
        let Some(text) = read_optional(&path).await? else {
            return Ok(Vec::new());
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str(l).map_err(|e| {
                    Error::Serialization(format!("bad transcript line in {}: {e}", path.display()))
                })
            })
            .collect()
    }

    async fn search_transcripts(
        &self,
        session: &AgentSession,
        source_id: &str,
        query: &str,
        session_id: Option<&str>,
        limit: usize,
    ) -> Result<Vec<TranscriptSearchResult>> {
        let query = query.trim();
        if query.is_empty() {
            return Err(Error::tool("query must not be empty."));
        }
        let needle = query.to_lowercase();
        let dir = self.transcripts_dir(session, source_id)?;
        let mut reader = match tokio::fs::read_dir(&dir).await {
            Ok(r) => r,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io_err(&dir, e)),
        };
        let mut files = Vec::new();
        while let Some(entry) = reader.next_entry().await.map_err(|e| io_err(&dir, e))? {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "jsonl") {
                files.push(path);
            }
        }
        files.sort();
        let expected = session_id.map(Self::transcript_stem);
        let mut results = Vec::new();
        for file in files {
            let stem = file
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let matched = match (&expected, session_id) {
                (Some(want), Some(id)) => {
                    if &stem != want {
                        continue;
                    }
                    Some(id.to_string())
                }
                _ => Self::decode_transcript_stem(&stem),
            };
            let text = tokio::fs::read_to_string(&file)
                .await
                .map_err(|e| io_err(&file, e))?;
            for (index, line) in text.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let Ok(message) = serde_json::from_str::<Message>(line) else {
                    continue;
                };
                let body = message.text().trim().to_string();
                if body.is_empty() || !body.to_lowercase().contains(&needle) {
                    continue;
                }
                results.push(TranscriptSearchResult {
                    session_id: matched.clone(),
                    line_number: index + 1,
                    role: message.role.as_str().to_string(),
                    text: body,
                });
                if results.len() >= limit {
                    return Ok(results);
                }
            }
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Recent turns
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpanKind {
    System,
    User,
    AssistantText,
    ToolCall,
}

fn has_tool_call(m: &Message) -> bool {
    m.contents
        .iter()
        .any(|c| matches!(c, Content::FunctionCall(_)))
}

fn is_tool_call_assistant(m: &Message) -> bool {
    m.role.as_str() == Role::ASSISTANT && has_tool_call(m)
}

fn is_reasoning_only_assistant(m: &Message) -> bool {
    m.role.as_str() == Role::ASSISTANT
        && !m.contents.is_empty()
        && m.contents
            .iter()
            .all(|c| matches!(c, Content::TextReasoning(_)))
}

/// Group messages into spans as upstream's compaction `group_messages`:
/// system and user messages alone, an assistant tool call with its reasoning
/// prefix and results as one `tool_call` span, other assistant messages
/// alone.
fn group_spans(messages: &[Message]) -> Vec<(SpanKind, usize, usize)> {
    let mut spans = Vec::new();
    let n = messages.len();
    let mut i = 0;
    let extend_tool = |mut k: usize| {
        while k < n && is_reasoning_only_assistant(&messages[k]) {
            k += 1;
        }
        while k < n && messages[k].role.as_str() == Role::TOOL {
            k += 1;
        }
        k
    };
    while i < n {
        let m = &messages[i];
        match m.role.as_str() {
            Role::SYSTEM => {
                spans.push((SpanKind::System, i, i));
                i += 1;
                continue;
            }
            Role::USER => {
                spans.push((SpanKind::User, i, i));
                i += 1;
                continue;
            }
            _ => {}
        }
        if is_reasoning_only_assistant(m) {
            let mut j = i;
            while j < n && is_reasoning_only_assistant(&messages[j]) {
                j += 1;
            }
            if j < n && is_tool_call_assistant(&messages[j]) {
                let k = extend_tool(j + 1);
                spans.push((SpanKind::ToolCall, i, k - 1));
                i = k;
                continue;
            }
        }
        if is_tool_call_assistant(m) {
            let k = extend_tool(i + 1);
            spans.push((SpanKind::ToolCall, i, k - 1));
            i = k;
            continue;
        }
        if m.role.as_str() == Role::TOOL {
            let mut k = i + 1;
            while k < n && messages[k].role.as_str() == Role::TOOL {
                k += 1;
            }
            spans.push((SpanKind::ToolCall, i, k - 1));
            i = k;
            continue;
        }
        spans.push((SpanKind::AssistantText, i, i));
        i += 1;
    }
    spans
}

/// The last `turn_count` turns of `messages` (a turn starts at a user
/// message), optionally without tool-call spans.
fn select_recent_turns(
    messages: &[Message],
    turn_count: usize,
    load_tool_turns: bool,
) -> Vec<Message> {
    if turn_count == 0 || messages.is_empty() {
        return Vec::new();
    }
    let spans = group_spans(messages);
    let users: Vec<usize> = spans
        .iter()
        .enumerate()
        .filter(|(_, s)| s.0 == SpanKind::User)
        .map(|(i, _)| i)
        .collect();
    let start = if users.is_empty() {
        0
    } else {
        users[users.len().saturating_sub(turn_count)]
    };
    spans[start..]
        .iter()
        .filter(|s| load_tool_turns || s.0 != SpanKind::ToolCall)
        .flat_map(|&(_, a, b)| messages[a..=b].iter().cloned())
        .collect()
}

fn format_for_model(messages: &[Message]) -> String {
    messages
        .iter()
        .filter_map(|m| {
            let text = m.text();
            let text = text.trim();
            (!text.is_empty()).then(|| format!("{}: {text}", m.role.as_str().to_uppercase()))
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

/// Rewrites or drops a message before it is written to the transcript.
pub type HistoryMessageFilter = Arc<dyn Fn(Message) -> Option<Message> + Send + Sync>;

/// A [`ContextProvider`] giving an agent durable, topic-based memory. See
/// the [module docs](self).
#[derive(Clone)]
pub struct MemoryContextProvider {
    inner: Arc<ProviderInner>,
}

struct ProviderInner {
    store: Arc<dyn MemoryStore>,
    source_id: String,
    context_prompt: String,
    index_line_limit: usize,
    index_line_length: usize,
    selection_limit: usize,
    recent_turns: usize,
    load_tool_turns: bool,
    max_extractions: usize,
    consolidation_interval: Duration,
    consolidation_min_sessions: usize,
    extraction_prompt: String,
    consolidation_prompt: String,
    chat_client: Option<Arc<dyn ChatClient>>,
    consolidation_client: Option<Arc<dyn ChatClient>>,
    history_message_filter: Option<HistoryMessageFilter>,
}

impl std::fmt::Debug for MemoryContextProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryContextProvider")
            .field("source_id", &self.inner.source_id)
            .field("recent_turns", &self.inner.recent_turns)
            .field("selection_limit", &self.inner.selection_limit)
            .finish_non_exhaustive()
    }
}

/// Builder for a [`MemoryContextProvider`].
pub struct MemoryContextProviderBuilder {
    inner: ProviderInner,
}

impl MemoryContextProviderBuilder {
    /// Set the `source_id`.
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.inner.source_id = source_id.into();
        self
    }
    /// Replace the heading of the injected memory message.
    pub fn context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner.context_prompt = prompt.into();
        self
    }
    /// The maximum number of pointer lines in `MEMORY.md` (default 200).
    pub fn index_line_limit(mut self, limit: usize) -> Self {
        self.inner.index_line_limit = limit;
        self
    }
    /// The maximum length of one pointer line (default 150).
    pub fn index_line_length(mut self, length: usize) -> Self {
        self.inner.index_line_length = length;
        self
    }
    /// How many topic files to auto-load per run (default 3).
    pub fn selection_limit(mut self, limit: usize) -> Self {
        self.inner.selection_limit = limit;
        self
    }
    /// How many recent transcript turns to replay (default 0).
    pub fn recent_turns(mut self, turns: usize) -> Self {
        self.inner.recent_turns = turns;
        self
    }
    /// Whether replayed turns include tool-call spans (default true).
    pub fn load_tool_turns(mut self, load: bool) -> Self {
        self.inner.load_tool_turns = load;
        self
    }
    /// The maximum number of memories extracted per run (default 5).
    pub fn max_extractions(mut self, max: usize) -> Self {
        self.inner.max_extractions = max;
        self
    }
    /// The minimum time between automatic consolidations (default 24 h).
    pub fn consolidation_interval(mut self, interval: Duration) -> Self {
        self.inner.consolidation_interval = interval;
        self
    }
    /// Sessions required before automatic consolidation (default 5).
    pub fn consolidation_min_sessions(mut self, sessions: usize) -> Self {
        self.inner.consolidation_min_sessions = sessions;
        self
    }
    /// Replace the extraction prompt.
    pub fn extraction_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner.extraction_prompt = prompt.into();
        self
    }
    /// Replace the consolidation prompt.
    pub fn consolidation_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.inner.consolidation_prompt = prompt.into();
        self
    }
    /// The chat client used to extract memories (and to consolidate, unless
    /// [`consolidation_client`](Self::consolidation_client) is set). Usually
    /// the agent's own client.
    pub fn chat_client(mut self, client: Arc<dyn ChatClient>) -> Self {
        self.inner.chat_client = Some(client);
        self
    }
    /// A separate (for example cheaper) client for consolidation.
    pub fn consolidation_client(mut self, client: Arc<dyn ChatClient>) -> Self {
        self.inner.consolidation_client = Some(client);
        self
    }
    /// Rewrite or drop messages before they are written to the transcript.
    pub fn history_message_filter(
        mut self,
        filter: impl Fn(Message) -> Option<Message> + Send + Sync + 'static,
    ) -> Self {
        self.inner.history_message_filter = Some(Arc::new(filter));
        self
    }

    /// Build, validating the limits.
    pub fn build(self) -> Result<MemoryContextProvider> {
        let i = &self.inner;
        if i.index_line_limit == 0 {
            return Err(Error::Configuration(
                "index_line_limit must be greater than 0.".into(),
            ));
        }
        if i.index_line_length == 0 {
            return Err(Error::Configuration(
                "index_line_length must be greater than 0.".into(),
            ));
        }
        if i.consolidation_min_sessions == 0 {
            return Err(Error::Configuration(
                "consolidation_min_sessions must be greater than 0.".into(),
            ));
        }
        Ok(MemoryContextProvider {
            inner: Arc::new(self.inner),
        })
    }
}

type LockMap = HashMap<(usize, String), Weak<tokio::sync::Mutex<()>>>;

impl MemoryContextProvider {
    /// A builder over `store`.
    pub fn builder(store: Arc<dyn MemoryStore>) -> MemoryContextProviderBuilder {
        MemoryContextProviderBuilder {
            inner: ProviderInner {
                store,
                source_id: DEFAULT_MEMORY_SOURCE_ID.into(),
                context_prompt: DEFAULT_MEMORY_CONTEXT_PROMPT.into(),
                index_line_limit: DEFAULT_MEMORY_INDEX_LINE_LIMIT,
                index_line_length: DEFAULT_MEMORY_INDEX_LINE_LENGTH,
                selection_limit: DEFAULT_MEMORY_SELECTION_LIMIT,
                recent_turns: 0,
                load_tool_turns: true,
                max_extractions: DEFAULT_MEMORY_MAX_EXTRACTIONS,
                consolidation_interval: DEFAULT_MEMORY_CONSOLIDATION_INTERVAL,
                consolidation_min_sessions: DEFAULT_MEMORY_CONSOLIDATION_MIN_SESSIONS,
                extraction_prompt: DEFAULT_MEMORY_EXTRACTION_PROMPT.into(),
                consolidation_prompt: DEFAULT_MEMORY_CONSOLIDATION_PROMPT.into(),
                chat_client: None,
                consolidation_client: None,
                history_message_filter: None,
            },
        }
    }

    /// A provider over `store` with the default settings.
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self::builder(store).build().expect("defaults are valid")
    }

    /// The provider's `source_id`.
    pub fn source_id(&self) -> &str {
        &self.inner.source_id
    }

    /// The lock serializing edits of one topic, or of the maintenance state,
    /// for `session`'s owner. Shared by every provider on the same store.
    fn lock(&self, session: &AgentSession, kind: &str, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        static LOCKS: OnceLock<Mutex<LockMap>> = OnceLock::new();
        let owner = self
            .inner
            .store
            .owner_id(session)
            .ok()
            .flatten()
            .unwrap_or_default();
        let key = (
            Arc::as_ptr(&self.inner.store) as *const () as usize,
            format!("{}:{owner}:{kind}:{key}", self.inner.source_id),
        );
        let mut map = LOCKS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(lock) = map.get(&key).and_then(Weak::upgrade) {
            return lock;
        }
        map.retain(|_, w| w.strong_count() > 0);
        let lock = Arc::new(tokio::sync::Mutex::new(()));
        map.insert(key, Arc::downgrade(&lock));
        lock
    }

    async fn rebuild_index(&self, session: &AgentSession) -> Result<Vec<MemoryIndexEntry>> {
        self.inner
            .store
            .rebuild_index(
                session,
                &self.inner.source_id,
                self.inner.index_line_limit,
                self.inner.index_line_length,
            )
            .await
    }

    fn select_topics(
        &self,
        entries: &[MemoryIndexEntry],
        input: &[Message],
    ) -> Vec<MemoryIndexEntry> {
        if self.inner.selection_limit == 0 || entries.is_empty() {
            return Vec::new();
        }
        let words: HashSet<String> = input.iter().flat_map(|m| keywords(&m.text())).collect();
        if words.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, &MemoryIndexEntry)> = entries
            .iter()
            .map(|e| {
                let topic_words = keywords(&format!("{} {}", e.topic, e.summary));
                (topic_words.intersection(&words).count(), e)
            })
            .collect();
        scored.sort_by(|a, b| {
            b.0.cmp(&a.0)
                .then_with(|| a.1.topic.to_lowercase().cmp(&b.1.topic.to_lowercase()))
        });
        scored
            .into_iter()
            .take(self.inner.selection_limit)
            .filter(|(score, _)| *score > 0)
            .map(|(_, e)| e.clone())
            .collect()
    }

    /// Add `memory` to `topic` (creating it), returning the updated record.
    pub async fn merge_memory(
        &self,
        session: &AgentSession,
        topic: &str,
        memory: &str,
    ) -> Result<MemoryTopicRecord> {
        self.merge_memory_at(session, topic, memory, now_secs())
            .await
    }

    async fn merge_memory_at(
        &self,
        session: &AgentSession,
        topic: &str,
        memory: &str,
        now: i64,
    ) -> Result<MemoryTopicRecord> {
        let topic = normalize_topic(topic)?;
        let memory = normalize_memory_text(memory)?;
        let lock = self.lock(session, "topic", &slugify_topic(&topic)?);
        let _guard = lock.lock().await;
        let store = &self.inner.store;
        let sid = session.session_id().to_string();
        let record = match store
            .get_topic(session, &self.inner.source_id, &topic)
            .await?
        {
            None => MemoryTopicRecord::new(
                &topic,
                None,
                &memory,
                std::slice::from_ref(&memory),
                format_memory_timestamp(now),
                &[sid],
            )?,
            Some(existing) => {
                let mut memories = existing.memories.clone();
                memories.push(memory);
                let mut sessions = existing.session_ids.clone();
                sessions.push(sid);
                MemoryTopicRecord::new(
                    &existing.topic,
                    Some(&existing.slug),
                    &existing.summary,
                    &memories,
                    format_memory_timestamp(now),
                    &sessions,
                )?
            }
        };
        store
            .write_topic(session, &record, &self.inner.source_id)
            .await?;
        Ok(record)
    }

    fn should_consolidate(&self, state: &MemoryMaintenanceState, now: i64) -> bool {
        if state.sessions_since_consolidation.len() < self.inner.consolidation_min_sessions {
            return false;
        }
        match state.last_consolidated_at.as_deref() {
            None => true,
            Some(at) => match parse_memory_timestamp(at) {
                None => true,
                Some(then) => now - then >= self.inner.consolidation_interval.as_secs() as i64,
            },
        }
    }

    /// Consolidate every topic now (or only when due, without `force`),
    /// returning how many topics were consolidated.
    pub async fn consolidate(&self, session: &AgentSession, force: bool) -> Result<usize> {
        self.run_consolidation(session, force, now_secs()).await
    }

    async fn run_consolidation(
        &self,
        session: &AgentSession,
        force: bool,
        now: i64,
    ) -> Result<usize> {
        let store = &self.inner.store;
        let source = &self.inner.source_id;
        let topics = {
            let lock = self.lock(session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = store.read_state(session, source).await?;
            if !force && !self.should_consolidate(&state, now) {
                return Ok(0);
            }
            let topics = store.list_topics(session, source).await?;
            if topics.is_empty() {
                state.last_consolidated_at = Some(format_memory_timestamp(now));
                state.sessions_since_consolidation.clear();
                store.write_state(session, &state, source).await?;
                return Ok(0);
            }
            topics
        };
        let client = self
            .inner
            .consolidation_client
            .clone()
            .or_else(|| self.inner.chat_client.clone());
        let mut succeeded = 0;
        for record in topics {
            let lock = self.lock(session, "topic", &record.slug);
            let _guard = lock.lock().await;
            let Some(current) = store.get_topic(session, source, &record.slug).await? else {
                continue;
            };
            if let Some(consolidated) = self
                .consolidate_topic(client.as_deref(), &current, now)
                .await?
            {
                store.write_topic(session, &consolidated, source).await?;
                succeeded += 1;
            }
        }
        // Only advance the window when something consolidated, so a run of
        // transient failures is retried rather than skipped.
        if succeeded > 0 {
            let lock = self.lock(session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = store.read_state(session, source).await?;
            state.last_consolidated_at = Some(format_memory_timestamp(now));
            state.sessions_since_consolidation.clear();
            store.write_state(session, &state, source).await?;
            self.rebuild_index(session).await?;
        }
        Ok(succeeded)
    }

    /// The consolidated record, or `None` when the model failed or answered
    /// malformed output.
    async fn consolidate_topic(
        &self,
        client: Option<&dyn ChatClient>,
        record: &MemoryTopicRecord,
        now: i64,
    ) -> Result<Option<MemoryTopicRecord>> {
        let stamp = format_memory_timestamp(now);
        let Some(client) = client else {
            return Ok(Some(MemoryTopicRecord::new(
                &record.topic,
                Some(&record.slug),
                &coerce_summary(&record.summary, &record.memories),
                &record.memories,
                stamp,
                &record.session_ids,
            )?));
        };
        let request = vec![
            Message::system(self.inner.consolidation_prompt.clone()),
            Message::user(serde_json::to_string(record)?),
        ];
        let response = match client.get_response(request, ChatOptions::default()).await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(topic = %record.topic, error = %e, "skipping memory consolidation");
                return Ok(None);
            }
        };
        let text = response.text();
        let text = text.trim();
        if text.is_empty() {
            tracing::warn!(topic = %record.topic, "skipping consolidation: empty response");
            return Ok(None);
        }
        let payload: Value = match serde_json::from_str(extract_json_text(text)) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(topic = %record.topic, error = %e, preview = %preview(text), "skipping consolidation: invalid JSON");
                return Ok(None);
            }
        };
        let (Some(summary), Some(memories)) = (
            payload.get("summary").and_then(Value::as_str),
            payload.get("memories").and_then(Value::as_array),
        ) else {
            tracing::warn!(topic = %record.topic, preview = %preview(text), "skipping consolidation: missing 'summary' or 'memories'");
            return Ok(None);
        };
        let memories: Vec<String> = memories
            .iter()
            .filter_map(|m| m.as_str().map(str::to_string))
            .collect();
        Ok(Some(MemoryTopicRecord::new(
            &record.topic,
            Some(&record.slug),
            summary,
            &memories,
            stamp,
            &record.session_ids,
        )?))
    }

    async fn extract_memories(
        &self,
        session: &AgentSession,
        request: &[Message],
        response: &[Message],
        now: i64,
    ) -> Result<Vec<MemoryTopicRecord>> {
        let Some(client) = self.inner.chat_client.as_ref() else {
            return Ok(Vec::new());
        };
        if self.inner.max_extractions == 0 || response.is_empty() {
            return Ok(Vec::new());
        }
        let delta = format_for_model(&[request, response].concat());
        if delta.is_empty() {
            return Ok(Vec::new());
        }
        let reply = match client
            .get_response(
                vec![
                    Message::system(self.inner.extraction_prompt.clone()),
                    Message::user(delta),
                ],
                ChatOptions::default(),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = %e, "skipping memory extraction: extractor call failed");
                return Ok(Vec::new());
            }
        };
        let text = reply.text();
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let payload: Value = match serde_json::from_str(extract_json_text(text)) {
            Ok(v) => v,
            Err(e) => {
                tracing::warn!(error = %e, "skipping memory extraction: invalid JSON");
                return Ok(Vec::new());
            }
        };
        let items = match &payload {
            Value::Object(map) => map.get("memories"),
            other => Some(other),
        };
        let Some(Value::Array(items)) = items else {
            tracing::warn!(preview = %preview(text), "skipping memory extraction: 'memories' is not a list");
            return Ok(Vec::new());
        };
        let mut updated = Vec::new();
        for item in items.iter().take(self.inner.max_extractions) {
            let (Some(topic), Some(memory)) = (
                item.get("topic").and_then(Value::as_str),
                item.get("memory").and_then(Value::as_str),
            ) else {
                tracing::warn!(item = %preview(&item.to_string()), "skipping memory item without 'topic' and 'memory' strings");
                continue;
            };
            match self.merge_memory_at(session, topic, memory, now).await {
                Ok(record) => updated.push(record),
                Err(Error::Tool(_)) => continue, // An empty topic or memory.
                Err(e) => return Err(e),
            }
        }
        Ok(updated)
    }

    /// The six memory tools, bound to `session`.
    pub fn tools(&self, session: &AgentSession) -> Vec<ToolDefinition> {
        let (p, s) = (self.clone(), session.clone());
        let list = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {}
            let (p, s) = (p.clone(), s.clone());
            FunctionTool::typed(
                "list_memory_topics",
                "List the current topic pointers recorded in MEMORY.md.",
                move |_: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        Ok(Value::String(serde_json::to_string(
                            &p.rebuild_index(&s).await?,
                        )?))
                    }
                },
            )
            .into_definition()
        };
        let read = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {
                /// Topic name or slug.
                topic: String,
            }
            let (p, s) = (p.clone(), s.clone());
            FunctionTool::typed(
                "read_memory_topic",
                "Read one topic memory file by topic name or slug.",
                move |args: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        let topic = normalize_topic(&args.topic)?;
                        match p
                            .inner
                            .store
                            .get_topic(&s, &p.inner.source_id, &topic)
                            .await?
                        {
                            Some(record) => Ok(Value::String(record.to_markdown())),
                            None => Err(Error::tool(format!(
                                "No memory topic named '{topic}' was found for this owner."
                            ))),
                        }
                    }
                },
            )
            .into_definition()
        };
        let write = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {
                /// Topic name.
                topic: String,
                /// One durable memory line.
                memory: String,
            }
            let (p, s) = (p.clone(), s.clone());
            FunctionTool::typed(
                "write_memory",
                "Add one durable memory line to a topic file.",
                move |args: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        let record = p.merge_memory(&s, &args.topic, &args.memory).await?;
                        p.rebuild_index(&s).await?;
                        Ok(Value::String(serde_json::to_string(&record)?))
                    }
                },
            )
            .into_definition()
        };
        let delete = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {
                /// Topic name or slug.
                topic: String,
            }
            let (p, s) = (p.clone(), s.clone());
            FunctionTool::typed(
                "delete_memory_topic",
                "Delete one topic memory file by topic name or slug.",
                move |args: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        let topic = normalize_topic(&args.topic)?;
                        let lock = p.lock(&s, "topic", &slugify_topic(&topic)?);
                        let _guard = lock.lock().await;
                        if !p
                            .inner
                            .store
                            .delete_topic(&s, &p.inner.source_id, &topic)
                            .await?
                        {
                            return Err(Error::tool(format!(
                                "No memory topic named '{topic}' was found for this owner."
                            )));
                        }
                        p.rebuild_index(&s).await?;
                        Ok(Value::String(format!("Deleted memory topic '{topic}'.")))
                    }
                },
            )
            .into_definition()
        };
        let search = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {
                /// Text to look for, ignoring case.
                query: String,
                /// Only search this session's transcript.
                #[serde(default)]
                session_id: Option<String>,
                /// The maximum number of results (default 20).
                #[serde(default)]
                limit: Option<usize>,
            }
            let (p, s) = (p.clone(), s.clone());
            FunctionTool::typed(
                "search_memory_transcripts",
                "Search the raw transcript archive for matching text snippets.",
                move |args: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        let results = p
                            .inner
                            .store
                            .search_transcripts(
                                &s,
                                &p.inner.source_id,
                                &args.query,
                                args.session_id.as_deref(),
                                args.limit.unwrap_or(20),
                            )
                            .await?;
                        Ok(Value::String(serde_json::to_string(&results)?))
                    }
                },
            )
            .into_definition()
        };
        let consolidate = {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {}
            FunctionTool::typed(
                "consolidate_memories",
                "Force an immediate consolidation pass across all topic files.",
                move |_: Args| {
                    let (p, s) = (p.clone(), s.clone());
                    async move {
                        let count = p.consolidate(&s, true).await?;
                        Ok(Value::String(
                            json!({"consolidated_topics": count}).to_string(),
                        ))
                    }
                },
            )
            .into_definition()
        };
        vec![list, read, write, delete, search, consolidate]
    }

    /// The after-run work for a successful run in `session`: transcript,
    /// maintenance state, extraction and consolidation.
    async fn record_run(
        &self,
        session: &AgentSession,
        request: &[Message],
        response: &[Message],
    ) -> Result<()> {
        let store = &self.inner.store;
        let source = &self.inner.source_id;
        let to_store: Vec<Message> = request
            .iter()
            .chain(response)
            .cloned()
            .filter_map(|m| match &self.inner.history_message_filter {
                Some(filter) => filter(m),
                None => Some(m),
            })
            .collect();
        if !to_store.is_empty() {
            store.append_transcript(session, source, &to_store).await?;
        }
        let now = now_secs();
        {
            let lock = self.lock(session, "state", "maintenance");
            let _guard = lock.lock().await;
            let mut state = store.read_state(session, source).await?;
            let sid = session.session_id().to_string();
            if !state.sessions_since_consolidation.contains(&sid) {
                state.sessions_since_consolidation.push(sid);
            }
            store.write_state(session, &state, source).await?;
        }
        if !self
            .extract_memories(session, request, response, now)
            .await?
            .is_empty()
        {
            self.rebuild_index(session).await?;
        }
        self.run_consolidation(session, false, now).await?;
        Ok(())
    }
}

#[async_trait]
impl ContextProvider for MemoryContextProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = ctx.session.clone().ok_or_else(|| {
            Error::Configuration("MemoryContextProvider requires a session.".into())
        })?;
        let store = &self.inner.store;
        let source = &self.inner.source_id;
        let entries = self.rebuild_index(&session).await?;
        let index_text = store.index_text(&session, source).await?;
        let recent = if self.inner.recent_turns > 0 {
            select_recent_turns(
                &store.load_transcript(&session, source).await?,
                self.inner.recent_turns,
                self.inner.load_tool_turns,
            )
        } else {
            Vec::new()
        };
        let mut loaded = Vec::new();
        for entry in self.select_topics(&entries, &ctx.input_messages) {
            if let Some(record) = store.get_topic(&session, source, &entry.slug).await? {
                loaded.push(format!(
                    "### topics/{}.md\n{}",
                    record.slug,
                    record.to_markdown()
                ));
            }
        }
        let loaded = if loaded.is_empty() {
            "- none auto-loaded for this turn".to_string()
        } else {
            loaded.join("\n\n")
        };
        ctx.tools.extend(self.tools(&session));
        ctx.add_instructions(MEMORY_INSTRUCTIONS.join("\n"));
        ctx.messages.extend(recent);
        ctx.messages.push(Message::user(format!(
            "{}\n\n### MEMORY.md\n{index_text}\n\n### Auto-loaded topic files\n{loaded}",
            self.inner.context_prompt
        )));
        Ok(())
    }

    async fn after_run_in_session(
        &self,
        session: &AgentSession,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        self.record_run(session, request_messages, response_messages)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_and_pointer_lines() {
        assert_eq!(
            slugify_topic("  User Preferences!! ").unwrap(),
            "user-preferences"
        );
        assert_eq!(slugify_topic("日本").unwrap(), "memory-topic");
        assert!(slugify_topic("   ").is_err());
        let entry = MemoryIndexEntry {
            topic: "t".into(),
            slug: "t".into(),
            summary: "x".repeat(200),
            updated_at: String::new(),
        };
        let line = entry.to_pointer_line(20);
        assert_eq!(line.chars().count(), 20);
        assert!(line.ends_with("..."));
    }

    #[test]
    fn markdown_round_trips_and_escapes_headings() {
        let record = MemoryTopicRecord::new(
            "Prefs",
            None,
            "# not a heading",
            &["likes tea".into(), "Likes Tea".into(), "#hashtag".into()],
            "2024-01-02T03:04:05+00:00",
            &["s1".into(), "s1".into(), "s2".into()],
        )
        .unwrap();
        assert_eq!(record.memories, ["likes tea", "#hashtag"]);
        assert_eq!(record.session_ids, ["s1", "s2"]);
        let md = record.to_markdown();
        assert!(md.contains("\n\\# not a heading\n"), "{md}");
        let back = MemoryTopicRecord::from_markdown(&md, None).unwrap();
        assert_eq!(back, record);
    }

    #[test]
    fn timestamps_round_trip() {
        let t = 1_704_164_645; // 2024-01-02T03:04:05Z
        assert_eq!(format_memory_timestamp(t), "2024-01-02T03:04:05+00:00");
        assert_eq!(parse_memory_timestamp("2024-01-02T03:04:05+00:00"), Some(t));
        assert_eq!(
            parse_memory_timestamp("2024-01-02T05:04:05.123+02:00"),
            Some(t)
        );
        assert_eq!(parse_memory_timestamp("2024-01-02T03:04:05Z"), Some(t));
        assert_eq!(parse_memory_timestamp("garbage"), None);
    }

    #[test]
    fn recent_turns_group_tool_calls() {
        use crate::types::FunctionCallContent;
        let call = Message {
            role: Role::assistant(),
            contents: vec![Content::FunctionCall(FunctionCallContent {
                call_id: "c".into(),
                name: "f".into(),
                arguments: None,
                protected_data: None,
                id: None,
            })],
            author_name: None,
            message_id: None,
            additional_properties: Default::default(),
        };
        let messages = vec![
            Message::user("one"),
            Message::assistant("a1"),
            Message::user("two"),
            call,
            Message::new(Role::tool(), "result"),
            Message::assistant("a2"),
        ];
        let last = select_recent_turns(&messages, 1, true);
        assert_eq!(last.len(), 4);
        let no_tools = select_recent_turns(&messages, 1, false);
        let texts: Vec<String> = no_tools.iter().map(Message::text).collect();
        assert_eq!(texts, ["two", "a2"]);
        assert_eq!(select_recent_turns(&messages, 5, true).len(), 6);
    }

    #[test]
    fn fenced_json_is_unwrapped() {
        assert_eq!(extract_json_text("```json\n{\"a\":1}\n```"), "{\"a\":1}");
        assert_eq!(extract_json_text(" {\"a\":1} "), "{\"a\":1}");
    }
}
