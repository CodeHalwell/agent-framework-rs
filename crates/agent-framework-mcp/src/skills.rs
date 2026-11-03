//! Agent Skills served over MCP (SEP-2640), the counterpart of upstream's
//! `MCPSkillsSource`.
//!
//! Discovery reads the well-known `skill://index.json` resource and
//! dispatches each entry by its `type` (case-insensitive):
//!
//! - `skill-md` (`name`, `description`, `url`) becomes a
//!   [`Skill`] whose instructions are
//!   the body of the `SKILL.md` resource at `url`.
//! - `archive`: `url` is a ZIP resource, downloaded and unpacked **in
//!   memory** (nothing touches disk). Its shallowest `SKILL.md` drives the
//!   skill (its frontmatter `name` must match the entry's), and the text files
//!   under that file's directory matching [`ArchiveOptions::resource_extensions`]
//!   within [`ArchiveOptions::resource_search_depth`] become the skill's
//!   resources. Bundled scripts are never runnable (at most a readable
//!   resource). A non-null `digest` must be `sha256:` plus 64 lowercase hex
//!   characters matching the downloaded bytes. Extraction is bounded by
//!   download size, file count and total uncompressed size; a member path
//!   with a `..` segment ("zip slip") rejects the whole archive; members whose
//!   paths collide case-insensitively keep the first. A later archive whose
//!   skill name repeats an earlier archive's is skipped.
//!
//! Pass the skills to a
//! [`SkillsProvider`](agent_framework_core::skills::SkillsProvider).
//!
//! # Divergences
//!
//! - `SKILL.md` bodies are read at discovery rather than on first
//!   `load_skill`: the Rust [`Skill`]
//!   carries its instructions. Resources referenced inside a `SKILL.md` are
//!   not prefetched.
//! - Entry types with no handler are skipped (as upstream).
//! - An absent, empty or malformed index yields no skills, as upstream; any
//!   other failure to read it is an error (so a cache keeps its previous
//!   skills rather than replacing them with none). An unreadable or invalid
//!   `SKILL.md` skips that skill.
//! - An archive whose resource is missing (MCP "not found" / "method not
//!   found" / "invalid params") or unusable is skipped; any other read error
//!   propagates, as upstream, so a transient failure is not mistaken for a
//!   missing skill.

use std::collections::HashMap;
use std::io::Read;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::skills::Skill;
use agent_framework_core::skills_files::{
    parse_skill_md, DEFAULT_RESOURCE_EXTENSIONS, DEFAULT_SEARCH_DEPTH,
};
use serde::Deserialize;

use crate::McpClient;

/// The well-known discovery resource.
pub const SKILLS_INDEX_URI: &str = "skill://index.json";

/// The default [`ArchiveOptions::max_file_count`].
pub const DEFAULT_ARCHIVE_MAX_FILE_COUNT: usize = 20;
/// The default [`ArchiveOptions::max_size_bytes`] (1 MiB).
pub const DEFAULT_ARCHIVE_MAX_SIZE_BYTES: usize = 1024 * 1024;
/// The default [`ArchiveOptions::max_uncompressed_size_bytes`] (1 MiB).
pub const DEFAULT_ARCHIVE_MAX_UNCOMPRESSED_SIZE_BYTES: usize = 1024 * 1024;

/// One `skill://index.json` entry.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillIndexEntry {
    /// The skill name.
    #[serde(default)]
    pub name: Option<String>,
    /// The entry type (`skill-md`, `archive`, ...).
    #[serde(default, rename = "type")]
    pub kind: Option<String>,
    /// A short description.
    #[serde(default)]
    pub description: Option<String>,
    /// The resource URI of the entry's content.
    #[serde(default)]
    pub url: Option<String>,
}

/// An index entry plus the fields only archives use.
#[derive(Debug, Deserialize)]
struct IndexEntry {
    #[serde(flatten)]
    entry: SkillIndexEntry,
    /// An archive's `sha256:<hex>` digest; any non-string value is invalid.
    #[serde(default)]
    digest: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SkillIndex {
    #[serde(default)]
    skills: Vec<IndexEntry>,
}

/// How `archive` entries are unpacked (upstream's `archive_*` options).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchiveOptions {
    /// Extensions (with the dot, case-insensitive) of the files that become
    /// resources. Default
    /// [`DEFAULT_RESOURCE_EXTENSIONS`].
    pub resource_extensions: Vec<String>,
    /// How deep under the `SKILL.md` directory resources are found (that
    /// directory is depth 1). Default
    /// [`DEFAULT_SEARCH_DEPTH`].
    pub resource_search_depth: usize,
    /// The most files one archive may hold.
    pub max_file_count: usize,
    /// The largest archive download, in bytes.
    pub max_size_bytes: usize,
    /// The most bytes one archive may decompress to, counted as produced.
    pub max_uncompressed_size_bytes: usize,
}

impl Default for ArchiveOptions {
    fn default() -> Self {
        Self {
            resource_extensions: DEFAULT_RESOURCE_EXTENSIONS
                .iter()
                .map(|e| e.to_string())
                .collect(),
            resource_search_depth: DEFAULT_SEARCH_DEPTH,
            max_file_count: DEFAULT_ARCHIVE_MAX_FILE_COUNT,
            max_size_bytes: DEFAULT_ARCHIVE_MAX_SIZE_BYTES,
            max_uncompressed_size_bytes: DEFAULT_ARCHIVE_MAX_UNCOMPRESSED_SIZE_BYTES,
        }
    }
}

async fn read_text(client: &McpClient, uri: &str) -> Option<String> {
    let result = client.read_resource(uri).await.ok()?;
    result.contents.into_iter().find_map(|c| c.text)
}

/// Whether `error` is an MCP "resource not found" (`-32002`), "method not
/// found" (`-32601`) or "invalid params" (`-32602`) error, which mean the
/// resource is not available rather than that the read failed.
fn is_not_found(error: &Error) -> bool {
    let text = error.to_string();
    ["-32002", "-32601", "-32602"]
        .iter()
        .any(|code| text.contains(&format!("MCP error {code}:")))
}

/// Discover the skills `client`'s server serves, with default
/// [`ArchiveOptions`]. See the [module docs](self).
pub async fn load_skills(client: &McpClient) -> Result<Vec<Skill>> {
    load_skills_with(client, &ArchiveOptions::default()).await
}

/// [`load_skills`] unpacking `archive` entries with `options`.
pub async fn load_skills_with(client: &McpClient, options: &ArchiveOptions) -> Result<Vec<Skill>> {
    let text = match client.read_resource(SKILLS_INDEX_URI).await {
        Ok(result) => result.contents.into_iter().find_map(|c| c.text),
        Err(e) if is_not_found(&e) => None,
        Err(e) => return Err(e),
    };
    let Some(text) = text else {
        return Ok(Vec::new());
    };
    let Ok(index) = serde_json::from_str::<SkillIndex>(&text) else {
        return Ok(Vec::new());
    };
    let kind_of = |e: &IndexEntry| {
        e.entry
            .kind
            .as_deref()
            .map(|k| k.trim().to_ascii_lowercase())
            .unwrap_or_default()
    };
    let mut skills = Vec::new();
    let mut archives = Vec::new();
    for entry in index.skills {
        match kind_of(&entry).as_str() {
            "skill-md" => {
                if let Some(skill) = skill_md(client, entry.entry).await {
                    skills.push(skill);
                }
            }
            "archive" => archives.push(entry),
            _ => {}
        }
    }
    let mut archive_names = std::collections::HashSet::new();
    for entry in archives {
        let Some(skill) = archive_skill(client, &entry, options).await? else {
            continue;
        };
        if !archive_names.insert(skill.name.clone()) {
            tracing::warn!(skill = %skill.name, "duplicate archive skill name: later entry skipped");
            continue;
        }
        skills.push(skill);
    }
    Ok(skills)
}

async fn skill_md(client: &McpClient, entry: SkillIndexEntry) -> Option<Skill> {
    let content = read_text(client, entry.url.as_deref()?).await?;
    let (frontmatter, body) = parse_skill_md(&content).ok()?;
    let description = entry
        .description
        .filter(|d| !d.is_empty())
        .unwrap_or(frontmatter.description);
    let mut skill = Skill::new(frontmatter.name, description).with_instructions(body);
    skill.license = frontmatter.license;
    skill.compatibility = frontmatter.compatibility;
    skill.allowed_tools = frontmatter.allowed_tools;
    skill.metadata = frontmatter.metadata.unwrap_or_default();
    Some(skill)
}

/// Download and unpack one `archive` entry: `Ok(None)` skips it.
async fn archive_skill(
    client: &McpClient,
    entry: &IndexEntry,
    options: &ArchiveOptions,
) -> Result<Option<Skill>> {
    let name = match entry.entry.name.as_deref().map(str::trim) {
        Some(name) if !name.is_empty() => name,
        _ => return Ok(None),
    };
    let url = match entry.entry.url.as_deref().map(str::trim) {
        Some(url) if !url.is_empty() => url,
        _ => return Ok(None),
    };
    let result = match client.read_resource(url).await {
        Ok(result) => result,
        Err(e) if is_not_found(&e) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some((data, mime_type)) = first_blob(result) else {
        tracing::warn!(skill = name, "archive resource returned no binary content");
        return Ok(None);
    };
    if data.is_empty() || data.len() > options.max_size_bytes {
        tracing::warn!(skill = name, "archive resource is empty or too large");
        return Ok(None);
    }
    if let Some(digest) = &entry.digest {
        if !digest_matches(digest, &data) {
            tracing::warn!(skill = name, "archive digest is invalid or does not match");
            return Ok(None);
        }
    }
    match skill_from_archive(name, url, &data, mime_type.as_deref(), options) {
        Ok(skill) => Ok(Some(skill)),
        Err(reason) => {
            tracing::warn!(skill = name, %reason, "skipping archive skill");
            Ok(None)
        }
    }
}

/// The first binary content of a read, base64-decoded (a `data:` prefix is
/// tolerated), with its MIME type.
fn first_blob(result: crate::protocol::ReadResourceResult) -> Option<(Vec<u8>, Option<String>)> {
    use base64::Engine as _;
    let content = result.contents.into_iter().find(|c| c.blob.is_some())?;
    let blob = content.blob?;
    let encoded = match blob.strip_prefix("data:") {
        Some(rest) => rest.split_once(',').map_or(rest, |(_, data)| data),
        None => blob.as_str(),
    };
    let data = base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .ok()?;
    Some((data, content.mime_type))
}

fn digest_matches(digest: &serde_json::Value, data: &[u8]) -> bool {
    use sha2::Digest as _;
    let Some(hex) = digest.as_str().and_then(|d| d.strip_prefix("sha256:")) else {
        return false;
    };
    if hex.len() != 64 || !hex.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')) {
        return false;
    }
    let actual: String = sha2::Sha256::digest(data)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    actual == hex
}

/// Whether `data` is a ZIP, by signature, then MIME type, then URL suffix
/// (a gzip signature is never one).
fn is_zip(data: &[u8], mime_type: Option<&str>, url: &str) -> bool {
    if data.starts_with(&[0x1f, 0x8b]) {
        return false;
    }
    if data.len() >= 4 && data[..2] == [0x50, 0x4b] && matches!(data[2], 0x03 | 0x05 | 0x07) {
        return true;
    }
    let media = mime_type.unwrap_or_default().trim().to_ascii_lowercase();
    if matches!(
        media.as_str(),
        "application/zip" | "application/x-zip-compressed"
    ) {
        return true;
    }
    url.to_ascii_lowercase().ends_with(".zip")
}

/// A member path as a clean relative path; `None` for a degenerate entry;
/// an error for a `..` escape.
fn normalize_member(path: &str) -> std::result::Result<Option<String>, String> {
    if path.trim().is_empty() {
        return Ok(None);
    }
    let normalized = path.replace('\\', "/");
    let normalized = normalized.trim_start_matches('/');
    if normalized.split('/').any(|s| s == "..") {
        return Err(format!(
            "archive member '{path}' attempts to escape the skill namespace"
        ));
    }
    let parts: Vec<&str> = normalized
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    Ok((!parts.is_empty()).then(|| parts.join("/")))
}

/// Read a ZIP's regular files into memory within the options' limits.
fn extract(
    data: &[u8],
    options: &ArchiveOptions,
) -> std::result::Result<Vec<(String, Vec<u8>)>, String> {
    let mut archive =
        zip::ZipArchive::new(std::io::Cursor::new(data)).map_err(|e| e.to_string())?;
    let mut remaining = options.max_uncompressed_size_bytes;
    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut count = 0;
    for i in 0..archive.len() {
        let mut member = archive.by_index(i).map_err(|e| e.to_string())?;
        if member.is_dir() {
            continue;
        }
        count += 1;
        if count > options.max_file_count {
            return Err(format!(
                "archive exceeds the maximum file count ({})",
                options.max_file_count
            ));
        }
        let raw = member.name().map_err(|e| e.to_string())?.into_owned();
        let Some(name) = normalize_member(&raw)? else {
            continue;
        };
        if !seen.insert(name.to_lowercase()) {
            tracing::warn!(member = %name, "skipping a duplicate archive member");
            continue;
        }
        // Count the bytes the decompressor produces, not the header's claim.
        let mut content = Vec::new();
        let limit = remaining as u64 + 1;
        (&mut member)
            .take(limit)
            .read_to_end(&mut content)
            .map_err(|e| e.to_string())?;
        if content.len() > remaining {
            return Err("archive exceeds the maximum uncompressed size".into());
        }
        remaining -= content.len();
        files.push((name, content));
    }
    Ok(files)
}

/// Build the skill an archive holds. See the [module docs](self).
fn skill_from_archive(
    name: &str,
    url: &str,
    data: &[u8],
    mime_type: Option<&str>,
    options: &ArchiveOptions,
) -> std::result::Result<Skill, String> {
    if !is_zip(data, mime_type, url) {
        return Err(format!(
            "unsupported archive media type '{}'",
            mime_type.unwrap_or("(none)")
        ));
    }
    let files = extract(data, options)?;
    let skill_md = files
        .iter()
        .map(|(key, _)| key)
        .filter(|key| {
            key.rsplit('/')
                .next()
                .is_some_and(|b| b.eq_ignore_ascii_case("SKILL.md"))
        })
        .min_by_key(|key| (key.matches('/').count(), (*key).clone()))
        .cloned()
        .ok_or("archive contains no SKILL.md")?;
    let content = files
        .iter()
        .find(|(key, _)| *key == skill_md)
        .map(|(_, bytes)| std::str::from_utf8(bytes))
        .expect("found above")
        .map_err(|_| "SKILL.md is not valid UTF-8")?;
    let (frontmatter, body) = parse_skill_md(content).map_err(|e| e.to_string())?;
    if frontmatter.name != name {
        return Err(format!(
            "SKILL.md frontmatter name '{}' does not match the entry name",
            frontmatter.name
        ));
    }
    let root = skill_md.rfind('/').map_or("", |i| &skill_md[..=i]);
    let allowed: Vec<String> = options
        .resource_extensions
        .iter()
        .map(|e| e.to_lowercase())
        .collect();
    let mut resources = HashMap::new();
    for (key, bytes) in &files {
        if *key == skill_md {
            continue;
        }
        let Some(rel) = key.strip_prefix(root).filter(|r| !r.is_empty()) else {
            continue;
        };
        if rel.matches('/').count() + 1 > options.resource_search_depth {
            continue;
        }
        let base = rel.rsplit('/').next().unwrap_or(rel);
        let suffix = match base.rfind('.') {
            Some(dot) if dot > 0 => base[dot..].to_lowercase(),
            _ => continue,
        };
        if !allowed.contains(&suffix) {
            continue;
        }
        if let Ok(text) = std::str::from_utf8(bytes) {
            resources.insert(rel.to_string(), text.to_string());
        }
    }
    let mut skill = Skill::new(frontmatter.name, frontmatter.description).with_instructions(body);
    skill.resources = resources;
    skill.license = frontmatter.license;
    skill.compatibility = frontmatter.compatibility;
    skill.allowed_tools = frontmatter.allowed_tools;
    skill.metadata = frontmatter.metadata.unwrap_or_default();
    Ok(skill)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for (name, content) in files {
            writer.start_file(*name, options).unwrap();
            writer.write_all(content).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    const SKILL: &[u8] = b"---\nname: packed\ndescription: Packed skill.\n---\nUse the table.";

    fn build(data: &[u8], options: &ArchiveOptions) -> std::result::Result<Skill, String> {
        skill_from_archive("packed", "skill://packed.zip", data, None, options)
    }

    #[test]
    fn an_archive_becomes_a_skill_with_its_text_resources() {
        let data = zip_of(&[
            ("packed/SKILL.md", SKILL),
            ("packed/table.csv", b"a,b"),
            ("packed/refs/notes.md", b"notes"),
            ("packed/refs/deep/too-deep.md", b"x"),
            ("packed/run.py", b"print(1)"),
            ("packed/blob.bin", b"\x00"),
            ("other/outside.md", b"outside"),
        ]);
        let skill = build(&data, &ArchiveOptions::default()).unwrap();
        assert_eq!(skill.name, "packed");
        assert_eq!(skill.description, "Packed skill.");
        assert_eq!(skill.instructions.trim(), "Use the table.");
        let mut names: Vec<&String> = skill.resources.keys().collect();
        names.sort();
        assert_eq!(names, ["refs/notes.md", "table.csv"]);
        assert!(skill.scripts.is_empty(), "bundled scripts never run");
        let deeper = ArchiveOptions {
            resource_search_depth: 3,
            ..Default::default()
        };
        assert!(build(&data, &deeper)
            .unwrap()
            .resources
            .contains_key("refs/deep/too-deep.md"));
    }

    #[test]
    fn unsafe_or_unusable_archives_are_rejected() {
        let defaults = ArchiveOptions::default();
        // Zip slip rejects the whole archive.
        let slip = zip_of(&[("SKILL.md", SKILL), ("../evil.md", b"x")]);
        assert!(build(&slip, &defaults).unwrap_err().contains("escape"));
        // Too many files.
        let many = zip_of(&[("SKILL.md", SKILL), ("a.md", b"a"), ("b.md", b"b")]);
        let two = ArchiveOptions {
            max_file_count: 2,
            ..Default::default()
        };
        assert!(build(&many, &two).unwrap_err().contains("file count"));
        // A decompression bomb, by produced bytes.
        let bomb = zip_of(&[("SKILL.md", SKILL), ("big.txt", &[b'a'; 4096])]);
        let small = ArchiveOptions {
            max_uncompressed_size_bytes: 1024,
            ..Default::default()
        };
        assert!(build(&bomb, &small).unwrap_err().contains("uncompressed"));
        // No SKILL.md; a mismatched name; gzip; garbage.
        assert!(build(&zip_of(&[("a.md", b"a")]), &defaults).is_err());
        let other = zip_of(&[("SKILL.md", b"---\nname: other\ndescription: d\n---\nx")]);
        assert!(build(&other, &defaults)
            .unwrap_err()
            .contains("does not match"));
        assert!(build(&[0x1f, 0x8b, 0, 0], &defaults).is_err());
        assert!(build(b"not a zip", &defaults).is_err());
    }

    #[test]
    fn the_shallowest_skill_md_wins_and_case_collisions_keep_the_first() {
        let data = zip_of(&[
            (
                "nested/deeper/SKILL.md",
                b"---\nname: wrong\ndescription: d\n---\nx",
            ),
            ("nested/skill.md", SKILL),
            ("nested/Data.txt", b"first"),
            ("nested/data.TXT", b"second"),
        ]);
        let skill = build(&data, &ArchiveOptions::default()).unwrap();
        assert_eq!(skill.name, "packed");
        assert_eq!(skill.resources["Data.txt"], "first");
        assert!(!skill.resources.contains_key("data.TXT"));
    }

    #[test]
    fn digests_are_checked_strictly() {
        use sha2::Digest as _;
        let data = b"payload";
        let hex: String = sha2::Sha256::digest(data)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert!(digest_matches(&format!("sha256:{hex}").into(), data));
        assert!(!digest_matches(
            &format!("sha256:{}", hex.to_uppercase()).into(),
            data
        ));
        assert!(!digest_matches(&format!("sha512:{hex}").into(), data));
        assert!(!digest_matches(&serde_json::json!(5), data));
        assert!(!digest_matches(
            &format!("sha256:{}", "0".repeat(64)).into(),
            data
        ));
    }

    #[test]
    fn mcp_not_found_errors_are_recognized() {
        assert!(is_not_found(&Error::service("MCP error -32002: not found")));
        assert!(is_not_found(&Error::service(
            "MCP error -32601: Method not found"
        )));
        assert!(!is_not_found(&Error::service("MCP error -32603: internal")));
        assert!(!is_not_found(&Error::service("connection reset")));
    }
}
