//! File-based skills: [`Skill`]s discovered from `SKILL.md` packages on
//! disk, per the [Agent Skills specification](https://agentskills.io/specification).
//!
//! Port of upstream's `FileSkillsSource` / `FileSkill` / `SkillFrontmatter`
//! (`_skills.py`), behind the `file-skills` feature.
//!
//! A skill is a directory holding a `SKILL.md`: YAML frontmatter (`name`,
//! `description`, and optionally `license`, `compatibility`,
//! `allowed-tools`, `metadata`) followed by the instructions. Each
//! configured path is either a skill directory or a parent of skill
//! directories, searched two levels deep; once a `SKILL.md` is found,
//! everything beneath it belongs to that skill. Files in the skill directory
//! with a resource extension (`.md`, `.json`, `.yaml`, `.yml`, `.csv`,
//! `.xml`, `.txt` by default), down to `search_depth` levels, become its
//! resources, named by their `/`-separated path relative to the skill.
//!
//! ```no_run
//! use agent_framework_core::skills::SkillsProvider;
//!
//! # fn demo() -> agent_framework_core::error::Result<()> {
//! let skills = SkillsProvider::from_paths(["./skills"])?;
//! # let _ = skills;
//! # Ok(())
//! # }
//! ```
//!
//! # Security
//!
//! Discovery fails closed on links: a symbolic link below a configured root
//! is never adopted as a skill or read as a resource, so a skill cannot pull
//! in files from outside its directory. Only load skills from trusted
//! sources; their instructions reach the model verbatim.
//!
//! # Divergences
//!
//! - Resources are read when the skills are loaded rather than on each
//!   `read_skill_resource` call; reload to pick up changes. Files over 1 MiB
//!   or not valid UTF-8 are skipped.
//! - Script files (`.py` by default, [`FileSkillsSource::with_script_extensions`])
//!   become [`SkillScript::file`]s named by their relative path, run through
//!   `run_skill_script` by the runner given to
//!   [`SkillsProvider::with_script_runner`]; there is no default runner.
//! - Frontmatter is parsed with `serde_yaml`, which rejects duplicate keys
//!   anywhere, including inside `metadata` (upstream keeps the first and
//!   warns there).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_yaml::Value as Yaml;

use crate::error::{Error, Result};
use crate::skills::{available_scripts_block, Skill, SkillScript, SkillsProvider};

/// The skill file name.
pub const SKILL_FILE_NAME: &str = "SKILL.md";
/// How deep skill directories are searched for below a configured path.
pub const MAX_SEARCH_DEPTH: usize = 2;
/// The longest valid skill name.
pub const MAX_NAME_LENGTH: usize = 64;
/// The longest valid description.
pub const MAX_DESCRIPTION_LENGTH: usize = 1024;
/// The longest valid compatibility note.
pub const MAX_COMPATIBILITY_LENGTH: usize = 500;
/// The default resource extensions.
pub const DEFAULT_RESOURCE_EXTENSIONS: &[&str] =
    &[".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".txt"];
/// The default script extensions, as upstream.
pub const DEFAULT_SCRIPT_EXTENSIONS: &[&str] = &[".py"];
/// The default resource search depth (the root plus one level).
pub const DEFAULT_SEARCH_DEPTH: usize = 2;
const MAX_RESOURCE_BYTES: u64 = 1024 * 1024;

/// A skill's `SKILL.md` frontmatter.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SkillFrontmatter {
    /// The name: lowercase letters, digits and single inner hyphens, at most
    /// 64 characters.
    pub name: String,
    /// What the skill does, at most 1024 characters.
    pub description: String,
    /// A license name or reference.
    pub license: Option<String>,
    /// Compatibility notes, at most 500 characters.
    pub compatibility: Option<String>,
    /// Space-delimited pre-approved tool names.
    pub allowed_tools: Option<String>,
    /// Free-form string metadata.
    pub metadata: Option<BTreeMap<String, String>>,
}

/// Check a skill name against the specification.
pub fn validate_skill_name(name: &str) -> Result<()> {
    let valid = !name.is_empty()
        && name.len() <= MAX_NAME_LENGTH
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !name.starts_with('-')
        && !name.ends_with('-')
        && !name.contains("--");
    if name.trim().is_empty() {
        return Err(Error::Configuration("Skill name cannot be empty.".into()));
    }
    if !valid {
        return Err(Error::Configuration(format!(
            "Invalid skill name '{name}': Must be {MAX_NAME_LENGTH} characters or fewer, using only lowercase \
             letters, numbers, and hyphens, and must not start or end with a hyphen or contain consecutive hyphens."
        )));
    }
    Ok(())
}

fn scalar_text(value: &Yaml) -> Option<String> {
    match value {
        Yaml::String(s) => Some(s.clone()),
        Yaml::Number(n) => Some(n.to_string()),
        Yaml::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Split a `SKILL.md` into its validated frontmatter and its body.
pub fn parse_skill_md(content: &str) -> Result<(SkillFrontmatter, String)> {
    let text = content.strip_prefix('\u{feff}').unwrap_or(content);
    let invalid = || {
        Error::Configuration(
            "SKILL.md does not contain valid YAML frontmatter delimited by '---'.".into(),
        )
    };
    let rest = text
        .strip_prefix("---\r\n")
        .or_else(|| text.strip_prefix("---\n"))
        .ok_or_else(invalid)?;
    let (yaml, body) = {
        let mut offset = 0;
        let mut found = None;
        for line in rest.split_inclusive('\n') {
            if line.trim_end_matches(['\r', '\n']) == "---" {
                found = Some((&rest[..offset], &rest[offset + line.len()..]));
                break;
            }
            offset += line.len();
        }
        found.ok_or_else(invalid)?
    };
    let root: Yaml = serde_yaml::from_str(yaml).map_err(|e| {
        Error::Configuration(format!("SKILL.md contains invalid YAML frontmatter: {e}"))
    })?;
    let Yaml::Mapping(map) = root else {
        return Err(Error::Configuration(
            "SKILL.md must contain a YAML frontmatter mapping.".into(),
        ));
    };
    let mut fm = SkillFrontmatter::default();
    for (key, value) in &map {
        let Some(key) = key.as_str() else { continue };
        let text = || {
            scalar_text(value).ok_or_else(|| {
                Error::Configuration(format!(
                    "SKILL.md frontmatter field '{key}' must be a text value."
                ))
            })
        };
        match key {
            "name" => fm.name = text()?,
            "description" => fm.description = text()?,
            "license" if !value.is_null() => fm.license = Some(text()?),
            "compatibility" if !value.is_null() => fm.compatibility = Some(text()?),
            "allowed-tools" if !value.is_null() => fm.allowed_tools = Some(text()?),
            "metadata" => match value {
                Yaml::Null => {}
                Yaml::Mapping(m) => {
                    let mut out = BTreeMap::new();
                    for (k, v) in m {
                        match (scalar_text(k), scalar_text(v)) {
                            (Some(k), Some(v)) => {
                                out.insert(k, v);
                            }
                            _ => tracing::warn!("SKILL.md metadata entry is not text; skipping it"),
                        }
                    }
                    fm.metadata = Some(out);
                }
                _ => tracing::warn!("SKILL.md has invalid metadata; expected a mapping"),
            },
            _ => {}
        }
    }
    validate_skill_name(&fm.name)?;
    if fm.description.trim().is_empty() {
        return Err(Error::Configuration(
            "Skill description cannot be empty.".into(),
        ));
    }
    if fm.description.chars().count() > MAX_DESCRIPTION_LENGTH {
        return Err(Error::Configuration(format!(
            "Skill '{}' has an invalid description: Must be {MAX_DESCRIPTION_LENGTH} characters or fewer.",
            fm.name
        )));
    }
    if fm
        .compatibility
        .as_ref()
        .is_some_and(|c| c.chars().count() > MAX_COMPATIBILITY_LENGTH)
    {
        return Err(Error::Configuration(format!(
            "Skill compatibility must be {MAX_COMPATIBILITY_LENGTH} characters or fewer."
        )));
    }
    Ok((fm, body.to_string()))
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Filters discovered resources: `(skill name, relative path) -> keep`.
pub type ResourceFilter = std::sync::Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;

/// Discovers [`Skill`]s from `SKILL.md` packages. See the [module docs](self).
#[derive(Clone)]
pub struct FileSkillsSource {
    paths: Vec<PathBuf>,
    resource_extensions: Vec<String>,
    script_extensions: Vec<String>,
    search_depth: usize,
    resource_filter: Option<ResourceFilter>,
    script_filter: Option<ResourceFilter>,
}

impl std::fmt::Debug for FileSkillsSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSkillsSource")
            .field("paths", &self.paths)
            .field("resource_extensions", &self.resource_extensions)
            .field("search_depth", &self.search_depth)
            .finish_non_exhaustive()
    }
}

impl FileSkillsSource {
    /// A source over skill directories, or parents of them.
    pub fn new<P: AsRef<Path>>(paths: impl IntoIterator<Item = P>) -> Self {
        Self {
            paths: paths
                .into_iter()
                .map(|p| p.as_ref().to_path_buf())
                .collect(),
            resource_extensions: DEFAULT_RESOURCE_EXTENSIONS
                .iter()
                .map(|e| e.to_string())
                .collect(),
            script_extensions: DEFAULT_SCRIPT_EXTENSIONS
                .iter()
                .map(|e| e.to_string())
                .collect(),
            search_depth: DEFAULT_SEARCH_DEPTH,
            resource_filter: None,
            script_filter: None,
        }
    }

    /// The extensions (with the dot) that make a file a script. Empty
    /// discovers none.
    pub fn with_script_extensions<S: Into<String>>(
        mut self,
        extensions: impl IntoIterator<Item = S>,
    ) -> Self {
        self.script_extensions = extensions
            .into_iter()
            .map(|e| e.into().to_lowercase())
            .collect();
        self
    }

    /// Keep only the scripts `filter(skill_name, relative_path)` accepts.
    pub fn with_script_filter(
        mut self,
        filter: impl Fn(&str, &str) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.script_filter = Some(std::sync::Arc::new(filter));
        self
    }

    /// The extensions (with the dot) that make a file a resource. Empty
    /// discovers none.
    pub fn with_resource_extensions<S: Into<String>>(
        mut self,
        extensions: impl IntoIterator<Item = S>,
    ) -> Self {
        self.resource_extensions = extensions
            .into_iter()
            .map(|e| e.into().to_lowercase())
            .collect();
        self
    }

    /// How many directory levels of a skill to search for resources (at
    /// least 1, the skill root).
    pub fn with_search_depth(mut self, depth: usize) -> Result<Self> {
        if depth < 1 {
            return Err(Error::Configuration(format!(
                "search_depth must be >= 1, got {depth}"
            )));
        }
        self.search_depth = depth;
        Ok(self)
    }

    /// Keep only the resources `filter(skill_name, relative_path)` accepts.
    pub fn with_resource_filter(
        mut self,
        filter: impl Fn(&str, &str) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.resource_filter = Some(std::sync::Arc::new(filter));
        self
    }

    /// Every directory holding a `SKILL.md`, never through a link.
    fn skill_directories(&self) -> Vec<PathBuf> {
        fn search(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
            let skill_file = dir.join(SKILL_FILE_NAME);
            if let Ok(meta) = std::fs::symlink_metadata(&skill_file) {
                if meta.file_type().is_symlink() {
                    tracing::warn!(dir = %dir.display(), "skipping skill: SKILL.md is a symbolic link");
                } else if meta.is_file() {
                    out.push(dir.to_path_buf());
                }
                return;
            }
            if depth >= MAX_SEARCH_DEPTH {
                return;
            }
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            let mut children: Vec<PathBuf> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.file_type().is_ok_and(|t| t.is_dir() && !t.is_symlink()))
                .map(|e| e.path())
                .collect();
            children.sort();
            for child in children {
                search(&child, depth + 1, out);
            }
        }
        let mut out = Vec::new();
        for root in &self.paths {
            search(root, 0, &mut out);
        }
        out
    }

    /// `(relative name, path)` of each resource file of a skill.
    fn resource_files(&self, skill_dir: &Path, skill_name: &str) -> Vec<(String, PathBuf)> {
        self.files(
            skill_dir,
            skill_name,
            &self.resource_extensions,
            self.resource_filter.as_ref(),
        )
    }

    /// `(relative name, path)` of each script file of a skill.
    fn script_files(&self, skill_dir: &Path, skill_name: &str) -> Vec<(String, PathBuf)> {
        self.files(
            skill_dir,
            skill_name,
            &self.script_extensions,
            self.script_filter.as_ref(),
        )
    }

    /// `(relative name, path)` of each file of a skill with one of
    /// `extensions` that `filter` keeps, never through a link.
    fn files(
        &self,
        skill_dir: &Path,
        skill_name: &str,
        extensions: &[String],
        filter: Option<&ResourceFilter>,
    ) -> Vec<(String, PathBuf)> {
        let mut out = Vec::new();
        let mut pending = vec![(skill_dir.to_path_buf(), 1usize)];
        while let Some((dir, depth)) = pending.pop() {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.filter_map(|e| e.ok()) {
                let Ok(kind) = entry.file_type() else {
                    continue;
                };
                if kind.is_symlink() {
                    continue;
                }
                let path = entry.path();
                if kind.is_dir() {
                    if depth < self.search_depth {
                        pending.push((path, depth + 1));
                    }
                    continue;
                }
                let Ok(relative) = path.strip_prefix(skill_dir) else {
                    continue;
                };
                let name = relative
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect::<Vec<_>>()
                    .join("/");
                if path
                    .file_name()
                    .is_some_and(|f| f.to_string_lossy().eq_ignore_ascii_case(SKILL_FILE_NAME))
                {
                    continue;
                }
                let lower = name.to_lowercase();
                if !extensions.iter().any(|e| lower.ends_with(e.as_str())) {
                    continue;
                }
                if filter.is_some_and(|f| !f(skill_name, &name)) {
                    continue;
                }
                out.push((name, path));
            }
        }
        out.sort();
        out
    }

    /// Discover and load the skills. A skill whose `SKILL.md` is invalid is
    /// skipped with an error log; a duplicate name keeps the first found.
    pub fn load(&self) -> Vec<Skill> {
        let mut skills: Vec<Skill> = Vec::new();
        for dir in self.skill_directories() {
            let file = dir.join(SKILL_FILE_NAME);
            let content = match std::fs::read_to_string(&file) {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(file = %file.display(), error = %e, "cannot read SKILL.md");
                    continue;
                }
            };
            let (fm, body) = match parse_skill_md(&content) {
                Ok(parsed) => parsed,
                Err(e) => {
                    tracing::error!(file = %file.display(), error = %e, "skipping invalid skill");
                    continue;
                }
            };
            if skills.iter().any(|s| s.name == fm.name) {
                tracing::warn!(skill = %fm.name, dir = %dir.display(), "duplicate skill name; keeping the first");
                continue;
            }
            let mut skill = Skill::new(fm.name.clone(), fm.description.clone());
            skill.path = Some(std::path::absolute(&dir).unwrap_or_else(|_| dir.clone()));
            skill.license = fm.license.clone();
            skill.compatibility = fm.compatibility.clone();
            skill.allowed_tools = fm.allowed_tools.clone();
            skill.metadata = fm.metadata.clone().unwrap_or_default();
            for (name, path) in self.script_files(&dir, &fm.name) {
                let full_path = std::path::absolute(&path).unwrap_or(path);
                skill = skill.with_script(SkillScript::file(name, full_path));
            }
            let mut listed = Vec::new();
            for (name, path) in self.resource_files(&dir, &fm.name) {
                let too_big =
                    std::fs::metadata(&path).map_or(true, |m| m.len() > MAX_RESOURCE_BYTES);
                if too_big {
                    tracing::warn!(resource = %name, "skipping a resource over 1 MiB");
                    continue;
                }
                match std::fs::read_to_string(&path) {
                    Ok(text) => {
                        listed.push(format!("  <resource name=\"{}\" />", xml_escape(&name)));
                        skill = skill.with_resource(name, text);
                    }
                    Err(e) => {
                        tracing::warn!(resource = %name, error = %e, "skipping an unreadable resource")
                    }
                }
            }
            let resources_block = if listed.is_empty() {
                "<available_resources />".to_string()
            } else {
                format!(
                    "<available_resources>\n{}\n</available_resources>",
                    listed.join("\n")
                )
            };
            let scripts_block = available_scripts_block(&skill.scripts);
            skill = skill.with_instructions(format!(
                "{}\n\n{resources_block}\n\n{scripts_block}",
                body.trim()
            ));
            skills.push(skill);
        }
        skills
    }
}

impl SkillsProvider {
    /// A provider over the skills discovered under `paths` (see
    /// [`FileSkillsSource`]).
    pub fn from_paths<P: AsRef<Path>>(
        paths: impl IntoIterator<Item = P>,
    ) -> crate::error::Result<Self> {
        Ok(Self::new(FileSkillsSource::new(paths).load()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_follow_the_specification() {
        for ok in ["pdf", "data-analysis", "a1-b2"] {
            assert!(validate_skill_name(ok).is_ok(), "{ok}");
        }
        for bad in [
            "",
            "Upper",
            "-lead",
            "trail-",
            "two--hyphens",
            "under_score",
            &"x".repeat(65),
        ] {
            assert!(validate_skill_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn frontmatter_parses_and_validates() {
        let (fm, body) = parse_skill_md(
            "---\nname: pdf-tools\ndescription: >\n  Work with PDFs.\nlicense: MIT\nallowed-tools: read write\nmetadata:\n  version: 2\n  owner: team\n---\n# Use\nStep one.\n",
        )
        .unwrap();
        assert_eq!(fm.name, "pdf-tools");
        assert_eq!(fm.description.trim(), "Work with PDFs.");
        assert_eq!(fm.allowed_tools.as_deref(), Some("read write"));
        assert_eq!(fm.metadata.unwrap()["version"], "2");
        assert_eq!(body, "# Use\nStep one.\n");
        assert!(parse_skill_md("no frontmatter").is_err());
        assert!(
            parse_skill_md("---\nname: x\n---\nbody").is_err(),
            "missing description"
        );
        assert!(parse_skill_md("---\nname: Bad\ndescription: d\n---\n").is_err());
    }
}
