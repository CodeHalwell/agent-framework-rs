//! Agent Skills served over MCP (SEP-2640), the counterpart of upstream's
//! `MCPSkillsSource`.
//!
//! Discovery reads the well-known `skill://index.json` resource. Each
//! `skill-md` entry (`name`, `description`, `url`) becomes a
//! [`Skill`](agent_framework_core::skills::Skill) whose instructions are the
//! body of the `SKILL.md` resource at `url`; pass the skills to a
//! [`SkillsProvider`](agent_framework_core::skills::SkillsProvider).
//!
//! # Divergences
//!
//! - `SKILL.md` bodies are read at discovery rather than on first
//!   `load_skill`: the Rust [`Skill`](agent_framework_core::skills::Skill)
//!   carries its instructions. Resources referenced inside a `SKILL.md` are
//!   not prefetched.
//! - `archive` entries (packaged ZIP skills) are skipped, as are entry types
//!   with no handler (upstream skips the latter too).
//! - An absent, unreadable or malformed index yields no skills, as upstream;
//!   an unreadable or invalid `SKILL.md` skips that skill.

use agent_framework_core::error::Result;
use agent_framework_core::skills::Skill;
use agent_framework_core::skills_files::parse_skill_md;
use serde::Deserialize;

use crate::McpClient;

/// The well-known discovery resource.
pub const SKILLS_INDEX_URI: &str = "skill://index.json";

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

#[derive(Debug, Deserialize)]
struct SkillIndex {
    #[serde(default)]
    skills: Vec<SkillIndexEntry>,
}

async fn read_text(client: &McpClient, uri: &str) -> Option<String> {
    let result = client.read_resource(uri).await.ok()?;
    result.contents.into_iter().find_map(|c| c.text)
}

/// Discover the skills `client`'s server serves. See the [module docs](self).
pub async fn load_skills(client: &McpClient) -> Result<Vec<Skill>> {
    let Some(text) = read_text(client, SKILLS_INDEX_URI).await else {
        return Ok(Vec::new());
    };
    let Ok(index) = serde_json::from_str::<SkillIndex>(&text) else {
        return Ok(Vec::new());
    };
    let mut skills = Vec::new();
    for entry in index.skills {
        let is_skill_md = entry
            .kind
            .as_deref()
            .is_some_and(|k| k.eq_ignore_ascii_case("skill-md"));
        let Some(url) = entry.url.as_deref().filter(|_| is_skill_md) else {
            continue;
        };
        let Some(content) = read_text(client, url).await else {
            continue;
        };
        let Ok((frontmatter, body)) = parse_skill_md(&content) else {
            continue;
        };
        let description = entry
            .description
            .filter(|d| !d.is_empty())
            .unwrap_or(frontmatter.description);
        let mut skill = Skill::new(frontmatter.name, description).with_instructions(body);
        skill.license = frontmatter.license;
        skill.compatibility = frontmatter.compatibility;
        skill.allowed_tools = frontmatter.allowed_tools;
        skill.metadata = frontmatter.metadata.unwrap_or_default();
        skills.push(skill);
    }
    Ok(skills)
}
