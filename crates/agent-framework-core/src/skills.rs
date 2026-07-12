//! Skills: progressive-disclosure capability packages.
//!
//! Rust equivalent of `agent_framework._skills` (upstream `_skills.py`,
//! ~4,370 lines). A [`Skill`] is a named capability package: a short
//! `description` (always visible to the model, so it can decide whether the
//! skill is relevant), a longer `instructions` body (revealed on demand), and
//! zero or more named `resources` (revealed on demand, individually).
//!
//! [`SkillsProvider`] is a [`ContextProvider`] that surfaces a set of skills
//! to an agent run with **progressive disclosure**: on every
//! [`ContextProvider::before_run`] it injects a compact catalog (name +
//! description for every skill) into the run's instructions, plus two
//! framework-generated [`FunctionTool`]s — `load_skill` and
//! `read_skill_resource` — that let the model pull in a skill's full
//! `instructions` or a specific resource only when it decides the skill is
//! relevant, instead of paying the token cost of every skill's full detail
//! on every turn. Once a skill has been loaded (via a `load_skill` call),
//! its full instructions are also injected on every subsequent `before_run`
//! for the lifetime of the provider, so the model does not have to re-load it
//! each turn.
//!
//! This is a deliberate **subset** of upstream: no MCP-backed skills (an
//! `@experimental` upstream feature) and no `run_skill_script` /
//! sandboxed script execution (upstream's third framework-generated tool) —
//! both are out of scope here. Only `load_skill` and `read_skill_resource`
//! are implemented.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;
use crate::memory::{ContextProvider, SessionContext};
use crate::tools::FunctionTool;

/// A named, progressive-disclosure capability package.
///
/// * `description` is short and always visible to the model (via
///   [`SkillsProvider`]'s catalog), so the model can judge relevance without
///   paying for the full detail.
/// * `instructions` is the full guidance for actually using the skill;
///   revealed only after the model calls `load_skill` (or if the skill was
///   already loaded on a previous turn of the same [`SkillsProvider`]).
/// * `resources` are additional named text blobs (reference docs, examples,
///   schemas, ...) revealed individually via `read_skill_resource`, so a
///   skill can carry more detail than is worth inlining into
///   `instructions` up front.
#[derive(Debug, Clone, Default)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub instructions: String,
    pub resources: HashMap<String, String>,
}

impl Skill {
    /// A new skill with an empty `instructions` body and no resources.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: String::new(),
            resources: HashMap::new(),
        }
    }

    /// Builder: set the skill's full instructions (revealed on `load_skill`).
    pub fn with_instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = instructions.into();
        self
    }

    /// Builder: attach a named resource (revealed on `read_skill_resource`).
    /// Calling this again with the same `name` replaces the prior content.
    pub fn with_resource(mut self, name: impl Into<String>, content: impl Into<String>) -> Self {
        self.resources.insert(name.into(), content.into());
        self
    }
}

/// A [`ContextProvider`] that attaches a set of [`Skill`]s to an agent run
/// with progressive disclosure.
///
/// Holds its skills behind an `Arc` (so `SkillsProvider` itself is cheap to
/// `Clone`, e.g. into a session's `context_providers`) and its set of
/// currently-loaded skill names behind an `Arc<Mutex<_>>`, shared with the
/// `load_skill` tool closure so that a call to the tool during a run is
/// visible to `before_run` on every subsequent run.
#[derive(Clone)]
pub struct SkillsProvider {
    skills: Arc<HashMap<String, Skill>>,
    loaded: Arc<Mutex<HashSet<String>>>,
}

impl SkillsProvider {
    /// Build a provider from a list of skills, keyed by [`Skill::name`]. If
    /// two skills share a name, the later one in `skills` wins.
    pub fn new(skills: Vec<Skill>) -> Self {
        let skills = skills.into_iter().map(|s| (s.name.clone(), s)).collect();
        Self {
            skills: Arc::new(skills),
            loaded: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// The short catalog injected into every run's instructions: one bullet
    /// per skill (name + description), plus a note describing how the model
    /// can pull in more detail.
    fn catalog(&self) -> String {
        let mut names: Vec<&String> = self.skills.keys().collect();
        names.sort();

        let mut lines = vec![
            "Available skills (progressive disclosure — each skill below is only \
             summarized; call the `load_skill` tool with a skill's name to reveal \
             its full instructions, and `read_skill_resource` to read one of its \
             named resources):"
                .to_string(),
        ];
        for name in names {
            let skill = &self.skills[name];
            lines.push(format!("- {}: {}", skill.name, skill.description));
        }
        lines.join("\n")
    }

    /// Full instructions for every skill currently marked as loaded, sorted
    /// by name for determinism. Empty when no skill has been loaded yet.
    fn loaded_instructions(&self) -> Vec<String> {
        let loaded = self.loaded.lock().unwrap();
        let mut names: Vec<&String> = loaded.iter().collect();
        names.sort();
        names
            .into_iter()
            .filter_map(|name| self.skills.get(name))
            .map(|skill| {
                format!(
                    "Full instructions for skill '{}':\n{}",
                    skill.name, skill.instructions
                )
            })
            .collect()
    }

    /// The `load_skill(skill_name: String) -> String` tool: marks the named
    /// skill as loaded (so its full instructions are injected on every
    /// subsequent `before_run`) and returns those instructions directly, so
    /// the model can act on them immediately without waiting for the next
    /// turn. An unknown skill name is not an error — it returns a clear
    /// message the model can read and recover from.
    fn load_skill_tool(&self) -> FunctionTool {
        let skills = Arc::clone(&self.skills);
        let loaded = Arc::clone(&self.loaded);
        FunctionTool::new(
            "load_skill",
            "Load a skill by name to reveal its full instructions. Use this once \
             a skill from the catalog looks relevant to the current task.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill to load, exactly as listed in the catalog."
                    }
                },
                "required": ["skill_name"]
            }),
            move |args: Value| {
                let skills = Arc::clone(&skills);
                let loaded = Arc::clone(&loaded);
                async move {
                    let skill_name = args
                        .get("skill_name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let response = match skills.get(&skill_name) {
                        Some(skill) => {
                            loaded.lock().unwrap().insert(skill_name);
                            skill.instructions.clone()
                        }
                        None => format!("No skill named '{skill_name}' is available."),
                    };
                    Ok(Value::String(response))
                }
            },
        )
    }

    /// The `read_skill_resource(skill_name: String, resource_name: String) ->
    /// String` tool: returns a named resource's content, or a clear
    /// not-found message for an unknown skill or resource name.
    fn read_skill_resource_tool(&self) -> FunctionTool {
        let skills = Arc::clone(&self.skills);
        FunctionTool::new(
            "read_skill_resource",
            "Read a named resource belonging to a skill (e.g. reference docs, \
             examples, or schemas the skill's instructions point to).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill that owns the resource."
                    },
                    "resource_name": {
                        "type": "string",
                        "description": "The name of the resource to read."
                    }
                },
                "required": ["skill_name", "resource_name"]
            }),
            move |args: Value| {
                let skills = Arc::clone(&skills);
                async move {
                    let skill_name = args
                        .get("skill_name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let resource_name = args
                        .get("resource_name")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    let response = match skills.get(&skill_name) {
                        Some(skill) => match skill.resources.get(&resource_name) {
                            Some(content) => content.clone(),
                            None => format!(
                                "Skill '{skill_name}' has no resource named '{resource_name}'."
                            ),
                        },
                        None => format!("No skill named '{skill_name}' is available."),
                    };
                    Ok(Value::String(response))
                }
            },
        )
    }

    // Note: upstream also generates a third tool, `run_skill_script`, backed
    // by sandboxed script execution. That — like MCP-backed skills — is out
    // of scope for this subset; only `load_skill` and `read_skill_resource`
    // are provided.
}

#[async_trait]
impl ContextProvider for SkillsProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        ctx.add_instructions(self.catalog());
        for instructions in self.loaded_instructions() {
            ctx.add_instructions(instructions);
        }

        ctx.tools.push(self.load_skill_tool().into_definition());
        ctx.tools
            .push(self.read_skill_resource_tool().into_definition());

        Ok(())
    }

    // after_run is a no-op: skills carry no per-run state to reconcile once
    // a run completes (unlike a HistoryProvider, which records messages
    // here). Uses the trait's default implementation.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    fn two_skills() -> Vec<Skill> {
        vec![
            Skill::new("weather", "Get current weather for a city")
                .with_instructions("Call the weather API with a city name and units.")
                .with_resource("api_reference", "GET /weather?city={city}"),
            Skill::new("translate", "Translate text between languages").with_instructions(
                "Detect the source language, then translate to the target language.",
            ),
        ]
    }

    #[tokio::test]
    async fn before_run_injects_catalog_and_adds_tools() {
        let provider = SkillsProvider::new(two_skills());
        let mut ctx = SessionContext::new(vec![Message::user("hi")]);

        provider.before_run(&mut ctx).await.unwrap();

        let instructions = ctx.instructions.clone().unwrap_or_default();
        assert!(instructions.contains("weather: Get current weather for a city"));
        assert!(instructions.contains("translate: Translate text between languages"));
        assert!(instructions.contains("load_skill"));
        assert!(instructions.contains("read_skill_resource"));
        // Neither skill has been loaded yet, so no full instructions leak in.
        assert!(!instructions.contains("Call the weather API"));

        assert_eq!(ctx.tools.len(), 2);
        assert!(ctx.tools.iter().any(|t| t.name == "load_skill"));
        assert!(ctx.tools.iter().any(|t| t.name == "read_skill_resource"));
        assert!(ctx.tools.iter().all(|t| t.is_executable()));
    }

    #[tokio::test]
    async fn load_skill_returns_instructions_and_persists_across_runs() {
        let provider = SkillsProvider::new(two_skills());
        let mut ctx = SessionContext::new(vec![]);
        provider.before_run(&mut ctx).await.unwrap();

        let load_tool = ctx.tools.iter().find(|t| t.name == "load_skill").unwrap();
        let executor = load_tool.executor.clone().unwrap();
        let result = executor
            .invoke(serde_json::json!({"skill_name": "weather"}))
            .await
            .unwrap();
        assert_eq!(
            result.as_str().unwrap(),
            "Call the weather API with a city name and units."
        );

        // A later before_run (e.g. next turn) must now also inject the full
        // instructions for the loaded skill, alongside the catalog.
        let mut ctx2 = SessionContext::new(vec![]);
        provider.before_run(&mut ctx2).await.unwrap();
        let instructions2 = ctx2.instructions.unwrap_or_default();
        assert!(instructions2.contains("Full instructions for skill 'weather':"));
        assert!(instructions2.contains("Call the weather API with a city name and units."));
        // The un-loaded skill's full instructions still must not appear.
        assert!(!instructions2.contains("Detect the source language"));
    }

    #[tokio::test]
    async fn load_skill_reports_unknown_skill_with_a_clear_message() {
        let provider = SkillsProvider::new(two_skills());
        let mut ctx = SessionContext::new(vec![]);
        provider.before_run(&mut ctx).await.unwrap();

        let load_tool = ctx.tools.iter().find(|t| t.name == "load_skill").unwrap();
        let executor = load_tool.executor.clone().unwrap();
        let result = executor
            .invoke(serde_json::json!({"skill_name": "nonexistent"}))
            .await
            .unwrap();
        assert_eq!(
            result.as_str().unwrap(),
            "No skill named 'nonexistent' is available."
        );
    }

    #[tokio::test]
    async fn read_skill_resource_returns_content_or_clear_not_found_messages() {
        let provider = SkillsProvider::new(two_skills());
        let mut ctx = SessionContext::new(vec![]);
        provider.before_run(&mut ctx).await.unwrap();

        let read_tool = ctx
            .tools
            .iter()
            .find(|t| t.name == "read_skill_resource")
            .unwrap();
        let executor = read_tool.executor.clone().unwrap();

        let found = executor
            .invoke(serde_json::json!({"skill_name": "weather", "resource_name": "api_reference"}))
            .await
            .unwrap();
        assert_eq!(found.as_str().unwrap(), "GET /weather?city={city}");

        let missing_resource = executor
            .invoke(serde_json::json!({"skill_name": "weather", "resource_name": "nope"}))
            .await
            .unwrap();
        assert_eq!(
            missing_resource.as_str().unwrap(),
            "Skill 'weather' has no resource named 'nope'."
        );

        let missing_skill = executor
            .invoke(serde_json::json!({"skill_name": "nope", "resource_name": "nope"}))
            .await
            .unwrap();
        assert_eq!(
            missing_skill.as_str().unwrap(),
            "No skill named 'nope' is available."
        );
    }
}
