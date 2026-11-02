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
//! Skills are built in code here, or discovered from `SKILL.md` packages on
//! disk with the `file-skills` feature (`skills_files`,
//! `SkillsProvider::from_paths`).
//!
//! A skill may also carry [`SkillScript`]s: inline scripts (a closure run
//! in-process) or file scripts (a file in the skill's directory, run by the
//! provider's [`SkillScriptRunner`], which the application supplies — there
//! is no default). When any skill has a script the provider adds a third
//! tool, `run_skill_script`, which **requires approval by default** because
//! it runs code; [`SkillsProvider::with_script_approval`] opts out for
//! trusted skills.
//!
//! This is still a subset of upstream: no MCP-backed skills, no class-based
//! skills, and no skill-source pipeline (filtering / caching / aggregating
//! sources).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::tools::{BoxFuture, FunctionTool};

/// An inline script's body: called with the model's `args`.
pub type InlineScriptFn = Arc<dyn Fn(Option<Value>) -> BoxFuture<Result<Value>> + Send + Sync>;

/// Runs file-based skill scripts: called with the owning skill, the script
/// and the model's `args` (for file scripts, an array of CLI argument
/// strings). The application decides how — a subprocess, a sandbox, a
/// hosted code interpreter.
pub type SkillScriptRunner =
    Arc<dyn Fn(Skill, SkillScript, Option<Value>) -> BoxFuture<Result<Value>> + Send + Sync>;

/// Where a [`SkillScript`]'s code lives.
#[derive(Clone)]
pub enum SkillScriptSource {
    /// Code-defined: run in-process.
    Inline(InlineScriptFn),
    /// A file in the skill's directory, run by the provider's
    /// [`SkillScriptRunner`].
    File {
        /// The script's absolute path.
        full_path: std::path::PathBuf,
    },
}

/// An executable script attached to a [`Skill`], run through the
/// `run_skill_script` tool.
#[derive(Clone)]
pub struct SkillScript {
    /// The name the model runs it by (for file scripts, the path relative to
    /// the skill, e.g. `scripts/process.py`). Matched case-insensitively.
    pub name: String,
    /// A short summary, if any.
    pub description: Option<String>,
    /// JSON Schema for `args`, advertised to the model, if any. File scripts
    /// take an array of strings.
    pub parameters_schema: Option<Value>,
    /// The script's code.
    pub source: SkillScriptSource,
}

impl std::fmt::Debug for SkillScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("SkillScript");
        d.field("name", &self.name)
            .field("description", &self.description);
        if let SkillScriptSource::File { full_path } = &self.source {
            d.field("full_path", full_path);
        }
        d.finish_non_exhaustive()
    }
}

impl SkillScript {
    /// An inline script running `f` with the model's `args`.
    pub fn inline<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Option<Value>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: None,
            parameters_schema: None,
            source: SkillScriptSource::Inline(Arc::new(move |args| Box::pin(f(args)))),
        }
    }

    /// A file script at `full_path`, run by the provider's runner.
    pub fn file(name: impl Into<String>, full_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            name: name.into(),
            description: None,
            parameters_schema: Some(
                serde_json::json!({"type": "array", "items": {"type": "string"}}),
            ),
            source: SkillScriptSource::File {
                full_path: full_path.into(),
            },
        }
    }

    /// Builder: a short summary.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Builder: the JSON Schema of `args`.
    pub fn with_parameters_schema(mut self, schema: Value) -> Self {
        self.parameters_schema = Some(schema);
        self
    }

    /// The file script's path, if it is one.
    pub fn full_path(&self) -> Option<&std::path::Path> {
        match &self.source {
            SkillScriptSource::File { full_path } => Some(full_path),
            SkillScriptSource::Inline(_) => None,
        }
    }
}

fn xml_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// The `<available_scripts>` block listing `scripts` (upstream
/// `_build_available_scripts_block`): `<available_scripts />` when there are
/// none, so the model does not invent names.
pub fn available_scripts_block(scripts: &[SkillScript]) -> String {
    if scripts.is_empty() {
        return "<available_scripts />".into();
    }
    let lines: Vec<String> = scripts
        .iter()
        .map(|script| {
            let mut attrs = format!("name=\"{}\"", xml_attr(&script.name));
            if let Some(d) = &script.description {
                attrs.push_str(&format!(" description=\"{}\"", xml_attr(d)));
            }
            match &script.parameters_schema {
                Some(schema) => format!(
                    "  <script {attrs}>\n    <parameters_schema>{}</parameters_schema>\n  </script>",
                    xml_attr(&schema.to_string())
                ),
                None => format!("  <script {attrs} />"),
            }
        })
        .collect();
    format!(
        "<available_scripts>\n{}\n</available_scripts>",
        lines.join("\n")
    )
}

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
    /// Scripts the model can run with `run_skill_script`.
    pub scripts: Vec<SkillScript>,
    /// The directory a file-based skill was loaded from.
    pub path: Option<std::path::PathBuf>,
    /// A license name or reference (frontmatter `license`).
    pub license: Option<String>,
    /// Compatibility notes (frontmatter `compatibility`).
    pub compatibility: Option<String>,
    /// Space-delimited pre-approved tool names (frontmatter `allowed-tools`).
    pub allowed_tools: Option<String>,
    /// Free-form string metadata (frontmatter `metadata`).
    pub metadata: std::collections::BTreeMap<String, String>,
}

impl Skill {
    /// A new skill with an empty `instructions` body and no resources.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            instructions: String::new(),
            resources: HashMap::new(),
            scripts: Vec::new(),
            path: None,
            license: None,
            compatibility: None,
            allowed_tools: None,
            metadata: Default::default(),
        }
    }

    /// Builder: attach a script (replacing one with the same name,
    /// case-insensitively).
    pub fn with_script(mut self, script: SkillScript) -> Self {
        self.scripts
            .retain(|s| !s.name.eq_ignore_ascii_case(&script.name));
        self.scripts.push(script);
        self
    }

    /// The script named `name`, matched case-insensitively.
    pub fn script(&self, name: &str) -> Option<&SkillScript> {
        self.scripts
            .iter()
            .find(|s| s.name.eq_ignore_ascii_case(name))
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
    script_runner: Option<SkillScriptRunner>,
    script_approval: bool,
}

impl SkillsProvider {
    /// Build a provider from a list of skills, keyed by [`Skill::name`]. If
    /// two skills share a name, the later one in `skills` wins.
    pub fn new(skills: Vec<Skill>) -> Self {
        let skills = skills.into_iter().map(|s| (s.name.clone(), s)).collect();
        Self {
            skills: Arc::new(skills),
            loaded: Arc::new(Mutex::new(HashSet::new())),
            script_runner: None,
            script_approval: true,
        }
    }

    /// Run file-based scripts with `runner`. Without one, running a file
    /// script answers with an error; inline scripts need none.
    pub fn with_script_runner<F, Fut>(mut self, runner: F) -> Self
    where
        F: Fn(Skill, SkillScript, Option<Value>) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<Value>> + Send + 'static,
    {
        self.script_runner = Some(Arc::new(move |skill, script, args| {
            Box::pin(runner(skill, script, args))
        }));
        self
    }

    /// Whether `run_skill_script` requires approval (the default, as
    /// upstream). Turn it off only for trusted skills.
    pub fn with_script_approval(mut self, required: bool) -> Self {
        self.script_approval = required;
        self
    }

    fn has_scripts(&self) -> bool {
        self.skills.values().any(|s| !s.scripts.is_empty())
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
        if self.has_scripts() {
            lines.push(
                "Use `run_skill_script` to run a script a loaded skill lists, using the name \
                 exactly as listed. Pass named script arguments inside `args` as a JSON object; \
                 for file-based scripts that take CLI-style arguments, pass `args` as an array \
                 of strings."
                    .to_string(),
            );
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
                            if skill.scripts.is_empty()
                                || skill.instructions.contains("<available_scripts")
                            {
                                skill.instructions.clone()
                            } else {
                                format!(
                                    "{}\n\n{}",
                                    skill.instructions,
                                    available_scripts_block(&skill.scripts)
                                )
                            }
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

    /// The `run_skill_script(skill_name, script_name, args?)` tool. Unknown
    /// names answer with an error message the model can read; a script's
    /// own failure is an error of the call, handled by the function loop.
    fn run_skill_script_tool(&self) -> FunctionTool {
        let skills = Arc::clone(&self.skills);
        let runner = self.script_runner.clone();
        let tool = FunctionTool::new(
            "run_skill_script",
            "Runs a script associated with a skill.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "The name of the skill."},
                    "script_name": {"type": "string", "description": "The name of the script, exactly as listed."},
                    "args": {
                        "description": "Arguments for the script: a JSON object of named arguments, or an array of strings for file-based scripts.",
                        "anyOf": [
                            {"type": "object"},
                            {"type": "array", "items": {"type": "string"}}
                        ]
                    }
                },
                "required": ["skill_name", "script_name"],
                "additionalProperties": false
            }),
            move |args: Value| {
                let skills = Arc::clone(&skills);
                let runner = runner.clone();
                async move {
                    let text = |k: &str| {
                        args.get(k)
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .trim()
                            .to_string()
                    };
                    let (skill_name, script_name) = (text("skill_name"), text("script_name"));
                    if skill_name.is_empty() {
                        return Ok(Value::String("Error: Skill name cannot be empty.".into()));
                    }
                    if script_name.is_empty() {
                        return Ok(Value::String("Error: Script name cannot be empty.".into()));
                    }
                    let Some(skill) = skills.get(&skill_name) else {
                        return Ok(Value::String(format!(
                            "Error: Skill '{skill_name}' not found."
                        )));
                    };
                    let Some(script) = skill.script(&script_name) else {
                        return Ok(Value::String(format!(
                            "Error: Script '{script_name}' not found in skill '{skill_name}'."
                        )));
                    };
                    let script_args = args.get("args").cloned().filter(|a| !a.is_null());
                    match &script.source {
                        SkillScriptSource::Inline(f) => f(script_args).await,
                        SkillScriptSource::File { full_path } => {
                            let Some(runner) = runner else {
                                return Err(Error::Configuration(format!(
                                    "Script '{}' requires a runner. Provide a script runner for file-based scripts.",
                                    script.name
                                )));
                            };
                            // Recheck the file: it may have been replaced by a
                            // link since discovery.
                            let linked = std::fs::symlink_metadata(full_path)
                                .map(|m| !m.is_file())
                                .unwrap_or(true);
                            if linked {
                                return Err(Error::Configuration(format!(
                                    "Script '{}' is no longer a regular file.",
                                    script.name
                                )));
                            }
                            runner(skill.clone(), script.clone(), script_args).await
                        }
                    }
                }
            },
        );
        if self.script_approval {
            tool.with_approval_mode(crate::tools::ApprovalMode::AlwaysRequire)
        } else {
            tool
        }
    }
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
        if self.has_scripts() {
            ctx.tools
                .push(self.run_skill_script_tool().into_definition());
        }

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
