//! Operating modes: a [`ContextProvider`] that tracks which mode (by default
//! `plan` or `execute`) an agent is in and tells it how to behave there.
//!
//! Port of upstream `_harness/_mode.py`. The current mode lives in the
//! session's state bag under the provider's `source_id` as
//! `{"current_mode": "<mode>"}`. Mode names are matched case-insensitively
//! and stored lower-cased.
//!
//! When application code changes the mode ([`set_agent_mode`],
//! [`AgentModeProvider::set_mode`]), the previous mode is remembered and the
//! next run injects a `user` message announcing the switch: a model that
//! earlier called `mode_set` itself tends to keep following that call over
//! a changed system prompt.
//!
//! # Divergences
//!
//! - Upstream's `get_agent_mode` / `set_agent_mode` take `source_id`,
//!   `default_mode` and `available_modes` keyword arguments. Here the free
//!   functions use the built-in configuration, and a provider's own
//!   configuration is used through [`AgentModeProvider::mode`],
//!   [`AgentModeProvider::set_mode`] and
//!   [`AgentModeProvider::set_mode_without_notification`] (upstream's
//!   `notify=False`).
//! - Configuration errors (no modes, duplicate modes, an unknown default
//!   mode) are reported by [`AgentModeProviderBuilder::build`] rather than a
//!   constructor exception.

use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::AgentSession;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::Message;

use super::read_state_object;

/// The default `source_id` (state key) of an [`AgentModeProvider`].
pub const DEFAULT_MODE_SOURCE_ID: &str = "agent_mode";

const MODE_GET_INSTRUCTIONS: &str = "Use the mode_get tool to check your current operating mode.\n";
const MODE_SET_INSTRUCTIONS: &str = "Use the mode_set tool to switch between modes as your work \
progresses. Only use mode_set if the user explicitly instructs/allows you to change modes.\n\n";
const MODE_SET_HIDDEN_INSTRUCTIONS: &str = "Mode changes are controlled by the application. Use \
its configured mode-change mechanism only when the user explicitly instructs/allows a mode \
change.\n\n";
const PLAN_MODE_TRANSITION: &str = "7. When approval is granted, always switch to execute mode \
(using the `mode_set` tool), and follow the steps for *Execute mode*.";
const PLAN_MODE_TRANSITION_HIDDEN: &str = "7. When approval is granted, use the application's \
configured mode-change mechanism to transition to execute mode. Follow the steps for *Execute \
mode* only after the mode has changed.";
const PREVIOUS_MODE_STATE_KEY: &str = "previous_mode_for_notification";
const CURRENT_MODE_STATE_KEY: &str = "current_mode";

/// The default provider instructions. `{current_mode}` is replaced with the
/// active mode and `{available_modes}` with one section per configured mode.
pub const DEFAULT_MODE_INSTRUCTIONS: &str = "## Agent Mode\n\n\
- You can operate in different modes. Depending on the mode you are in, \
you will be required to follow different processes.\n\n\
Use the mode_get tool to check your current operating mode.\n\
Use the mode_set tool to switch between modes as your work progresses. \
Only use mode_set if the user explicitly instructs/allows you to change modes.\n\n\
You are currently operating in the {current_mode} mode.\n\n\
### Mandatory Mode based Workflow\n\n\
For every new substantive user request, including short factual questions, \
your behavior is determined by the mode you are in.\n\n\
{available_modes}\n";

/// The `user` message injected after an external mode change.
/// `{previous_mode}` and `{current_mode}` are substituted.
pub const DEFAULT_MODE_CHANGE_NOTIFICATION: &str = "[Mode changed: The operating mode has been \
switched from \"{previous_mode}\" to \"{current_mode}\". You must now adjust your behavior to \
match the \"{current_mode}\" mode.]";

/// The built-in instructions for `plan` mode.
pub const PLAN_MODE_INSTRUCTIONS: &str = "Use this mode when analyzing requirements, breaking \
down tasks, and creating plans. This is the interactive mode — ask clarifying questions, discuss \
options, and get user approval before proceeding.\n\n\
Process to follow when in plan mode:\n\
1. Analyze the request with the purpose of building a research plan.\n\
2. Create a list of todo items.\n\
3. If needed, use the provided tools to do some exploratory checks to help build a plan and \
determine what clarifying questions you may need from the user.\n\
4. Ask for clarifications from the user where needed.\n   \
1. Ask each clarification one by one.\n   \
2. When asking for clarification and you have specific options in mind, present them to the \
user, so they can choose the option instead of having to retype the entire response.\n   \
3. Do not proceed until you have received all the needed clarifications.\n   \
4. Do short exploratory research if it helps with being able to ask sensible clarifications \
from the user.\n\
5. Write the plan to a memory file, so that it is retained even if compaction happens. Make \
sure to update the plan file if the user requests changes.\n\
6. Present the plan to the user and ask for approval to switch to execute mode and process the \
plan.\n\
7. When approval is granted, always switch to execute mode (using the `mode_set` tool), and \
follow the steps for *Execute mode*.";

/// The built-in instructions for `execute` mode.
pub const EXECUTE_MODE_INSTRUCTIONS: &str = "Determine the type of ask:\n\
1. Simple question that doesn't require any further work to answer.\n\
2. Any other work, including complex user request that requires a multi-step process to \
satisfy.\n\n\
If 1. just answer the question directly.\n\
If 2. Work autonomously using your best judgment — do not ask the user questions or wait for \
feedback and follow the following process:\n\
1. If you don't have a plan or tasks yet, analyze the user request and create tasks and a plan. \
(**Skip this step if you came from plan mode**)\n\
2. Work autonomously — use your best judgment to make decisions and keep progressing without \
asking the user questions. The goal is to have a complete, useful result ready when the user \
returns.\n\
3. If you encounter ambiguity or an unexpected situation during execution, choose the most \
reasonable option, note your choice, and keep going.\n\
4. Mark tasks as completed as you finish them.\n\
5. Continue working, thinking and calling tools until you have the research result for the user.";

/// A validated mode configuration: `(normalized name, display name)` pairs
/// in declaration order, plus the normalized default.
#[derive(Debug, Clone)]
struct ModeConfig {
    source_id: String,
    modes: Vec<(String, String)>,
    default_mode: String,
}

impl ModeConfig {
    fn new(
        source_id: String,
        display_modes: impl IntoIterator<Item = String>,
        default_mode: Option<&str>,
    ) -> Result<Self> {
        let mut modes: Vec<(String, String)> = Vec::new();
        for mode in display_modes {
            let display = mode.trim().to_string();
            let normalized = display.to_lowercase();
            if modes.iter().any(|(n, _)| *n == normalized) {
                return Err(Error::Configuration(format!(
                    "Duplicate mode configured: {mode}."
                )));
            }
            modes.push((normalized, display));
        }
        if modes.is_empty() {
            return Err(Error::Configuration(
                "at least one agent mode must be configured.".into(),
            ));
        }
        let mut config = Self {
            source_id,
            default_mode: modes[0].0.clone(),
            modes,
        };
        if let Some(default_mode) = default_mode {
            config.default_mode = config.normalize(default_mode)?;
        }
        Ok(config)
    }

    fn builtin() -> Self {
        Self::new(
            DEFAULT_MODE_SOURCE_ID.to_string(),
            ["plan".to_string(), "execute".to_string()],
            None,
        )
        .expect("the built-in modes are valid")
    }

    fn normalize(&self, mode: &str) -> Result<String> {
        let normalized = mode.trim().to_lowercase();
        if self.modes.iter().any(|(n, _)| *n == normalized) {
            Ok(normalized)
        } else {
            let supported: Vec<String> = self.modes.iter().map(|(_, d)| format!("'{d}'")).collect();
            Err(Error::tool(format!(
                "Invalid mode: {mode}. Supported modes are {}.",
                supported.join(", ")
            )))
        }
    }

    fn display<'a>(&'a self, normalized: &'a str) -> &'a str {
        self.modes
            .iter()
            .find(|(n, _)| n == normalized)
            .map(|(_, d)| d.as_str())
            .unwrap_or(normalized)
    }

    /// Read the current mode, resetting to the default when nothing (or a
    /// mode no longer configured) is stored.
    fn get(&self, session: &AgentSession) -> Result<String> {
        let mut state = read_state_object(session, &self.source_id)?;
        if let Some(Value::String(current)) = state.get(CURRENT_MODE_STATE_KEY) {
            if let Ok(normalized) = self.normalize(current) {
                return Ok(normalized);
            }
        }
        state.insert(
            CURRENT_MODE_STATE_KEY.into(),
            Value::String(self.default_mode.clone()),
        );
        session.state.insert(&self.source_id, Value::Object(state));
        Ok(self.default_mode.clone())
    }

    fn set(&self, session: &AgentSession, mode: &str, notify: bool) -> Result<String> {
        let normalized = self.normalize(mode)?;
        let mut state = read_state_object(session, &self.source_id)?;
        let previous = state
            .insert(
                CURRENT_MODE_STATE_KEY.into(),
                Value::String(normalized.clone()),
            )
            .and_then(|v| v.as_str().map(str::to_string));
        if notify {
            if let Some(previous) = previous.filter(|p| *p != normalized) {
                state.insert(PREVIOUS_MODE_STATE_KEY.into(), Value::String(previous));
            }
        } else {
            state.remove(PREVIOUS_MODE_STATE_KEY);
        }
        session.state.insert(&self.source_id, Value::Object(state));
        Ok(normalized)
    }

    fn take_previous(&self, session: &AgentSession) -> Result<Option<String>> {
        let mut state = read_state_object(session, &self.source_id)?;
        let previous = state.remove(PREVIOUS_MODE_STATE_KEY);
        if previous.is_some() {
            session.state.insert(&self.source_id, Value::Object(state));
        }
        Ok(previous.and_then(|v| v.as_str().map(str::to_string)))
    }
}

/// The current mode of `session` under the built-in configuration
/// (`plan`/`execute`, default `plan`, source id `"agent_mode"`).
///
/// Use [`AgentModeProvider::mode`] for a provider with its own configuration.
pub fn get_agent_mode(session: &AgentSession) -> Result<String> {
    ModeConfig::builtin().get(session)
}

/// Set the mode of `session` under the built-in configuration, returning the
/// normalized mode. When the mode actually changes, the next run announces
/// the switch to the agent.
///
/// Use [`AgentModeProvider::set_mode`] for a provider with its own
/// configuration.
pub fn set_agent_mode(session: &AgentSession, mode: &str) -> Result<String> {
    ModeConfig::builtin().set(session, mode, true)
}

struct ModeInner {
    config: ModeConfig,
    /// `(normalized mode, instructions)` in declaration order.
    mode_instructions: Vec<(String, String)>,
    instructions: Option<String>,
    expose_mode_set: bool,
    expose_mode_get: bool,
}

/// Tracks an agent's operating mode per session and gives it `mode_set` /
/// `mode_get` tools.
///
/// Register it as a context provider on the agent. On every run it adds
/// instructions describing the configured modes and the current one, the
/// tools, and, after an external mode change, a `user` message announcing it.
///
/// Cloning yields a handle onto the same provider.
#[derive(Clone)]
pub struct AgentModeProvider {
    inner: Arc<ModeInner>,
}

impl std::fmt::Debug for AgentModeProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentModeProvider")
            .field("source_id", &self.inner.config.source_id)
            .field("available_modes", &self.available_modes())
            .field("default_mode", &self.inner.config.default_mode)
            .finish_non_exhaustive()
    }
}

impl Default for AgentModeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentModeProvider {
    /// A provider with the built-in `plan` and `execute` modes.
    pub fn new() -> Self {
        Self::builder()
            .build()
            .expect("the default mode configuration is valid")
    }

    /// Start configuring a provider.
    pub fn builder() -> AgentModeProviderBuilder {
        AgentModeProviderBuilder::default()
    }

    /// The state key this provider uses.
    pub fn source_id(&self) -> &str {
        &self.inner.config.source_id
    }

    /// The configured modes (normalized), in declaration order.
    pub fn available_modes(&self) -> Vec<&str> {
        self.inner
            .config
            .modes
            .iter()
            .map(|(n, _)| n.as_str())
            .collect()
    }

    /// The mode a session starts in.
    pub fn default_mode(&self) -> &str {
        &self.inner.config.default_mode
    }

    /// The current mode of `session`.
    pub fn mode(&self, session: &AgentSession) -> Result<String> {
        self.inner.config.get(session)
    }

    /// Change the mode of `session` from application code. The agent is
    /// told about the change on its next run.
    pub fn set_mode(&self, session: &AgentSession, mode: &str) -> Result<String> {
        self.inner.config.set(session, mode, true)
    }

    /// Change the mode without announcing it (upstream's `notify=False`),
    /// for a replacement mode tool whose result the agent already saw. Also
    /// clears any pending announcement.
    pub fn set_mode_without_notification(
        &self,
        session: &AgentSession,
        mode: &str,
    ) -> Result<String> {
        self.inner.config.set(session, mode, false)
    }

    /// The instructions this provider adds for a session in `current_mode`.
    pub fn render_instructions(&self, current_mode: &str) -> String {
        let inner = &self.inner;
        let mode_lines: String = inner
            .mode_instructions
            .iter()
            .map(|(mode, text)| format!("#### {}\n\n{text}\n\n", inner.config.display(mode)))
            .collect();
        let mut instructions = match &inner.instructions {
            Some(custom) => custom.clone(),
            None => {
                let mut text = DEFAULT_MODE_INSTRUCTIONS.to_string();
                if !inner.expose_mode_get {
                    text = text.replace(MODE_GET_INSTRUCTIONS, "");
                }
                if !inner.expose_mode_set {
                    text = text.replace(MODE_SET_INSTRUCTIONS, MODE_SET_HIDDEN_INSTRUCTIONS);
                }
                text
            }
        };
        instructions = instructions
            .replace("{available_modes}", &mode_lines)
            .replace("{current_mode}", current_mode);
        instructions
    }

    /// The mode tools (as configured) bound to `session`.
    pub fn tools(&self, session: &AgentSession) -> Vec<ToolDefinition> {
        let mut tools = Vec::new();
        if self.inner.expose_mode_set {
            #[derive(Deserialize, schemars::JsonSchema)]
            struct Args {
                /// The mode to switch to.
                mode: String,
            }
            let provider = self.clone();
            let session = session.clone();
            tools.push(
                FunctionTool::typed(
                    "mode_set",
                    "Switch the agent's operating mode.",
                    move |args: Args| {
                        let provider = provider.clone();
                        let session = session.clone();
                        async move {
                            let mode = provider.set_mode_without_notification(&session, &args.mode)?;
                            Ok(json!({ "mode": mode, "message": format!("Mode changed to '{mode}'.") }))
                        }
                    },
                )
                .into_definition(),
            );
        }
        if self.inner.expose_mode_get {
            let provider = self.clone();
            let session = session.clone();
            tools.push(
                FunctionTool::new(
                    "mode_get",
                    "Get the agent's current operating mode.",
                    crate::tools::empty_schema(),
                    move |_args| {
                        let provider = provider.clone();
                        let session = session.clone();
                        async move { Ok(json!({ "mode": provider.mode(&session)? })) }
                    },
                )
                .into_definition(),
            );
        }
        tools
    }
}

#[async_trait]
impl ContextProvider for AgentModeProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let session = ctx.session.clone().ok_or_else(|| {
            Error::Configuration("AgentModeProvider requires an agent session.".into())
        })?;
        let current = self.mode(&session)?;
        // Pop the external-change marker so the agent sees it only once.
        let previous = self.inner.config.take_previous(&session)?;
        ctx.add_instructions(self.render_instructions(&current));
        ctx.tools.extend(self.tools(&session));
        if let Some(previous) = previous.filter(|p| *p != current) {
            let config = &self.inner.config;
            let notification = DEFAULT_MODE_CHANGE_NOTIFICATION
                .replace("{previous_mode}", config.display(&previous))
                .replace("{current_mode}", config.display(&current));
            ctx.messages.push(Message::user(notification));
        }
        Ok(())
    }
}

/// Configures an [`AgentModeProvider`].
#[derive(Debug, Clone)]
pub struct AgentModeProviderBuilder {
    source_id: String,
    default_mode: Option<String>,
    mode_instructions: Option<Vec<(String, String)>>,
    instructions: Option<String>,
    expose_mode_set: bool,
    expose_mode_get: bool,
}

impl Default for AgentModeProviderBuilder {
    fn default() -> Self {
        Self {
            source_id: DEFAULT_MODE_SOURCE_ID.to_string(),
            default_mode: None,
            mode_instructions: None,
            instructions: None,
            expose_mode_set: true,
            expose_mode_get: true,
        }
    }
}

impl AgentModeProviderBuilder {
    /// The state key (default `"agent_mode"`).
    pub fn source_id(mut self, source_id: impl Into<String>) -> Self {
        self.source_id = source_id.into();
        self
    }

    /// The mode a session starts in (default: the first configured mode).
    pub fn default_mode(mut self, mode: impl Into<String>) -> Self {
        self.default_mode = Some(mode.into());
        self
    }

    /// Replace the built-in modes with `(mode, instructions)` pairs, in the
    /// order they should be listed. Custom text is used verbatim.
    pub fn mode_instructions<K, V>(mut self, modes: impl IntoIterator<Item = (K, V)>) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.mode_instructions = Some(
            modes
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        );
        self
    }

    /// Replace the provider instructions. `{current_mode}` and
    /// `{available_modes}` are substituted; the text is otherwise used as
    /// given, even when a tool is hidden.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.instructions = Some(instructions.into());
        self
    }

    /// Whether to contribute the `mode_set` tool (default `true`). With it
    /// hidden, the application controls mode changes and the default
    /// instructions say so.
    pub fn expose_mode_set(mut self, expose: bool) -> Self {
        self.expose_mode_set = expose;
        self
    }

    /// Whether to contribute the `mode_get` tool (default `true`).
    pub fn expose_mode_get(mut self, expose: bool) -> Self {
        self.expose_mode_get = expose;
        self
    }

    /// Validate and build the provider. Errors when no modes are configured,
    /// two modes normalize to the same name, or the default mode is not one
    /// of them.
    pub fn build(self) -> Result<AgentModeProvider> {
        let mode_instructions = match self.mode_instructions {
            Some(custom) => custom,
            None => {
                let plan = if self.expose_mode_set {
                    PLAN_MODE_INSTRUCTIONS.to_string()
                } else {
                    PLAN_MODE_INSTRUCTIONS
                        .replace(PLAN_MODE_TRANSITION, PLAN_MODE_TRANSITION_HIDDEN)
                };
                vec![
                    ("plan".to_string(), plan),
                    ("execute".to_string(), EXECUTE_MODE_INSTRUCTIONS.to_string()),
                ]
            }
        };
        let config = ModeConfig::new(
            self.source_id,
            mode_instructions.iter().map(|(m, _)| m.clone()),
            self.default_mode.as_deref(),
        )?;
        let mode_instructions = mode_instructions
            .into_iter()
            .map(|(m, text)| (m.trim().to_lowercase(), text))
            .collect();
        Ok(AgentModeProvider {
            inner: Arc::new(ModeInner {
                config,
                mode_instructions,
                instructions: self.instructions,
                expose_mode_set: self.expose_mode_set,
                expose_mode_get: self.expose_mode_get,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn invoke(tools: &[ToolDefinition], name: &str, args: Value) -> Result<Value> {
        tools
            .iter()
            .find(|t| t.name == name)
            .unwrap()
            .executor
            .as_ref()
            .unwrap()
            .invoke(args)
            .await
    }

    #[test]
    fn get_and_set_agent_mode_manage_session_state() {
        let session = AgentSession::new();
        assert_eq!(get_agent_mode(&session).unwrap(), "plan");
        assert_eq!(
            session.state.get(DEFAULT_MODE_SOURCE_ID),
            Some(json!({"current_mode": "plan"}))
        );
        assert_eq!(set_agent_mode(&session, " EXECUTE ").unwrap(), "execute");
        assert_eq!(get_agent_mode(&session).unwrap(), "execute");
        assert!(set_agent_mode(&session, "unknown").is_err());
    }

    #[test]
    fn helpers_reject_non_object_state() {
        let session = AgentSession::new();
        session.state.insert(DEFAULT_MODE_SOURCE_ID, json!("plan"));
        assert!(get_agent_mode(&session).is_err());
        assert!(set_agent_mode(&session, "plan").is_err());
    }

    #[test]
    fn builder_validates_configuration() {
        assert!(AgentModeProvider::builder()
            .mode_instructions(Vec::<(String, String)>::new())
            .build()
            .is_err());
        assert!(AgentModeProvider::builder()
            .mode_instructions([("Plan", "a"), ("plan", "b")])
            .build()
            .is_err());
        assert!(AgentModeProvider::builder()
            .default_mode("missing")
            .build()
            .is_err());
        let provider = AgentModeProvider::builder()
            .mode_instructions([(" Draft ", "draft it"), ("Review", "review it")])
            .build()
            .unwrap();
        assert_eq!(provider.available_modes(), ["draft", "review"]);
        assert_eq!(provider.default_mode(), "draft");
    }

    #[test]
    fn default_mode_falls_back_when_stored_mode_is_not_configured() {
        let provider = AgentModeProvider::builder()
            .mode_instructions([("draft", "d"), ("review", "r")])
            .default_mode("review")
            .build()
            .unwrap();
        let session = AgentSession::new();
        session
            .state
            .insert(DEFAULT_MODE_SOURCE_ID, json!({"current_mode": "plan"}));
        assert_eq!(provider.mode(&session).unwrap(), "review");
    }

    #[test]
    fn set_records_previous_mode_only_for_real_external_changes() {
        let session = AgentSession::new();
        let provider = AgentModeProvider::new();
        provider.set_mode(&session, "plan").unwrap();
        assert!(
            session.state.get(DEFAULT_MODE_SOURCE_ID).unwrap()[PREVIOUS_MODE_STATE_KEY].is_null()
        );
        provider.set_mode(&session, "execute").unwrap();
        assert_eq!(
            session.state.get(DEFAULT_MODE_SOURCE_ID).unwrap()[PREVIOUS_MODE_STATE_KEY],
            "plan"
        );
        // A silent change clears the pending announcement.
        provider
            .set_mode_without_notification(&session, "plan")
            .unwrap();
        assert!(
            session.state.get(DEFAULT_MODE_SOURCE_ID).unwrap()[PREVIOUS_MODE_STATE_KEY].is_null()
        );
    }

    #[tokio::test]
    async fn before_run_injects_instructions_tools_and_one_time_notification() {
        let provider = AgentModeProvider::new();
        let session = AgentSession::new();
        provider.set_mode(&session, "plan").unwrap();
        provider.set_mode(&session, "execute").unwrap();

        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(session.clone());
        provider.before_run(&mut ctx).await.unwrap();
        let instructions = ctx.instructions.clone().unwrap();
        assert!(instructions.contains("You are currently operating in the execute mode."));
        assert!(instructions.contains("#### plan\n\n"));
        assert!(instructions.contains("#### execute\n\n"));
        assert!(instructions.contains("mode_get"));
        assert_eq!(
            ctx.tools
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            ["mode_set", "mode_get"]
        );
        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(
            ctx.messages[0].text(),
            "[Mode changed: The operating mode has been switched from \"plan\" to \"execute\". \
             You must now adjust your behavior to match the \"execute\" mode.]"
        );

        // The announcement is delivered once.
        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(session.clone());
        provider.before_run(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }

    #[tokio::test]
    async fn tools_update_mode_and_return_json() {
        let provider = AgentModeProvider::new();
        let session = AgentSession::new();
        let tools = provider.tools(&session);
        assert_eq!(
            invoke(&tools, "mode_get", json!({})).await.unwrap(),
            json!({"mode": "plan"})
        );
        assert_eq!(
            invoke(&tools, "mode_set", json!({"mode": "Execute"}))
                .await
                .unwrap(),
            json!({"mode": "execute", "message": "Mode changed to 'execute'."})
        );
        assert_eq!(provider.mode(&session).unwrap(), "execute");
        assert!(invoke(&tools, "mode_set", json!({"mode": "nope"}))
            .await
            .is_err());

        // The agent's own switch is not announced back to it.
        let mut ctx = SessionContext::new(vec![]);
        ctx.session = Some(session);
        provider.before_run(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }

    #[test]
    fn hidden_tools_adjust_default_instructions_only() {
        let provider = AgentModeProvider::builder()
            .expose_mode_set(false)
            .expose_mode_get(false)
            .build()
            .unwrap();
        let session = AgentSession::new();
        assert!(provider.tools(&session).is_empty());
        let text = provider.render_instructions("plan");
        assert!(!text.contains("mode_get"));
        assert!(!text.contains("Use the mode_set tool"));
        assert!(text.contains("Mode changes are controlled by the application."));
        assert!(text.contains("use the application's configured mode-change mechanism"));

        // Custom instructions are not rewritten.
        let custom = AgentModeProvider::builder()
            .expose_mode_set(false)
            .instructions("Mode: {current_mode}. Use mode_set.\n{available_modes}")
            .mode_instructions([("a", "alpha")])
            .build()
            .unwrap();
        assert_eq!(
            custom.render_instructions("a"),
            "Mode: a. Use mode_set.\n#### a\n\nalpha\n\n"
        );
    }

    #[test]
    fn default_instructions_preserve_upstream_wording() {
        let text = AgentModeProvider::new().render_instructions("plan");
        assert!(text.starts_with("## Agent Mode\n\n"));
        assert!(text.contains(MODE_GET_INSTRUCTIONS));
        assert!(text.contains(MODE_SET_INSTRUCTIONS));
        assert!(text.contains(PLAN_MODE_TRANSITION));
    }
}
