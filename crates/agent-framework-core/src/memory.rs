//! Context / memory providers.
//!
//! Rust equivalent of `agent_framework._memory`. A [`ContextProvider`] injects
//! extra instructions, messages, and tools into an agent invocation without
//! persisting them to the conversation history.
//!
//! Upstream renamed `ContextProvider.invoking`/`invoked` to `before_run`/
//! `after_run` and removed the `thread_created` hook entirely; `before_run`
//! mutates a [`SessionContext`] in place instead of returning a value. There
//! is no aggregate wrapper any more — consumers hold a
//! `Vec<Arc<dyn ContextProvider>>` and iterate it directly.

use async_trait::async_trait;

use crate::error::{Error, Result};
use crate::tools::ToolDefinition;
use crate::types::Message;

/// Per-invocation context a provider contributes to a run. Providers mutate
/// this in place in before_run. Rust equivalent of upstream SessionContext.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Local session identifier (from the thread), for provider scoping.
    pub session_id: Option<String>,
    /// Service-managed session/conversation id, when applicable.
    pub service_session_id: Option<String>,
    /// The session the run belongs to, when the caller of `before_run` has
    /// one (an [`Agent`](crate::agent::Agent) run always does). Clones share
    /// the session's state bag, so a provider can read and persist
    /// per-session state through it. Mirrors the `session` argument of
    /// upstream's `before_run`.
    pub session: Option<crate::session::AgentSession>,
    /// The agent running, when the caller of `before_run` is one (an
    /// [`Agent`](crate::agent::Agent) run always sets it). Mirrors the
    /// `agent` argument of upstream's `before_run`, as the identity a
    /// provider may scope by rather than the agent object itself.
    pub agent: Option<AgentInfo>,
    /// The run's input messages. A provider may rewrite them (for example
    /// to strip an attachment it has analyzed); the model receives the
    /// rewritten input, while `after_run` still sees the caller's original.
    pub input_messages: Vec<Message>,
    /// Extra system instructions to inject (providers append via add_instructions).
    pub instructions: Option<String>,
    /// Extra context messages to inject ahead of history.
    pub messages: Vec<Message>,
    /// Extra tools to make available for this run.
    pub tools: Vec<ToolDefinition>,
}

/// The [`Message::additional_properties`] key carrying a context message's
/// source attribution: `{"source_id", "source_type"?, "origin_session_ids"?}`
/// (upstream `_attribution`). The run's own input carries none.
pub const ATTRIBUTION_KEY: &str = "_attribution";

/// Tag `message` as contributed by `source_id`, merging with any attribution
/// it already carries (an existing `source_id`/`source_type` is kept; origin
/// session ids accumulate without duplicates).
pub fn attribute_message(
    message: &mut Message,
    source_id: &str,
    source_type: Option<&str>,
    origin_session_ids: &[String],
) {
    let entry = message
        .additional_properties
        .entry(ATTRIBUTION_KEY.to_string())
        .or_insert_with(|| serde_json::json!({}));
    if !entry.is_object() {
        *entry = serde_json::json!({});
    }
    let map = entry.as_object_mut().expect("an object");
    map.entry("source_id")
        .or_insert_with(|| serde_json::json!(source_id));
    if let Some(source_type) = source_type {
        map.entry("source_type")
            .or_insert_with(|| serde_json::json!(source_type));
    }
    if !origin_session_ids.is_empty() {
        let mut origins: Vec<String> = map
            .get("origin_session_ids")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        for id in origin_session_ids {
            if !origins.contains(id) {
                origins.push(id.clone());
            }
        }
        map.insert("origin_session_ids".into(), serde_json::json!(origins));
    }
}

/// Tag `message` with a source (upstream `WithAgentRequestMessageSource`),
/// returning it.
pub fn with_message_source(mut message: Message, source_type: &str, source_id: &str) -> Message {
    attribute_message(&mut message, source_id, Some(source_type), &[]);
    message
}

/// The type of the source a message was attributed to, if any (upstream
/// `GetAgentRequestMessageSourceType`).
pub fn message_source_type(message: &Message) -> Option<&str> {
    message
        .additional_properties
        .get(ATTRIBUTION_KEY)?
        .get("source_type")?
        .as_str()
}

/// The source a context message was attributed to, if any.
pub fn message_source_id(message: &Message) -> Option<&str> {
    message
        .additional_properties
        .get(ATTRIBUTION_KEY)?
        .get("source_id")?
        .as_str()
}

/// The identity of the agent a [`SessionContext`] belongs to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentInfo {
    /// The agent's id.
    pub id: String,
    /// The agent's name, if it has one.
    pub name: Option<String>,
    /// The agent's description, if it has one.
    pub description: Option<String>,
}

impl SessionContext {
    pub fn new(input_messages: Vec<Message>) -> Self {
        Self {
            input_messages,
            ..Default::default()
        }
    }
    /// Add context messages from `source_id`, each tagged with its source
    /// (see [`ATTRIBUTION_KEY`]). Mirrors upstream's `extend_messages`.
    pub fn extend_messages(&mut self, source_id: &str, messages: Vec<Message>) {
        self.extend_messages_from(source_id, None, messages, &[]);
    }

    /// As [`extend_messages`](Self::extend_messages), also recording the
    /// provider's type and the sessions the messages came from when that is
    /// not the current one (cross-session memory), for audit and governance.
    pub fn extend_messages_from(
        &mut self,
        source_id: &str,
        source_type: Option<&str>,
        messages: Vec<Message>,
        origin_session_ids: &[String],
    ) {
        for mut message in messages {
            attribute_message(&mut message, source_id, source_type, origin_session_ids);
            self.messages.push(message);
        }
    }

    /// The context messages, optionally only those from `sources` and not
    /// from `exclude_sources`; unattributed messages count as from no source.
    pub fn get_messages(
        &self,
        sources: Option<&[&str]>,
        exclude_sources: Option<&[&str]>,
    ) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| {
                let source = message_source_id(m);
                sources.is_none_or(|s| source.is_some_and(|id| s.contains(&id)))
                    && exclude_sources.is_none_or(|s| source.is_none_or(|id| !s.contains(&id)))
            })
            .cloned()
            .collect()
    }

    /// Append instructions, newline-concatenating with any already present.
    pub fn add_instructions(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.instructions = match self.instructions.take() {
            Some(existing) => Some(format!("{existing}\n{s}")),
            None => Some(s),
        };
    }
}

/// A source of per-invocation context (memory, RAG, etc.).
/// Upstream renamed invoking/invoked -> before_run/after_run and REMOVED
/// thread_created. before_run mutates the SessionContext in place instead of
/// returning a Context.
#[async_trait]
pub trait ContextProvider: Send + Sync {
    /// Called before the model is invoked; mutate ctx to inject instructions,
    /// messages, and/or tools. Read ctx.input_messages / ctx.session_id.
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()>;

    /// Called after an invocation completes, on BOTH success and failure.
    /// On success, error is None and response_messages holds the output.
    /// On failure, error is Some and response_messages is empty.
    async fn after_run(
        &self,
        _request_messages: &[Message],
        _response_messages: &[Message],
        _error: Option<&Error>,
    ) -> Result<()> {
        Ok(())
    }

    /// The completion hook with the run's session, which is what agents
    /// call. Mirrors the `session` argument of upstream's `after_run`.
    ///
    /// The default forwards to [`after_run`](Self::after_run), so a provider
    /// that does not need the session implements only that; one that does
    /// (to find per-session or per-owner storage, say) overrides this.
    async fn after_run_in_session(
        &self,
        session: &crate::session::AgentSession,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        let _ = session;
        self.after_run(request_messages, response_messages, error)
            .await
    }

    /// Whether this provider manages conversation history (a
    /// [`HistoryProvider`](crate::history::HistoryProvider)). [`Agent`](crate::agent::Agent)
    /// and [`WorkflowAgent`](crate::workflow::WorkflowAgent) use this to
    /// detect an already-attached history provider among a session's
    /// `context_providers` and avoid auto-attaching a redundant
    /// [`InMemoryHistoryProvider`](crate::history::InMemoryHistoryProvider).
    /// Defaults to `false`; history providers override it to `true`.
    fn is_history_provider(&self) -> bool {
        false
    }

    /// The id this provider's context messages are attributed to (see
    /// [`ATTRIBUTION_KEY`]). Defaults to the provider's type name; upstream's
    /// `source_id`.
    fn source_id(&self) -> String {
        let name = std::any::type_name::<Self>();
        let base = name.split('<').next().unwrap_or(name);
        base.rsplit("::").next().unwrap_or(base).to_string()
    }

    /// The conversation history this provider keeps only in process memory,
    /// for a [session store](crate::session_store) snapshot to carry.
    ///
    /// Defaults to `None`, which is right for every provider that is not a
    /// history provider and for one that persists its own messages (a file
    /// or database history): restoring those means reattaching them, not
    /// copying their messages. [`InMemoryHistoryProvider`](crate::history::InMemoryHistoryProvider)
    /// returns its messages.
    #[cfg(feature = "experimental-session-store")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental-session-store")))]
    fn snapshot_history(&self) -> Option<Vec<Message>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_instructions_sets_when_none() {
        let mut ctx = SessionContext::new(vec![]);
        assert!(ctx.instructions.is_none());
        ctx.add_instructions("be brief");
        assert_eq!(ctx.instructions.as_deref(), Some("be brief"));
    }

    #[test]
    fn add_instructions_newline_concatenates() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.add_instructions("first");
        ctx.add_instructions("second");
        ctx.add_instructions("third");
        assert_eq!(ctx.instructions.as_deref(), Some("first\nsecond\nthird"));
    }

    #[test]
    fn new_sets_input_messages_and_defaults_rest() {
        let messages = vec![Message::user("hi")];
        let ctx = SessionContext::new(messages.clone());
        assert_eq!(ctx.input_messages.len(), messages.len());
        assert_eq!(ctx.input_messages[0].text(), "hi");
        assert!(ctx.session_id.is_none());
        assert!(ctx.service_session_id.is_none());
        assert!(ctx.session.is_none());
        assert!(ctx.instructions.is_none());
        assert!(ctx.messages.is_empty());
        assert!(ctx.tools.is_empty());
    }
}
