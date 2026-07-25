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
    /// The run's input messages (read-only for providers).
    pub input_messages: Vec<Message>,
    /// Extra system instructions to inject (providers append via add_instructions).
    pub instructions: Option<String>,
    /// Extra context messages to inject ahead of history.
    pub messages: Vec<Message>,
    /// Extra tools to make available for this run.
    pub tools: Vec<ToolDefinition>,
}

/// Identifies the provider contributing context messages, for attribution.
///
/// Mirrors the `source: str | object` parameter of upstream's
/// `SessionContext.extend_messages`: a bare id, or an id plus the provider's
/// type name. Rust has no stable runtime type name for an
/// `Arc<dyn ContextProvider>`, so a provider that wants `source_type` recorded
/// supplies it explicitly.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ContextSource {
    pub source_id: String,
    pub source_type: Option<String>,
}

impl ContextSource {
    pub fn new(source_id: impl Into<String>) -> Self {
        Self {
            source_id: source_id.into(),
            source_type: None,
        }
    }

    /// Also record the provider's type name, matching what upstream derives
    /// from `type(source).__name__`.
    pub fn with_type(mut self, source_type: impl Into<String>) -> Self {
        self.source_type = Some(source_type.into());
        self
    }
}

impl From<&str> for ContextSource {
    fn from(id: &str) -> Self {
        ContextSource::new(id)
    }
}

impl From<String> for ContextSource {
    fn from(id: String) -> Self {
        ContextSource::new(id)
    }
}

/// The `additional_properties` key attribution is recorded under. Matches
/// upstream's `_attribution` exactly — it is a cross-language wire contract,
/// read by downstream context observers.
pub const ATTRIBUTION_KEY: &str = "_attribution";

/// Return `ids` in first-seen order with duplicates removed. Mirrors upstream's
/// `_deduplicate_origin_session_ids`.
fn deduplicate_origin_session_ids<'a>(ids: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    for id in ids {
        if seen.insert(id) {
            out.push(id.to_string());
        }
    }
    out
}

impl SessionContext {
    pub fn new(input_messages: Vec<Message>) -> Self {
        Self {
            input_messages,
            ..Default::default()
        }
    }
    /// Append instructions, newline-concatenating with any already present.
    pub fn add_instructions(&mut self, s: impl Into<String>) {
        let s = s.into();
        self.instructions = match self.instructions.take() {
            Some(existing) => Some(format!("{existing}\n{s}")),
            None => Some(s),
        };
    }

    /// Add context messages attributed to `source`.
    ///
    /// Each message is stamped with an `_attribution` entry in its
    /// `additional_properties` recording which provider contributed it, then
    /// appended to [`SessionContext::messages`]. Mirrors upstream's
    /// `SessionContext.extend_messages`.
    ///
    /// Prefer this over pushing onto `messages` directly: an unattributed
    /// context message is indistinguishable from the user's own input once it
    /// reaches history.
    pub fn extend_messages(
        &mut self,
        source: impl Into<ContextSource>,
        messages: impl IntoIterator<Item = Message>,
    ) {
        self.extend_messages_from_sessions(source, messages, &[]);
    }

    /// Add context messages attributed to `source` and to the sessions that
    /// originally produced them.
    ///
    /// `origin_session_ids` is for providers injecting content stored under
    /// *other* sessions — cross-session memory. The ids describe the
    /// contributing sessions for the whole call rather than pairing
    /// positionally with messages, since one composed message can have several
    /// origins. They surface under
    /// `additional_properties["_attribution"]["origin_session_ids"]` so
    /// downstream observers can detect cross-session content for governance or
    /// audit. Pass an empty slice when content originates in the current
    /// session: an absent field means no origin information was supplied,
    /// which is distinct from "originated here" (upstream #7041).
    pub fn extend_messages_from_sessions(
        &mut self,
        source: impl Into<ContextSource>,
        messages: impl IntoIterator<Item = Message>,
        origin_session_ids: &[String],
    ) {
        let source = source.into();
        let origins = deduplicate_origin_session_ids(origin_session_ids.iter().map(String::as_str));

        for mut message in messages {
            let attribution = message
                .additional_properties
                .entry(ATTRIBUTION_KEY.to_string())
                .or_insert_with(|| serde_json::json!({}));

            let Some(map) = attribution.as_object_mut() else {
                // A non-object `_attribution` is someone else's data; leave it
                // rather than clobbering it.
                self.messages.push(message);
                continue;
            };

            // Existing keys win — the first provider to attribute a message
            // owns it, matching upstream's `setdefault`.
            map.entry("source_id")
                .or_insert_with(|| serde_json::json!(source.source_id));
            if let Some(source_type) = &source.source_type {
                map.entry("source_type")
                    .or_insert_with(|| serde_json::json!(source_type));
            }
            // Origins are the exception: they accumulate across providers, so a
            // message composed from several sessions lists all of them.
            if !origins.is_empty() {
                let existing: Vec<String> = map
                    .get("origin_session_ids")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let merged = deduplicate_origin_session_ids(
                    existing
                        .iter()
                        .map(String::as_str)
                        .chain(origins.iter().map(String::as_str)),
                );
                map.insert("origin_session_ids".into(), serde_json::json!(merged));
            }

            self.messages.push(message);
        }
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
        assert!(ctx.instructions.is_none());
        assert!(ctx.messages.is_empty());
        assert!(ctx.tools.is_empty());
    }
}

#[cfg(test)]
mod attribution_tests {
    use super::*;
    use crate::types::Role;

    fn attribution(msg: &Message) -> &serde_json::Value {
        msg.additional_properties
            .get(ATTRIBUTION_KEY)
            .expect("message must carry attribution")
    }

    fn origins(msg: &Message) -> Vec<String> {
        attribution(msg)
            .get("origin_session_ids")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn extend_messages_stamps_the_source_id() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.extend_messages("rag", [Message::user("recalled")]);

        assert_eq!(ctx.messages.len(), 1);
        assert_eq!(attribution(&ctx.messages[0])["source_id"], "rag");
        assert_eq!(ctx.messages[0].role, Role::user());
    }

    #[test]
    fn extend_messages_records_source_type_when_supplied() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.extend_messages(
            ContextSource::new("rag").with_type("VectorMemoryProvider"),
            [Message::user("recalled")],
        );

        let attr = attribution(&ctx.messages[0]);
        assert_eq!(attr["source_id"], "rag");
        assert_eq!(attr["source_type"], "VectorMemoryProvider");
    }

    /// Upstream #7041: content pulled from other sessions is marked with the
    /// sessions it came from, so an observer can tell cross-session content
    /// apart from content that originated here.
    #[test]
    fn origin_session_ids_are_recorded_and_deduplicated() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.extend_messages_from_sessions(
            "cross-session-memory",
            [Message::user("from elsewhere")],
            &["sess-a".into(), "sess-b".into(), "sess-a".into()],
        );

        // First-seen order, no duplicates.
        assert_eq!(origins(&ctx.messages[0]), vec!["sess-a", "sess-b"]);
    }

    /// An absent field means "no origin information supplied", which is
    /// deliberately distinct from "originated in this session".
    #[test]
    fn no_origins_leaves_the_field_absent() {
        let mut ctx = SessionContext::new(vec![]);
        ctx.extend_messages("rag", [Message::user("local")]);

        assert!(attribution(&ctx.messages[0])
            .get("origin_session_ids")
            .is_none());
    }

    /// A message composed from several sessions accumulates every origin, so
    /// origins merge where the other attribution keys are first-writer-wins.
    #[test]
    fn origins_accumulate_while_source_id_is_first_writer_wins() {
        let mut msg = Message::user("composed");
        let mut ctx = SessionContext::new(vec![]);

        ctx.extend_messages_from_sessions("first", [msg.clone()], &["sess-a".into()]);
        msg = ctx.messages.pop().unwrap();

        ctx.extend_messages_from_sessions("second", [msg], &["sess-b".into(), "sess-a".into()]);

        let out = &ctx.messages[0];
        assert_eq!(
            attribution(out)["source_id"],
            "first",
            "the first provider to attribute a message keeps ownership"
        );
        assert_eq!(origins(out), vec!["sess-a", "sess-b"]);
    }

    /// A pre-existing non-object `_attribution` belongs to someone else and is
    /// left intact rather than overwritten.
    #[test]
    fn non_object_attribution_is_left_untouched() {
        let mut msg = Message::user("odd");
        msg.additional_properties
            .insert(ATTRIBUTION_KEY.into(), serde_json::json!("not-a-map"));

        let mut ctx = SessionContext::new(vec![]);
        ctx.extend_messages("rag", [msg]);

        assert_eq!(
            *attribution(&ctx.messages[0]),
            serde_json::json!("not-a-map")
        );
    }
}
