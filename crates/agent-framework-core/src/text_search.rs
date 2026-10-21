//! Retrieval-augmented generation over any text search.
//!
//! Port of .NET's `TextSearchProvider` (`Microsoft.Agents.AI`) and Go's
//! `agent/textsearch` (v0.2.0). [`TextSearchProvider`] is a
//! [`ContextProvider`] wrapping a search function you supply — a vector
//! store query, a web search, a keyword index — and puts its results in
//! front of the model in one of two ways ([`TextSearchBehavior`]):
//!
//! - **Before each run** (the default): the run's input text is searched and
//!   the formatted results are added as a `user` message ahead of it.
//! - **On demand**: the model gets a `Search` tool and calls it when it wants
//!   to.
//!
//! With [`TextSearchProvider::with_recent_message_memory`], the search input
//! also carries the last few messages of the conversation, kept in the
//! session's state, so a follow-up such as "and the second one?" is
//! searched with the question it follows.
//!
//! ```
//! use agent_framework_core::text_search::{TextSearchProvider, TextSearchResult};
//!
//! let provider = TextSearchProvider::new(|query: String| async move {
//!     // Look `query` up in your index.
//!     Ok(vec![TextSearchResult::new("Returns are accepted within 30 days.")
//!         .with_source("Returns policy", "https://example.com/returns")])
//! });
//! # let _ = provider;
//! ```
//!
//! # Divergences
//!
//! - **No message filters.** Upstream filters the search and storage input
//!   so history and other providers' messages are not searched or
//!   remembered. A Rust provider only ever receives the run's own input and
//!   response, so there is nothing to filter out.
//! - **Logging.** Result counts and failures go to `tracing`; the query and
//!   formatted results are logged (at `trace`) only after
//!   [`TextSearchProvider::with_sensitive_logging`], in place of upstream's
//!   redactor.

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::memory::{ContextProvider, SessionContext};
use crate::session::SessionState;
use crate::tools::{FunctionTool, ToolDefinition};
use crate::types::{Message, Role};

/// The default session-state key for recent-message memory.
pub const DEFAULT_TEXT_SEARCH_STATE_KEY: &str = "TextSearchProvider";
const DEFAULT_TOOL_NAME: &str = "Search";
const DEFAULT_TOOL_DESCRIPTION: &str =
    "Allows searching for additional information to help answer the user question.";
const DEFAULT_CONTEXT_PROMPT: &str = "## Additional Context\nConsider the following information from source documents when responding to the user:";
const DEFAULT_CITATIONS_PROMPT: &str = "Include citations to the source document with document name and link if document name and link is available.";

/// Runs that recorded memory input and have not finished yet are capped at
/// this many; see [`TextSearchProvider`]'s `after_run`.
const MAX_PENDING_RUNS: usize = 64;

/// One search hit.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextSearchResult {
    /// The source document's name, if known.
    pub source_name: Option<String>,
    /// A link to the source document, if known.
    pub source_link: Option<String>,
    /// The text to give the model.
    pub text: String,
    /// The search backend's own value for this hit, for a custom formatter.
    pub raw_representation: Option<Value>,
}

impl TextSearchResult {
    /// A result carrying `text` and nothing else.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Default::default()
        }
    }

    /// This result with a source name and link.
    pub fn with_source(mut self, name: impl Into<String>, link: impl Into<String>) -> Self {
        self.source_name = Some(name.into());
        self.source_link = Some(link.into());
        self
    }
}

/// When a [`TextSearchProvider`] searches.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TextSearchBehavior {
    /// Search the run's input before every run and add the results.
    #[default]
    BeforeInvoke,
    /// Give the model a search tool to call when it wants to.
    OnDemandFunctionCalling,
}

type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;
type SearchFn = Arc<dyn Fn(String) -> BoxFuture<Result<Vec<TextSearchResult>>> + Send + Sync>;
type Formatter = Arc<dyn Fn(&[TextSearchResult]) -> Result<String> + Send + Sync>;

/// A run that took memory input in `before_run`, waiting for `after_run`.
struct PendingRun {
    /// The run's input, which `after_run` receives again and is matched on.
    input: Vec<Message>,
    /// The run's session state bag (a shared handle).
    state: SessionState,
}

/// A [`ContextProvider`] that adds text-search results to a run. See the
/// [module docs](self).
///
/// Cloning yields a handle onto the same provider.
#[derive(Clone)]
pub struct TextSearchProvider {
    search: SearchFn,
    behavior: TextSearchBehavior,
    tool_name: String,
    tool_description: String,
    context_prompt: String,
    citations_prompt: String,
    formatter: Option<Formatter>,
    memory_limit: usize,
    memory_roles: Vec<Role>,
    state_key: String,
    sensitive_logging: bool,
    pending: Arc<Mutex<VecDeque<PendingRun>>>,
}

impl std::fmt::Debug for TextSearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSearchProvider")
            .field("behavior", &self.behavior)
            .field("tool_name", &self.tool_name)
            .field("memory_limit", &self.memory_limit)
            .field("state_key", &self.state_key)
            .finish_non_exhaustive()
    }
}

impl TextSearchProvider {
    /// A provider searching with `search`, which takes the query text.
    pub fn new<F, Fut>(search: F) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Vec<TextSearchResult>>> + Send + 'static,
    {
        Self {
            search: Arc::new(move |query| Box::pin(search(query))),
            behavior: TextSearchBehavior::default(),
            tool_name: DEFAULT_TOOL_NAME.to_string(),
            tool_description: DEFAULT_TOOL_DESCRIPTION.to_string(),
            context_prompt: DEFAULT_CONTEXT_PROMPT.to_string(),
            citations_prompt: DEFAULT_CITATIONS_PROMPT.to_string(),
            formatter: None,
            memory_limit: 0,
            memory_roles: vec![Role::user()],
            state_key: DEFAULT_TEXT_SEARCH_STATE_KEY.to_string(),
            sensitive_logging: false,
            pending: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// When to search (default [`TextSearchBehavior::BeforeInvoke`]).
    pub fn with_behavior(mut self, behavior: TextSearchBehavior) -> Self {
        self.behavior = behavior;
        self
    }

    /// The on-demand tool's name and description (default `Search`).
    pub fn with_tool(mut self, name: impl Into<String>, description: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self.tool_description = description.into();
        self
    }

    /// The line put before the results.
    pub fn with_context_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.context_prompt = prompt.into();
        self
    }

    /// The line put after the results, asking for citations.
    pub fn with_citations_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.citations_prompt = prompt.into();
        self
    }

    /// Format results yourself instead of the default layout. Called for
    /// every search, including one with no results.
    pub fn with_context_formatter(
        mut self,
        formatter: impl Fn(&[TextSearchResult]) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        self.formatter = Some(Arc::new(formatter));
        self
    }

    /// Include up to `limit` recent conversation messages in the search
    /// input (default 0, off). Only messages with a role in
    /// [`Self::with_recent_message_roles`] (default `user`) are kept. The
    /// memory lives in the session state under [`Self::with_state_key`], so a
    /// run without a session has none.
    pub fn with_recent_message_memory(mut self, limit: usize) -> Self {
        self.memory_limit = limit;
        self
    }

    /// The roles recent-message memory keeps (default `user` only).
    pub fn with_recent_message_roles(mut self, roles: Vec<Role>) -> Self {
        self.memory_roles = roles;
        self
    }

    /// The session-state key for recent-message memory (default
    /// [`DEFAULT_TEXT_SEARCH_STATE_KEY`]). A blank key is refused.
    pub fn with_state_key(mut self, key: impl Into<String>) -> Result<Self> {
        let key = key.into();
        if key.trim().is_empty() {
            return Err(Error::Configuration(
                "text search state_key must be a non-empty string".into(),
            ));
        }
        self.state_key = key;
        Ok(self)
    }

    /// Log queries and formatted results at `trace` level. Off by default:
    /// both carry conversation content.
    pub fn with_sensitive_logging(mut self, enabled: bool) -> Self {
        self.sensitive_logging = enabled;
        self
    }

    /// The session-state key recent-message memory uses.
    pub fn state_key(&self) -> &str {
        &self.state_key
    }

    /// The on-demand search tool. Added to every run under
    /// [`TextSearchBehavior::OnDemandFunctionCalling`]; it takes one string
    /// argument, `userQuestion`.
    pub fn search_tool(&self) -> ToolDefinition {
        let provider = self.clone();
        FunctionTool::new(
            self.tool_name.clone(),
            self.tool_description.clone(),
            json!({
                "type": "object",
                "properties": { "userQuestion": { "type": "string" } },
                "required": ["userQuestion"],
            }),
            move |args: Value| {
                let provider = provider.clone();
                async move {
                    let question = args
                        .get("userQuestion")
                        .and_then(Value::as_str)
                        .ok_or_else(|| Error::tool("missing string argument 'userQuestion'"))?
                        .to_string();
                    Ok(Value::String(provider.search_and_format(question).await?))
                }
            },
        )
        .into_definition()
    }

    /// Run a search and format its results.
    pub async fn search_and_format(&self, query: String) -> Result<String> {
        let results = (self.search)(query.clone()).await?;
        tracing::debug!(count = results.len(), "text search completed");
        let formatted = self.format_results(&results)?;
        if self.sensitive_logging {
            tracing::trace!(query = %query, output = %formatted, "text search results");
        }
        Ok(formatted)
    }

    fn format_results(&self, results: &[TextSearchResult]) -> Result<String> {
        if let Some(formatter) = &self.formatter {
            return formatter(results);
        }
        if results.is_empty() {
            return Ok(String::new());
        }
        let mut out = String::new();
        out.push_str(&self.context_prompt);
        out.push('\n');
        for result in results {
            if let Some(name) = result
                .source_name
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                out.push_str(&format!("SourceDocName: {name}\n"));
            }
            if let Some(link) = result
                .source_link
                .as_deref()
                .filter(|s| !s.trim().is_empty())
            {
                out.push_str(&format!("SourceDocLink: {link}\n"));
            }
            out.push_str(&format!("Contents: {}\n----\n", result.text));
        }
        out.push_str(&self.citations_prompt);
        out.push_str("\n\n");
        Ok(out)
    }

    fn recent_messages(&self, state: &SessionState) -> Vec<String> {
        state
            .get(&self.state_key)
            .and_then(|v| v.get("recentMessagesText").cloned())
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default()
    }

    fn remember(&self, state: &SessionState, messages: &[&Message]) {
        let mut recent = self.recent_messages(state);
        recent.extend(
            messages
                .iter()
                .filter(|m| self.memory_roles.contains(&m.role))
                .map(|m| m.text())
                .filter(|t| !t.trim().is_empty()),
        );
        if recent.len() > self.memory_limit {
            recent.drain(..recent.len() - self.memory_limit);
        }
        state.insert(
            self.state_key.clone(),
            json!({ "recentMessagesText": recent }),
        );
    }
}

#[async_trait]
impl ContextProvider for TextSearchProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        if self.memory_limit > 0 {
            if let Some(session) = &ctx.session {
                let mut pending = self.pending.lock().unwrap();
                if pending.len() >= MAX_PENDING_RUNS {
                    pending.pop_front();
                }
                pending.push_back(PendingRun {
                    input: ctx.input_messages.clone(),
                    state: session.state.clone(),
                });
            }
        }
        if self.behavior == TextSearchBehavior::OnDemandFunctionCalling {
            ctx.tools.push(self.search_tool());
            return Ok(());
        }
        let mut parts = match (&ctx.session, self.memory_limit) {
            (Some(session), limit) if limit > 0 => self.recent_messages(&session.state),
            _ => Vec::new(),
        };
        parts.extend(
            ctx.input_messages
                .iter()
                .map(Message::text)
                .filter(|t| !t.trim().is_empty()),
        );
        let query = parts.join("\n");
        // A failed search must not fail the run: the model answers without
        // the extra context, as upstream does.
        match self.search_and_format(query).await {
            Ok(formatted) if !formatted.is_empty() => {
                ctx.messages.push(Message::user(formatted));
            }
            Ok(_) => {}
            Err(e) => tracing::error!(error = %e, "text search failed"),
        }
        Ok(())
    }

    /// Records the run's messages in recent-message memory. `after_run`
    /// receives no session, so the run is matched by its input to the state
    /// handle `before_run` recorded (as the harness mode provider does). A
    /// failed run records nothing, and neither does a run whose input
    /// matches in-flight runs of more than one session, since it cannot be
    /// told apart from them.
    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if self.memory_limit == 0 {
            return Ok(());
        }
        // The read-modify-write of the memory below also happens under this
        // lock, so two runs on clones of one session cannot lose each
        // other's messages.
        let mut pending = self.pending.lock().unwrap();
        let matches: Vec<usize> = pending
            .iter()
            .enumerate()
            .filter(|(_, p)| p.input == request_messages)
            .map(|(i, _)| i)
            .collect();
        let Some(&first) = matches.first() else {
            return Ok(());
        };
        let ambiguous = matches
            .iter()
            .any(|&i| !pending[i].state.shares_storage_with(&pending[first].state));
        if ambiguous {
            // Runs of different sessions with the same input are in flight,
            // and nothing says which one this is. Recording into the wrong
            // session would leak one conversation into another's searches,
            // so all of them are dropped and none is remembered.
            for &i in matches.iter().rev() {
                pending.remove(i);
            }
            tracing::debug!("text search memory skipped: concurrent runs of different sessions share this input");
            return Ok(());
        }
        let run = pending.remove(first).expect("index from this queue");
        if error.is_none() {
            let messages: Vec<&Message> =
                request_messages.iter().chain(response_messages).collect();
            self.remember(&run.state, &messages);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_format_matches_upstream() {
        let provider = TextSearchProvider::new(|_| async { Ok(vec![]) });
        let formatted = provider
            .format_results(&[
                TextSearchResult::new("Doc one").with_source("One", "http://one"),
                TextSearchResult::new("Doc two"),
            ])
            .unwrap();
        assert_eq!(
            formatted,
            "## Additional Context\nConsider the following information from source documents when responding to the user:\n\
             SourceDocName: One\nSourceDocLink: http://one\nContents: Doc one\n----\n\
             Contents: Doc two\n----\n\
             Include citations to the source document with document name and link if document name and link is available.\n\n"
        );
        assert_eq!(provider.format_results(&[]).unwrap(), "");
    }

    #[test]
    fn blank_state_key_is_refused() {
        assert!(TextSearchProvider::new(|_| async { Ok(vec![]) })
            .with_state_key(" ")
            .is_err());
    }
}
