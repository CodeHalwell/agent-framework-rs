//! Conversation-history compaction.
//!
//! Rust equivalent of (a self-contained subset of) upstream `_compaction.py`
//! (see `UPSTREAM_DRIFT.md` §9). Upstream's `_compaction.py` is a large
//! (1500+ line), annotation-driven system: it groups messages into logical
//! spans (system / user / assistant-text / tool-call), stamps grouping and
//! token-count metadata onto `Message.additional_properties`, and never
//! deletes history — it flags messages `_excluded` and lets the client filter
//! them out when building the payload sent to the model. It ships seven
//! strategies (`Truncation`, `SlidingWindow`, `SelectiveToolCall`,
//! `ToolResult`, LLM-backed `Summarization`, `TokenBudgetComposed`,
//! `ContextWindow`) plus a `CompactionProvider(ContextProvider)` that wires a
//! strategy into the client's `get_response` loop.
//!
//! This module intentionally delivers a smaller, dependency-free surface:
//! the [`Tokenizer`] and [`CompactionStrategy`] abstractions upstream
//! defines, plus four concrete, non-LLM strategies that mirror upstream's
//! `Truncation`, `SlidingWindow`, `ContextWindow`/`TokenBudget`, and
//! `ToolResult` (renamed [`SelectiveToolResult`] here to avoid confusion with
//! `Content::FunctionResult`... "tool result" is the plain-English name).
//! Compaction here works by *returning a reduced list* rather than annotating
//! messages in place — simpler, and sufficient for the strategies included.
//! Wiring a strategy into the client's `get_response` loop (upstream's
//! `CompactionProvider`) is intentionally out of scope for this change; see
//! `UPSTREAM_DRIFT.md` §9.
//!
//! Compaction never errors on content: given any message list it returns a
//! (possibly unchanged) retained subset that satisfies the strategy's
//! constraint.

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::memory::{ContextProvider, SessionContext};
use crate::types::{Content, Message, Role};

/// Counts tokens for a piece of text. Rust equivalent of upstream
/// `TokenizerProtocol`.
pub trait Tokenizer: Send + Sync {
    /// Count the tokens represented by `text`.
    fn count_tokens(&self, text: &str) -> usize;
}

/// A dependency-free default tokenizer using a ~4-characters-per-token
/// heuristic. Mirrors upstream's `CharacterEstimatorTokenizer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ApproxTokenizer;

impl Tokenizer for ApproxTokenizer {
    fn count_tokens(&self, text: &str) -> usize {
        text.chars().count().div_ceil(4)
    }
}

/// Sum the token counts of a message's text content (text and reasoning
/// content items) using `tokenizer`.
pub fn count_message_tokens(tokenizer: &dyn Tokenizer, message: &Message) -> usize {
    message
        .contents
        .iter()
        .filter_map(Content::as_text)
        .map(|text| tokenizer.count_tokens(text))
        .sum()
}

/// A strategy that reduces a message list to fit some constraint.
///
/// Compaction never errors on content — it always returns *some* retained
/// subset of `messages`, in original order. Rust equivalent of upstream
/// `CompactionStrategy`.
pub trait CompactionStrategy: Send + Sync {
    /// Return the retained messages (in original order) after compaction.
    fn compact(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message>;
}

/// Returns the number of leading messages with `Role::system()`.
fn leading_system_count(messages: &[Message]) -> usize {
    messages
        .iter()
        .take_while(|m| m.role == Role::system())
        .count()
}

/// Keep the most recent `max_messages`, always preserving any leading system
/// message(s) at the front. Mirrors upstream's `Truncation` strategy.
#[derive(Debug, Clone, Copy)]
pub struct Truncation {
    pub max_messages: usize,
}

impl Truncation {
    pub fn new(max_messages: usize) -> Self {
        Self { max_messages }
    }
}

impl CompactionStrategy for Truncation {
    fn compact(&self, messages: &[Message], _tokenizer: &dyn Tokenizer) -> Vec<Message> {
        if messages.len() <= self.max_messages {
            return messages.to_vec();
        }
        let sys_count = leading_system_count(messages);
        let mut out: Vec<Message> = messages[..sys_count].to_vec();

        if sys_count >= self.max_messages {
            // The system prefix alone already fills (or exceeds) the budget;
            // keep just the system prefix, truncated to the budget.
            out.truncate(self.max_messages);
            return out;
        }

        let remaining_budget = self.max_messages - sys_count;
        let rest = &messages[sys_count..];
        let start = rest.len().saturating_sub(remaining_budget);
        out.extend_from_slice(&rest[start..]);
        out
    }
}

/// Keep leading system message(s) + the last `window` non-system messages.
/// Mirrors upstream's `SlidingWindow` strategy.
#[derive(Debug, Clone, Copy)]
pub struct SlidingWindow {
    pub window: usize,
}

impl SlidingWindow {
    pub fn new(window: usize) -> Self {
        Self { window }
    }
}

impl CompactionStrategy for SlidingWindow {
    fn compact(&self, messages: &[Message], _tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let sys_count = leading_system_count(messages);
        let mut out: Vec<Message> = messages[..sys_count].to_vec();
        let rest = &messages[sys_count..];
        let start = rest.len().saturating_sub(self.window);
        out.extend_from_slice(&rest[start..]);
        out
    }
}

/// Keep leading system message(s), then walk from the newest message
/// backward accumulating token counts, keeping messages until adding the
/// next would exceed `max_tokens`. Returns the kept messages in original
/// order. Mirrors upstream's `ContextWindow`/token-budget strategy.
#[derive(Debug, Clone, Copy)]
pub struct TokenBudget {
    pub max_tokens: usize,
}

impl TokenBudget {
    pub fn new(max_tokens: usize) -> Self {
        Self { max_tokens }
    }
}

impl CompactionStrategy for TokenBudget {
    fn compact(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let sys_count = leading_system_count(messages);
        let system_prefix = &messages[..sys_count];
        let rest = &messages[sys_count..];

        let mut used: usize = system_prefix
            .iter()
            .map(|m| count_message_tokens(tokenizer, m))
            .sum();

        // Walk from newest to oldest over the non-system tail, keeping
        // messages until adding the next would exceed the budget. The
        // newest non-system message is always kept, even if it alone (plus
        // the system prefix) exceeds the budget — compaction never reduces
        // a non-empty tail to nothing.
        let mut kept_rest: Vec<&Message> = Vec::new();
        for message in rest.iter().rev() {
            let cost = count_message_tokens(tokenizer, message);
            if !kept_rest.is_empty() && used + cost > self.max_tokens {
                break;
            }
            used += cost;
            kept_rest.push(message);
        }
        kept_rest.reverse();

        let mut out: Vec<Message> = system_prefix.to_vec();
        out.extend(kept_rest.into_iter().cloned());
        out
    }
}

/// Whether a message carries any `Content::FunctionResult` (tool-result)
/// content.
fn has_tool_result(message: &Message) -> bool {
    message
        .contents
        .iter()
        .any(|c| matches!(c, Content::FunctionResult(_)))
}

/// Drop `Content::FunctionResult` (tool-result) content from all but the last
/// `keep_last` messages that carry tool results — they are the bulkiest and
/// least useful once stale. Text and other content is left intact. Messages
/// that become empty after stripping are dropped entirely. Mirrors upstream's
/// `ToolResult` strategy.
#[derive(Debug, Clone, Copy)]
pub struct SelectiveToolResult {
    pub keep_last: usize,
}

impl SelectiveToolResult {
    pub fn new(keep_last: usize) -> Self {
        Self { keep_last }
    }
}

impl CompactionStrategy for SelectiveToolResult {
    fn compact(&self, messages: &[Message], _tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let tool_result_count = messages.iter().filter(|m| has_tool_result(m)).count();
        let mut strip_budget = tool_result_count.saturating_sub(self.keep_last);

        let mut out = Vec::with_capacity(messages.len());
        for message in messages {
            if has_tool_result(message) && strip_budget > 0 {
                strip_budget -= 1;
                let contents: Vec<Content> = message
                    .contents
                    .iter()
                    .filter(|c| !matches!(c, Content::FunctionResult(_)))
                    .cloned()
                    .collect();
                if contents.is_empty() {
                    continue;
                }
                let mut stripped = message.clone();
                stripped.contents = contents;
                out.push(stripped);
            } else {
                out.push(message.clone());
            }
        }
        out
    }
}

/// Convenience free function: compact `messages` with `strategy` and
/// `tokenizer`.
pub fn compact(
    messages: &[Message],
    strategy: &dyn CompactionStrategy,
    tokenizer: &dyn Tokenizer,
) -> Vec<Message> {
    strategy.compact(messages, tokenizer)
}

/// A [`ContextProvider`] that compacts the accumulated message list —
/// typically the run's history, once a [`HistoryProvider`](crate::history::HistoryProvider)
/// has prepended it in `before_run` — down to fit a [`CompactionStrategy`]'s
/// constraint before it reaches the model. Rust equivalent of (a subset of)
/// upstream's `CompactionProvider` (see module docs and `UPSTREAM_DRIFT.md`
/// §9).
///
/// Register it via [`AgentBuilder::with_compaction`](crate::agent::AgentBuilder::with_compaction),
/// which attaches it as one of the agent's own context providers — those run
/// *after* the session's (which is where a history provider, auto-attached
/// or explicit, lives — see [`Agent::combined_providers`](crate::agent::Agent)),
/// so compaction always sees the full, history-prepended message list for the
/// run.
pub struct CompactionProvider {
    strategy: Arc<dyn CompactionStrategy>,
    tokenizer: Box<dyn Tokenizer>,
}

impl CompactionProvider {
    /// A compaction provider using `strategy` with the default
    /// [`ApproxTokenizer`].
    pub fn new(strategy: impl CompactionStrategy + 'static) -> Self {
        Self::with_tokenizer(strategy, ApproxTokenizer)
    }

    /// A compaction provider using `strategy` and an explicit `tokenizer`.
    pub fn with_tokenizer(
        strategy: impl CompactionStrategy + 'static,
        tokenizer: impl Tokenizer + 'static,
    ) -> Self {
        Self {
            strategy: Arc::new(strategy),
            tokenizer: Box::new(tokenizer),
        }
    }
}

#[async_trait]
impl ContextProvider for CompactionProvider {
    /// Replace `ctx.messages` (the accumulated history + any earlier
    /// provider-injected messages) with the strategy's compacted subset.
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        ctx.messages = self.strategy.compact(&ctx.messages, &*self.tokenizer);
        Ok(())
    }

    // `after_run` is intentionally a no-op (the default from `ContextProvider`):
    // compaction only shapes the outgoing request, it never observes or
    // records the run's outcome.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FunctionResultContent;
    use serde_json::json;

    fn text(role: Role, s: &str) -> Message {
        Message::new(role, s)
    }

    fn tool_result_message(call_id: &str, result: &str) -> Message {
        Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent::new(
                call_id,
                Some(json!(result)),
            ))],
        )
    }

    // ---- ApproxTokenizer -------------------------------------------------

    #[test]
    fn approx_tokenizer_uses_four_chars_per_token_ceiling() {
        let t = ApproxTokenizer;
        assert_eq!(t.count_tokens(""), 0);
        assert_eq!(t.count_tokens("abcd"), 1);
        assert_eq!(t.count_tokens("abcde"), 2); // ceil(5/4) = 2
        assert_eq!(t.count_tokens("abcdefgh"), 2);
        assert_eq!(t.count_tokens("abcdefghi"), 3); // ceil(9/4) = 3
    }

    #[test]
    fn count_message_tokens_sums_text_content() {
        let t = ApproxTokenizer;
        let msg = Message::with_contents(
            Role::user(),
            vec![Content::text("abcd"), Content::text("abcdefgh")],
        );
        // 1 + 2 = 3
        assert_eq!(count_message_tokens(&t, &msg), 3);
    }

    // ---- Truncation --------------------------------------------------------

    #[test]
    fn truncation_keeps_most_recent_messages() {
        let messages = vec![
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
            text(Role::assistant(), "4"),
        ];
        let strategy = Truncation::new(2);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text(), "3");
        assert_eq!(out[1].text(), "4");
    }

    #[test]
    fn truncation_preserves_leading_system_messages() {
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
            text(Role::assistant(), "4"),
        ];
        let strategy = Truncation::new(2);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        // system preserved + 1 most recent (budget of 2 total)
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Role::system());
        assert_eq!(out[0].text(), "sys");
        assert_eq!(out[1].text(), "4");
    }

    #[test]
    fn truncation_preserves_multiple_leading_system_messages() {
        let messages = vec![
            text(Role::system(), "sys1"),
            text(Role::system(), "sys2"),
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
        ];
        let strategy = Truncation::new(3);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text(), "sys1");
        assert_eq!(out[1].text(), "sys2");
        assert_eq!(out[2].text(), "2");
    }

    #[test]
    fn truncation_noop_when_under_budget() {
        let messages = vec![text(Role::user(), "1"), text(Role::assistant(), "2")];
        let strategy = Truncation::new(10);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out, messages);
    }

    // ---- SlidingWindow -------------------------------------------------

    #[test]
    fn sliding_window_keeps_system_plus_last_n_non_system() {
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
        ];
        let strategy = SlidingWindow::new(2);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text(), "sys");
        assert_eq!(out[1].text(), "2");
        assert_eq!(out[2].text(), "3");
    }

    #[test]
    fn sliding_window_with_no_system_message() {
        let messages = vec![
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
        ];
        let strategy = SlidingWindow::new(1);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text(), "3");
    }

    // ---- TokenBudget --------------------------------------------------

    /// A tokenizer with a fixed per-message-call cost, for deterministic
    /// tests independent of exact text length.
    struct FixedTokenizer(usize);
    impl Tokenizer for FixedTokenizer {
        fn count_tokens(&self, _text: &str) -> usize {
            self.0
        }
    }

    #[test]
    fn token_budget_keeps_only_what_fits_from_the_newest_backward() {
        let messages = vec![
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
            text(Role::assistant(), "4"),
        ];
        // Each message costs a fixed 10 tokens; budget for 2 messages.
        let tokenizer = FixedTokenizer(10);
        let strategy = TokenBudget::new(25);
        let out = compact(&messages, &strategy, &tokenizer);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text(), "3");
        assert_eq!(out[1].text(), "4");
    }

    #[test]
    fn token_budget_preserves_leading_system_message_and_counts_it() {
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
        ];
        let tokenizer = FixedTokenizer(10);
        // System (10) + budget for one more message (<=20 total).
        let strategy = TokenBudget::new(20);
        let out = compact(&messages, &strategy, &tokenizer);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, Role::system());
        assert_eq!(out[1].text(), "3");
    }

    #[test]
    fn token_budget_keeps_at_least_the_newest_message_even_if_it_alone_exceeds_budget() {
        let messages = vec![text(Role::user(), "1"), text(Role::assistant(), "2")];
        let tokenizer = FixedTokenizer(100);
        let strategy = TokenBudget::new(1);
        let out = compact(&messages, &strategy, &tokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text(), "2");
    }

    #[test]
    fn token_budget_keeps_everything_when_it_all_fits() {
        let messages = vec![text(Role::user(), "1"), text(Role::assistant(), "2")];
        let tokenizer = FixedTokenizer(1);
        let strategy = TokenBudget::new(1000);
        let out = compact(&messages, &strategy, &tokenizer);
        assert_eq!(out, messages);
    }

    // ---- SelectiveToolResult --------------------------------------------

    #[test]
    fn selective_tool_result_strips_stale_results_and_keeps_recent_ones() {
        let messages = vec![
            text(Role::user(), "ask 1"),
            tool_result_message("c1", "result 1"),
            text(Role::user(), "ask 2"),
            tool_result_message("c2", "result 2"),
            text(Role::user(), "ask 3"),
            tool_result_message("c3", "result 3"),
        ];
        let strategy = SelectiveToolResult::new(1);
        let out = compact(&messages, &strategy, &ApproxTokenizer);

        // The two oldest tool-result messages become empty and are dropped;
        // the newest tool-result message is kept intact.
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].text(), "ask 1");
        assert_eq!(out[1].text(), "ask 2");
        assert_eq!(out[2].text(), "ask 3");
        assert!(has_tool_result(&out[3]));
        assert_eq!(out[3].function_results()[0].call_id, "c3");
    }

    #[test]
    fn selective_tool_result_keeps_text_alongside_a_stripped_tool_result() {
        let mixed = Message::with_contents(
            Role::tool(),
            vec![
                Content::text("some accompanying text"),
                Content::FunctionResult(FunctionResultContent::new("c1", Some(json!("r1")))),
            ],
        );
        let messages = vec![
            mixed,
            tool_result_message("c2", "result 2"),
            tool_result_message("c3", "result 3"),
        ];
        let strategy = SelectiveToolResult::new(2);
        let out = compact(&messages, &strategy, &ApproxTokenizer);

        // First message's tool result is stripped (only the two most recent
        // tool-result-bearing messages are kept intact), but its text survives.
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].text(), "some accompanying text");
        assert!(!has_tool_result(&out[0]));
        assert!(has_tool_result(&out[1]));
        assert!(has_tool_result(&out[2]));
    }

    #[test]
    fn selective_tool_result_noop_when_keep_last_covers_all() {
        let messages = vec![
            tool_result_message("c1", "result 1"),
            tool_result_message("c2", "result 2"),
        ];
        let strategy = SelectiveToolResult::new(5);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out, messages);
    }

    #[test]
    fn selective_tool_result_ignores_messages_without_tool_results() {
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "hi"),
            text(Role::assistant(), "hello"),
        ];
        let strategy = SelectiveToolResult::new(0);
        let out = compact(&messages, &strategy, &ApproxTokenizer);
        assert_eq!(out, messages);
    }

    // ---- CompactionProvider ---------------------------------------------

    #[tokio::test]
    async fn compaction_provider_before_run_replaces_ctx_messages_with_compacted_subset() {
        let provider = CompactionProvider::new(Truncation::new(2));
        let mut ctx = SessionContext::new(vec![]);
        ctx.messages = vec![
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
            text(Role::assistant(), "4"),
        ];
        provider.before_run(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].text(), "3");
        assert_eq!(ctx.messages[1].text(), "4");
    }

    #[tokio::test]
    async fn compaction_provider_with_tokenizer_uses_the_supplied_tokenizer() {
        struct FixedTokenizer(usize);
        impl Tokenizer for FixedTokenizer {
            fn count_tokens(&self, _text: &str) -> usize {
                self.0
            }
        }
        let provider = CompactionProvider::with_tokenizer(TokenBudget::new(25), FixedTokenizer(10));
        let mut ctx = SessionContext::new(vec![]);
        ctx.messages = vec![
            text(Role::user(), "1"),
            text(Role::assistant(), "2"),
            text(Role::user(), "3"),
            text(Role::assistant(), "4"),
        ];
        provider.before_run(&mut ctx).await.unwrap();
        // Budget of 25 with a fixed 10-token cost per message keeps 2 messages.
        assert_eq!(ctx.messages.len(), 2);
        assert_eq!(ctx.messages[0].text(), "3");
        assert_eq!(ctx.messages[1].text(), "4");
    }

    #[tokio::test]
    async fn compaction_provider_after_run_is_a_noop() {
        let provider = CompactionProvider::new(Truncation::new(1));
        provider
            .after_run(&[Message::new(Role::user(), "hi")], &[], None)
            .await
            .unwrap();
    }
}
