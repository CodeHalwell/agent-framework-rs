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

/// Replace the payload of `Content::FunctionResult` (tool-result) content in
/// all but the last `keep_last` messages that carry tool results — they are the
/// bulkiest and least useful once stale. Text and other content is left intact.
///
/// The result content itself is **kept**, with its payload swapped for
/// [`OMITTED_TOOL_RESULT`], rather than deleted. Deleting it would leave the
/// matching assistant `tool_calls` entry unanswered, which providers reject
/// outright — so the size win would come at the cost of a 400 on the next
/// request. Mirrors the intent of upstream's `ToolResultCompactionStrategy`,
/// which likewise *replaces* stale tool groups with a compact stand-in instead
/// of removing them; upstream summarizes the group with an LLM, while this
/// port (which has no summarizing strategy) substitutes a fixed marker.
#[derive(Debug, Clone, Copy)]
pub struct SelectiveToolResult {
    pub keep_last: usize,
}

impl SelectiveToolResult {
    pub fn new(keep_last: usize) -> Self {
        Self { keep_last }
    }
}

/// Stand-in payload left in place of a tool result this strategy compacts.
///
/// The result *content* is kept (only its payload is replaced) so the exchange
/// stays paired with its function call — see [`SelectiveToolResult`].
pub const OMITTED_TOOL_RESULT: &str = "[tool result omitted by compaction]";

impl CompactionStrategy for SelectiveToolResult {
    fn compact(&self, messages: &[Message], _tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let tool_result_count = messages.iter().filter(|m| has_tool_result(m)).count();
        let mut strip_budget = tool_result_count.saturating_sub(self.keep_last);

        let mut out = Vec::with_capacity(messages.len());
        for message in messages {
            if has_tool_result(message) && strip_budget > 0 {
                strip_budget -= 1;
                let mut compacted = message.clone();
                for content in &mut compacted.contents {
                    if let Content::FunctionResult(fr) = content {
                        fr.result =
                            Some(serde_json::Value::String(OMITTED_TOOL_RESULT.to_string()));
                    }
                }
                out.push(compacted);
            } else {
                out.push(message.clone());
            }
        }
        out
    }
}

/// Strip function-call / function-result contents that lost their counterpart,
/// dropping any message left empty by the strip.
///
/// Compaction cuts a message list at an arbitrary point, so a strategy can
/// easily retain one half of a tool exchange: [`TokenBudget`] drops an
/// expensive call-bearing message while keeping its cheap result, and
/// [`SelectiveToolResult`] strips old results while their calls stay put.
/// Either half alone is not merely wasteful — it is *invalid*. Providers
/// require every assistant `tool_calls` entry to be answered by a matching
/// tool message and reject a tool message that answers nothing, so an orphan
/// turns the next request into a 400 rather than a slightly worse completion.
///
/// Mirrors the invariant upstream added in "Keep call and result occurrences
/// atomic in compaction" (#7406). Upstream enforces it by linking call and
/// result into one indivisible span before any strategy runs; this port has no
/// span/group model, so it enforces the same invariant as a repair pass over
/// the retained set — the observable guarantee (never emit a half-exchange) is
/// identical.
///
/// Note this *drops* an unmatched call rather than replacing it with a summary
/// the way upstream's `ToolResultCompactionStrategy` does; this port has no
/// LLM-summarizing compaction strategy to build that summary with.
fn drop_orphaned_tool_exchanges(messages: Vec<Message>) -> Vec<Message> {
    use std::collections::{HashMap, HashSet, VecDeque};

    // Pair by *occurrence*, not by id membership. Call ids are not guaranteed
    // unique across a conversation — providers may reuse one for a later
    // invocation, and some surfaces derive the id from the tool name — so
    // comparing sets of ids would call `[call(c1), result(c1), call(c1)]`
    // balanced and leave the second call unanswered. Each result is instead
    // matched to the oldest still-unanswered call sharing its id; whatever is
    // left unmatched on either side is an orphan.
    let paired: HashSet<(usize, usize)> = {
        let mut unanswered: HashMap<&str, VecDeque<(usize, usize)>> = HashMap::new();
        let mut paired = HashSet::new();
        for (mi, message) in messages.iter().enumerate() {
            for (ci, content) in message.contents.iter().enumerate() {
                match content {
                    Content::FunctionCall(fc) if !fc.call_id.is_empty() => {
                        unanswered
                            .entry(fc.call_id.as_str())
                            .or_default()
                            .push_back((mi, ci));
                    }
                    Content::FunctionResult(fr) if !fr.call_id.is_empty() => {
                        if let Some(call_site) = unanswered
                            .get_mut(fr.call_id.as_str())
                            .and_then(VecDeque::pop_front)
                        {
                            paired.insert(call_site);
                            paired.insert((mi, ci));
                        }
                    }
                    _ => {}
                }
            }
        }
        paired
    };

    // Fast path: nothing is orphaned, so the (common) balanced conversation is
    // returned without rebuilding it.
    let has_orphan = messages.iter().enumerate().any(|(mi, message)| {
        message.contents.iter().enumerate().any(|(ci, content)| {
            matches!(
                content,
                Content::FunctionCall(_) | Content::FunctionResult(_)
            ) && !paired.contains(&(mi, ci))
        })
    });
    if !has_orphan {
        return messages;
    }

    let mut out = Vec::with_capacity(messages.len());
    for (mi, mut message) in messages.into_iter().enumerate() {
        let had_contents = !message.contents.is_empty();
        let mut ci = 0;
        message.contents.retain(|content| {
            let keep = !matches!(
                content,
                Content::FunctionCall(_) | Content::FunctionResult(_)
            ) || paired.contains(&(mi, ci));
            ci += 1;
            keep
        });
        // A message that carried only an orphaned half is dropped outright; one
        // that was already empty is left alone (not this pass's business).
        if had_contents && message.contents.is_empty() {
            continue;
        }
        out.push(message);
    }
    out
}

/// Ensure compaction never reduces a conversation to system messages alone.
///
/// A budget smaller than the system prefix leaves [`Truncation`] and
/// [`SlidingWindow`] returning only system messages — a "conversation" with no
/// turn for the model to answer, which is useless rather than merely short.
/// Mirrors upstream's `_minimum_retained_group_ids` (#7219): the most recent
/// non-system message is retained even when that pushes the result back over
/// the limit.
///
/// The fallback is stripped of function-call/result contents before being
/// reinstated, since its counterpart is by definition not in the retained set —
/// so this can never manufacture the orphan
/// [`drop_orphaned_tool_exchanges`] exists to remove.
fn ensure_non_system_message(original: &[Message], mut retained: Vec<Message>) -> Vec<Message> {
    if retained.iter().any(|m| m.role != Role::system()) {
        return retained;
    }
    // Prefer ordinary content: the latest non-system message with something in
    // it besides a tool exchange. Its call/result contents are stripped, since
    // their counterparts are by definition not in the retained set — so this
    // can never manufacture the orphan `drop_orphaned_tool_exchanges` removes.
    let plain = original
        .iter()
        .rev()
        .filter(|m| m.role != Role::system())
        .find_map(|m| {
            let mut candidate = m.clone();
            candidate
                .contents
                .retain(|c| !matches!(c, Content::FunctionCall(_) | Content::FunctionResult(_)));
            (!candidate.contents.is_empty()).then_some(candidate)
        });
    if let Some(plain) = plain {
        retained.push(plain);
        return retained;
    }
    // Nothing but a tool exchange, then — a conversation whose only non-system
    // content is a call and its result. Stripping both halves (as the branch
    // above does) would leave the result system-only after all, so retain the
    // latest *complete* exchange instead: both halves together are valid, and
    // they give the model something to answer.
    retained.extend(latest_complete_tool_exchange(original));
    retained
}

/// The most recent complete call/result pair, as one or two messages reduced to
/// just that pair's contents (one when both halves share a message).
///
/// Returns empty when no result has a preceding unanswered call — there is no
/// complete exchange to reinstate, and a half is worse than nothing.
fn latest_complete_tool_exchange(messages: &[Message]) -> Vec<Message> {
    use std::collections::{HashMap, VecDeque};

    let mut unanswered: HashMap<&str, VecDeque<(usize, usize)>> = HashMap::new();
    let mut last_pair: Option<((usize, usize), (usize, usize))> = None;
    for (mi, message) in messages.iter().enumerate() {
        if message.role == Role::system() {
            continue;
        }
        for (ci, content) in message.contents.iter().enumerate() {
            match content {
                Content::FunctionCall(fc) if !fc.call_id.is_empty() => {
                    unanswered
                        .entry(fc.call_id.as_str())
                        .or_default()
                        .push_back((mi, ci));
                }
                Content::FunctionResult(fr) if !fr.call_id.is_empty() => {
                    if let Some(call_site) = unanswered
                        .get_mut(fr.call_id.as_str())
                        .and_then(VecDeque::pop_front)
                    {
                        last_pair = Some((call_site, (mi, ci)));
                    }
                }
                _ => {}
            }
        }
    }
    let Some(((call_mi, call_ci), (result_mi, result_ci))) = last_pair else {
        return Vec::new();
    };
    // A result always pairs with a call that came before it, so emitting the
    // call's message first preserves the original order.
    if call_mi == result_mi {
        vec![reduced_to(&messages[call_mi], &[call_ci, result_ci])]
    } else {
        vec![
            reduced_to(&messages[call_mi], &[call_ci]),
            reduced_to(&messages[result_mi], &[result_ci]),
        ]
    }
}

/// Clone `message` keeping only the contents at `keep` (content indices).
fn reduced_to(message: &Message, keep: &[usize]) -> Message {
    let mut out = message.clone();
    let mut index = 0;
    out.contents.retain(|_| {
        let keep_this = keep.contains(&index);
        index += 1;
        keep_this
    });
    out
}

/// Apply the invariants every compaction result must satisfy, whatever
/// strategy produced it: no half tool exchanges, and never system-only.
///
/// Order matters: the orphan repair runs *first*, because it can itself strip
/// a conversation down to system messages only (a retained tool result whose
/// call fell outside the budget is removed, and it may have been the sole
/// non-system message). Running the minimum-retention check afterwards catches
/// that case too. The reverse order silently leaves a system-only result.
/// Neither pass can undo the other: the repair is a no-op on the orphan-free
/// message the fallback reinstates.
fn finalize_compaction(original: &[Message], retained: Vec<Message>) -> Vec<Message> {
    let retained = drop_orphaned_tool_exchanges(retained);
    ensure_non_system_message(original, retained)
}

/// Compact `messages` with `strategy` and `tokenizer`.
///
/// This is the supported entry point: it runs the strategy and then enforces
/// the invariants in [`finalize_compaction`]. Calling
/// [`CompactionStrategy::compact`] directly gives the strategy's raw output
/// without them.
pub fn compact(
    messages: &[Message],
    strategy: &dyn CompactionStrategy,
    tokenizer: &dyn Tokenizer,
) -> Vec<Message> {
    finalize_compaction(messages, strategy.compact(messages, tokenizer))
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
        let retained = self.strategy.compact(&ctx.messages, &*self.tokenizer);
        ctx.messages = finalize_compaction(&ctx.messages, retained);
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

    fn tool_call_message(call_id: &str, name: &str) -> Message {
        Message::with_contents(
            Role::assistant(),
            vec![Content::FunctionCall(
                crate::types::FunctionCallContent::new(call_id, name, None),
            )],
        )
    }

    fn call_ids_in(messages: &[Message], f: fn(&Content) -> bool) -> Vec<String> {
        messages
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter(|c| f(c))
            .filter_map(|c| match c {
                Content::FunctionCall(fc) => Some(fc.call_id.clone()),
                Content::FunctionResult(fr) => Some(fr.call_id.clone()),
                _ => None,
            })
            .collect()
    }

    // ---- compaction invariants (upstream #7406 / #7219) --------------------

    #[test]
    fn selective_tool_result_compacts_the_payload_without_orphaning_the_call() {
        // Deleting the stale result would leave call_1's assistant `tool_calls`
        // entry unanswered, which providers reject with a 400. The result
        // content stays; only its payload is replaced.
        let messages = vec![
            tool_call_message("call_1", "get_weather"),
            tool_result_message("call_1", "a very long stale payload"),
            tool_call_message("call_2", "get_time"),
            tool_result_message("call_2", "noon"),
        ];
        let out = compact(&messages, &SelectiveToolResult::new(1), &ApproxTokenizer);

        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["call_1", "call_2"]
        );
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
            vec!["call_1", "call_2"]
        );
        assert_eq!(
            out[1].function_results()[0].result,
            Some(json!(OMITTED_TOOL_RESULT))
        );
        // The most recent exchange keeps its real payload.
        assert_eq!(out[3].function_results()[0].result, Some(json!("noon")));
    }

    #[test]
    fn a_budget_cut_between_call_and_result_orphans_neither() {
        let mut call_msg = tool_call_message("call_1", "get_weather");
        call_msg
            .contents
            .push(Content::text("let me look that up for you right now"));
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "hi"),
            call_msg,
            tool_result_message("call_1", "sunny"),
        ];
        let out = compact(&messages, &TokenBudget::new(4), &ApproxTokenizer);
        // The expensive call-bearing message fell outside the budget, so its
        // cheap result must not survive alone answering nothing.
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))).is_empty());
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn a_reused_call_id_is_paired_by_occurrence_not_by_id() {
        // Call ids are not guaranteed unique across a conversation. Comparing
        // *sets* of ids called this balanced (both sides are {c1}) and returned
        // early, leaving the second call unanswered.
        let messages = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
            tool_call_message("c1", "get_weather"),
        ];
        let out = compact(&messages, &SlidingWindow::new(10), &ApproxTokenizer);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"],
            "the second, unanswered c1 call must be dropped"
        );
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
            vec!["c1"]
        );
    }

    #[test]
    fn two_full_exchanges_reusing_one_call_id_both_survive() {
        let messages = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "rainy"),
        ];
        let out = compact(&messages, &SlidingWindow::new(10), &ApproxTokenizer);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn a_result_preceding_its_call_is_not_treated_as_paired() {
        // A result can only answer a call that came before it.
        let messages = vec![
            tool_result_message("c1", "sunny"),
            tool_call_message("c1", "get_weather"),
        ];
        let out = compact(&messages, &SlidingWindow::new(10), &ApproxTokenizer);
        assert!(out.is_empty(), "both halves are orphans, got {out:?}");
    }

    #[test]
    fn an_intact_tool_exchange_is_left_alone() {
        let messages = vec![
            tool_call_message("call_1", "get_weather"),
            tool_result_message("call_1", "sunny"),
        ];
        let out = compact(&messages, &SlidingWindow::new(10), &ApproxTokenizer);
        assert_eq!(out.len(), 2);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["call_1"]
        );
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
            vec!["call_1"]
        );
    }

    #[test]
    fn an_orphan_strip_keeps_the_rest_of_its_message() {
        // Only the orphaned call content is removed; sibling text survives and
        // the message itself is not dropped.
        let mut call_msg = tool_call_message("call_1", "get_weather");
        call_msg.contents.push(Content::text("checking now"));
        let out = compact(&[call_msg], &SlidingWindow::new(10), &ApproxTokenizer);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text(), "checking now");
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn compaction_never_returns_system_messages_only() {
        // A budget smaller than the system prefix used to leave a
        // "conversation" with no turn for the model to answer.
        let messages = vec![
            text(Role::system(), "sys"),
            text(Role::user(), "hello"),
            text(Role::assistant(), "hi"),
        ];
        for out in [
            compact(&messages, &Truncation::new(1), &ApproxTokenizer),
            compact(&messages, &SlidingWindow::new(0), &ApproxTokenizer),
        ] {
            assert!(
                out.iter().any(|m| m.role != Role::system()),
                "expected a non-system message to be retained, got {:?}",
                out.iter().map(|m| m.role.as_str()).collect::<Vec<_>>()
            );
            // Upstream accepts exceeding the limit rather than emitting a
            // useless projection; the retained turn is the most recent one.
            assert_eq!(out.last().unwrap().text(), "hi");
        }
    }

    #[test]
    fn the_minimum_retained_message_is_never_a_half_exchange() {
        // The only non-system messages are a tool exchange, so the fallback
        // must reinstate the call-bearing message's *text*, not the orphan.
        let mut call_msg = tool_call_message("call_1", "get_weather");
        call_msg.contents.push(Content::text("checking now"));
        let messages = vec![
            text(Role::system(), "sys"),
            call_msg,
            tool_result_message("call_1", "sunny"),
        ];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        assert!(out.iter().any(|m| m.role != Role::system()));
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))).is_empty());
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn a_conversation_of_only_a_tool_exchange_retains_it_whole() {
        // Stripping both halves (the ordinary-content fallback) would leave the
        // result system-only after all, so the complete exchange is reinstated
        // atomically instead.
        let messages = vec![
            text(Role::system(), "sys"),
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
        ];
        for out in [
            compact(&messages, &Truncation::new(1), &ApproxTokenizer),
            compact(&messages, &SlidingWindow::new(0), &ApproxTokenizer),
        ] {
            assert!(
                out.iter().any(|m| m.role != Role::system()),
                "expected the tool exchange to be retained, got {:?}",
                out.iter().map(|m| m.role.as_str()).collect::<Vec<_>>()
            );
            assert_eq!(
                call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
                vec!["c1"]
            );
            assert_eq!(
                call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
                vec!["c1"]
            );
        }
    }

    #[test]
    fn the_latest_complete_exchange_is_the_one_reinstated() {
        let messages = vec![
            text(Role::system(), "sys"),
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
            tool_call_message("c2", "get_time"),
            tool_result_message("c2", "noon"),
        ];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c2"]
        );
    }

    #[test]
    fn an_incomplete_tool_exchange_is_not_reinstated() {
        // A half exchange is worse than nothing: there is no complete pair to
        // fall back to, so the result stays system-only rather than becoming
        // invalid.
        let messages = vec![
            text(Role::system(), "sys"),
            tool_call_message("c1", "get_weather"),
        ];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        assert!(out.iter().all(|m| m.role == Role::system()));
    }

    #[test]
    fn an_all_system_conversation_stays_all_system() {
        // Nothing to reinstate: the fallback must not invent a turn.
        let messages = vec![text(Role::system(), "a"), text(Role::system(), "b")];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        assert!(out.iter().all(|m| m.role == Role::system()));
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
    fn approx_tokenizer_does_not_inflate_non_ascii_text() {
        // Upstream counted tokens off a JSON serialization with ensure_ascii=True,
        // so CJK text was measured as inflated `\uXXXX` escapes rather than the
        // characters the model actually sees (#7124). This port counts the
        // characters directly, so the escape inflation never existed here — pin
        // that: 4 CJK characters cost the same as 4 ASCII ones, not 6x more.
        let t = ApproxTokenizer;
        assert_eq!(t.count_tokens("日本語訳"), t.count_tokens("abcd"));

        let msg = Message::with_contents(Role::user(), vec![Content::text("日本語訳")]);
        assert_eq!(count_message_tokens(&t, &msg), 1);
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
            tool_call_message("c1", "t1"),
            tool_result_message("c1", "result 1"),
            tool_call_message("c2", "t2"),
            tool_result_message("c2", "result 2"),
            tool_call_message("c3", "t3"),
            tool_result_message("c3", "result 3"),
        ];
        let strategy = SelectiveToolResult::new(1);
        let out = compact(&messages, &strategy, &ApproxTokenizer);

        // Every message survives — the two oldest results keep their content
        // (so their calls stay answered) with only the payload replaced.
        assert_eq!(out.len(), 6);
        assert_eq!(
            out[1].function_results()[0].result,
            Some(json!(OMITTED_TOOL_RESULT))
        );
        assert_eq!(
            out[3].function_results()[0].result,
            Some(json!(OMITTED_TOOL_RESULT))
        );
        assert_eq!(out[5].function_results()[0].result, Some(json!("result 3")));
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
            tool_call_message("c1", "t1"),
            mixed,
            tool_call_message("c2", "t2"),
            tool_result_message("c2", "result 2"),
            tool_call_message("c3", "t3"),
            tool_result_message("c3", "result 3"),
        ];
        let strategy = SelectiveToolResult::new(2);
        let out = compact(&messages, &strategy, &ApproxTokenizer);

        // The first message's tool-result payload is compacted (only the two
        // most recent tool-result-bearing messages keep theirs), but its
        // accompanying text survives untouched alongside it.
        assert_eq!(out.len(), 6);
        assert_eq!(out[1].text(), "some accompanying text");
        assert_eq!(
            out[1].function_results()[0].result,
            Some(json!(OMITTED_TOOL_RESULT))
        );
        assert_eq!(out[3].function_results()[0].result, Some(json!("result 2")));
        assert_eq!(out[5].function_results()[0].result, Some(json!("result 3")));
    }

    #[test]
    fn selective_tool_result_noop_when_keep_last_covers_all() {
        let messages = vec![
            tool_call_message("c1", "t1"),
            tool_result_message("c1", "result 1"),
            tool_call_message("c2", "t2"),
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
