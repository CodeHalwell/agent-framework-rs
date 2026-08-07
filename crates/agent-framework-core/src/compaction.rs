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
/// [`OMITTED_TOOL_RESULT`] — on `result` and, for a failed call, on
/// `exception` too, since provider converters render `exception` *instead of*
/// `result` and leaving it would send the original error verbatim — rather
/// than deleted. Deleting it would leave the
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
                        // A failed call carries its payload — often a stack
                        // trace, the bulkiest thing here — in `exception`, and
                        // every provider converter renders `exception` *instead
                        // of* `result`. Replacing only `result` would leave the
                        // original error sent verbatim and the marker ignored,
                        // making this a no-op for exactly the results most worth
                        // compacting. Replaced rather than cleared, so the model
                        // still sees that the call failed.
                        if fr.exception.is_some() {
                            fr.exception = Some(OMITTED_TOOL_RESULT.to_string());
                        }
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
fn drop_orphaned_tool_exchanges(messages: Vec<Message>, pending: &[Message]) -> Vec<Message> {
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
        // `pending` is scanned as a continuation of `messages` (its indices
        // start past the end) so a call here can pair with a result that only
        // arrives later, but it is never itself modified.
        for (mi, message) in messages.iter().chain(pending.iter()).enumerate() {
            for (ci, content) in message.contents.iter().enumerate() {
                match content {
                    Content::FunctionCall(fc) if !fc.call_id.is_empty() => {
                        unanswered
                            .entry(fc.call_id.as_str())
                            .or_default()
                            .push_back((mi, ci));
                    }
                    Content::FunctionResult(fr) if !fr.call_id.is_empty() => {
                        // Nearest *preceding* outstanding call, not the oldest.
                        // With FIFO a result in `pending` paired with a stale
                        // historical call instead of the pending call directly
                        // before it, leaving that one unanswered while the
                        // historical one was kept — two calls, one result.
                        if let Some(call_site) = unanswered
                            .get_mut(fr.call_id.as_str())
                            .and_then(VecDeque::pop_back)
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
        // A tool-role message is never standalone content: its text belongs to
        // the exchange. Stripping its result and keeping the text yields a tool
        // message with no `tool_call_id`, which the OpenAI converter drops
        // entirely (leaving the conversation system-only after all) and which
        // Gemini emits as a bare text part under its `function` role. The
        // complete-exchange fallback below handles these properly.
        .filter(|m| m.role != Role::system() && m.role != Role::tool())
        .find_map(|m| {
            let mut candidate = m.clone();
            // Reasoning is dropped alongside the exchange halves, not kept as
            // standalone content. It renders nothing on its own — Gemini's
            // request builder deliberately skips reasoning parts, and the
            // OpenAI converter has no mapping for it — so a message reduced to
            // reasoning alone would qualify here, return early, and leave the
            // request effectively system-only while bypassing the
            // complete-exchange fallback below.
            candidate.contents.retain(|c| {
                !matches!(
                    c,
                    Content::FunctionCall(_)
                        | Content::FunctionResult(_)
                        | Content::TextReasoning(_)
                )
            });
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

/// Put back a call that the *strategy* dropped but that an incoming result
/// answers.
///
/// Pairing against `pending` stops the repair from stripping a **retained**
/// call, but the strategy runs first and may have excluded the call from
/// `retained` altogether — `SlidingWindow::new(0)` over a history holding
/// `call(c1)` while the run's input holds `result(c1)`, for instance. Nothing
/// downstream can fix that: the call is gone from the retained set, and
/// `pending` is the caller's input, which is not ours to edit. The request
/// would go out with a result answering nothing.
///
/// So the call is reinstated from `original` — reduced to just that content, so
/// no unrelated payload rides back in with it — and appended after the retained
/// messages, which keeps it ahead of the `pending` result that follows.
fn reinstate_calls_answered_by_pending(
    original: &[Message],
    mut retained: Vec<Message>,
    pending: &[Message],
) -> Vec<Message> {
    use std::collections::HashMap;

    // How many results in `pending` want a call, per id.
    // Only results that `pending` does not answer *itself*. A run's input can
    // carry a complete exchange of its own, and it may reuse an id that an
    // older unanswered call also used. Counting every result would reinstate
    // that stale historical call, and the FIFO repair would then pair the
    // pending result with it and leave the pending call unanswered — an
    // invalid exchange assembled out of two valid halves.
    let wanted = unanswered_result_counts(pending);
    if wanted.is_empty() {
        return retained;
    }
    let mut wanted = wanted;
    // Subtract the calls the retained set can already answer with. Counted by
    // *occurrence*, not id membership — a completed pair in `retained` answers
    // nothing further.
    for (_, _, call_id) in unanswered_call_sites(&retained) {
        if let Some(count) = wanted.get_mut(call_id) {
            *count = count.saturating_sub(1);
        }
    }

    // Choose from the calls left unanswered *in the original history*, not from
    // every call sharing the id. Ids are reused: in
    // `call(c1, old) -> result(c1) -> call(c1, new)` the first occurrence is
    // already answered, and an incoming result answers the *new* call. Picking
    // the first id match would replay the old call's name and arguments and
    // attach the new result to it, corrupting the tool history.
    let mut chosen: Vec<(usize, usize)> = Vec::new();
    let mut unanswered_by_id: HashMap<&str, Vec<(usize, usize)>> = HashMap::new();
    for (mi, ci, call_id) in unanswered_call_sites(original) {
        unanswered_by_id.entry(call_id).or_default().push((mi, ci));
    }
    for (call_id, count) in &wanted {
        if *count == 0 {
            continue;
        }
        if let Some(sites) = unanswered_by_id.get(call_id) {
            // The most recent unanswered occurrences are the ones an incoming
            // result answers.
            let take_from = sites.len().saturating_sub(*count);
            chosen.extend_from_slice(&sites[take_from..]);
        }
    }
    // Append in original order so several recovered calls keep their sequence.
    chosen.sort_unstable();
    for (mi, ci) in chosen {
        let mut message = reduced_to(&original[mi], &[ci]);
        // Carry the Gemini-style replay signature across with the call. When it
        // sits on the preceding reasoning content — the supported fallback
        // placement — reducing to the call alone would strip the only copy, and
        // the reasoning message the request builder backfills from is not in
        // the retained set either.
        if let Some(Content::FunctionCall(fc)) = message.contents.first_mut() {
            if fc.protected_data.is_none() {
                fc.protected_data = preceding_reasoning_signature(&original[mi], ci);
            }
        }
        retained.push(message);
    }
    retained
}

/// The replay signature of the reasoning content immediately preceding the
/// content at `index`, if any.
///
/// Only a *contiguous* run of reasoning content counts, matching the request
/// builder's rule that a backfilled signature applies solely to a call directly
/// following its reasoning — any other content in between clears it.
fn preceding_reasoning_signature(message: &Message, index: usize) -> Option<String> {
    for content in message.contents[..index].iter().rev() {
        match content {
            Content::TextReasoning(t) => {
                if let Some(signature) = t.protected_data.as_deref().filter(|s| !s.is_empty()) {
                    return Some(signature.to_string());
                }
            }
            _ => return None,
        }
    }
    None
}

/// How many function results in `messages` no preceding call in `messages`
/// answers, per call id.
///
/// The mirror of [`unanswered_call_sites`], pairing by occurrence for the same
/// reason: ids are reused, so membership alone cannot tell a self-contained
/// exchange from one that reaches back into history.
fn unanswered_result_counts(messages: &[Message]) -> std::collections::HashMap<&str, usize> {
    use std::collections::HashMap;

    let mut available_calls: HashMap<&str, usize> = HashMap::new();
    let mut unanswered: HashMap<&str, usize> = HashMap::new();
    for content in messages.iter().flat_map(|m| m.contents.iter()) {
        match content {
            Content::FunctionCall(fc) if !fc.call_id.is_empty() => {
                *available_calls.entry(fc.call_id.as_str()).or_insert(0) += 1;
            }
            Content::FunctionResult(fr) if !fr.call_id.is_empty() => {
                match available_calls.get_mut(fr.call_id.as_str()) {
                    Some(count) if *count > 0 => *count -= 1,
                    _ => *unanswered.entry(fr.call_id.as_str()).or_insert(0) += 1,
                }
            }
            _ => {}
        }
    }
    unanswered.retain(|_, count| *count > 0);
    unanswered
}

/// The `(message index, content index, call id)` of every function call in
/// `messages` that no later result answers.
///
/// Pairs by occurrence — each result consumes the oldest still-unanswered call
/// sharing its id — because ids are not unique across a conversation.
fn unanswered_call_sites(messages: &[Message]) -> Vec<(usize, usize, &str)> {
    use std::collections::{HashMap, VecDeque};

    let mut unanswered: HashMap<&str, VecDeque<(usize, usize)>> = HashMap::new();
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
                    if let Some(sites) = unanswered.get_mut(fr.call_id.as_str()) {
                        // Same nearest-preceding rule as the orphan repair, so
                        // both agree on which occurrence a result answers.
                        sites.pop_back();
                    }
                }
                _ => {}
            }
        }
    }
    let mut out: Vec<(usize, usize, &str)> = unanswered
        .into_iter()
        .flat_map(|(id, sites)| sites.into_iter().map(move |(mi, ci)| (mi, ci, id)))
        .collect();
    out.sort_unstable();
    out
}

/// Apply the invariants every compaction result must satisfy, whatever
/// strategy produced it: no half tool exchanges, and never system-only.
///
/// `pending` is a list that will be appended *after* this runs and is not ours
/// to modify — for [`CompactionProvider`] that is the run's own input. It
/// participates in pairing so a retained call answered by an incoming result is
/// not mistaken for an orphan, but is never stripped or returned.
///
/// Order matters: the orphan repair runs *first*, because it can itself strip
/// a conversation down to system messages only (a retained tool result whose
/// call fell outside the budget is removed, and it may have been the sole
/// non-system message). Running the minimum-retention check afterwards catches
/// that case too. The reverse order silently leaves a system-only result.
/// Neither pass can undo the other: the repair is a no-op on the orphan-free
/// message the fallback reinstates.
fn finalize_compaction(
    original: &[Message],
    retained: Vec<Message>,
    pending: &[Message],
) -> Vec<Message> {
    let retained = reinstate_calls_answered_by_pending(original, retained, pending);
    let retained = drop_orphaned_tool_exchanges(retained, pending);
    // A non-system message in `pending` already gives the model something to
    // answer, so the minimum-retention rule has nothing to enforce.
    if pending.iter().any(|m| m.role != Role::system()) {
        return retained;
    }
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
    finalize_compaction(messages, strategy.compact(messages, tokenizer), &[])
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
        // Pair against the run's input as well as history. `prepare_request`
        // appends `input_messages` *after* every provider has run, so a
        // declaration-only (frontend) tool call sitting in history is routinely
        // answered by a result that arrives in this run's input. Repairing
        // history alone would see that call as unanswered, drop it, and leave
        // the incoming result unmatched — manufacturing exactly the invalid
        // conversation this pass exists to prevent.
        ctx.messages = finalize_compaction(&ctx.messages, retained, &ctx.input_messages);
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
    fn selective_tool_result_compacts_a_failed_calls_exception_too() {
        // Every provider converter renders `exception` instead of `result`, so
        // replacing only `result` left the original stack trace sent verbatim —
        // a no-op for exactly the results most worth compacting.
        let failed = Message::with_contents(
            Role::tool(),
            vec![Content::FunctionResult(FunctionResultContent {
                call_id: "c1".into(),
                result: None,
                exception: Some("Traceback: ...a very long stack trace...".into()),
            })],
        );
        let messages = vec![
            tool_call_message("c1", "t1"),
            failed,
            tool_call_message("c2", "t2"),
            tool_result_message("c2", "fresh"),
        ];
        let out = compact(&messages, &SelectiveToolResult::new(1), &ApproxTokenizer);

        let compacted = &out[1].function_results()[0];
        // Still visibly a failure, just without the payload.
        assert_eq!(compacted.exception.as_deref(), Some(OMITTED_TOOL_RESULT));
        // The most recent exchange is untouched.
        assert_eq!(out[3].function_results()[0].result, Some(json!("fresh")));
    }

    #[test]
    fn selective_tool_result_does_not_invent_an_exception_on_a_successful_result() {
        let messages = vec![
            tool_call_message("c1", "t1"),
            tool_result_message("c1", "stale"),
            tool_call_message("c2", "t2"),
            tool_result_message("c2", "fresh"),
        ];
        let out = compact(&messages, &SelectiveToolResult::new(1), &ApproxTokenizer);
        assert!(out[1].function_results()[0].exception.is_none());
    }

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
    fn a_call_answered_by_the_runs_input_is_not_dropped_as_an_orphan() {
        // The frontend/declaration-only tool flow: the model's call sits in
        // history, and the caller supplies its result in the *next* run's
        // input. `prepare_request` appends that input after every provider has
        // run, so a repair that only sees history would drop the call and leave
        // the incoming result unmatched — the exact invalid conversation this
        // pass exists to prevent.
        let history = vec![
            text(Role::user(), "what's the weather?"),
            tool_call_message("c1", "get_weather"),
        ];
        let input = vec![tool_result_message("c1", "sunny")];

        let retained = SlidingWindow::new(10).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);

        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"],
            "the call must survive: its result arrives in this run's input"
        );
        // The input itself is never returned or modified by the repair.
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))).is_empty());

        // Contrast, pinning the bug this fixes: without visibility of the
        // pending input the same call *is* dropped as an orphan.
        let retained = SlidingWindow::new(10).compact(&history, &ApproxTokenizer);
        let blind = finalize_compaction(&history, retained, &[]);
        assert!(call_ids_in(&blind, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn a_call_the_strategy_dropped_is_reinstated_when_the_input_answers_it() {
        // Pairing against pending stops a *retained* call being stripped, but
        // the strategy runs first: SlidingWindow(0) excludes the call from
        // history entirely, and the incoming result then answers nothing.
        let history = vec![
            text(Role::user(), "what's the weather?"),
            tool_call_message("c1", "get_weather"),
        ];
        let input = vec![tool_result_message("c1", "sunny")];

        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        assert!(
            call_ids_in(&retained, |c| matches!(c, Content::FunctionCall(_))).is_empty(),
            "precondition: the strategy really did drop the call"
        );

        let out = finalize_compaction(&history, retained, &input);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"],
            "the call must be put back so the incoming result answers something"
        );
    }

    #[test]
    fn a_reused_call_id_reinstates_the_unanswered_occurrence() {
        // `call(c1, old) -> result(c1) -> call(c1, new)`: the first occurrence
        // is already answered, so an incoming result answers the *new* call.
        // Picking the first id match replayed the old call's name and arguments
        // and attached the new result to it.
        let mut old_call = tool_call_message("c1", "get_weather");
        old_call.contents = vec![Content::FunctionCall(
            crate::types::FunctionCallContent::new(
                "c1",
                "get_weather",
                Some(crate::types::FunctionArguments::Raw(
                    "{\"city\":\"old\"}".into(),
                )),
            ),
        )];
        let mut new_call = tool_call_message("c1", "get_weather");
        new_call.contents = vec![Content::FunctionCall(
            crate::types::FunctionCallContent::new(
                "c1",
                "get_weather",
                Some(crate::types::FunctionArguments::Raw(
                    "{\"city\":\"new\"}".into(),
                )),
            ),
        )];
        let history = vec![old_call, tool_result_message("c1", "old answer"), new_call];
        let input = vec![tool_result_message("c1", "new answer")];

        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);

        let calls: Vec<&crate::types::FunctionCallContent> = out
            .iter()
            .flat_map(|m| m.contents.iter())
            .filter_map(Content::as_function_call)
            .collect();
        assert_eq!(calls.len(), 1);
        let args = calls[0].parse_arguments().unwrap();
        assert_eq!(
            args.get("city").and_then(|v| v.as_str()),
            Some("new"),
            "the unanswered (new) occurrence must be the one reinstated"
        );
    }

    #[test]
    fn a_self_contained_pending_exchange_reinstates_nothing() {
        // The run's input carries its own call *and* result, reusing an id that
        // an older unanswered call also used. Counting every pending result
        // pulled the stale historical call back, and the FIFO repair then
        // paired the pending result with it, leaving the pending call
        // unanswered — an invalid exchange assembled from two valid halves.
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
        ];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty(),
            "pending answers itself; nothing should be reinstated, got {out:?}"
        );
    }

    #[test]
    fn a_pending_result_beyond_what_pending_answers_still_reinstates() {
        // Two results, only one answered within pending: the extra one reaches
        // back into history and must still recover its call.
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "first"),
            tool_result_message("c1", "second"),
        ];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"]
        );
    }

    #[test]
    fn a_pending_exchange_pairs_with_its_own_call_not_a_historical_one() {
        // Retained history holds an unanswered c1 call and the input carries its
        // own c1 call + result. Pairing with the *oldest* matching call attached
        // the pending result to the historical call, kept that call, and left
        // the pending call unanswered — two calls, one result.
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
        ];
        let retained = SlidingWindow::new(10).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty(),
            "the unanswered historical call must be stripped, got {out:?}"
        );
    }

    #[test]
    fn a_reinstated_call_keeps_its_reasoning_signature() {
        // The signature sits on the preceding reasoning content (the supported
        // fallback placement). Reducing to the call alone stripped the only
        // copy, and the reasoning message is not retained either, so the
        // request builder had nothing left to backfill from.
        let signed = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(crate::types::TextReasoningContent {
                    text: "thinking".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::FunctionCall(crate::types::FunctionCallContent::new(
                    "c1",
                    "get_weather",
                    None,
                )),
            ],
        );
        let history = vec![signed];
        let input = vec![tool_result_message("c1", "sunny")];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);

        let call = out
            .iter()
            .flat_map(|m| m.contents.iter())
            .find_map(Content::as_function_call)
            .expect("the call is reinstated");
        assert_eq!(call.protected_data.as_deref(), Some("c2ln"));
    }

    #[test]
    fn a_reinstated_call_gains_no_signature_from_unrelated_content() {
        let mut msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(crate::types::TextReasoningContent {
                    text: "thinking".into(),
                    protected_data: Some("c2ln".into()),
                    ..Default::default()
                }),
                Content::text("intervening"),
            ],
        );
        msg.contents.push(Content::FunctionCall(
            crate::types::FunctionCallContent::new("c1", "get_weather", None),
        ));
        let history = vec![msg];
        let input = vec![tool_result_message("c1", "sunny")];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);

        let call = out
            .iter()
            .flat_map(|m| m.contents.iter())
            .find_map(Content::as_function_call)
            .expect("the call is reinstated");
        assert!(call.protected_data.is_none());
    }

    #[test]
    fn a_completed_exchange_in_history_is_not_reinstated() {
        // Nothing is outstanding, so an incoming result for a *fresh* call id
        // pulls back nothing.
        let history = vec![
            tool_call_message("c1", "get_weather"),
            tool_result_message("c1", "sunny"),
        ];
        let input = vec![tool_result_message("c2", "noon")];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn a_reinstated_call_carries_nothing_else_from_its_message() {
        let mut call_msg = tool_call_message("c1", "get_weather");
        call_msg
            .contents
            .push(Content::text("a large unrelated payload"));
        let history = vec![call_msg];
        let input = vec![tool_result_message("c1", "sunny")];

        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].contents.len(), 1, "only the call rides back in");
        assert!(out[0].text().is_empty());
    }

    #[test]
    fn nothing_is_reinstated_when_the_input_carries_no_results() {
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![text(Role::user(), "hello")];
        let retained = SlidingWindow::new(0).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn an_already_retained_call_is_not_reinstated_twice() {
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![tool_result_message("c1", "sunny")];
        let retained = SlidingWindow::new(10).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"]
        );
    }

    #[test]
    fn the_minimum_retention_fallback_never_picks_a_tool_role_message() {
        // A tool message's text belongs to its exchange. Keeping it without the
        // result yields a tool message with no tool_call_id, which the OpenAI
        // converter drops outright (system-only after all) and which Gemini
        // emits as a bare text part under its `function` role.
        let mut result_msg = tool_result_message("c1", "sunny");
        result_msg.contents.push(Content::text("explanatory text"));
        let messages = vec![
            text(Role::system(), "sys"),
            tool_call_message("c1", "get_weather"),
            result_msg,
        ];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        // The complete exchange comes back instead of a bare tool-role text.
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"]
        );
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
            vec!["c1"]
        );
        for m in &out {
            if m.role == Role::tool() {
                assert!(
                    m.contents
                        .iter()
                        .any(|c| matches!(c, Content::FunctionResult(_))),
                    "a tool-role message must carry its result"
                );
            }
        }
    }

    #[test]
    fn a_call_with_no_answer_anywhere_is_still_dropped() {
        let history = vec![tool_call_message("c1", "get_weather")];
        let input = vec![text(Role::user(), "never mind")];
        let retained = SlidingWindow::new(10).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))).is_empty());
    }

    #[test]
    fn pending_input_satisfies_the_minimum_retention_rule() {
        // The run's own input already gives the model something to answer, so
        // no history turn needs reinstating over the limit.
        let history = vec![text(Role::system(), "sys"), text(Role::user(), "older")];
        let input = vec![text(Role::user(), "current question")];
        let retained = Truncation::new(1).compact(&history, &ApproxTokenizer);
        let out = finalize_compaction(&history, retained, &input);
        assert!(out.iter().all(|m| m.role == Role::system()));
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
    fn reasoning_alone_does_not_qualify_as_the_minimum_retained_message() {
        // `[TextReasoning, FunctionCall]` + its result: stripping the call left
        // reasoning behind, which passed the non-empty check and returned early,
        // bypassing the complete-exchange fallback. Reasoning renders nothing
        // standalone, so the request was effectively system-only anyway.
        let signed = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(crate::types::TextReasoningContent {
                    text: "thinking".into(),
                    ..Default::default()
                }),
                Content::FunctionCall(crate::types::FunctionCallContent::new(
                    "c1",
                    "get_weather",
                    None,
                )),
            ],
        );
        let messages = vec![
            text(Role::system(), "sys"),
            signed,
            tool_result_message("c1", "sunny"),
        ];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);

        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionCall(_))),
            vec!["c1"],
            "the complete exchange must be restored, got {out:?}"
        );
        assert_eq!(
            call_ids_in(&out, |c| matches!(c, Content::FunctionResult(_))),
            vec!["c1"]
        );
    }

    #[test]
    fn text_beside_reasoning_still_qualifies() {
        let msg = Message::with_contents(
            Role::assistant(),
            vec![
                Content::TextReasoning(crate::types::TextReasoningContent {
                    text: "thinking".into(),
                    ..Default::default()
                }),
                Content::text("the answer"),
            ],
        );
        let messages = vec![text(Role::system(), "sys"), msg];
        let out = compact(&messages, &Truncation::new(1), &ApproxTokenizer);
        assert_eq!(out.last().unwrap().text(), "the answer");
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
