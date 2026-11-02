//! The message-group model, triggers, composition and LLM summarization.
//!
//! Port of upstream's compaction group model (`group_messages`,
//! .NET `CompactionMessageIndex` / `CompactionMessageGroup` /
//! `CompactionGroupKind`), `CompactionTriggers`, `PipelineCompactionStrategy`
//! and the LLM-backed `SummarizationStrategy`.
//!
//! A [`CompactionMessageIndex`] splits a conversation into atomic groups —
//! a system message, a user message, an assistant text reply, a tool-call
//! exchange (the assistant message requesting calls together with the tool
//! messages answering them), or an earlier summary — numbered by user turn
//! and costed in tokens and bytes. Strategies decide on groups, never on
//! single messages, so a cut cannot separate a call from its result.
//!
//! [`AsyncCompactionStrategy`] is the asynchronous form of
//! [`CompactionStrategy`] that summarization needs; every synchronous strategy
//! is one too, so [`PipelineCompactionStrategy`] and
//! [`TriggeredCompactionStrategy`] compose either kind, and
//! [`CompactionProvider::with_async_strategy`](super::CompactionProvider::with_async_strategy)
//! (or `AgentBuilder::with_async_compaction`) runs them.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{count_message_tokens, ApproxTokenizer, CompactionStrategy, Tokenizer};
use crate::client::ChatClient;
use crate::error::{Error, Result};
use crate::types::{ChatOptions, Content, Message, Role};

/// The [`Message::additional_properties`] key marking a summary message
/// [`SummarizationCompactionStrategy`] wrote, with the ids of the messages it
/// replaced (`{"summary_of_message_ids": [...]}`).
pub const COMPACTION_SUMMARY_KEY: &str = "_compaction_summary";

/// What a [`CompactionMessageGroup`] holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactionGroupKind {
    /// A system message.
    System,
    /// A user message; each starts a turn.
    User,
    /// An assistant reply without tool calls.
    AssistantText,
    /// An assistant message requesting tool calls, with the tool messages
    /// answering them.
    ToolCall,
    /// A summary an earlier compaction wrote.
    Summary,
}

/// An atomic run of messages a strategy keeps or drops as a whole.
#[derive(Debug, Clone)]
pub struct CompactionMessageGroup {
    /// What the group holds.
    pub kind: CompactionGroupKind,
    /// The messages, in conversation order.
    pub messages: Vec<Message>,
    /// The zero-based user turn the group belongs to; `None` for system
    /// groups. Groups before the first user message are in turn 0.
    pub turn_index: Option<usize>,
    /// The group's token cost under the index's tokenizer.
    pub token_count: usize,
    /// The group's size as serialized JSON, in bytes.
    pub byte_count: usize,
    /// Whether a strategy has excluded the group.
    pub is_excluded: bool,
    /// Why it was excluded, if it was.
    pub exclude_reason: Option<String>,
}

impl CompactionMessageGroup {
    /// The number of messages in the group.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }
}

fn is_summary(message: &Message) -> bool {
    message
        .additional_properties
        .contains_key(COMPACTION_SUMMARY_KEY)
}

fn has_tool_call(message: &Message) -> bool {
    message
        .contents
        .iter()
        .any(|c| matches!(c, Content::FunctionCall(_)))
}

fn is_tool_result_message(message: &Message) -> bool {
    message.role == Role::tool()
        || (!message.contents.is_empty()
            && message
                .contents
                .iter()
                .all(|c| matches!(c, Content::FunctionResult(_))))
}

/// A conversation split into [`CompactionMessageGroup`]s. See the
/// [module docs](self).
#[derive(Clone)]
pub struct CompactionMessageIndex {
    groups: Vec<CompactionMessageGroup>,
    /// The tokenizer later groups are costed with; `None` when the index was
    /// built with a borrowed one (the [`ApproxTokenizer`] then costs them).
    tokenizer: Option<Arc<dyn Tokenizer>>,
    raw_message_count: usize,
}

impl std::fmt::Debug for CompactionMessageIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompactionMessageIndex")
            .field("groups", &self.groups)
            .field("raw_message_count", &self.raw_message_count)
            .finish_non_exhaustive()
    }
}

impl CompactionMessageIndex {
    /// Group `messages`, costing each group with `tokenizer`.
    pub fn new(messages: &[Message], tokenizer: Arc<dyn Tokenizer>) -> Self {
        let mut index = Self::counted(messages, &*tokenizer);
        index.tokenizer = Some(tokenizer);
        index
    }

    /// Group `messages`, costing them with a borrowed `tokenizer`.
    pub fn counted(messages: &[Message], tokenizer: &dyn Tokenizer) -> Self {
        let mut index = Self {
            groups: Vec::new(),
            tokenizer: None,
            raw_message_count: messages.len(),
        };
        let mut turn = 0usize;
        let mut seen_user = false;
        let mut i = 0;
        while i < messages.len() {
            let message = &messages[i];
            let (kind, end) = if is_summary(message) {
                (CompactionGroupKind::Summary, i + 1)
            } else if message.role == Role::system() {
                (CompactionGroupKind::System, i + 1)
            } else if message.role == Role::user() && !is_tool_result_message(message) {
                if seen_user {
                    turn += 1;
                }
                seen_user = true;
                (CompactionGroupKind::User, i + 1)
            } else if has_tool_call(message) || is_tool_result_message(message) {
                // The call message, then every tool message answering it.
                let mut end = i + 1;
                while end < messages.len() && is_tool_result_message(&messages[end]) {
                    end += 1;
                }
                (CompactionGroupKind::ToolCall, end)
            } else {
                (CompactionGroupKind::AssistantText, i + 1)
            };
            let turn_index = (kind != CompactionGroupKind::System).then_some(turn);
            let group = Self::make_group(tokenizer, kind, messages[i..end].to_vec(), turn_index);
            index.groups.push(group);
            i = end;
        }
        index
    }

    /// Group `messages` with the default [`ApproxTokenizer`].
    pub fn from_messages(messages: &[Message]) -> Self {
        Self::new(messages, Arc::new(ApproxTokenizer))
    }

    fn make_group(
        tokenizer: &dyn Tokenizer,
        kind: CompactionGroupKind,
        messages: Vec<Message>,
        turn_index: Option<usize>,
    ) -> CompactionMessageGroup {
        let token_count = messages
            .iter()
            .map(|m| count_message_tokens(tokenizer, m))
            .sum();
        let byte_count = messages
            .iter()
            .map(|m| serde_json::to_vec(m).map_or(0, |b| b.len()))
            .sum();
        CompactionMessageGroup {
            kind,
            messages,
            turn_index,
            token_count,
            byte_count,
            is_excluded: false,
            exclude_reason: None,
        }
    }

    fn costed(
        &self,
        kind: CompactionGroupKind,
        messages: Vec<Message>,
        turn_index: Option<usize>,
    ) -> CompactionMessageGroup {
        match &self.tokenizer {
            Some(t) => Self::make_group(&**t, kind, messages, turn_index),
            None => Self::make_group(&ApproxTokenizer, kind, messages, turn_index),
        }
    }

    /// Append a group.
    pub fn add_group(
        &mut self,
        kind: CompactionGroupKind,
        messages: Vec<Message>,
        turn_index: Option<usize>,
    ) {
        let group = self.costed(kind, messages, turn_index);
        self.groups.push(group);
    }

    /// Insert a group at `position` (clamped to the end).
    pub fn insert_group(
        &mut self,
        position: usize,
        kind: CompactionGroupKind,
        messages: Vec<Message>,
        turn_index: Option<usize>,
    ) {
        let group = self.costed(kind, messages, turn_index);
        let position = position.min(self.groups.len());
        self.groups.insert(position, group);
    }

    /// The groups, in conversation order.
    pub fn groups(&self) -> &[CompactionMessageGroup] {
        &self.groups
    }

    /// The groups, mutably (to exclude or include them).
    pub fn groups_mut(&mut self) -> &mut [CompactionMessageGroup] {
        &mut self.groups
    }

    /// Exclude group `position` for `reason`.
    pub fn exclude(&mut self, position: usize, reason: impl Into<String>) {
        if let Some(group) = self.groups.get_mut(position) {
            group.is_excluded = true;
            group.exclude_reason = Some(reason.into());
        }
    }

    /// The tokenizer groups are costed with, when the index owns one.
    pub fn tokenizer(&self) -> Option<&Arc<dyn Tokenizer>> {
        self.tokenizer.as_ref()
    }

    /// Every message, excluded or not.
    pub fn get_all_messages(&self) -> Vec<Message> {
        self.groups
            .iter()
            .flat_map(|g| g.messages.iter().cloned())
            .collect()
    }

    /// The messages of groups not excluded: what is sent to the model.
    pub fn get_included_messages(&self) -> Vec<Message> {
        self.included()
            .flat_map(|g| g.messages.iter().cloned())
            .collect()
    }

    /// The groups of user turn `turn`.
    pub fn get_turn_groups(&self, turn: usize) -> Vec<&CompactionMessageGroup> {
        self.groups
            .iter()
            .filter(|g| g.turn_index == Some(turn))
            .collect()
    }

    fn included(&self) -> impl Iterator<Item = &CompactionMessageGroup> {
        self.groups.iter().filter(|g| !g.is_excluded)
    }

    fn turns<'a>(groups: impl Iterator<Item = &'a CompactionMessageGroup>) -> usize {
        groups
            .filter(|g| g.kind == CompactionGroupKind::User)
            .count()
    }

    /// The number of messages the index was built from.
    pub fn raw_message_count(&self) -> usize {
        self.raw_message_count
    }
    /// All groups.
    pub fn total_group_count(&self) -> usize {
        self.groups.len()
    }
    /// All messages.
    pub fn total_message_count(&self) -> usize {
        self.groups.iter().map(|g| g.messages.len()).sum()
    }
    /// All tokens.
    pub fn total_token_count(&self) -> usize {
        self.groups.iter().map(|g| g.token_count).sum()
    }
    /// All bytes.
    pub fn total_byte_count(&self) -> usize {
        self.groups.iter().map(|g| g.byte_count).sum()
    }
    /// All user turns.
    pub fn total_turn_count(&self) -> usize {
        Self::turns(self.groups.iter())
    }
    /// Groups not excluded.
    pub fn included_group_count(&self) -> usize {
        self.included().count()
    }
    /// Non-system groups not excluded.
    pub fn included_non_system_group_count(&self) -> usize {
        self.included()
            .filter(|g| g.kind != CompactionGroupKind::System)
            .count()
    }
    /// Messages not excluded.
    pub fn included_message_count(&self) -> usize {
        self.included().map(|g| g.messages.len()).sum()
    }
    /// Tokens not excluded.
    pub fn included_token_count(&self) -> usize {
        self.included().map(|g| g.token_count).sum()
    }
    /// Bytes not excluded.
    pub fn included_byte_count(&self) -> usize {
        self.included().map(|g| g.byte_count).sum()
    }
    /// User turns not excluded.
    pub fn included_turn_count(&self) -> usize {
        Self::turns(self.included())
    }
}

/// Decides, from a conversation's [`CompactionMessageIndex`], whether a
/// strategy should run.
pub type CompactionTrigger = Arc<dyn Fn(&CompactionMessageIndex) -> bool + Send + Sync>;

/// Ready-made [`CompactionTrigger`]s (upstream `CompactionTriggers`), all
/// measured over the included groups.
pub struct CompactionTriggers;

impl CompactionTriggers {
    /// Always run.
    pub fn always() -> CompactionTrigger {
        Arc::new(|_| true)
    }
    /// Never run.
    pub fn never() -> CompactionTrigger {
        Arc::new(|_| false)
    }
    /// Run when every trigger fires (and, for none, always).
    pub fn all(triggers: Vec<CompactionTrigger>) -> CompactionTrigger {
        Arc::new(move |index| triggers.iter().all(|t| t(index)))
    }
    /// Run when any trigger fires (and, for none, never).
    pub fn any(triggers: Vec<CompactionTrigger>) -> CompactionTrigger {
        Arc::new(move |index| triggers.iter().any(|t| t(index)))
    }
    /// Run when the conversation costs more than `tokens`.
    pub fn tokens_exceed(tokens: usize) -> CompactionTrigger {
        Arc::new(move |index| index.included_token_count() > tokens)
    }
    /// Run when the conversation costs fewer than `tokens`.
    pub fn tokens_below(tokens: usize) -> CompactionTrigger {
        Arc::new(move |index| index.included_token_count() < tokens)
    }
    /// Run when there are more than `messages` messages.
    pub fn messages_exceed(messages: usize) -> CompactionTrigger {
        Arc::new(move |index| index.included_message_count() > messages)
    }
    /// Run when there are more than `turns` user turns.
    pub fn turns_exceed(turns: usize) -> CompactionTrigger {
        Arc::new(move |index| index.included_turn_count() > turns)
    }
    /// Run when there are more than `groups` groups.
    pub fn groups_exceed(groups: usize) -> CompactionTrigger {
        Arc::new(move |index| index.included_group_count() > groups)
    }
    /// Run when the conversation holds a tool-call exchange.
    pub fn has_tool_calls() -> CompactionTrigger {
        Arc::new(|index| {
            index
                .included()
                .any(|g| g.kind == CompactionGroupKind::ToolCall)
        })
    }
}

/// The asynchronous form of [`CompactionStrategy`], for strategies that
/// call out (summarization). Every [`CompactionStrategy`] is one.
#[async_trait]
pub trait AsyncCompactionStrategy: Send + Sync {
    /// Return the retained messages after compaction.
    async fn compact_async(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message>;
}

#[async_trait]
impl<T: CompactionStrategy> AsyncCompactionStrategy for T {
    async fn compact_async(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        self.compact(messages, tokenizer)
    }
}

/// Runs strategies in order, each on the previous one's output (upstream
/// `PipelineCompactionStrategy`).
#[derive(Clone)]
pub struct PipelineCompactionStrategy {
    strategies: Vec<Arc<dyn AsyncCompactionStrategy>>,
}

impl PipelineCompactionStrategy {
    /// A pipeline of `strategies`.
    pub fn new(strategies: Vec<Arc<dyn AsyncCompactionStrategy>>) -> Self {
        Self { strategies }
    }

    /// The strategies, in order.
    pub fn strategies(&self) -> &[Arc<dyn AsyncCompactionStrategy>] {
        &self.strategies
    }
}

#[async_trait]
impl AsyncCompactionStrategy for PipelineCompactionStrategy {
    async fn compact_async(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let mut current = messages.to_vec();
        for strategy in &self.strategies {
            current = strategy.compact_async(&current, tokenizer).await;
        }
        current
    }
}

/// Runs a strategy only when its trigger fires (the trigger gating of
/// upstream's strategy base class).
#[derive(Clone)]
pub struct TriggeredCompactionStrategy {
    trigger: CompactionTrigger,
    strategy: Arc<dyn AsyncCompactionStrategy>,
}

impl TriggeredCompactionStrategy {
    /// Run `strategy` when `trigger` fires.
    pub fn new(
        trigger: CompactionTrigger,
        strategy: impl AsyncCompactionStrategy + 'static,
    ) -> Self {
        Self {
            trigger,
            strategy: Arc::new(strategy),
        }
    }
}

#[async_trait]
impl AsyncCompactionStrategy for TriggeredCompactionStrategy {
    async fn compact_async(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let index = CompactionMessageIndex::counted(messages, tokenizer);
        if (self.trigger)(&index) {
            self.strategy.compact_async(messages, tokenizer).await
        } else {
            messages.to_vec()
        }
    }
}

/// The default summarization instruction, as upstream.
pub const DEFAULT_SUMMARIZATION_PROMPT: &str = "**Generate a clear and complete summary of the entire conversation in no more than five sentences.**

The summary must always:
- Reflect contributions from both the user and the assistant
- Preserve context to support ongoing dialogue
- Incorporate any previously provided summary
- Emphasize the most relevant and meaningful points

The summary must never:
- Offer critique, correction, interpretation, or speculation
- Highlight errors, misunderstandings, or judgments of accuracy
- Comment on events or ideas not present in the conversation
- Omit any details included in an earlier summary
";

/// The default token budget for a summarization request.
pub const DEFAULT_SUMMARY_INPUT_TOKEN_BUDGET: usize = 8_000;

fn format_summary_message(index: usize, message: &Message) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    for content in &message.contents {
        let rendered = match content {
            Content::FunctionCall(fc) => Some(format!(
                "called {}({})",
                fc.name,
                match &fc.arguments {
                    Some(crate::types::FunctionArguments::Raw(raw)) => raw.clone(),
                    Some(crate::types::FunctionArguments::Object(map)) => {
                        serde_json::to_string(map).unwrap_or_default()
                    }
                    None => String::new(),
                }
            )),
            Content::FunctionResult(fr) => Some(format!(
                "result of {}: {}",
                fr.call_id,
                match &fr.result {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => fr.exception.clone().unwrap_or_default(),
                }
            )),
            _ => None,
        };
        match rendered {
            Some(text) => {
                if !pending.is_empty() {
                    parts.push(pending.join(" "));
                    pending.clear();
                }
                parts.push(text);
            }
            None => {
                if let Content::Text(t) = content {
                    if !t.text.is_empty() {
                        pending.push(&t.text);
                    }
                }
            }
        }
    }
    if !pending.is_empty() {
        parts.push(pending.join(" "));
    }
    let mut text = parts.join("; ");
    if text.is_empty() {
        text = message
            .contents
            .iter()
            .map(|c| {
                serde_json::to_value(c)
                    .ok()
                    .and_then(|v| v.get("type").and_then(Value::as_str).map(str::to_string))
                    .unwrap_or_else(|| "content".into())
            })
            .collect::<Vec<_>>()
            .join(", ");
    }
    format!("{index}. [{}] {text}", message.role)
}

fn format_messages_for_summary(messages: &[Message], start: usize) -> String {
    messages
        .iter()
        .enumerate()
        .map(|(i, m)| format_summary_message(start + i, m))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Summarizes the oldest groups with a chat client once the conversation
/// grows past `target_count + threshold` non-system messages, keeping about
/// `target_count` of the newest (upstream `SummarizationStrategy`).
///
/// **Security:** the summary replaces the original messages and is trusted
/// as an assistant message from then on, so a compromised summarizer could
/// plant instructions that persist (indirect prompt injection). Only point
/// it at a model you trust as much as the primary one.
///
/// A failed or empty summarization leaves the conversation unchanged.
#[derive(Clone)]
pub struct SummarizationCompactionStrategy {
    client: Arc<dyn ChatClient>,
    target_count: usize,
    threshold: usize,
    prompt: String,
    max_summary_input_tokens: Option<usize>,
}

impl std::fmt::Debug for SummarizationCompactionStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SummarizationCompactionStrategy")
            .field("target_count", &self.target_count)
            .field("threshold", &self.threshold)
            .field("max_summary_input_tokens", &self.max_summary_input_tokens)
            .finish_non_exhaustive()
    }
}

impl SummarizationCompactionStrategy {
    /// Summarize with `client`, keeping 4 messages and allowing 2 more
    /// before summarizing, as upstream.
    pub fn new(client: Arc<dyn ChatClient>) -> Self {
        Self {
            client,
            target_count: 4,
            threshold: 2,
            prompt: DEFAULT_SUMMARIZATION_PROMPT.to_string(),
            max_summary_input_tokens: Some(DEFAULT_SUMMARY_INPUT_TOKEN_BUDGET),
        }
    }

    /// Keep about `target_count` non-system messages (at least 1).
    pub fn with_target_count(mut self, target_count: usize) -> Result<Self> {
        if target_count == 0 {
            return Err(Error::Configuration(
                "target_count must be greater than 0.".into(),
            ));
        }
        self.target_count = target_count;
        Ok(self)
    }

    /// Allow `threshold` messages beyond the target before summarizing.
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold;
        self
    }

    /// The summarization instruction.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Cap the summarization request at `tokens` (prompt plus transcript;
    /// whole groups are taken until the next would exceed it), or lift the
    /// cap with `None`.
    pub fn with_max_summary_input_tokens(mut self, tokens: Option<usize>) -> Result<Self> {
        if tokens == Some(0) {
            return Err(Error::Configuration(
                "max_summary_input_tokens must be greater than 0.".into(),
            ));
        }
        self.max_summary_input_tokens = tokens;
        Ok(self)
    }

    /// The groups to summarize, oldest first, within the input budget.
    fn select<'a>(
        &self,
        candidates: &[(usize, &'a CompactionMessageGroup)],
        tokenizer: &dyn Tokenizer,
    ) -> Vec<(usize, &'a CompactionMessageGroup)> {
        let Some(budget) = self.max_summary_input_tokens else {
            return candidates.to_vec();
        };
        let prompt_tokens = tokenizer.count_tokens(&self.prompt);
        let separator = tokenizer.count_tokens("\n");
        let mut selected = Vec::new();
        let (mut count, mut text_tokens) = (0usize, 0usize);
        for &(position, group) in candidates {
            let text = format_messages_for_summary(&group.messages, count + 1);
            let mut candidate = text_tokens + tokenizer.count_tokens(&text);
            if !selected.is_empty() {
                candidate += separator;
            }
            if prompt_tokens + candidate > budget {
                if selected.is_empty() {
                    continue;
                }
                break;
            }
            selected.push((position, group));
            count += group.messages.len();
            text_tokens = candidate;
        }
        selected
    }
}

#[async_trait]
impl AsyncCompactionStrategy for SummarizationCompactionStrategy {
    async fn compact_async(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message> {
        let index = CompactionMessageIndex::counted(messages, tokenizer);
        let non_system: Vec<(usize, &CompactionMessageGroup)> = index
            .groups()
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.is_excluded && g.kind != CompactionGroupKind::System)
            .collect();
        let count: usize = non_system.iter().map(|(_, g)| g.messages.len()).sum();
        if count <= self.target_count + self.threshold {
            return messages.to_vec();
        }
        // Keep whole groups from the newest back until the target is met.
        let mut keep = std::collections::HashSet::new();
        let mut retained = 0usize;
        for (position, group) in non_system.iter().rev() {
            if retained >= self.target_count && !keep.is_empty() {
                break;
            }
            keep.insert(*position);
            retained += group.messages.len();
        }
        let candidates: Vec<_> = non_system
            .iter()
            .filter(|(p, _)| !keep.contains(p))
            .copied()
            .collect();
        let selected = self.select(&candidates, tokenizer);
        if selected.is_empty() {
            if !candidates.is_empty() {
                tracing::warn!(
                    "skipping summarization compaction: no complete message group fits within max_summary_input_tokens"
                );
            }
            return messages.to_vec();
        }
        let to_summarize: Vec<Message> = selected
            .iter()
            .flat_map(|(_, g)| g.messages.iter().cloned())
            .collect();
        let request = vec![
            Message::system(self.prompt.clone()),
            Message::user(format_messages_for_summary(&to_summarize, 1)),
        ];
        let summary = match self
            .client
            .get_response(request, ChatOptions::default())
            .await
        {
            Ok(response) => response.text().trim().to_string(),
            Err(e) => {
                tracing::warn!(error = %e, "skipping summarization compaction: summary generation failed");
                return messages.to_vec();
            }
        };
        if summary.is_empty() {
            tracing::warn!("skipping summarization compaction: the summarizer returned no text");
            return messages.to_vec();
        }
        let summarized: std::collections::HashSet<usize> =
            selected.iter().map(|(p, _)| *p).collect();
        let mut summary_message = Message::assistant(summary);
        summary_message.message_id = Some(format!("summary_{}", messages.len()));
        let ids: Vec<&str> = to_summarize
            .iter()
            .filter_map(|m| m.message_id.as_deref())
            .collect();
        summary_message.additional_properties.insert(
            COMPACTION_SUMMARY_KEY.into(),
            json!({"summary_of_message_ids": ids}),
        );
        let mut origins: Vec<String> = Vec::new();
        for m in &to_summarize {
            if let Some(found) = m
                .additional_properties
                .get(crate::memory::ATTRIBUTION_KEY)
                .and_then(|a| a.get("origin_session_ids"))
                .and_then(Value::as_array)
            {
                for id in found.iter().filter_map(Value::as_str) {
                    if !origins.iter().any(|o| o == id) {
                        origins.push(id.to_string());
                    }
                }
            }
        }
        if !origins.is_empty() {
            summary_message.additional_properties.insert(
                crate::memory::ATTRIBUTION_KEY.into(),
                json!({"origin_session_ids": origins}),
            );
        }
        let first = *summarized.iter().min().expect("non-empty");
        let mut out = Vec::with_capacity(messages.len());
        for (position, group) in index.groups().iter().enumerate() {
            if position == first {
                out.push(summary_message.clone());
            }
            if !summarized.contains(&position) {
                out.extend(group.messages.iter().cloned());
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::ChatStream;
    use crate::types::{ChatResponse, FunctionCallContent, FunctionResultContent};
    use std::sync::Mutex;

    fn conversation() -> Vec<Message> {
        vec![
            Message::system("be brief"),
            Message::user("weather?"),
            Message::with_contents(
                Role::assistant(),
                vec![Content::FunctionCall(FunctionCallContent::new(
                    "c1", "weather", None,
                ))],
            ),
            Message::with_contents(
                Role::tool(),
                vec![Content::FunctionResult(FunctionResultContent::new(
                    "c1",
                    Some(json!("sunny")),
                ))],
            ),
            Message::assistant("It is sunny."),
            Message::user("thanks"),
            Message::assistant("You're welcome."),
        ]
    }

    #[test]
    fn messages_group_into_atomic_kinds_and_turns() {
        let index = CompactionMessageIndex::from_messages(&conversation());
        let kinds: Vec<_> = index.groups().iter().map(|g| g.kind).collect();
        use CompactionGroupKind::*;
        assert_eq!(
            kinds,
            [System, User, ToolCall, AssistantText, User, AssistantText]
        );
        assert_eq!(
            index.groups()[2].message_count(),
            2,
            "call and result together"
        );
        let turns: Vec<_> = index.groups().iter().map(|g| g.turn_index).collect();
        assert_eq!(turns, [None, Some(0), Some(0), Some(0), Some(1), Some(1)]);
        assert_eq!(index.total_turn_count(), 2);
        assert_eq!(index.get_turn_groups(1).len(), 2);
        assert_eq!(index.total_message_count(), 7);
        assert_eq!(index.raw_message_count(), 7);
        assert!(index.total_token_count() > 0);
        assert!(index.total_byte_count() > 0);
    }

    #[test]
    fn excluding_a_group_changes_the_included_view_only() {
        let mut index = CompactionMessageIndex::from_messages(&conversation());
        index.exclude(2, "old tool output");
        assert_eq!(index.included_message_count(), 5);
        assert_eq!(index.total_message_count(), 7);
        assert_eq!(index.included_non_system_group_count(), 4);
        assert_eq!(
            index.groups()[2].exclude_reason.as_deref(),
            Some("old tool output")
        );
        assert_eq!(index.get_included_messages().len(), 5);
        index.insert_group(
            0,
            CompactionGroupKind::Summary,
            vec![Message::assistant("s")],
            None,
        );
        assert_eq!(index.groups()[0].kind, CompactionGroupKind::Summary);
    }

    #[test]
    fn triggers_measure_the_included_conversation() {
        let index = CompactionMessageIndex::from_messages(&conversation());
        assert!(CompactionTriggers::always()(&index));
        assert!(!CompactionTriggers::never()(&index));
        assert!(CompactionTriggers::messages_exceed(6)(&index));
        assert!(!CompactionTriggers::messages_exceed(7)(&index));
        assert!(CompactionTriggers::turns_exceed(1)(&index));
        assert!(CompactionTriggers::groups_exceed(5)(&index));
        assert!(CompactionTriggers::has_tool_calls()(&index));
        assert!(CompactionTriggers::tokens_below(1_000_000)(&index));
        assert!(!CompactionTriggers::tokens_exceed(1_000_000)(&index));
        assert!(CompactionTriggers::all(vec![
            CompactionTriggers::always(),
            CompactionTriggers::has_tool_calls()
        ])(&index));
        assert!(!CompactionTriggers::any(vec![])(&index));
    }

    #[tokio::test]
    async fn pipelines_and_triggers_compose_strategies() {
        let messages = conversation();
        let pipeline = PipelineCompactionStrategy::new(vec![
            Arc::new(super::super::SlidingWindow::new(4)),
            Arc::new(super::super::Truncation::new(3)),
        ]);
        let out = pipeline.compact_async(&messages, &ApproxTokenizer).await;
        assert!(out.len() <= 4);
        let gated = TriggeredCompactionStrategy::new(
            CompactionTriggers::messages_exceed(100),
            super::super::Truncation::new(1),
        );
        assert_eq!(
            gated.compact_async(&messages, &ApproxTokenizer).await.len(),
            7
        );
    }

    #[derive(Default)]
    struct Summarizer {
        seen: Mutex<Vec<Vec<Message>>>,
        fail: bool,
    }

    #[async_trait]
    impl ChatClient for Summarizer {
        async fn get_response(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            if self.fail {
                return Err(Error::service("down"));
            }
            Ok(ChatResponse::from_text(
                "They asked about weather; it was sunny.",
            ))
        }
        async fn get_streaming_response(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!("not used")
        }
    }

    #[tokio::test]
    async fn summarization_replaces_the_oldest_groups_with_a_summary() {
        let client = Arc::new(Summarizer::default());
        let strategy = SummarizationCompactionStrategy::new(client.clone())
            .with_target_count(2)
            .unwrap()
            .with_threshold(0);
        let out = strategy
            .compact_async(&conversation(), &ApproxTokenizer)
            .await;
        // System kept, then the summary, then the newest groups (≥ 2 messages).
        assert_eq!(out[0].role, Role::system());
        assert!(out[1]
            .additional_properties
            .contains_key(COMPACTION_SUMMARY_KEY));
        assert_eq!(out[1].text(), "They asked about weather; it was sunny.");
        let rest: Vec<String> = out[2..].iter().map(Message::text).collect();
        assert_eq!(rest, ["thanks", "You're welcome."]);
        let request = client.seen.lock().unwrap()[0].clone();
        assert_eq!(request[0].role, Role::system());
        let transcript = request[1].text();
        assert!(transcript.contains("1. [user] weather?"), "{transcript}");
        assert!(transcript.contains("called weather"), "{transcript}");

        // Below the trigger, nothing is summarized.
        let quiet = SummarizationCompactionStrategy::new(client.clone());
        assert_eq!(
            quiet
                .compact_async(&conversation()[..3], &ApproxTokenizer)
                .await
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn a_failed_summary_leaves_the_conversation_unchanged() {
        let client = Arc::new(Summarizer {
            fail: true,
            ..Default::default()
        });
        let strategy = SummarizationCompactionStrategy::new(client)
            .with_target_count(1)
            .unwrap()
            .with_threshold(0);
        let out = strategy
            .compact_async(&conversation(), &ApproxTokenizer)
            .await;
        assert_eq!(out.len(), 7);
        assert!(
            SummarizationCompactionStrategy::new(Arc::new(Summarizer::default()))
                .with_target_count(0)
                .is_err()
        );
    }

    #[tokio::test]
    async fn an_earlier_summary_is_its_own_group() {
        let client = Arc::new(Summarizer::default());
        let strategy = SummarizationCompactionStrategy::new(client)
            .with_target_count(2)
            .unwrap()
            .with_threshold(0);
        let once = strategy
            .compact_async(&conversation(), &ApproxTokenizer)
            .await;
        let index = CompactionMessageIndex::from_messages(&once);
        assert_eq!(index.groups()[1].kind, CompactionGroupKind::Summary);
    }
}
