//! Server-Sent Events parsing for Azure AI Foundry agent runs.
//!
//! Two layers: [`SseDecoder`] turns a raw byte/char stream into discrete
//! [`SseEvent`]s (an `event:` name plus its `data:` payload), and
//! [`AgentEventMapper`] turns those events into framework
//! [`ChatResponseUpdate`]s, tracking the run id so streamed message deltas
//! coalesce into a single assistant message. The Azure AI run event names
//! follow the Assistants streaming convention (`thread.run.*`,
//! `thread.message.delta`, `thread.run.step.*`).

use std::collections::VecDeque;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::streaming::Utf8StreamDecoder;
use agent_framework_core::types::{
    Annotation, ChatResponseUpdate, Content, FinishReason, Role, TextContent, TextSpanRegion,
    UsageContent,
};
use futures::StreamExt;
use serde_json::Value;

use crate::convert;

// ---------------------------------------------------------------------------
// SSE framing
// ---------------------------------------------------------------------------

/// A single Server-Sent Event: an optional `event:` name and its `data:`
/// payload (multiple `data:` lines are joined with newlines).
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE framer. Feed it raw text with [`push`](Self::push); it
/// returns whatever complete events (blocks terminated by a blank line) have
/// accumulated, buffering any partial trailing block.
#[derive(Default)]
pub struct SseDecoder {
    buffer: String,
}

impl SseDecoder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append `chunk` and return any newly-complete events.
    pub fn push(&mut self, chunk: &str) -> Vec<SseEvent> {
        // Normalize CRLF so the blank-line boundary is always "\n\n".
        self.buffer.push_str(&chunk.replace("\r\n", "\n"));
        let mut events = Vec::new();
        while let Some(idx) = self.buffer.find("\n\n") {
            let block: String = self.buffer[..idx].to_string();
            self.buffer.drain(..idx + 2);
            if let Some(ev) = parse_event_block(&block) {
                events.push(ev);
            }
        }
        events
    }
}

fn parse_event_block(block: &str) -> Option<SseEvent> {
    let mut event = None;
    let mut data_lines: Vec<String> = Vec::new();
    for line in block.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            data_lines.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        }
        // `id:`, `retry:`, and comment lines (`:` …) are ignored.
    }
    if event.is_none() && data_lines.is_empty() {
        return None;
    }
    Some(SseEvent {
        event,
        data: data_lines.join("\n"),
    })
}

// ---------------------------------------------------------------------------
// Event → update mapping
// ---------------------------------------------------------------------------

/// The result of handling one SSE event.
pub enum EventOutcome {
    /// Zero or more updates to yield to the consumer.
    Updates(Vec<ChatResponseUpdate>),
    /// The run failed; the stream should end with this (already-classified —
    /// see [`convert::classify_last_error`]) error.
    Failed(Error),
    /// End of stream (`event: done` / `data: [DONE]`).
    Done,
}

/// Maps Azure AI run SSE events to [`ChatResponseUpdate`]s, remembering the
/// active run id so all updates in a turn share a `conversation_id`,
/// `response_id`, and `message_id`.
pub struct AgentEventMapper {
    thread_id: String,
    response_id: Option<String>,
}

impl AgentEventMapper {
    pub fn new(thread_id: impl Into<String>) -> Self {
        Self {
            thread_id: thread_id.into(),
            response_id: None,
        }
    }

    fn base_update(&self) -> ChatResponseUpdate {
        ChatResponseUpdate {
            conversation_id: Some(self.thread_id.clone()),
            response_id: self.response_id.clone(),
            message_id: self.response_id.clone(),
            role: Some(Role::assistant()),
            ..Default::default()
        }
    }

    pub fn handle(&mut self, ev: &SseEvent) -> EventOutcome {
        let event = ev.event.as_deref().unwrap_or("");
        if event == "done" || ev.data.trim() == "[DONE]" {
            return EventOutcome::Done;
        }
        let data: Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(_) => return EventOutcome::Updates(Vec::new()),
        };

        match event {
            "thread.run.created" | "thread.run.queued" | "thread.run.in_progress" => {
                if let Some(id) = data.get("id").and_then(Value::as_str) {
                    self.response_id = Some(id.to_string());
                }
                let mut u = self.base_update();
                u.model = data.get("model").and_then(Value::as_str).map(String::from);
                EventOutcome::Updates(vec![u])
            }
            "thread.run.requires_action" => {
                let run_id = data
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                if self.response_id.is_none() {
                    self.response_id = Some(run_id.clone());
                }
                let mut u = self.base_update();
                u.contents = convert::required_action_contents(&data, &run_id);
                u.finish_reason = Some(FinishReason::tool_calls());
                EventOutcome::Updates(vec![u])
            }
            "thread.run.failed" => EventOutcome::Failed(convert::classify_last_error(&data)),
            "thread.run.completed" => {
                let mut u = self.base_update();
                u.finish_reason = Some(FinishReason::stop());
                EventOutcome::Updates(vec![u])
            }
            "thread.run.cancelled" | "thread.run.expired" => {
                let mut u = self.base_update();
                u.finish_reason = Some(FinishReason::new(event.trim_start_matches("thread.run.")));
                EventOutcome::Updates(vec![u])
            }
            "thread.run.step.created" => {
                if let Some(rid) = data.get("run_id").and_then(Value::as_str) {
                    self.response_id = Some(rid.to_string());
                }
                EventOutcome::Updates(Vec::new())
            }
            "thread.run.step.completed" => match convert::parse_usage(&data) {
                Some(usage) => {
                    let mut u = self.base_update();
                    u.contents = vec![Content::Usage(UsageContent { details: usage })];
                    EventOutcome::Updates(vec![u])
                }
                None => EventOutcome::Updates(Vec::new()),
            },
            "thread.message.delta" => {
                let contents = parse_message_delta(&data);
                if contents.is_empty() {
                    EventOutcome::Updates(Vec::new())
                } else {
                    let mut u = self.base_update();
                    u.contents = contents;
                    EventOutcome::Updates(vec![u])
                }
            }
            _ => EventOutcome::Updates(Vec::new()),
        }
    }
}

fn parse_message_delta(data: &Value) -> Vec<Content> {
    let mut out = Vec::new();
    let content = data
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(Value::as_array);
    for block in content.into_iter().flatten() {
        if block.get("type").and_then(Value::as_str) != Some("text") {
            continue;
        }
        let text_obj = block.get("text");
        let value = text_obj
            .and_then(|t| t.get("value"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let citations = parse_annotations(text_obj.and_then(|t| t.get("annotations")));
        if !value.is_empty() || citations.is_some() {
            let mut tc = TextContent::new(value);
            tc.annotations = citations;
            out.push(Content::Text(tc));
        }
    }
    out
}

fn parse_annotations(annotations: Option<&Value>) -> Option<Vec<Annotation>> {
    let annotations = annotations?.as_array()?;
    let mut out = Vec::new();
    for a in annotations {
        if a.get("type").and_then(Value::as_str) != Some("url_citation") {
            continue;
        }
        let uc = a.get("url_citation");
        let start = a.get("start_index").and_then(Value::as_i64);
        let end = a.get("end_index").and_then(Value::as_i64);
        let regions = match (start, end) {
            (Some(_), Some(_)) => Some(vec![TextSpanRegion {
                start_index: start,
                end_index: end,
            }]),
            _ => None,
        };
        out.push(Annotation {
            title: uc
                .and_then(|u| u.get("title"))
                .and_then(Value::as_str)
                .map(String::from),
            url: uc
                .and_then(|u| u.get("url"))
                .and_then(Value::as_str)
                .map(String::from),
            annotated_regions: regions,
            ..Default::default()
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// reqwest byte-stream wrapper
// ---------------------------------------------------------------------------

type ByteStream =
    std::pin::Pin<Box<dyn futures::Stream<Item = reqwest::Result<bytes::Bytes>> + Send>>;

struct StreamState {
    utf8: Utf8StreamDecoder,
    byte_stream: ByteStream,
    decoder: SseDecoder,
    mapper: AgentEventMapper,
    queued: VecDeque<ChatResponseUpdate>,
    done: bool,
}

/// Turn a streaming Azure AI run response into a stream of
/// [`ChatResponseUpdate`]s.
pub fn parse_agent_sse_stream(
    resp: reqwest::Response,
    thread_id: String,
) -> impl futures::Stream<Item = Result<ChatResponseUpdate>> + Send {
    let state = StreamState {
        byte_stream: Box::pin(resp.bytes_stream()),
        decoder: SseDecoder::new(),
        utf8: Utf8StreamDecoder::new(),
        mapper: AgentEventMapper::new(thread_id),
        queued: VecDeque::new(),
        done: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        loop {
            if let Some(update) = state.queued.pop_front() {
                return Some((Ok(update), state));
            }
            if state.done {
                return None;
            }
            match state.byte_stream.next().await {
                Some(Ok(bytes)) => {
                    let chunk = state.utf8.push(&bytes);
                    for ev in state.decoder.push(&chunk) {
                        match state.mapper.handle(&ev) {
                            EventOutcome::Updates(updates) => state.queued.extend(updates),
                            EventOutcome::Failed(err) => {
                                state.done = true;
                                return Some((Err(err), state));
                            }
                            EventOutcome::Done => {
                                state.done = true;
                                break;
                            }
                        }
                    }
                }
                Some(Err(e)) => {
                    state.done = true;
                    return Some((Err(Error::service(format!("stream error: {e}"))), state));
                }
                None => {
                    state.done = true;
                    if let Some(update) = state.queued.pop_front() {
                        return Some((Ok(update), state));
                    }
                    return None;
                }
            }
        }
    })
}

/// Drive the decoder + mapper over a complete SSE text body, collecting all
/// updates. Used by [`parse_agent_sse_stream`]'s tests and by the non-network
/// fixtures; stops at a `done` sentinel and surfaces a failed run as an error.
#[cfg(test)]
pub fn updates_from_sse(sse: &str, thread_id: &str) -> Result<Vec<ChatResponseUpdate>> {
    let mut decoder = SseDecoder::new();
    let mut mapper = AgentEventMapper::new(thread_id);
    let mut updates = Vec::new();
    for ev in decoder.push(sse) {
        match mapper.handle(&ev) {
            EventOutcome::Updates(u) => updates.extend(u),
            EventOutcome::Failed(err) => return Err(err),
            EventOutcome::Done => break,
        }
    }
    Ok(updates)
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::ChatResponse;

    #[test]
    fn decoder_frames_events_across_chunk_boundaries() {
        let mut d = SseDecoder::new();
        // A partial event split across two pushes.
        assert!(d
            .push("event: thread.run.created\ndata: {\"id\"")
            .is_empty());
        let events = d.push(":\"run_1\"}\n\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("thread.run.created"));
        assert_eq!(events[0].data, "{\"id\":\"run_1\"}");
    }

    #[test]
    fn decoder_handles_crlf_and_done() {
        let mut d = SseDecoder::new();
        let events = d.push("event: done\r\ndata: [DONE]\r\n\r\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.as_deref(), Some("done"));
        assert_eq!(events[0].data, "[DONE]");
    }

    /// A realistic run: created → step created → message deltas → step
    /// completed (usage) → run completed → done.
    #[test]
    fn text_run_produces_text_and_usage() {
        let sse = concat!(
            "event: thread.run.created\ndata: {\"id\":\"run_1\",\"model\":\"gpt-4o\"}\n\n",
            "event: thread.run.step.created\ndata: {\"id\":\"step_1\",\"run_id\":\"run_1\"}\n\n",
            "event: thread.message.delta\ndata: {\"id\":\"msg_1\",\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":{\"value\":\"Hello\"}}]}}\n\n",
            "event: thread.message.delta\ndata: {\"id\":\"msg_1\",\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":{\"value\":\" world\"}}]}}\n\n",
            "event: thread.run.step.completed\ndata: {\"id\":\"step_1\",\"usage\":{\"prompt_tokens\":8,\"completion_tokens\":2,\"total_tokens\":10}}\n\n",
            "event: thread.run.completed\ndata: {\"id\":\"run_1\",\"status\":\"completed\"}\n\n",
            "event: done\ndata: [DONE]\n\n",
        );
        let updates = updates_from_sse(sse, "thread_9").unwrap();
        // Aggregate to verify end-to-end behavior.
        let resp = ChatResponse::from_updates(updates);
        assert_eq!(resp.text(), "Hello world");
        assert_eq!(resp.conversation_id.as_deref(), Some("thread_9"));
        assert_eq!(resp.finish_reason, Some(FinishReason::stop()));
        assert_eq!(resp.usage_details.unwrap().total_token_count, Some(10));
    }

    #[test]
    fn requires_action_surfaces_function_calls() {
        let sse = concat!(
            "event: thread.run.created\ndata: {\"id\":\"run_5\"}\n\n",
            "event: thread.run.requires_action\ndata: {\"id\":\"run_5\",\"status\":\"requires_action\",\"required_action\":{\"type\":\"submit_tool_outputs\",\"submit_tool_outputs\":{\"tool_calls\":[{\"id\":\"call_x\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]}}}\n\n",
        );
        let updates = updates_from_sse(sse, "thread_1").unwrap();
        let resp = ChatResponse::from_updates(updates);
        let calls = resp.function_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(
            convert::decode_call_id(&calls[0].call_id),
            Some(("run_5".into(), "call_x".into()))
        );
        assert_eq!(resp.finish_reason, Some(FinishReason::tool_calls()));
    }

    #[test]
    fn failed_run_becomes_error() {
        let sse = concat!(
            "event: thread.run.created\ndata: {\"id\":\"run_2\"}\n\n",
            "event: thread.run.failed\ndata: {\"id\":\"run_2\",\"status\":\"failed\",\"last_error\":{\"code\":\"rate_limit\",\"message\":\"too many\"}}\n\n",
        );
        let err = updates_from_sse(sse, "thread_1").unwrap_err();
        assert!(err.to_string().contains("too many"));
        // A non-content-filter code (or none at all) stays the generic
        // variant, matching upstream's undifferentiated `ServiceResponseException`.
        assert!(matches!(err, Error::Service(_)));
    }

    #[test]
    fn content_filter_run_failure_becomes_content_filter_error() {
        let sse = concat!(
            "event: thread.run.created\ndata: {\"id\":\"run_3\"}\n\n",
            "event: thread.run.failed\ndata: {\"id\":\"run_3\",\"status\":\"failed\",\"last_error\":{\"code\":\"content_filter\",\"message\":\"The response was filtered\"}}\n\n",
        );
        let err = updates_from_sse(sse, "thread_1").unwrap_err();
        assert!(err.to_string().contains("The response was filtered"));
        assert!(matches!(err, Error::ServiceContentFilter { .. }));
    }

    #[test]
    fn message_delta_carries_url_citation() {
        let sse = "event: thread.message.delta\ndata: {\"id\":\"msg_1\",\"delta\":{\"content\":[{\"index\":0,\"type\":\"text\",\"text\":{\"value\":\"see docs\",\"annotations\":[{\"type\":\"url_citation\",\"start_index\":0,\"end_index\":8,\"url_citation\":{\"url\":\"https://example.com\",\"title\":\"Docs\"}}]}}]}}\n\n";
        let updates = updates_from_sse(sse, "thread_1").unwrap();
        let text = &updates[0].contents[0];
        let Content::Text(tc) = text else {
            panic!("expected text")
        };
        let ann = tc.annotations.as_ref().unwrap();
        assert_eq!(ann[0].url.as_deref(), Some("https://example.com"));
        assert_eq!(ann[0].title.as_deref(), Some("Docs"));
    }
}
