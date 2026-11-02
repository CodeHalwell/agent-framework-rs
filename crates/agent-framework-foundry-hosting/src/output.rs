//! Projects agent output onto Responses output items and stream events
//! (upstream `_ResponseEventTracker`).
//!
//! One tracker serves both the streamed and the non-streamed path, so a
//! response's `output` is identical either way: contents are folded in
//! order into output items, one active item at a time, and every step
//! produces the agentserver's stream events (discarded when not
//! streaming).
//!
//! | Content | Output item |
//! | --- | --- |
//! | text | `message` with an `output_text` part (citations as `url_citation` annotations) |
//! | refusal text | `message` with a `refusal` part |
//! | reasoning | `reasoning` with a summary (and `encrypted_content` from `protected_data`) |
//! | function call | `function_call`, arguments streamed |
//! | function result | `function_call_output` |
//! | MCP server tool call / result | `mcp_call` (completed or failed by its result); an unmatched result is a `custom_tool_call_output` |
//! | image generation result | `image_generation_call` with the base64 image |
//! | shell tool call / result | `shell_call` / `shell_call_output` |
//! | function approval request | `mcp_approval_request` |
//! | OAuth consent request | `oauth_consent_request` (safe HTTPS links only) |
//! | usage | the response's `usage` |
//!
//! A `content_filter` / `length` finish reason makes the response
//! `incomplete` (`incomplete_details.reason`).

use std::collections::HashSet;

use agent_framework_core::types::{
    AgentResponse, AgentResponseUpdate, Annotation, Content, FunctionArguments, UsageDetails,
};
use base64::Engine;
use serde_json::{json, Value};

use crate::scope::ids;

/// The item being streamed.
enum Active {
    Message {
        index: usize,
        part: usize,
        refusal: bool,
        message_id: Option<String>,
    },
    Reasoning {
        index: usize,
        encrypted: Option<String>,
    },
    Mcp {
        index: usize,
        call_id: String,
    },
}

/// Folds contents into output items and stream events.
pub(crate) struct OutputTracker {
    response_id: String,
    items: Vec<Value>,
    active: Option<Active>,
    buffer: String,
    annotations: Vec<Value>,
    usage: Option<UsageDetails>,
    incomplete: Option<&'static str>,
    events: Vec<(String, Value)>,
    consent_origins: Option<Vec<String>>,
    consents: HashSet<(String, String)>,
    /// Function calls still receiving arguments, in first-seen order. They
    /// accumulate per `call_id` (providers may interleave several calls'
    /// deltas) and are finalized when their result arrives or the response
    /// ends.
    open_calls: Vec<OpenCall>,
    /// `(mcp_approval_request id, original approval request)` pairs, to be
    /// saved so a later `mcp_approval_response` can be resolved.
    approvals: Vec<(String, Content)>,
    /// Computer calls already emitted in this response, by `call_id`, so a
    /// replayed call is not emitted twice (upstream
    /// `_outstanding_computer_calls`).
    #[cfg(feature = "experimental-computer-use")]
    computer_calls: HashSet<String>,
    /// Container files announced by MCP tool results, by filename, in first
    /// announced order, until output text names them (upstream
    /// `_pending_container_file_citations`).
    pending_container_files: Vec<ContainerFileCitation>,
}

/// A container file an MCP tool result announced (upstream
/// `_ContainerFileCitation`).
#[derive(Debug, Clone, PartialEq)]
struct ContainerFileCitation {
    container_id: String,
    file_id: String,
    filename: String,
}

/// The container files an MCP tool result announces in its `_meta` or its
/// content blocks' `_meta` under `container_file_citations` (a citation, a
/// list of them, or either JSON-encoded), each with `file_id`, `filename`
/// and `container_id` (or the `_meta` block's own `container_id`); invalid
/// entries are skipped (upstream
/// `_container_file_citations_from_function_result`).
fn container_file_citations(
    result: &agent_framework_core::types::FunctionResultContent,
) -> Vec<ContainerFileCitation> {
    let Some(payload) = result
        .additional_properties
        .get(agent_framework_core::types::MCP_TOOL_RESULT_HOST_PAYLOAD_KEY)
        .and_then(Value::as_object)
    else {
        return Vec::new();
    };
    let mut blocks: Vec<&serde_json::Map<String, Value>> = Vec::new();
    if let Some(meta) = payload.get("_meta").and_then(Value::as_object) {
        blocks.push(meta);
    }
    for item in payload
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(meta) = item.get("_meta").and_then(Value::as_object) {
            blocks.push(meta);
        }
    }
    let nonempty = |v: Option<&Value>| {
        v.and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let mut citations = Vec::new();
    for meta in blocks {
        let mut raw = meta.get("container_file_citations").cloned();
        if let Some(Value::String(encoded)) = &raw {
            match serde_json::from_str::<Value>(encoded) {
                Ok(decoded) => raw = Some(decoded),
                Err(_) => continue,
            }
        }
        let values = match raw {
            Some(Value::Object(citation)) => vec![Value::Object(citation)],
            Some(Value::Array(values)) => values,
            _ => continue,
        };
        let fallback = meta.get("container_id");
        for value in &values {
            let Some(citation) = value.as_object() else {
                continue;
            };
            // Upstream's `or`: a falsy container id falls back.
            let falsy = |v: &Value| match v {
                Value::Null => true,
                Value::Bool(b) => !b,
                Value::Number(n) => n.as_f64() == Some(0.0),
                Value::String(s) => s.is_empty(),
                Value::Array(a) => a.is_empty(),
                Value::Object(o) => o.is_empty(),
            };
            let container = citation
                .get("container_id")
                .filter(|v| !falsy(v))
                .or(fallback);
            let (Some(container_id), Some(file_id), Some(filename)) = (
                nonempty(container),
                nonempty(citation.get("file_id")),
                nonempty(citation.get("filename")),
            ) else {
                continue;
            };
            citations.push(ContainerFileCitation {
                container_id,
                file_id,
                filename,
            });
        }
    }
    citations
}

/// Whether `text[start..end]` (char indices) is not embedded in another
/// filename-like token (upstream `_has_container_filename_boundaries`).
fn has_filename_boundaries(text: &[char], start: usize, end: usize) -> bool {
    let token = |c: char| c.is_alphanumeric() || matches!(c, '.' | '_' | '-');
    let before = start.checked_sub(1).map(|i| text[i]);
    let after = text.get(end).copied();
    !before.is_some_and(token) && !after.is_some_and(token)
}

struct OpenCall {
    index: usize,
    call_id: String,
    name: String,
    arguments: String,
}

/// A content the tracker refuses (fails the response).
#[derive(Debug)]
pub(crate) struct OutputError(pub String);

impl std::fmt::Display for OutputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl OutputTracker {
    pub(crate) fn new(response_id: &str, consent_origins: Option<Vec<String>>) -> Self {
        Self {
            response_id: response_id.to_string(),
            items: Vec::new(),
            active: None,
            buffer: String::new(),
            annotations: Vec::new(),
            usage: None,
            incomplete: None,
            events: Vec::new(),
            consent_origins,
            consents: HashSet::new(),
            open_calls: Vec::new(),
            approvals: Vec::new(),
            #[cfg(feature = "experimental-computer-use")]
            computer_calls: HashSet::new(),
            pending_container_files: Vec::new(),
        }
    }

    /// Remember the container files `result` announces (a later announcement
    /// of a filename replaces the earlier one).
    fn note_container_files(
        &mut self,
        result: &agent_framework_core::types::FunctionResultContent,
    ) {
        for citation in container_file_citations(result) {
            match self
                .pending_container_files
                .iter_mut()
                .find(|c| c.filename == citation.filename)
            {
                Some(existing) => *existing = citation,
                None => self.pending_container_files.push(citation),
            }
        }
    }

    /// `container_file_citation` annotations for the pending files `text`
    /// names (upstream `_container_file_annotations_for_text`): every
    /// occurrence with filename boundaries is a candidate; longer matches
    /// win, then earlier ones, each file is cited once, citations never
    /// overlap, and a cited file stops being pending. Indices count
    /// characters.
    fn container_file_annotations(&mut self, text: &str) -> Vec<Value> {
        let chars: Vec<char> = text.chars().collect();
        let mut candidates: Vec<(usize, usize, usize)> = Vec::new();
        for (i, citation) in self.pending_container_files.iter().enumerate() {
            let needle: Vec<char> = citation.filename.chars().collect();
            if needle.is_empty() || needle.len() > chars.len() {
                continue;
            }
            for start in 0..=chars.len() - needle.len() {
                let end = start + needle.len();
                if chars[start..end] == needle[..] && has_filename_boundaries(&chars, start, end) {
                    candidates.push((start, end, i));
                }
            }
        }
        let files = &self.pending_container_files;
        candidates.sort_by(|a, b| {
            (b.1 - b.0)
                .cmp(&(a.1 - a.0))
                .then(a.0.cmp(&b.0))
                .then(files[a.2].filename.cmp(&files[b.2].filename))
        });
        let mut selected: Vec<(usize, usize, usize)> = Vec::new();
        for candidate in candidates {
            let (start, end, i) = candidate;
            if selected.iter().any(|s| s.2 == i)
                || selected.iter().any(|s| start < s.1 && s.0 < end)
            {
                continue;
            }
            selected.push(candidate);
        }
        selected.sort_by_key(|s| s.0);
        let annotations = selected
            .iter()
            .map(|&(start, end, i)| {
                let c = &files[i];
                json!({
                    "type": "container_file_citation",
                    "container_id": c.container_id,
                    "file_id": c.file_id,
                    "filename": c.filename,
                    "start_index": start,
                    "end_index": end,
                })
            })
            .collect();
        let cited: HashSet<usize> = selected.iter().map(|s| s.2).collect();
        let mut index = 0;
        self.pending_container_files.retain(|_| {
            let keep = !cited.contains(&index);
            index += 1;
            keep
        });
        annotations
    }

    /// The events produced since the last call.
    pub(crate) fn drain(&mut self) -> Vec<(String, Value)> {
        std::mem::take(&mut self.events)
    }

    fn emit(&mut self, kind: &str, payload: Value) {
        self.events.push((kind.to_string(), payload));
    }

    /// Note a finish reason: only the ones meaning the model stopped early.
    pub(crate) fn record_finish_reason(&mut self, reason: Option<&str>) {
        match reason {
            Some("content_filter") => self.incomplete = Some("content_filter"),
            Some("length") if self.incomplete.is_none() => {
                self.incomplete = Some("max_output_tokens")
            }
            _ => {}
        }
    }

    /// Fold one streamed update.
    pub(crate) fn handle_update(
        &mut self,
        update: &AgentResponseUpdate,
    ) -> Result<(), OutputError> {
        self.record_finish_reason(update.finish_reason.as_ref().map(|f| f.0.as_str()));
        for content in &update.contents {
            self.handle(content, update.message_id.as_deref(), false)?;
        }
        Ok(())
    }

    /// Fold a completed response: each message's parts keep their
    /// boundaries (upstream `complete_message=True`).
    pub(crate) fn handle_response(&mut self, response: &AgentResponse) -> Result<(), OutputError> {
        for (i, message) in response.messages.iter().enumerate() {
            let message_id = message
                .message_id
                .clone()
                .unwrap_or_else(|| format!("m{i}"));
            for content in &message.contents {
                self.handle(content, Some(&message_id), true)?;
            }
        }
        if let Some(usage) = &response.usage_details {
            self.add_usage(usage);
        }
        self.record_finish_reason(response.finish_reason.as_ref().map(|f| f.0.as_str()));
        Ok(())
    }

    fn add_usage(&mut self, usage: &UsageDetails) {
        let total = self.usage.get_or_insert_with(UsageDetails::default);
        fn add(a: &mut Option<u64>, b: Option<u64>) {
            if let Some(b) = b {
                *a = Some(a.unwrap_or(0) + b);
            }
        }
        add(&mut total.input_token_count, usage.input_token_count);
        add(&mut total.output_token_count, usage.output_token_count);
        add(&mut total.total_token_count, usage.total_token_count);
        add(
            &mut total.cache_creation_input_token_count,
            usage.cache_creation_input_token_count,
        );
        add(
            &mut total.cache_read_input_token_count,
            usage.cache_read_input_token_count,
        );
        add(
            &mut total.reasoning_output_token_count,
            usage.reasoning_output_token_count,
        );
    }

    /// Fold one content item.
    pub(crate) fn handle(
        &mut self,
        content: &Content,
        message_id: Option<&str>,
        new_part: bool,
    ) -> Result<(), OutputError> {
        match content {
            Content::Text(text) => {
                self.ensure_message(text.refusal, message_id, new_part);
                for annotation in annotations_to_output_text(text.annotations.as_deref()) {
                    if !self.annotations.contains(&annotation) {
                        self.annotations.push(annotation);
                    }
                }
                if !text.text.is_empty() {
                    self.buffer.push_str(&text.text);
                    if let Some(Active::Message {
                        index,
                        part,
                        refusal,
                        ..
                    }) = &self.active
                    {
                        let (index, part, refusal) = (*index, *part, *refusal);
                        let id = self.items[index]["id"].clone();
                        let kind = if refusal {
                            "response.refusal.delta"
                        } else {
                            "response.output_text.delta"
                        };
                        self.emit(
                            kind,
                            json!({"item_id": id, "output_index": index, "content_index": part, "delta": text.text}),
                        );
                    }
                }
            }
            Content::TextReasoning(reasoning) => {
                if !matches!(self.active, Some(Active::Reasoning { .. })) {
                    self.close();
                    let id = ids::new_id("rs", Some(&self.response_id));
                    let index = self.push_item(json!({
                        "type": "reasoning", "id": id, "summary": [],
                        "encrypted_content": null, "status": "in_progress",
                    }));
                    self.emit(
                        "response.reasoning_summary_part.added",
                        json!({"item_id": id, "output_index": index, "summary_index": 0,
                               "part": {"type": "summary_text", "text": ""}}),
                    );
                    self.active = Some(Active::Reasoning {
                        index,
                        encrypted: None,
                    });
                }
                if let Some(Active::Reasoning { index, encrypted }) = &mut self.active {
                    if reasoning.protected_data.is_some() {
                        encrypted.clone_from(&reasoning.protected_data);
                    }
                    let index = *index;
                    if !reasoning.text.is_empty() {
                        self.buffer.push_str(&reasoning.text);
                        let id = self.items[index]["id"].clone();
                        self.emit(
                            "response.reasoning_summary_text.delta",
                            json!({"item_id": id, "output_index": index, "summary_index": 0, "delta": reasoning.text}),
                        );
                    }
                }
            }
            Content::FunctionCall(call) => {
                let delta = arguments_string(call.arguments.as_ref());
                let position = self
                    .open_calls
                    .iter()
                    .position(|c| c.call_id == call.call_id);
                let position = match position {
                    Some(p) => p,
                    None => {
                        // A new call ends the item being streamed, not other calls.
                        self.close();
                        let id = ids::function_call_id(Some(&self.response_id));
                        let index = self.push_item(json!({
                            "type": "function_call", "id": id, "call_id": call.call_id,
                            "name": call.name, "arguments": "", "status": "in_progress",
                        }));
                        self.open_calls.push(OpenCall {
                            index,
                            call_id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: String::new(),
                        });
                        self.open_calls.len() - 1
                    }
                };
                let open = &mut self.open_calls[position];
                if open.name.is_empty() && !call.name.is_empty() {
                    open.name = call.name.clone();
                    self.items[open.index]["name"] = json!(call.name);
                }
                let open = &mut self.open_calls[position];
                if !delta.is_empty() {
                    open.arguments.push_str(&delta);
                    let index = open.index;
                    let id = self.items[index]["id"].clone();
                    self.emit(
                        "response.function_call_arguments.delta",
                        json!({"item_id": id, "output_index": index, "delta": delta}),
                    );
                }
            }
            Content::FunctionResult(result) => {
                // The call is complete before its result.
                if let Some(p) = self
                    .open_calls
                    .iter()
                    .position(|c| c.call_id == result.call_id)
                {
                    let call = self.open_calls.remove(p);
                    self.finish_call(call);
                }
                self.close();
                self.note_container_files(result);
                let output = match (&result.result, &result.exception) {
                    (Some(value), _) => json_text(value),
                    (None, Some(exception)) => format!("Error: {exception}"),
                    (None, None) => String::new(),
                };
                self.simple_item(json!({
                    "type": "function_call_output",
                    "id": ids::new_id("fco", Some(&self.response_id)),
                    "call_id": result.call_id, "output": output, "status": "completed",
                }));
            }
            Content::McpServerToolCall(call) => {
                let same = matches!(&self.active, Some(Active::Mcp { call_id, .. }) if *call_id == call.call_id);
                if !same {
                    self.close();
                    let id = ids::new_id("mcp", Some(&self.response_id));
                    let index = self.push_item(json!({
                        "type": "mcp_call", "id": id,
                        "server_label": call.server_name.clone().unwrap_or_else(|| "default".into()),
                        "name": call.tool_name, "arguments": "", "status": "in_progress",
                    }));
                    self.emit(
                        "response.mcp_call.in_progress",
                        json!({"item_id": id, "output_index": index}),
                    );
                    self.active = Some(Active::Mcp {
                        index,
                        call_id: call.call_id.clone(),
                    });
                }
                let delta = arguments_string(call.arguments.as_ref());
                if !delta.is_empty() {
                    self.buffer.push_str(&delta);
                    if let Some(Active::Mcp { index, .. }) = &self.active {
                        let index = *index;
                        let id = self.items[index]["id"].clone();
                        self.emit(
                            "response.mcp_call_arguments.delta",
                            json!({"item_id": id, "output_index": index, "delta": delta}),
                        );
                    }
                }
            }
            Content::McpServerToolResult(result) => {
                let matched = matches!(&self.active, Some(Active::Mcp { call_id, .. }) if *call_id == result.call_id);
                if matched {
                    let Some(Active::Mcp { index, .. }) = self.active.take() else {
                        unreachable!()
                    };
                    let arguments = std::mem::take(&mut self.buffer);
                    let id = self.items[index]["id"].clone();
                    self.emit(
                        "response.mcp_call_arguments.done",
                        json!({"item_id": id, "output_index": index, "arguments": arguments}),
                    );
                    let item = &mut self.items[index];
                    item["arguments"] = json!(arguments);
                    if let Some(error) = &result.error {
                        item["status"] = json!("failed");
                        item["error"] = json!(json_text(error));
                    } else {
                        item["status"] = json!("completed");
                        item["output"] = json!(mcp_output_text(result.output.as_ref()));
                    }
                    let failed = result.error.is_some();
                    self.emit(
                        if failed {
                            "response.mcp_call.failed"
                        } else {
                            "response.mcp_call.completed"
                        },
                        json!({"item_id": id, "output_index": index}),
                    );
                    let item = self.items[index].clone();
                    self.emit(
                        "response.output_item.done",
                        json!({"output_index": index, "item": item}),
                    );
                } else {
                    self.close();
                    self.simple_item(json!({
                        "type": "custom_tool_call_output",
                        "id": ids::new_id("ctco", Some(&self.response_id)),
                        "call_id": result.call_id,
                        "output": mcp_output_text(result.output.as_ref()),
                    }));
                }
            }
            Content::ImageGenerationToolResult(result) => {
                self.close();
                let image = result
                    .outputs
                    .as_ref()
                    .and_then(image_generation_result)
                    .unwrap_or_else(|| {
                        tracing::warn!("image generation result has no base64 image");
                        String::new()
                    });
                let id = ids::new_id("ig", Some(&self.response_id));
                let index = self.push_item(json!({
                    "type": "image_generation_call", "id": id, "status": "in_progress", "result": null,
                }));
                for state in ["in_progress", "generating", "completed"] {
                    self.emit(
                        &format!("response.image_generation_call.{state}"),
                        json!({"item_id": id, "output_index": index}),
                    );
                }
                self.items[index]["status"] = json!("completed");
                self.items[index]["result"] = json!(image);
                let item = self.items[index].clone();
                self.emit(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item}),
                );
            }
            #[cfg(feature = "experimental-computer-use")]
            Content::ComputerToolCall(call) => {
                if !self.computer_calls.insert(call.call_id.clone()) {
                    return Ok(());
                }
                self.close();
                let item = crate::computer::call_item(call, &self.response_id)?;
                self.simple_item(item);
            }
            #[cfg(feature = "experimental-computer-use")]
            Content::ComputerToolResult(result) => {
                self.close();
                let item = crate::computer::result_item(result, &self.response_id)?;
                self.computer_calls.remove(&result.call_id);
                self.simple_item(item);
            }
            Content::ShellToolCall(call) => {
                self.close();
                self.simple_item(json!({
                    "type": "shell_call",
                    "id": ids::new_id("lsh", Some(&self.response_id)),
                    "call_id": call.call_id.clone().unwrap_or_default(),
                    "action": {
                        "commands": call.commands.clone().unwrap_or_default(),
                        "timeout_ms": call.timeout_ms,
                        "max_output_length": call.max_output_length,
                    },
                    "environment": {"type": "local"},
                    "status": call.status.clone().unwrap_or_else(|| "completed".into()),
                }));
            }
            Content::ShellToolResult(result) => {
                self.close();
                let output: Vec<Value> = result
                    .outputs
                    .iter()
                    .flatten()
                    .filter_map(|c| match c {
                        Content::ShellCommandOutput(o) => Some(json!({
                            "stdout": o.stdout.clone().unwrap_or_default(),
                            "stderr": o.stderr.clone().unwrap_or_default(),
                            "outcome": if o.timed_out == Some(true) {
                                json!({"type": "timeout"})
                            } else {
                                json!({"type": "exit", "exit_code": o.exit_code.unwrap_or(0)})
                            },
                        })),
                        _ => None,
                    })
                    .collect();
                self.simple_item(json!({
                    "type": "shell_call_output",
                    "id": ids::new_id("lsho", Some(&self.response_id)),
                    "call_id": result.call_id.clone().unwrap_or_default(),
                    "output": output,
                    "status": "completed",
                    "max_output_length": result.max_output_length.unwrap_or(0),
                }));
            }
            Content::FunctionApprovalRequest(request) => {
                self.close();
                let id = ids::new_id("mcpr", Some(&self.response_id));
                self.approvals.push((id.clone(), content.clone()));
                self.simple_item(json!({
                    "type": "mcp_approval_request",
                    "id": id,
                    "server_label": "agent_framework",
                    "name": request.function_call.name,
                    "arguments": arguments_string(request.function_call.arguments.as_ref()),
                }));
            }
            Content::OauthConsentRequest(request) => {
                self.close();
                if !is_allowed_consent_link(&request.consent_link, self.consent_origins.as_deref())
                {
                    return Err(OutputError(
                        "OAuth consent request content must include an allowed safe HTTPS consent link."
                            .into(),
                    ));
                }
                let key = (request.consent_link.clone(), "agent_framework".to_string());
                if self.consents.insert(key) {
                    self.simple_item(json!({
                        "type": "oauth_consent_request",
                        "id": ids::new_id("oacr", Some(&self.response_id)),
                        "response_id": self.response_id,
                        "consent_link": request.consent_link,
                        "server_label": "agent_framework",
                    }));
                }
            }
            Content::Usage(usage) => self.add_usage(&usage.details),
            other => {
                self.close();
                tracing::debug!(
                    content = ?std::mem::discriminant(other),
                    "content type is not projected onto a Responses output item"
                );
            }
        }
        Ok(())
    }

    fn push_item(&mut self, item: Value) -> usize {
        let index = self.items.len();
        self.items.push(item.clone());
        self.emit(
            "response.output_item.added",
            json!({"output_index": index, "item": item}),
        );
        index
    }

    /// An item emitted whole: `added` (in progress) then `done`.
    fn simple_item(&mut self, item: Value) {
        let mut added = item.clone();
        if added.get("status").is_some() {
            added["status"] = json!("in_progress");
        }
        let index = self.push_item(added);
        self.items[index] = item.clone();
        self.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        );
    }

    fn ensure_message(&mut self, refusal: bool, message_id: Option<&str>, new_part: bool) {
        if let Some(Active::Message {
            refusal: active_refusal,
            message_id: active_id,
            ..
        }) = &self.active
        {
            let changed = matches!((message_id, active_id), (Some(a), Some(b)) if a != b);
            if *active_refusal == refusal && !changed && !new_part {
                return;
            }
            if !changed {
                // Same message, another part.
                self.close_part();
                self.open_part(refusal, message_id);
                return;
            }
        }
        self.close();
        let id = ids::message_id(Some(&self.response_id));
        let index = self.push_item(json!({
            "id": id, "type": "message", "status": "in_progress", "role": "assistant", "content": [],
        }));
        self.active = Some(Active::Message {
            index,
            part: 0,
            refusal,
            message_id: message_id.map(str::to_string),
        });
        self.open_part_at(index, 0, refusal);
    }

    fn open_part(&mut self, refusal: bool, message_id: Option<&str>) {
        let index = match &self.active {
            Some(Active::Message { index, .. }) => *index,
            _ => return,
        };
        let part = self.items[index]["content"]
            .as_array()
            .map(Vec::len)
            .unwrap_or(0);
        self.active = Some(Active::Message {
            index,
            part,
            refusal,
            message_id: message_id.map(str::to_string),
        });
        self.open_part_at(index, part, refusal);
    }

    fn open_part_at(&mut self, index: usize, part: usize, refusal: bool) {
        let body = if refusal {
            json!({"type": "refusal", "refusal": ""})
        } else {
            json!({"type": "output_text", "text": "", "annotations": []})
        };
        if let Some(content) = self.items[index]["content"].as_array_mut() {
            content.push(body.clone());
        }
        let id = self.items[index]["id"].clone();
        self.emit(
            "response.content_part.added",
            json!({"item_id": id, "output_index": index, "content_index": part, "part": body}),
        );
    }

    /// Close the active message's content part.
    fn close_part(&mut self) {
        let Some(Active::Message {
            index,
            part,
            refusal,
            ..
        }) = &self.active
        else {
            return;
        };
        let (index, part, refusal) = (*index, *part, *refusal);
        let text = std::mem::take(&mut self.buffer);
        let id = self.items[index]["id"].clone();
        let body = if refusal {
            self.emit(
                "response.refusal.done",
                json!({"item_id": id, "output_index": index, "content_index": part, "refusal": text}),
            );
            json!({"type": "refusal", "refusal": text})
        } else {
            self.emit(
                "response.output_text.done",
                json!({"item_id": id, "output_index": index, "content_index": part, "text": text, "logprobs": []}),
            );
            let mut annotations = std::mem::take(&mut self.annotations);
            for annotation in self.container_file_annotations(&text) {
                if !annotations.contains(&annotation) {
                    annotations.push(annotation);
                }
            }
            for (i, annotation) in annotations.iter().enumerate() {
                self.emit(
                    "response.output_text.annotation.added",
                    json!({"item_id": id, "output_index": index, "content_index": part,
                           "annotation_index": i, "annotation": annotation}),
                );
            }
            json!({"type": "output_text", "text": text, "annotations": annotations})
        };
        self.items[index]["content"][part] = body.clone();
        self.emit(
            "response.content_part.done",
            json!({"item_id": id, "output_index": index, "content_index": part, "part": body}),
        );
    }

    /// Close the active item.
    pub(crate) fn close(&mut self) {
        match self.active {
            Some(Active::Message { index, .. }) => {
                self.close_part();
                self.items[index]["status"] = json!("completed");
                self.done(index);
            }
            Some(Active::Reasoning {
                index,
                ref encrypted,
            }) => {
                let encrypted = encrypted.clone();
                let text = std::mem::take(&mut self.buffer);
                let id = self.items[index]["id"].clone();
                self.emit(
                    "response.reasoning_summary_text.done",
                    json!({"item_id": id, "output_index": index, "summary_index": 0, "text": text}),
                );
                self.emit(
                    "response.reasoning_summary_part.done",
                    json!({"item_id": id, "output_index": index, "summary_index": 0,
                           "part": {"type": "summary_text", "text": text}}),
                );
                let item = &mut self.items[index];
                item["summary"] = json!([{"type": "summary_text", "text": text}]);
                item["encrypted_content"] = json!(encrypted);
                item["status"] = json!("completed");
                self.done(index);
            }
            Some(Active::Mcp { index, .. }) => {
                let arguments = std::mem::take(&mut self.buffer);
                let id = self.items[index]["id"].clone();
                self.emit(
                    "response.mcp_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": arguments}),
                );
                self.emit(
                    "response.mcp_call.completed",
                    json!({"item_id": id, "output_index": index}),
                );
                self.items[index]["arguments"] = json!(arguments);
                self.items[index]["status"] = json!("completed");
                self.done(index);
            }
            None => {}
        }
        self.active = None;
        self.buffer.clear();
    }

    fn done(&mut self, index: usize) {
        let item = self.items[index].clone();
        self.emit(
            "response.output_item.done",
            json!({"output_index": index, "item": item}),
        );
    }

    /// Close everything and return `(output, usage, incomplete reason)`.
    /// The approval requests emitted so far, with their wire ids.
    /// The id of the most recent `mcp_approval_request` emitted.
    pub(crate) fn last_approval_id(&self) -> Option<String> {
        self.approvals.last().map(|(id, _)| id.clone())
    }

    pub(crate) fn take_approvals(&mut self) -> Vec<(String, Content)> {
        std::mem::take(&mut self.approvals)
    }

    fn finish_call(&mut self, call: OpenCall) {
        let id = self.items[call.index]["id"].clone();
        self.emit(
            "response.function_call_arguments.done",
            json!({"item_id": id, "output_index": call.index, "name": call.name, "arguments": call.arguments}),
        );
        self.items[call.index]["arguments"] = json!(call.arguments);
        self.items[call.index]["status"] = json!("completed");
        self.done(call.index);
    }

    /// Close the active item and every open function call (end of output).
    pub(crate) fn close_all(&mut self) {
        self.close();
        for call in std::mem::take(&mut self.open_calls) {
            self.finish_call(call);
        }
    }

    pub(crate) fn finish(mut self) -> (Vec<Value>, Option<Value>, Option<&'static str>) {
        self.close_all();
        let usage = self.usage.as_ref().map(usage_json);
        (self.items, usage, self.incomplete)
    }
}

/// The Responses `usage` object.
pub(crate) fn usage_json(usage: &UsageDetails) -> Value {
    let input = usage.input_token_count.unwrap_or(0);
    let output = usage.output_token_count.unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {
            "cached_tokens": usage.cache_read_input_token_count.unwrap_or(0),
            "cache_write_tokens": usage.cache_creation_input_token_count.unwrap_or(0),
        },
        "output_tokens": output,
        "output_tokens_details": {
            "reasoning_tokens": usage.reasoning_output_token_count.unwrap_or(0),
        },
        "total_tokens": usage.total_token_count.unwrap_or(input + output),
    })
}

fn arguments_string(arguments: Option<&FunctionArguments>) -> String {
    match arguments {
        None => String::new(),
        Some(FunctionArguments::Raw(raw)) => raw.clone(),
        Some(FunctionArguments::Object(map)) => serde_json::to_string(map).unwrap_or_default(),
    }
}

fn json_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Text of an MCP mapping (`{"text": …}` or `{"type": "text", "text": …}`).
fn mcp_mapping_text(value: &Value) -> Option<String> {
    value
        .get("text")
        .and_then(Value::as_str)
        .map(str::to_string)
}

/// MCP output as the string `mcp_call.output` expects.
fn mcp_output_text(output: Option<&Value>) -> String {
    match output {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(map @ Value::Object(_)) => mcp_mapping_text(map).unwrap_or_else(|| map.to_string()),
        Some(Value::Array(entries)) => {
            let mut parts = Vec::new();
            for entry in entries {
                match entry {
                    Value::String(s) => parts.push(s.clone()),
                    other => match mcp_mapping_text(other) {
                        Some(text) => parts.push(text),
                        None => return Value::Array(entries.clone()).to_string(),
                    },
                }
            }
            parts.concat()
        }
        Some(other) => other.to_string(),
    }
}

fn is_base64(value: &str) -> bool {
    !value.is_empty()
        && base64::engine::general_purpose::STANDARD
            .decode(value)
            .is_ok()
}

/// The base64 image of an image generation result: a base64 string, or the
/// payload of a base64 `data:` URI (directly or as a content's `uri`).
fn image_generation_result(outputs: &Value) -> Option<String> {
    let candidates: Vec<&Value> = match outputs {
        Value::Array(items) => items.iter().collect(),
        other => vec![other],
    };
    for item in candidates {
        let text = match item {
            Value::String(s) => s.as_str(),
            Value::Object(o) => match o.get("uri").and_then(Value::as_str) {
                Some(uri) => uri,
                None => continue,
            },
            _ => continue,
        };
        if let Some(rest) = text.strip_prefix("data:") {
            if let Some((meta, payload)) = rest.split_once(',') {
                if meta.ends_with(";base64") && is_base64(payload) {
                    return Some(payload.to_string());
                }
            }
        } else if is_base64(text) {
            return Some(text.to_string());
        }
    }
    None
}

/// Framework citations as Responses output-text annotations (upstream
/// `_annotations_to_output_text`): container-file citations (a
/// `container_id` extra), URL citations, file citations and file paths (an
/// `index` extra). Source-coordinate regions describe the cited source, not
/// the generated text, so they are not emitted as spans.
fn annotations_to_output_text(annotations: Option<&[Annotation]>) -> Vec<Value> {
    let mut out = Vec::new();
    for annotation in annotations.unwrap_or_default() {
        let regions: Vec<(i64, i64)> = if annotation.annotated_regions_coordinate_space
            == Some(agent_framework_core::types::RegionCoordinateSpace::Source)
        {
            Vec::new()
        } else {
            annotation
                .annotated_regions
                .iter()
                .flatten()
                .filter_map(|r| Some((r.start_index?, r.end_index?)))
                .collect()
        };
        let extras = &annotation.additional_properties;
        let container_id = extras
            .get("container_id")
            .and_then(Value::as_str)
            .filter(|c| !c.is_empty());
        let index = extras
            .get("index")
            .and_then(Value::as_i64)
            .filter(|i| *i >= 0);
        let file_id = annotation.file_id.as_deref().filter(|f| !f.is_empty());
        let url = annotation.url.as_deref().filter(|u| !u.is_empty());
        match (container_id, file_id, url, index) {
            (Some(container_id), Some(file_id), _, _) => {
                for (start, end) in &regions {
                    out.push(json!({
                        "type": "container_file_citation",
                        "container_id": container_id,
                        "file_id": file_id,
                        "filename": url.unwrap_or_default(),
                        "start_index": start,
                        "end_index": end,
                    }));
                }
            }
            (None, None, Some(url), _) => {
                for (start, end) in &regions {
                    out.push(json!({
                        "type": "url_citation",
                        "url": url,
                        "title": annotation.title.clone().unwrap_or_default(),
                        "start_index": start,
                        "end_index": end,
                    }));
                }
            }
            (None, Some(file_id), Some(filename), Some(index)) => out.push(json!({
                "type": "file_citation",
                "file_id": file_id,
                "filename": filename,
                "index": index,
            })),
            (None, Some(file_id), None, Some(index)) => out.push(json!({
                "type": "file_path",
                "file_id": file_id,
                "index": index,
            })),
            _ => {}
        }
    }
    out
}

/// An absolute HTTPS URL without credentials, safe to surface as an action,
/// and (when `allowed` is set) on one of the allowed origins.
pub(crate) fn is_allowed_consent_link(link: &str, allowed: Option<&[String]>) -> bool {
    if link.is_empty() || link.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    let Ok(url) = reqwest::Url::parse(link) else {
        return false;
    };
    if url.scheme() != "https"
        || url.host_str().is_none_or(str::is_empty)
        || !url.username().is_empty()
        || url.password().is_some()
    {
        return false;
    }
    match allowed {
        None => true,
        Some(origins) => {
            let origin = url.origin().ascii_serialization();
            origins
                .iter()
                .any(|o| o.trim_end_matches('/').eq_ignore_ascii_case(&origin))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_file_citations_prefer_longer_earlier_matches_once() {
        use agent_framework_core::types::{
            FunctionResultContent, MCP_TOOL_RESULT_HOST_PAYLOAD_KEY,
        };
        let mut result = FunctionResultContent::new("c1", Some(json!("ok")));
        result.additional_properties.insert(
            MCP_TOOL_RESULT_HOST_PAYLOAD_KEY.into(),
            json!({"_meta": {"container_id": "cntr", "container_file_citations": [
                {"file_id": "f1", "filename": "report.csv"},
                {"file_id": "f2", "filename": "final report.csv"},
                {"file_id": "f3", "filename": "a.txt", "container_id": 0},
                {"file_id": "f4", "filename": "later.txt"},
                {"file_id": "", "filename": "nofile.txt"},
            ]}}),
        );
        let mut tracker = OutputTracker::new("caresp_x", None);
        tracker.note_container_files(&result);
        let annotations =
            tracker.container_file_annotations("final report.csv, report.csv and a.txt, a.txt");
        let cited: Vec<(&str, u64, u64)> = annotations
            .iter()
            .map(|a| {
                (
                    a["file_id"].as_str().unwrap(),
                    a["start_index"].as_u64().unwrap(),
                    a["end_index"].as_u64().unwrap(),
                )
            })
            .collect();
        // The longer name wins the overlap; each file is cited once.
        assert_eq!(cited, [("f2", 0, 16), ("f1", 18, 28), ("f3", 33, 38)]);
        assert_eq!(
            annotations[2]["container_id"], "cntr",
            "a falsy id falls back"
        );
        // Cited files stop pending; the rest wait for later text.
        assert!(tracker.container_file_annotations("report.csv").is_empty());
        assert_eq!(
            tracker.container_file_annotations("see later.txt")[0]["file_id"],
            "f4"
        );
    }
    use agent_framework_core::types::{
        FunctionCallContent, FunctionResultContent, Message, Role, TextContent, TextSpanRegion,
    };

    fn kinds(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(k, _)| k.as_str()).collect()
    }

    #[test]
    fn contents_fold_into_ordered_items() {
        let mut t = OutputTracker::new("caresp_x", None);
        let reasoning = agent_framework_core::types::TextReasoningContent {
            text: "thinking".into(),
            protected_data: Some("enc".into()),
            ..Default::default()
        };
        t.handle(&Content::TextReasoning(reasoning), None, false)
            .unwrap();
        t.handle(
            &Content::FunctionCall(FunctionCallContent::new("c1", "lookup", None)),
            None,
            false,
        )
        .unwrap();
        t.handle(
            &Content::FunctionResult(FunctionResultContent::new("c1", Some(json!({"a": 1})))),
            None,
            false,
        )
        .unwrap();
        t.handle(&Content::Text(TextContent::refusal("no")), None, false)
            .unwrap();
        let (items, usage, incomplete) = t.finish();
        let types: Vec<&str> = items.iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            [
                "reasoning",
                "function_call",
                "function_call_output",
                "message"
            ]
        );
        assert_eq!(items[0]["summary"][0]["text"], "thinking");
        assert_eq!(items[0]["encrypted_content"], "enc");
        assert_eq!(items[2]["output"], r#"{"a":1}"#);
        assert_eq!(
            items[3]["content"][0],
            json!({"type": "refusal", "refusal": "no"})
        );
        assert!(usage.is_none() && incomplete.is_none());
    }

    #[test]
    fn citations_become_url_annotations() {
        let mut text = TextContent::new("see source");
        text.annotations = Some(vec![Annotation {
            url: Some("https://example.com".into()),
            title: Some("Ex".into()),
            annotated_regions: Some(vec![TextSpanRegion {
                start_index: Some(4),
                end_index: Some(10),
            }]),
            ..Default::default()
        }]);
        let mut t = OutputTracker::new("caresp_x", None);
        t.handle(&Content::Text(text), None, false).unwrap();
        t.close();
        let events = t.drain();
        assert!(kinds(&events).contains(&"response.output_text.annotation.added"));
        let (items, _, _) = t.finish();
        assert_eq!(
            items[0]["content"][0]["annotations"][0],
            json!({"type": "url_citation", "url": "https://example.com", "title": "Ex", "start_index": 4, "end_index": 10})
        );
    }

    #[test]
    fn container_and_file_citations_become_their_annotations() {
        let region = || {
            Some(vec![TextSpanRegion {
                start_index: Some(0),
                end_index: Some(5),
            }])
        };
        let extras = |pairs: &[(&str, Value)]| {
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect::<serde_json::Map<String, Value>>()
        };
        let mut text = TextContent::new("plot.png");
        text.annotations = Some(vec![
            Annotation {
                file_id: Some("cfile_1".into()),
                url: Some("plot.png".into()),
                annotated_regions: region(),
                additional_properties: extras(&[("container_id", json!("cntr_1"))]),
                ..Default::default()
            },
            Annotation {
                file_id: Some("file_2".into()),
                url: Some("doc.pdf".into()),
                additional_properties: extras(&[("index", json!(3))]),
                ..Default::default()
            },
            Annotation {
                file_id: Some("file_3".into()),
                additional_properties: extras(&[("index", json!(1))]),
                ..Default::default()
            },
        ]);
        let mut t = OutputTracker::new("caresp_x", None);
        t.handle(&Content::Text(text), None, false).unwrap();
        t.close();
        let (items, _, _) = t.finish();
        assert_eq!(
            items[0]["content"][0]["annotations"],
            json!([
                {"type": "container_file_citation", "container_id": "cntr_1", "file_id": "cfile_1",
                 "filename": "plot.png", "start_index": 0, "end_index": 5},
                {"type": "file_citation", "file_id": "file_2", "filename": "doc.pdf", "index": 3},
                {"type": "file_path", "file_id": "file_3", "index": 1},
            ])
        );
    }

    #[test]
    fn a_completed_response_keeps_message_boundaries_and_usage() {
        let mut response = AgentResponse {
            messages: vec![
                Message::with_contents(
                    Role::assistant(),
                    vec![Content::text("a"), Content::text("b")],
                ),
                Message::assistant("c"),
            ],
            usage_details: Some(UsageDetails {
                input_token_count: Some(3),
                output_token_count: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        };
        response.finish_reason = Some(agent_framework_core::types::FinishReason::new("length"));
        let mut t = OutputTracker::new("caresp_x", None);
        t.handle_response(&response).unwrap();
        let (items, usage, incomplete) = t.finish();
        assert_eq!(items.len(), 2, "two messages");
        assert_eq!(
            items[0]["content"].as_array().unwrap().len(),
            2,
            "two parts"
        );
        assert_eq!(usage.unwrap()["total_tokens"], 7);
        assert_eq!(incomplete, Some("max_output_tokens"));
    }

    #[test]
    fn consent_links_must_be_safe_https() {
        assert!(is_allowed_consent_link("https://login.example.com/x", None));
        assert!(!is_allowed_consent_link("http://login.example.com/x", None));
        assert!(!is_allowed_consent_link(
            "https://u:p@login.example.com/",
            None
        ));
        assert!(!is_allowed_consent_link("javascript:alert(1)", None));
        let allowed = vec!["https://login.example.com".to_string()];
        assert!(is_allowed_consent_link(
            "https://login.example.com/x",
            Some(&allowed)
        ));
        assert!(!is_allowed_consent_link(
            "https://evil.example.com/x",
            Some(&allowed)
        ));
    }

    #[test]
    fn image_results_are_base64_payloads() {
        assert_eq!(
            image_generation_result(&json!("data:image/png;base64,aGk=")).as_deref(),
            Some("aGk=")
        );
        assert_eq!(
            image_generation_result(&json!([{"uri": "data:image/png;base64,aGk="}])).as_deref(),
            Some("aGk=")
        );
        assert_eq!(image_generation_result(&json!("https://x/y.png")), None);
    }
}

#[cfg(test)]
mod interleaving_tests {
    use super::*;
    use agent_framework_core::types::FunctionCallContent;

    fn delta(call_id: &str, name: &str, args: &str) -> Content {
        Content::FunctionCall(FunctionCallContent::new(
            call_id,
            name,
            Some(FunctionArguments::Raw(args.into())),
        ))
    }

    #[test]
    fn interleaved_call_deltas_accumulate_per_call() {
        let mut t = OutputTracker::new("caresp_x", None);
        for content in [
            delta("a", "first", r#"{"x":"#),
            delta("b", "second", r#"{"y":"#),
            delta("a", "", "1}"),
            delta("b", "", "2}"),
        ] {
            t.handle(&content, None, false).unwrap();
        }
        let (items, _, _) = t.finish();
        assert_eq!(items.len(), 2, "one item per call: {items:?}");
        assert_eq!(items[0]["call_id"], "a");
        assert_eq!(items[0]["name"], "first");
        assert_eq!(items[0]["arguments"], r#"{"x":1}"#);
        assert_eq!(items[1]["call_id"], "b");
        assert_eq!(items[1]["arguments"], r#"{"y":2}"#);
        assert!(items.iter().all(|i| i["status"] == "completed"));
    }
}
