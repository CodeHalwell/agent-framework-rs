//! Computer-use pauses for [`AgentExecutor`](super::AgentExecutor)
//! (experimental `COMPUTER_USE`).
//!
//! When an agent's reply carries pending computer calls, the executor pauses
//! with one `request_info` whose payload is a [`ComputerUseRequest`]: the
//! conversation so far (reply included) and the pending calls. The
//! application runs the actions and answers with the results — one
//! [`ComputerToolResultContent`] per call, as a single value or an array.
//! Each result must answer its call and acknowledge exactly the call's
//! pending safety checks (upstream `_validate_computer_tool_result`). The
//! executor then resumes the agent with the results appended as a tool
//! message.
//!
//! Like [`AgentApprovalExecutor`](super::AgentApprovalExecutor), the state
//! rides in the request payload, so a pause survives checkpoint and resume.
//!
//! **Cancelling.** A call can be cancelled instead of answered: reply with
//! `{"call_id": "...", "cancelled": true}` in its place, or with
//! `{"cancelled": true}` to cancel every pending call. As upstream, a
//! cancelled call ends the turn: the agent is not run again, each cancelled
//! call becomes an [`ErrorContent`](crate::types::ErrorContent) (error code
//! [`CANCELLED_COMPUTER_CALL`]) after any results, and the executor yields
//! the conversation as its output without sending it downstream.
//!
//! **Streaming.** While a run carries a pending computer call, its later
//! updates are held back and emitted after the run ends without the still
//! pending calls (upstream's deferred computer updates), so the pause's
//! `request_info` is the only event carrying them.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::types::{ComputerToolCallContent, ComputerToolResultContent, Content, Message, Role};
use crate::workflow::{RequestResponse, WorkflowContext};

/// The error code of the [`ErrorContent`](crate::types::ErrorContent) a
/// cancelled computer call becomes.
pub const CANCELLED_COMPUTER_CALL: &str = "cancelled_computer_call";

/// What an answer to a computer-use pause asks the executor to do.
pub(super) enum Resumption {
    /// Run the agent again over this conversation (results appended).
    Resume(Value),
    /// A call was cancelled: end the turn with this conversation.
    Cancelled(Vec<Message>),
}

/// The `request_info` payload of a computer-use pause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputerUseRequest {
    /// The conversation so far, including the reply carrying the calls.
    pub conversation: Vec<Message>,
    /// The pending computer calls, in reply order.
    pub computer_calls: Vec<ComputerToolCallContent>,
}

/// The reply's computer calls still awaiting the application.
fn pending_calls(reply: &[Message]) -> Vec<ComputerToolCallContent> {
    let contents = || reply.iter().flat_map(|m| m.contents.iter());
    let answered: std::collections::HashSet<&str> = contents()
        .filter_map(|c| match c {
            Content::ComputerToolResult(r) => Some(r.call_id.as_str()),
            _ => None,
        })
        .collect();
    contents()
        .filter_map(|c| match c {
            Content::ComputerToolCall(call)
                if call.user_input_request && !answered.contains(call.call_id.as_str()) =>
            {
                Some(call.clone())
            }
            _ => None,
        })
        .collect()
}

/// Pause with a [`ComputerUseRequest`] when the reply has pending computer
/// calls; `true` when paused.
pub(super) async fn pause_for_pending_calls(
    conversation: &[Message],
    reply: &[Message],
    ctx: &WorkflowContext,
) -> Result<bool> {
    let computer_calls = pending_calls(reply);
    if computer_calls.is_empty() {
        return Ok(false);
    }
    let request = ComputerUseRequest {
        conversation: conversation.to_vec(),
        computer_calls,
    };
    let payload = serde_json::to_value(&request)
        .map_err(|e| Error::Workflow(format!("failed to serialize computer-use request: {e}")))?;
    ctx.request_info(payload).await?;
    Ok(true)
}

/// One entry of an answer: a result, or the cancellation of a call.
enum Answer {
    Result(ComputerToolResultContent),
    Cancelled(String),
}

fn parse_answer(value: &Value) -> Result<Answer> {
    if value.get("cancelled").and_then(Value::as_bool) == Some(true) {
        let call_id = value
            .get("call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Workflow("a cancellation names its call_id".into()))?;
        return Ok(Answer::Cancelled(call_id.to_string()));
    }
    parse_result(value).map(Answer::Result)
}

/// Parse one result: the bare struct or its `Content` form.
fn parse_result(value: &Value) -> Result<ComputerToolResultContent> {
    if let Ok(Content::ComputerToolResult(result)) =
        serde_json::from_value::<Content>(value.clone())
    {
        return Ok(result);
    }
    serde_json::from_value(value.clone())
        .map_err(|e| Error::Workflow(format!("invalid computer tool result: {e}")))
}

/// If `message` answers a computer-use pause, validate it and say how to
/// continue: resume the agent with the results appended as a tool message,
/// or end the turn when a call was cancelled. `Ok(None)` when the message is
/// not such an answer.
pub(super) fn resume_conversation(message: &Value) -> Result<Option<Resumption>> {
    let Some(response) = RequestResponse::from_message(message) else {
        return Ok(None);
    };
    let Ok(request) = serde_json::from_value::<ComputerUseRequest>(response.original_request)
    else {
        return Ok(None);
    };
    let cancel_all = response.data.get("cancelled").and_then(Value::as_bool) == Some(true)
        && response.data.get("call_id").is_none();
    let answers = if cancel_all {
        request
            .computer_calls
            .iter()
            .map(|c| Answer::Cancelled(c.call_id.clone()))
            .collect()
    } else {
        match &response.data {
            Value::Array(items) => items.iter().map(parse_answer).collect::<Result<Vec<_>>>()?,
            single => vec![parse_answer(single)?],
        }
    };
    if answers.len() != request.computer_calls.len() {
        return Err(Error::Workflow(format!(
            "expected {} computer result(s), got {}",
            request.computer_calls.len(),
            answers.len()
        )));
    }
    // Answers follow the calls' order whatever order they arrive in.
    let mut results = Vec::new();
    let mut cancelled = Vec::new();
    for call in &request.computer_calls {
        let answer = answers
            .iter()
            .find(|a| match a {
                Answer::Result(r) => r.call_id == call.call_id,
                Answer::Cancelled(id) => *id == call.call_id,
            })
            .ok_or_else(|| {
                Error::Workflow(format!("no result for computer call {}", call.call_id))
            })?;
        match answer {
            Answer::Result(result) => {
                call.validate_result(result)
                    .map_err(|e| Error::Workflow(e.to_string()))?;
                results.push(Content::ComputerToolResult(result.clone()));
            }
            Answer::Cancelled(id) => cancelled.push(Content::Error(crate::types::ErrorContent {
                message: Some(format!(
                    "Computer call {id} was cancelled without a result."
                )),
                error_code: Some(CANCELLED_COMPUTER_CALL.to_string()),
                details: None,
            })),
        }
    }
    let mut conversation = request.conversation;
    // The answered calls no longer await the application.
    for message in &mut conversation {
        for content in &mut message.contents {
            if let Content::ComputerToolCall(call) = content {
                if request
                    .computer_calls
                    .iter()
                    .any(|c| c.call_id == call.call_id)
                {
                    call.user_input_request = false;
                }
            }
        }
    }
    if !results.is_empty() {
        conversation.push(Message::with_contents(Role::tool(), results));
    }
    if !cancelled.is_empty() {
        conversation.push(Message::with_contents(Role::assistant(), cancelled));
        return Ok(Some(Resumption::Cancelled(conversation)));
    }
    serde_json::to_value(&conversation)
        .map(|v| Some(Resumption::Resume(v)))
        .map_err(|e| Error::Workflow(format!("failed to serialize conversation: {e}")))
}

/// Whether a streamed update carries a computer call still awaiting the
/// application.
pub(super) fn has_pending_call(contents: &[Content]) -> bool {
    contents
        .iter()
        .any(|c| matches!(c, Content::ComputerToolCall(call) if call.user_input_request))
}

/// A held-back update, emitted after the run: calls the run completed are no
/// longer pending, and the still-pending ones are dropped (the pause carries
/// them). `None` when nothing is left.
pub(super) fn finalize_deferred(
    mut contents: Vec<Content>,
    completed: &std::collections::HashSet<String>,
) -> Option<Vec<Content>> {
    for content in &mut contents {
        if let Content::ComputerToolCall(call) = content {
            if completed.contains(&call.call_id) {
                call.user_input_request = false;
            }
        }
    }
    contents.retain(|c| !matches!(c, Content::ComputerToolCall(call) if call.user_input_request));
    (!contents.is_empty()).then_some(contents)
}
