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
//! Divergence: upstream also lets an application cancel a pending call (an
//! error result) and defers the agent's own computer updates while paused;
//! here every pending call must be answered.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::types::{ComputerToolCallContent, ComputerToolResultContent, Content, Message, Role};
use crate::workflow::{RequestResponse, WorkflowContext};

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

/// If `message` answers a computer-use pause, validate the results and return
/// the conversation to resume the agent with (the results appended as a tool
/// message). `Ok(None)` when the message is not such an answer.
pub(super) fn resume_conversation(message: &Value) -> Result<Option<Value>> {
    let Some(response) = RequestResponse::from_message(message) else {
        return Ok(None);
    };
    let Ok(request) = serde_json::from_value::<ComputerUseRequest>(response.original_request)
    else {
        return Ok(None);
    };
    let results = match &response.data {
        Value::Array(items) => items.iter().map(parse_result).collect::<Result<Vec<_>>>()?,
        single => vec![parse_result(single)?],
    };
    if results.len() != request.computer_calls.len() {
        return Err(Error::Workflow(format!(
            "expected {} computer result(s), got {}",
            request.computer_calls.len(),
            results.len()
        )));
    }
    // Results follow the calls' order whatever order they arrive in.
    let mut ordered = Vec::with_capacity(results.len());
    for call in &request.computer_calls {
        let result = results
            .iter()
            .find(|r| r.call_id == call.call_id)
            .ok_or_else(|| {
                Error::Workflow(format!("no result for computer call {}", call.call_id))
            })?;
        call.validate_result(result)
            .map_err(|e| Error::Workflow(e.to_string()))?;
        ordered.push(Content::ComputerToolResult(result.clone()));
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
    conversation.push(Message::with_contents(Role::tool(), ordered));
    serde_json::to_value(&conversation)
        .map(Some)
        .map_err(|e| Error::Workflow(format!("failed to serialize conversation: {e}")))
}
