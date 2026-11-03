//! Computer use on the Responses host (experimental `COMPUTER_USE`): an
//! agent's computer calls and results become `computer_call` /
//! `computer_call_output` output items, and a client's
//! `computer_call` / `computer_call_output` input items become computer-use
//! content (upstream `_responses.py`).

use agent_framework_core::types::{
    ComputerSafetyCheck, ComputerToolCallContent, ComputerToolResultContent, Content, DataContent,
    HostedFileContent, Message, Role, UriContent,
};
use serde_json::{json, Map, Value};

use crate::output::OutputError;
use crate::scope::ids;

/// Marks a call in the preview single-`action` form.
const ACTION_FORMAT_KEY: &str = "computer_action_format";

/// Keep a provider item id the platform accepts; otherwise mint one.
fn item_id(id: Option<&str>, prefix: &str, response_id: &str) -> String {
    match id {
        Some(id) if ids::partition_key(id).is_some() => id.to_string(),
        Some(id) => {
            tracing::warn!(id, "remapping a computer item id to an AgentServer id");
            ids::new_id(prefix, Some(response_id))
        }
        None => ids::new_id(prefix, Some(response_id)),
    }
}

fn status(status: Option<&str>, default: &str) -> Result<String, OutputError> {
    let status = status.unwrap_or(default);
    if !matches!(status, "in_progress" | "completed" | "incomplete") {
        return Err(OutputError(format!(
            "Unsupported computer item status: {status:?}"
        )));
    }
    Ok(status.to_string())
}

/// The `computer_call` output item for an agent's computer call.
pub(crate) fn call_item(
    call: &ComputerToolCallContent,
    response_id: &str,
) -> Result<Value, OutputError> {
    if call.id.is_empty() || call.call_id.is_empty() || call.actions.is_empty() {
        return Err(OutputError(
            "A computer call requires an item id, call_id, and actions.".into(),
        ));
    }
    let mut item = Map::new();
    item.insert("type".into(), json!("computer_call"));
    item.insert(
        "id".into(),
        json!(item_id(Some(&call.id), "cu", response_id)),
    );
    item.insert("call_id".into(), json!(call.call_id));
    if call.additional_properties.get(ACTION_FORMAT_KEY) == Some(&json!("single")) {
        if call.actions.len() != 1 {
            return Err(OutputError(
                "A preview computer call requires exactly one action.".into(),
            ));
        }
        item.insert("action".into(), call.actions[0].clone());
    } else {
        item.insert("actions".into(), json!(call.actions));
    }
    item.insert(
        "pending_safety_checks".into(),
        json!(call.pending_safety_checks.clone().unwrap_or_default()),
    );
    item.insert(
        "status".into(),
        json!(status(call.status.as_deref(), "completed")?),
    );
    Ok(Value::Object(item))
}

/// The `computer_call_output` output item for a computer result.
pub(crate) fn result_item(
    result: &ComputerToolResultContent,
    response_id: &str,
) -> Result<Value, OutputError> {
    let screenshot = match result.screenshot.as_deref() {
        Some(Content::Uri(u)) => json!({"type": "computer_screenshot", "image_url": u.uri}),
        Some(Content::Data(d)) => json!({"type": "computer_screenshot", "image_url": d.uri}),
        Some(Content::HostedFile(f)) => {
            json!({"type": "computer_screenshot", "file_id": f.file_id})
        }
        _ => {
            return Err(OutputError(
                "A computer result requires a call_id and screenshot.".into(),
            ))
        }
    };
    if result.call_id.is_empty() {
        return Err(OutputError(
            "A computer result requires a call_id and screenshot.".into(),
        ));
    }
    let mut item = Map::new();
    item.insert("type".into(), json!("computer_call_output"));
    item.insert(
        "id".into(),
        json!(item_id(result.id.as_deref(), "cuo", response_id)),
    );
    item.insert("call_id".into(), json!(result.call_id));
    item.insert("output".into(), screenshot);
    item.insert(
        "status".into(),
        json!(status(result.status.as_deref(), "completed")?),
    );
    if let Some(checks) = &result.acknowledged_safety_checks {
        item.insert("acknowledged_safety_checks".into(), json!(checks));
    }
    Ok(Value::Object(item))
}

fn safety_checks(value: Option<&Value>) -> Result<Option<Vec<ComputerSafetyCheck>>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(v) => serde_json::from_value(v.clone())
            .map(Some)
            .map_err(|_| "Computer safety checks require an id.".to_string()),
    }
}

fn screenshot_from_output(output: &Value) -> Result<Content, String> {
    if output.get("type").and_then(Value::as_str) != Some("computer_screenshot") {
        return Err("Computer call output must contain a computer screenshot.".into());
    }
    let image_url = output
        .get("image_url")
        .and_then(Value::as_str)
        .filter(|u| !u.is_empty());
    let file_id = output
        .get("file_id")
        .and_then(Value::as_str)
        .filter(|f| !f.is_empty());
    Ok(match (image_url, file_id) {
        (Some(url), _) if url.starts_with("data:") => Content::Data(DataContent {
            uri: url.into(),
            media_type: DataContent::media_type_from_uri(url).ok(),
        }),
        (Some(url), _) => Content::Uri(UriContent {
            uri: url.into(),
            media_type: "image/png".into(),
        }),
        (None, Some(file_id)) => Content::HostedFile(HostedFileContent {
            file_id: file_id.into(),
        }),
        (None, None) => {
            return Err("Computer screenshot is missing its image URL or file ID.".into())
        }
    })
}

/// A client's `computer_call` / `computer_call_output` input item as a
/// message (assistant call / tool result); `None` for any other item.
pub(crate) fn item_to_message(item: &Value) -> Result<Option<Message>, String> {
    let str_field = |key: &str| item.get(key).and_then(Value::as_str).unwrap_or_default();
    match item.get("type").and_then(Value::as_str) {
        Some("computer_call") => {
            let (actions, single) = match (item.get("actions"), item.get("action")) {
                (Some(Value::Array(actions)), _) if !actions.is_empty() => (actions.clone(), false),
                (_, Some(action)) if !action.is_null() => (vec![action.clone()], true),
                _ => return Err("Computer call is missing its ordered actions.".into()),
            };
            if str_field("id").is_empty() || str_field("call_id").is_empty() {
                return Err("Computer call is missing its item id or call_id.".into());
            }
            let mut additional_properties = Map::new();
            if single {
                additional_properties.insert(ACTION_FORMAT_KEY.into(), json!("single"));
            }
            Ok(Some(Message::with_contents(
                Role::assistant(),
                vec![Content::ComputerToolCall(ComputerToolCallContent {
                    id: str_field("id").into(),
                    call_id: str_field("call_id").into(),
                    actions,
                    status: item
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    pending_safety_checks: safety_checks(item.get("pending_safety_checks"))?,
                    user_input_request: true,
                    additional_properties,
                })],
            )))
        }
        Some("computer_call_output") => {
            if str_field("call_id").is_empty() {
                return Err("Computer call output is missing its call_id.".into());
            }
            let screenshot = screenshot_from_output(item.get("output").unwrap_or(&Value::Null))?;
            Ok(Some(Message::with_contents(
                Role::tool(),
                vec![Content::ComputerToolResult(ComputerToolResultContent {
                    call_id: str_field("call_id").into(),
                    id: item.get("id").and_then(Value::as_str).map(str::to_string),
                    status: item
                        .get("status")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    screenshot: Some(Box::new(screenshot)),
                    acknowledged_safety_checks: safety_checks(
                        item.get("acknowledged_safety_checks"),
                    )?,
                    additional_properties: Map::new(),
                })],
            )))
        }
        _ => Ok(None),
    }
}
