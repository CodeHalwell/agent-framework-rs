//! Expose a built [`Workflow`] as an [`Agent`]. Rust analogue of
//! `_workflows/_agent.py` (`WorkflowAgent`).
//!
//! [`WorkflowAgent::run`] feeds the input messages to the workflow (as the
//! JSON-serialized conversation), runs it, and maps the workflow's
//! [`Output`](WorkflowEvent::Output) events into the response messages:
//! `Vec<ChatMessage>` / `ChatMessage` / string payloads become messages, other
//! payloads are JSON-stringified. Any pending request-info events are surfaced
//! as `user_input_requests` on the response (mirroring Python, which maps
//! `RequestInfoEvent` to a `request_info` function-approval request).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::Value;

use crate::agent::{Agent, AgentRunStream};
use crate::error::Result;
use crate::threads::AgentThread;
use crate::tools::{AiFunction, ToolDefinition};
use crate::types::{
    AgentRunResponse, AgentRunResponseUpdate, ChatMessage, Content, FunctionApprovalRequestContent,
    FunctionArguments, FunctionCallContent, Role,
};
use crate::workflow::{Workflow, WorkflowEvent};

/// The synthetic function name used to surface a pending request as a
/// user-input request (matches Python's `REQUEST_INFO_FUNCTION_NAME`).
const REQUEST_INFO_FUNCTION_NAME: &str = "request_info";

/// An [`Agent`] that wraps a [`Workflow`] and exposes it through the agent
/// interface. Rust analogue of `WorkflowAgent`.
#[derive(Clone)]
pub struct WorkflowAgent {
    workflow: Workflow,
    id: String,
    name: Option<String>,
    description: Option<String>,
}

impl WorkflowAgent {
    /// Wrap `workflow` as an agent named `name`.
    pub fn new(workflow: Workflow, name: impl Into<String>) -> Self {
        let name = name.into();
        Self {
            workflow,
            id: format!("workflow_agent_{}", uuid::Uuid::new_v4().simple()),
            name: Some(name),
            description: None,
        }
    }

    /// Set an explicit id.
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }

    /// Set a description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// The wrapped workflow.
    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    /// The agent description, if any.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Stream the workflow's agent activity as [`AgentRunResponseUpdate`]s.
    ///
    /// Maps the engine's `run_stream` events: each
    /// [`AgentRunUpdate`](WorkflowEvent::AgentRunUpdate) carries an agent message
    /// (emitted as a delta) and each [`RequestInfo`](WorkflowEvent::RequestInfo)
    /// becomes a user-input request update. Other events are ignored.
    pub fn run_stream(&self, messages: Vec<ChatMessage>) -> AgentRunStream {
        let input = serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(Vec::new()));
        let name = self.name.clone();
        let stream = self.workflow.run_stream(input);
        let mapped = stream.filter_map(move |event| {
            let name = name.clone();
            async move { convert_event(&event, name.as_deref()).map(Ok) }
        });
        Box::pin(mapped)
    }

    /// Wrap this workflow-agent as a [`ToolDefinition`] callable by another
    /// agent (mirrors [`ChatAgent::as_tool`](crate::agent::ChatAgent::as_tool)).
    /// The tool takes a single `task` string and returns the response text.
    pub fn as_tool(&self) -> ToolDefinition {
        let agent = Arc::new(self.clone());
        let tool_name = self.name.clone().unwrap_or_else(|| self.id.clone());
        let description = self.description.clone().unwrap_or_default();
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "task": { "type": "string", "description": format!("Task for {tool_name}") } },
            "required": ["task"],
        });
        AiFunction::new(tool_name, description, schema, move |args: Value| {
            let agent = agent.clone();
            async move {
                let task = args
                    .get("task")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let response = agent.run(vec![ChatMessage::user(task)], None).await?;
                Ok(Value::String(response.text()))
            }
        })
        .into_definition()
    }

    /// Map the workflow's outputs into response messages.
    fn outputs_to_messages(&self, outputs: Vec<Value>) -> Vec<ChatMessage> {
        let mut messages = Vec::new();
        for out in outputs {
            if let Ok(msgs) = serde_json::from_value::<Vec<ChatMessage>>(out.clone()) {
                messages.extend(msgs);
                continue;
            }
            if let Ok(msg) = serde_json::from_value::<ChatMessage>(out.clone()) {
                messages.push(msg);
                continue;
            }
            match out {
                Value::String(s) => messages.push(ChatMessage::assistant(s)),
                other => messages.push(ChatMessage::assistant(other.to_string())),
            }
        }
        messages
    }
}

/// Convert a pending request into a user-input (function-approval) message.
fn request_message(request_id: &str, data: &Value, name: Option<&str>) -> ChatMessage {
    let mut args: HashMap<String, Value> = HashMap::new();
    args.insert(
        "request_id".to_string(),
        Value::String(request_id.to_string()),
    );
    args.insert("data".to_string(), data.clone());
    let call = FunctionCallContent::new(
        request_id,
        REQUEST_INFO_FUNCTION_NAME,
        Some(FunctionArguments::Object(args)),
    );
    let approval = FunctionApprovalRequestContent {
        id: request_id.to_string(),
        function_call: call.clone(),
    };
    let mut msg = ChatMessage::with_contents(
        Role::assistant(),
        vec![
            Content::FunctionCall(call),
            Content::FunctionApprovalRequest(approval),
        ],
    );
    msg.author_name = name.map(str::to_string);
    msg
}

/// Convert a live workflow event into an agent update (used by `run_stream`).
fn convert_event(event: &WorkflowEvent, name: Option<&str>) -> Option<AgentRunResponseUpdate> {
    match event {
        WorkflowEvent::AgentRunUpdate { update, .. } => {
            let msg: ChatMessage = serde_json::from_value(update.clone()).ok()?;
            Some(AgentRunResponseUpdate {
                contents: msg.contents,
                role: Some(msg.role),
                author_name: msg.author_name.or_else(|| name.map(str::to_string)),
                message_id: msg.message_id,
                ..Default::default()
            })
        }
        WorkflowEvent::RequestInfo {
            request_id,
            request_data,
            ..
        } => {
            let msg = request_message(request_id, request_data, name);
            Some(AgentRunResponseUpdate {
                contents: msg.contents,
                role: Some(msg.role),
                author_name: msg.author_name,
                ..Default::default()
            })
        }
        _ => None,
    }
}

#[async_trait]
impl Agent for WorkflowAgent {
    async fn run(
        &self,
        messages: Vec<ChatMessage>,
        _thread: Option<&mut AgentThread>,
    ) -> Result<AgentRunResponse> {
        let input = serde_json::to_value(&messages).unwrap_or_else(|_| Value::Array(Vec::new()));
        let run = self.workflow.run(input).await?;

        let mut response_messages = self.outputs_to_messages(run.outputs());

        // Surface any outstanding requests as user-input requests.
        for pending in run.pending_requests() {
            response_messages.push(request_message(
                &pending.request_id,
                &pending.request_data,
                self.name.as_deref(),
            ));
        }

        Ok(AgentRunResponse {
            messages: response_messages,
            ..Default::default()
        })
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
}

/// Extension trait adding [`Workflow::as_agent`](WorkflowAgentExt::as_agent) so
/// a built workflow can be exposed as an [`Agent`] fluently.
pub trait WorkflowAgentExt {
    /// Wrap this workflow as a [`WorkflowAgent`] named `name`.
    fn as_agent(&self, name: impl Into<String>) -> WorkflowAgent;
}

impl WorkflowAgentExt for Workflow {
    fn as_agent(&self, name: impl Into<String>) -> WorkflowAgent {
        WorkflowAgent::new(self.clone(), name)
    }
}
