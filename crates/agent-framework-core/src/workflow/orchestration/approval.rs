//! Post-agent human approval. Rust analogue of upstream's shared
//! `AgentApprovalExecutor` (`agent_framework_orchestrations`), which replaced
//! the old pre-agent `RequestInfoInterceptor` (see `UPSTREAM_DRIFT.md` §12).
//!
//! The old engine paused *before* an agent ran, handing the human a bare
//! `str` response. The new engine pauses **after** the agent responds: the
//! human reviews the agent's actual reply and answers with a structured
//! response where an *empty* value means "approved" (the reply is forwarded
//! downstream as-is) and a *non-empty* value means "revise" (treated as
//! feedback, the agent is re-invoked, and a fresh approval request is raised)
//! — enabling **iterate-until-approved** loops.
//!
//! ## Divergence from Python
//!
//! Upstream wraps a two-node sub-workflow (an agent node feeding a dedicated
//! approval node) and keeps loop state in that sub-workflow. This port keeps
//! [`AgentApprovalExecutor`] a single [`Executor`] — matching the rest of this
//! crate's orchestration style (see the [module docs](super)) — and threads
//! all loop state through the request/response payload itself rather than
//! executor-local state: the `request_info` payload carries the
//! conversation-so-far plus the agent's pending reply, and the human's
//! response (routed back to this same executor as a message, per
//! [`WorkflowContext::request_info`]) carries a [`RequestResponse`] whose
//! `original_request` is exactly that payload. The executor itself is
//! therefore stateless — nothing to snapshot/restore — while still
//! supporting a full multi-round approve/revise loop (not a one-shot
//! approval), and it survives checkpoint/resume for free because
//! `PendingRequest::request_data` (which carries this payload) is already
//! part of `WorkflowCheckpoint`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{parse_conversation, run_agent_and_emit};
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::types::Message;
use crate::workflow::{Executor, RequestResponse, WorkflowContext};

/// The `request_info` payload emitted while an agent's reply awaits human
/// approval: the conversation the agent replied to, plus its pending reply.
/// Round-trips through [`RequestResponse::original_request`] so the executor
/// needs no state of its own.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// The conversation the agent replied to (excludes the pending reply).
    pub conversation: Vec<Message>,
    /// The agent's reply, awaiting approval.
    pub reply: Vec<Message>,
}

/// True when a human response counts as approval: `null`, an empty string
/// (after trimming), or an empty array/object. Anything else — including a
/// non-empty string, number, `false`/`true`, or a non-empty array/object — is
/// treated as revision feedback.
fn is_approval(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::String(s) => s.trim().is_empty(),
        Value::Array(a) => a.is_empty(),
        Value::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// Render a non-empty response value as feedback text appended to the
/// conversation. Strings are used verbatim; anything else is rendered as JSON
/// so structured feedback (e.g. `{"comment": "..."}`) is not silently lost.
fn feedback_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// An [`Executor`] that runs an [`SupportsAgentRun`] over the incoming
/// conversation, then pauses for human approval of its reply before
/// forwarding it downstream. Rust analogue of `AgentApprovalExecutor`.
///
/// A non-empty human response is treated as revision feedback: the agent is
/// re-invoked with the feedback appended, and a new approval request is
/// raised — repeating until an empty (approving) response arrives.
pub struct AgentApprovalExecutor {
    id: String,
    agent: Arc<dyn SupportsAgentRun>,
    /// When true, yield the approved conversation as a workflow output (in
    /// addition to forwarding it downstream when [`Self::also_send`] is set).
    /// Mirrors [`AgentExecutor::with_output`](super::AgentExecutor::with_output).
    emit_output: bool,
    /// When true, forward the approved conversation downstream even though
    /// `emit_output` is also set. Mirrors
    /// [`AgentExecutor::with_also_send`](super::AgentExecutor::with_also_send).
    also_send: bool,
}

impl AgentApprovalExecutor {
    /// Wrap `agent` in a post-agent approval gate with the given executor id.
    pub fn new(id: impl Into<String>, agent: Arc<dyn SupportsAgentRun>) -> Self {
        Self {
            id: id.into(),
            agent,
            emit_output: false,
            also_send: false,
        }
    }

    /// When set, yield the approved conversation as a workflow output. See
    /// [`AgentExecutor::with_output`](super::AgentExecutor::with_output).
    pub fn with_output(mut self, emit_output: bool) -> Self {
        self.emit_output = emit_output;
        self
    }

    /// When set, forward the approved conversation downstream even if
    /// [`Self::with_output`] is also set. See
    /// [`AgentExecutor::with_also_send`](super::AgentExecutor::with_also_send).
    pub fn with_also_send(mut self, also_send: bool) -> Self {
        self.also_send = also_send;
        self
    }

    /// Run the inner agent over `conversation` and raise a fresh approval
    /// request for its reply.
    async fn run_and_request(
        &self,
        conversation: Vec<Message>,
        ctx: &WorkflowContext,
    ) -> Result<()> {
        let response =
            run_agent_and_emit(&self.agent, conversation.clone(), &self.id, &self.id, ctx).await?;
        let request = ApprovalRequest {
            conversation,
            reply: response.messages,
        };
        let payload = serde_json::to_value(&request)
            .map_err(|e| Error::Workflow(format!("failed to serialize approval request: {e}")))?;
        ctx.request_info(payload).await
    }

    /// Handle a routed-back human response to an outstanding approval
    /// request: forward on approval, or re-invoke the agent on revision.
    async fn handle_response(&self, resp: RequestResponse, ctx: &WorkflowContext) -> Result<()> {
        let request: ApprovalRequest = serde_json::from_value(resp.original_request)
            .map_err(|e| Error::Workflow(format!("invalid approval request payload: {e}")))?;

        if is_approval(&resp.data) {
            let mut full = request.conversation;
            full.extend(request.reply);
            let payload = serde_json::to_value(&full)
                .map_err(|e| Error::Workflow(format!("failed to serialize conversation: {e}")))?;
            if self.emit_output {
                ctx.yield_output(payload.clone()).await?;
            }
            if !self.emit_output || self.also_send {
                ctx.send_message(payload).await?;
            }
            Ok(())
        } else {
            let mut conversation = request.conversation;
            conversation.extend(request.reply);
            conversation.push(Message::user(feedback_text(&resp.data)));
            self.run_and_request(conversation, ctx).await
        }
    }
}

#[async_trait]
impl Executor for AgentApprovalExecutor {
    fn id(&self) -> &str {
        &self.id
    }

    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        if let Some(resp) = RequestResponse::from_message(&message) {
            self.handle_response(resp, &ctx).await
        } else {
            let conversation = parse_conversation(&message)?;
            self.run_and_request(conversation, &ctx).await
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::client::{ChatClient, ChatStream};
    use crate::types::{ChatOptions, ChatResponse, ChatResponseUpdate};
    use async_trait::async_trait;
    use futures::StreamExt;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn empty_values_are_approval() {
        assert!(is_approval(&Value::Null));
        assert!(is_approval(&serde_json::json!("")));
        assert!(is_approval(&serde_json::json!("   ")));
        assert!(is_approval(&serde_json::json!([])));
        assert!(is_approval(&serde_json::json!({})));
    }

    #[test]
    fn non_empty_values_are_revision() {
        assert!(!is_approval(&serde_json::json!("please redo this")));
        assert!(!is_approval(&serde_json::json!(["x"])));
        assert!(!is_approval(&serde_json::json!({"comment": "no"})));
        assert!(!is_approval(&serde_json::json!(false)));
    }

    /// A chat client that returns successive canned replies on each call,
    /// repeating the last once exhausted.
    struct ScriptedClient {
        replies: Vec<String>,
        calls: AtomicUsize,
    }

    impl ScriptedClient {
        fn new(replies: Vec<&str>) -> Self {
            Self {
                replies: replies.into_iter().map(String::from).collect(),
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl ChatClient for ScriptedClient {
        async fn get_response(
            &self,
            _messages: Vec<Message>,
            _options: ChatOptions,
        ) -> Result<ChatResponse> {
            let i = self.calls.fetch_add(1, Ordering::SeqCst);
            let text = self
                .replies
                .get(i)
                .or_else(|| self.replies.last())
                .cloned()
                .unwrap_or_default();
            Ok(ChatResponse::from_text(text))
        }

        async fn get_streaming_response(
            &self,
            messages: Vec<Message>,
            options: ChatOptions,
        ) -> Result<ChatStream> {
            let resp = self.get_response(messages, options).await?;
            let updates: Vec<Result<ChatResponseUpdate>> = resp
                .messages
                .into_iter()
                .map(|m| {
                    Ok(ChatResponseUpdate {
                        contents: m.contents,
                        role: Some(m.role),
                        ..Default::default()
                    })
                })
                .collect();
            Ok(futures::stream::iter(updates).boxed())
        }
    }

    fn scripted_agent(id: &str, replies: Vec<&str>) -> Arc<dyn SupportsAgentRun> {
        Arc::new(
            Agent::builder(ScriptedClient::new(replies))
                .id(id)
                .name(id)
                .build(),
        )
    }

    /// Drives the executor directly (no `Workflow`/runner involved): the
    /// first `execute` raises a request; feeding the routed-back response
    /// through a second `execute` either forwards the reply (approval) or
    /// raises a fresh request with a re-invoked reply (revision).
    #[tokio::test]
    async fn approve_forwards_reply_revise_re_invokes_and_requests_again() {
        use crate::workflow::shared_state::SharedState;

        let agent = scripted_agent("a", vec!["draft-1", "draft-2"]);
        let exec = AgentApprovalExecutor::new("approver", agent).with_also_send(true);

        let ctx = WorkflowContext::new("approver".into(), vec![], SharedState::default());
        exec.execute(serde_json::json!("hello"), ctx.clone())
            .await
            .unwrap();
        let (_, _, _, requests) = ctx.take();
        assert_eq!(requests.len(), 1);
        let request: ApprovalRequest = serde_json::from_value(requests[0].data.clone()).unwrap();
        assert_eq!(
            request.reply.iter().map(|m| m.text()).collect::<Vec<_>>(),
            ["draft-1"]
        );

        // Revise: fed back as a RequestResponse, this must re-invoke the
        // agent and raise a new request rather than forwarding.
        let revise = RequestResponse {
            request_id: "r1".into(),
            data: serde_json::json!("make it punchier"),
            original_request: requests[0].data.clone(),
        };
        let ctx2 = WorkflowContext::new("approver".into(), vec![], SharedState::default());
        exec.execute(serde_json::to_value(&revise).unwrap(), ctx2.clone())
            .await
            .unwrap();
        let (sent, outputs, _, requests2) = ctx2.take();
        assert!(sent.is_empty());
        assert!(outputs.is_empty());
        assert_eq!(requests2.len(), 1);
        let request2: ApprovalRequest = serde_json::from_value(requests2[0].data.clone()).unwrap();
        assert_eq!(
            request2.reply.iter().map(|m| m.text()).collect::<Vec<_>>(),
            ["draft-2"]
        );

        // Approve: an empty response forwards the reply downstream.
        let approve = RequestResponse {
            request_id: "r2".into(),
            data: Value::Null,
            original_request: requests2[0].data.clone(),
        };
        let ctx3 = WorkflowContext::new("approver".into(), vec![], SharedState::default());
        exec.execute(serde_json::to_value(&approve).unwrap(), ctx3.clone())
            .await
            .unwrap();
        let (sent, _, _, requests3) = ctx3.take();
        assert!(requests3.is_empty());
        assert_eq!(sent.len(), 1);
        let conv: Vec<Message> = serde_json::from_value(sent[0].data.clone()).unwrap();
        let texts: Vec<String> = conv.iter().map(|m| m.text()).collect();
        assert!(texts.contains(&"draft-2".to_string()));
    }
}
