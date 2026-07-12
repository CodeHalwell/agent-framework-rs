//! Sequential orchestration: a pipeline of agents that each see and extend the
//! running conversation. Rust analogue of `_sequential.py`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::{AgentApprovalExecutor, AgentExecutor};
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::workflow::{Executor, Workflow, WorkflowBuilder};

/// Builder for a sequential pipeline of agents. Rust analogue of
/// `SequentialBuilder`. Each participant sees the running conversation and
/// appends its reply; the final conversation is yielded as output.
#[derive(Default)]
pub struct SequentialBuilder {
    participants: Vec<Arc<dyn SupportsAgentRun>>,
    name: Option<String>,
    output_from: Vec<String>,
    intermediate_output_from: Vec<String>,
    request_info: bool,
}

impl SequentialBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the ordered list of participants.
    pub fn participants(
        mut self,
        agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>,
    ) -> Self {
        self.participants = agents.into_iter().collect();
        self
    }

    /// Append a participant to the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, agent: Arc<dyn SupportsAgentRun>) -> Self {
        self.participants.push(agent);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Designate participants (by [`SupportsAgentRun::id`]) whose turn output
    /// becomes a terminal [`WorkflowEvent::Output`](crate::workflow::WorkflowEvent::Output)
    /// event, resolved to the pipeline's internal executor ids at
    /// [`build`](Self::build) time and forwarded to
    /// [`WorkflowBuilder::output_from`].
    ///
    /// When neither this nor [`Self::intermediate_output_from`] is called,
    /// the builder preserves its current default: only the *last* participant
    /// yields output. Once either is called, only the designated
    /// participants yield at all — every participant still forwards the
    /// running conversation to the next stage regardless of designation, so
    /// the pipeline itself is unaffected.
    ///
    /// Rejected at `build()` if an id does not match any registered
    /// participant, or overlaps [`Self::intermediate_output_from`].
    pub fn output_from(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.output_from.extend(ids.into_iter().map(Into::into));
        self
    }

    /// Designate participants (by [`SupportsAgentRun::id`]) whose turn output
    /// becomes a non-terminal
    /// [`WorkflowEvent::Intermediate`](crate::workflow::WorkflowEvent::Intermediate)
    /// event rather than the workflow's final output. See
    /// [`Self::output_from`] for the full designation semantics.
    pub fn intermediate_output_from(
        mut self,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.intermediate_output_from
            .extend(ids.into_iter().map(Into::into));
        self
    }

    /// Opt in to post-agent human approval: every participant's turn pauses
    /// the workflow (a [`WorkflowEvent::RequestInfo`](crate::workflow::WorkflowEvent::RequestInfo)
    /// event, surfaced as a [`PendingRequest`](crate::workflow::PendingRequest))
    /// until a response is supplied. An empty response approves the reply and
    /// the pipeline continues to the next participant; a non-empty response is
    /// treated as revision feedback and re-invokes that participant, pausing
    /// again — repeating until approved. Rust analogue of upstream's
    /// post-agent `.with_request_info()` (see `UPSTREAM_DRIFT.md` §12),
    /// implemented via [`AgentApprovalExecutor`]. Default (not called): no
    /// pausing, matching prior behavior.
    pub fn with_request_info(mut self) -> Self {
        self.request_info = true;
        self
    }

    /// Validate and build the sequential workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "sequential workflow needs at least one participant".into(),
            ));
        }

        let n = self.participants.len();
        let last = n - 1;
        let mut ids = Vec::with_capacity(n);
        let mut agent_id_to_exec: HashMap<String, String> = HashMap::new();
        for (i, agent) in self.participants.iter().enumerate() {
            let exec_id = format!("seq_{i}");
            agent_id_to_exec.insert(agent.id().to_string(), exec_id.clone());
            ids.push(exec_id);
        }

        let resolve = |names: &[String], label: &str| -> Result<Vec<String>> {
            names
                .iter()
                .map(|n| {
                    agent_id_to_exec.get(n).cloned().ok_or_else(|| {
                        Error::Workflow(format!(
                            "{label} references unknown participant agent id '{n}'"
                        ))
                    })
                })
                .collect()
        };
        let output_exec_ids = resolve(&self.output_from, "output_from")?;
        let intermediate_exec_ids =
            resolve(&self.intermediate_output_from, "intermediate_output_from")?;
        let designated: HashSet<&str> = output_exec_ids
            .iter()
            .chain(intermediate_exec_ids.iter())
            .map(String::as_str)
            .collect();
        let has_designation = !designated.is_empty();

        let mut builder = WorkflowBuilder::new();
        for (i, agent) in self.participants.into_iter().enumerate() {
            let exec_id = ids[i].clone();
            let emit = if has_designation {
                designated.contains(exec_id.as_str())
            } else {
                i == last
            };
            let exec: Arc<dyn Executor> = if self.request_info {
                Arc::new(
                    AgentApprovalExecutor::new(exec_id.clone(), agent)
                        .with_output(emit)
                        .with_also_send(has_designation),
                )
            } else {
                Arc::new(
                    AgentExecutor::new(exec_id.clone(), agent)
                        .with_output(emit)
                        .with_also_send(has_designation),
                )
            };
            builder = builder.add_executor(exec);
        }
        builder = builder.set_start(ids[0].clone()).add_chain(ids);
        if !output_exec_ids.is_empty() {
            builder = builder.output_from(output_exec_ids);
        }
        if !intermediate_exec_ids.is_empty() {
            builder = builder.intermediate_output_from(intermediate_exec_ids);
        }
        if let Some(name) = self.name {
            builder = builder.name(name);
        }
        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::Agent;
    use crate::client::{ChatClient, ChatStream};
    use crate::types::{ChatOptions, ChatResponse, ChatResponseUpdate, Message};
    use crate::workflow::WorkflowEvent;
    use async_trait::async_trait;
    use futures::StreamExt;

    /// A chat client that always returns the same canned text.
    struct EchoClient(String);

    #[async_trait]
    impl ChatClient for EchoClient {
        async fn get_response(
            &self,
            _messages: Vec<Message>,
            _options: ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse::from_text(self.0.clone()))
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

    fn agent(id: &str, reply: &str) -> Arc<dyn SupportsAgentRun> {
        Arc::new(
            Agent::builder(EchoClient(reply.to_string()))
                .id(id)
                .name(id)
                .build(),
        )
    }

    /// Designating an interior participant as the output source makes its
    /// turn the workflow's terminal output, while the later, otherwise-final
    /// participant becomes a non-terminal intermediate event instead.
    #[tokio::test]
    async fn output_from_redesignates_terminal_output() {
        let a = agent("a", "reply-A");
        let b = agent("b", "reply-B");
        let c = agent("c", "reply-C");

        let workflow = SequentialBuilder::new()
            .participants(vec![a, b, c])
            .output_from(["b"])
            .intermediate_output_from(["c"])
            .build()
            .unwrap();

        let run = workflow.run("start").await.unwrap();

        // The final output is B's turn, not C's (the default last stage).
        let output = run.last_output().expect("a final output");
        let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
        let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
        assert!(texts.contains(&"reply-A".to_string()));
        assert!(texts.contains(&"reply-B".to_string()));
        assert!(
            !texts.contains(&"reply-C".to_string()),
            "C's turn should not be part of the terminal output: {texts:?}"
        );

        // C's turn shows up as a non-terminal Intermediate event instead.
        let intermediate_texts: Vec<String> = run
            .events()
            .iter()
            .filter_map(|e| match e {
                WorkflowEvent::Intermediate { data, .. } => {
                    let conv: Vec<Message> = serde_json::from_value(data.clone()).ok()?;
                    conv.last().map(|m| m.text())
                }
                _ => None,
            })
            .collect();
        assert!(
            intermediate_texts.iter().any(|t| t == "reply-C"),
            "expected an Intermediate event carrying C's turn: {intermediate_texts:?}"
        );

        // Only one terminal Output event was emitted (B's turn).
        let output_events = run
            .events()
            .iter()
            .filter(|e| matches!(e, WorkflowEvent::Output { .. }))
            .count();
        assert_eq!(output_events, 1);
    }

    /// Unknown ids passed to output_from are rejected at build time.
    #[tokio::test]
    async fn output_from_rejects_unknown_id() {
        let a = agent("a", "reply-A");
        let err = match SequentialBuilder::new()
            .participants(vec![a])
            .output_from(["not-a-participant"])
            .build()
        {
            Ok(_) => panic!("expected an error"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("not-a-participant"));
    }
}
