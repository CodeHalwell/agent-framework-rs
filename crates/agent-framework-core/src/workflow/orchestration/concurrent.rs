//! Concurrent orchestration: fan out an input to several agents and fan their
//! replies back in. Rust analogue of `_concurrent.py`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::{parse_conversation, AgentExecutor};
use crate::agent::SupportsAgentRun;
use crate::error::{Error, Result};
use crate::types::Message;
use crate::workflow::{Executor, Workflow, WorkflowBuilder, WorkflowContext};

/// A dispatcher that broadcasts its input to all concurrent participants.
struct DispatchExecutor {
    id: String,
}

#[async_trait]
impl Executor for DispatchExecutor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        let conversation = parse_conversation(&message)?;
        let payload = serde_json::to_value(&conversation)
            .map_err(|e| Error::Workflow(format!("serialize error: {e}")))?;
        ctx.send_message(payload).await?;
        Ok(())
    }
}

/// The default aggregator: collects each participant's final conversation and
/// yields the union of the initial prompt plus each agent's last reply.
struct AggregateExecutor {
    id: String,
}

#[async_trait]
impl Executor for AggregateExecutor {
    fn id(&self) -> &str {
        &self.id
    }
    async fn execute(&self, message: Value, ctx: WorkflowContext) -> Result<()> {
        // message is an array of conversations (one per participant).
        let conversations = match &message {
            Value::Array(items) => items,
            _ => return Err(Error::Workflow("aggregator expected an array".into())),
        };
        let mut merged: Vec<Message> = Vec::new();
        let mut seeded = false;
        for conv_value in conversations {
            let conv = parse_conversation(conv_value)?;
            if !seeded {
                // Seed with everything except the last (the shared prompt).
                if conv.len() > 1 {
                    merged.extend(conv[..conv.len() - 1].iter().cloned());
                }
                seeded = true;
            }
            if let Some(last) = conv.last() {
                merged.push(last.clone());
            }
        }
        let payload = serde_json::to_value(&merged)
            .map_err(|e| Error::Workflow(format!("serialize error: {e}")))?;
        ctx.yield_output(payload).await?;
        Ok(())
    }
}

/// Builder for a concurrent fan-out/fan-in over agents. Rust analogue of
/// `ConcurrentBuilder`.
#[derive(Default)]
pub struct ConcurrentBuilder {
    participants: Vec<Arc<dyn SupportsAgentRun>>,
    name: Option<String>,
    output_from: Vec<String>,
    intermediate_output_from: Vec<String>,
}

impl ConcurrentBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the participants that run concurrently.
    pub fn participants(
        mut self,
        agents: impl IntoIterator<Item = Arc<dyn SupportsAgentRun>>,
    ) -> Self {
        self.participants = agents.into_iter().collect();
        self
    }

    /// Add a participant.
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

    /// Designate participants (by [`SupportsAgentRun::id`]) whose individual
    /// reply becomes a terminal [`WorkflowEvent::Output`](crate::workflow::WorkflowEvent::Output)
    /// event, resolved to the fan-out's internal executor ids at
    /// [`build`](Self::build) time and forwarded to
    /// [`WorkflowBuilder::output_from`].
    ///
    /// When neither this nor [`Self::intermediate_output_from`] is called,
    /// the builder preserves its current default: every participant fans
    /// into the built-in aggregator, whose merged reply is the sole output.
    /// Once either is called, designated participants *additionally* yield
    /// their own reply as output/intermediate output — every participant
    /// still fans into the aggregator regardless of designation, so if the
    /// aggregator itself is not designated, its merged reply is
    /// automatically demoted to a non-terminal
    /// [`WorkflowEvent::Intermediate`](crate::workflow::WorkflowEvent::Intermediate)
    /// event (see [`WorkflowBuilder::output_from`] for the precedence rule).
    ///
    /// Rejected at `build()` if an id does not match any registered
    /// participant, or overlaps [`Self::intermediate_output_from`].
    pub fn output_from(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.output_from.extend(ids.into_iter().map(Into::into));
        self
    }

    /// Designate participants (by [`SupportsAgentRun::id`]) whose individual
    /// reply becomes a non-terminal
    /// [`WorkflowEvent::Intermediate`](crate::workflow::WorkflowEvent::Intermediate)
    /// event. See [`Self::output_from`] for the full designation semantics.
    pub fn intermediate_output_from(
        mut self,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.intermediate_output_from
            .extend(ids.into_iter().map(Into::into));
        self
    }

    /// Validate and build the concurrent workflow.
    pub fn build(self) -> Result<Workflow> {
        if self.participants.is_empty() {
            return Err(Error::Workflow(
                "concurrent workflow needs at least one participant".into(),
            ));
        }

        let mut agent_id_to_exec: HashMap<String, String> = HashMap::new();
        for (i, agent) in self.participants.iter().enumerate() {
            agent_id_to_exec.insert(agent.id().to_string(), format!("agent_{i}"));
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

        let mut builder = WorkflowBuilder::new()
            .add_executor(Arc::new(DispatchExecutor {
                id: "dispatch".into(),
            }))
            .add_executor(Arc::new(AggregateExecutor {
                id: "aggregate".into(),
            }))
            .set_start("dispatch");

        let mut agent_ids = Vec::new();
        for (i, agent) in self.participants.into_iter().enumerate() {
            let id = format!("agent_{i}");
            let exec = AgentExecutor::new(id.clone(), agent)
                .with_output(designated.contains(id.as_str()))
                // Participants must always feed the fan-in barrier,
                // regardless of designation — otherwise the aggregator (and
                // the run) would hang waiting on a source that never sends.
                .with_also_send(true);
            builder = builder.add_executor(Arc::new(exec));
            agent_ids.push(id);
        }
        builder = builder.add_fan_out("dispatch", agent_ids.clone());
        builder = builder.add_fan_in(agent_ids, "aggregate");
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
    use crate::types::{ChatOptions, ChatResponse, ChatResponseUpdate};
    use crate::workflow::WorkflowEvent;
    use async_trait::async_trait;
    use futures::StreamExt;

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

    /// Designating a participant as an intermediate-output source surfaces
    /// its own reply as a non-terminal `Intermediate` event while the
    /// default aggregator output (now not designated as output) is demoted
    /// from the terminal output too — the workflow's only remaining terminal
    /// output comes from the other, output-designated participant.
    #[tokio::test]
    async fn output_from_and_intermediate_output_from_redesignate_yields() {
        let a = agent("a", "from-A");
        let b = agent("b", "from-B");

        let workflow = ConcurrentBuilder::new()
            .participants(vec![a, b])
            .output_from(["a"])
            .intermediate_output_from(["b"])
            .build()
            .unwrap();

        let run = workflow.run("question").await.unwrap();

        // A's reply is the terminal output.
        let output = run.last_output().expect("a final output");
        let conversation: Vec<Message> = serde_json::from_value(output).unwrap();
        let texts: Vec<String> = conversation.iter().map(|m| m.text()).collect();
        assert!(texts.iter().any(|t| t == "from-A"));

        // B's reply and the aggregator's merged reply both show up as
        // non-terminal Intermediate events instead.
        let intermediate_count = run
            .events()
            .iter()
            .filter(|e| matches!(e, WorkflowEvent::Intermediate { .. }))
            .count();
        assert!(
            intermediate_count >= 1,
            "expected at least one Intermediate event, got: {:?}",
            run.events()
        );

        let output_events = run
            .events()
            .iter()
            .filter(|e| matches!(e, WorkflowEvent::Output { .. }))
            .count();
        assert_eq!(output_events, 1, "only A's reply should be terminal output");
    }

    /// Default behavior (no designation) is unchanged: only the aggregator's
    /// merged reply is the terminal output.
    #[tokio::test]
    async fn default_output_is_the_aggregated_reply() {
        let a = agent("a", "from-A");
        let b = agent("b", "from-B");

        let workflow = ConcurrentBuilder::new()
            .participants(vec![a, b])
            .build()
            .unwrap();

        let run = workflow.run("question").await.unwrap();
        let output_events = run
            .events()
            .iter()
            .filter(|e| matches!(e, WorkflowEvent::Output { .. }))
            .count();
        assert_eq!(output_events, 1);
        let intermediate_events = run
            .events()
            .iter()
            .filter(|e| matches!(e, WorkflowEvent::Intermediate { .. }))
            .count();
        assert_eq!(intermediate_events, 0);
    }
}
