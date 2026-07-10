//! The workflow builder, the [`Workflow`] type, and the superstep runner.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

use super::context::{WorkflowContext, WorkflowMessage};
use super::edge::{Case, Default as SwitchDefault, EdgeGroup};
use super::events::{WorkflowEvent, WorkflowRunState};
use super::executor::Executor;
use crate::error::{Error, Result};

const DEFAULT_MAX_ITERATIONS: usize = 100;

/// Fluent builder for a [`Workflow`]. Rust equivalent of `WorkflowBuilder`.
pub struct WorkflowBuilder {
    executors: HashMap<String, Arc<dyn Executor>>,
    edge_groups: Vec<EdgeGroup>,
    start: Option<String>,
    max_iterations: usize,
    name: Option<String>,
    description: Option<String>,
}

impl std::default::Default for WorkflowBuilder {
    fn default() -> Self {
        Self {
            executors: HashMap::new(),
            edge_groups: Vec::new(),
            start: None,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            name: None,
            description: None,
        }
    }
}

impl WorkflowBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an executor. Executors are keyed by their [`Executor::id`].
    pub fn add_executor(mut self, executor: Arc<dyn Executor>) -> Self {
        self.executors.insert(executor.id().to_string(), executor);
        self
    }

    /// Set the entry-point executor (by id). The initial message is delivered
    /// here.
    pub fn set_start(mut self, id: impl Into<String>) -> Self {
        self.start = Some(id.into());
        self
    }

    /// Add a single directed edge, with an optional condition on the message.
    pub fn add_edge(mut self, source: impl Into<String>, target: impl Into<String>) -> Self {
        self.edge_groups.push(EdgeGroup::Single {
            source: source.into(),
            target: target.into(),
            condition: None,
        });
        self
    }

    /// Add a single directed edge guarded by a condition.
    pub fn add_conditional_edge(
        mut self,
        source: impl Into<String>,
        target: impl Into<String>,
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::Single {
            source: source.into(),
            target: target.into(),
            condition: Some(Arc::new(condition)),
        });
        self
    }

    /// Broadcast from `source` to all `targets`.
    pub fn add_fan_out(
        mut self,
        source: impl Into<String>,
        targets: impl IntoIterator<Item = String>,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::FanOut {
            source: source.into(),
            targets: targets.into_iter().collect(),
            selection: None,
        });
        self
    }

    /// Fan in from `sources` to `target` (barrier).
    pub fn add_fan_in(
        mut self,
        sources: impl IntoIterator<Item = String>,
        target: impl Into<String>,
    ) -> Self {
        self.edge_groups.push(EdgeGroup::FanIn {
            sources: sources.into_iter().collect(),
            target: target.into(),
        });
        self
    }

    /// Add a switch/case group: evaluate `cases` in order, falling back to
    /// `default`.
    pub fn add_switch(
        mut self,
        source: impl Into<String>,
        cases: Vec<Case>,
        default: SwitchDefault,
    ) -> Self {
        let source = source.into();
        let mut targets: Vec<String> = cases.iter().map(|c| c.target.clone()).collect();
        targets.push(default.target.clone());
        let conds: Vec<Case> = cases;
        let default_target = default.target.clone();
        let selection = Arc::new(move |msg: &Value, _candidates: &[String]| {
            for case in &conds {
                if (case.condition)(msg) {
                    return vec![case.target.clone()];
                }
            }
            vec![default_target.clone()]
        });
        self.edge_groups.push(EdgeGroup::FanOut {
            source,
            targets,
            selection: Some(selection),
        });
        self
    }

    /// Chain executors sequentially (sugar for consecutive edges).
    pub fn add_chain(mut self, ids: impl IntoIterator<Item = String>) -> Self {
        let ids: Vec<String> = ids.into_iter().collect();
        for pair in ids.windows(2) {
            self = self.add_edge(pair[0].clone(), pair[1].clone());
        }
        self
    }

    pub fn set_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Validate and build the workflow.
    pub fn build(self) -> Result<Workflow> {
        let start = self
            .start
            .ok_or_else(|| Error::Workflow("no start executor set".into()))?;
        if !self.executors.contains_key(&start) {
            return Err(Error::Workflow(format!(
                "start executor '{start}' is not registered"
            )));
        }
        for group in &self.edge_groups {
            for id in group.sources().into_iter().chain(group.targets()) {
                if !self.executors.contains_key(&id) {
                    return Err(Error::Workflow(format!(
                        "edge references unknown executor '{id}'"
                    )));
                }
            }
        }
        Ok(Workflow {
            executors: self.executors,
            edge_groups: self.edge_groups,
            start,
            max_iterations: self.max_iterations,
            name: self.name,
            description: self.description,
        })
    }
}

/// A built, runnable workflow graph.
pub struct Workflow {
    executors: HashMap<String, Arc<dyn Executor>>,
    edge_groups: Vec<EdgeGroup>,
    start: String,
    max_iterations: usize,
    name: Option<String>,
    description: Option<String>,
}

/// The result of running a workflow to completion.
#[derive(Debug, Clone)]
pub struct WorkflowRunResult {
    pub events: Vec<WorkflowEvent>,
}

impl WorkflowRunResult {
    /// All workflow-level outputs, in order.
    pub fn outputs(&self) -> Vec<Value> {
        self.events
            .iter()
            .filter_map(|e| e.as_output().cloned())
            .collect()
    }

    /// The last output, if any.
    pub fn last_output(&self) -> Option<Value> {
        self.events
            .iter()
            .rev()
            .find_map(|e| e.as_output().cloned())
    }
}

impl Workflow {
    /// The workflow name, if set.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    /// The workflow description, if set.
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }

    /// Run the workflow to completion, returning all observed events.
    pub async fn run(&self, input: impl Into<Value>) -> Result<WorkflowRunResult> {
        let mut events = vec![
            WorkflowEvent::Started,
            WorkflowEvent::Status(WorkflowRunState::InProgress),
        ];

        // Buffer of in-transit messages, keyed by target executor id.
        let mut pending: Vec<WorkflowMessage> = vec![WorkflowMessage {
            data: input.into(),
            source_id: "__start__".to_string(),
            target_id: Some(self.start.clone()),
        }];
        // Fan-in accumulation across supersteps, keyed by target then source so
        // the barrier only fires once every distinct source has delivered and
        // the collected messages are ordered deterministically.
        let mut fanin_buffer: HashMap<String, HashMap<String, Value>> = HashMap::new();

        let mut iteration = 0usize;
        while !pending.is_empty() {
            if iteration >= self.max_iterations {
                let error = format!("max_iterations ({}) exceeded", self.max_iterations);
                events.push(WorkflowEvent::Failed {
                    error: error.clone(),
                });
                events.push(WorkflowEvent::Status(WorkflowRunState::Failed));
                return Err(Error::Workflow(error));
            }
            events.push(WorkflowEvent::SuperStepStarted(iteration));

            // Group this superstep's deliveries by target executor.
            let deliveries = std::mem::take(&mut pending);
            let mut by_target: HashMap<String, Vec<WorkflowMessage>> = HashMap::new();
            for msg in deliveries {
                let targets = self.resolve_targets(&msg);
                for target in targets {
                    by_target.entry(target).or_default().push(msg.clone());
                }
            }

            let mut next: Vec<WorkflowMessage> = Vec::new();

            for (target_id, msgs) in by_target {
                let executor = match self.executors.get(&target_id) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                // Determine whether this target is a fan-in sink.
                let fanin = self.fanin_group_for(&target_id);
                if let Some(sources) = fanin {
                    // Accumulate by source; only fire when every distinct source
                    // has delivered a message.
                    let buf = fanin_buffer.entry(target_id.clone()).or_default();
                    for m in &msgs {
                        buf.insert(m.source_id.clone(), m.data.clone());
                    }
                    if buf.len() < sources.len() {
                        continue;
                    }
                    // Collect in the order the sources were declared for
                    // deterministic downstream behavior.
                    let mut collected_vec = Vec::with_capacity(sources.len());
                    for source in &sources {
                        if let Some(val) = buf.remove(source) {
                            collected_vec.push(val);
                        }
                    }
                    fanin_buffer.remove(&target_id);
                    let collected = Value::Array(collected_vec);
                    self.run_executor(&executor, collected, sources, &mut events, &mut next)
                        .await?;
                } else {
                    for m in msgs {
                        let source_ids = vec![m.source_id.clone()];
                        self.run_executor(&executor, m.data, source_ids, &mut events, &mut next)
                            .await?;
                    }
                }
            }

            events.push(WorkflowEvent::SuperStepCompleted(iteration));
            pending = next;
            iteration += 1;
        }

        events.push(WorkflowEvent::Status(WorkflowRunState::Idle));
        Ok(WorkflowRunResult { events })
    }

    async fn run_executor(
        &self,
        executor: &Arc<dyn Executor>,
        data: Value,
        source_ids: Vec<String>,
        events: &mut Vec<WorkflowEvent>,
        next: &mut Vec<WorkflowMessage>,
    ) -> Result<()> {
        let id = executor.id().to_string();
        events.push(WorkflowEvent::ExecutorInvoked {
            executor_id: id.clone(),
        });
        let ctx = WorkflowContext::new(id.clone(), source_ids);
        match executor.execute(data, ctx.clone()).await {
            Ok(()) => {
                let (sent, outputs, custom, requests) = ctx.take();
                for out in outputs {
                    events.push(WorkflowEvent::Output {
                        data: out,
                        source_executor_id: id.clone(),
                    });
                }
                for ev in custom {
                    events.push(ev);
                }
                for (request_id, request_data) in requests {
                    events.push(WorkflowEvent::RequestInfo {
                        request_id,
                        source_executor_id: id.clone(),
                        request_data,
                    });
                }
                next.extend(sent);
                events.push(WorkflowEvent::ExecutorCompleted { executor_id: id });
                Ok(())
            }
            Err(e) => {
                events.push(WorkflowEvent::ExecutorFailed {
                    executor_id: id,
                    error: e.to_string(),
                });
                Err(e)
            }
        }
    }

    /// Resolve which executors a message should be delivered to.
    fn resolve_targets(&self, msg: &WorkflowMessage) -> Vec<String> {
        // Explicit target wins.
        if let Some(t) = &msg.target_id {
            return vec![t.clone()];
        }
        let mut targets = Vec::new();
        for group in &self.edge_groups {
            match group {
                EdgeGroup::Single {
                    source,
                    target,
                    condition,
                } if *source == msg.source_id => {
                    if condition.as_ref().map(|c| c(&msg.data)).unwrap_or(true) {
                        targets.push(target.clone());
                    }
                }
                EdgeGroup::FanOut {
                    source,
                    targets: outs,
                    selection,
                } if *source == msg.source_id => match selection {
                    Some(sel) => targets.extend(sel(&msg.data, outs)),
                    None => targets.extend(outs.clone()),
                },
                EdgeGroup::FanIn { sources, target } if sources.contains(&msg.source_id) => {
                    targets.push(target.clone());
                }
                _ => {}
            }
        }
        targets
    }

    /// If `target` is the sink of a fan-in group, return its source ids.
    fn fanin_group_for(&self, target: &str) -> Option<Vec<String>> {
        for group in &self.edge_groups {
            if let EdgeGroup::FanIn { sources, target: t } = group {
                if t == target {
                    return Some(sources.clone());
                }
            }
        }
        None
    }
}
