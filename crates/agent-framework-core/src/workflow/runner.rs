//! The workflow builder, the [`Workflow`] type, the [`WorkflowRun`] handle, and
//! the superstep runner.

use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use super::checkpoint::{CheckpointStorage, WorkflowCheckpoint};
use super::context::{WorkflowContext, WorkflowMessage};
use super::edge::{Case, Default as SwitchDefault, EdgeGroup};
use super::events::{WorkflowEvent, WorkflowRunState};
use super::executor::Executor;
use super::request_info::{PendingRequest, RequestResponse};
use super::shared_state::SharedState;
use super::validation::validate_workflow_graph;
use super::viz::WorkflowViz;
use crate::error::{Error, Result};

const DEFAULT_MAX_ITERATIONS: usize = 100;

/// FNV-1a (64-bit) hash of `bytes`. Implemented inline to avoid a hashing
/// dependency; used only to fingerprint a workflow's topology, so collision
/// resistance beyond "changed graphs almost always differ" is not required.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A stable, normalized descriptor for one edge group.
///
/// Opaque runtime closures (a [`EdgeGroup::Single`] condition, a
/// [`EdgeGroup::FanOut`] selection) cannot be inspected, so they contribute
/// only their *kind* — whether a condition/selection is present, plus any
/// declared switch-case labels — never their behavior. Target and source lists
/// are sorted so the descriptor is independent of declaration order.
fn edge_group_descriptor(group: &EdgeGroup) -> String {
    match group {
        EdgeGroup::Single {
            source,
            target,
            condition,
        } => {
            let kind = if condition.is_some() {
                "conditional"
            } else {
                "plain"
            };
            format!("single:{source}->{target}:{kind}")
        }
        EdgeGroup::FanOut {
            source,
            targets,
            selection,
            case_labels,
        } => {
            let mut targets = targets.clone();
            targets.sort_unstable();
            let kind = if selection.is_some() {
                "selection"
            } else {
                "broadcast"
            };
            let labels = case_labels
                .as_ref()
                .map(|labels| {
                    let mut labels = labels.clone();
                    labels.sort_unstable();
                    labels.join(",")
                })
                .unwrap_or_default();
            format!(
                "fanout:{source}->[{}]:{kind}:labels=[{labels}]",
                targets.join(",")
            )
        }
        EdgeGroup::FanIn { sources, target } => {
            let mut sources = sources.clone();
            sources.sort_unstable();
            format!("fanin:[{}]->{target}", sources.join(","))
        }
    }
}

/// Compute a deterministic signature of a built workflow's graph.
///
/// The signature is the FNV-1a hash (hex) of a canonical rendering of the
/// start executor, the sorted executor ids, and the sorted, normalized
/// [`edge_group_descriptor`]s. It is stable across processes and independent of
/// executor/edge insertion order, and changes whenever a node or edge is
/// added, removed, retargeted, or has its condition/selection *presence* or
/// switch labels change. Because conditions/selections are opaque closures,
/// changing only a predicate's *body* (same presence, same labels) does not
/// change the signature — documented, and acceptable for a
/// resume-compatibility guard.
pub(crate) fn compute_graph_signature(
    executors: &HashMap<String, Arc<dyn Executor>>,
    edge_groups: &[EdgeGroup],
    start: &str,
) -> String {
    let mut ids: Vec<&str> = executors.keys().map(String::as_str).collect();
    ids.sort_unstable();

    let mut edges: Vec<String> = edge_groups.iter().map(edge_group_descriptor).collect();
    edges.sort_unstable();

    let mut canonical = String::new();
    canonical.push_str("start=");
    canonical.push_str(start);
    canonical.push_str("\nnodes=");
    canonical.push_str(&ids.join(","));
    canonical.push_str("\nedges=");
    for edge in &edges {
        canonical.push('\n');
        canonical.push_str(edge);
    }

    format!("v1-{:016x}", fnv1a_64(canonical.as_bytes()))
}

/// Immutable, shared definition of a built workflow graph. Held behind an `Arc`
/// so a [`WorkflowRun`] can be driven on its own (including in a spawned task).
pub(crate) struct WorkflowShared {
    pub executors: HashMap<String, Arc<dyn Executor>>,
    pub edge_groups: Vec<EdgeGroup>,
    pub start: String,
    pub max_iterations: usize,
    pub name: Option<String>,
    pub description: Option<String>,
    pub checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
    pub id: String,
    /// Deterministic fingerprint of the graph topology, checked on resume.
    pub graph_signature: String,
}

/// Fluent builder for a [`Workflow`]. Rust equivalent of `WorkflowBuilder`.
pub struct WorkflowBuilder {
    executors: HashMap<String, Arc<dyn Executor>>,
    edge_groups: Vec<EdgeGroup>,
    start: Option<String>,
    max_iterations: usize,
    name: Option<String>,
    description: Option<String>,
    checkpoint_storage: Option<Arc<dyn CheckpointStorage>>,
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
            checkpoint_storage: None,
        }
    }
}

impl WorkflowBuilder {
    /// Create a new, empty builder.
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

    /// Add a single directed edge.
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
            case_labels: None,
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
        let mut targets: Vec<String> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for (i, case) in cases.iter().enumerate() {
            targets.push(case.target.clone());
            labels.push(case.label.clone().unwrap_or_else(|| format!("case {i}")));
        }
        targets.push(default.target.clone());
        labels.push("default".to_string());

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
            case_labels: Some(labels),
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

    /// Set the maximum number of supersteps before the run fails.
    pub fn set_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n.max(1);
        self
    }

    /// Set the workflow name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the workflow description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable automatic checkpointing to `storage` at each superstep boundary.
    pub fn with_checkpointing(mut self, storage: Arc<dyn CheckpointStorage>) -> Self {
        self.checkpoint_storage = Some(storage);
        self
    }

    /// Validate and build the workflow.
    pub fn build(self) -> Result<Workflow> {
        let start = self
            .start
            .ok_or_else(|| Error::Workflow("no start executor set".into()))?;

        // Full graph validation (start presence, duplicate edges, connectivity).
        validate_workflow_graph(&self.executors, &self.edge_groups, &start)
            .map_err(|e| Error::Workflow(e.to_string()))?;

        let graph_signature = compute_graph_signature(&self.executors, &self.edge_groups, &start);

        Ok(Workflow {
            shared: Arc::new(WorkflowShared {
                executors: self.executors,
                edge_groups: self.edge_groups,
                start,
                max_iterations: self.max_iterations,
                name: self.name,
                description: self.description,
                checkpoint_storage: self.checkpoint_storage,
                id: uuid::Uuid::new_v4().to_string(),
                graph_signature,
            }),
        })
    }
}

/// A built, runnable workflow graph.
#[derive(Clone)]
pub struct Workflow {
    pub(crate) shared: Arc<WorkflowShared>,
}

impl Workflow {
    /// The workflow name, if set.
    pub fn name(&self) -> Option<&str> {
        self.shared.name.as_deref()
    }
    /// The workflow description, if set.
    pub fn description(&self) -> Option<&str> {
        self.shared.description.as_deref()
    }
    /// The workflow's unique id.
    pub fn id(&self) -> &str {
        &self.shared.id
    }
    /// The start executor id.
    pub fn start_executor_id(&self) -> &str {
        &self.shared.start
    }

    /// This workflow's deterministic graph signature (see
    /// [`WorkflowBuilder::build`]). Checkpoints record the signature of the
    /// graph that produced them; [`Workflow::run_from_checkpoint`] refuses to
    /// resume a checkpoint whose signature does not match this workflow's.
    pub fn graph_signature(&self) -> &str {
        &self.shared.graph_signature
    }

    /// Run the workflow to completion (or until it pauses awaiting external
    /// input), returning the run handle with observed events and final state.
    pub async fn run(&self, input: impl Into<Value>) -> Result<WorkflowRun> {
        let mut run = WorkflowRun::new(self.shared.clone(), None);
        run.start(input.into()).await?;
        Ok(run)
    }

    /// Run the workflow, streaming events as they happen.
    ///
    /// Returns a [`WorkflowRunStream`] that yields [`WorkflowEvent`]s live; call
    /// [`WorkflowRunStream::into_run`] after the stream ends to obtain the final
    /// [`WorkflowRun`] state.
    pub fn run_stream(&self, input: impl Into<Value>) -> WorkflowRunStream {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let shared = self.shared.clone();
        let input = input.into();
        let handle = tokio::spawn(async move {
            let mut run = WorkflowRun::new(shared, Some(tx));
            let result = run.start(input).await;
            // Drop the streaming sender so the receiver observes end-of-stream;
            // otherwise the sender lives on inside the returned run and the
            // stream would never terminate.
            run.event_tx = None;
            result?;
            Ok(run)
        });
        WorkflowRunStream {
            rx,
            handle: Some(handle),
        }
    }

    /// Restore a run from a checkpoint and continue it to completion (or pause).
    ///
    /// Restores the in-flight message queue, iteration count, shared state,
    /// executor states, and any outstanding requests.
    ///
    /// Before restoring, the checkpoint's recorded [graph
    /// signature](Workflow::graph_signature) is compared against this
    /// workflow's: a mismatch is a hard [`Error::Workflow`] (the topology
    /// changed since the checkpoint was saved, so resuming could misroute
    /// messages or drop executor state). A legacy checkpoint carrying no
    /// signature is resumed with a warning. Use
    /// [`Workflow::run_from_checkpoint_unchecked`] to bypass the check.
    pub async fn run_from_checkpoint(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Result<WorkflowRun> {
        self.run_from_checkpoint_inner(checkpoint_id, storage, true)
            .await
    }

    /// Like [`Workflow::run_from_checkpoint`], but skips graph-signature
    /// validation (a mismatch is logged, not rejected). For deliberately
    /// resuming a checkpoint into an intentionally-evolved graph, where the
    /// caller accepts responsibility for compatibility.
    pub async fn run_from_checkpoint_unchecked(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
    ) -> Result<WorkflowRun> {
        self.run_from_checkpoint_inner(checkpoint_id, storage, false)
            .await
    }

    async fn run_from_checkpoint_inner(
        &self,
        checkpoint_id: &str,
        storage: Arc<dyn CheckpointStorage>,
        validate: bool,
    ) -> Result<WorkflowRun> {
        let cp = storage
            .load(checkpoint_id)
            .await?
            .ok_or_else(|| Error::Workflow(format!("checkpoint '{checkpoint_id}' not found")))?;
        self.check_graph_signature(&cp, validate)?;
        let mut run = WorkflowRun::new(self.shared.clone(), None);
        run.restore(cp).await?;
        Ok(run)
    }

    /// Compare a checkpoint's graph signature against this workflow's.
    ///
    /// * No signature (legacy checkpoint) → warn and continue.
    /// * Matching signature → continue.
    /// * Mismatch with `validate` → [`Error::Workflow`] naming both signatures.
    /// * Mismatch without `validate` → warn and continue.
    fn check_graph_signature(&self, cp: &WorkflowCheckpoint, validate: bool) -> Result<()> {
        let expected = &self.shared.graph_signature;
        if cp.graph_signature.is_empty() {
            tracing::warn!(
                workflow_id = %self.shared.id,
                expected = %expected,
                "resuming a checkpoint with no graph signature (written before signature \
                 validation existed); skipping the graph-compatibility check"
            );
            return Ok(());
        }
        if &cp.graph_signature == expected {
            return Ok(());
        }
        if validate {
            return Err(Error::Workflow(format!(
                "checkpoint graph signature mismatch: checkpoint '{}' was saved for graph \
                 signature '{}', but this workflow's graph signature is '{}'. The workflow's \
                 executors or edges changed since the checkpoint was written, so resuming could \
                 misroute messages or drop executor state. Rebuild the identical graph, or call \
                 `run_from_checkpoint_unchecked` to override.",
                cp.checkpoint_id, cp.graph_signature, expected
            )));
        }
        tracing::warn!(
            workflow_id = %self.shared.id,
            checkpoint_signature = %cp.graph_signature,
            expected = %expected,
            "resuming a checkpoint whose graph signature does not match this workflow \
             (unchecked); resume may misbehave if the topology is incompatible"
        );
        Ok(())
    }

    /// A visualization helper (Mermaid / Graphviz DOT) for this workflow.
    pub fn viz(&self) -> WorkflowViz<'_> {
        WorkflowViz::new(self)
    }
}

/// A live workflow run: owns the pending message queue, fan-in buffers,
/// iteration count, shared state, and outstanding requests, so a run can pause
/// (awaiting external input) and later resume.
pub struct WorkflowRun {
    shared: Arc<WorkflowShared>,
    shared_state: SharedState,
    queue: Vec<WorkflowMessage>,
    fanin: HashMap<String, HashMap<String, Value>>,
    iteration: usize,
    pending_requests: BTreeMap<String, PendingRequest>,
    events: Vec<WorkflowEvent>,
    state: WorkflowRunState,
    event_tx: Option<UnboundedSender<WorkflowEvent>>,
}

impl WorkflowRun {
    fn new(shared: Arc<WorkflowShared>, event_tx: Option<UnboundedSender<WorkflowEvent>>) -> Self {
        Self {
            shared,
            shared_state: SharedState::new(),
            queue: Vec::new(),
            fanin: HashMap::new(),
            iteration: 0,
            pending_requests: BTreeMap::new(),
            events: Vec::new(),
            state: WorkflowRunState::Started,
            event_tx,
        }
    }

    fn emit(&mut self, event: WorkflowEvent) {
        if let Some(tx) = &self.event_tx {
            // Ignore send errors: the receiver may have been dropped.
            let _ = tx.send(event.clone());
        }
        self.events.push(event);
    }

    async fn start(&mut self, input: Value) -> Result<()> {
        self.emit(WorkflowEvent::Started);
        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));
        self.queue.push(WorkflowMessage {
            data: input,
            source_id: "__start__".to_string(),
            target_id: Some(self.shared.start.clone()),
        });
        self.drive().await
    }

    async fn restore(&mut self, cp: WorkflowCheckpoint) -> Result<()> {
        self.shared_state.import(cp.shared_state).await;
        self.iteration = cp.iteration_count;
        self.queue = cp.messages;
        self.pending_requests = cp
            .pending_requests
            .into_iter()
            .map(|p| (p.request_id.clone(), p))
            .collect();
        for (id, state) in cp.executor_states {
            if let Some(ex) = self.shared.executors.get(&id) {
                ex.restore_state(state).await?;
            }
        }

        self.emit(WorkflowEvent::Started);
        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));
        // Re-surface any outstanding requests so consumers observe them.
        let pending_events: Vec<WorkflowEvent> = self
            .pending_requests
            .values()
            .map(|pr| WorkflowEvent::RequestInfo {
                request_id: pr.request_id.clone(),
                source_executor_id: pr.source_executor_id.clone(),
                request_data: pr.request_data.clone(),
            })
            .collect();
        for ev in pending_events {
            self.emit(ev);
        }
        self.drive().await
    }

    /// Drive the superstep loop, emitting the terminal status on success or a
    /// `Failed` status on error.
    async fn drive(&mut self) -> Result<()> {
        match self.run_loop().await {
            Ok(()) => Ok(()),
            Err(e) => {
                self.state = WorkflowRunState::Failed;
                self.emit(WorkflowEvent::Failed {
                    error: e.to_string(),
                });
                self.emit(WorkflowEvent::Status(WorkflowRunState::Failed));
                Err(e)
            }
        }
    }

    async fn run_loop(&mut self) -> Result<()> {
        let mut emitted_pending = false;
        while !self.queue.is_empty() {
            if self.iteration >= self.shared.max_iterations {
                return Err(Error::Workflow(format!(
                    "max_iterations ({}) exceeded",
                    self.shared.max_iterations
                )));
            }
            let step = self.iteration + 1;
            self.emit(WorkflowEvent::SuperStepStarted(step));

            // Group this superstep's deliveries by target executor.
            let deliveries = std::mem::take(&mut self.queue);
            let mut by_target: HashMap<String, Vec<WorkflowMessage>> = HashMap::new();
            for msg in deliveries {
                for target in self.resolve_targets(&msg) {
                    by_target.entry(target).or_default().push(msg.clone());
                }
            }

            // Deterministic iteration order within a superstep.
            let mut targets: Vec<String> = by_target.keys().cloned().collect();
            targets.sort();

            let mut next: Vec<WorkflowMessage> = Vec::new();
            for target_id in targets {
                let msgs = by_target.remove(&target_id).unwrap_or_default();
                let executor = match self.shared.executors.get(&target_id) {
                    Some(e) => e.clone(),
                    None => continue,
                };

                if let Some(sources) = self.fanin_group_for(&target_id) {
                    // Barrier: accumulate by source, fire once every source has
                    // delivered, collecting in declared order.
                    let buf = self.fanin.entry(target_id.clone()).or_default();
                    for m in &msgs {
                        buf.insert(m.source_id.clone(), m.data.clone());
                    }
                    if buf.len() < sources.len() {
                        continue;
                    }
                    let mut collected = Vec::with_capacity(sources.len());
                    for source in &sources {
                        if let Some(v) = buf.remove(source) {
                            collected.push(v);
                        }
                    }
                    self.fanin.remove(&target_id);
                    self.run_executor(
                        &executor,
                        Value::Array(collected),
                        sources,
                        &mut next,
                        &mut emitted_pending,
                    )
                    .await?;
                } else {
                    for m in msgs {
                        let source_ids = vec![m.source_id.clone()];
                        self.run_executor(
                            &executor,
                            m.data,
                            source_ids,
                            &mut next,
                            &mut emitted_pending,
                        )
                        .await?;
                    }
                }
            }

            self.emit(WorkflowEvent::SuperStepCompleted(step));
            self.queue = next;
            self.iteration += 1;
            self.maybe_checkpoint(step).await;
        }

        // Terminal status: distinguish idle from paused-with-requests.
        if self.pending_requests.is_empty() {
            self.state = WorkflowRunState::Idle;
            self.emit(WorkflowEvent::Status(WorkflowRunState::Idle));
        } else {
            self.state = WorkflowRunState::IdleWithPendingRequests;
            self.emit(WorkflowEvent::Status(
                WorkflowRunState::IdleWithPendingRequests,
            ));
        }
        Ok(())
    }

    async fn run_executor(
        &mut self,
        executor: &Arc<dyn Executor>,
        data: Value,
        source_ids: Vec<String>,
        next: &mut Vec<WorkflowMessage>,
        emitted_pending: &mut bool,
    ) -> Result<()> {
        let id = executor.id().to_string();
        self.emit(WorkflowEvent::ExecutorInvoked {
            executor_id: id.clone(),
        });
        let ctx = WorkflowContext::new(id.clone(), source_ids, self.shared_state.clone());
        match executor.execute(data, ctx.clone()).await {
            Ok(()) => {
                let (sent, outputs, custom, requests) = ctx.take();
                for out in outputs {
                    self.emit(WorkflowEvent::Output {
                        data: out,
                        source_executor_id: id.clone(),
                    });
                }
                for ev in custom {
                    self.emit(ev);
                }
                for draft in requests {
                    let pending = PendingRequest {
                        request_id: draft.request_id.clone(),
                        source_executor_id: id.clone(),
                        reply_to_executor_id: draft.reply_to,
                        request_data: draft.data.clone(),
                    };
                    self.pending_requests
                        .insert(draft.request_id.clone(), pending);
                    self.emit(WorkflowEvent::RequestInfo {
                        request_id: draft.request_id,
                        source_executor_id: id.clone(),
                        request_data: draft.data,
                    });
                    if !*emitted_pending {
                        *emitted_pending = true;
                        self.emit(WorkflowEvent::Status(
                            WorkflowRunState::InProgressPendingRequests,
                        ));
                    }
                }
                next.extend(sent);
                self.emit(WorkflowEvent::ExecutorCompleted { executor_id: id });
                Ok(())
            }
            Err(e) => {
                self.emit(WorkflowEvent::ExecutorFailed {
                    executor_id: id,
                    error: e.to_string(),
                });
                Err(e)
            }
        }
    }

    async fn maybe_checkpoint(&self, step: usize) {
        let Some(storage) = self.shared.checkpoint_storage.clone() else {
            return;
        };
        let mut executor_states: HashMap<String, Value> = HashMap::new();
        for (id, ex) in &self.shared.executors {
            if let Some(state) = ex.snapshot_state().await {
                executor_states.insert(id.clone(), state);
            }
        }
        let shared_state = self.shared_state.export().await;
        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert("superstep".to_string(), Value::from(self.iteration as u64));
        metadata.insert(
            "checkpoint_type".to_string(),
            Value::from(format!("superstep_{step}")),
        );
        let checkpoint = WorkflowCheckpoint::new(
            self.shared.id.clone(),
            self.shared.name.clone(),
            self.iteration,
            self.queue.clone(),
            executor_states,
            shared_state,
            self.pending_requests.values().cloned().collect(),
            metadata,
            self.shared.graph_signature.clone(),
        );
        if let Err(e) = storage.save(checkpoint).await {
            tracing::warn!("failed to save checkpoint: {e}");
        }
    }

    /// Deliver a single response, resuming execution.
    pub async fn send_response(
        &mut self,
        request_id: impl Into<String>,
        value: Value,
    ) -> Result<()> {
        let mut map = HashMap::new();
        map.insert(request_id.into(), value);
        self.send_responses(map).await
    }

    /// Deliver responses to outstanding requests and resume execution.
    ///
    /// Each response is routed back to the executor that made the corresponding
    /// request as a [`RequestResponse`] message.
    pub async fn send_responses(&mut self, responses: HashMap<String, Value>) -> Result<()> {
        if self.pending_requests.is_empty() {
            return Err(Error::Workflow("no pending requests to respond to".into()));
        }
        for id in responses.keys() {
            if !self.pending_requests.contains_key(id) {
                return Err(Error::Workflow(format!(
                    "response provided for unknown request id '{id}'"
                )));
            }
        }

        self.state = WorkflowRunState::InProgress;
        self.emit(WorkflowEvent::Status(WorkflowRunState::InProgress));

        // Deliver in deterministic order.
        let mut ids: Vec<String> = responses.keys().cloned().collect();
        ids.sort();
        for id in ids {
            if let Some(value) = responses.get(&id) {
                self.deliver_response(&id, value.clone());
            }
        }
        self.drive().await
    }

    fn deliver_response(&mut self, request_id: &str, value: Value) {
        if let Some(pending) = self.pending_requests.remove(request_id) {
            let response = RequestResponse {
                request_id: pending.request_id.clone(),
                data: value,
                original_request: pending.request_data.clone(),
            };
            let data = serde_json::to_value(response).unwrap_or(Value::Null);
            self.queue.push(WorkflowMessage {
                data,
                source_id: pending.source_executor_id.clone(),
                target_id: Some(pending.reply_to_executor_id.clone()),
            });
        }
    }

    /// Resolve which executors a message should be delivered to.
    fn resolve_targets(&self, msg: &WorkflowMessage) -> Vec<String> {
        // Explicit target wins (direct sends and routed responses).
        if let Some(t) = &msg.target_id {
            return vec![t.clone()];
        }
        let mut targets = Vec::new();
        for group in &self.shared.edge_groups {
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
                    ..
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
        for group in &self.shared.edge_groups {
            if let EdgeGroup::FanIn { sources, target: t } = group {
                if t == target {
                    return Some(sources.clone());
                }
            }
        }
        None
    }

    /// All events observed during the run, in order.
    pub fn events(&self) -> &[WorkflowEvent] {
        &self.events
    }

    /// The current run state.
    pub fn state(&self) -> WorkflowRunState {
        self.state
    }

    /// Outstanding requests awaiting responses, ordered by request id.
    pub fn pending_requests(&self) -> Vec<PendingRequest> {
        self.pending_requests.values().cloned().collect()
    }

    /// All workflow-level outputs, in order.
    pub fn outputs(&self) -> Vec<Value> {
        self.events
            .iter()
            .filter_map(|e| e.as_output().cloned())
            .collect()
    }

    /// The last workflow output, if any.
    pub fn last_output(&self) -> Option<Value> {
        self.events
            .iter()
            .rev()
            .find_map(|e| e.as_output().cloned())
    }

    /// A handle to the run-scoped shared state (for inspection in tests/tools).
    pub fn shared_state(&self) -> SharedState {
        self.shared_state.clone()
    }
}

/// A live stream of [`WorkflowEvent`]s from a running workflow, plus a way to
/// obtain the final [`WorkflowRun`] once the stream is exhausted.
pub struct WorkflowRunStream {
    rx: UnboundedReceiver<WorkflowEvent>,
    handle: Option<tokio::task::JoinHandle<Result<WorkflowRun>>>,
}

impl WorkflowRunStream {
    /// Await the driving task and return the final run state.
    ///
    /// Drains any remaining buffered events first so the run completes.
    pub async fn into_run(mut self) -> Result<WorkflowRun> {
        while self.rx.recv().await.is_some() {}
        match self.handle.take() {
            Some(handle) => handle
                .await
                .map_err(|e| Error::Workflow(format!("workflow task failed: {e}")))?,
            None => Err(Error::Workflow("run already taken from stream".into())),
        }
    }
}

impl futures::Stream for WorkflowRunStream {
    type Item = WorkflowEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}
