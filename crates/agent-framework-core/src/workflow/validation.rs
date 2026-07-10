//! Workflow graph validation, run at [`WorkflowBuilder::build`](super::WorkflowBuilder::build).
//!
//! Rust equivalent of Python's `_validation.py`, restricted to the structural
//! checks that make sense for a `serde_json::Value`-typed engine (Python's
//! static type-compatibility checks are intentionally omitted).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use super::edge::EdgeGroup;
use super::executor::Executor;

/// The category of a workflow validation failure. Mirrors Python's
/// `ValidationTypeEnum` (subset relevant to this engine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationType {
    /// A start executor was not registered.
    StartNotRegistered,
    /// The same directed edge was declared more than once.
    EdgeDuplication,
    /// An edge referenced an executor that is not registered.
    UnknownExecutor,
    /// One or more executors are unreachable from the start node.
    GraphConnectivity,
}

impl fmt::Display for ValidationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ValidationType::StartNotRegistered => "START_NOT_REGISTERED",
            ValidationType::EdgeDuplication => "EDGE_DUPLICATION",
            ValidationType::UnknownExecutor => "UNKNOWN_EXECUTOR",
            ValidationType::GraphConnectivity => "GRAPH_CONNECTIVITY",
        };
        f.write_str(s)
    }
}

/// A typed workflow validation error carrying its [`ValidationType`].
///
/// [`WorkflowBuilder::build`](super::WorkflowBuilder::build) surfaces this as an
/// `Error::Workflow` (using this error's `Display` as the message). Callers who
/// want the structured category can call [`validate_workflow_graph`] directly,
/// which returns this type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowValidationError {
    pub validation_type: ValidationType,
    pub message: String,
}

impl WorkflowValidationError {
    fn new(validation_type: ValidationType, message: impl Into<String>) -> Self {
        Self {
            validation_type,
            message: message.into(),
        }
    }
}

impl fmt::Display for WorkflowValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.validation_type, self.message)
    }
}

impl std::error::Error for WorkflowValidationError {}

/// Validate a workflow graph. Rust analogue of `validate_workflow_graph`.
///
/// Checks, in order:
/// 1. the start executor is registered;
/// 2. every edge references a registered executor;
/// 3. no directed `(source, target)` edge is duplicated;
/// 4. every executor is reachable from the start node.
pub fn validate_workflow_graph(
    executors: &HashMap<String, Arc<dyn Executor>>,
    edge_groups: &[EdgeGroup],
    start: &str,
) -> Result<(), WorkflowValidationError> {
    // 1. Start must be registered.
    if !executors.contains_key(start) {
        return Err(WorkflowValidationError::new(
            ValidationType::StartNotRegistered,
            format!("start executor '{start}' is not registered"),
        ));
    }

    // 2. Every edge endpoint must be a known executor.
    for group in edge_groups {
        for id in group.sources().into_iter().chain(group.targets()) {
            if !executors.contains_key(&id) {
                return Err(WorkflowValidationError::new(
                    ValidationType::UnknownExecutor,
                    format!("edge references unknown executor '{id}'"),
                ));
            }
        }
    }

    // 3. Duplicate-edge detection over flattened directed edges (deduping
    //    identical targets within a single fan-out/switch group, which are not
    //    logically distinct edges).
    let mut seen: HashSet<(String, String)> = HashSet::new();
    for group in edge_groups {
        let mut group_seen: HashSet<(String, String)> = HashSet::new();
        for edge in group.flat_edges() {
            if !group_seen.insert(edge.clone()) {
                // Repeated within the same group: ignore.
                continue;
            }
            if !seen.insert(edge.clone()) {
                return Err(WorkflowValidationError::new(
                    ValidationType::EdgeDuplication,
                    format!(
                        "duplicate edge detected: '{}' -> '{}'. Each edge must be unique.",
                        edge.0, edge.1
                    ),
                ));
            }
        }
    }

    // 4. Connectivity: every executor must be reachable from the start node.
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for group in edge_groups {
        for (src, tgt) in group.flat_edges() {
            adjacency.entry(src).or_default().push(tgt);
        }
    }
    let reachable = reachable_from(&adjacency, start);
    let mut unreachable: Vec<String> = executors
        .keys()
        .filter(|id| !reachable.contains(*id))
        .cloned()
        .collect();
    if !unreachable.is_empty() {
        unreachable.sort();
        return Err(WorkflowValidationError::new(
            ValidationType::GraphConnectivity,
            format!(
                "the following executors are unreachable from the start executor '{start}': {unreachable:?}. \
                 This may indicate a disconnected workflow graph."
            ),
        ));
    }

    Ok(())
}

fn reachable_from(adjacency: &HashMap<String, Vec<String>>, start: &str) -> HashSet<String> {
    let mut visited: HashSet<String> = HashSet::new();
    let mut stack = vec![start.to_string()];
    while let Some(node) = stack.pop() {
        if visited.insert(node.clone()) {
            if let Some(neighbors) = adjacency.get(&node) {
                for n in neighbors {
                    if !visited.contains(n) {
                        stack.push(n.clone());
                    }
                }
            }
        }
    }
    visited
}
