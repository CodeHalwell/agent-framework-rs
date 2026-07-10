//! Edges and edge groups connecting executors.

use serde_json::Value;
use std::sync::Arc;

/// A predicate deciding whether a message should traverse an edge.
pub type Condition = Arc<dyn Fn(&Value) -> bool + Send + Sync>;

/// A runtime target selector for multi-selection / switch edges: given the
/// message and the candidate target ids, return the ids to route to.
pub type Selection = Arc<dyn Fn(&Value, &[String]) -> Vec<String> + Send + Sync>;

/// A group of edges sharing routing semantics. Rust equivalent of the
/// `EdgeGroup` hierarchy.
#[derive(Clone)]
pub enum EdgeGroup {
    /// A single conditional edge from `source` to `target`.
    Single {
        source: String,
        target: String,
        condition: Option<Condition>,
    },
    /// Broadcast from `source` to all `targets` (optionally filtered by a
    /// selection function — used for switch/case and multi-selection).
    FanOut {
        source: String,
        targets: Vec<String>,
        selection: Option<Selection>,
    },
    /// Barrier: `target` runs once all `sources` have delivered, receiving the
    /// collected messages as a JSON array.
    FanIn {
        sources: Vec<String>,
        target: String,
    },
}

impl EdgeGroup {
    /// The source executor ids for this group.
    pub fn sources(&self) -> Vec<String> {
        match self {
            EdgeGroup::Single { source, .. } => vec![source.clone()],
            EdgeGroup::FanOut { source, .. } => vec![source.clone()],
            EdgeGroup::FanIn { sources, .. } => sources.clone(),
        }
    }

    /// The target executor ids for this group.
    pub fn targets(&self) -> Vec<String> {
        match self {
            EdgeGroup::Single { target, .. } => vec![target.clone()],
            EdgeGroup::FanOut { targets, .. } => targets.clone(),
            EdgeGroup::FanIn { target, .. } => vec![target.clone()],
        }
    }
}

/// A switch/case branch: if `condition` matches, route to `target`.
pub struct Case {
    pub condition: Condition,
    pub target: String,
}

impl Case {
    pub fn new(
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
        target: impl Into<String>,
    ) -> Self {
        Self {
            condition: Arc::new(condition),
            target: target.into(),
        }
    }
}

/// The default branch of a switch/case group.
pub struct Default {
    pub target: String,
}

impl Default {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}
