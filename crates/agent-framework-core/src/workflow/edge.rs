//! Edges and edge groups connecting executors.

use serde_json::Value;
use std::future::Future;
use std::sync::Arc;

use crate::tools::BoxFuture;

/// A predicate deciding whether a message should traverse an edge.
///
/// Evaluated **asynchronously**, mirroring upstream's `Edge.should_route`
/// becoming `async` (see `UPSTREAM_DRIFT.md` §10, `EdgeCondition = Callable[[Any],
/// bool | Awaitable[bool]]`). Callers normally build one via
/// [`wrap_sync_condition`] (from an existing sync predicate — used by the sync
/// builder methods to stay backward compatible) or [`wrap_async_condition`]
/// (from a genuinely async predicate — used by the `*_async` builder methods)
/// rather than constructing the `Arc` directly.
pub type Condition = Arc<dyn Fn(&Value) -> BoxFuture<bool> + Send + Sync>;

/// A runtime target selector for multi-selection / switch edges: given the
/// message and the candidate target ids, return the ids to route to.
///
/// Evaluated asynchronously for the same reason as [`Condition`] — a switch's
/// selection function awaits each [`Case::condition`](Case) in turn.
pub type Selection = Arc<dyn Fn(&Value, &[String]) -> BoxFuture<Vec<String>> + Send + Sync>;

/// Wrap a synchronous `Fn(&Value) -> bool` into an async [`Condition`].
///
/// The sync closure is invoked eagerly, right when the condition is called —
/// not deferred inside the returned future — so wrapping does not change
/// evaluation order or timing for existing sync call sites; only the return
/// type becomes an (already-resolved) future.
pub(crate) fn wrap_sync_condition(f: impl Fn(&Value) -> bool + Send + Sync + 'static) -> Condition {
    Arc::new(move |v: &Value| {
        let result = f(v);
        Box::pin(async move { result }) as BoxFuture<bool>
    })
}

/// Wrap an async `Fn(&Value) -> impl Future<Output = bool>` into a
/// [`Condition`].
pub(crate) fn wrap_async_condition<F, Fut>(f: F) -> Condition
where
    F: Fn(&Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = bool> + Send + 'static,
{
    Arc::new(move |v: &Value| Box::pin(f(v)) as BoxFuture<bool>)
}

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
    ///
    /// `case_labels`, when present, is index-aligned with `targets` and carries
    /// human-readable labels for visualization (e.g. switch-case names). It has
    /// no effect on routing.
    FanOut {
        source: String,
        targets: Vec<String>,
        selection: Option<Selection>,
        case_labels: Option<Vec<String>>,
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

    /// Whether this edge group carries a runtime routing predicate — a
    /// [`Single`](EdgeGroup::Single) edge's `condition` or a
    /// [`FanOut`](EdgeGroup::FanOut) group's `selection` — as opposed to being
    /// an unconditional edge, a plain broadcast, or a fan-in barrier (which
    /// never carries one). Mirrors upstream's `Edge.has_condition`.
    pub fn has_condition(&self) -> bool {
        match self {
            EdgeGroup::Single { condition, .. } => condition.is_some(),
            EdgeGroup::FanOut { selection, .. } => selection.is_some(),
            EdgeGroup::FanIn { .. } => false,
        }
    }

    /// Flattened directed `(source, target)` edges implied by this group. Used
    /// by validation and visualization.
    pub(crate) fn flat_edges(&self) -> Vec<(String, String)> {
        match self {
            EdgeGroup::Single { source, target, .. } => vec![(source.clone(), target.clone())],
            EdgeGroup::FanOut {
                source, targets, ..
            } => targets
                .iter()
                .map(|t| (source.clone(), t.clone()))
                .collect(),
            EdgeGroup::FanIn { sources, target } => sources
                .iter()
                .map(|s| (s.clone(), target.clone()))
                .collect(),
        }
    }
}

/// A switch/case branch: if `condition` matches, route to `target`.
pub struct Case {
    pub condition: Condition,
    pub target: String,
    /// Optional human-readable label used in visualization.
    pub label: Option<String>,
}

impl Case {
    /// A case routing to `target` when the synchronous `condition` holds.
    pub fn new(
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
        target: impl Into<String>,
    ) -> Self {
        Self {
            condition: wrap_sync_condition(condition),
            target: target.into(),
            label: None,
        }
    }

    /// A case routing to `target` when the async `condition` holds.
    pub fn new_async<F, Fut>(condition: F, target: impl Into<String>) -> Self
    where
        F: Fn(&Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self {
            condition: wrap_async_condition(condition),
            target: target.into(),
            label: None,
        }
    }

    /// A case with an explicit visualization label.
    pub fn labeled(
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
        target: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            condition: wrap_sync_condition(condition),
            target: target.into(),
            label: Some(label.into()),
        }
    }

    /// An async case with an explicit visualization label.
    pub fn labeled_async<F, Fut>(
        condition: F,
        target: impl Into<String>,
        label: impl Into<String>,
    ) -> Self
    where
        F: Fn(&Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        Self {
            condition: wrap_async_condition(condition),
            target: target.into(),
            label: Some(label.into()),
        }
    }
}

/// The default branch of a switch/case group.
pub struct Default {
    pub target: String,
}

impl Default {
    /// The default branch, routing to `target` when no case matches.
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_condition_true_for_conditional_single_and_selection_fanout() {
        let single = EdgeGroup::Single {
            source: "a".into(),
            target: "b".into(),
            condition: Some(wrap_sync_condition(|_| true)),
        };
        assert!(single.has_condition());

        let fanout = EdgeGroup::FanOut {
            source: "a".into(),
            targets: vec!["b".into(), "c".into()],
            selection: Some(Arc::new(|_: &Value, candidates: &[String]| {
                let candidates = candidates.to_vec();
                Box::pin(async move { candidates }) as BoxFuture<Vec<String>>
            })),
            case_labels: None,
        };
        assert!(fanout.has_condition());
    }

    #[test]
    fn has_condition_false_for_plain_edge_broadcast_and_fanin() {
        let plain = EdgeGroup::Single {
            source: "a".into(),
            target: "b".into(),
            condition: None,
        };
        assert!(!plain.has_condition());

        let broadcast = EdgeGroup::FanOut {
            source: "a".into(),
            targets: vec!["b".into(), "c".into()],
            selection: None,
            case_labels: None,
        };
        assert!(!broadcast.has_condition());

        let fanin = EdgeGroup::FanIn {
            sources: vec!["a".into(), "b".into()],
            target: "c".into(),
        };
        assert!(!fanin.has_condition());
    }
}
