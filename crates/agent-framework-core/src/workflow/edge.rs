//! Edges and edge groups connecting executors.

use serde_json::Value;
use std::future::Future;
use std::sync::Arc;

use crate::error::Result;
use crate::tools::BoxFuture;

/// A predicate deciding whether a message should traverse an edge.
///
/// Evaluated **asynchronously**, mirroring upstream's `Edge.should_route`
/// becoming `async` (see `UPSTREAM_DRIFT.md` §10, `EdgeCondition = Callable[[Any],
/// bool | Awaitable[bool]]`), and **fallibly**: a predicate that cannot reach
/// a verdict says so instead of guessing one.
///
/// # Why a predicate needs an error channel
///
/// A realistic predicate inspects the payload — deserializes it, reads a
/// field, parses a date. Any of those can fail, and with a plain `bool` the
/// only thing a failed predicate can return is `false`. For a switch/case
/// group that means falling through to the default branch, so a *broken*
/// predicate is indistinguishable from one that correctly declined, and the
/// message is quietly delivered somewhere nobody chose. Upstream hit this and
/// fixed it from the other direction, by no longer swallowing predicate
/// exceptions (#8490).
///
/// Writing an infallible predicate is unchanged: every API that takes one
/// accepts a closure returning either `bool` or [`Result<bool>`], via
/// [`IntoConditionResult`]. Returning `Err` aborts the run with that error
/// rather than routing anywhere.
///
/// Callers normally build one with [`wrap_sync_condition`] or
/// [`wrap_async_condition`] rather than constructing the `Arc` directly.
pub type Condition = Arc<dyn Fn(&Value) -> BoxFuture<Result<bool>> + Send + Sync>;

/// A runtime target selector for multi-selection / switch edges: given the
/// message and the candidate target ids, return the ids to route to.
///
/// Evaluated asynchronously and fallibly for the same reasons as
/// [`Condition`] — a switch's selection function awaits each
/// [`Case::condition`](Case) in turn, and must be able to pass on a failure
/// rather than fall through to the default branch. Build one with
/// [`wrap_selection`] or [`wrap_async_selection`].
pub type Selection = Arc<dyn Fn(&Value, &[String]) -> BoxFuture<Result<Vec<String>>> + Send + Sync>;

/// What a condition closure may return: `bool` for a predicate that cannot
/// fail, or [`Result<bool>`] for one that can.
///
/// This is what keeps a fallible predicate from being a breaking change to
/// every builder that takes one: both forms satisfy the same bound, so
/// `|v| v["urgent"] == true` and `|v| Ok(serde_json::from_value::<Order>(v.clone())?.urgent)`
/// are both accepted where a condition is wanted.
pub trait IntoConditionResult: Send + 'static {
    /// The verdict, or the reason there isn't one.
    fn into_condition_result(self) -> Result<bool>;
}

impl IntoConditionResult for bool {
    fn into_condition_result(self) -> Result<bool> {
        Ok(self)
    }
}

impl IntoConditionResult for Result<bool> {
    fn into_condition_result(self) -> Result<bool> {
        self
    }
}

/// What a selection closure may return: the chosen target ids, or the reason
/// they could not be chosen. The [`Condition`] story, for [`Selection`].
pub trait IntoSelectionResult: Send + 'static {
    /// The chosen target ids, or the reason there are none.
    fn into_selection_result(self) -> Result<Vec<String>>;
}

impl IntoSelectionResult for Vec<String> {
    fn into_selection_result(self) -> Result<Vec<String>> {
        Ok(self)
    }
}

impl IntoSelectionResult for Result<Vec<String>> {
    fn into_selection_result(self) -> Result<Vec<String>> {
        self
    }
}

/// Wrap a synchronous predicate into a [`Condition`].
///
/// The closure may return `bool` or [`Result<bool>`] (see
/// [`IntoConditionResult`]). It is invoked eagerly, right when the condition
/// is called — not deferred inside the returned future — so wrapping does not
/// change evaluation order or timing for a sync call site; only the return
/// type becomes an (already-resolved) future.
pub fn wrap_sync_condition<R: IntoConditionResult>(
    f: impl Fn(&Value) -> R + Send + Sync + 'static,
) -> Condition {
    Arc::new(move |v: &Value| {
        let result = f(v).into_condition_result();
        Box::pin(async move { result }) as BoxFuture<Result<bool>>
    })
}

/// Wrap an async predicate into a [`Condition`]. The future may resolve to
/// `bool` or [`Result<bool>`] (see [`IntoConditionResult`]).
pub fn wrap_async_condition<F, Fut, R>(f: F) -> Condition
where
    F: Fn(&Value) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    R: IntoConditionResult,
{
    Arc::new(move |v: &Value| {
        let fut = f(v);
        Box::pin(async move { fut.await.into_condition_result() }) as BoxFuture<Result<bool>>
    })
}

/// Wrap a synchronous target selector into a [`Selection`]. The closure may
/// return `Vec<String>` or [`Result<Vec<String>>`] (see
/// [`IntoSelectionResult`]).
pub fn wrap_selection<R: IntoSelectionResult>(
    f: impl Fn(&Value, &[String]) -> R + Send + Sync + 'static,
) -> Selection {
    Arc::new(move |v: &Value, candidates: &[String]| {
        let result = f(v, candidates).into_selection_result();
        Box::pin(async move { result }) as BoxFuture<Result<Vec<String>>>
    })
}

/// Wrap an async target selector into a [`Selection`].
pub fn wrap_async_selection<F, Fut, R>(f: F) -> Selection
where
    F: Fn(&Value, &[String]) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = R> + Send + 'static,
    R: IntoSelectionResult,
{
    Arc::new(move |v: &Value, candidates: &[String]| {
        let fut = f(v, candidates);
        Box::pin(async move { fut.await.into_selection_result() }) as BoxFuture<Result<Vec<String>>>
    })
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
    pub fn new<R: IntoConditionResult>(
        condition: impl Fn(&Value) -> R + Send + Sync + 'static,
        target: impl Into<String>,
    ) -> Self {
        Self {
            condition: wrap_sync_condition(condition),
            target: target.into(),
            label: None,
        }
    }

    /// A case routing to `target` when the async `condition` holds.
    pub fn new_async<F, Fut, R>(condition: F, target: impl Into<String>) -> Self
    where
        F: Fn(&Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoConditionResult,
    {
        Self {
            condition: wrap_async_condition(condition),
            target: target.into(),
            label: None,
        }
    }

    /// A case with an explicit visualization label.
    pub fn labeled<R: IntoConditionResult>(
        condition: impl Fn(&Value) -> R + Send + Sync + 'static,
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
    pub fn labeled_async<F, Fut, R>(
        condition: F,
        target: impl Into<String>,
        label: impl Into<String>,
    ) -> Self
    where
        F: Fn(&Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoConditionResult,
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
            selection: Some(wrap_selection(|_: &Value, candidates: &[String]| {
                candidates.to_vec()
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
