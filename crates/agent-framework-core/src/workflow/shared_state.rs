//! Shared, run-scoped key→JSON state accessible to every executor.
//!
//! Rust equivalent of Python's `State` (`_workflows/_state.py`). A single
//! [`SharedState`] instance is created per workflow run and cloned into each
//! [`WorkflowContext`](super::WorkflowContext), so all executors in a run
//! observe the same underlying store. Sub-workflows get their own isolated
//! store (a nested run creates a fresh [`SharedState`]).
//!
//! # Superstep semantics
//!
//! State uses the Pregel-style **staged** model upstream adopted: writes are
//! buffered in a *pending* map and only folded into *committed* state at
//! superstep boundaries, when the runner calls [`SharedState::commit`] after a
//! superstep's executors all finish (and before checkpointing). Reads check
//! pending first, then committed, so an executor sees its own writes; but
//! [`SharedState::export`] returns **committed only**, so a checkpoint taken at
//! a superstep boundary reflects exactly the writes of completed supersteps —
//! never a half-written mid-superstep value. A superstep that fails never
//! reaches `commit`, so its partial writes are discarded rather than persisted.

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A staged pending write: either a new/updated value or a tombstone marking
/// the key for deletion from committed state at [`SharedState::commit`].
#[derive(Clone)]
enum Pending {
    Set(Value),
    Delete,
}

#[derive(Default)]
struct Inner {
    committed: HashMap<String, Value>,
    pending: HashMap<String, Pending>,
}

/// A thread-safe, async, string→JSON store shared by all executors in a run.
///
/// Cheap to clone (an `Arc` handle); clones share the same underlying store.
/// Guards against concurrent access with a [`tokio::sync::RwLock`]. See the
/// module docs for the staged (pending/committed) superstep semantics.
///
/// Warning: keys beginning with `_` are reserved for internal framework use.
#[derive(Clone, Default)]
pub struct SharedState {
    inner: Arc<RwLock<Inner>>,
}

impl SharedState {
    /// Create an empty shared state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a value by key, if present. Checks the pending buffer first (so an
    /// executor observes its own writes) and then committed state.
    pub async fn get(&self, key: &str) -> Option<Value> {
        let guard = self.inner.read().await;
        match guard.pending.get(key) {
            Some(Pending::Set(v)) => Some(v.clone()),
            Some(Pending::Delete) => None,
            None => guard.committed.get(key).cloned(),
        }
    }

    /// Stage a value for `key`, overwriting any existing pending write. The
    /// value is visible to subsequent [`SharedState::get`] calls but is not
    /// folded into committed state until [`SharedState::commit`].
    pub async fn set(&self, key: impl Into<String>, value: impl Into<Value>) {
        self.inner
            .write()
            .await
            .pending
            .insert(key.into(), Pending::Set(value.into()));
    }

    /// Whether a key exists in pending (as a non-tombstone) or committed state.
    pub async fn has(&self, key: &str) -> bool {
        let guard = self.inner.read().await;
        match guard.pending.get(key) {
            Some(Pending::Set(_)) => true,
            Some(Pending::Delete) => false,
            None => guard.committed.contains_key(key),
        }
    }

    /// Stage a deletion of `key`, returning whether the key currently exists.
    ///
    /// If the key exists only in the pending buffer it is removed there; if it
    /// exists in committed state a tombstone is staged so the key is removed at
    /// [`SharedState::commit`].
    pub async fn delete(&self, key: &str) -> bool {
        let mut guard = self.inner.write().await;
        match guard.pending.get(key) {
            Some(Pending::Delete) => false,
            Some(Pending::Set(_)) => {
                // Staged but not committed: dropping the pending write suffices,
                // unless committed also holds the key (then keep a tombstone).
                if guard.committed.contains_key(key) {
                    guard.pending.insert(key.to_string(), Pending::Delete);
                } else {
                    guard.pending.remove(key);
                }
                true
            }
            None => {
                if guard.committed.contains_key(key) {
                    guard.pending.insert(key.to_string(), Pending::Delete);
                    true
                } else {
                    false
                }
            }
        }
    }

    /// Atomically read-modify-write a key under the write lock, staging the
    /// result into the pending buffer.
    ///
    /// The closure receives the current value (pending-first, then committed)
    /// or `None` and returns the new value to stage. Rust analogue of Python's
    /// hold/`set_within_hold` pattern for a single key.
    pub async fn update<F>(&self, key: impl Into<String>, f: F)
    where
        F: FnOnce(Option<Value>) -> Value,
    {
        let mut guard = self.inner.write().await;
        let key = key.into();
        let current = match guard.pending.get(&key) {
            Some(Pending::Set(v)) => Some(v.clone()),
            Some(Pending::Delete) => None,
            None => guard.committed.get(&key).cloned(),
        };
        guard.pending.insert(key, Pending::Set(f(current)));
    }

    /// Fold all pending writes into committed state and clear the buffer.
    ///
    /// Called by the runner at each superstep boundary, after the superstep's
    /// executors finish successfully and before checkpointing.
    pub async fn commit(&self) {
        let mut guard = self.inner.write().await;
        let pending = std::mem::take(&mut guard.pending);
        for (k, entry) in pending {
            match entry {
                Pending::Set(v) => {
                    guard.committed.insert(k, v);
                }
                Pending::Delete => {
                    guard.committed.remove(&k);
                }
            }
        }
    }

    /// Discard all pending writes without committing them.
    pub async fn discard(&self) {
        self.inner.write().await.pending.clear();
    }

    /// Export a snapshot copy of the **committed** state (used for
    /// checkpointing). Pending writes are deliberately excluded.
    pub async fn export(&self) -> HashMap<String, Value> {
        self.inner.read().await.committed.clone()
    }

    /// Merge a serialized state map into committed state (used on restore).
    pub async fn import(&self, state: HashMap<String, Value>) {
        let mut guard = self.inner.write().await;
        for (k, v) in state {
            guard.committed.insert(k, v);
        }
    }

    /// Remove all entries, both committed and pending.
    pub async fn clear(&self) {
        let mut guard = self.inner.write().await;
        guard.committed.clear();
        guard.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn writes_are_staged_until_commit() {
        let s = SharedState::new();
        s.set("k", json!(1)).await;
        // Visible to the writer via pending-first reads...
        assert_eq!(s.get("k").await, Some(json!(1)));
        assert!(s.has("k").await);
        // ...but not yet in committed/exported state.
        assert!(s.export().await.is_empty());
        s.commit().await;
        assert_eq!(s.export().await.get("k"), Some(&json!(1)));
    }

    #[tokio::test]
    async fn discard_drops_pending_but_keeps_committed() {
        let s = SharedState::new();
        s.set("k", json!(1)).await;
        s.commit().await;
        s.set("k", json!(2)).await;
        s.discard().await;
        assert_eq!(s.get("k").await, Some(json!(1)));
    }

    #[tokio::test]
    async fn delete_tombstones_committed_key_until_commit() {
        let s = SharedState::new();
        s.set("k", json!(1)).await;
        s.commit().await;
        assert!(s.delete("k").await);
        // Hidden immediately from reads, but only removed from committed at commit.
        assert_eq!(s.get("k").await, None);
        assert!(!s.has("k").await);
        assert!(s.export().await.contains_key("k"));
        s.commit().await;
        assert!(!s.export().await.contains_key("k"));
        // Deleting an absent key reports no existing entry.
        assert!(!s.delete("missing").await);
    }

    #[tokio::test]
    async fn delete_of_pending_only_key_drops_the_write() {
        let s = SharedState::new();
        s.set("k", json!(1)).await;
        assert!(s.delete("k").await);
        assert_eq!(s.get("k").await, None);
        s.commit().await;
        assert!(s.export().await.is_empty());
    }

    #[tokio::test]
    async fn update_reads_pending_first() {
        let s = SharedState::new();
        s.set("n", json!(1)).await;
        s.update("n", |cur| {
            json!(cur.and_then(|v| v.as_i64()).unwrap_or(0) + 10)
        })
        .await;
        assert_eq!(s.get("n").await, Some(json!(11)));
    }
}
