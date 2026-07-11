//! Shared, run-scoped key→JSON state accessible to every executor.
//!
//! Rust equivalent of Python's `SharedState`. A single [`SharedState`] instance
//! is created per workflow run and cloned into each
//! [`WorkflowContext`](super::WorkflowContext), so all executors in a run
//! observe the same underlying map. Sub-workflows get their own isolated store
//! (a nested run creates a fresh [`SharedState`]).

use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// A thread-safe, async, string→JSON store shared by all executors in a run.
///
/// Cheap to clone (an `Arc` handle); clones share the same underlying map.
/// Guards against concurrent access with a [`tokio::sync::RwLock`].
///
/// Warning: keys beginning with `_` are reserved for internal framework use.
#[derive(Clone, Default)]
pub struct SharedState {
    inner: Arc<RwLock<HashMap<String, Value>>>,
}

impl SharedState {
    /// Create an empty shared state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a value by key, if present.
    pub async fn get(&self, key: &str) -> Option<Value> {
        self.inner.read().await.get(key).cloned()
    }

    /// Set a value, overwriting any existing entry.
    pub async fn set(&self, key: impl Into<String>, value: impl Into<Value>) {
        self.inner.write().await.insert(key.into(), value.into());
    }

    /// Whether a key exists.
    pub async fn has(&self, key: &str) -> bool {
        self.inner.read().await.contains_key(key)
    }

    /// Delete a key, returning whether it existed.
    pub async fn delete(&self, key: &str) -> bool {
        self.inner.write().await.remove(key).is_some()
    }

    /// Atomically read-modify-write a key under the write lock.
    ///
    /// The closure receives the current value (or `None`) and returns the new
    /// value to store. Rust analogue that fills the role of Python's
    /// hold/`set_within_hold` pattern for a single key.
    pub async fn update<F>(&self, key: impl Into<String>, f: F)
    where
        F: FnOnce(Option<Value>) -> Value,
    {
        let mut guard = self.inner.write().await;
        let key = key.into();
        let current = guard.get(&key).cloned();
        guard.insert(key, f(current));
    }

    /// Export a snapshot copy of the entire state (used for checkpointing).
    pub async fn export(&self) -> HashMap<String, Value> {
        self.inner.read().await.clone()
    }

    /// Merge a serialized state map into the current state.
    pub async fn import(&self, state: HashMap<String, Value>) {
        let mut guard = self.inner.write().await;
        for (k, v) in state {
            guard.insert(k, v);
        }
    }

    /// Remove all entries.
    pub async fn clear(&self) {
        self.inner.write().await.clear();
    }
}
