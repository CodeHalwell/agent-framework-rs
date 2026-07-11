//! Workflow checkpointing: snapshot structs and storage backends.
//!
//! Rust equivalent of Python's `_checkpoint.py` / `_checkpoint_summary.py`. A
//! [`WorkflowCheckpoint`] captures the full run state at a superstep boundary so
//! a workflow can pause and resume, including across process restarts (via
//! [`FileCheckpointStorage`]). Because all workflow data is `serde_json::Value`,
//! no per-type marker encoding (Python's `_checkpoint_encoding`) is required.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use super::context::WorkflowMessage;
use super::request_info::PendingRequest;
use crate::error::{Error, Result};

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn default_version() -> String {
    "1.0".to_string()
}

/// A complete snapshot of a workflow's execution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCheckpoint {
    /// Unique id for this checkpoint.
    pub checkpoint_id: String,
    /// Id of the workflow this checkpoint belongs to.
    pub workflow_id: String,
    /// Optional human-readable workflow name.
    #[serde(default)]
    pub workflow_name: Option<String>,
    /// Creation time in milliseconds since the Unix epoch.
    pub timestamp_millis: u64,
    /// The superstep iteration count at snapshot time.
    pub iteration_count: usize,
    /// In-flight messages awaiting delivery on resume.
    pub messages: Vec<WorkflowMessage>,
    /// Per-executor serialized state, keyed by executor id.
    pub executor_states: HashMap<String, Value>,
    /// The run's shared state map.
    pub shared_state: HashMap<String, Value>,
    /// Requests outstanding at snapshot time.
    pub pending_requests: Vec<PendingRequest>,
    /// Partially-satisfied fan-in barriers: `target -> (source -> value)`.
    ///
    /// When a fan-in target has received some but not all of its sources'
    /// messages, the already-delivered messages are buffered on the runner
    /// (`WorkflowRun::fanin`) until the barrier fires. If a checkpoint is taken
    /// between source deliveries (they can arrive in different supersteps),
    /// this buffer must be captured or the barrier can never complete on
    /// resume. Empty for legacy checkpoints written before this field existed
    /// (serde default), and for checkpoints with no in-flight fan-in.
    #[serde(default)]
    pub fanin_state: HashMap<String, HashMap<String, Value>>,
    /// Additional metadata (e.g. superstep index, checkpoint type).
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// A deterministic signature of the workflow graph (executor ids + edge
    /// topology) this checkpoint was produced from, used to detect resuming
    /// against an incompatible graph. Empty for legacy checkpoints written
    /// before signatures existed (serde default), which resume with a warning
    /// rather than a hard error. See `runner::compute_graph_signature`.
    #[serde(default)]
    pub graph_signature: String,
    /// Checkpoint format version.
    #[serde(default = "default_version")]
    pub version: String,
}

impl WorkflowCheckpoint {
    /// Create a checkpoint with a fresh id and current timestamp.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        workflow_id: String,
        workflow_name: Option<String>,
        iteration_count: usize,
        messages: Vec<WorkflowMessage>,
        executor_states: HashMap<String, Value>,
        shared_state: HashMap<String, Value>,
        pending_requests: Vec<PendingRequest>,
        fanin_state: HashMap<String, HashMap<String, Value>>,
        metadata: HashMap<String, Value>,
        graph_signature: String,
    ) -> Self {
        Self {
            checkpoint_id: uuid::Uuid::new_v4().to_string(),
            workflow_id,
            workflow_name,
            timestamp_millis: now_millis(),
            iteration_count,
            messages,
            executor_states,
            shared_state,
            pending_requests,
            fanin_state,
            metadata,
            graph_signature,
            version: default_version(),
        }
    }
}

/// A human-readable summary of a checkpoint. Rust analogue of
/// `WorkflowCheckpointSummary`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowCheckpointSummary {
    pub checkpoint_id: String,
    pub timestamp_millis: u64,
    pub iteration_count: usize,
    /// Distinct message targets in transit, sorted.
    pub targets: Vec<String>,
    /// Executor ids with persisted state, sorted.
    pub executor_ids: Vec<String>,
    /// A coarse status string: `idle`, `awaiting request response`, or
    /// `awaiting next superstep`.
    pub status: String,
    /// Ids of requests outstanding at snapshot time.
    pub pending_request_ids: Vec<String>,
}

/// Summarize a checkpoint. Rust analogue of `get_checkpoint_summary`.
pub fn get_checkpoint_summary(checkpoint: &WorkflowCheckpoint) -> WorkflowCheckpointSummary {
    let mut targets: Vec<String> = checkpoint
        .messages
        .iter()
        .filter_map(|m| m.target_id.clone())
        .collect();
    targets.sort();
    targets.dedup();

    let mut executor_ids: Vec<String> = checkpoint.executor_states.keys().cloned().collect();
    executor_ids.sort();

    let mut pending_request_ids: Vec<String> = checkpoint
        .pending_requests
        .iter()
        .map(|r| r.request_id.clone())
        .collect();
    pending_request_ids.sort();

    let status = if !pending_request_ids.is_empty() {
        "awaiting request response"
    } else if checkpoint.messages.is_empty() {
        "idle"
    } else {
        "awaiting next superstep"
    }
    .to_string();

    WorkflowCheckpointSummary {
        checkpoint_id: checkpoint.checkpoint_id.clone(),
        timestamp_millis: checkpoint.timestamp_millis,
        iteration_count: checkpoint.iteration_count,
        targets,
        executor_ids,
        status,
        pending_request_ids,
    }
}

/// Storage backend for workflow checkpoints.
///
/// Rust analogue of Python's `CheckpointStorage` protocol.
#[async_trait]
pub trait CheckpointStorage: Send + Sync {
    /// Persist a checkpoint, returning its id.
    async fn save(&self, checkpoint: WorkflowCheckpoint) -> Result<String>;

    /// Load a checkpoint by id, or `None` if it does not exist.
    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>>;

    /// List checkpoints, optionally filtered by workflow id.
    async fn list(&self, workflow_id: Option<&str>) -> Result<Vec<WorkflowCheckpoint>>;

    /// Delete a checkpoint by id, returning whether it existed.
    async fn delete(&self, checkpoint_id: &str) -> Result<bool>;
}

/// In-memory checkpoint storage for testing and development.
#[derive(Clone, Default)]
pub struct InMemoryCheckpointStorage {
    checkpoints: Arc<Mutex<HashMap<String, WorkflowCheckpoint>>>,
}

impl InMemoryCheckpointStorage {
    /// Create empty in-memory storage.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl CheckpointStorage for InMemoryCheckpointStorage {
    async fn save(&self, checkpoint: WorkflowCheckpoint) -> Result<String> {
        let id = checkpoint.checkpoint_id.clone();
        self.checkpoints
            .lock()
            .unwrap()
            .insert(id.clone(), checkpoint);
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        Ok(self.checkpoints.lock().unwrap().get(checkpoint_id).cloned())
    }

    async fn list(&self, workflow_id: Option<&str>) -> Result<Vec<WorkflowCheckpoint>> {
        let mut out: Vec<WorkflowCheckpoint> = self
            .checkpoints
            .lock()
            .unwrap()
            .values()
            .filter(|cp| workflow_id.is_none_or(|w| cp.workflow_id == w))
            .cloned()
            .collect();
        out.sort_by_key(|a| a.timestamp_millis);
        Ok(out)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        Ok(self
            .checkpoints
            .lock()
            .unwrap()
            .remove(checkpoint_id)
            .is_some())
    }
}

/// File-based checkpoint storage: one JSON file per checkpoint in a directory.
#[derive(Clone)]
pub struct FileCheckpointStorage {
    dir: PathBuf,
}

impl FileCheckpointStorage {
    /// Open (creating if necessary) file storage rooted at `dir`.
    pub fn new(dir: impl Into<PathBuf>) -> Result<Self> {
        let dir = dir.into();
        std::fs::create_dir_all(&dir)
            .map_err(|e| Error::Workflow(format!("failed to create checkpoint dir: {e}")))?;
        Ok(Self { dir })
    }

    fn path_for(&self, checkpoint_id: &str) -> PathBuf {
        self.dir.join(format!("{checkpoint_id}.json"))
    }

    async fn read_file(path: &Path) -> Result<WorkflowCheckpoint> {
        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| Error::Workflow(format!("failed to read checkpoint: {e}")))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::Workflow(format!("failed to parse checkpoint: {e}")))
    }
}

#[async_trait]
impl CheckpointStorage for FileCheckpointStorage {
    async fn save(&self, checkpoint: WorkflowCheckpoint) -> Result<String> {
        let id = checkpoint.checkpoint_id.clone();
        let path = self.path_for(&id);
        let json = serde_json::to_vec_pretty(&checkpoint)
            .map_err(|e| Error::Workflow(format!("failed to serialize checkpoint: {e}")))?;
        // Write atomically via a temp file + rename.
        let tmp = path.with_extension("json.tmp");
        tokio::fs::write(&tmp, &json)
            .await
            .map_err(|e| Error::Workflow(format!("failed to write checkpoint: {e}")))?;
        tokio::fs::rename(&tmp, &path)
            .await
            .map_err(|e| Error::Workflow(format!("failed to finalize checkpoint: {e}")))?;
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        let path = self.path_for(checkpoint_id);
        match Self::read_file(&path).await {
            Ok(cp) => Ok(Some(cp)),
            Err(_) if !path.exists() => Ok(None),
            Err(e) => Err(e),
        }
    }

    async fn list(&self, workflow_id: Option<&str>) -> Result<Vec<WorkflowCheckpoint>> {
        let mut out = Vec::new();
        let mut entries = tokio::fs::read_dir(&self.dir)
            .await
            .map_err(|e| Error::Workflow(format!("failed to list checkpoint dir: {e}")))?;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            if let Ok(cp) = Self::read_file(&path).await {
                if workflow_id.is_none_or(|w| cp.workflow_id == w) {
                    out.push(cp);
                }
            }
        }
        out.sort_by_key(|a| a.timestamp_millis);
        Ok(out)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        let path = self.path_for(checkpoint_id);
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(Error::Workflow(format!("failed to delete checkpoint: {e}"))),
        }
    }
}
