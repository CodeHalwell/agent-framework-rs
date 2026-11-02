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

/// Reject any checkpoint id that is not a strict filename-safe token, so it can
/// never be used to traverse outside a [`FileCheckpointStorage`] directory.
///
/// Allowed: a non-empty string of ASCII alphanumerics, `-`, and `_` (which
/// admits UUIDs — the ids [`WorkflowCheckpoint::new`] generates — and any
/// reasonable custom id). Everything else is rejected, including path
/// separators (`/`, `\`), `.`/`..` traversal, absolute paths, drive prefixes
/// (`C:`), and the empty string.
fn validate_checkpoint_id(checkpoint_id: &str) -> Result<()> {
    if checkpoint_id.is_empty() {
        return Err(Error::Workflow(
            "checkpoint id must not be empty".to_string(),
        ));
    }
    if !checkpoint_id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(Error::Workflow(format!(
            "invalid checkpoint id {checkpoint_id:?}: only ASCII alphanumerics, '-', and '_' are allowed"
        )));
    }
    Ok(())
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

/// Order two checkpoints as [`CheckpointStorage::list`] returns them:
/// oldest first, by `(timestamp_millis, iteration_count, checkpoint_id)`.
///
/// Sorting on `timestamp_millis` alone is not a total order, and that matters.
/// Checkpoints are written one per superstep boundary, so two of them can
/// share a millisecond, and then the tie is broken by whatever order the
/// backend happened to enumerate: `HashMap` iteration order for
/// [`InMemoryCheckpointStorage`] (which varies between runs of the same
/// program), directory order for [`FileCheckpointStorage`], and the server's
/// row order for a remote store. A caller that reads the end of the list as
/// "the newest checkpoint" could therefore resume from a *superseded* one,
/// and do it intermittently.
///
/// `iteration_count` is what settles it: within one workflow run it is the
/// superstep counter, so the higher of two checkpoints sharing a timestamp is
/// the later one. That is the same supersession upstream recovers by walking
/// `previous_checkpoint_id` (#9261); this port has no such lineage field and
/// does not need one, because the superstep counter already encodes it.
/// `checkpoint_id` is the final key so that checkpoints from *different* runs
/// landing on the same timestamp and superstep still come back in a stable
/// order rather than an arbitrary one.
pub fn compare_checkpoints(a: &WorkflowCheckpoint, b: &WorkflowCheckpoint) -> std::cmp::Ordering {
    a.timestamp_millis
        .cmp(&b.timestamp_millis)
        .then_with(|| a.iteration_count.cmp(&b.iteration_count))
        .then_with(|| a.checkpoint_id.cmp(&b.checkpoint_id))
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
    ///
    /// Implementations return them oldest first, in
    /// [`compare_checkpoints`] order — a total order, so the last element is
    /// always *the* newest checkpoint and never an arbitrary pick among
    /// several sharing a millisecond. An implementation outside this crate
    /// should sort with that comparator rather than by `timestamp_millis`
    /// alone.
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
        out.sort_by(compare_checkpoints);
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

    /// Resolve the on-disk path for `checkpoint_id`, after validating that the
    /// id cannot escape the configured checkpoint directory.
    ///
    /// Checkpoint ids reach [`CheckpointStorage::load`]/[`CheckpointStorage::delete`]
    /// from callers, so a value like `../secret` or an absolute path must never
    /// be joined onto `self.dir` verbatim: `PathBuf::join` with an absolute
    /// second component silently discards the root entirely. We reject anything
    /// that is not a strict filename-safe token (see [`validate_checkpoint_id`]),
    /// which forbids path separators, `.`/`..`, drive prefixes, and empty ids.
    fn path_for(&self, checkpoint_id: &str) -> Result<PathBuf> {
        validate_checkpoint_id(checkpoint_id)?;
        let path = self.dir.join(format!("{checkpoint_id}.json"));
        // Defense in depth: after joining, the parent must still be exactly the
        // configured directory. With the strict grammar above this always holds,
        // but the check keeps the invariant explicit and local to path building.
        if path.parent() != Some(self.dir.as_path()) {
            return Err(Error::Workflow(format!(
                "checkpoint id {checkpoint_id:?} escapes the checkpoint directory"
            )));
        }
        Ok(path)
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
        // Validate even on save: a caller can construct a checkpoint with an
        // arbitrary id, and we must not write outside the configured directory.
        let path = self.path_for(&id)?;
        let json = serde_json::to_vec_pretty(&checkpoint)
            .map_err(|e| Error::Workflow(format!("failed to serialize checkpoint: {e}")))?;
        // Write atomically via a temp file + rename. The temp name carries a
        // nonce rather than being derived from the checkpoint id alone: two
        // saves of the *same* id would otherwise share one temp path, so one
        // would write into the file the other was still writing (a checkpoint
        // that parses as neither) and then fail its own rename with
        // `NotFound`, because the first save had already moved that file away.
        // With a name per save, each writes its own file and the renames are
        // atomic, so a reader sees one complete checkpoint or the other —
        // whichever renamed last, which is what concurrent saves of one id
        // mean. Mirrors upstream #7757.
        let tmp = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        if let Err(e) = tokio::fs::write(&tmp, &json).await {
            // Best effort: a partial temp file left in the checkpoint
            // directory is never read (`list` matches `.json` only), but it
            // is still litter.
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::Workflow(format!("failed to write checkpoint: {e}")));
        }
        if let Err(e) = tokio::fs::rename(&tmp, &path).await {
            let _ = tokio::fs::remove_file(&tmp).await;
            return Err(Error::Workflow(format!(
                "failed to finalize checkpoint: {e}"
            )));
        }
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        let path = self.path_for(checkpoint_id)?;
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
        out.sort_by(compare_checkpoints);
        Ok(out)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        let path = self.path_for(checkpoint_id)?;
        match tokio::fs::remove_file(&path).await {
            Ok(()) => Ok(true),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(e) => Err(Error::Workflow(format!("failed to delete checkpoint: {e}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("afr-checkpoint-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Build a checkpoint with an explicit id, timestamp and superstep, so a
    /// timestamp tie can be set up deliberately.
    fn cp_at(id: &str, timestamp_millis: u64, iteration_count: usize) -> WorkflowCheckpoint {
        WorkflowCheckpoint {
            checkpoint_id: id.to_string(),
            workflow_id: "wf".to_string(),
            workflow_name: None,
            timestamp_millis,
            iteration_count,
            messages: Vec::new(),
            executor_states: HashMap::new(),
            shared_state: HashMap::new(),
            pending_requests: Vec::new(),
            fanin_state: HashMap::new(),
            metadata: HashMap::new(),
            graph_signature: String::new(),
            version: default_version(),
        }
    }

    #[test]
    fn checkpoints_sharing_a_timestamp_order_by_superstep() {
        // Checkpoints are written one per superstep boundary, so two can land
        // in the same millisecond. The later superstep supersedes the earlier
        // one and has to sort after it, however the backend enumerated them.
        let earlier = cp_at("a", 1_000, 3);
        let later = cp_at("b", 1_000, 4);
        assert_eq!(
            compare_checkpoints(&earlier, &later),
            std::cmp::Ordering::Less
        );
        assert_eq!(
            compare_checkpoints(&later, &earlier),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn a_newer_timestamp_wins_over_a_higher_superstep() {
        // Timestamp is still the primary key: a later checkpoint from a
        // shorter run must not be ordered before one from a longer earlier run.
        let older_longer_run = cp_at("a", 1_000, 9);
        let newer_shorter_run = cp_at("b", 2_000, 1);
        assert_eq!(
            compare_checkpoints(&older_longer_run, &newer_shorter_run),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn a_full_tie_is_broken_deterministically_by_id() {
        // Different runs can collide on both timestamp and superstep. The
        // order between them is arbitrary but must not vary between calls.
        let a = cp_at("aaa", 1_000, 2);
        let b = cp_at("bbb", 1_000, 2);
        assert_eq!(compare_checkpoints(&a, &b), std::cmp::Ordering::Less);
        assert_eq!(compare_checkpoints(&a, &a), std::cmp::Ordering::Equal);
    }

    #[tokio::test]
    async fn in_memory_list_puts_the_newest_checkpoint_last_despite_a_timestamp_tie() {
        // The real defect: `InMemoryCheckpointStorage` is backed by a
        // `HashMap`, whose iteration order varies between runs of the same
        // program. Sorting on `timestamp_millis` alone left the tie to that
        // order, so a caller reading the end of the list as "the newest
        // checkpoint" could resume from the superseded one — intermittently.
        //
        // Saved newest-first so a stable sort on timestamp alone would leave
        // the superseded checkpoint last for at least some hash orders.
        let storage = InMemoryCheckpointStorage::new();
        for (id, iteration) in [("z-newest", 7usize), ("a-superseded", 6)] {
            storage.save(cp_at(id, 5_000, iteration)).await.unwrap();
        }

        let listed = storage.list(Some("wf")).await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(
            listed.last().unwrap().checkpoint_id,
            "z-newest",
            "the highest superstep must be last: {:?}",
            listed
                .iter()
                .map(|c| (&c.checkpoint_id, c.iteration_count))
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn file_list_puts_the_newest_checkpoint_last_despite_a_timestamp_tie() {
        // Same contract for the file store, where the tie would otherwise
        // fall to directory enumeration order.
        let dir = tmp_dir();
        let storage = FileCheckpointStorage::new(&dir).unwrap();
        for (id, iteration) in [("z-newest", 2usize), ("a-superseded", 1)] {
            storage.save(cp_at(id, 5_000, iteration)).await.unwrap();
        }

        let listed = storage.list(Some("wf")).await.unwrap();
        assert_eq!(listed.len(), 2);
        assert_eq!(listed.last().unwrap().checkpoint_id, "z-newest");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn validate_checkpoint_id_accepts_uuids_and_safe_tokens() {
        assert!(validate_checkpoint_id(&uuid::Uuid::new_v4().to_string()).is_ok());
        assert!(validate_checkpoint_id("my-checkpoint_01").is_ok());
        assert!(validate_checkpoint_id("abc123").is_ok());
    }

    #[test]
    fn validate_checkpoint_id_rejects_traversal_and_absolute_ids() {
        for bad in [
            "",
            "..",
            ".",
            "../other-file",
            "../../etc/passwd",
            "sub/dir",
            "a/b",
            "with space",
            "with.dot",
            // Unix absolute.
            "/etc/passwd",
            // Windows-style traversal and drive prefixes.
            "..\\other",
            "C:\\Windows\\System32\\config",
            "\\\\server\\share",
        ] {
            assert!(
                validate_checkpoint_id(bad).is_err(),
                "expected {bad:?} to be rejected"
            );
        }
    }

    #[tokio::test]
    async fn file_storage_rejects_traversal_on_load_and_delete() {
        let dir = tmp_dir();
        let storage = FileCheckpointStorage::new(&dir).unwrap();

        // Plant a sibling file outside the checkpoint dir that a traversal id
        // would otherwise reach.
        let outside = dir.parent().unwrap().join("afr-checkpoint-secret.json");
        std::fs::write(&outside, "{}").unwrap();

        let escape_id = format!("../{}", outside.file_stem().unwrap().to_string_lossy());
        assert!(storage.load(&escape_id).await.is_err());
        assert!(storage.delete(&escape_id).await.is_err());
        // The outside file must still be present — delete never reached it.
        assert!(
            outside.exists(),
            "traversal delete must not touch outside files"
        );

        // An absolute id must not replace the storage root either.
        let abs = outside.to_string_lossy().replace(".json", "");
        assert!(storage.load(&abs).await.is_err());
        assert!(storage.delete(&abs).await.is_err());

        std::fs::remove_file(&outside).ok();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn file_storage_round_trips_a_valid_checkpoint() {
        let dir = tmp_dir();
        let storage = FileCheckpointStorage::new(&dir).unwrap();
        let cp = WorkflowCheckpoint::new(
            "wf-1".to_string(),
            None,
            0,
            Vec::new(),
            HashMap::new(),
            HashMap::new(),
            Vec::new(),
            HashMap::new(),
            HashMap::new(),
            String::new(),
        );
        let id = storage.save(cp).await.unwrap();
        assert!(storage.load(&id).await.unwrap().is_some());
        assert!(storage.delete(&id).await.unwrap());
        assert!(storage.load(&id).await.unwrap().is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn concurrent_saves_of_one_checkpoint_id_do_not_corrupt_it() {
        // A shared temp path made two saves of the same id write into one
        // file — so the survivor could be a mix of both, and the loser's
        // rename failed with `NotFound` because the winner had already moved
        // that file away (upstream #7757).
        let dir = tmp_dir();
        let storage = FileCheckpointStorage::new(&dir).unwrap();
        let checkpoint = |iteration: usize| {
            let mut cp = WorkflowCheckpoint::new(
                "wf-1".to_string(),
                None,
                iteration,
                Vec::new(),
                HashMap::new(),
                HashMap::new(),
                Vec::new(),
                HashMap::new(),
                HashMap::new(),
                String::new(),
            );
            // One id, many writers.
            cp.checkpoint_id = "same-id".to_string();
            // Big enough that a write cannot land in one atomic chunk, which
            // is what makes an interleaved write visible at all.
            cp.metadata.insert(
                "filler".to_string(),
                serde_json::json!("x".repeat(200_000 + iteration)),
            );
            cp
        };

        let saves = (0..8).map(|i| storage.save(checkpoint(i)));
        let results = futures::future::join_all(saves).await;
        for result in &results {
            assert!(result.is_ok(), "no save may fail: {result:?}");
        }

        // Whichever renamed last wins, and what it wrote is one whole
        // checkpoint rather than a blend of several.
        let loaded = storage
            .load("same-id")
            .await
            .expect("the surviving checkpoint parses")
            .expect("it is there");
        let filler = loaded.metadata["filler"].as_str().unwrap();
        assert_eq!(filler.len(), 200_000 + loaded.iteration_count);

        // And no temp files are left behind.
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "left behind {leftovers:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
