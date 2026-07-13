//! A [`CheckpointStorage`] backed by Azure Cosmos DB (NoSQL / SQL API),
//! ported from .NET's `Microsoft.Agents.AI.CosmosNoSql.CosmosCheckpointStorage`
//! over the raw Cosmos REST API (see [`crate::client`] and [`crate::auth`])
//! rather than the `Microsoft.Azure.Cosmos` SDK — the checkpoint-storage
//! sibling of [`crate::CosmosChatMessageStore`].
//!
//! Every [`WorkflowCheckpoint`] is its own document/item:
//!
//! ```json
//! {
//!   "id": "<checkpoint.checkpoint_id>",
//!   "workflowId": "<checkpoint.workflow_id>",
//!   "checkpoint": "<WorkflowCheckpoint, JSON-serialized to a STRING>"
//! }
//! ```
//!
//! `checkpoint` is a JSON **string** (double-encoded), not a nested object —
//! the same round-tripping convention [`crate::chat_message_store`] uses for
//! its `message` field (see that module's docs for the rationale: it keeps
//! this store agnostic to the exact shape of the serialized type).
//!
//! # Partitioning: `/id`, not `/workflowId`
//!
//! Unlike [`crate::CosmosChatMessageStore`] (partitioned by `threadId`,
//! since many messages share a thread and are listed together),
//! [`CheckpointStorage::load`] and [`CheckpointStorage::delete`] are keyed
//! by `checkpoint_id` alone — no `workflow_id` is available at that call
//! site to scope a partition-key lookup. So this store partitions by `id`
//! itself (each checkpoint is a single-item partition), which keeps
//! [`CosmosCheckpointStorage::save`]/`load`/`delete` cheap point
//! operations. The tradeoff is [`CheckpointStorage::list`]: since
//! `workflowId` isn't the partition key, it runs a cross-partition query
//! ([`crate::client::CosmosRestClient::query_documents_cross_partition`])
//! rather than a single-partition one.
//!
//! # Divergences from .NET
//!
//! - **Auth**: master key only (HMAC request signing — see [`crate::auth`]).
//!   Same divergence as [`crate::CosmosChatMessageStore`]; see `PARITY.md`.
//! - **Partitioning**: .NET's `CosmosCheckpointStorage` partitions by
//!   `workflowId` (checkpoints for one workflow share a partition, and
//!   `DeleteAsync`/point reads there always have a `workflowId` in hand via
//!   the surrounding `CosmosDBCheckpointManager`). This port's
//!   [`CheckpointStorage`] trait signatures don't carry `workflow_id`
//!   through to `load`/`delete`, so `/id` partitioning (above) is used
//!   instead — functionally equivalent, but trades cheaper single-workflow
//!   `list` calls for cheaper point reads/deletes.
//! - **Existence check on delete**: Cosmos's `DELETE` doesn't report
//!   whether a document existed, but [`CheckpointStorage::delete`] must
//!   return that as a `bool`. This store does a point [`Self`]-internal
//!   read-then-delete (two round trips, not atomic as a unit) to answer it,
//!   matching the "best effort, not linearizable" caveat
//!   [`crate::CosmosChatMessageStore`]'s module docs already carry for its
//!   own multi-request operations.

use async_trait::async_trait;
use serde_json::Value;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::workflow::{CheckpointStorage, WorkflowCheckpoint};

use crate::client::CosmosRestClient;

/// Cosmos DB container partition key path used by
/// [`CosmosCheckpointStorage`]: `/id` — every checkpoint document is its
/// own partition. See the module docs for why this differs from
/// [`crate::DEFAULT_PARTITION_KEY_PATH`]'s `/threadId`.
pub const DEFAULT_PARTITION_KEY_PATH: &str = "/id";

/// Build the Cosmos document for one [`WorkflowCheckpoint`]. The document
/// `id` is the checkpoint's own `checkpoint_id` (also this store's
/// partition key value, per the [module docs](self)) so `save`/`load`/
/// `delete` are all single-document point operations addressed by
/// `checkpoint_id` alone. `workflowId` is duplicated out to a top-level
/// field (even though it's also embedded in the serialized `checkpoint`
/// string) purely so [`CheckpointStorage::list`]'s `workflow_id` filter can
/// be expressed as a Cosmos `WHERE` clause without deserializing every
/// candidate document first.
fn build_checkpoint_document(checkpoint: &WorkflowCheckpoint) -> Result<Value> {
    let checkpoint_json = serde_json::to_string(checkpoint)?;
    Ok(serde_json::json!({
        "id": checkpoint.checkpoint_id,
        "workflowId": checkpoint.workflow_id,
        "checkpoint": checkpoint_json,
    }))
}

/// Parse one Cosmos document back into a [`WorkflowCheckpoint`], requiring
/// a string `checkpoint` field (see the [module docs](self) for the
/// double-encoded-string shape). Mirrors
/// [`crate::chat_message_store::parse_message_document`]'s strictness: a
/// malformed or missing `checkpoint` field fails outright rather than
/// being silently skipped.
fn parse_checkpoint_document(doc: &Value) -> Result<WorkflowCheckpoint> {
    let raw = doc
        .get("checkpoint")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::Serialization("Cosmos document is missing a string 'checkpoint' field".into())
        })?;
    serde_json::from_str(raw).map_err(Error::from)
}

/// Cosmos DB (NoSQL API)-backed [`CheckpointStorage`]: every checkpoint is
/// its own document, partitioned by its own `id` (see the module
/// docs). Sibling of [`crate::CosmosChatMessageStore`], reusing the
/// same `CosmosRestClient` HTTP/auth plumbing.
///
/// ```no_run
/// use agent_framework_cosmos::CosmosCheckpointStorage;
/// use agent_framework_core::workflow::CheckpointStorage;
///
/// # async fn demo(checkpoint: agent_framework_core::workflow::WorkflowCheckpoint) -> agent_framework_core::error::Result<()> {
/// let storage = CosmosCheckpointStorage::new(
///     "https://my-account.documents.azure.com:443/",
///     "<base64 master key>",
///     "agent-framework",
///     "workflow-checkpoints",
/// )?;
/// storage.ensure_created().await?;
///
/// let id = storage.save(checkpoint).await?;
/// let loaded = storage.load(&id).await?;
/// println!("loaded: {}", loaded.is_some());
/// # Ok(())
/// # }
/// ```
pub struct CosmosCheckpointStorage {
    client: CosmosRestClient,
    database_id: String,
    container_id: String,
}

impl CosmosCheckpointStorage {
    /// Create checkpoint storage for the given Cosmos DB account/database/
    /// container. No network I/O happens here beyond decoding `key`, which
    /// must be valid base64 (the account's master key) or this returns an
    /// error. Call [`Self::ensure_created`] before first use if the
    /// database/container might not already exist.
    pub fn new(
        account_endpoint: impl Into<String>,
        key: impl Into<String>,
        database_id: impl Into<String>,
        container_id: impl Into<String>,
    ) -> Result<Self> {
        let client = CosmosRestClient::new(account_endpoint, key)?;
        Ok(Self {
            client,
            database_id: database_id.into(),
            container_id: container_id.into(),
        })
    }

    /// The configured database id.
    pub fn database_id(&self) -> &str {
        &self.database_id
    }

    /// The configured container id.
    pub fn container_id(&self) -> &str {
        &self.container_id
    }

    /// Create the database and container if they don't already exist yet
    /// (`409 Conflict` from either is treated as success). The container is
    /// created with partition key [`DEFAULT_PARTITION_KEY_PATH`] (`/id`).
    /// Mirrors [`crate::CosmosChatMessageStore::ensure_created`].
    pub async fn ensure_created(&self) -> Result<()> {
        self.client
            .create_database_if_not_exists(&self.database_id)
            .await?;
        self.client
            .create_container_if_not_exists(
                &self.database_id,
                &self.container_id,
                DEFAULT_PARTITION_KEY_PATH,
            )
            .await?;
        Ok(())
    }
}

#[async_trait]
impl CheckpointStorage for CosmosCheckpointStorage {
    async fn save(&self, checkpoint: WorkflowCheckpoint) -> Result<String> {
        let id = checkpoint.checkpoint_id.clone();
        let document = build_checkpoint_document(&checkpoint)?;
        self.client
            .upsert_document(&self.database_id, &self.container_id, &id, &document)
            .await?;
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        let doc = self
            .client
            .get_document(
                &self.database_id,
                &self.container_id,
                checkpoint_id,
                checkpoint_id,
            )
            .await?;
        doc.as_ref().map(parse_checkpoint_document).transpose()
    }

    async fn list(&self, workflow_id: Option<&str>) -> Result<Vec<WorkflowCheckpoint>> {
        let docs = match workflow_id {
            Some(workflow_id) => {
                self.client
                    .query_documents_cross_partition(
                        &self.database_id,
                        &self.container_id,
                        "SELECT * FROM c WHERE c.workflowId = @workflowId",
                        &[("@workflowId", Value::String(workflow_id.to_string()))],
                    )
                    .await?
            }
            None => {
                self.client
                    .query_documents_cross_partition(
                        &self.database_id,
                        &self.container_id,
                        "SELECT * FROM c",
                        &[],
                    )
                    .await?
            }
        };
        let mut out: Vec<WorkflowCheckpoint> = docs
            .iter()
            .map(parse_checkpoint_document)
            .collect::<Result<_>>()?;
        out.sort_by_key(|cp| cp.timestamp_millis);
        Ok(out)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        // Cosmos DELETE doesn't report whether the document existed; point
        // GET-then-DELETE to answer that, at the cost of not being atomic
        // as a unit (see the [module docs](self) divergence note).
        let existed = self
            .client
            .get_document(
                &self.database_id,
                &self.container_id,
                checkpoint_id,
                checkpoint_id,
            )
            .await?
            .is_some();
        if existed {
            self.client
                .delete_document(
                    &self.database_id,
                    &self.container_id,
                    checkpoint_id,
                    checkpoint_id,
                )
                .await?;
        }
        Ok(existed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::workflow::WorkflowCheckpoint;
    use std::collections::HashMap;

    const TEST_KEY: &str =
        "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==";

    fn storage() -> CosmosCheckpointStorage {
        CosmosCheckpointStorage::new(
            "https://acct.documents.azure.com",
            TEST_KEY,
            "agent-framework",
            "workflow-checkpoints",
        )
        .expect("valid config")
    }

    fn sample_checkpoint(checkpoint_id: &str, workflow_id: &str) -> WorkflowCheckpoint {
        WorkflowCheckpoint {
            checkpoint_id: checkpoint_id.to_string(),
            workflow_id: workflow_id.to_string(),
            workflow_name: Some("my-workflow".to_string()),
            timestamp_millis: 1_700_000_000_000,
            iteration_count: 3,
            messages: Vec::new(),
            executor_states: {
                let mut m = HashMap::new();
                m.insert("executor-a".to_string(), serde_json::json!({"count": 2}));
                m
            },
            shared_state: {
                let mut m = HashMap::new();
                m.insert("k".to_string(), serde_json::json!("v"));
                m
            },
            pending_requests: Vec::new(),
            fanin_state: HashMap::new(),
            metadata: HashMap::new(),
            graph_signature: "sig-123".to_string(),
            version: "1.0".to_string(),
        }
    }

    // region: construction

    #[test]
    fn database_and_container_ids_are_preserved() {
        let s = storage();
        assert_eq!(s.database_id(), "agent-framework");
        assert_eq!(s.container_id(), "workflow-checkpoints");
    }

    #[test]
    fn invalid_master_key_is_rejected() {
        let result = CosmosCheckpointStorage::new(
            "https://acct.documents.azure.com",
            "not-valid-base64!!!",
            "db",
            "coll",
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_account_endpoint_is_rejected() {
        assert!(CosmosCheckpointStorage::new("", TEST_KEY, "db", "coll").is_err());
    }

    // endregion

    // region: build_checkpoint_document / parse_checkpoint_document round trip (no server required)

    #[test]
    fn build_checkpoint_document_shape() {
        let checkpoint = sample_checkpoint("cp-1", "wf-1");
        let doc = build_checkpoint_document(&checkpoint).unwrap();
        assert_eq!(doc["id"], serde_json::json!("cp-1"));
        assert_eq!(doc["workflowId"], serde_json::json!("wf-1"));
        // `checkpoint` is a JSON *string* (double-encoded), not a nested object.
        assert!(doc["checkpoint"].is_string());
        let inner: WorkflowCheckpoint =
            serde_json::from_str(doc["checkpoint"].as_str().unwrap()).unwrap();
        assert_eq!(inner.checkpoint_id, "cp-1");
        assert_eq!(inner.iteration_count, 3);
    }

    #[test]
    fn parse_checkpoint_document_round_trips_full_checkpoint() {
        let checkpoint = sample_checkpoint("cp-2", "wf-2");
        let doc = build_checkpoint_document(&checkpoint).unwrap();
        let parsed = parse_checkpoint_document(&doc).unwrap();
        assert_eq!(parsed.checkpoint_id, checkpoint.checkpoint_id);
        assert_eq!(parsed.workflow_id, checkpoint.workflow_id);
        assert_eq!(parsed.workflow_name, checkpoint.workflow_name);
        assert_eq!(parsed.iteration_count, checkpoint.iteration_count);
        assert_eq!(parsed.executor_states, checkpoint.executor_states);
        assert_eq!(parsed.shared_state, checkpoint.shared_state);
        assert_eq!(parsed.graph_signature, checkpoint.graph_signature);
        assert_eq!(parsed.version, checkpoint.version);
    }

    #[test]
    fn parse_checkpoint_document_requires_string_checkpoint_field() {
        let doc = serde_json::json!({"id": "cp-1", "workflowId": "wf-1"});
        let err = parse_checkpoint_document(&doc).unwrap_err();
        assert!(err.to_string().contains("checkpoint"));
    }

    #[test]
    fn parse_checkpoint_document_rejects_malformed_checkpoint_json() {
        let doc = serde_json::json!({"id": "cp-1", "workflowId": "wf-1", "checkpoint": "not json"});
        assert!(parse_checkpoint_document(&doc).is_err());
    }

    // endregion

    // region: id / partition key convention (pure)

    #[test]
    fn document_id_is_checkpoint_id_not_workflow_id() {
        // save()/load()/delete() address documents by checkpoint_id alone
        // (see module docs), so the document `id` — and this store's
        // partition key value — must be the checkpoint id, not the
        // workflow id.
        let checkpoint = sample_checkpoint("the-checkpoint-id", "some-other-workflow-id");
        let doc = build_checkpoint_document(&checkpoint).unwrap();
        assert_eq!(doc["id"], serde_json::json!("the-checkpoint-id"));
        assert_ne!(doc["id"], doc["workflowId"]);
    }

    #[test]
    fn default_partition_key_path_is_id() {
        assert_eq!(DEFAULT_PARTITION_KEY_PATH, "/id");
    }

    // endregion
}
