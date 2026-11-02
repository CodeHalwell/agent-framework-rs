//! Hosted state over Foundry state stores: agent sessions, workflow
//! checkpoints and function-approval requests (upstream `_state_store.py`).
//!
//! Store names follow upstream: `agent_sessions`, `checkpoints/<context>`
//! and `function_approvals` locally; hosted, each is suffixed
//! `/v2/<scope storage key>` (and a hashed context), so a store name never
//! reveals the user or session id.

use std::sync::Arc;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::types::Content;
use agent_framework_core::workflow::{compare_checkpoints, CheckpointStorage, WorkflowCheckpoint};
use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::config::FoundryHostConfig;
use crate::scope::{store_name, FoundryRequestContext, FoundryRequestScope};
use crate::storage::{FoundryStateStore, FoundryStorage, StorageError, StoreOptions};

/// The trusted scope stores are keyed by: `Some` only when hosted.
pub fn store_scope(
    config: &FoundryHostConfig,
    context: &FoundryRequestContext,
) -> Result<Option<FoundryRequestScope>> {
    if !config.is_hosted {
        return Ok(None);
    }
    FoundryRequestScope::from_context(config, context, None).map(Some)
}

pub(crate) async fn open(
    storage: &FoundryStorage,
    name: &str,
    context: &FoundryRequestContext,
) -> Result<FoundryStateStore> {
    Ok(storage
        .get_or_create(
            name,
            StoreOptions {
                user_id: context.user_id.clone(),
                call_id: context.call_id.clone(),
                ..Default::default()
            },
        )
        .await?)
}

pub(crate) fn object(value: Value, what: &str) -> Result<Map<String, Value>> {
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(Error::Serialization(format!(
            "{what} did not serialize to a JSON object"
        ))),
    }
}

/// Workflow checkpoints in a Foundry state store scoped to one context
/// (a conversation or invocation session).
pub struct FoundryCheckpointStore {
    storage: FoundryStorage,
    name: String,
    context: FoundryRequestContext,
}

impl std::fmt::Debug for FoundryCheckpointStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryCheckpointStore")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FoundryCheckpointStore {
    /// The default store root.
    pub const DEFAULT_ROOT: &'static str = "checkpoints";

    /// A store for `context_id`'s checkpoints.
    pub fn new(
        storage: FoundryStorage,
        context: FoundryRequestContext,
        context_id: &str,
        scope: Option<&FoundryRequestScope>,
    ) -> Result<Self> {
        if context_id.is_empty() {
            return Err(Error::Configuration(
                "context_id must be provided to initialize a FoundryCheckpointStore.".into(),
            ));
        }
        Ok(Self {
            storage,
            name: store_name(Self::DEFAULT_ROOT, scope, Some(context_id)),
            context,
        })
    }

    /// The state-store name.
    pub fn store_name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl CheckpointStorage for FoundryCheckpointStore {
    async fn save(&self, checkpoint: WorkflowCheckpoint) -> Result<String> {
        let id = checkpoint.checkpoint_id.clone();
        let value = object(serde_json::to_value(&checkpoint)?, "checkpoint")?;
        let store = open(&self.storage, &self.name, &self.context).await?;
        store.set_item(&id, value, None).await?;
        Ok(id)
    }

    async fn load(&self, checkpoint_id: &str) -> Result<Option<WorkflowCheckpoint>> {
        let store = open(&self.storage, &self.name, &self.context).await?;
        match store.get_item(checkpoint_id).await? {
            None => Ok(None),
            Some(item) => Ok(Some(serde_json::from_value(Value::Object(item.value))?)),
        }
    }

    async fn list(&self, workflow_id: Option<&str>) -> Result<Vec<WorkflowCheckpoint>> {
        let store = open(&self.storage, &self.name, &self.context).await?;
        let mut out = Vec::new();
        for key in store.list_keys().await? {
            if let Some(item) = store.get_item(&key).await? {
                let checkpoint: WorkflowCheckpoint =
                    serde_json::from_value(Value::Object(item.value))?;
                if workflow_id.is_none_or(|w| checkpoint.workflow_id == w) {
                    out.push(checkpoint);
                }
            }
        }
        out.sort_by(compare_checkpoints);
        Ok(out)
    }

    async fn delete(&self, checkpoint_id: &str) -> Result<bool> {
        let store = open(&self.storage, &self.name, &self.context).await?;
        match store.delete_item(checkpoint_id, None).await {
            Ok(()) => Ok(true),
            Err(StorageError::NotFound(_)) => Ok(false),
            Err(e) => Err(e.into()),
        }
    }
}

/// Stores pending function-approval requests between turns.
#[async_trait]
pub trait FunctionApprovalStore: Send + Sync {
    /// Save a request under `id`; an existing id is an error.
    async fn save_approval_request(&self, id: &str, request: &Content) -> Result<()>;
    /// Load the request saved under `id`.
    async fn load_approval_request(&self, id: &str) -> Result<Content>;
}

/// A process-local [`FunctionApprovalStore`] (the Responses server's default
/// when no [`FunctionApprovalStoreProvider`] is configured).
#[derive(Debug, Default)]
pub struct InMemoryFunctionApprovalStore {
    requests: std::sync::Mutex<std::collections::HashMap<String, Content>>,
}

#[async_trait]
impl FunctionApprovalStore for InMemoryFunctionApprovalStore {
    async fn save_approval_request(&self, id: &str, request: &Content) -> Result<()> {
        let mut requests = self.requests.lock().unwrap();
        if requests.contains_key(id) {
            return Err(agent_framework_core::error::Error::Other(format!(
                "Approval request with ID '{id}' already exists."
            )));
        }
        requests.insert(id.to_string(), request.clone());
        Ok(())
    }

    async fn load_approval_request(&self, id: &str) -> Result<Content> {
        self.requests
            .lock()
            .unwrap()
            .get(id)
            .cloned()
            .ok_or_else(|| {
                agent_framework_core::error::Error::Other(format!(
                    "Approval request with ID '{id}' does not exist."
                ))
            })
    }
}

/// Function-approval requests in a Foundry state store.
pub struct FoundryFunctionApprovalStore {
    storage: FoundryStorage,
    name: String,
    context: FoundryRequestContext,
}

impl std::fmt::Debug for FoundryFunctionApprovalStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryFunctionApprovalStore")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FoundryFunctionApprovalStore {
    /// The default store root.
    pub const DEFAULT_ROOT: &'static str = "function_approvals";

    /// A store for one request.
    pub fn new(
        storage: FoundryStorage,
        context: FoundryRequestContext,
        scope: Option<&FoundryRequestScope>,
    ) -> Self {
        Self {
            storage,
            name: store_name(Self::DEFAULT_ROOT, scope, None),
            context,
        }
    }
}

#[async_trait]
impl FunctionApprovalStore for FoundryFunctionApprovalStore {
    async fn save_approval_request(&self, id: &str, request: &Content) -> Result<()> {
        let value = object(serde_json::to_value(request)?, "approval request")?;
        let store = open(&self.storage, &self.name, &self.context).await?;
        match store.create_item(id, value).await {
            Ok(_) => Ok(()),
            Err(StorageError::Conflict(_)) => Err(Error::Configuration(format!(
                "Approval request with ID '{id}' already exists."
            ))),
            Err(e) => Err(e.into()),
        }
    }

    async fn load_approval_request(&self, id: &str) -> Result<Content> {
        let store = open(&self.storage, &self.name, &self.context).await?;
        match store.get_item(id).await? {
            Some(item) => Ok(serde_json::from_value(Value::Object(item.value))?),
            None => Err(Error::Configuration(format!(
                "Approval request with ID '{id}' does not exist."
            ))),
        }
    }
}

/// Supplies function-approval stores per request.
pub trait FunctionApprovalStoreProvider: Send + Sync {
    /// The store for a request with `context`.
    fn approval_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
    ) -> Result<Arc<dyn FunctionApprovalStore>>;
}

impl FunctionApprovalStoreProvider for FoundryStorage {
    fn approval_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
    ) -> Result<Arc<dyn FunctionApprovalStore>> {
        let scope = store_scope(config, context)?;
        Ok(Arc::new(FoundryFunctionApprovalStore::new(
            self.clone(),
            context.clone(),
            scope.as_ref(),
        )))
    }
}

/// Supplies checkpoint stores per request and context (upstream
/// `ContextScopedStoreProvider`).
pub trait CheckpointStoreProvider: Send + Sync {
    /// The store for `context_id` in a request with `context`.
    fn checkpoint_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
        context_id: &str,
    ) -> Result<Arc<dyn CheckpointStorage>>;
}

impl CheckpointStoreProvider for FoundryStorage {
    fn checkpoint_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
        context_id: &str,
    ) -> Result<Arc<dyn CheckpointStorage>> {
        let scope = store_scope(config, context)?;
        Ok(Arc::new(FoundryCheckpointStore::new(
            self.clone(),
            context.clone(),
            context_id,
            scope.as_ref(),
        )?))
    }
}
