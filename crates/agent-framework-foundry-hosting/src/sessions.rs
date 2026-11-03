//! Agent sessions in Foundry storage (the `session-store` feature, which
//! turns on core's experimental `AgentSessionStore`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework_core::agent::SupportsAgentRun;
use agent_framework_core::error::{Error, Result};
use agent_framework_core::session::AgentSession;
use agent_framework_core::session_store::{
    deserialize_session, serialize_session, AgentSessionStore, AgentSessionStoreKey,
    InMemoryAgentSessionStore,
};
use async_trait::async_trait;
use serde_json::Value;

use crate::config::FoundryHostConfig;
use crate::scope::{store_name, FoundryRequestContext, FoundryRequestScope};
use crate::storage::{FoundryStorage, StorageError};
use crate::stores::{object, open, store_scope};

/// The item key of `agent`'s session under `key`: the JSON array
/// `[agent id, session id, {partitions}]`. Each part is a JSON string (or
/// object of strings), so no id or partition can spell another key, and
/// each agent's sessions are its own (the `AgentSessionStore` contract).
///
/// The Rust host's own format: upstream's `FoundryAgentSessionStore` keys a
/// session by its bare id (it has no agent or partitions), and the Rust host
/// already names its sessions differently (`response-…`), so the two never
/// shared records.
fn session_item_key(agent: &dyn SupportsAgentRun, key: &AgentSessionStoreKey) -> String {
    serde_json::to_string(&(agent.id(), key.session_id(), key.partitions()))
        .expect("strings serialize")
}

/// Agent sessions in a Foundry state store, with optimistic concurrency: a
/// session loaded through this store is written back with `If-Match`, so a
/// concurrent writer's update is never silently overwritten.
pub struct FoundryAgentSessionStore {
    storage: FoundryStorage,
    name: String,
    context: FoundryRequestContext,
    scoped: bool,
    etags: Mutex<HashMap<String, Option<String>>>,
}

impl std::fmt::Debug for FoundryAgentSessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryAgentSessionStore")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FoundryAgentSessionStore {
    /// The default store root.
    pub const DEFAULT_ROOT: &'static str = "agent_sessions";

    /// A store for one request: `root` (default [`Self::DEFAULT_ROOT`]),
    /// scoped when `scope` is set.
    pub fn new(
        storage: FoundryStorage,
        context: FoundryRequestContext,
        root: Option<&str>,
        scope: Option<&FoundryRequestScope>,
    ) -> Self {
        Self {
            storage,
            name: store_name(root.unwrap_or(Self::DEFAULT_ROOT), scope, None),
            context,
            scoped: scope.is_some(),
            etags: Mutex::new(HashMap::new()),
        }
    }

    /// The state-store name.
    pub fn store_name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl AgentSessionStore for FoundryAgentSessionStore {
    async fn save_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
        session: &AgentSession,
    ) -> Result<()> {
        let item = session_item_key(agent, key);
        let value = object(serialize_session(session), "session")?;
        let store = open(&self.storage, &self.name, &self.context).await?;
        let known = self.etags.lock().unwrap().get(&item).cloned();
        let result = match known {
            Some(Some(etag)) => store.set_item(&item, value, Some(&etag)).await,
            // Loaded as absent (or hosted): create, so a racing creator wins.
            Some(None) => store.create_item(&item, value).await,
            None if self.scoped => store.create_item(&item, value).await,
            None => store.set_item(&item, value, None).await,
        };
        let written = result.map_err(|e| match e {
            StorageError::Conflict(_) | StorageError::PreconditionFailed(_) => Error::Service(
                "Another request advanced this agent session; reload before writing.".into(),
            ),
            other => other.into(),
        })?;
        if written.etag.is_empty() {
            return Err(Error::Service(
                "Foundry agent session storage did not return an ETag after saving.".into(),
            ));
        }
        self.etags.lock().unwrap().insert(item, Some(written.etag));
        Ok(())
    }

    async fn get_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<Option<AgentSession>> {
        let item = session_item_key(agent, key);
        let store = open(&self.storage, &self.name, &self.context).await?;
        let Some(found) = store.get_item(&item).await? else {
            self.etags.lock().unwrap().insert(item, None);
            return Ok(None);
        };
        if found.etag.is_empty() {
            return Err(Error::Service(
                "Foundry agent session storage returned a loaded session without an ETag.".into(),
            ));
        }
        self.etags.lock().unwrap().insert(item, Some(found.etag));
        deserialize_session(agent, &Value::Object(found.value)).map(Some)
    }

    async fn delete_session(
        &self,
        agent: &dyn SupportsAgentRun,
        key: &AgentSessionStoreKey,
    ) -> Result<()> {
        let item = session_item_key(agent, key);
        let store = open(&self.storage, &self.name, &self.context).await?;
        match store.delete_item(&item, None).await {
            Ok(()) | Err(StorageError::NotFound(_)) => {}
            Err(e) => return Err(e.into()),
        }
        self.etags.lock().unwrap().remove(&item);
        Ok(())
    }
}

/// Supplies the session store for one request (upstream `StoreProvider`).
pub trait SessionStoreProvider: Send + Sync {
    /// The store for a request with `context`.
    fn session_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
    ) -> Result<Arc<dyn AgentSessionStore>>;
}

/// Sessions in Foundry storage, one store per request scope.
#[derive(Debug, Clone)]
pub struct FoundrySessionStoreProvider {
    storage: FoundryStorage,
    root: Option<String>,
}

impl FoundrySessionStoreProvider {
    /// Sessions under `root` (default `agent_sessions`).
    pub fn new(storage: FoundryStorage, root: Option<String>) -> Self {
        Self { storage, root }
    }
}

impl SessionStoreProvider for FoundrySessionStoreProvider {
    fn session_store(
        &self,
        config: &FoundryHostConfig,
        context: &FoundryRequestContext,
    ) -> Result<Arc<dyn AgentSessionStore>> {
        let scope = store_scope(config, context)?;
        Ok(Arc::new(FoundryAgentSessionStore::new(
            self.storage.clone(),
            context.clone(),
            self.root.as_deref(),
            scope.as_ref(),
        )))
    }
}

/// One process-local store shared by every request (the local default).
#[derive(Clone, Default)]
pub struct InMemorySessionStoreProvider {
    store: InMemoryAgentSessionStore,
}

impl std::fmt::Debug for InMemorySessionStoreProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemorySessionStoreProvider")
            .finish_non_exhaustive()
    }
}

impl SessionStoreProvider for InMemorySessionStoreProvider {
    fn session_store(
        &self,
        _config: &FoundryHostConfig,
        _context: &FoundryRequestContext,
    ) -> Result<Arc<dyn AgentSessionStore>> {
        Ok(Arc::new(self.store.clone()))
    }
}
