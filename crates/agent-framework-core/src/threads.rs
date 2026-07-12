//! Conversation threads and message stores.
//!
//! Rust equivalent of `agent_framework._threads`. A thread is *either*
//! service-managed (identified by a conversation id, history stored remotely)
//! or local (history kept in a [`ChatMessageStore`]) — never both.

use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::{Error, Result};
use crate::memory::AggregateContextProvider;
use crate::types::ChatMessage;

/// The `type` discriminator Python's `AgentThreadState` serializes with; kept
/// identical so a Rust-serialized thread is readable by the Python/.NET
/// stores (`_threads.py:155`, via `SerializationMixin`).
pub(crate) const AGENT_THREAD_STATE_TYPE: &str = "agent_thread_state";
/// The `type` discriminator Python's `ChatMessageStoreState` serializes with
/// (`_threads.py:120`).
pub(crate) const CHAT_MESSAGE_STORE_STATE_TYPE: &str = "chat_message_store_state";

/// Storage abstraction for a conversation's message history.
#[async_trait]
pub trait ChatMessageStore: Send + Sync {
    /// Return the stored messages in ascending chronological order.
    async fn list_messages(&self) -> Result<Vec<ChatMessage>>;

    /// Append messages to the store.
    async fn add_messages(&self, messages: Vec<ChatMessage>) -> Result<()>;

    /// Serialize the store's state.
    ///
    /// The default shape mirrors Python's `ChatMessageStoreState.to_dict()`
    /// (`{"type":"chat_message_store_state","messages":[...]}`), so a thread
    /// serialized by [`AgentThread::serialize`] round-trips through Python's
    /// `ChatMessageStore.deserialize`. Stores that persist history externally
    /// (e.g. Redis) override this to serialize a pointer instead.
    async fn serialize(&self) -> Result<Value> {
        let msgs = self.list_messages().await?;
        Ok(serde_json::json!({
            "type": CHAT_MESSAGE_STORE_STATE_TYPE,
            "messages": msgs,
        }))
    }

    /// Restore the store's messages from previously serialized state, mirroring
    /// Python's `ChatMessageStore.update_from_state` (`_threads.py:262-275`).
    ///
    /// The default is a no-op so that stores which persist history externally
    /// (and therefore serialize only a pointer, not the messages) are not
    /// forced to implement it; [`InMemoryChatMessageStore`] overrides it.
    async fn update_from_state(&self, _serialized_store_state: &Value) -> Result<()> {
        Ok(())
    }
}

/// Default in-memory [`ChatMessageStore`].
#[derive(Default, Clone)]
pub struct InMemoryChatMessageStore {
    messages: Arc<Mutex<Vec<ChatMessage>>>,
}

impl InMemoryChatMessageStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_messages(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages: Arc::new(Mutex::new(messages)),
        }
    }
}

#[async_trait]
impl ChatMessageStore for InMemoryChatMessageStore {
    async fn list_messages(&self) -> Result<Vec<ChatMessage>> {
        Ok(self.messages.lock().await.clone())
    }
    async fn add_messages(&self, messages: Vec<ChatMessage>) -> Result<()> {
        self.messages.lock().await.extend(messages);
        Ok(())
    }
    async fn update_from_state(&self, serialized_store_state: &Value) -> Result<()> {
        // Replace the in-memory history with the serialized messages whenever
        // the state carries a `messages` list — INCLUDING an empty one, so
        // restoring an intentionally empty thread clears stale history from a
        // reused store. (Deliberate divergence: Python *appends* the restored
        // messages onto an existing store, `_threads.py:499-505`, so it can
        // neither clear nor deduplicate; replace is the honest semantic for a
        // restore operation.) A state without a `messages` key restores
        // nothing.
        let Some(raw) = serialized_store_state.get("messages") else {
            return Ok(());
        };
        if raw.is_null() {
            return Ok(());
        }
        let messages: Vec<ChatMessage> = serde_json::from_value(raw.clone()).map_err(|e| {
            Error::Serialization(format!("failed to restore chat message store: {e}"))
        })?;
        *self.messages.lock().await = messages;
        Ok(())
    }
}

/// A conversation thread, either service-managed or backed by a local store.
#[derive(Clone, Default)]
pub struct AgentThread {
    service_thread_id: Option<String>,
    message_store: Option<Arc<dyn ChatMessageStore>>,
    /// Context providers associated with this thread (memory/RAG injection).
    pub context_provider: Option<Arc<AggregateContextProvider>>,
}

impl AgentThread {
    /// A fresh, uninitialized thread.
    pub fn new() -> Self {
        Self::default()
    }

    /// A service-managed thread identified by a conversation id.
    pub fn service(id: impl Into<String>) -> Self {
        Self {
            service_thread_id: Some(id.into()),
            message_store: None,
            context_provider: None,
        }
    }

    /// A local thread backed by the given message store.
    pub fn local(store: Arc<dyn ChatMessageStore>) -> Self {
        Self {
            service_thread_id: None,
            message_store: Some(store),
            context_provider: None,
        }
    }

    /// Attach context providers to this thread.
    pub fn with_context_provider(mut self, provider: Arc<AggregateContextProvider>) -> Self {
        self.context_provider = Some(provider);
        self
    }

    /// Whether the thread has been initialized (has an id or a store).
    pub fn is_initialized(&self) -> bool {
        self.service_thread_id.is_some() || self.message_store.is_some()
    }

    /// The service-side conversation id, if any.
    pub fn service_thread_id(&self) -> Option<&str> {
        self.service_thread_id.as_deref()
    }

    /// Set the service-side conversation id. Fails if a local store is set.
    pub fn set_service_thread_id(&mut self, id: impl Into<String>) -> Result<()> {
        if self.message_store.is_some() {
            return Err(Error::other(
                "cannot set service_thread_id on a thread with a local message store",
            ));
        }
        self.service_thread_id = Some(id.into());
        Ok(())
    }

    /// Adopt a service-managed conversation id returned by the chat service
    /// (e.g. an OpenAI Responses `previous_response_id` or an Azure AI thread
    /// id), so follow-up runs continue the same service conversation.
    ///
    /// Succeeds when the thread has no local store or its local store is
    /// still empty (the unused store is discarded — the service owns the
    /// history from here on). A thread that already accumulated local
    /// history is left unchanged and `Ok(false)` is returned.
    pub async fn try_adopt_service_thread_id(&mut self, id: &str) -> Result<bool> {
        if self.service_thread_id.as_deref() == Some(id) {
            return Ok(false);
        }
        match &self.message_store {
            None => {
                self.service_thread_id = Some(id.to_string());
                Ok(true)
            }
            Some(store) => {
                if store.list_messages().await?.is_empty() {
                    self.message_store = None;
                    self.service_thread_id = Some(id.to_string());
                    Ok(true)
                } else {
                    tracing::debug!(
                        "not adopting service thread id {id}: thread has local history"
                    );
                    Ok(false)
                }
            }
        }
    }

    /// The local message store, if any.
    pub fn message_store(&self) -> Option<&Arc<dyn ChatMessageStore>> {
        self.message_store.as_ref()
    }

    /// Ensure a local message store exists (creating an in-memory one), then
    /// return it. Fails if the thread is service-managed.
    pub fn ensure_local_store(&mut self) -> Result<Arc<dyn ChatMessageStore>> {
        if self.service_thread_id.is_some() {
            return Err(Error::other(
                "cannot add a local store to a service-managed thread",
            ));
        }
        if self.message_store.is_none() {
            self.message_store = Some(Arc::new(InMemoryChatMessageStore::new()));
        }
        Ok(self.message_store.clone().unwrap())
    }

    /// Notify the thread of new messages. For service-managed threads this is a
    /// no-op (the service tracks history); for local threads the messages are
    /// appended to the store.
    pub async fn on_new_messages(&mut self, messages: Vec<ChatMessage>) -> Result<()> {
        if self.service_thread_id.is_some() {
            return Ok(());
        }
        let store = self.ensure_local_store()?;
        store.add_messages(messages).await
    }

    /// Return the current history for seeding a request (empty for
    /// service-managed threads).
    pub async fn list_messages(&self) -> Result<Vec<ChatMessage>> {
        match &self.message_store {
            Some(store) => store.list_messages().await,
            None => Ok(Vec::new()),
        }
    }

    /// Construct a thread from an explicit `service_thread_id` XOR a
    /// `message_store`, validating that both are not set at once.
    ///
    /// Mirrors the invariant enforced by Python's `AgentThread.__init__`
    /// (`_threads.py:342-343`): a thread is *either* service-managed or
    /// locally backed, never both.
    pub fn try_from_parts(
        service_thread_id: Option<String>,
        message_store: Option<Arc<dyn ChatMessageStore>>,
    ) -> Result<Self> {
        if service_thread_id.is_some() && message_store.is_some() {
            return Err(Error::other(
                "Only the service_thread_id or message_store may be set, but not both.",
            ));
        }
        Ok(Self {
            service_thread_id,
            message_store,
            context_provider: None,
        })
    }

    /// Serialize the thread's state to a JSON value.
    ///
    /// The wire shape matches Python's `AgentThread.serialize()`
    /// (`_threads.py:421-434`, via `AgentThreadState.to_dict(exclude_none=False)`):
    /// a `type`-tagged object carrying `service_thread_id` **xor**
    /// `chat_message_store_state` (the other key present as `null`), so the
    /// blob can be round-tripped by the Python/.NET stores. The message-store
    /// state is whatever the backing [`ChatMessageStore::serialize`] produces.
    pub async fn serialize(&self) -> Result<Value> {
        let chat_message_store_state = match &self.message_store {
            Some(store) => store.serialize().await?,
            None => Value::Null,
        };
        Ok(serde_json::json!({
            "type": AGENT_THREAD_STATE_TYPE,
            "service_thread_id": self.service_thread_id,
            "chat_message_store_state": chat_message_store_state,
        }))
    }

    /// Reconstruct a thread from serialized state produced by
    /// [`AgentThread::serialize`], mirroring Python's `AgentThread.deserialize`
    /// (`_threads.py:436-476`).
    ///
    /// A `service_thread_id` yields a service-managed thread. Otherwise, when
    /// `chat_message_store_state` is present, the provided `message_store`
    /// (or a fresh [`InMemoryChatMessageStore`]) is populated from it via
    /// [`ChatMessageStore::update_from_state`]. A state carrying neither yields
    /// an uninitialized thread.
    pub async fn deserialize(
        serialized_thread_state: &Value,
        message_store: Option<Arc<dyn ChatMessageStore>>,
    ) -> Result<Self> {
        let service_thread_id = serialized_thread_state
            .get("service_thread_id")
            .and_then(Value::as_str)
            .map(str::to_string);
        let store_state = serialized_thread_state
            .get("chat_message_store_state")
            .filter(|v| !v.is_null());

        if service_thread_id.is_some() && store_state.is_some() {
            return Err(Error::other(
                "A thread cannot have both a service_thread_id and a chat_message_store.",
            ));
        }
        if let Some(id) = service_thread_id {
            return Ok(Self::service(id));
        }
        let Some(store_state) = store_state else {
            return Ok(Self::new());
        };
        let store = message_store.unwrap_or_else(|| Arc::new(InMemoryChatMessageStore::new()));
        store.update_from_state(store_state).await?;
        Ok(Self::local(store))
    }

    /// Restore this thread's state in place from serialized state, mirroring
    /// Python's `AgentThread.update_from_thread_state` (`_threads.py:478-505`).
    ///
    /// If the state carries a `service_thread_id`, it is adopted (subject to
    /// the service-xor-store invariant). Otherwise a `chat_message_store_state`
    /// is loaded into the existing store, or into a freshly created in-memory
    /// store when the thread has none.
    pub async fn update_from_state(&mut self, serialized_thread_state: &Value) -> Result<()> {
        if let Some(id) = serialized_thread_state
            .get("service_thread_id")
            .and_then(Value::as_str)
        {
            self.set_service_thread_id(id)?;
            return Ok(());
        }
        let Some(store_state) = serialized_thread_state
            .get("chat_message_store_state")
            .filter(|v| !v.is_null())
        else {
            return Ok(());
        };
        match &self.message_store {
            Some(store) => store.update_from_state(store_state).await?,
            None => {
                let store = Arc::new(InMemoryChatMessageStore::new());
                store.update_from_state(store_state).await?;
                self.message_store = Some(store);
            }
        }
        // The restored state is local (its `service_thread_id` was absent or
        // null), so this thread becomes local: keeping a previous service id
        // alongside a store would break the service-xor-store invariant —
        // `serialize()` would emit both fields and `on_new_messages()` would
        // treat the thread as service-managed and stop recording history.
        // (Deliberate divergence: Python leaves the stale id in place here,
        // `_threads.py:493-505`, inheriting exactly that broken state.)
        self.service_thread_id = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ChatMessage;

    #[tokio::test]
    async fn restoring_an_empty_state_clears_stale_history() {
        // A reused store with prior messages must end up empty when the
        // restored thread state carries an (intentionally) empty list.
        let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
            ChatMessage::user("stale"),
        ]));
        store
            .update_from_state(&serde_json::json!({ "messages": [] }))
            .await
            .unwrap();
        assert!(store.list_messages().await.unwrap().is_empty());

        // A state with no `messages` key restores nothing (unchanged).
        let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
            ChatMessage::user("kept"),
        ]));
        store
            .update_from_state(&serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(store.list_messages().await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn restoring_local_state_onto_a_service_thread_clears_the_service_id() {
        let mut thread = AgentThread::service("svc-9");
        let state = serde_json::json!({
            "type": "agent_thread_state",
            "service_thread_id": null,
            "chat_message_store_state": {
                "type": "chat_message_store_state",
                "messages": [ChatMessage::user("hello")],
            },
        });
        thread.update_from_state(&state).await.unwrap();
        assert!(thread.service_thread_id().is_none());

        // The thread now behaves as local end to end: history records and a
        // re-serialize emits the local shape (store state set, id null).
        thread
            .on_new_messages(vec![ChatMessage::user("again")])
            .await
            .unwrap();
        let reserialized = thread.serialize().await.unwrap();
        assert!(reserialized["service_thread_id"].is_null());
        let messages = reserialized["chat_message_store_state"]["messages"]
            .as_array()
            .unwrap();
        assert_eq!(messages.len(), 2);
    }

    #[tokio::test]
    async fn serialize_wire_shape_matches_python_keys() {
        // Local thread with history -> `chat_message_store_state` populated,
        // `service_thread_id` present as null. Keys match Python's
        // AgentThreadState.to_dict(exclude_none=False).
        let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
            ChatMessage::user("hello"),
        ]));
        let thread = AgentThread::local(store);
        let state = thread.serialize().await.unwrap();
        assert_eq!(state["type"], "agent_thread_state");
        assert!(state.get("service_thread_id").unwrap().is_null());
        let store_state = &state["chat_message_store_state"];
        assert_eq!(store_state["type"], "chat_message_store_state");
        assert!(store_state["messages"].is_array());

        // Service thread -> `service_thread_id` set, store state null.
        let svc = AgentThread::service("thread_abc123");
        let svc_state = svc.serialize().await.unwrap();
        assert_eq!(svc_state["service_thread_id"], "thread_abc123");
        assert!(svc_state.get("chat_message_store_state").unwrap().is_null());
    }

    #[tokio::test]
    async fn roundtrip_local_thread_with_messages() {
        let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
            ChatMessage::user("q1"),
            ChatMessage::assistant("a1"),
        ]));
        let thread = AgentThread::local(store);
        let state = thread.serialize().await.unwrap();

        let restored = AgentThread::deserialize(&state, None).await.unwrap();
        let msgs = restored.list_messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text(), "q1");
        assert_eq!(msgs[1].text(), "a1");
        assert!(restored.service_thread_id().is_none());
    }

    #[tokio::test]
    async fn roundtrip_service_thread() {
        let thread = AgentThread::service("thread_xyz");
        let state = thread.serialize().await.unwrap();
        let restored = AgentThread::deserialize(&state, None).await.unwrap();
        assert_eq!(restored.service_thread_id(), Some("thread_xyz"));
        assert!(restored.message_store().is_none());
    }

    #[tokio::test]
    async fn deserialize_uses_provided_store() {
        let store = Arc::new(InMemoryChatMessageStore::with_messages(vec![
            ChatMessage::user("seed"),
        ]));
        let state = AgentThread::local(store).serialize().await.unwrap();

        // A provided (distinct) store instance should be the one populated.
        let target = Arc::new(InMemoryChatMessageStore::new());
        let restored = AgentThread::deserialize(&state, Some(target.clone()))
            .await
            .unwrap();
        assert_eq!(restored.list_messages().await.unwrap().len(), 1);
        // The passed-in store instance holds the messages.
        assert_eq!(target.list_messages().await.unwrap()[0].text(), "seed");
    }

    #[tokio::test]
    async fn store_update_from_state_replaces_messages() {
        let store = InMemoryChatMessageStore::with_messages(vec![ChatMessage::user("old")]);
        let state = serde_json::json!({
            "type": "chat_message_store_state",
            "messages": [ChatMessage::user("new")],
        });
        store.update_from_state(&state).await.unwrap();
        let msgs = store.list_messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text(), "new");
    }

    #[test]
    fn try_from_parts_rejects_service_id_plus_store() {
        // `AgentThread` is not `Debug`, so match rather than `unwrap_err`.
        let store: Arc<dyn ChatMessageStore> = Arc::new(InMemoryChatMessageStore::new());
        match AgentThread::try_from_parts(Some("id".into()), Some(store)) {
            Err(e) => assert!(e.to_string().contains("not both")),
            Ok(_) => panic!("expected a validation error"),
        }
    }

    #[tokio::test]
    async fn deserialize_rejects_service_id_plus_store_state() {
        // A malformed blob carrying both keys populated must be rejected.
        let bad = serde_json::json!({
            "type": "agent_thread_state",
            "service_thread_id": "svc",
            "chat_message_store_state": {"type": "chat_message_store_state", "messages": []},
        });
        match AgentThread::deserialize(&bad, None).await {
            Err(e) => assert!(e.to_string().contains("cannot have both")),
            Ok(_) => panic!("expected a validation error"),
        }
    }
}
