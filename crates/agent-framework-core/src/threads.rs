//! Conversation threads and message stores.
//!
//! Rust equivalent of `agent_framework._threads`. A thread is *either*
//! service-managed (identified by a conversation id, history stored remotely)
//! or local (history kept in a [`ChatMessageStore`]) — never both.

use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::error::{Error, Result};
use crate::memory::AggregateContextProvider;
use crate::types::ChatMessage;

/// Storage abstraction for a conversation's message history.
#[async_trait]
pub trait ChatMessageStore: Send + Sync {
    /// Return the stored messages in ascending chronological order.
    async fn list_messages(&self) -> Result<Vec<ChatMessage>>;

    /// Append messages to the store.
    async fn add_messages(&self, messages: Vec<ChatMessage>) -> Result<()>;

    /// Serialize the store's state.
    async fn serialize(&self) -> Result<serde_json::Value> {
        let msgs = self.list_messages().await?;
        Ok(serde_json::json!({ "messages": msgs }))
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
}

/// A conversation thread, either service-managed or backed by a local store.
#[derive(Clone, Default)]
pub struct AgentThread {
    // Shared across clones (like the message store) so a service id adopted
    // inside a cloned thread — e.g. during `run_stream` — is visible on the
    // caller's original thread.
    service_thread_id: Arc<std::sync::RwLock<Option<String>>>,
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
            service_thread_id: Arc::new(std::sync::RwLock::new(Some(id.into()))),
            message_store: None,
            context_provider: None,
        }
    }

    /// A local thread backed by the given message store.
    pub fn local(store: Arc<dyn ChatMessageStore>) -> Self {
        Self {
            service_thread_id: Arc::new(std::sync::RwLock::new(None)),
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
        self.service_thread_id.read().unwrap().is_some() || self.message_store.is_some()
    }

    /// The service-side conversation id, if any.
    pub fn service_thread_id(&self) -> Option<String> {
        self.service_thread_id.read().unwrap().clone()
    }

    /// Set the service-side conversation id. Fails if a local store is set.
    pub fn set_service_thread_id(&mut self, id: impl Into<String>) -> Result<()> {
        if self.message_store.is_some() {
            return Err(Error::other(
                "cannot set service_thread_id on a thread with a local message store",
            ));
        }
        *self.service_thread_id.write().unwrap() = Some(id.into());
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
        if self.service_thread_id.read().unwrap().as_deref() == Some(id) {
            return Ok(false);
        }
        match &self.message_store {
            None => {
                *self.service_thread_id.write().unwrap() = Some(id.to_string());
                Ok(true)
            }
            Some(store) => {
                if store.list_messages().await?.is_empty() {
                    self.message_store = None;
                    *self.service_thread_id.write().unwrap() = Some(id.to_string());
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
        if self.service_thread_id.read().unwrap().is_some() {
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
        if self.service_thread_id.read().unwrap().is_some() {
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
}
