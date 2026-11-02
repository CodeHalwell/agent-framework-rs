//! A Valkey-backed [`HistoryProvider`] that resolves its conversation from
//! the session on every run.
//!
//! Port of upstream .NET `Microsoft.Agents.AI.Valkey`
//! (`ValkeyChatHistoryProvider`). Valkey speaks the Redis protocol, so this
//! uses the same client as [`RedisChatMessageStore`](crate::RedisChatMessageStore).
//! The difference is where the conversation comes from: the Redis store is
//! bound to one session id when it is built, while this provider keeps a
//! `{"conversationId": ..}` state object in each session's state bag (under
//! [`ValkeyChatHistoryProvider::state_key`]), so one provider serves every
//! session of an agent.
//!
//! Messages are one `LIST` per conversation at `{key_prefix}:{conversation}`,
//! appended with a single `RPUSH`, trimmed with `LTRIM` when
//! [`max_messages`](ValkeyChatHistoryProvider::with_max_messages) is set, and
//! read with `LRANGE` (only the tail when
//! [`max_messages_to_retrieve`](ValkeyChatHistoryProvider::with_max_messages_to_retrieve)
//! is set). Malformed entries are skipped with a warning.
//!
//! # Divergences
//!
//! - The key's two halves go through
//!   [`storage_key_segment`], as every Redis key in this port does, so a
//!   prefix or conversation id containing `:` cannot address another
//!   conversation's list. Literal-safe ids (UUIDs, the default prefix) are
//!   unchanged, so keys match the .NET provider's in the common case.
//! - Messages are serialized in this port's JSON message shape, which is not
//!   the .NET `ChatMessage` shape: a list is not shared with .NET agents.

use std::sync::Arc;

use async_trait::async_trait;
use redis::AsyncCommands;
use serde_json::{json, Value};

use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::HistoryProvider;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::session::AgentSession;
use agent_framework_core::storage_keys::storage_key_segment;
use agent_framework_core::types::Message;

use crate::internal::{map_redis_err, LazyConnection};

/// The default key prefix, as upstream's.
pub const DEFAULT_VALKEY_KEY_PREFIX: &str = "chat_history";
/// The default session-state key, upstream's type name.
pub const DEFAULT_VALKEY_STATE_KEY: &str = "ValkeyChatHistoryProvider";

/// Picks the conversation id for a session that has none yet.
pub type ConversationIdInitializer = Arc<dyn Fn(&AgentSession) -> String + Send + Sync>;
/// Filters a list of messages.
pub type MessageFilter = Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>;

/// A Valkey-backed history provider. See the [module docs](self).
#[derive(Clone)]
pub struct ValkeyChatHistoryProvider {
    conn: Arc<LazyConnection>,
    key_prefix: String,
    state_key: String,
    max_messages: Option<usize>,
    max_messages_to_retrieve: Option<usize>,
    initializer: ConversationIdInitializer,
    provide_output_filter: Option<MessageFilter>,
    store_request_filter: Option<MessageFilter>,
    store_response_filter: Option<MessageFilter>,
}

impl std::fmt::Debug for ValkeyChatHistoryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValkeyChatHistoryProvider")
            .field("key_prefix", &self.key_prefix)
            .field("state_key", &self.state_key)
            .field("max_messages", &self.max_messages)
            .field("max_messages_to_retrieve", &self.max_messages_to_retrieve)
            .finish_non_exhaustive()
    }
}

impl ValkeyChatHistoryProvider {
    /// A provider for the server at `url` (`redis://` or `rediss://`). A new
    /// session's conversation id is its session id. No connection is made
    /// until the first run.
    pub fn new(url: &str) -> Result<Self> {
        Ok(Self {
            conn: Arc::new(LazyConnection::open(url)?),
            key_prefix: DEFAULT_VALKEY_KEY_PREFIX.into(),
            state_key: DEFAULT_VALKEY_STATE_KEY.into(),
            max_messages: None,
            max_messages_to_retrieve: None,
            initializer: Arc::new(|s: &AgentSession| s.session_id().to_string()),
            provide_output_filter: None,
            store_request_filter: None,
            store_response_filter: None,
        })
    }

    /// The key prefix (default `chat_history`).
    pub fn with_key_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.key_prefix = prefix.into();
        self
    }

    /// The session-state key (default `ValkeyChatHistoryProvider`).
    pub fn with_state_key(mut self, key: impl Into<String>) -> Self {
        self.state_key = key.into();
        self
    }

    /// Keep at most `max` messages per conversation, trimming the oldest.
    /// Must be greater than zero.
    pub fn with_max_messages(mut self, max: usize) -> Result<Self> {
        if max == 0 {
            return Err(Error::Configuration(
                "max_messages must be greater than zero.".into(),
            ));
        }
        self.max_messages = Some(max);
        Ok(self)
    }

    /// Read at most the last `max` messages; zero reads nothing.
    pub fn with_max_messages_to_retrieve(mut self, max: usize) -> Self {
        self.max_messages_to_retrieve = Some(max);
        self
    }

    /// Choose a new session's conversation id (default: its session id).
    pub fn with_conversation_id_initializer(
        mut self,
        f: impl Fn(&AgentSession) -> String + Send + Sync + 'static,
    ) -> Self {
        self.initializer = Arc::new(f);
        self
    }

    /// Filter the messages handed to the model.
    pub fn with_provide_output_filter(
        mut self,
        f: impl Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.provide_output_filter = Some(Arc::new(f));
        self
    }

    /// Filter the run's request messages before they are stored.
    pub fn with_store_request_filter(
        mut self,
        f: impl Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.store_request_filter = Some(Arc::new(f));
        self
    }

    /// Filter the run's response messages before they are stored.
    pub fn with_store_response_filter(
        mut self,
        f: impl Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    ) -> Self {
        self.store_response_filter = Some(Arc::new(f));
        self
    }

    /// The session-state key the conversation id lives under.
    pub fn state_key(&self) -> &str {
        &self.state_key
    }

    /// `session`'s conversation id, initializing and recording it if absent.
    pub fn conversation_id(&self, session: &AgentSession) -> Result<String> {
        if let Some(id) = session.state.get(&self.state_key).and_then(|s| {
            s.get("conversationId")
                .and_then(Value::as_str)
                .map(str::to_string)
        }) {
            return Ok(id);
        }
        let id = (self.initializer)(session);
        if id.trim().is_empty() {
            return Err(Error::Configuration(
                "The conversation id must not be empty or whitespace.".into(),
            ));
        }
        session
            .state
            .insert(&self.state_key, json!({"conversationId": id}));
        Ok(id)
    }

    /// The list key for `session`.
    pub fn key(&self, session: &AgentSession) -> Result<String> {
        Ok(format!(
            "{}:{}",
            storage_key_segment(&self.key_prefix, "~p-"),
            storage_key_segment(&self.conversation_id(session)?, "~s-")
        ))
    }

    /// The stored messages for `session`, oldest first, honoring
    /// `max_messages_to_retrieve`.
    pub async fn messages(&self, session: &AgentSession) -> Result<Vec<Message>> {
        let start = match self.max_messages_to_retrieve {
            // LRANGE key 0 -1 would return everything.
            Some(0) => return Ok(Vec::new()),
            Some(n) => -(n as isize),
            None => 0,
        };
        let key = self.key(session)?;
        let mut conn = self.conn.get().await?;
        let raw: Vec<String> = conn.lrange(&key, start, -1).await.map_err(map_redis_err)?;
        Ok(raw
            .iter()
            .filter(|r| !r.is_empty())
            .filter_map(|r| match serde_json::from_str::<Message>(r) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(key = %key, error = %e, "skipping a malformed history entry");
                    None
                }
            })
            .collect())
    }

    /// Append `messages` to `session`'s conversation in one `RPUSH`, then
    /// trim to `max_messages`.
    pub async fn append(&self, session: &AgentSession, messages: &[Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let key = self.key(session)?;
        let payload = messages
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut conn = self.conn.get().await?;
        let _: () = conn.rpush(&key, payload).await.map_err(map_redis_err)?;
        if let Some(max) = self.max_messages {
            let _: () = conn
                .ltrim(&key, -(max as isize), -1)
                .await
                .map_err(map_redis_err)?;
        }
        Ok(())
    }

    /// Delete `session`'s conversation.
    pub async fn clear(&self, session: &AgentSession) -> Result<()> {
        let key = self.key(session)?;
        let mut conn = self.conn.get().await?;
        let _: () = conn.del(&key).await.map_err(map_redis_err)?;
        Ok(())
    }

    /// The number of stored messages for `session`.
    pub async fn message_count(&self, session: &AgentSession) -> Result<usize> {
        let key = self.key(session)?;
        let mut conn = self.conn.get().await?;
        conn.llen(&key).await.map_err(map_redis_err)
    }
}

fn session_of(ctx: &SessionContext) -> Result<&AgentSession> {
    ctx.session
        .as_ref()
        .ok_or_else(|| Error::Configuration("ValkeyChatHistoryProvider requires a session.".into()))
}

#[async_trait]
impl ContextProvider for ValkeyChatHistoryProvider {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let mut stored = self.messages(session_of(ctx)?).await?;
        if let Some(filter) = &self.provide_output_filter {
            stored = filter(stored);
        }
        // History goes ahead of whatever earlier providers added.
        stored.append(&mut ctx.messages);
        ctx.messages = stored;
        Ok(())
    }

    async fn after_run_in_session(
        &self,
        session: &AgentSession,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(());
        }
        let mut request = request_messages.to_vec();
        if let Some(filter) = &self.store_request_filter {
            request = filter(request);
        }
        let mut response = response_messages.to_vec();
        if let Some(filter) = &self.store_response_filter {
            response = filter(response);
        }
        request.extend(response);
        self.append(session, &request).await
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for ValkeyChatHistoryProvider {}
