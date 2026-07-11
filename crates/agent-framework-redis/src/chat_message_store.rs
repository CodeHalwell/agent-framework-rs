//! A [`ChatMessageStore`] backed by a Redis `LIST`.
//!
//! Mirrors the Python `agent_framework_redis.RedisChatMessageStore`: each
//! conversation thread owns one Redis list at key `{key_prefix}:{thread_id}`,
//! messages are appended with `RPUSH` (chronological order, oldest first),
//! read back with `LRANGE`, and — when `max_messages` is configured —
//! trimmed to the most recent N entries with `LTRIM` after every write. Each
//! list element is one message, JSON-serialized with `serde_json`
//! (equivalent to the Python store's `ChatMessage.to_json()` /
//! `ChatMessage.from_json()` round trip).

use async_trait::async_trait;
use redis::AsyncCommands;
use uuid::Uuid;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::threads::ChatMessageStore;
use agent_framework_core::types::ChatMessage;

use crate::internal::{map_redis_err, LazyConnection};

/// Default Redis key prefix, matching the Python store's default.
pub const DEFAULT_KEY_PREFIX: &str = "chat_messages";

/// Redis-backed [`ChatMessageStore`]: one Redis `LIST` per conversation
/// thread, JSON-serialized messages, optional automatic trimming.
///
/// ```no_run
/// use agent_framework_redis::RedisChatMessageStore;
/// use agent_framework_core::threads::ChatMessageStore;
/// use agent_framework_core::types::ChatMessage;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let store = RedisChatMessageStore::new("redis://127.0.0.1:6379", None)?
///     .with_key_prefix("my_app")
///     .with_max_messages(100);
///
/// store.add_messages(vec![ChatMessage::user("hello")]).await?;
/// let history = store.list_messages().await?;
/// println!("{} messages for thread {}", history.len(), store.thread_id());
/// # Ok(())
/// # }
/// ```
pub struct RedisChatMessageStore {
    conn: LazyConnection,
    redis_url: String,
    thread_id: String,
    key_prefix: String,
    max_messages: Option<usize>,
}

impl RedisChatMessageStore {
    /// Create a store for `redis_url`, optionally pinned to an existing
    /// `thread_id`. When `thread_id` is `None` a fresh id is generated as
    /// `thread_{uuid}`, matching the Python store's `f"thread_{uuid4()}"`.
    ///
    /// The connection to Redis is *not* established here; only the URL is
    /// parsed. Errors if `redis_url` cannot be parsed as a Redis connection
    /// string.
    pub fn new(redis_url: impl Into<String>, thread_id: Option<String>) -> Result<Self> {
        let redis_url = redis_url.into();
        let conn = LazyConnection::open(&redis_url)?;
        Ok(Self {
            conn,
            redis_url,
            thread_id: thread_id.unwrap_or_else(|| format!("thread_{}", Uuid::new_v4())),
            key_prefix: DEFAULT_KEY_PREFIX.to_string(),
            max_messages: None,
        })
    }

    /// Namespace Redis keys under `key_prefix` (builder style). Defaults to
    /// `"chat_messages"`.
    pub fn with_key_prefix(mut self, key_prefix: impl Into<String>) -> Self {
        self.key_prefix = key_prefix.into();
        self
    }

    /// Automatically trim the list to the most recent `max_messages`
    /// entries after every `add_messages` call (builder style).
    pub fn with_max_messages(mut self, max_messages: usize) -> Self {
        self.max_messages = Some(max_messages);
        self
    }

    /// This thread's id (auto-generated if not supplied to [`Self::new`]).
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// The configured key prefix.
    pub fn key_prefix(&self) -> &str {
        &self.key_prefix
    }

    /// The configured message limit, if any.
    pub fn max_messages(&self) -> Option<usize> {
        self.max_messages
    }

    /// The Redis key holding this thread's messages: `{key_prefix}:{thread_id}`.
    pub fn redis_key(&self) -> String {
        format!("{}:{}", self.key_prefix, self.thread_id)
    }

    /// Remove all messages for this thread (`DEL` on [`Self::redis_key`]).
    pub async fn clear(&self) -> Result<()> {
        let mut conn = self.conn.get().await?;
        let _: () = conn.del(self.redis_key()).await.map_err(map_redis_err)?;
        Ok(())
    }

    /// Ping the Redis server, returning `true` on success. Equivalent to the
    /// Python store's `ping()` convenience method.
    pub async fn ping(&self) -> bool {
        let Ok(mut conn) = self.conn.get().await else {
            return false;
        };
        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .is_ok()
    }

    /// Reconstruct a store from a value previously produced by
    /// [`ChatMessageStore::serialize`] (the `redis_store_state` shape).
    /// Mirrors the Python store's `deserialize` classmethod; the
    /// `ChatMessageStore` trait itself has no restore hook, so this is
    /// provided as an inherent associated function.
    pub fn from_state(state: &serde_json::Value) -> Result<Self> {
        let thread_id = state
            .get("thread_id")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::Configuration("state is missing 'thread_id'".into()))?
            .to_string();
        let redis_url = state
            .get("redis_url")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| Error::Configuration("state is missing 'redis_url'".into()))?
            .to_string();
        let key_prefix = state
            .get("key_prefix")
            .and_then(serde_json::Value::as_str)
            .unwrap_or(DEFAULT_KEY_PREFIX)
            .to_string();
        let max_messages = state
            .get("max_messages")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize);

        let mut store = Self::new(redis_url, Some(thread_id))?;
        store.key_prefix = key_prefix;
        store.max_messages = max_messages;
        Ok(store)
    }

    fn serialize_message(message: &ChatMessage) -> Result<String> {
        Ok(serde_json::to_string(message)?)
    }

    fn deserialize_message(data: &str) -> Result<ChatMessage> {
        Ok(serde_json::from_str(data)?)
    }
}

#[async_trait]
impl ChatMessageStore for RedisChatMessageStore {
    async fn list_messages(&self) -> Result<Vec<ChatMessage>> {
        let mut conn = self.conn.get().await?;
        let raw: Vec<String> = conn
            .lrange(self.redis_key(), 0, -1)
            .await
            .map_err(map_redis_err)?;
        raw.iter().map(|s| Self::deserialize_message(s)).collect()
    }

    async fn add_messages(&self, messages: Vec<ChatMessage>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.get().await?;
        let key = self.redis_key();

        // Atomic batch append, mirroring the Python store's
        // `pipeline(transaction=True)` + repeated RPUSH.
        let mut pipe = redis::pipe();
        pipe.atomic();
        for message in &messages {
            let payload = Self::serialize_message(message)?;
            pipe.rpush(&key, payload);
        }
        let _: () = pipe.query_async(&mut conn).await.map_err(map_redis_err)?;

        if let Some(max) = self.max_messages {
            let len: usize = conn.llen(&key).await.map_err(map_redis_err)?;
            if len > max {
                let _: () = conn
                    .ltrim(&key, -(max as isize), -1)
                    .await
                    .map_err(map_redis_err)?;
            }
        }
        Ok(())
    }

    /// Serialize the store's *configuration* (thread id, Redis URL, key
    /// prefix, message limit) rather than the message contents — Redis
    /// already persists the messages durably, so only the pointer back to
    /// them needs to survive. This mirrors the Python store's `serialize()`
    /// / `RedisStoreState`, including the `"type"` discriminator field.
    async fn serialize(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "type": "redis_store_state",
            "thread_id": self.thread_id,
            "redis_url": self.redis_url,
            "key_prefix": self.key_prefix,
            "max_messages": self.max_messages,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::{Content, Role, TextContent};
    use std::collections::HashMap;

    fn store(thread_id: &str) -> RedisChatMessageStore {
        // `redis://.../0` parses without touching the network — safe to
        // construct in hermetic unit tests.
        RedisChatMessageStore::new("redis://127.0.0.1:6379/0", Some(thread_id.to_string()))
            .expect("valid redis url")
    }

    // region: key construction

    #[test]
    fn redis_key_uses_default_prefix() {
        let s = store("t1");
        assert_eq!(s.redis_key(), "chat_messages:t1");
    }

    #[test]
    fn redis_key_uses_custom_prefix() {
        let s = store("t1").with_key_prefix("custom_messages");
        assert_eq!(s.redis_key(), "custom_messages:t1");
    }

    #[test]
    fn thread_id_auto_generated_when_absent() {
        let s = RedisChatMessageStore::new("redis://127.0.0.1:6379/0", None).unwrap();
        assert!(s.thread_id().starts_with("thread_"));
        // "thread_" (7) + a UUIDv4 (36) = 43 chars.
        assert!(s.thread_id().len() > 10);
        // Two auto-generated stores must not collide.
        let s2 = RedisChatMessageStore::new("redis://127.0.0.1:6379/0", None).unwrap();
        assert_ne!(s.thread_id(), s2.thread_id());
    }

    #[test]
    fn explicit_thread_id_is_preserved() {
        let s = store("user123_session456");
        assert_eq!(s.thread_id(), "user123_session456");
        assert_eq!(s.redis_key(), "chat_messages:user123_session456");
    }

    #[test]
    fn max_messages_defaults_to_none() {
        let s = store("t1");
        assert_eq!(s.max_messages(), None);
    }

    #[test]
    fn with_max_messages_sets_limit() {
        let s = store("t1").with_max_messages(100);
        assert_eq!(s.max_messages(), Some(100));
    }

    #[test]
    fn invalid_redis_url_is_rejected() {
        let result = RedisChatMessageStore::new("not-a-redis-url", None);
        assert!(result.is_err());
    }

    // endregion

    // region: message JSON round trip (no server required)

    #[test]
    fn message_serialization_roundtrip_simple() {
        let message = ChatMessage::new(Role::user(), "Hello").with_author("tester");
        let serialized = RedisChatMessageStore::serialize_message(&message).unwrap();
        assert!(serialized.contains("Hello"));
        let deserialized = RedisChatMessageStore::deserialize_message(&serialized).unwrap();
        assert_eq!(deserialized.role, message.role);
        assert_eq!(deserialized.text(), "Hello");
        assert_eq!(deserialized.author_name.as_deref(), Some("tester"));
    }

    #[test]
    fn message_serialization_roundtrip_complex_content() {
        let mut additional_properties = HashMap::new();
        additional_properties.insert("metadata".to_string(), serde_json::json!("test"));
        let message = ChatMessage {
            role: Role::assistant(),
            contents: vec![
                Content::Text(TextContent::new("Hello")),
                Content::Text(TextContent::new("World")),
            ],
            author_name: Some("TestBot".to_string()),
            message_id: Some("complex_msg".to_string()),
            additional_properties,
        };

        let serialized = RedisChatMessageStore::serialize_message(&message).unwrap();
        let deserialized = RedisChatMessageStore::deserialize_message(&serialized).unwrap();

        assert_eq!(deserialized.role, Role::assistant());
        assert_eq!(deserialized.text(), "Hello World");
        assert_eq!(deserialized.author_name.as_deref(), Some("TestBot"));
        assert_eq!(deserialized.message_id.as_deref(), Some("complex_msg"));
        assert_eq!(
            deserialized.additional_properties.get("metadata"),
            Some(&serde_json::json!("test"))
        );
    }

    #[test]
    fn deserialize_rejects_malformed_json() {
        assert!(RedisChatMessageStore::deserialize_message("not json").is_err());
    }

    // endregion

    // region: serialize()/from_state() config round trip (no server required)

    #[tokio::test]
    async fn serialize_produces_python_compatible_shape() {
        let s = store("test_thread_123");
        let state = s.serialize().await.unwrap();
        assert_eq!(
            state,
            serde_json::json!({
                "type": "redis_store_state",
                "thread_id": "test_thread_123",
                "redis_url": "redis://127.0.0.1:6379/0",
                "key_prefix": "chat_messages",
                "max_messages": null,
            })
        );
    }

    #[tokio::test]
    async fn serialize_then_from_state_round_trips() {
        let s = store("test_thread_123")
            .with_key_prefix("custom")
            .with_max_messages(50);
        let state = s.serialize().await.unwrap();

        let restored = RedisChatMessageStore::from_state(&state).unwrap();
        assert_eq!(restored.thread_id(), "test_thread_123");
        assert_eq!(restored.key_prefix(), "custom");
        assert_eq!(restored.max_messages(), Some(50));
        assert_eq!(restored.redis_key(), "custom:test_thread_123");
    }

    #[test]
    fn from_state_requires_thread_id_and_redis_url() {
        assert!(RedisChatMessageStore::from_state(&serde_json::json!({})).is_err());
        assert!(
            RedisChatMessageStore::from_state(&serde_json::json!({"thread_id": "t1"})).is_err()
        );
    }

    // endregion
}
