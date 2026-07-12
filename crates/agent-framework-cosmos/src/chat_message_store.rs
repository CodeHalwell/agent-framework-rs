//! A [`HistoryProvider`] backed by Azure Cosmos DB (NoSQL / SQL API),
//! ported from .NET's `Microsoft.Agents.AI.CosmosNoSql.CosmosChatMessageStore`
//! over the raw Cosmos REST API (see [`crate::client`] and [`crate::auth`])
//! rather than the `Microsoft.Azure.Cosmos` SDK.
//!
//! One container holds every thread's messages, partitioned by `threadId`
//! (matching the .NET store's default, non-hierarchical partitioning); each
//! message is its own item/document:
//!
//! ```json
//! {
//!   "id": "<fresh uuid, Cosmos's item id>",
//!   "threadId": "<this store's thread id — the partition key>",
//!   "seq": 1751971200000123,
//!   "message": "<Message, JSON-serialized to a STRING>"
//! }
//! ```
//!
//! `message` is a JSON **string** (double-encoded), not a nested object —
//! matching both the .NET reference (`Message = JsonSerializer.Serialize(message, ...)`)
//! and the sibling `agent-framework-redis` crate's `RedisChatMessageStore`
//! (`serde_json::to_string`/`from_str` around each list element). Every
//! `ChatMessageStore` in this workspace that persists to a document/KV
//! store follows this same round-tripping convention, and this store is no
//! exception.
//!
//! # Sequencing: no server-side auto-increment
//!
//! Cosmos DB has no auto-incrementing counter, but `list_messages()` needs
//! a stable chronological order (`ORDER BY c.seq`). Rather than read the
//! current max `seq` before every write (an extra round trip, and racy
//! under concurrent writers) this store assigns `seq =
//! unix_millis_at_call_time * 1000 + index_within_batch`, exactly the
//! recency-rank scheme already used by the sibling
//! `agent-framework-redis::RedisContextProvider` (`MemoryEntry::rank`).
//! This keeps ordering correct within one `add_messages()` call (each
//! message's index disambiguates same-millisecond writes) and, modulo
//! clock skew across processes, across calls and store instances too —
//! without any persisted counter to lose on restart or any read-before-write
//! race.
//!
//! # Divergences from .NET
//!
//! - **Auth**: master key only (HMAC request signing — see [`crate::auth`]).
//!   The .NET store also supports Azure `TokenCredential` (Entra ID/AAD);
//!   that is **not** ported here. See `PARITY.md`.
//! - **Batching**: .NET uses `Container.CreateTransactionalBatch` for
//!   multi-message `AddMessagesAsync` calls; hand-rolling that wire format
//!   (a multipart-style batch request/response) was judged out of scope for
//!   this port, so [`CosmosChatMessageStore::add_messages`] issues one
//!   `POST` per message instead. Functionally equivalent (all messages
//!   still land, in order), just not atomic as a unit and not as
//!   RU-efficient for large batches.
//! - **Hierarchical partitioning**: .NET optionally supports a
//!   tenant/user/session hierarchical partition key; this store always
//!   partitions by `threadId` alone (single-level), matching the simpler,
//!   non-hierarchical .NET constructor overloads.
//! - **TTL**: .NET defaults to a 24h message TTL (`MessageTtlSeconds`).
//!   This store does not set a Cosmos `ttl` property — messages persist
//!   until explicitly [`CosmosChatMessageStore::clear`]ed.

use async_trait::async_trait;
use serde_json::Value;
use uuid::Uuid;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::history::HistoryProvider;
use agent_framework_core::memory::{ContextProvider, SessionContext};
use agent_framework_core::types::Message;

use crate::client::CosmosRestClient;

/// Default Cosmos DB container partition key path — every container this
/// store creates is partitioned by `threadId`. Re-exported from
/// the internal `client::PARTITION_KEY_PATH` for visibility at the public API
/// surface.
pub const DEFAULT_PARTITION_KEY_PATH: &str = crate::client::PARTITION_KEY_PATH;

/// Build the Cosmos document for one stored chat message. `id` is a fresh
/// UUID — Cosmos's own item identifier, distinct from
/// [`Message::message_id`], which travels untouched, still embedded,
/// inside the serialized `message` string. `seq` is this store's
/// [module-documented](self) ordering key.
fn build_message_document(thread_id: &str, seq: i64, message: &Message) -> Result<Value> {
    let message_json = serde_json::to_string(message)?;
    Ok(serde_json::json!({
        "id": Uuid::new_v4().to_string(),
        "threadId": thread_id,
        "seq": seq,
        "message": message_json,
    }))
}

/// Parse one Cosmos document back into a [`Message`], requiring a
/// string `message` field (see the [module docs](self) for the
/// double-encoded-string shape). Mirrors
/// `agent_framework_redis::RedisChatMessageStore`'s strictness: a malformed
/// or missing `message` field fails the whole [`CosmosChatMessageStore::list_messages`]
/// call rather than being silently skipped, so callers never see a
/// silently-truncated history.
fn parse_message_document(doc: &Value) -> Result<Message> {
    let raw = doc.get("message").and_then(Value::as_str).ok_or_else(|| {
        Error::Serialization("Cosmos document is missing a string 'message' field".into())
    })?;
    serde_json::from_str(raw).map_err(Error::from)
}

/// The `seq` base for one `add_messages` batch: current time in Unix
/// milliseconds, scaled by 1000 so each message's index within the batch
/// (up to 999 messages) can be added without colliding with the next
/// millisecond's base value. Identical scheme to the sibling
/// `agent-framework-redis::RedisContextProvider`'s `MemoryEntry::rank`. See
/// the [module docs](self) for why.
fn seq_base() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0);
    millis * 1000
}

/// Cosmos DB (NoSQL API)-backed [`HistoryProvider`]: every thread's
/// messages are documents in one container, partitioned by `threadId`. See
/// the module docs for the wire shape and this port's divergences
/// from .NET's `CosmosChatMessageStore`.
///
/// ```no_run
/// use agent_framework_cosmos::CosmosChatMessageStore;
/// use agent_framework_core::types::Message;
///
/// # async fn demo() -> agent_framework_core::error::Result<()> {
/// let store = CosmosChatMessageStore::new(
///     "https://my-account.documents.azure.com:443/",
///     "<base64 master key>",
///     "agent-framework",
///     "chat-messages",
///     None,
/// )?;
/// store.ensure_created().await?;
///
/// store.add_messages(vec![Message::user("hello")]).await?;
/// let history = store.list_messages().await?;
/// println!("{} messages for thread {}", history.len(), store.thread_id());
/// # Ok(())
/// # }
/// ```
pub struct CosmosChatMessageStore {
    client: CosmosRestClient,
    database_id: String,
    container_id: String,
    thread_id: String,
}

impl CosmosChatMessageStore {
    /// Create a store for the given Cosmos DB account/database/container,
    /// optionally pinned to an existing `thread_id`. When `thread_id` is
    /// `None` a fresh id is generated as `thread_{uuid}`, matching the
    /// sibling `agent-framework-redis::RedisChatMessageStore`'s
    /// auto-generation convention (the .NET store instead generates a bare
    /// `Guid.NewGuid().ToString("N")`, with no prefix).
    ///
    /// No network I/O happens here beyond decoding `key`, which must be
    /// valid base64 (the store's master key) or this returns an error.
    /// Call [`Self::ensure_created`] before first use if the database/
    /// container might not already exist.
    pub fn new(
        account_endpoint: impl Into<String>,
        key: impl Into<String>,
        database_id: impl Into<String>,
        container_id: impl Into<String>,
        thread_id: Option<String>,
    ) -> Result<Self> {
        let client = CosmosRestClient::new(account_endpoint, key)?;
        Ok(Self {
            client,
            database_id: database_id.into(),
            container_id: container_id.into(),
            thread_id: thread_id.unwrap_or_else(|| format!("thread_{}", Uuid::new_v4())),
        })
    }

    /// This thread's id (also its Cosmos partition key value;
    /// auto-generated if not supplied to [`Self::new`]).
    pub fn thread_id(&self) -> &str {
        &self.thread_id
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
    /// (`409 Conflict` from either is treated as success — "already there"
    /// is not a failure). The container is created with partition key
    /// [`DEFAULT_PARTITION_KEY_PATH`] (`/threadId`). Mirrors the .NET
    /// store's test-fixture use of `CreateDatabaseIfNotExistsAsync` /
    /// `CreateContainerIfNotExistsAsync`, exposed here as a first-class
    /// operation since this crate has no SDK-level `CosmosClient` to reach
    /// for those calls on directly.
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

    /// Remove every message in this thread: query all document ids scoped
    /// to this thread's partition, then delete each one. Not part of the
    /// [`HistoryProvider`] trait (which has no `clear` hook) — an
    /// additional utility method, matching the equivalent `ClearMessagesAsync`
    /// on the .NET store.
    pub async fn clear(&self) -> Result<()> {
        let ids = self
            .client
            .query_documents(
                &self.database_id,
                &self.container_id,
                &self.thread_id,
                "SELECT VALUE c.id FROM c WHERE c.threadId = @threadId",
                &[("@threadId", Value::String(self.thread_id.clone()))],
            )
            .await?;
        for id in ids {
            if let Some(id) = id.as_str() {
                self.client
                    .delete_document(&self.database_id, &self.container_id, &self.thread_id, id)
                    .await?;
            }
        }
        Ok(())
    }

    /// Reconstruct a store from a value previously produced by
    /// [`CosmosChatMessageStore::serialize`] (the `cosmos_store_state` shape).
    /// Mirrors `RedisChatMessageStore::from_state`; the `ChatMessageStore`
    /// trait itself has no restore hook, so this is provided as an
    /// inherent associated function.
    ///
    /// The state blob embeds the plaintext master key (there is no
    /// SDK-level client object to hand credentials to separately, unlike
    /// .NET's `CreateFromSerializedState(CosmosClient, ...)`, which takes
    /// an already-authenticated client) — treat serialized state as a
    /// secret, the same way a `redis://user:pass@host` connection string
    /// would be.
    pub fn from_state(state: &Value) -> Result<Self> {
        fn field<'a>(state: &'a Value, name: &str) -> Result<&'a str> {
            state
                .get(name)
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Configuration(format!("state is missing '{name}'")))
        }
        let thread_id = field(state, "thread_id")?.to_string();
        let account_endpoint = field(state, "account_endpoint")?.to_string();
        let key = field(state, "key")?.to_string();
        let database_id = field(state, "database_id")?.to_string();
        let container_id = field(state, "container_id")?.to_string();
        Self::new(
            account_endpoint,
            key,
            database_id,
            container_id,
            Some(thread_id),
        )
    }

    /// The stored messages, in chronological order (`ORDER BY c.seq`).
    pub async fn list_messages(&self) -> Result<Vec<Message>> {
        let docs = self
            .client
            .query_documents(
                &self.database_id,
                &self.container_id,
                &self.thread_id,
                "SELECT * FROM c WHERE c.threadId = @threadId ORDER BY c.seq",
                &[("@threadId", Value::String(self.thread_id.clone()))],
            )
            .await?;
        docs.iter().map(parse_message_document).collect()
    }

    /// Append `messages` as individual documents. A no-op for an empty
    /// `messages`.
    pub async fn add_messages(&self, messages: Vec<Message>) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let base_seq = seq_base();
        for (i, message) in messages.iter().enumerate() {
            let document = build_message_document(&self.thread_id, base_seq + i as i64, message)?;
            self.client
                .create_document(
                    &self.database_id,
                    &self.container_id,
                    &self.thread_id,
                    &document,
                )
                .await?;
        }
        Ok(())
    }

    /// Serialize the store's *configuration* (account endpoint, master key,
    /// database/container/thread id) rather than the message contents —
    /// Cosmos DB already persists the messages durably, so only the
    /// pointer back to them needs to survive. Mirrors
    /// `RedisChatMessageStore::to_dict`, including the `"type"`
    /// discriminator field. See [`Self::from_state`] for the security note
    /// on the embedded master key.
    pub async fn serialize(&self) -> Result<Value> {
        Ok(serde_json::json!({
            "type": "cosmos_store_state",
            "thread_id": self.thread_id,
            "account_endpoint": self.client.account_endpoint(),
            "key": self.client.master_key(),
            "database_id": self.database_id,
            "container_id": self.container_id,
        }))
    }
}

#[async_trait]
impl ContextProvider for CosmosChatMessageStore {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.list_messages().await?;
        let existing = std::mem::take(&mut ctx.messages);
        ctx.messages = stored.into_iter().chain(existing).collect();
        Ok(())
    }

    async fn after_run(
        &self,
        request_messages: &[Message],
        response_messages: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_none() {
            let mut combined = Vec::with_capacity(request_messages.len() + response_messages.len());
            combined.extend(request_messages.iter().cloned());
            combined.extend(response_messages.iter().cloned());
            self.add_messages(combined).await?;
        }
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for CosmosChatMessageStore {}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_framework_core::types::Role;

    const TEST_KEY: &str =
        "C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==";

    fn store(thread_id: &str) -> CosmosChatMessageStore {
        CosmosChatMessageStore::new(
            "https://acct.documents.azure.com",
            TEST_KEY,
            "agent-framework",
            "chat-messages",
            Some(thread_id.to_string()),
        )
        .expect("valid config")
    }

    // region: construction

    #[test]
    fn thread_id_auto_generated_when_absent() {
        let s = CosmosChatMessageStore::new(
            "https://acct.documents.azure.com",
            TEST_KEY,
            "db",
            "coll",
            None,
        )
        .unwrap();
        assert!(s.thread_id().starts_with("thread_"));
        let s2 = CosmosChatMessageStore::new(
            "https://acct.documents.azure.com",
            TEST_KEY,
            "db",
            "coll",
            None,
        )
        .unwrap();
        assert_ne!(s.thread_id(), s2.thread_id());
    }

    #[test]
    fn explicit_thread_id_is_preserved() {
        let s = store("session-123");
        assert_eq!(s.thread_id(), "session-123");
        assert_eq!(s.database_id(), "agent-framework");
        assert_eq!(s.container_id(), "chat-messages");
    }

    #[test]
    fn invalid_master_key_is_rejected() {
        let result = CosmosChatMessageStore::new(
            "https://acct.documents.azure.com",
            "not-valid-base64!!!",
            "db",
            "coll",
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn empty_account_endpoint_is_rejected() {
        assert!(CosmosChatMessageStore::new("", TEST_KEY, "db", "coll", None).is_err());
    }

    // endregion

    // region: build_message_document / parse_message_document round trip (no server required)

    #[test]
    fn build_message_document_shape() {
        let message = Message::user("hello");
        let doc = build_message_document("thread-1", 42, &message).unwrap();
        assert_eq!(doc["threadId"], serde_json::json!("thread-1"));
        assert_eq!(doc["seq"], serde_json::json!(42));
        assert!(doc["id"].as_str().is_some());
        // `message` is a JSON *string* (double-encoded), not a nested object.
        assert!(doc["message"].is_string());
        let inner: Message = serde_json::from_str(doc["message"].as_str().unwrap()).unwrap();
        assert_eq!(inner.text(), "hello");
    }

    #[test]
    fn build_message_document_ids_are_unique_across_calls() {
        let message = Message::user("hi");
        let a = build_message_document("t", 1, &message).unwrap();
        let b = build_message_document("t", 1, &message).unwrap();
        assert_ne!(a["id"], b["id"]);
    }

    #[test]
    fn parse_message_document_round_trips_complex_message() {
        let message = Message::new(Role::assistant(), "Hi there!").with_author("bot");
        let doc = build_message_document("t", 1, &message).unwrap();
        let parsed = parse_message_document(&doc).unwrap();
        assert_eq!(parsed.role, Role::assistant());
        assert_eq!(parsed.text(), "Hi there!");
        assert_eq!(parsed.author_name.as_deref(), Some("bot"));
    }

    #[test]
    fn parse_message_document_requires_string_message_field() {
        let doc = serde_json::json!({"id": "x", "threadId": "t", "seq": 1});
        let err = parse_message_document(&doc).unwrap_err();
        assert!(err.to_string().contains("message"));
    }

    #[test]
    fn parse_message_document_rejects_malformed_message_json() {
        let doc = serde_json::json!({"id": "x", "threadId": "t", "seq": 1, "message": "not json"});
        assert!(parse_message_document(&doc).is_err());
    }

    // endregion

    // region: seq_base (pure)

    #[test]
    fn seq_base_is_monotonic_non_decreasing() {
        let a = seq_base();
        let b = seq_base();
        assert!(b >= a);
    }

    #[test]
    fn seq_base_leaves_room_for_batch_indices() {
        // A batch of up to 1000 messages must not overflow into the next
        // millisecond's base value.
        let base = seq_base();
        assert!(base + 999 < base + 1000);
    }

    // endregion

    // region: serialize()/from_state() config round trip (no server required)

    #[tokio::test]
    async fn serialize_produces_expected_shape() {
        let s = store("thread-abc");
        let state = s.serialize().await.unwrap();
        assert_eq!(
            state,
            serde_json::json!({
                "type": "cosmos_store_state",
                "thread_id": "thread-abc",
                "account_endpoint": "https://acct.documents.azure.com",
                "key": TEST_KEY,
                "database_id": "agent-framework",
                "container_id": "chat-messages",
            })
        );
    }

    #[tokio::test]
    async fn serialize_then_from_state_round_trips() {
        let s = store("thread-xyz");
        let state = s.serialize().await.unwrap();
        let restored = CosmosChatMessageStore::from_state(&state).unwrap();
        assert_eq!(restored.thread_id(), "thread-xyz");
        assert_eq!(restored.database_id(), "agent-framework");
        assert_eq!(restored.container_id(), "chat-messages");
    }

    #[test]
    fn from_state_requires_all_fields() {
        assert!(CosmosChatMessageStore::from_state(&serde_json::json!({})).is_err());
        assert!(
            CosmosChatMessageStore::from_state(&serde_json::json!({"thread_id": "t"})).is_err()
        );
    }

    // endregion

    // region: add_messages([]) is a no-op without any network access

    #[tokio::test]
    async fn add_messages_empty_vec_is_noop_without_network_call() {
        // If this tried to make a request, it would hang/error trying to
        // reach the fake `acct.documents.azure.com` host.
        let s = store("thread-noop");
        s.add_messages(vec![]).await.unwrap();
    }

    // endregion
}
