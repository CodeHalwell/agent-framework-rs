//! # agent-framework-cosmos
//!
//! Azure Cosmos DB (NoSQL / SQL API)-backed
//! [`ChatMessageStore`](agent_framework_core::threads::ChatMessageStore) for
//! `agent-framework-rs`, porting `Microsoft.Agents.AI.CosmosNoSql` from the
//! .NET Agent Framework.
//!
//! - [`CosmosChatMessageStore`] — one container holds every thread's
//!   messages as individual documents, partitioned by `threadId`. Talks
//!   directly to the [Cosmos DB REST
//!   API](https://learn.microsoft.com/en-us/rest/api/cosmos-db/) with
//!   master-key (HMAC-SHA256) request signing — see the [`auth`] module —
//!   rather than depending on the `azure_data_cosmos`/`Microsoft.Azure.Cosmos`
//!   SDK. **Only master-key authentication is implemented**; Entra ID/AAD
//!   (`TokenCredential`) auth, which the .NET package also supports, is not
//!   ported — see `PARITY.md`.
//!
//! ```no_run
//! use agent_framework_cosmos::CosmosChatMessageStore;
//! use agent_framework_core::threads::ChatMessageStore;
//! use agent_framework_core::types::ChatMessage;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! let store = CosmosChatMessageStore::new(
//!     "https://my-account.documents.azure.com:443/",
//!     "<base64 master key>",
//!     "agent-framework",
//!     "chat-messages",
//!     None,
//! )?;
//! // Creates the database/container (partition key /threadId) if missing.
//! store.ensure_created().await?;
//!
//! store.add_messages(vec![ChatMessage::user("Hello!")]).await?;
//! let history = store.list_messages().await?;
//! println!("{} messages", history.len());
//! # Ok(())
//! # }
//! ```

mod auth;
mod chat_message_store;
mod client;
mod dates;

pub use chat_message_store::{CosmosChatMessageStore, DEFAULT_PARTITION_KEY_PATH};
pub use client::DEFAULT_API_VERSION;
