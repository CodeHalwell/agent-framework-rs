//! # agent-framework-cosmos
//!
//! Azure Cosmos DB (NoSQL / SQL API)-backed
//! [`HistoryProvider`](agent_framework_core::history::HistoryProvider) for
//! `agent-framework-rs`, porting `Microsoft.Agents.AI.CosmosNoSql` from the
//! .NET Agent Framework.
//!
//! - [`CosmosChatMessageStore`] — one container holds every thread's
//!   messages as individual documents, partitioned by `threadId`. Talks
//!   directly to the [Cosmos DB REST
//!   API](https://learn.microsoft.com/en-us/rest/api/cosmos-db/) — see the
//!   internal `auth` module — rather than depending on the
//!   `azure_data_cosmos`/`Microsoft.Azure.Cosmos` SDK.
//!
//! # Authentication
//!
//! Both modes the .NET `Microsoft.Agents.AI.CosmosNoSql` package supports are
//! available, and every type here offers a constructor for each:
//!
//! - **Master key** ([`CosmosChatMessageStore::new`],
//!   [`CosmosCheckpointStorage::new`]) — per-request HMAC-SHA256 signing.
//! - **Microsoft Entra ID** ([`CosmosChatMessageStore::with_token_credential`],
//!   [`CosmosCheckpointStorage::with_token_credential`]) — a bearer token from
//!   any [`agent_framework_azure::TokenCredential`], so a workload
//!   authenticates as itself with no key anywhere in the deployment. This is
//!   the only mode that works on an account with `disableLocalAuth` set,
//!   which many tenants require by policy.
//!
//! Entra ID's Cosmos DB RBAC grants **data-plane** actions only, so
//! `ensure_created` — a control-plane operation — cannot succeed with a token
//! however the principal is assigned; the database and container must be
//! provisioned out of band (ARM/Bicep, the Azure CLI, or the portal). The
//! call reports that rather than surfacing a bare `403`. See
//! [`CosmosChatMessageStore::with_token_credential`] for the role assignment
//! the data operations do need.
//!
//! ```no_run
//! use agent_framework_cosmos::CosmosChatMessageStore;
//! use agent_framework_core::types::Message;
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
//! store.add_messages(vec![Message::user("Hello!")]).await?;
//! let history = store.list_messages().await?;
//! println!("{} messages", history.len());
//! # Ok(())
//! # }
//! ```

mod auth;
mod chat_message_store;
mod checkpoint_storage;
mod client;
mod dates;

pub use chat_message_store::{CosmosChatMessageStore, DEFAULT_PARTITION_KEY_PATH};
pub use checkpoint_storage::{
    CosmosCheckpointStorage, DEFAULT_PARTITION_KEY_PATH as DEFAULT_CHECKPOINT_PARTITION_KEY_PATH,
};
pub use client::DEFAULT_API_VERSION;
