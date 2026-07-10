//! # agent-framework-redis
//!
//! Redis-backed [`ChatMessageStore`](agent_framework_core::threads::ChatMessageStore)
//! and [`ContextProvider`](agent_framework_core::memory::ContextProvider) for
//! `agent-framework-rs`, porting `agent_framework_redis` from the Python
//! Agent Framework.
//!
//! - [`RedisChatMessageStore`] — one Redis `LIST` per conversation thread,
//!   JSON-serialized messages, optional automatic trimming. A close mirror
//!   of the Python `RedisChatMessageStore`.
//! - [`RedisContextProvider`] — scoped long-term memory storage/retrieval.
//!   Ports the Python `RedisProvider`'s *scoping* semantics
//!   (application/agent/user/thread id) but **not** its RediSearch-backed
//!   vector/full-text search — see the [`context_provider`] module docs for
//!   the simplification this crate implements instead, and why.
//!
//! Both types connect lazily: constructing them only parses the Redis URL
//! (via [`redis::Client::open`]); the actual
//! [`MultiplexedConnection`](redis::aio::MultiplexedConnection) is
//! established on the first call that needs it.
//!
//! ```no_run
//! use agent_framework_core::prelude::*;
//! use agent_framework_redis::{RedisChatMessageStore, RedisContextProvider};
//! use std::sync::Arc;
//!
//! # async fn demo(client: impl ChatClient + 'static) -> Result<()> {
//! let store = RedisChatMessageStore::new("redis://127.0.0.1:6379", None)?
//!     .with_max_messages(200);
//!
//! let memory = RedisContextProvider::new("redis://127.0.0.1:6379")?.with_user_id("user-42");
//! let mut providers = AggregateContextProvider::new();
//! providers.add(Arc::new(memory));
//!
//! let agent = ChatAgent::builder(client)
//!     .instructions("You are a helpful assistant.")
//!     .context_provider(Arc::new(providers))
//!     .build();
//!
//! let mut thread = AgentThread::local(Arc::new(store));
//! let response = agent
//!     .run(vec![ChatMessage::user("Hello!")], Some(&mut thread))
//!     .await?;
//! println!("{}", response.text());
//! # Ok(())
//! # }
//! ```

pub mod chat_message_store;
pub mod context_provider;
mod internal;

pub use chat_message_store::{
    RedisChatMessageStore, DEFAULT_KEY_PREFIX as DEFAULT_STORE_KEY_PREFIX,
};
pub use context_provider::{
    RedisContextProvider, DEFAULT_CONTEXT_PROMPT,
    DEFAULT_KEY_PREFIX as DEFAULT_PROVIDER_KEY_PREFIX, DEFAULT_LIMIT,
};
