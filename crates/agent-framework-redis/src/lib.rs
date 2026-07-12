//! # agent-framework-redis
//!
//! Redis-backed [`HistoryProvider`](agent_framework_core::history::HistoryProvider)
//! and [`ContextProvider`](agent_framework_core::memory::ContextProvider) for
//! `agent-framework-rs`, porting `agent_framework_redis` from the Python
//! Agent Framework.
//!
//! - [`RedisChatMessageStore`] — one Redis `LIST` per session,
//!   JSON-serialized messages, optional automatic trimming. A close mirror
//!   of the Python `RedisChatMessageStore`, adapted to the
//!   [`HistoryProvider`](agent_framework_core::history::HistoryProvider)
//!   contract (`before_run`/`after_run`) now that conversation history lives
//!   in a context provider rather than on the session/thread itself.
//! - [`RedisContextProvider`] — scoped long-term memory storage/retrieval.
//!   Ports the Python `RedisProvider`'s *scoping* semantics
//!   (application/agent/user/thread id). When the connected server has
//!   RediSearch loaded (Redis Stack), retrieval is backed by a real
//!   `FT.SEARCH` BM25 full-text index; otherwise it falls back to a
//!   `SCAN`+token-match path over plain Redis. Vector/hybrid search is
//!   **not** ported — see the [`context_provider`] module docs for the full
//!   picture.
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
//!
//! let agent = Agent::builder(client)
//!     .instructions("You are a helpful assistant.")
//!     .context_provider(Arc::new(memory))
//!     .build();
//!
//! let mut session = AgentSession::new().with_context_providers(vec![Arc::new(store)]);
//! let response = agent
//!     .run(vec![Message::user("Hello!")], Some(&mut session))
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
