//! Shared, crate-private plumbing used by both [`crate::RedisChatMessageStore`]
//! and [`crate::RedisContextProvider`].

use agent_framework_core::error::{Error, Result};
use redis::aio::MultiplexedConnection;
use tokio::sync::OnceCell;

/// A `redis::Client` plus a lazily-established multiplexed async connection.
///
/// [`redis::aio::MultiplexedConnection`] is a cheap, `Clone`-able handle onto
/// a single background connection (commands from many clones are
/// multiplexed over one socket), so [`LazyConnection::get`] hands out clones
/// rather than requiring callers to serialize on a lock. The connection
/// itself is only opened on the *first* call to `get`, mirroring the
/// Python store's behavior of not touching the network until a store
/// operation actually runs.
pub(crate) struct LazyConnection {
    client: redis::Client,
    cell: OnceCell<MultiplexedConnection>,
}

impl LazyConnection {
    /// Parse `url` into a [`redis::Client`] (no network I/O yet).
    pub(crate) fn open(url: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| Error::Configuration(format!("invalid Redis URL '{url}': {e}")))?;
        Ok(Self {
            client,
            cell: OnceCell::new(),
        })
    }

    /// Return a clone of the shared multiplexed connection, establishing it
    /// on first use.
    pub(crate) async fn get(&self) -> Result<MultiplexedConnection> {
        self.cell
            .get_or_try_init(|| async { self.client.get_multiplexed_async_connection().await })
            .await
            .cloned()
            .map_err(map_redis_err)
    }
}

/// Map a [`redis::RedisError`] onto the framework's [`Error::Service`] variant.
pub(crate) fn map_redis_err(e: redis::RedisError) -> Error {
    Error::service(format!("redis error: {e}"))
}
