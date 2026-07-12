//! Redis-backed conversation history and long-term memory:
//! `RedisChatMessageStore` is a `HistoryProvider` that keeps a session's
//! messages in a Redis LIST, and `RedisContextProvider` stores/retrieves
//! scoped long-term memories.
//!
//! Prerequisite: a reachable Redis server (default `redis://127.0.0.1:6379`,
//! override with REDIS_URL) -- e.g. `docker run --rm -p 6379:6379 redis:7`.
//! The example skips gracefully (exit 0) when Redis is unreachable. Note the
//! provider's retrieval is recency-based, not vector search (see PARITY.md).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example redis_memory
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;

/// An offline stand-in for a model so this example only needs Redis.
#[derive(Clone)]
struct CannedClient;

#[async_trait]
impl ChatClient for CannedClient {
    async fn get_response(
        &self,
        messages: Vec<Message>,
        _options: ChatOptions,
    ) -> Result<ChatResponse> {
        // The provider injects remembered context as extra messages, so the
        // "model" can prove the memories arrived by echoing the count.
        Ok(ChatResponse::from_text(format!(
            "(canned reply) I received {} message(s) of context/history.",
            messages.len()
        )))
    }

    async fn get_streaming_response(
        &self,
        messages: Vec<Message>,
        options: ChatOptions,
    ) -> Result<ChatStream> {
        let resp = self.get_response(messages, options).await?;
        let updates = resp.messages.into_iter().map(|m| {
            Ok(ChatResponseUpdate {
                contents: m.contents,
                role: Some(m.role),
                ..Default::default()
            })
        });
        Ok(futures::stream::iter(updates.collect::<Vec<_>>()).boxed())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".into());

    // Conversation history: one Redis LIST per session id, trimmed to the
    // most recent 200 messages. Connections are lazy; `ping()` checks
    // reachability so we can skip cleanly on machines without Redis.
    let store = Arc::new(
        RedisChatMessageStore::new(&url, Some("example-session".into()))?.with_max_messages(200),
    );
    if !store.ping().await {
        println!("redis not reachable at {url} -- skipping (start one and re-run)");
        return Ok(());
    }
    store.clear().await?; // fresh demo state on every run

    // Long-term memory, scoped by user id: `after_run()` persists each turn,
    // `before_run()` retrieves recent memories into the next request's context.
    let memory = RedisContextProvider::new(&url)?
        .with_user_id("user-42")
        .with_limit(5);

    let agent = Agent::builder(CannedClient)
        .instructions("You are a helpful assistant.")
        .context_provider(Arc::new(memory))
        .build();

    // Both turns share the Redis-backed session, so the second request also
    // carries the first turn's history read back from Redis. Attaching the
    // store as a context provider is what makes it a `HistoryProvider` for
    // this session -- `Agent` won't layer on its own `InMemoryHistoryProvider`
    // since one is already present.
    let mut session =
        AgentSession::new().with_context_providers(vec![store.clone() as Arc<dyn ContextProvider>]);
    for text in ["I love hiking in the Alps.", "What do you know about me?"] {
        let response = agent
            .run(vec![Message::user(text)], Some(&mut session))
            .await?;
        println!("user: {text}\nagent: {}\n", response.text());
    }

    let persisted = store.list_messages().await?;
    println!(
        "{} message(s) persisted in redis key '{}'",
        persisted.len(),
        store.redis_key()
    );

    Ok(())
}
