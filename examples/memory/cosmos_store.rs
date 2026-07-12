//! Azure Cosmos DB (NoSQL) as a conversation store: `CosmosChatMessageStore`
//! is a `HistoryProvider` that persists a session's messages as documents
//! partitioned by thread id, authenticating with the account's master key
//! (HMAC-signed REST -- no SDK).
//!
//! Skips gracefully unless configured (works against the Cosmos DB Emulator
//! too -- its well-known endpoint/key work here):
//!   COSMOS_ENDPOINT      e.g. https://<account>.documents.azure.com
//!   COSMOS_KEY           the account master key (base64)
//!   COSMOS_DATABASE_ID   optional (default agent-framework-example)
//!   COSMOS_CONTAINER_ID  optional (default chat-messages)
//!
//! ```bash
//! COSMOS_ENDPOINT=https://... COSMOS_KEY=... \
//! cargo run -p agent-framework-examples --example cosmos_store
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(key)) = (
        std::env::var("COSMOS_ENDPOINT"),
        std::env::var("COSMOS_KEY"),
    ) else {
        println!("set COSMOS_ENDPOINT and COSMOS_KEY to run this example");
        return Ok(());
    };
    let database =
        std::env::var("COSMOS_DATABASE_ID").unwrap_or_else(|_| "agent-framework-example".into());
    let container = std::env::var("COSMOS_CONTAINER_ID").unwrap_or_else(|_| "chat-messages".into());

    // Construction only decodes the key; all I/O is lazy. Pinning thread_id
    // lets a later process resume the same conversation; None generates one.
    let store = Arc::new(CosmosChatMessageStore::new(
        &endpoint,
        &key,
        &database,
        &container,
        Some("example-thread".into()),
    )?);

    // Create the database/container on first run (409 "already exists" is
    // treated as success). Partition key is /threadId.
    store.ensure_created().await?;

    // Any `HistoryProvider` can back an `AgentSession`'s conversation
    // history; here we exercise the store directly so the example needs no
    // model credentials.
    store
        .add_messages(vec![
            Message::user("What is the capital of France?"),
            Message::assistant("Paris."),
        ])
        .await?;

    let history = store.list_messages().await?;
    println!(
        "{} message(s) persisted for thread '{}':",
        history.len(),
        store.thread_id()
    );
    for msg in &history {
        println!("  {}: {}", msg.role, msg.text());
    }

    // To drive an agent with it instead:
    //   let mut session = AgentSession::new()
    //       .with_context_providers(vec![store.clone() as Arc<dyn ContextProvider>]);
    //   agent.run(vec![Message::user("...")], Some(&mut session)).await?;

    Ok(())
}
