//! Long-term memory with Microsoft Foundry's managed memory stores:
//! `FoundryMemoryProvider` searches a memory store for relevant memories and
//! injects them into the next request (`before_run()`), then writes the turn
//! back so later runs can retrieve it (`after_run()`).
//!
//! It speaks the Foundry **project** data plane directly --
//! `POST {endpoint}/memory_stores/{name}:search_memories` and
//! `:update_memories` -- against the same project endpoint
//! `FoundryChatClient` takes, authenticated with `FOUNDRY_SCOPE`.
//!
//! Memories are isolated by **scope**: pin one with `with_scope` (a user or
//! tenant id), or leave it unset to use the session id. Nothing is stored or
//! retrieved when neither is available.
//!
//! Prerequisites: a Foundry project with a memory store, and Entra ID
//! credentials (`az login` or a managed identity). Skips gracefully unless
//! FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MEMORY_STORE are set.
//!
//! ```bash
//! az login && FOUNDRY_PROJECT_ENDPOINT=https://<res>.services.ai.azure.com/api/projects/<proj> \
//! FOUNDRY_MEMORY_STORE=my-store FOUNDRY_MODEL=gpt-4o-mini \
//! cargo run -p agent-framework-examples --example foundry_memory
//! ```

use std::sync::Arc;

use agent_framework::azure::DefaultAzureCredential;
use agent_framework::foundry::{FoundryChatClient, FoundryMemoryProvider, FOUNDRY_SCOPE};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(store)) = (
        std::env::var("FOUNDRY_PROJECT_ENDPOINT"),
        std::env::var("FOUNDRY_MEMORY_STORE"),
    ) else {
        println!("set FOUNDRY_PROJECT_ENDPOINT and FOUNDRY_MEMORY_STORE to run this example");
        return Ok(());
    };
    let model = std::env::var("FOUNDRY_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());

    // One credential serves both the chat client and the memory store: they
    // are two surfaces of the same project, on the same audience.
    let credential = Arc::new(DefaultAzureCredential::new(FOUNDRY_SCOPE));

    // `with_update_delay(0)` writes immediately. The service otherwise waits
    // 300s and restarts that timer on each new update, batching a burst of
    // turns into one write -- better in production, unhelpful in a demo that
    // wants to read back what it just stored.
    let memory = FoundryMemoryProvider::new(&endpoint, &store, credential.clone())
        .with_scope("user-42")
        .with_update_delay(0);

    let agent = Agent::builder(FoundryChatClient::with_token_credential(
        &endpoint, &model, credential,
    ))
    .instructions("You are a helpful assistant.")
    .context_provider(Arc::new(memory))
    .build();

    // Teach it a fact...
    let first = agent
        .run_once("Remember this: my favorite tea is Earl Grey.")
        .await?;
    println!("agent: {}", first.text());

    // ...then ask again. The provider searches the store for this scope's
    // memories and injects what it finds ahead of the question. (Extraction
    // is asynchronous, so a brand-new memory can take a moment to appear.)
    let second = agent.run_once("What is my favorite tea?").await?;
    println!("agent: {}", second.text());

    Ok(())
}
