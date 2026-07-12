//! Long-term memory with the hosted Mem0 API: `Mem0Provider` persists each
//! turn to Mem0 (`invoked()`) and retrieves relevant memories into the next
//! request's context (`invoking()`), scoped by user/agent/application/thread
//! id.
//!
//! Prerequisites: a Mem0 account and MEM0_API_KEY (https://mem0.ai), plus
//! OPENAI_API_KEY for the model. Skips gracefully when MEM0_API_KEY is unset.
//!
//! ```bash
//! MEM0_API_KEY=m0-... OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example mem0_memory
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("MEM0_API_KEY").is_err() {
        println!("set MEM0_API_KEY (and OPENAI_API_KEY) to run this example");
        return Ok(());
    }

    // Reads MEM0_API_KEY and optional MEM0_API_BASE. At least one scope id
    // (user/agent/application/thread) is required so memories don't bleed
    // across users; `with_scope_to_per_operation_thread_id(true)` is the
    // alternative when you want per-thread isolation instead.
    let memory = Mem0Provider::from_env()?.with_user_id("user-42");

    let mut providers = AggregateContextProvider::new();
    providers.add(Arc::new(memory));

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("gpt-4o-mini")?)
        .instructions("You are a helpful assistant.")
        .context_provider(Arc::new(providers))
        .build();

    // Teach it a fact...
    let first = agent
        .run_once("Remember this: my favorite tea is Earl Grey.")
        .await?;
    println!("agent: {}", first.text());

    // ...then ask again. Even on a brand-new thread, the provider searches
    // Mem0 for this user's memories and injects what it finds. (Mem0 indexes
    // asynchronously, so brand-new memories can take a moment to appear.)
    let second = agent.run_once("What is my favorite tea?").await?;
    println!("agent: {}", second.text());

    Ok(())
}
