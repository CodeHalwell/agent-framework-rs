//! Azure AI Search as a long-term-memory `ContextProvider`:
//! `AzureAISearchProvider` runs a hybrid/semantic search against a search
//! index on every turn and injects the retrieved documents as extra
//! instructions for the model -- the same `invoking()`/`AggregateContextProvider`
//! wiring `redis_memory.rs` and `mem0_memory.rs` use, backed by a different
//! store.
//!
//! Skips gracefully unless configured:
//!   AZURE_SEARCH_ENDPOINT   e.g. https://<service>.search.windows.net
//!   AZURE_SEARCH_API_KEY    an admin or query api-key
//!   AZURE_SEARCH_INDEX      the index to query
//! plus OPENAI_API_KEY for the model.
//!
//! ```bash
//! AZURE_SEARCH_ENDPOINT=https://my-search.search.windows.net \
//! AZURE_SEARCH_API_KEY=... AZURE_SEARCH_INDEX=my-index \
//! OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example azure_ai_search
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(api_key), Ok(index), Ok(client)) = (
        std::env::var("AZURE_SEARCH_ENDPOINT"),
        std::env::var("AZURE_SEARCH_API_KEY"),
        std::env::var("AZURE_SEARCH_INDEX"),
        OpenAIChatCompletionClient::from_env("gpt-4o-mini"),
    ) else {
        println!(
            "set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX, \
             and OPENAI_API_KEY to run this example"
        );
        return Ok(());
    };

    // Hybrid search: `.with_top` caps how many documents come back;
    // `.with_semantic_configuration("...")` turns on the semantic reranker
    // (only if your index has one configured); `.with_vector_field("...")`
    // adds a vector query too (server-side vectorization by default, or
    // client-side via `.with_embedding_function`). Authentication can be a
    // Microsoft Entra ID token instead of an api-key via
    // `AzureAISearchProvider::with_token_credential`.
    let search = AzureAISearchProvider::with_api_key(endpoint, index, api_key).with_top(5);

    let mut providers = AggregateContextProvider::new();
    providers.add(Arc::new(search));

    let agent = Agent::builder(client)
        .name("assistant")
        .instructions("Answer using the retrieved context when it's relevant.")
        .context_provider(Arc::new(providers))
        .build();

    // Every run's `invoking()` hook queries the index with the latest user
    // message and folds the results into the request as extra instructions
    // -- the agent never calls Azure AI Search directly.
    let response = agent
        .run_once("What does the documentation say? Summarize in two sentences.")
        .await?;
    println!("{}", response.text());

    Ok(())
}
