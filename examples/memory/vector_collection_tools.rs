//! Hand a vector collection to an agent as tools.
//!
//! `vector_store.rs` shows a collection driven by *your* code. This shows one
//! driven by the *model*: `VectorCollectionContextProvider` generates a
//! search / get / upsert / delete tool over any `VectorCollection`, so the
//! agent can look things up and write them down on its own.
//!
//! Three things worth noticing, because each is a decision rather than a
//! default:
//!
//! * **Writes ask first.** `upsert` and `delete` are `AlwaysRequire` by
//!   default, so a run that decides to tidy up the collection pauses for a
//!   human. Reads are free.
//! * **The scope filter groups, it does not authorize.** Every generated tool
//!   is scoped to `tenant == "acme"` below: the other tenant's record is
//!   invisible to search, reads as absent to `get`, and is not deleted. That
//!   is reliable between cooperating parties and is not a security boundary —
//!   see the module docs.
//! * **The model never authors an embedding.** `embed_from_field("text")`
//!   says which field the vector is derived from; the provider embeds it on
//!   the way in, and the vector field is not even in the tool's schema.
//!
//! The store here is the in-memory one so the example runs offline, but the
//! provider takes `Arc<dyn VectorCollection>` — point it at
//! `AzureAISearchStore` or `CosmosVectorStore` and nothing below changes.
//!
//! Skips gracefully unless `OPENAI_API_KEY` is set (the agent and the
//! embeddings both need it).
//!
//! ```bash
//! OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example vector_collection_tools
//! ```

use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::vectors::{
    Filter, InMemoryVectorStore, VectorCollection, VectorCollectionContextProvider, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_core::client::EmbeddingClient;
use serde_json::json;

/// `text-embedding-3-small`'s width.
const DIMENSIONS: usize = 1536;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(chat), Ok(embedder)) = (
        OpenAIChatCompletionClient::from_env("gpt-4o-mini"),
        OpenAIEmbeddingClient::from_env("text-embedding-3-small"),
    ) else {
        println!("set OPENAI_API_KEY to run this example");
        return Ok(());
    };
    let embedder: Arc<dyn EmbeddingClient> = Arc::new(embedder);

    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("tenant").with_type("str"),
        VectorStoreField::vector("embedding", DIMENSIONS),
    ])?;
    let store = InMemoryVectorStore::new();
    let collection: Arc<dyn VectorCollection> =
        Arc::from(store.get_collection("notes", definition)?);
    collection.ensure_collection_exists().await?;

    // Two tenants' records in one collection. Only one of them is this
    // agent's to see.
    let seed = [
        (
            "n1",
            "The office coffee machine is on the third floor.",
            "acme",
        ),
        ("n2", "Expenses are filed in the portal by the 5th.", "acme"),
        ("n3", "Our launch date is March 14th.", "other"),
    ];
    let embeddings = embedder
        .get_embeddings(seed.iter().map(|(_, t, _)| t.to_string()).collect(), None)
        .await?;
    collection
        .upsert(
            seed.iter()
                .zip(embeddings.embeddings.iter())
                .map(|((id, text, tenant), embedding)| {
                    json!({
                        "id": id,
                        "text": text,
                        "tenant": tenant,
                        "embedding": embedding.vector,
                    })
                })
                .collect(),
        )
        .await?;

    let provider = VectorCollectionContextProvider::builder(Arc::clone(&collection), embedder)
        .tool_prefix("notes")
        .scope_filter(Filter::eq("tenant", "acme")?)
        .embed_from_field("text")
        .build()?;
    println!(
        "tools: {:?}",
        provider
            .tools()
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>()
    );

    let agent = Agent::builder(chat)
        .instructions("Answer from the notes collection. Say so if you find nothing.")
        .context_provider(Arc::new(provider))
        .build();

    // The agent reaches for `notes_search` on its own.
    let answer = agent.run_once("where is the coffee machine?").await?;
    println!("\nQ: where is the coffee machine?\nA: {}", answer.text());

    // The other tenant's note is not reachable through these tools, whatever
    // the model asks for.
    let scoped = agent.run_once("when is the launch date?").await?;
    println!("\nQ: when is the launch date?\nA: {}", scoped.text());

    Ok(())
}
