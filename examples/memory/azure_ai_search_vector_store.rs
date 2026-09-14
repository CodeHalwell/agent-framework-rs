//! Azure AI Search as a **vector store**: create an index from a collection
//! definition, upsert records, and run a filtered vector search.
//!
//! This is the other half of `azure_ai_search.rs`. That example *reads* from
//! an index someone else built, through a `ContextProvider`. This one owns
//! the index: `AzureAISearchStore` creates it, writes records into it, and
//! searches it through the provider-agnostic `VectorCollection` trait — so
//! the same code runs against `InMemoryVectorStore` in a test and against
//! Azure in production.
//!
//! The filter is the portable kind: `Filter::gte("year", 2020)` is translated
//! into the OData `year ge 2020` here, and evaluated in process by the
//! in-memory store, without the caller writing either dialect.
//!
//! Skips gracefully unless configured:
//!   AZURE_SEARCH_ENDPOINT   e.g. https://<service>.search.windows.net
//!   AZURE_SEARCH_API_KEY    an **admin** api-key (a query key cannot create
//!                           an index)
//! plus OPENAI_API_KEY, which embeds the documents and the query.
//!
//! ```bash
//! AZURE_SEARCH_ENDPOINT=https://my-search.search.windows.net \
//! AZURE_SEARCH_API_KEY=... OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example azure_ai_search_vector_store
//! ```

use agent_framework::azure_ai_search::AzureAISearchStore;
use agent_framework::prelude::*;
use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::vectors::{
    Filter, FilterGroup, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use serde_json::json;

/// `text-embedding-3-small`'s width.
const DIMENSIONS: usize = 1536;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(api_key), Ok(embedder)) = (
        std::env::var("AZURE_SEARCH_ENDPOINT"),
        std::env::var("AZURE_SEARCH_API_KEY"),
        OpenAIEmbeddingClient::from_env("text-embedding-3-small"),
    ) else {
        println!(
            "set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY (admin), and OPENAI_API_KEY to run \
             this example"
        );
        return Ok(());
    };

    // The collection's shape. `indexed()` means "filterable" and
    // `full_text_indexed()` means "searchable" — the two Azure AI Search
    // needs to be told at index-creation time, because neither can be turned
    // on later without rebuilding the index.
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("title").full_text_indexed(),
        VectorStoreField::data("year").with_type("int").indexed(),
        VectorStoreField::data("kind").indexed(),
        VectorStoreField::vector("embedding", DIMENSIONS),
    ])?;

    let store = AzureAISearchStore::with_api_key(endpoint, api_key);
    let docs = store.collection("af-rs-demo", definition)?;

    // Creates the index when it is absent, and leaves an existing one alone
    // rather than rewriting its schema. `build_index()` returns the same
    // schema as JSON if you would rather create it yourself with extras the
    // definition cannot express (a semantic configuration, a scoring profile).
    docs.ensure_collection_exists().await?;

    let documents = [
        ("d1", "Quarterly earnings", 2024, "report"),
        ("d2", "Safety handbook", 2019, "manual"),
        ("d3", "Product roadmap", 2026, "report"),
    ];
    let embeddings = embedder
        .get_embeddings(
            documents
                .iter()
                .map(|(_, title, ..)| title.to_string())
                .collect(),
            None,
        )
        .await?;
    let records: Vec<_> = documents
        .iter()
        .zip(embeddings.embeddings.iter())
        .map(|((id, title, year, kind), embedding)| {
            json!({
                "id": id,
                "title": title,
                "year": year,
                "kind": kind,
                "embedding": embedding.vector,
            })
        })
        .collect();
    docs.upsert(records).await?;
    println!("upserted {} documents", documents.len());

    // Azure AI Search indexes asynchronously: a document is durable as soon
    // as the write returns, but may not be searchable for a moment after.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    let query = embedder
        .get_embeddings(vec!["how did the business do?".into()], None)
        .await?;
    let query_vector = query.embeddings[0].vector.clone();

    // One portable filter, two conditions: reports from 2020 onwards. The
    // connector turns this into `(kind eq 'report' and year ge 2020)`.
    let filter = FilterGroup::and(vec![
        Filter::eq("kind", "report")?.into(),
        Filter::gte("year", 2020)?.into(),
    ])?;

    let hits = docs
        .search(
            query_vector.clone(),
            &VectorSearchOptions::new(5).with_filter(filter),
        )
        .await?;
    println!("\nvector search (filtered):");
    for hit in &hits {
        println!(
            "  {:.3}  {}  ({})",
            hit.score.unwrap_or_default(),
            hit.record["title"].as_str().unwrap_or_default(),
            hit.record["year"]
        );
    }

    // Keyword-hybrid retrieval — a full-text query fused with the vector one
    // by the service. Not on the `VectorCollection` trait, since that takes
    // only a vector, so it is reached through the concrete collection type.
    let hybrid = docs
        .search_hybrid("earnings", query_vector, &VectorSearchOptions::new(3))
        .await?;
    println!("\nkeyword-hybrid search:");
    for hit in &hybrid {
        println!(
            "  {:.3}  {}",
            hit.score.unwrap_or_default(),
            hit.record["title"].as_str().unwrap_or_default()
        );
    }

    // Leave the service as we found it.
    docs.ensure_collection_deleted().await?;
    Ok(())
}
