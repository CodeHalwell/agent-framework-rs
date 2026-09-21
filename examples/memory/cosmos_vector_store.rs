//! Azure Cosmos DB for NoSQL as a **vector store**: create a container with
//! a vector policy, upsert records, and run a filtered vector search.
//!
//! This is the other half of `cosmos_store.rs`. That example persists a
//! conversation's history in a Cosmos container; this one stores embeddings
//! in one and searches them through the provider-agnostic `VectorCollection`
//! trait — the same trait `azure_ai_search_vector_store.rs` and
//! `InMemoryVectorStore` implement, so this code moves between them
//! unchanged.
//!
//! Two Cosmos-specific things the definition has to get right, both enforced
//! before the first request rather than at the service:
//!
//! * the key field is stored as `id` — a Cosmos item is addressed by that
//!   property, and the container partitions on it;
//! * the distance function is one Cosmos computes. `cosine_similarity`,
//!   `dot_prod`, and `euclidean_distance` map onto its three; anything else
//!   is refused rather than quietly ranked by a different metric.
//!
//! Skips gracefully unless configured:
//!   AZURE_COSMOS_ENDPOINT   e.g. https://<account>.documents.azure.com:443/
//!   AZURE_COSMOS_KEY        the account's primary key
//!   AZURE_COSMOS_DATABASE   an existing database (default: agent-framework)
//! plus OPENAI_API_KEY, which embeds the documents and the query.
//!
//! The account needs the **vector search** capability enabled
//! (`az cosmosdb update --capabilities EnableNoSQLVectorSearch ...`); without
//! it the service rejects the container's vector policy.
//!
//! ```bash
//! AZURE_COSMOS_ENDPOINT=https://my-account.documents.azure.com:443/ \
//! AZURE_COSMOS_KEY=... OPENAI_API_KEY=sk-... \
//! cargo run -p agent-framework-examples --example cosmos_vector_store
//! ```

use agent_framework::cosmos::CosmosVectorStore;
use agent_framework::prelude::*;
use agent_framework_core::client::EmbeddingClient;
use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterGroup, VectorCollection, VectorSearchOptions,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde_json::json;

/// `text-embedding-3-small`'s width.
const DIMENSIONS: usize = 1536;

#[tokio::main]
async fn main() -> Result<()> {
    let (Ok(endpoint), Ok(key), Ok(embedder)) = (
        std::env::var("AZURE_COSMOS_ENDPOINT"),
        std::env::var("AZURE_COSMOS_KEY"),
        OpenAIEmbeddingClient::from_env("text-embedding-3-small"),
    ) else {
        println!(
            "set AZURE_COSMOS_ENDPOINT, AZURE_COSMOS_KEY, and OPENAI_API_KEY to run this example"
        );
        return Ok(());
    };
    let database =
        std::env::var("AZURE_COSMOS_DATABASE").unwrap_or_else(|_| "agent-framework".to_string());

    // The collection's shape. `indexed()` is the default for a data field in
    // Cosmos (everything under `/*` is indexed); marking a field *not*
    // indexed excludes its path, and this connector then refuses to filter on
    // it rather than let the query scan the container.
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("title").with_type("str"),
        VectorStoreField::data("year").with_type("int"),
        VectorStoreField::vector("embedding", DIMENSIONS)
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
    ])?;

    let store = CosmosVectorStore::new(endpoint, key, database)?;
    // A fresh container per run. The cleanup at the end of this example
    // deletes it, and a fixed name would mean deleting whatever was already
    // sitting at that name in the account — which is not the example's to
    // destroy.
    let container = format!(
        "af-vector-demo-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or_default()
    );
    let docs = store.collection(&container, definition)?;

    // Creates the container with its vector and indexing policies, or — when
    // it already exists, which for the fresh name above it will not — checks
    // that what is there can serve the searches below. A vector policy cannot
    // be changed after creation, so a mismatch is reported here rather than
    // discovered as a wrongly-ranked result.
    docs.ensure_collection_exists().await?;

    let corpus = [
        ("d1", "Q1 earnings call", 2021),
        ("d2", "Q4 earnings call", 2024),
        ("d3", "Employee handbook", 2019),
    ];
    let embeddings = embedder
        .get_embeddings(
            corpus
                .iter()
                .map(|(_, title, _)| title.to_string())
                .collect(),
            None,
        )
        .await?;

    let records: Vec<_> = corpus
        .iter()
        .zip(embeddings.embeddings.iter())
        .map(|((id, title, year), embedding)| {
            json!({
                "id": id,
                "title": title,
                "year": year,
                "embedding": embedding.vector,
            })
        })
        .collect();
    docs.upsert(records).await?;
    println!("upserted {} records", corpus.len());

    let query = embedder
        .get_embeddings(vec!["quarterly results".to_string()], None)
        .await?;
    let query_vector = query.embeddings[0].vector.clone();

    // A portable filter: `year >= 2020 AND title contains "earnings"`, which
    // this connector translates into Cosmos SQL with every literal bound as a
    // parameter. The same expression runs against Azure AI Search or the
    // in-memory store without a change.
    let filter = FilterGroup::and(vec![
        Filter::gte("year", 2020)?.into(),
        Filter::contains_text("title", "earnings")?.into(),
    ])?;
    let hits = docs
        .search(
            query_vector.clone(),
            &VectorSearchOptions::new(3).with_filter(filter),
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

    // Point reads by key: the partition key *is* the id, so each of these is
    // a single-partition read rather than a query.
    let fetched = docs.get(vec![json!("d2"), json!("missing")], false).await?;
    println!("\npoint reads:");
    for (key, record) in ["d2", "missing"].iter().zip(fetched.iter()) {
        match record {
            Some(r) => println!("  {key}: {}", r["title"]),
            None => println!("  {key}: (not found)"),
        }
    }

    // Leave the account as we found it: this container is this run's own, so
    // deleting it takes nothing with it.
    docs.ensure_collection_deleted().await?;
    println!("\ndeleted {container}");
    Ok(())
}
