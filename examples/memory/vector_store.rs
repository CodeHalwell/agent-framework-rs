//! Vector stores: describe a collection's fields once, then read, write, and
//! search it through a provider-agnostic trait pair.
//!
//! - `VectorStoreField` describes one field: the key, plain data, or a vector
//!   of a fixed dimensionality (with its index kind and distance function).
//! - `VectorStoreCollectionDefinition` collects the fields and validates the
//!   shape (exactly one key, at least one vector, no duplicate names).
//! - `VectorCollection` is a collection's data plane: create/drop, upsert,
//!   get, delete, search.
//! - `VectorStore` is the connection that hands collections out.
//!
//! `InMemoryVectorStore` implements both, with brute-force scoring -- right
//! for tests and for getting a pipeline working before a real store is
//! provisioned, wrong for a few million records. Swapping in a real provider
//! means changing the one line that constructs the store: everything below it
//! is written against `Box<dyn VectorCollection>`.
//!
//! Records are `serde_json::Value` objects keyed by *logical* field name, so
//! a typed struct converts at the boundary with `serde_json::to_value`. The
//! separate *storage* name is how you keep an idiomatic Rust field name over
//! a store whose column is called something else -- shown at the end.
//!
//! Runs fully offline: the "embeddings" here are hand-written vectors, so no
//! embedding model is needed. In real use they would come from an
//! `EmbeddingClient` (see `providers/openai_embeddings.rs`).
//!
//! ```bash
//! cargo run -p agent-framework-examples --example vector_store
//! ```

use agent_framework::prelude::*;
use agent_framework::vectors::{
    DistanceFunction, IndexKind, InMemoryVectorStore, VectorSearchOptions, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// The record type, as you would actually write it. `serde` handles the
/// conversion to and from the `Value` the collection stores.
#[derive(Debug, Serialize, Deserialize)]
struct Doc {
    id: String,
    title: String,
    body: String,
    embedding: Vec<f32>,
}

/// Stand-in for a real embedding model: three axes -- "rust-ness",
/// "database-ness", "cooking-ness" -- so the search results below are
/// predictable and checkable by eye.
fn embed(rust: f32, database: f32, cooking: f32) -> Vec<f32> {
    vec![rust, database, cooking]
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("== 1. describing the collection ==\n");

    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("title").full_text_indexed(),
        VectorStoreField::data("body"),
        VectorStoreField::vector("embedding", 3)
            .with_index_kind(IndexKind::new(IndexKind::HNSW))
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
    ])?;

    println!("  fields:        {:?}", definition.names());
    println!("  key field:     {}", definition.key_field().name);
    println!(
        "  vector fields: {:?}",
        definition
            .vector_fields()
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );
    println!(
        "  data fields:   {:?}",
        definition
            .data_fields()
            .iter()
            .map(|f| f.name.as_str())
            .collect::<Vec<_>>()
    );

    // The definition is validated on construction, so a malformed collection
    // fails here rather than on the first write.
    let bad = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::key("also_id"),
        VectorStoreField::vector("embedding", 3),
    ]);
    println!("  two key fields -> {:?}", bad.err().map(|e| e.to_string()));

    println!("\n== 2. writing records ==\n");

    let store = InMemoryVectorStore::new();
    let collection = store.get_collection("docs", definition.clone())?;
    collection.ensure_collection_exists().await?;
    println!("  collection exists: {}", collection.collection_exists().await?);

    let docs = vec![
        Doc {
            id: "d1".into(),
            title: "Ownership and borrowing".into(),
            body: "How the borrow checker keeps references valid.".into(),
            embedding: embed(1.0, 0.1, 0.0),
        },
        Doc {
            id: "d2".into(),
            title: "Async Rust with tokio".into(),
            body: "Tasks, the reactor, and why blocking calls hurt.".into(),
            embedding: embed(0.9, 0.2, 0.0),
        },
        Doc {
            id: "d3".into(),
            title: "Indexing strategies in Postgres".into(),
            body: "B-tree, GIN, and when a partial index pays off.".into(),
            embedding: embed(0.1, 1.0, 0.0),
        },
        Doc {
            id: "d4".into(),
            title: "A decent bolognese".into(),
            body: "Soffritto, milk, and four hours you will not regret.".into(),
            embedding: embed(0.0, 0.0, 1.0),
        },
    ];

    let records: Vec<_> = docs
        .iter()
        .map(serde_json::to_value)
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| Error::Serialization(e.to_string()))?;
    let keys = collection.upsert(records).await?;
    println!("  upserted keys: {keys:?}");

    println!("\n== 3. searching ==\n");

    // "Tell me about Rust" -- close to d1/d2, far from d3/d4.
    let query = embed(1.0, 0.0, 0.0);
    let results = collection
        .search(query.clone(), &VectorSearchOptions::new(3))
        .await?;
    for result in &results {
        println!(
            "  score={:.4}  {}",
            result.score.unwrap_or_default(),
            result.record["title"].as_str().unwrap_or_default()
        );
    }
    println!(
        "\n  cosine_similarity is higher-is-closer ({:?}), so the top row is the\n  \
         best match. A distance metric would rank the other way -- ask the\n  \
         field's DistanceFunction rather than assuming.",
        DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY).higher_is_closer()
    );

    // Options: page past the first hit, and ask for the vectors back (they
    // are omitted by default -- they are large and rarely wanted).
    println!("\n  with skip(1) and include_vectors:");
    let paged = collection
        .search(
            query,
            &VectorSearchOptions::new(2)
                .with_skip(1)
                .with_include_vectors(true),
        )
        .await?;
    for result in &paged {
        println!(
            "    {}  embedding={}",
            result.record["title"].as_str().unwrap_or_default(),
            result.record["embedding"]
        );
    }

    println!("\n== 4. get and delete ==\n");

    // `get` preserves the order of the keys you asked for and leaves a `None`
    // in the slot of anything missing, so the two lists stay aligned.
    let fetched = collection
        .get(vec![json!("d1"), json!("nope"), json!("d4")], false)
        .await?;
    for (key, record) in ["d1", "nope", "d4"].iter().zip(&fetched) {
        match record {
            Some(r) => println!("  {key:<5} -> {}", r["title"]),
            None => println!("  {key:<5} -> (not found)"),
        }
    }

    collection.delete(vec![json!("d4")]).await?;
    let after = collection.get(vec![json!("d4")], false).await?;
    println!("  after delete, d4 -> {:?}", after[0].is_some());

    // Round-tripping back to the typed struct is just serde.
    let d1 = collection.get(vec![json!("d1")], true).await?;
    let doc: Doc = serde_json::from_value(d1[0].clone().unwrap())
        .map_err(|e| Error::Serialization(e.to_string()))?;
    println!("  typed round trip: {} ({} dims)", doc.title, doc.embedding.len());

    println!("\n== 5. logical vs. storage names ==\n");

    // The store's column is `doc_text`; your Rust code says `body`. The
    // definition translates, so nothing above the boundary has to know.
    let renamed = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_storage_name("_id"),
        VectorStoreField::data("body").with_storage_name("doc_text"),
        VectorStoreField::vector("embedding", 3).with_storage_name("vec"),
    ])?;
    let logical = json!({ "id": "x1", "body": "hello", "embedding": [1.0, 0.0, 0.0] });
    let stored = renamed.to_storage(&logical)?;
    println!("  logical: {logical}");
    println!("  stored:  {stored}");
    println!("  back:    {}", renamed.from_storage(&stored, true)?);

    println!("\n== 6. listing and tearing down ==\n");
    println!("  collections: {:?}", store.list_collection_names().await?);
    collection.ensure_collection_deleted().await?;
    println!("  after drop:  {:?}", store.list_collection_names().await?);

    println!(
        "\nnote: everything after the `InMemoryVectorStore::new()` line is written\n\
         against the traits, so pointing this at a real store is a one-line change.\n\
         For a ready-made retrieval provider, see `memory/azure_ai_search.rs`\n\
         (a ContextProvider) and `memory/redis_memory.rs`."
    );

    Ok(())
}
