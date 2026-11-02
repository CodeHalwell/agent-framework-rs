//! Live tests for [`RedisVectorStore`] against a Redis server with Search
//! and JSON (Redis 8, or Redis Stack). Set `AF_REDIS_SEARCH_TEST_URL` to run
//! them; without it each test returns early. The plain `redis-server` the
//! other integration tests spawn has no Search module, so these never spawn
//! one.
#![cfg(feature = "experimental-vector-stores")]

use agent_framework_core::vectors::{
    Filter, VectorCollection, VectorSearchOptions, VectorSearchResult, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_redis::{RedisStorageType, RedisVectorStore};
use serde_json::json;

fn store() -> Option<RedisVectorStore> {
    let url = std::env::var("AF_REDIS_SEARCH_TEST_URL").ok()?;
    let namespace = format!("t{}", uuid::Uuid::new_v4().simple());
    Some(
        RedisVectorStore::new(&url)
            .unwrap()
            .with_namespace(namespace)
            .unwrap(),
    )
}

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::data("year").with_type("int").indexed(),
        VectorStoreField::data("tags").with_type("list").indexed(),
        VectorStoreField::vector("embedding", 3),
    ])
    .unwrap()
}

async fn round_trip(storage: RedisStorageType) {
    let Some(store) = store() else {
        eprintln!("AF_REDIS_SEARCH_TEST_URL not set; skipping");
        return;
    };
    let docs = store
        .collection_with_storage("docs", definition(), storage)
        .unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.ensure_collection_exists().await.unwrap();
    assert_eq!(store.list_collection_names().await.unwrap(), ["docs"]);
    let records = vec![
        json!({"id": "a", "title": "alpha", "year": 2020, "tags": ["x"], "embedding": [1.0, 0.0, 0.0]}),
        json!({"id": "b", "title": "beta", "year": 2024, "tags": ["x", "y"], "embedding": [0.0, 1.0, 0.0]}),
    ];
    docs.upsert(records.clone()).await.unwrap();
    let got = docs
        .get(vec![json!("b"), json!("missing")], true)
        .await
        .unwrap();
    assert_eq!(got, vec![Some(records[1].clone()), None]);

    let hits = docs
        .search(vec![0.9, 0.1, 0.0], &VectorSearchOptions::new(2))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], "a");
    assert_eq!(
        hits[0].score_kind.as_deref(),
        Some(VectorSearchResult::SCORE_KIND_DISTANCE)
    );
    assert!(hits[0].score < hits[1].score);

    let filtered = docs
        .search(
            vec![0.9, 0.1, 0.0],
            &VectorSearchOptions::new(2).with_filter(Filter::contains("tags", "y").unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(filtered.len(), 1);
    assert_eq!(filtered[0].record["id"], "b");

    docs.delete(vec![json!("a")]).await.unwrap();
    assert_eq!(docs.get(vec![json!("a")], false).await.unwrap(), vec![None]);
    docs.ensure_collection_deleted().await.unwrap();
    assert!(!docs.collection_exists().await.unwrap());
}

#[tokio::test]
async fn hash_collections_round_trip() {
    round_trip(RedisStorageType::Hash).await;
}

#[tokio::test]
async fn json_collections_round_trip() {
    round_trip(RedisStorageType::Json).await;
}

#[tokio::test]
async fn an_incompatible_existing_index_is_refused() {
    let Some(store) = store() else {
        return;
    };
    let docs = store.collection("docs", definition()).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    let narrower = store
        .collection(
            "docs",
            VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id"),
                VectorStoreField::vector("embedding", 3),
            ])
            .unwrap(),
        )
        .unwrap();
    assert!(narrower.ensure_collection_exists().await.is_err());
    docs.ensure_collection_deleted().await.unwrap();
}
