//! Tests against a real PostgreSQL with pgvector. Set `AF_POSTGRES_TEST_URL`
//! (for example `host=127.0.0.1 port=5432 user=postgres`) to run them;
//! without it they pass vacuously. Each test uses its own table.

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterGroup, FilterOperator, IndexKind, VectorSearchOptions,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_postgres::PostgresStore;
use serde_json::json;

async fn store() -> Option<PostgresStore> {
    let url = std::env::var("AF_POSTGRES_TEST_URL").ok()?;
    let store = PostgresStore::connect(&url).await.expect("connect");
    Some(store)
}

fn definition(distance: Option<&str>, index: Option<&str>) -> VectorStoreCollectionDefinition {
    let mut vector = VectorStoreField::vector("embedding", 2);
    if let Some(d) = distance {
        vector = vector.with_distance_function(DistanceFunction::new(d));
    }
    if let Some(i) = index {
        vector = vector.with_index_kind(IndexKind::new(i));
    }
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::data("year").with_type("int"),
        VectorStoreField::data("score").with_type("float"),
        VectorStoreField::data("tags").with_type("list"),
        VectorStoreField::data("meta").with_type("dict"),
        vector,
    ])
    .unwrap()
}

fn rows() -> Vec<serde_json::Value> {
    vec![
        json!({"id": 1, "title": "north", "year": 2001, "score": 1.5, "tags": ["a", "b"], "meta": {"k": 1}, "embedding": [1.0, 0.0]}),
        json!({"id": 2, "title": "east_100%", "year": 2010, "score": 2.0, "tags": ["b"], "meta": null, "embedding": [0.0, 1.0]}),
        json!({"id": 3, "title": "northeast", "year": null, "score": 3.0, "tags": [], "meta": {}, "embedding": [0.7, 0.7]}),
    ]
}

#[tokio::test]
async fn crud_filters_and_search() {
    let Some(store) = store().await else { return };
    let docs = store
        .get_collection("af_test_crud", definition(None, None))
        .unwrap();
    docs.ensure_collection_deleted().await.unwrap();
    assert!(!docs.collection_exists().await.unwrap());
    docs.ensure_collection_exists().await.unwrap();
    docs.ensure_collection_exists().await.unwrap();
    assert!(store
        .list_collection_names()
        .await
        .unwrap()
        .contains(&"af_test_crud".to_string()));

    assert_eq!(
        docs.upsert(rows()).await.unwrap(),
        [json!(1), json!(2), json!(3)]
    );
    // Upsert replaces; a key repeated in one batch keeps the last value.
    docs.upsert(vec![
        json!({"id": 1, "title": "x", "year": 1, "score": 0.0, "tags": [], "meta": null, "embedding": [1.0, 0.0]}),
        json!({"id": 1, "title": "north", "year": 2001, "score": 1.5, "tags": ["a", "b"], "meta": {"k": 1}, "embedding": [1.0, 0.0]}),
    ])
    .await
    .unwrap();

    let got = docs
        .get(vec![json!(3), json!(9), json!(1)], true)
        .await
        .unwrap();
    assert_eq!(got[0].as_ref().unwrap()["title"], "northeast");
    assert!(got[1].is_none());
    let one = got[2].as_ref().unwrap();
    assert_eq!(one["tags"], json!(["a", "b"]));
    assert_eq!(one["meta"], json!({"k": 1}));
    assert_eq!(one["score"], json!(1.5));
    assert_eq!(one["embedding"], json!([1.0, 0.0]));

    let search = |filter: Option<agent_framework_core::vectors::FilterExpression>| {
        let mut options = VectorSearchOptions::new(10);
        if let Some(f) = filter {
            options = options.with_filter(f);
        }
        options
    };
    let ids = |hits: &[agent_framework_core::vectors::VectorSearchResult]| {
        hits.iter()
            .map(|h| h.record["id"].as_i64().unwrap())
            .collect::<Vec<_>>()
    };

    let hits = docs.search(vec![1.0, 0.1], &search(None)).await.unwrap();
    assert_eq!(ids(&hits), [1, 3, 2]);
    assert!(
        hits[0].score.unwrap() < hits[1].score.unwrap(),
        "cosine distance ascends"
    );
    assert!(hits[0].record.get("embedding").is_none());

    let filtered = docs
        .search(
            vec![1.0, 0.1],
            &search(Some(Filter::gte("year", 2005).unwrap().into())),
        )
        .await
        .unwrap();
    assert_eq!(ids(&filtered), [2], "null years never compare");
    let text = Filter::new("title", FilterOperator::ContainsText, Some(json!("_100%"))).unwrap();
    assert_eq!(
        ids(&docs
            .search(vec![1.0, 0.0], &search(Some(text.into())))
            .await
            .unwrap()),
        [2]
    );
    let wildcard = Filter::new("title", FilterOperator::ContainsText, Some(json!("%"))).unwrap();
    assert_eq!(
        ids(&docs
            .search(vec![1.0, 0.0], &search(Some(wildcard.into())))
            .await
            .unwrap()),
        [2]
    );
    let contains = Filter::new("tags", FilterOperator::Contains, Some(json!("b"))).unwrap();
    let mut with_b = ids(&docs
        .search(vec![1.0, 0.0], &search(Some(contains.into())))
        .await
        .unwrap());
    with_b.sort();
    assert_eq!(with_b, [1, 2]);
    let either = FilterGroup::or(vec![
        Filter::eq("year", 2001).unwrap().into(),
        Filter::is_null("year").unwrap().into(),
    ])
    .unwrap();
    assert_eq!(
        ids(&docs
            .search(vec![1.0, 0.0], &search(Some(either)))
            .await
            .unwrap()),
        [1, 3]
    );
    let float_eq = Filter::eq("score", 2).unwrap();
    assert_eq!(
        ids(&docs
            .search(vec![1.0, 0.0], &search(Some(float_eq.into())))
            .await
            .unwrap()),
        [2]
    );
    let mismatch = Filter::eq("year", "2001").unwrap();
    assert!(docs
        .search(vec![1.0, 0.0], &search(Some(mismatch.into())))
        .await
        .unwrap()
        .is_empty());

    let page = docs
        .search(vec![1.0, 0.1], &VectorSearchOptions::new(1).with_skip(1))
        .await
        .unwrap();
    assert_eq!(ids(&page), [3]);

    docs.delete(vec![json!(1), json!(2)]).await.unwrap();
    assert!(docs.get(vec![json!(1)], false).await.unwrap()[0].is_none());
    assert!(docs.upsert(vec![json!({"id": 5, "title": 1, "year": 1, "score": 1, "tags": [], "meta": {}, "embedding": [1, 0]})]).await.is_err());
    docs.ensure_collection_deleted().await.unwrap();
}

#[tokio::test]
async fn similarity_scores_and_ann_indexes() {
    let Some(store) = store().await else { return };
    let docs = store
        .get_collection(
            "af_test_sim",
            definition(Some("cosine_similarity"), Some("hnsw")),
        )
        .unwrap();
    docs.ensure_collection_deleted().await.unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(rows()).await.unwrap();
    let hits = docs
        .search(vec![1.0, 0.0], &VectorSearchOptions::new(3))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], 1);
    assert!(
        (hits[0].score.unwrap() - 1.0).abs() < 1e-6,
        "similarity, not distance"
    );
    docs.ensure_collection_deleted().await.unwrap();

    let ivf = store
        .collection(
            "af_test_ivf",
            definition(Some("euclidean_distance"), Some("ivf_flat")),
        )
        .unwrap();
    use agent_framework_core::vectors::VectorCollection;
    ivf.ensure_collection_deleted().await.unwrap();
    let err = ivf.ensure_collection_exists().await.unwrap_err();
    assert!(
        err.to_string().contains("IVFFlat requires training data"),
        "{err}"
    );
    ivf.ensure_collection_exists_with(false).await.unwrap();
    ivf.upsert(rows()).await.unwrap();
    ivf.ensure_collection_exists().await.unwrap();
    let hits = ivf
        .search(vec![0.0, 1.0], &VectorSearchOptions::new(1))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], 2);
    ivf.ensure_collection_deleted().await.unwrap();
}
