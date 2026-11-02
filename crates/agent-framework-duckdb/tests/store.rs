//! Tests against an in-memory DuckDB database.

#![cfg(feature = "experimental-vector-stores")]

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterGroup, FilterOperator, IndexKind, VectorSearchOptions,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_duckdb::DuckDbStore;
use serde_json::json;

async fn store() -> Option<DuckDbStore> {
    Some(DuckDbStore::in_memory().expect("open"))
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
        VectorStoreField::data("title").with_type("str"),
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
    assert!(
        docs.search(vec![1.0, 0.0], &search(Some(contains.into())))
            .await
            .is_err(),
        "list containment is not supported, as upstream"
    );
    let prefix = Filter::new("title", FilterOperator::StartsWith, Some(json!("north"))).unwrap();
    let mut north = ids(&docs
        .search(vec![1.0, 0.0], &search(Some(prefix.into())))
        .await
        .unwrap());
    north.sort();
    assert_eq!(north, [1, 3]);
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
async fn similarity_and_dot_product_order_descending() {
    let store = store().await.unwrap();
    for (name, distance) in [("af_sim", "cosine_similarity"), ("af_dot", "dot_prod")] {
        let docs = store
            .get_collection(name, definition(Some(distance), None))
            .unwrap();
        docs.ensure_collection_exists().await.unwrap();
        docs.upsert(rows()).await.unwrap();
        let hits = docs
            .search(vec![1.0, 0.0], &VectorSearchOptions::new(3))
            .await
            .unwrap();
        assert_eq!(hits[0].record["id"], 1, "{distance}");
        assert!(
            hits[0].score.unwrap() >= hits[1].score.unwrap(),
            "{distance}"
        );
        docs.ensure_collection_deleted().await.unwrap();
        assert!(!docs.collection_exists().await.unwrap());
    }
}

#[tokio::test]
async fn euclidean_distance_and_unsupported_metrics() {
    let store = store().await.unwrap();
    let docs = store
        .get_collection("af_l2", definition(Some("euclidean_distance"), None))
        .unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(rows()).await.unwrap();
    let hits = docs
        .search(vec![0.0, 1.0], &VectorSearchOptions::new(1))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], 2);
    assert!(hits[0].score.unwrap().abs() < 1e-6);

    let bad = store.get_collection("af_bad", definition(Some("hamming"), None));
    let failed = match bad {
        Err(_) => true,
        Ok(c) => c.ensure_collection_exists().await.is_err(),
    };
    assert!(failed, "hamming is not supported");
}

#[tokio::test]
async fn the_search_statement_is_parameterised() {
    let store = store().await.unwrap();
    let docs = store.collection("af_sql", definition(None, None)).unwrap();
    let (sql, params) = docs
        .search_sql(
            &[1.0, 0.0],
            &VectorSearchOptions::new(5)
                .with_filter(Filter::eq("title", "x'; DROP TABLE t; --").unwrap()),
        )
        .unwrap();
    assert!(sql.contains("array_cosine_distance"), "{sql}");
    assert!(sql.contains("LIMIT ? OFFSET ?"), "{sql}");
    assert!(!sql.contains("DROP TABLE"), "values are bound, not inlined");
    assert!(params.iter().any(|p| p.contains("DROP TABLE")));
}

#[test]
fn explicit_data_indexes_are_refused() {
    let store = DuckDbStore::in_memory().unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    assert!(store.collection("af_idx", definition).is_err());
}

#[tokio::test]
async fn uuid_shaped_string_keys_stay_distinct() {
    let store = DuckDbStore::in_memory().unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("n").with_type("int"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    let docs = store.get_collection("af_str_keys", definition).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    let lower = "a1b2c3d4-0000-0000-0000-000000000001";
    let upper = "A1B2C3D4-0000-0000-0000-000000000001";
    docs.upsert(vec![
        json!({"id": lower, "n": 1, "embedding": [1.0, 0.0]}),
        json!({"id": upper, "n": 2, "embedding": [0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let got = docs
        .get(vec![json!(lower), json!(upper)], false)
        .await
        .unwrap();
    assert_eq!(got[0].as_ref().unwrap()["n"], 1);
    assert_eq!(got[1].as_ref().unwrap()["n"], 2);
}

#[tokio::test]
async fn integer_filters_beyond_2_53_are_exact() {
    let store = DuckDbStore::in_memory().unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("big").with_type("int"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    let docs = store.get_collection("af_big_ints", definition).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(vec![
        json!({"id": 1, "big": 9_007_199_254_740_992_i64, "embedding": [1.0, 0.0]}),
        json!({"id": 2, "big": 9_007_199_254_740_993_i64, "embedding": [0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let ids = |hits: Vec<agent_framework_core::vectors::VectorSearchResult>| {
        hits.iter()
            .map(|h| h.record["id"].as_i64().unwrap())
            .collect::<Vec<_>>()
    };
    let eq = Filter::eq("big", 9_007_199_254_740_993_i64).unwrap();
    let hits = docs
        .search(vec![1.0, 0.0], &VectorSearchOptions::new(5).with_filter(eq))
        .await
        .unwrap();
    assert_eq!(ids(hits), [2]);
    let gt = Filter::gt("big", 9_007_199_254_740_992_i64).unwrap();
    let hits = docs
        .search(vec![1.0, 0.0], &VectorSearchOptions::new(5).with_filter(gt))
        .await
        .unwrap();
    assert_eq!(ids(hits), [2]);
}

#[tokio::test]
async fn integral_float_filters_on_int_columns_compare_exactly() {
    let store = DuckDbStore::in_memory().unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("big").with_type("int"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    let docs = store.get_collection("af_float_ops", definition).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(vec![
        json!({"id": 1, "big": 9_007_199_254_740_992_i64, "embedding": [1.0, 0.0]}),
        json!({"id": 2, "big": 9_007_199_254_740_993_i64, "embedding": [0.0, 1.0]}),
        json!({"id": 3, "big": 2020, "embedding": [1.0, 1.0]}),
    ])
    .await
    .unwrap();
    let ids = |hits: Vec<agent_framework_core::vectors::VectorSearchResult>| {
        let mut ids: Vec<i64> = hits
            .iter()
            .map(|h| h.record["id"].as_i64().unwrap())
            .collect();
        ids.sort();
        ids
    };
    let docs = &docs;
    let search = |filter: Filter| async move {
        docs.search(
            vec![1.0, 0.0],
            &VectorSearchOptions::new(5).with_filter(filter),
        )
        .await
    };
    // 2^53 as a float names exactly 2^53, not its neighbour 2^53 + 1.
    let eq = Filter::eq("big", 9_007_199_254_740_992.0_f64).unwrap();
    assert_eq!(ids(search(eq).await.unwrap()), [1]);
    let member = Filter::any_of("big", vec![json!(9_007_199_254_740_992.0_f64)]).unwrap();
    assert_eq!(ids(search(member).await.unwrap()), [1]);
    let gte = Filter::gte("big", 9_007_199_254_740_993.0_f64).unwrap();
    // 2^53 + 1 is not representable: the float rounds to 2^53.
    assert_eq!(ids(search(gte).await.unwrap()), [1, 2]);
    let small = Filter::eq("big", 2020.0_f64).unwrap();
    assert_eq!(ids(search(small).await.unwrap()), [3]);
    // An integral float beyond BIGINT equals nothing, and is refused for
    // ordered comparisons.
    let huge = Filter::eq("big", 1e19_f64).unwrap();
    assert!(ids(search(huge).await.unwrap()).is_empty());
    assert!(search(Filter::gt("big", 1e19_f64).unwrap()).await.is_err());
}

#[tokio::test]
async fn integer_filters_on_float_columns_compare_exactly() {
    let store = DuckDbStore::in_memory().unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("int"),
        VectorStoreField::data("score").with_type("float"),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    let docs = store.get_collection("af_int_on_float", definition).unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(vec![
        json!({"id": 1, "score": 9_007_199_254_740_992.0_f64, "embedding": [1.0, 0.0]}),
        json!({"id": 2, "score": 2.0, "embedding": [0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let docs = &docs;
    let search = |filter: Filter| async move {
        docs.search(
            vec![1.0, 0.0],
            &VectorSearchOptions::new(5).with_filter(filter),
        )
        .await
    };
    let ids = |hits: Vec<agent_framework_core::vectors::VectorSearchResult>| {
        let mut ids: Vec<i64> = hits
            .iter()
            .map(|h| h.record["id"].as_i64().unwrap())
            .collect();
        ids.sort();
        ids
    };
    // 2^53 + 1 is not a double, so no stored float equals it.
    let eq = Filter::eq("score", 9_007_199_254_740_993_i64).unwrap();
    assert!(ids(search(eq).await.unwrap()).is_empty());
    let member = Filter::any_of("score", vec![json!(9_007_199_254_740_993_i64)]).unwrap();
    assert!(ids(search(member).await.unwrap()).is_empty());
    // A representable integer still compares numerically.
    assert_eq!(
        ids(search(Filter::eq("score", 2).unwrap()).await.unwrap()),
        [2]
    );
    assert_eq!(
        ids(
            search(Filter::eq("score", 9_007_199_254_740_992_i64).unwrap())
                .await
                .unwrap()
        ),
        [1]
    );
    // An ordered comparison it cannot make exactly is refused.
    assert!(
        search(Filter::gt("score", 9_007_199_254_740_993_i64).unwrap())
            .await
            .is_err()
    );
}
