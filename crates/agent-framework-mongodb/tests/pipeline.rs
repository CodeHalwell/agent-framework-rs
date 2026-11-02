//! Offline tests of what the MongoDB store sends: index definitions,
//! `$vectorSearch` pipelines, prefilters and BSON documents. No server is
//! contacted (the driver connects lazily). Set `AF_MONGODB_TEST_URL` (an
//! Atlas cluster with Vector Search) to also run the live round trip.

#![cfg(feature = "experimental-vector-stores")]

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, FilterOperator, IndexKind,
    VectorCollection, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use agent_framework_mongodb::mongodb::bson::{doc, oid::ObjectId, Bson, Document};
use agent_framework_mongodb::{
    default_index_name, MongoDbCollection, MongoDbSearchOptions, MongoDbStore, SCORE_FIELD,
};
use serde_json::json;

async fn store() -> MongoDbStore {
    // Port 1: nothing listens, and nothing here performs I/O.
    MongoDbStore::connect("mongodb://127.0.0.1:1", "af_test")
        .await
        .unwrap()
}

fn definition() -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("number")
            .with_type("float")
            .indexed(),
        VectorStoreField::data("integer").with_type("int").indexed(),
        VectorStoreField::data("flag").with_type("bool").indexed(),
        VectorStoreField::data("tags").with_type("list"),
        VectorStoreField::data("when").with_type("datetime"),
        VectorStoreField::vector("text_vector", 3).with_storage_name("dense_text"),
        VectorStoreField::vector("image_vector", 2)
            .with_storage_name("dense_image")
            .with_distance_function(DistanceFunction::new("dot_prod")),
    ])
    .unwrap()
}

async fn collection() -> MongoDbCollection {
    store()
        .await
        .collection("docs", definition())
        .unwrap()
        .with_index_name("text_vector", "text_vector_index")
        .unwrap()
}

fn vector_field<'a>(c: &'a MongoDbCollection, name: &str) -> &'a VectorStoreField {
    c.definition().try_get_field(name).unwrap()
}

#[tokio::test]
async fn index_names_and_definitions() {
    let c = collection().await;
    assert_eq!(
        c.index_name(vector_field(&c, "text_vector")),
        "text_vector_index"
    );
    let generated = c.index_name(vector_field(&c, "image_vector"));
    assert!(generated.starts_with("af_dense_image_"), "{generated}");
    assert_eq!(generated.len(), "af_dense_image_".len() + 12);
    assert_eq!(generated, default_index_name("docs", "dense_image"));
    assert_eq!(
        c.index_definition(vector_field(&c, "text_vector")).unwrap(),
        doc! {
            "fields": [
                { "type": "vector", "path": "dense_text", "numDimensions": 3_i64, "similarity": "cosine" },
                { "type": "filter", "path": "_id" },
                { "type": "filter", "path": "number" },
                { "type": "filter", "path": "integer" },
                { "type": "filter", "path": "flag" },
            ]
        }
    );
    let image = c
        .index_definition(vector_field(&c, "image_vector"))
        .unwrap();
    assert_eq!(
        image.get_array("fields").unwrap()[0]
            .as_document()
            .unwrap()
            .get_str("similarity")
            .unwrap(),
        "dotProduct"
    );
    // Odd characters fold to '_', the readable part is capped at 32.
    let long = default_index_name("c", &format!("a.b{}", "x".repeat(40)));
    assert!(long.starts_with("af_a_bxxx"), "{long}");
    assert_eq!(long.len(), 3 + 32 + 1 + 12);
}

#[tokio::test]
async fn model_validation_happens_before_io() {
    let s = store().await;
    let reserved = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("other")
            .with_type("str")
            .with_storage_name("_id"),
        VectorStoreField::vector("v", 2),
    ])
    .unwrap();
    assert!(s.collection("c", reserved).is_err(), "_id is the key's");
    let too_big = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::vector("v", 8193),
    ])
    .unwrap();
    assert!(s.collection("c", too_big).is_err());
    let ivf = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::vector("v", 2).with_index_kind(IndexKind::new("ivf_flat")),
    ])
    .unwrap();
    assert!(s.collection("c", ivf).is_err());
    let indexed_list = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("tags").with_type("list").indexed(),
        VectorStoreField::vector("v", 2),
    ])
    .unwrap();
    assert!(s.collection("c", indexed_list).is_err());
    let full_text = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("t")
            .with_type("str")
            .full_text_indexed(),
        VectorStoreField::vector("v", 2),
    ])
    .unwrap();
    assert!(s.collection("c", full_text).is_err());
    // Two fields may not resolve to one index name.
    let c = s.collection("docs", definition()).unwrap();
    let image = c.index_name(c.definition().try_get_field("image_vector").unwrap());
    assert!(c.with_index_name("text_vector", image).is_err());
}

#[tokio::test]
async fn existing_indexes_compare_semantically() {
    let c = collection().await;
    let field = vector_field(&c, "text_vector");
    let mut fields = c
        .index_definition(field)
        .unwrap()
        .get_array("fields")
        .unwrap()
        .clone();
    fields.reverse();
    // Int32 dimensions and "quantization: none" are the same index.
    if let Some(Bson::Document(v)) = fields.last_mut() {
        v.insert("numDimensions", 3_i32);
        v.insert("quantization", "none");
    }
    let ready = doc! {
        "name": "text_vector_index",
        "type": "vectorSearch",
        "queryable": true,
        "status": "READY",
        "latestDefinition": { "fields": fields.clone() },
        "extraStatusMetadata": { "ignored": true },
    };
    assert!(c.validate_search_index(field, &ready).unwrap());
    let pending = doc! { "name": "text_vector_index", "status": "PENDING" };
    assert!(!c.validate_search_index(field, &pending).unwrap());
    let mut wrong = ready.clone();
    wrong.insert(
        "latestDefinition",
        doc! { "fields": [{ "type": "vector", "path": "dense_text", "numDimensions": 4, "similarity": "cosine" }] },
    );
    assert!(c.validate_search_index(field, &wrong).is_err());
    let mut text = ready;
    text.insert("type", "search");
    assert!(c.validate_search_index(field, &text).is_err());
}

#[tokio::test]
async fn ann_search_pipeline_shape() {
    let c = collection().await;
    let options = VectorSearchOptions::new(3)
        .with_vector_field_name("text_vector")
        .with_skip(2)
        .with_filter(Filter::eq("integer", 1).unwrap());
    let pipeline = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &options,
            &MongoDbSearchOptions {
                score_threshold: Some(0.75),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        pipeline[0],
        doc! {
            "$vectorSearch": {
                "index": "text_vector_index",
                "path": "dense_text",
                "queryVector": [1.0, 0.0, 0.0],
                "limit": 5_i64,
                "numCandidates": 100_i64,
                "filter": { "integer": { "$eq": 1_i64 } },
            }
        }
    );
    assert_eq!(
        pipeline[1],
        doc! { "$set": { SCORE_FIELD: { "$meta": "vectorSearchScore" } } }
    );
    let stages: Vec<&str> = pipeline
        .iter()
        .map(|s| s.keys().next().unwrap().as_str())
        .collect();
    assert_eq!(
        stages,
        [
            "$vectorSearch",
            "$set",
            "$match",
            "$skip",
            "$limit",
            "$project"
        ]
    );
    assert_eq!(
        pipeline[5],
        doc! { "$project": { "dense_text": 0, "dense_image": 0 } }
    );

    // Including vectors drops the projection; no skip drops $skip.
    let plain = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3)
                .with_vector_field_name("text_vector")
                .with_include_vectors(true),
            &MongoDbSearchOptions::default(),
        )
        .unwrap();
    let stages: Vec<&str> = plain
        .iter()
        .map(|s| s.keys().next().unwrap().as_str())
        .collect();
    assert_eq!(stages, ["$vectorSearch", "$set", "$limit"]);
}

#[tokio::test]
async fn candidate_windows() {
    let c = collection().await;
    let search = |options: VectorSearchOptions, mongo: MongoDbSearchOptions| {
        c.search_pipeline(&[1.0, 0.0, 0.0], &options, &mongo)
            .map(|p| p[0].get_document("$vectorSearch").unwrap().clone())
    };
    let exact = search(
        VectorSearchOptions::new(3)
            .with_vector_field_name("text_vector")
            .with_skip(1),
        MongoDbSearchOptions {
            exact: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(exact.get_bool("exact"), Ok(true));
    assert!(!exact.contains_key("numCandidates"));

    let explicit = search(
        VectorSearchOptions::new(501).with_vector_field_name("text_vector"),
        MongoDbSearchOptions {
            num_candidates: Some(10_000),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(explicit.get_i64("limit"), Ok(501));
    assert_eq!(explicit.get_i64("numCandidates"), Ok(10_000));

    let capped = search(
        VectorSearchOptions::new(600).with_vector_field_name("text_vector"),
        MongoDbSearchOptions::default(),
    )
    .unwrap();
    assert_eq!(capped.get_i64("numCandidates"), Ok(10_000));

    let large_exact = search(
        VectorSearchOptions::new(20_000).with_vector_field_name("text_vector"),
        MongoDbSearchOptions {
            exact: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(large_exact.get_i64("limit"), Ok(20_000));

    for (options, mongo) in [
        (
            VectorSearchOptions::new(10_001).with_vector_field_name("text_vector"),
            MongoDbSearchOptions::default(),
        ),
        (
            VectorSearchOptions::new(5).with_vector_field_name("text_vector"),
            MongoDbSearchOptions {
                num_candidates: Some(4),
                ..Default::default()
            },
        ),
        (
            VectorSearchOptions::new(5).with_vector_field_name("text_vector"),
            MongoDbSearchOptions {
                num_candidates: Some(10_001),
                ..Default::default()
            },
        ),
        (
            VectorSearchOptions::new(5).with_vector_field_name("text_vector"),
            MongoDbSearchOptions {
                exact: true,
                num_candidates: Some(10),
                ..Default::default()
            },
        ),
        (
            VectorSearchOptions::new(5).with_vector_field_name("text_vector"),
            MongoDbSearchOptions {
                score_threshold: Some(1.5),
                ..Default::default()
            },
        ),
    ] {
        assert!(search(options, mongo).is_err());
    }
    // The query vector must match the field's dimensions.
    assert!(c
        .search_pipeline(
            &[1.0, 0.0],
            &VectorSearchOptions::new(1).with_vector_field_name("text_vector"),
            &MongoDbSearchOptions::default()
        )
        .is_err());
}

#[tokio::test]
async fn prefilters() {
    let c = collection().await;
    let f = |e: FilterExpression| c.vector_filter(&e);
    assert_eq!(
        f(
            Filter::new("number", FilterOperator::Between, Some(json!([1, 3])))
                .unwrap()
                .into()
        )
        .unwrap(),
        doc! { "number": { "$gte": 1_i64, "$lte": 3_i64 } }
    );
    assert_eq!(
        f(Filter::eq("integer", true).unwrap().into()).unwrap(),
        doc! { "_id": { "$in": [] } },
        "a mismatched type matches nothing"
    );
    assert_eq!(
        f(Filter::eq("integer", 2.0).unwrap().into()).unwrap(),
        doc! { "integer": { "$eq": 2_i64 } },
        "an integral float is an integer"
    );
    assert_eq!(
        f(
            Filter::new("flag", FilterOperator::In, Some(json!([true, "x"])))
                .unwrap()
                .into()
        )
        .unwrap(),
        doc! { "flag": { "$in": [true] } }
    );
    assert_eq!(
        f(Filter::eq("id", "a").unwrap().into()).unwrap(),
        doc! { "_id": { "$eq": "a" } },
        "the key filters as _id"
    );
    let group = FilterGroup::or(vec![
        Filter::gte("integer", 1).unwrap().into(),
        Filter::new("number", FilterOperator::Lt, Some(json!(2.5)))
            .unwrap()
            .into(),
    ])
    .unwrap();
    assert_eq!(
        f(group).unwrap(),
        doc! { "$or": [ { "integer": { "$gte": 1_i64 } }, { "number": { "$lt": 2.5 } } ] }
    );

    // Not indexed, or not faithfully expressible: refused, never dropped.
    for refused in [
        Filter::eq("text", "hello").unwrap().into(),
        Filter::new("integer", FilterOperator::Ne, Some(json!(1)))
            .unwrap()
            .into(),
        Filter::new("integer", FilterOperator::NotIn, Some(json!([1])))
            .unwrap()
            .into(),
        Filter::is_null("integer").unwrap().into(),
        Filter::new("integer", FilterOperator::Exists, None)
            .unwrap()
            .into(),
        Filter::eq("integer", serde_json::Value::Null)
            .unwrap()
            .into(),
        Filter::new("integer", FilterOperator::Gt, Some(json!("x")))
            .unwrap()
            .into(),
        FilterGroup::not(Filter::eq("integer", 1).unwrap().into()).unwrap(),
        Filter::eq("text_vector", 1).unwrap().into(),
        Filter::eq("missing", 1).unwrap().into(),
    ] {
        let e: FilterExpression = refused;
        assert!(f(e.clone()).is_err(), "{e:?}");
    }

    // A provider prefilter is conjoined with the portable one.
    let pipeline = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(1)
                .with_vector_field_name("text_vector")
                .with_filter(Filter::eq("flag", true).unwrap())
                .with_provider_filter(r#"{"integer": {"$lt": 9}}"#),
            &MongoDbSearchOptions::default(),
        )
        .unwrap();
    assert_eq!(
        pipeline[0]
            .get_document("$vectorSearch")
            .unwrap()
            .get_document("filter")
            .unwrap(),
        &doc! { "$and": [ { "flag": { "$eq": true } }, { "integer": { "$lt": 9_i32 } } ] }
    );
}

#[tokio::test]
async fn object_id_keys_keep_native_identity() {
    let s = store().await;
    let c = s
        .collection(
            "oid",
            VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("ObjectId"),
                VectorStoreField::vector("vector", 2),
            ])
            .unwrap(),
        )
        .unwrap();
    let id = ObjectId::new();
    assert_eq!(
        c.vector_filter(&Filter::eq("id", id.to_hex()).unwrap().into())
            .unwrap(),
        doc! { "_id": { "$eq": id } }
    );
    let document = c
        .to_document(&json!({"id": id.to_hex(), "vector": [1, 0]}))
        .unwrap();
    assert_eq!(document.get_object_id("_id"), Ok(id));
    let back = c.from_document(document, true).unwrap();
    assert_eq!(back, json!({"id": id.to_hex(), "vector": [1.0, 0.0]}));
    assert!(c
        .to_document(&json!({"id": "not-an-oid", "vector": [1, 0]}))
        .is_err());
}

#[tokio::test]
async fn documents_round_trip_and_fail_before_io() {
    let c = collection().await;
    let record = json!({
        "id": "a",
        "text": "hello",
        "number": 1.5,
        "integer": 7,
        "flag": true,
        "tags": ["x", {"k": 1}],
        "when": "2024-05-01T12:00:00Z",
        "text_vector": [1, 0, 0],
        "image_vector": null,
        "undeclared": "dropped",
    });
    let document = c.to_document(&record).unwrap();
    assert_eq!(document.get_str("_id"), Ok("a"));
    assert!(!document.contains_key("id"));
    assert!(!document.contains_key("undeclared"));
    assert_eq!(document.get_i64("integer"), Ok(7));
    assert!(matches!(document.get("when"), Some(Bson::DateTime(_))));
    assert_eq!(
        document.get_array("dense_text").unwrap(),
        &vec![Bson::Double(1.0), Bson::Double(0.0), Bson::Double(0.0)]
    );
    let mut with_score: Document = document.clone();
    with_score.insert(SCORE_FIELD, 0.9);
    let back = c.from_document(with_score, false).unwrap();
    assert_eq!(back["id"], "a");
    assert_eq!(back["integer"], 7);
    assert_eq!(back["tags"], json!(["x", {"k": 1}]));
    assert_eq!(back["when"], "2024-05-01T12:00:00Z");
    assert!(back.get("text_vector").is_none());
    assert!(back.get(SCORE_FIELD).is_none());

    let bad = [
        json!({"id": 1, "text_vector": [1, 0, 0]}),
        json!({"id": "a", "integer": 1.5, "text_vector": [1, 0, 0]}),
        json!({"id": "a", "integer": true, "text_vector": [1, 0, 0]}),
        json!({"id": "a", "flag": "yes", "text_vector": [1, 0, 0]}),
        json!({"id": "a", "when": "yesterday", "text_vector": [1, 0, 0]}),
        json!({"id": "a", "text_vector": [1, 0]}),
        json!({"id": "a", "text_vector": [1, "x", 0]}),
        json!({"id": "a", "tags": "not a list", "text_vector": [1, 0, 0]}),
        json!({"id": "a", "integer": 9_223_372_036_854_775_808_u64, "text_vector": [1, 0, 0]}),
    ];
    for record in bad {
        assert!(c.to_document(&record).is_err(), "{record}");
    }
    // Oversized documents are refused before any write.
    let huge = json!({"id": "a", "text": "x".repeat(17 * 1024 * 1024), "text_vector": [1, 0, 0]});
    assert!(c.to_document(&huge).is_err());
    assert!(c.upsert(vec![huge]).await.is_err());
}

#[tokio::test]
async fn live_round_trip() {
    let Ok(url) = std::env::var("AF_MONGODB_TEST_URL") else {
        return;
    };
    let store = MongoDbStore::connect(&url, "af_test").await.unwrap();
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("year").with_type("int").indexed(),
        VectorStoreField::vector("embedding", 2),
    ])
    .unwrap();
    let docs = store.get_collection("af_live", definition).unwrap();
    docs.ensure_collection_deleted().await.unwrap();
    docs.ensure_collection_exists().await.unwrap();
    docs.upsert(vec![
        json!({"id": "a", "year": 2001, "embedding": [1.0, 0.0]}),
        json!({"id": "b", "year": 2010, "embedding": [0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let got = docs.get(vec![json!("b"), json!("z")], false).await.unwrap();
    assert_eq!(got[0].as_ref().unwrap()["year"], 2010);
    assert!(got[1].is_none());
    // Atlas indexes asynchronously; search results may lag the upsert.
    let _ = docs
        .search(
            vec![1.0, 0.0],
            &VectorSearchOptions::new(2).with_filter(Filter::gte("year", 2005).unwrap()),
        )
        .await
        .unwrap();
    docs.ensure_collection_deleted().await.unwrap();
}
