//! Offline tests of what the Azure DocumentDB store sends: index specs,
//! `cosmosSearch` pipelines, MQL filters and BSON documents. No server is
//! contacted (the driver connects lazily). Set `AF_DOCUMENTDB_TEST_URL`
//! (a DocumentDB connection string) to also run the live round trip.

#![cfg(feature = "experimental-vector-stores")]

use agent_framework_azure_documentdb::driver::bson::{doc, Bson, Document};
use agent_framework_azure_documentdb::{
    index_name, matching_index, AzureDocumentDbCollection, AzureDocumentDbStore,
    DocumentDbSearchOptions, IndexParameters, SCORE_FIELD,
};
use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterExpression, FilterGroup, FilterOperator, IndexKind,
    VectorCollection, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use serde_json::{json, Value};

async fn store() -> AzureDocumentDbStore {
    // tls=false is refused for real use; here the driver never connects.
    let client =
        agent_framework_azure_documentdb::driver::Client::with_uri_str("mongodb://127.0.0.1:1")
            .await
            .unwrap();
    AzureDocumentDbStore::from_client(client, "af_test").unwrap()
}

fn definition(index_kind: &str, distance: &str) -> VectorStoreCollectionDefinition {
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("body").with_type("str"),
        VectorStoreField::data("category")
            .with_type("str")
            .indexed(),
        VectorStoreField::data("number").with_type("int").indexed(),
        VectorStoreField::data("ratio").with_type("float").indexed(),
        VectorStoreField::data("flag").with_type("bool").indexed(),
        VectorStoreField::data("tags").with_type("list").indexed(),
        VectorStoreField::vector("embedding", 3)
            .with_storage_name("contentVector")
            .with_index_kind(IndexKind::new(index_kind))
            .with_distance_function(DistanceFunction::new(distance)),
    ])
    .unwrap()
}

async fn collection() -> AzureDocumentDbCollection {
    store()
        .await
        .collection("documents", definition("ivf_flat", "DEFAULT"))
        .unwrap()
}

#[tokio::test]
async fn index_specs_use_documentdb_commands() {
    let c = store()
        .await
        .collection("documents", definition("hnsw", "dot_prod"))
        .unwrap()
        .with_index_parameters(
            "embedding",
            IndexParameters {
                m: Some(12),
                ef_construction: Some(48),
                ..Default::default()
            },
        )
        .unwrap();
    let vector = c.definition().try_get_field("embedding").unwrap();
    let spec = c.vector_index_spec(vector).unwrap();
    assert_eq!(
        spec.get_document("key").unwrap(),
        &doc! { "contentVector": "cosmosSearch" }
    );
    assert_eq!(
        spec.get_document("cosmosSearchOptions").unwrap(),
        &doc! { "kind": "vector-hnsw", "dimensions": 3_i64, "similarity": "IP", "m": 12_i64, "efConstruction": 48_i64 }
    );
    assert_eq!(
        spec.get_str("name").unwrap(),
        index_name("documents", "contentVector", "vector")
    );
    assert_eq!(spec.get_str("name").unwrap().len(), "af_vector_".len() + 32);
    let category = c.definition().try_get_field("category").unwrap();
    assert_eq!(
        c.data_index_spec(category).get_document("key").unwrap(),
        &doc! { "category": 1 }
    );

    // Defaults per kind, and parameter ranges checked before I/O.
    let ivf = collection().await;
    let spec = ivf
        .vector_index_spec(ivf.definition().try_get_field("embedding").unwrap())
        .unwrap();
    assert_eq!(
        spec.get_document("cosmosSearchOptions").unwrap(),
        &doc! { "kind": "vector-ivf", "dimensions": 3_i64, "similarity": "COS", "numLists": 1_i64 }
    );
    let disk = store()
        .await
        .collection("documents", definition("disk_ann", "euclidean_distance"))
        .unwrap();
    let spec = disk
        .vector_index_spec(disk.definition().try_get_field("embedding").unwrap())
        .unwrap();
    assert_eq!(
        spec.get_document("cosmosSearchOptions").unwrap(),
        &doc! { "kind": "vector-diskann", "dimensions": 3_i64, "similarity": "L2", "maxDegree": 32_i64, "lBuild": 50_i64 }
    );
    for bad in [
        IndexParameters {
            m: Some(1),
            ..Default::default()
        },
        IndexParameters {
            m: Some(20),
            ef_construction: Some(30),
            ..Default::default()
        },
        IndexParameters {
            num_lists: Some(4),
            ..Default::default()
        },
    ] {
        let c = store()
            .await
            .collection("documents", definition("hnsw", "DEFAULT"))
            .unwrap();
        assert!(
            c.with_index_parameters("embedding", bad).is_err(),
            "{bad:?}"
        );
    }
}

#[tokio::test]
async fn index_compatibility() {
    let c = store()
        .await
        .collection("documents", definition("hnsw", "DEFAULT"))
        .unwrap();
    let expected = c
        .vector_index_spec(c.definition().try_get_field("embedding").unwrap())
        .unwrap();
    // Documented defaults omitted, lowercase similarity, another name.
    let mut existing = vec![doc! {
        "name": "administrator_name",
        "key": { "contentVector": "cosmosSearch" },
        "cosmosSearch": { "kind": "vector-hnsw", "dimensions": 3_i32, "similarity": "cos" },
    }];
    assert!(matching_index(&existing, &expected).unwrap());
    existing[0]
        .get_document_mut("cosmosSearch")
        .unwrap()
        .insert("m", 32);
    assert!(matching_index(&existing, &expected).is_err());
    assert!(!matching_index(&[doc! { "name": "_id_", "key": { "_id": 1 } }], &expected).unwrap());

    let filter = c.data_index_spec(c.definition().try_get_field("category").unwrap());
    assert!(matching_index(
        &[doc! { "name": "existing", "key": { "category": 1_i32 } }],
        &filter
    )
    .unwrap());
    for option in [
        doc! { "unique": true },
        doc! { "sparse": true },
        doc! { "partialFilterExpression": { "category": { "$exists": true } } },
        doc! { "expireAfterSeconds": 60 },
        doc! { "collation": { "locale": "en" } },
    ] {
        let mut index = doc! { "name": "existing", "key": { "category": 1 } };
        index.extend(option);
        assert!(matching_index(&[index], &filter).is_err());
    }
    // Same name, other key: incompatible.
    let name = filter.get_str("name").unwrap();
    assert!(matching_index(&[doc! { "name": name, "key": { "other": 1 } }], &filter).is_err());
}

#[tokio::test]
async fn filters_use_mongo_wire_operators() {
    let c = collection().await;
    let f = |e: FilterExpression| c.filter_document(&e);
    let cases: Vec<(FilterExpression, Document)> = vec![
        (
            Filter::eq("category", "db").unwrap().into(),
            doc! { "category": { "$eq": "db" } },
        ),
        (
            Filter::new("category", FilterOperator::Ne, Some(json!("db")))
                .unwrap()
                .into(),
            doc! { "$and": [ { "category": { "$exists": true } }, { "category": { "$ne": "db" } } ] },
        ),
        (
            Filter::eq("category", Value::Null).unwrap().into(),
            doc! { "$and": [ { "category": { "$exists": true } }, { "category": { "$eq": Bson::Null } } ] },
        ),
        (
            Filter::is_null("category").unwrap().into(),
            doc! { "$and": [ { "category": { "$exists": true } }, { "category": { "$eq": Bson::Null } } ] },
        ),
        (
            Filter::new("category", FilterOperator::IsNotNull, None)
                .unwrap()
                .into(),
            doc! { "$and": [ { "category": { "$exists": true } }, { "category": { "$ne": Bson::Null } } ] },
        ),
        (
            Filter::new("category", FilterOperator::Exists, None)
                .unwrap()
                .into(),
            doc! { "category": { "$exists": true } },
        ),
        (
            Filter::new("number", FilterOperator::Between, Some(json!([1, 3])))
                .unwrap()
                .into(),
            doc! { "number": { "$gte": 1_i64, "$lte": 3_i64 } },
        ),
        (
            Filter::new("ratio", FilterOperator::Lt, Some(json!(0.5)))
                .unwrap()
                .into(),
            doc! { "ratio": { "$lt": 0.5 } },
        ),
        (
            Filter::new("number", FilterOperator::NotIn, Some(json!([1, 2])))
                .unwrap()
                .into(),
            doc! { "$and": [ { "number": { "$exists": true } }, { "number": { "$ne": Bson::Null } }, { "number": { "$nin": [1_i64, 2_i64] } } ] },
        ),
        (
            Filter::new("tags", FilterOperator::Contains, Some(json!("azure")))
                .unwrap()
                .into(),
            doc! { "$and": [ { "tags": { "$exists": true } }, { "tags": { "$ne": Bson::Null } }, { "tags": { "$in": ["azure"] } } ] },
        ),
        (
            Filter::new(
                "tags",
                FilterOperator::ContainsAll,
                Some(json!(["azure", 1])),
            )
            .unwrap()
            .into(),
            doc! { "$and": [ { "tags": { "$exists": true } }, { "tags": { "$ne": Bson::Null } }, { "tags": { "$all": ["azure", 1_i64] } } ] },
        ),
        (
            Filter::eq("id", "one").unwrap().into(),
            doc! { "_id": { "$eq": "one" } },
        ),
    ];
    for (expression, expected) in cases {
        assert_eq!(f(expression.clone()).unwrap(), expected, "{expression:?}");
    }

    // Values stay values; NOT is $nor.
    let payload = "'; drop collection documents; --";
    let group = FilterGroup::and(vec![
        Filter::eq("category", payload).unwrap().into(),
        FilterGroup::not(Filter::eq("flag", false).unwrap().into()).unwrap(),
    ])
    .unwrap();
    assert_eq!(
        f(group).unwrap(),
        doc! { "$and": [ { "category": { "$eq": payload } }, { "$nor": [ { "flag": { "$eq": false } } ] } ] }
    );

    // Empty membership.
    let false_filter = doc! { "_id": { "$exists": false } };
    let present =
        doc! { "$and": [ { "number": { "$exists": true } }, { "number": { "$ne": Bson::Null } } ] };
    assert_eq!(
        f(Filter::new("number", FilterOperator::In, Some(json!([])))
            .unwrap()
            .into())
        .unwrap(),
        false_filter
    );
    assert_eq!(
        f(
            Filter::new("number", FilterOperator::NotIn, Some(json!([])))
                .unwrap()
                .into()
        )
        .unwrap(),
        present
    );
    assert_eq!(
        f(
            Filter::new("tags", FilterOperator::ContainsAny, Some(json!([])))
                .unwrap()
                .into()
        )
        .unwrap(),
        false_filter
    );
    // Membership never matches null operands.
    assert_eq!(
        f(Filter::new(
            "category",
            FilterOperator::In,
            Some(json!([null, "database"]))
        )
        .unwrap()
        .into())
        .unwrap(),
        doc! { "category": { "$in": ["database"] } }
    );
    assert_eq!(
        f(
            Filter::new("category", FilterOperator::In, Some(json!([null])))
                .unwrap()
                .into()
        )
        .unwrap(),
        false_filter
    );
    // Booleans and numbers stay distinct.
    assert_eq!(
        f(Filter::eq("number", true).unwrap().into()).unwrap(),
        false_filter
    );
    assert_eq!(
        f(Filter::eq("flag", 1).unwrap().into()).unwrap(),
        false_filter
    );
    assert_eq!(
        f(Filter::new("number", FilterOperator::Ne, Some(json!(true)))
            .unwrap()
            .into())
        .unwrap(),
        doc! { "number": { "$exists": true } }
    );

    // Refused before I/O, never dropped.
    for refused in [
        Filter::eq("body", "not-indexed").unwrap().into(),
        Filter::is_null("embedding").unwrap().into(),
        Filter::eq("category.nested", "x").unwrap().into(),
        Filter::new("category", FilterOperator::ContainsText, Some(json!("x")))
            .unwrap()
            .into(),
        Filter::new("category", FilterOperator::StartsWith, Some(json!("x")))
            .unwrap()
            .into(),
        Filter::new("tags", FilterOperator::Contains, Some(json!(["nested"])))
            .unwrap()
            .into(),
        Filter::new("tags", FilterOperator::In, Some(json!([["azure"]])))
            .unwrap()
            .into(),
        Filter::new("flag", FilterOperator::Gt, Some(json!(true)))
            .unwrap()
            .into(),
        Filter::new("number", FilterOperator::Gt, Some(json!("x")))
            .unwrap()
            .into(),
    ] {
        let e: FilterExpression = refused;
        assert!(f(e.clone()).is_err(), "{e:?}");
    }
}

#[tokio::test]
async fn search_pipeline_shape() {
    let c = collection().await;
    let pipeline = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(2)
                .with_skip(1)
                .with_filter(Filter::eq("category", "database").unwrap()),
            &DocumentDbSearchOptions {
                k: Some(10),
                n_probes: Some(1),
                score_threshold: Some(0.8),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        pipeline[0],
        doc! { "$search": { "cosmosSearch": {
            "path": "contentVector",
            "vector": [1.0, 0.0, 0.0],
            "k": 10_i64,
            "nProbes": 1_i64,
            "filter": { "category": { "$eq": "database" } },
        } } }
    );
    let project = pipeline[1].get_document("$project").unwrap();
    assert_eq!(
        project.get_document(SCORE_FIELD).unwrap(),
        &doc! { "$meta": "searchScore" }
    );
    assert!(!project.contains_key("contentVector"));
    assert_eq!(project.get_i32("_id"), Ok(1));
    assert_eq!(
        pipeline[2],
        doc! { "$match": { SCORE_FIELD: { "$gte": 0.8 } } }
    );
    assert_eq!(
        pipeline[3..],
        [doc! { "$skip": 1_i64 }, doc! { "$limit": 2_i64 }]
    );

    // Default k is skip + top; including vectors projects them.
    let plain = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(3).with_include_vectors(true),
            &DocumentDbSearchOptions::default(),
        )
        .unwrap();
    let search = plain[0]
        .get_document("$search")
        .unwrap()
        .get_document("cosmosSearch")
        .unwrap();
    assert_eq!(search.get_i64("k"), Ok(3));
    assert!(!search.contains_key("filter"));
    assert_eq!(
        plain[1]
            .get_document("$project")
            .unwrap()
            .get_i32("contentVector"),
        Ok(1)
    );
    assert_eq!(plain.len(), 3);
}

#[tokio::test]
async fn euclidean_threshold_is_a_maximum() {
    let c = store()
        .await
        .collection("documents", definition("ivf_flat", "euclidean_distance"))
        .unwrap();
    let pipeline = c
        .search_pipeline(
            &[1.0, 0.0, 0.0],
            &VectorSearchOptions::new(2).with_skip(1),
            &DocumentDbSearchOptions {
                score_threshold: Some(0.5),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(
        pipeline[2],
        doc! { "$match": { SCORE_FIELD: { "$lte": 0.5 } } }
    );
    assert_eq!(
        pipeline[3..],
        [doc! { "$skip": 1_i64 }, doc! { "$limit": 2_i64 }]
    );
}

#[tokio::test]
async fn search_option_bounds_and_service_names() {
    let s = store().await;
    let run = |kind: &str, native: DocumentDbSearchOptions| {
        let c = s
            .collection("documents", definition(kind, "DEFAULT"))
            .unwrap();
        c.search_pipeline(&[1.0, 0.0, 0.0], &VectorSearchOptions::new(3), &native)
            .map(|p| {
                p[0].get_document("$search")
                    .unwrap()
                    .get_document("cosmosSearch")
                    .unwrap()
                    .clone()
            })
    };
    let opts = |f: fn(&mut DocumentDbSearchOptions)| {
        let mut o = DocumentDbSearchOptions::default();
        f(&mut o);
        o
    };
    for (kind, native) in [
        ("ivf_flat", opts(|o| o.ef_search = Some(10))),
        ("ivf_flat", opts(|o| o.n_probes = Some(2))),
        ("hnsw", opts(|o| o.n_probes = Some(1))),
        ("disk_ann", opts(|o| o.ef_search = Some(40))),
        ("disk_ann", opts(|o| o.k = Some(1001))),
        (
            "disk_ann",
            opts(|o| {
                o.k = Some(50);
                o.l_search = Some(40);
            }),
        ),
        ("disk_ann", opts(|o| o.l_search = Some(9))),
        ("ivf_flat", opts(|o| o.k = Some(2))),
        ("ivf_flat", opts(|o| o.score_threshold = Some(f64::NAN))),
    ] {
        assert!(run(kind, native.clone()).is_err(), "{kind} {native:?}");
    }
    assert_eq!(
        run("hnsw", opts(|o| o.ef_search = Some(75)))
            .unwrap()
            .get_i64("efSearch"),
        Ok(75)
    );
    assert_eq!(
        run("disk_ann", opts(|o| o.l_search = Some(80)))
            .unwrap()
            .get_i64("lSearch"),
        Ok(80)
    );
    assert_eq!(
        run("disk_ann", DocumentDbSearchOptions::default())
            .unwrap()
            .get_i64("lSearch"),
        Ok(40),
        "lSearch defaults to max(40, k)"
    );
}

#[tokio::test]
async fn model_validation_happens_before_io() {
    let s = store().await;
    let with =
        |fields: Vec<VectorStoreField>| VectorStoreCollectionDefinition::new(fields).unwrap();
    let vector = || VectorStoreField::vector("v", 3);
    for (name, definition) in [
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("ObjectId"),
                vector(),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                VectorStoreField::data("s")
                    .with_type("str")
                    .with_storage_name(SCORE_FIELD),
                vector(),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                VectorStoreField::data("d").with_type("datetime"),
                vector(),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                VectorStoreField::data("t")
                    .with_type("str")
                    .full_text_indexed(),
                vector(),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                VectorStoreField::vector("v", 2001),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                vector().with_index_kind(IndexKind::new("flat")),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                vector().with_distance_function(DistanceFunction::new("cosine_distance")),
            ]),
        ),
        (
            "c",
            with(vec![
                VectorStoreField::key("id").with_type("str"),
                VectorStoreField::data("a.b").with_type("str"),
                vector(),
            ]),
        ),
        (
            "system.x",
            with(vec![VectorStoreField::key("id").with_type("str"), vector()]),
        ),
        (
            "a$b",
            with(vec![VectorStoreField::key("id").with_type("str"), vector()]),
        ),
    ] {
        assert!(s.collection(name, definition).is_err(), "{name}");
    }
}

#[tokio::test]
async fn documents_round_trip_and_fail_before_io() {
    let c = collection().await;
    let record = json!({
        "id": "one", "body": "Azure", "category": "database", "number": 1,
        "ratio": 1, "flag": true, "tags": ["a", {"k": 1}], "embedding": [1, 0, 0],
    });
    let document = c.to_document(&record).unwrap();
    assert_eq!(document.get_str("_id"), Ok("one"));
    assert_eq!(document.get_f64("ratio"), Ok(1.0));
    assert_eq!(
        document.get_array("contentVector").unwrap(),
        &vec![Bson::Int64(1), Bson::Int64(0), Bson::Int64(0)]
    );
    let mut returned = document.clone();
    returned.insert(SCORE_FIELD, 0.9);
    let back = c.from_document(returned, false).unwrap();
    assert_eq!(back["id"], "one");
    assert_eq!(back["tags"], json!(["a", {"k": 1}]));
    assert!(back.get("embedding").is_none());
    // A key of the wrong type coming back is a bad response.
    let mut wrong = document.clone();
    wrong.insert("_id", 7_i64);
    assert!(c.from_document(wrong, false).is_err());

    let base = record.as_object().unwrap().clone();
    let with = |k: &str, v: Value| {
        let mut r = base.clone();
        r.insert(k.into(), v);
        Value::Object(r)
    };
    let mut missing = base.clone();
    missing.remove("body");
    for bad in [
        Value::Object(missing),
        with("id", json!(1)),
        with("id", json!("x".repeat(2049))),
        with("number", json!(1.5)),
        with("flag", json!(1)),
        with("tags", json!({"a": 1})),
        with("tags", json!([{"$bad": 1}])),
        with("tags", json!([{"a.b": 1}])),
        with("embedding", json!([1, 0])),
        with("embedding", json!([1, true, 0])),
    ] {
        assert!(c.to_document(&bad).is_err(), "{bad}");
    }
    let huge = with("body", json!("x".repeat(17 * 1024 * 1024)));
    assert!(c.upsert(vec![record.clone(), huge]).await.is_err());
    // No keys: no I/O.
    assert!(c.get(vec![], false).await.unwrap().is_empty());
    c.delete(vec![]).await.unwrap();
}

#[tokio::test]
async fn connection_rules() {
    assert!(
        AzureDocumentDbStore::connect("mongodb://h/?tls=false", "db")
            .await
            .is_err()
    );
    assert!(
        AzureDocumentDbStore::connect("mongodb://h/?retryWrites=true", "db")
            .await
            .is_err()
    );
    assert!(AzureDocumentDbStore::connect("", "db").await.is_err());
    assert!(AzureDocumentDbStore::connect("mongodb://h/", "bad.name")
        .await
        .is_err());
    let ok = AzureDocumentDbStore::connect("mongodb://h/", "db")
        .await
        .unwrap();
    assert_eq!(ok.database_name(), "db");
}

#[tokio::test]
async fn live_round_trip() {
    let Ok(url) = std::env::var("AF_DOCUMENTDB_TEST_URL") else {
        return;
    };
    let store = AzureDocumentDbStore::connect(&url, "af_test")
        .await
        .unwrap();
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
    let hits = docs
        .search(
            vec![0.0, 1.0],
            &VectorSearchOptions::new(1).with_filter(Filter::gte("year", 2005).unwrap()),
        )
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], "b");
    docs.ensure_collection_deleted().await.unwrap();
}

#[tokio::test]
async fn list_fields_refuse_whole_value_equality() {
    let c = collection().await;
    for op in [FilterOperator::Eq, FilterOperator::Ne] {
        let e: FilterExpression = Filter::new("tags", op, Some(json!([1, 2]))).unwrap().into();
        let err = c.filter_document(&e).unwrap_err().to_string();
        assert!(err.contains("whole values"), "{err}");
    }
    // A null operand is still an is-null test.
    let e: FilterExpression = Filter::eq("tags", Value::Null).unwrap().into();
    assert!(c.filter_document(&e).is_ok());
}
