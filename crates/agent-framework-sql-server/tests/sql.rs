//! Offline tests of the statements the SQL Server store sends, and a live
//! round trip against a real SQL Server 2025 when `AF_SQL_SERVER_TEST_URL`
//! (an ADO connection string) is set; without it the live test passes
//! vacuously.

#![cfg(feature = "experimental-vector-stores")]

use agent_framework_core::vectors::{
    DistanceFunction, Filter, FilterGroup, FilterOperator, IndexKind, VectorSearchOptions,
    VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
};
use agent_framework_sql_server::{
    quote_identifier, Param, SqlServerCollection, SqlServerStore, MAX_PARAMETERS,
};
use serde_json::json;

fn store() -> SqlServerStore {
    SqlServerStore::connect("server=tcp:localhost,1433;user=sa;password=x").unwrap()
}

fn definition(distance: Option<&str>) -> VectorStoreCollectionDefinition {
    let mut vector = VectorStoreField::vector("embedding", 2);
    if let Some(d) = distance {
        vector = vector.with_distance_function(DistanceFunction::new(d));
    }
    VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("title").with_type("str").indexed(),
        VectorStoreField::data("year").with_type("int"),
        VectorStoreField::data("score").with_type("float"),
        VectorStoreField::data("when").with_type("datetime"),
        VectorStoreField::data("tags").with_type("list"),
        vector,
    ])
    .unwrap()
}

fn docs(distance: Option<&str>) -> SqlServerCollection {
    store().collection("docs", definition(distance)).unwrap()
}

#[test]
fn identifiers_are_bracket_quoted_and_bounded() {
    assert_eq!(quote_identifier("a]b").unwrap(), "[a]]b]");
    assert!(quote_identifier("").is_err());
    assert!(quote_identifier("a\0b").is_err());
    assert!(quote_identifier(&"x".repeat(129)).is_err());
    assert!(quote_identifier(&"x".repeat(128)).is_ok());
}

#[test]
fn create_table_uses_native_types_and_guarded_indexes() {
    let statements = docs(None).create_statements().unwrap();
    let create = &statements[0];
    assert!(
        create.sql.contains("CREATE TABLE [dbo].[docs]"),
        "{}",
        create.sql
    );
    assert!(create
        .sql
        .contains("[id] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NOT NULL PRIMARY KEY"));
    assert!(create
        .sql
        .contains("[title] NVARCHAR(450) COLLATE Latin1_General_100_BIN2 NULL"));
    assert!(create.sql.contains("[year] BIGINT NULL"));
    assert!(create.sql.contains("[score] FLOAT(53) NULL"));
    assert!(create.sql.contains("[when] DATETIME2(7) NULL"));
    assert!(create.sql.contains("[tags] NVARCHAR(MAX) NULL"));
    assert!(create.sql.contains("[embedding] VECTOR(2) NULL"));
    assert_eq!(
        create.params,
        [Param::Text("dbo".into()), Param::Text("docs".into())],
        "catalog values are bound"
    );
    assert_eq!(statements.len(), 2, "one index, for the indexed field");
    assert!(statements[1].sql.contains("CREATE INDEX [af_sql_"));
    assert!(statements[1].sql.contains("ON [dbo].[docs] ([title])"));
}

#[test]
fn unsupported_definitions_fail_before_io() {
    let s = store();
    let bad = |fields: Vec<VectorStoreField>| {
        s.collection("t", VectorStoreCollectionDefinition::new(fields).unwrap())
            .is_err()
    };
    let key = || VectorStoreField::key("id").with_type("int");
    assert!(bad(vec![key(), VectorStoreField::vector("v", 1999)]));
    assert!(bad(vec![
        key(),
        VectorStoreField::vector("v", 2).with_type("float16")
    ]));
    assert!(bad(vec![
        key(),
        VectorStoreField::vector("v", 2).with_distance_function(DistanceFunction::new("hamming"))
    ]));
    assert!(bad(vec![
        key(),
        VectorStoreField::vector("v", 2).with_index_kind(IndexKind::new("hnsw"))
    ]));
    assert!(bad(vec![
        key(),
        VectorStoreField::data("b").with_type("bytes")
    ]));
    assert!(bad(vec![
        key(),
        VectorStoreField::data("j").with_type("dict").indexed()
    ]));
    assert!(
        bad(vec![key(), VectorStoreField::data("x")]),
        "types are required"
    );
    assert!(bad(vec![VectorStoreField::key("id").with_type("float")]));
    assert!(s.clone().with_schema("").is_err());
    assert!(SqlServerStore::connect("  ").is_err());
}

#[test]
fn search_binds_the_vector_filters_and_paging_in_order() {
    let options = VectorSearchOptions::new(5)
        .with_skip(2)
        .with_filter(Filter::eq("title", "x'; DROP TABLE t; --").unwrap());
    let statement = docs(None).search_sql(&[1.0, 0.0], &options).unwrap();
    let sql = &statement.sql;
    assert!(
        sql.contains("VECTOR_DISTANCE('cosine', CAST(@P1 AS VECTOR(2)), t.[embedding])"),
        "{sql}"
    );
    assert!(sql.contains("CONVERT(VARBINARY(MAX), t.[title]) = CONVERT(VARBINARY(MAX), CONVERT(NVARCHAR(MAX), @P2))"), "{sql}");
    assert!(sql.ends_with(
        "ORDER BY d.[distance] ASC, t.[id] ASC OFFSET @P3 ROWS FETCH NEXT @P4 ROWS ONLY"
    ));
    assert!(!sql.contains("DROP TABLE"), "values are bound, not inlined");
    assert!(
        !sql.contains("t.[embedding] AS NVARCHAR"),
        "vectors excluded by default"
    );
    assert_eq!(
        statement.params,
        [
            Param::Text("[1.0,0.0]".into()),
            Param::Text("x'; DROP TABLE t; --".into()),
            Param::BigInt(2),
            Param::BigInt(5),
        ]
    );
    let with_vectors = docs(None)
        .search_sql(
            &[1.0, 0.0],
            &VectorSearchOptions::new(1).with_include_vectors(true),
        )
        .unwrap();
    assert!(with_vectors
        .sql
        .contains("CAST(t.[embedding] AS NVARCHAR(MAX))"));
}

#[test]
fn metrics_map_to_vector_distance() {
    for (distance, metric) in [
        ("cosine_similarity", "'cosine'"),
        ("euclidean_distance", "'euclidean'"),
        ("dot_prod", "'dot'"),
        ("negative_dot_prod", "'dot'"),
    ] {
        let sql = docs(Some(distance))
            .search_sql(&[1.0, 0.0], &VectorSearchOptions::new(1))
            .unwrap()
            .sql;
        assert!(
            sql.contains(&format!("VECTOR_DISTANCE({metric}")),
            "{distance}"
        );
        assert!(sql.contains("ORDER BY d.[distance] ASC"), "{distance}");
    }
    assert!(docs(None)
        .search_sql(&[1.0], &VectorSearchOptions::new(1))
        .is_err());
    assert!(docs(None)
        .search_sql(&[f32::NAN, 0.0], &VectorSearchOptions::new(1))
        .is_err());
}

fn filter_sql(
    filter: impl Into<agent_framework_core::vectors::FilterExpression>,
) -> (String, Vec<Param>) {
    let statement = docs(None)
        .search_sql(
            &[1.0, 0.0],
            &VectorSearchOptions::new(1).with_filter(filter),
        )
        .unwrap();
    let head = "WHERE t.[embedding] IS NOT NULL AND (";
    let start = statement.sql.find(head).unwrap() + head.len();
    let end = statement
        .sql
        .rfind(") AND d.[distance] IS NOT NULL")
        .unwrap();
    let predicate = statement.sql[start..end].to_string();
    let params = statement.params[1..statement.params.len() - 2].to_vec();
    (predicate, params)
}

fn filter_err(filter: impl Into<agent_framework_core::vectors::FilterExpression>) -> bool {
    docs(None)
        .search_sql(
            &[1.0, 0.0],
            &VectorSearchOptions::new(1).with_filter(filter),
        )
        .is_err()
}

#[test]
fn text_filters_escape_sql_server_wildcards() {
    let f = Filter::new(
        "title",
        FilterOperator::ContainsText,
        Some(json!("1_0%[a]!")),
    )
    .unwrap();
    let (sql, params) = filter_sql(f);
    assert!(
        sql.contains(
            "t.[title] IS NOT NULL AND t.[title] COLLATE Latin1_General_100_BIN2 LIKE @P2 ESCAPE '!'"
        ),
        "{sql}"
    );
    assert_eq!(params, [Param::Text("%1!_0!%[[]a]!!%".into())]);
    let starts = Filter::new("title", FilterOperator::StartsWith, Some(json!("ab"))).unwrap();
    assert_eq!(filter_sql(starts).1, [Param::Text("ab%".into())]);
    let ends = Filter::new("title", FilterOperator::EndsWith, Some(json!("ab"))).unwrap();
    assert_eq!(filter_sql(ends).1, [Param::Text("%ab".into())]);
    assert!(filter_err(
        Filter::new("year", FilterOperator::ContainsText, Some(json!("1"))).unwrap()
    ));
    assert!(filter_err(
        Filter::new("title", FilterOperator::ContainsText, Some(json!("a\0"))).unwrap()
    ));
}

#[test]
fn groups_membership_and_ordering_are_two_valued() {
    let either = FilterGroup::or(vec![
        Filter::eq("year", 2001).unwrap().into(),
        Filter::is_null("year").unwrap().into(),
    ])
    .unwrap();
    let (sql, params) = filter_sql(either);
    assert!(
        sql.contains("(t.[year] IS NOT NULL AND t.[year] = @P2) OR t.[year] IS NULL"),
        "{sql}"
    );
    assert_eq!(params, [Param::BigInt(2001)]);

    let not_in = Filter::new("year", FilterOperator::NotIn, Some(json!([1, 2]))).unwrap();
    let (sql, params) = filter_sql(not_in);
    assert!(sql.contains("t.[year] IS NOT NULL AND (NOT ("), "{sql}");
    assert_eq!(params, [Param::BigInt(1), Param::BigInt(2)]);
    let empty_in = Filter::new("year", FilterOperator::In, Some(json!([]))).unwrap();
    assert!(filter_sql(empty_in).0.contains("(1 = 0)"));

    let (sql, params) = filter_sql(Filter::gte("score", 1.5).unwrap());
    assert!(sql.contains("t.[score] >= @P2"), "{sql}");
    assert_eq!(params, [Param::Float(1.5)]);
    let between = Filter::new("year", FilterOperator::Between, Some(json!([1, 9]))).unwrap();
    assert!(filter_sql(between).0.contains("BETWEEN @P2 AND @P3"));
    let (sql, params) = filter_sql(Filter::gt("when", "2024-01-01T00:00:00+02:00").unwrap());
    assert!(
        sql.contains("SWITCHOFFSET(CAST(@P2 AS DATETIMEOFFSET(7)), '+00:00')"),
        "{sql}"
    );
    assert_eq!(params, [Param::Text("2024-01-01T00:00:00+02:00".into())]);

    assert!(
        filter_err(Filter::gt("title", "a").unwrap()),
        "no ordering on str"
    );
    assert!(
        filter_err(Filter::gt("when", "2024-01-01T00:00:00").unwrap()),
        "timezone"
    );
    assert!(filter_err(Filter::gt("year", true).unwrap()));
    assert!(filter_err(Filter::eq("tags", json!(["a"])).unwrap()));
    assert!(filter_err(Filter::eq("embedding", 1).unwrap()));
    assert!(filter_err(Filter::eq("nested.path", 1).unwrap()));
    assert!(filter_err(Filter::eq("missing", 1).unwrap()));
    assert!(filter_err(
        Filter::new("tags", FilterOperator::Contains, Some(json!("a"))).unwrap()
    ));
}

#[test]
fn mismatched_equality_never_matches_and_numbers_respect_column_range() {
    assert_eq!(filter_sql(Filter::eq("year", "2001").unwrap()).0, "1 = 0");
    assert_eq!(filter_sql(Filter::eq("title", 1).unwrap()).0, "1 = 0");
    assert_eq!(filter_sql(Filter::eq("year", true).unwrap()).0, "1 = 0");
    assert_eq!(
        filter_sql(Filter::eq("score", 1e20).unwrap()).1,
        [Param::Float(1e20)],
        "float columns accept integral values beyond bigint"
    );
    assert!(
        filter_err(Filter::eq("year", 1e20).unwrap()),
        "int columns refuse them"
    );
    let (sql, _) = filter_sql(Filter::new("year", FilterOperator::Exists, None).unwrap());
    assert_eq!(sql, "1 = 1");
}

#[test]
fn upsert_locks_then_updates_or_inserts_with_typed_placeholders() {
    let (statements, keys) = docs(None)
        .upsert_statements(&[json!({
            "id": "a", "title": "t", "year": 2001, "score": 1.5,
            "when": "2024-05-01T12:00:00Z", "tags": ["x"], "embedding": [1.0, 0.0]
        })])
        .unwrap();
    assert_eq!(keys, [json!("a")]);
    let sql = &statements[0].sql;
    assert!(sql.starts_with(
        "IF EXISTS (SELECT 1 FROM [dbo].[docs] WITH (UPDLOCK, HOLDLOCK) WHERE [id] = @P1)"
    ));
    assert!(
        sql.contains("UPDATE [dbo].[docs] SET [title] = @P2"),
        "{sql}"
    );
    assert!(
        sql.contains("[embedding] = CAST(@P7 AS VECTOR(2))"),
        "{sql}"
    );
    assert!(sql.contains("ELSE INSERT INTO [dbo].[docs] ([id], [title], [year], [score], [when], [tags], [embedding])"));
    assert_eq!(statements[0].params[5], Param::Text("[\"x\"]".into()));
    assert_eq!(statements[0].params[6], Param::Text("[1.0,0.0]".into()));

    let bad = |record: serde_json::Value| docs(None).upsert_statements(&[record]).is_err();
    assert!(bad(json!({"id": "trailing ", "embedding": [1.0, 0.0]})));
    assert!(bad(json!({"id": "x".repeat(451)})));
    assert!(bad(json!({"id": "a", "year": "2001"})));
    assert!(bad(json!({"id": "a", "when": "2024-05-01T12:00:00"})));
    assert!(bad(json!({"id": "a", "embedding": [1.0]})));
    assert!(bad(json!({"id": "a", "tags": {"not": "a list"}})));
    assert!(bad(json!({"title": "no key"})));
}

#[test]
fn the_parameter_budget_is_enforced() {
    let many: Vec<serde_json::Value> = (0..=MAX_PARAMETERS).map(|i| json!(i)).collect();
    let f = Filter::new("year", FilterOperator::In, Some(json!(many))).unwrap();
    assert!(filter_err(f));
}

#[tokio::test]
async fn live_round_trip() {
    let Ok(url) = std::env::var("AF_SQL_SERVER_TEST_URL") else {
        return;
    };
    let store = SqlServerStore::connect(&url).unwrap();
    let docs = store
        .get_collection(
            "af_sql_test_]';--",
            VectorStoreCollectionDefinition::new(vec![
                VectorStoreField::key("id").with_type("int"),
                VectorStoreField::data("title").with_type("str").indexed(),
                VectorStoreField::data("year").with_type("int"),
                VectorStoreField::vector("embedding", 2),
            ])
            .unwrap(),
        )
        .unwrap();
    docs.ensure_collection_deleted().await.unwrap();
    docs.ensure_collection_exists().await.unwrap();
    assert!(docs.collection_exists().await.unwrap());
    docs.upsert(vec![
        json!({"id": 1, "title": "north", "year": 2001, "embedding": [1.0, 0.0]}),
        json!({"id": 2, "title": "east ", "year": 2010, "embedding": [0.0, 1.0]}),
    ])
    .await
    .unwrap();
    let got = docs.get(vec![json!(2), json!(9)], true).await.unwrap();
    assert_eq!(got[0].as_ref().unwrap()["title"], "east ");
    assert!(got[1].is_none());
    let hits = docs
        .search(vec![1.0, 0.1], &VectorSearchOptions::new(2))
        .await
        .unwrap();
    assert_eq!(hits[0].record["id"], 1);
    let exact = docs
        .search(
            vec![1.0, 0.0],
            &VectorSearchOptions::new(2).with_filter(Filter::eq("title", "east").unwrap()),
        )
        .await
        .unwrap();
    assert!(exact.is_empty(), "trailing spaces are significant");
    docs.delete(vec![json!(1)]).await.unwrap();
    assert!(docs.get(vec![json!(1)], false).await.unwrap()[0].is_none());
    docs.ensure_collection_deleted().await.unwrap();
}
