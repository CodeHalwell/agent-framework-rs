//! A MongoDB Atlas [`VectorStore`]: one collection per vector collection,
//! BSON documents keyed by `_id`, and `$vectorSearch` over Atlas Vector
//! Search indexes.
//!
//! Port of upstream `agent_framework_mongodb` (`MongoDBCollection`,
//! `MongoDBStore`) over the official `mongodb` driver (rustls TLS).
//!
//! ```no_run
//! use agent_framework_core::vectors::{
//!     Filter, VectorSearchOptions, VectorStore, VectorStoreCollectionDefinition, VectorStoreField,
//! };
//! use agent_framework_mongodb::MongoDbStore;
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! // MONGODB_URI and MONGODB_DATABASE_NAME (and optionally MONGODB_APP_NAME).
//! let store = MongoDbStore::from_env().await?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     // Only indexed fields can prefilter a vector search.
//!     VectorStoreField::data("year").with_type("int").indexed(),
//!     VectorStoreField::vector("embedding", 3),
//! ])?;
//! let notes = store.get_collection("notes", definition)?;
//! // Creates the collection and its vector search index, then waits until
//! // Atlas reports the index queryable.
//! notes.ensure_collection_exists().await?;
//! let hits = notes
//!     .search(
//!         vec![0.1, 0.2, 0.3],
//!         &VectorSearchOptions::new(5).with_filter(Filter::gte("year", 2020)?),
//!     )
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```
//!
//! # Model
//!
//! - The key is stored as `_id`; it is `str`, `int` or `ObjectId` (a
//!   24-character hex string in records).
//! - Data types: `str`, `int` (BSON `Int64`), `float`, `bool`, `list` (and
//!   `tuple` / `set` / `Sequence`), `dict`, `ObjectId` and `datetime`
//!   (RFC 3339 text in records, BSON `DateTime` in the store). Untyped
//!   fields convert as plain JSON.
//! - Each vector field gets one Atlas Vector Search index, named
//!   `af_<field>_<hash>` unless [`MongoDbCollection::with_index_name`]
//!   names it. Indexed data fields (and `_id`) become its `filter` paths.
//!   An existing index whose definition differs is an error, not a
//!   migration.
//! - Search is `$vectorSearch` (approximate by default, with
//!   `numCandidates = min(20 × (skip + top), 10000)`; see
//!   [`MongoDbSearchOptions`] for exact search). Scores are Atlas's
//!   normalized `vectorSearchScore`, higher is closer
//!   ([`VectorSearchResult::SCORE_KIND_RELEVANCE`]).
//! - Portable filters become `$vectorSearch` prefilters: `eq`, `in`, the
//!   ordered operators and `between`, combined with `and` / `or`, over
//!   indexed `str`, `int`, `float`, `bool`, `ObjectId` and `datetime`
//!   fields. Anything else is refused rather than dropped.
//!   [`VectorSearchOptions::provider_filter`] takes an MQL prefilter as a
//!   JSON object, conjoined with the portable one.
//!
//! # Divergences
//!
//! - The `bytes` type is refused: records are JSON, which has no bytes.
//! - Upserts are one `replaceOne` (with `upsert`) per record, in order,
//!   rather than one `bulkWrite`; a failure leaves the earlier records
//!   written.
//! - There is no filtered (non-vector) retrieval, so upstream's `$expr`
//!   retrieval-filter compiler has no counterpart.
//! - Auto-generated `ObjectId` keys are not offered: the Rust field model
//!   has no `is_auto_generated`.

#![cfg(feature = "experimental-vector-stores")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]

pub mod codec;

use std::collections::HashMap;
use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use async_trait::async_trait;
use futures::TryStreamExt;
use mongodb::bson::{doc, oid::ObjectId, Bson, DateTime, Document};
use mongodb::{Client, Database, SearchIndexModel, SearchIndexType};
use serde_json::Value;

pub use mongodb;

use codec::{bson_to_json, json_to_bson, KeyRules};

/// The field the search score is copied into before projection.
pub const SCORE_FIELD: &str = "__af_vector_search_score";
/// Atlas's cap on `numCandidates` (and on an approximate search window).
pub const MAX_NUM_CANDIDATES: u32 = 10_000;
const VECTOR_FILTER_TYPES: &[&str] = &["str", "int", "float", "bool", "ObjectId", "datetime"];
const LIST_TYPES: &[&str] = &["list", "tuple", "set", "Sequence"];
const INDEX_TERMINAL_FAILURES: &[&str] = &["FAILED", "DELETING"];

fn service(context: &str, e: mongodb::error::Error) -> Error {
    Error::service(format!("MongoDB {context} failed: {e}"))
}

fn invalid_response(message: impl Into<String>) -> Error {
    Error::service(message.into())
}

fn non_empty(value: &str, name: &str) -> Result<()> {
    if value.trim().is_empty() || value.contains('\0') {
        return Err(Error::Configuration(format!(
            "{name} must be a non-empty string without NUL characters."
        )));
    }
    Ok(())
}

/// The Atlas `similarity` for a field's distance function.
fn similarity(field: &VectorStoreField) -> Result<&'static str> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(|d| d.as_str())
            .unwrap_or("DEFAULT")
        {
            "DEFAULT" | "cosine_similarity" | "cosine_distance" => "cosine",
            "dot_prod" => "dotProduct",
            "euclidean_distance" => "euclidean",
            other => {
                return Err(Error::Configuration(format!(
                    "MongoDB distance function '{other}' is not supported."
                )))
            }
        },
    )
}

/// Upstream's default index name: `af_<readable>_<sha256[:12]>`.
pub fn default_index_name(collection_name: &str, field_storage_name: &str) -> String {
    let digest = codec::sha256_hex(&format!("{collection_name}\0{field_storage_name}"));
    let readable: String = field_storage_name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(32)
        .collect();
    let readable = if readable.is_empty() {
        "vector".to_string()
    } else {
        readable
    };
    format!("af_{readable}_{}", &digest[..12])
}

fn storage_name(field: &VectorStoreField) -> &str {
    if field.field_type == FieldType::Key {
        "_id"
    } else {
        field.effective_storage_name()
    }
}

/// Whether a JSON value can stand for a value of the field's declared type.
fn matches_type(field: &VectorStoreField, value: &Value) -> bool {
    match (field.type_.as_deref(), value) {
        (_, Value::Null) => true,
        (Some("str"), Value::String(_)) => true,
        (Some("int"), Value::Number(n)) => {
            n.is_i64()
                || n.as_f64()
                    .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
        }
        (Some("float"), Value::Number(_)) => true,
        (Some("bool"), Value::Bool(_)) => true,
        (Some("ObjectId"), Value::String(s)) => ObjectId::parse_str(s).is_ok(),
        (Some("datetime"), Value::String(s)) => DateTime::parse_rfc3339_str(s).is_ok(),
        (Some("list"), Value::Array(_)) => true,
        (Some("dict"), Value::Object(_)) => true,
        _ => false,
    }
}

/// A filter operand as BSON, typed for the field.
fn filter_operand(field: &VectorStoreField, value: &Value) -> Result<Bson> {
    let name = format!("Filter value for '{}'", field.name);
    match (field.type_.as_deref(), value) {
        (Some("int"), Value::Number(n)) if !n.is_i64() => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            if !f.is_finite() || f.fract() != 0.0 {
                return Err(Error::Configuration(format!(
                    "Filter value for integer field '{}' must be an integer.",
                    field.name
                )));
            }
            if !(-9.223_372_036_854_776e18..9.223_372_036_854_776e18).contains(&f) {
                return Err(Error::Configuration(format!(
                    "Filter value for integer field '{}' exceeds signed 64-bit range.",
                    field.name
                )));
            }
            Ok(Bson::Int64(f as i64))
        }
        (Some("ObjectId"), Value::String(s)) => ObjectId::parse_str(s)
            .map(Bson::ObjectId)
            .map_err(|_| Error::Configuration(format!("{name} must be an ObjectId."))),
        (Some("datetime"), Value::String(s)) => DateTime::parse_rfc3339_str(s)
            .map(Bson::DateTime)
            .map_err(|_| Error::Configuration(format!("{name} must be an RFC 3339 datetime."))),
        _ => json_to_bson(value, &name, KeyRules::NoNul),
    }
}

/// A key value as BSON.
fn key_to_bson(field: &VectorStoreField, value: &Value) -> Result<Bson> {
    let expected = field.type_.as_deref().unwrap_or("str, int, or ObjectId");
    let mismatch = || {
        Error::Configuration(format!(
            "MongoDB key values must match the declared '{expected}' key type."
        ))
    };
    match (field.type_.as_deref(), value) {
        (Some("str"), Value::String(s)) => Ok(Bson::String(s.clone())),
        (Some("int"), Value::Number(n)) => n.as_i64().map(Bson::Int64).ok_or_else(mismatch),
        (Some("ObjectId"), Value::String(s)) => ObjectId::parse_str(s)
            .map(Bson::ObjectId)
            .map_err(|_| mismatch()),
        _ => Err(mismatch()),
    }
}

/// A data value as BSON, checked against the declared type.
fn data_to_bson(field: &VectorStoreField, value: &Value) -> Result<Bson> {
    if value.is_null() {
        return Ok(Bson::Null);
    }
    let name = format!("MongoDB field '{}'", field.name);
    let mismatch = |what: &str| Error::Configuration(format!("{name} must contain {what}."));
    let kind = field.type_.as_deref();
    match kind {
        Some("str") => value
            .as_str()
            .map(|s| Bson::String(s.to_string()))
            .ok_or_else(|| mismatch("a string")),
        Some("int") => value
            .as_i64()
            .filter(|_| value.is_number())
            .map(Bson::Int64)
            .ok_or_else(|| mismatch("a signed 64-bit integer")),
        Some("float") => value
            .as_f64()
            .filter(|f| f.is_finite())
            .map(Bson::Double)
            .ok_or_else(|| mismatch("a finite number")),
        Some("bool") => value
            .as_bool()
            .map(Bson::Boolean)
            .ok_or_else(|| mismatch("a boolean")),
        Some(k) if LIST_TYPES.contains(&k) => match value {
            Value::Array(_) => json_to_bson(value, &field.name, KeyRules::NoNul),
            _ => Err(mismatch("a collection")),
        },
        Some("dict") => match value {
            Value::Object(_) => json_to_bson(value, &field.name, KeyRules::NoNul),
            _ => Err(mismatch("a mapping")),
        },
        Some("ObjectId") => value
            .as_str()
            .and_then(|s| ObjectId::parse_str(s).ok())
            .map(Bson::ObjectId)
            .ok_or_else(|| mismatch("an ObjectId hex string")),
        Some("datetime") => value
            .as_str()
            .and_then(|s| DateTime::parse_rfc3339_str(s).ok())
            .map(Bson::DateTime)
            .ok_or_else(|| mismatch("an RFC 3339 datetime")),
        _ => json_to_bson(value, &field.name, KeyRules::NoNul),
    }
}

/// Compile a portable filter into a `$vectorSearch` prefilter.
fn vector_filter(
    definition: &VectorStoreCollectionDefinition,
    expression: &FilterExpression,
) -> Result<Document> {
    let filter: &Filter = match expression {
        FilterExpression::Group(group) => {
            let op = match group.operator {
                FilterGroupOperator::Not => {
                    return Err(Error::Configuration(
                        "MongoDB vector prefilters do not support portable NOT semantics.".into(),
                    ))
                }
                FilterGroupOperator::And => "$and",
                FilterGroupOperator::Or => "$or",
            };
            let children = group
                .filters
                .iter()
                .map(|c| vector_filter(definition, c).map(Bson::Document))
                .collect::<Result<Vec<_>>>()?;
            return Ok(doc! { op: children });
        }
        FilterExpression::Condition(f) => f,
    };
    if filter.field_name.contains('.') {
        return Err(Error::Configuration(
            "MongoDB portable filters support declared top-level fields only.".into(),
        ));
    }
    let field = definition
        .try_get_field(&filter.field_name)
        .ok_or_else(|| {
            Error::Configuration(format!(
                "Unknown MongoDB filter field '{}'.",
                filter.field_name
            ))
        })?;
    if field.field_type == FieldType::Vector {
        return Err(Error::Configuration(
            "MongoDB portable filters do not operate on vector fields.".into(),
        ));
    }
    if field.field_type == FieldType::Data && field.is_indexed != Some(true) {
        return Err(Error::Configuration(format!(
            "MongoDB vector prefilter field '{}' must be indexed.",
            field.name
        )));
    }
    let kind = field.type_.as_deref().unwrap_or_default();
    if !VECTOR_FILTER_TYPES.contains(&kind) {
        return Err(Error::Configuration(format!(
            "MongoDB vector prefilters do not support declared type '{kind}'."
        )));
    }
    let name = storage_name(field);
    let value = filter.value.clone().unwrap_or(Value::Null);
    let null_refused =
        || Error::Configuration("MongoDB vector prefilters do not index null values.".into());
    Ok(match &filter.operator {
        FilterOperator::Eq => {
            if value.is_null() {
                return Err(null_refused());
            }
            if matches_type(field, &value) {
                doc! { name: { "$eq": filter_operand(field, &value)? } }
            } else {
                // A value of another type can never match.
                doc! { "_id": { "$in": [] } }
            }
        }
        FilterOperator::In => {
            let items = value.as_array().cloned().unwrap_or_default();
            if items.iter().any(Value::is_null) {
                return Err(null_refused());
            }
            let prepared = items
                .iter()
                .filter(|i| matches_type(field, i))
                .map(|i| filter_operand(field, i))
                .collect::<Result<Vec<_>>>()?;
            doc! { name: { "$in": prepared } }
        }
        op @ (FilterOperator::Gt
        | FilterOperator::Gte
        | FilterOperator::Lt
        | FilterOperator::Lte
        | FilterOperator::Between) => {
            let operands = if *op == FilterOperator::Between {
                value.as_array().cloned().unwrap_or_default()
            } else {
                vec![value]
            };
            if operands
                .iter()
                .any(|o| o.is_null() || !matches_type(field, o))
            {
                return Err(Error::Configuration(format!(
                    "MongoDB ordered vector prefilter operands must match field '{}'.",
                    field.name
                )));
            }
            let prepared = operands
                .iter()
                .map(|o| filter_operand(field, o))
                .collect::<Result<Vec<_>>>()?;
            let condition = match op {
                FilterOperator::Between => {
                    doc! { "$gte": prepared[0].clone(), "$lte": prepared[1].clone() }
                }
                FilterOperator::Gt => doc! { "$gt": prepared[0].clone() },
                FilterOperator::Gte => doc! { "$gte": prepared[0].clone() },
                FilterOperator::Lt => doc! { "$lt": prepared[0].clone() },
                _ => doc! { "$lte": prepared[0].clone() },
            };
            doc! { name: condition }
        }
        other => {
            return Err(Error::Configuration(format!(
                "MongoDB vector prefilters do not faithfully support portable operator '{other:?}'."
            )))
        }
    })
}

/// A canonical, order-independent form of a vector search index
/// definition, for comparing an existing index with the expected one.
fn semantic_index_definition(definition: &Document) -> Result<Vec<String>> {
    let fields = definition.get_array("fields").map_err(|_| {
        Error::Configuration(
            "MongoDB vector search index definition must contain a fields array.".into(),
        )
    })?;
    let mut normalized = Vec::new();
    for raw in fields {
        let field = raw.as_document().ok_or_else(|| {
            Error::Configuration("MongoDB vector search index fields must be documents.".into())
        })?;
        let kind = field.get("type").cloned().unwrap_or(Bson::Null);
        let path = field.get("path").cloned().unwrap_or(Bson::Null);
        let mut semantic: Vec<(String, Value)> = match kind.as_str() {
            Some("filter") => vec![
                ("path".into(), bson_to_json(path)),
                ("type".into(), bson_to_json(kind)),
            ],
            Some("vector") => {
                let mut items = vec![
                    (
                        "numDimensions".into(),
                        bson_to_json(field.get("numDimensions").cloned().unwrap_or(Bson::Null)),
                    ),
                    ("path".into(), bson_to_json(path)),
                    (
                        "similarity".into(),
                        bson_to_json(field.get("similarity").cloned().unwrap_or(Bson::Null)),
                    ),
                    ("type".into(), bson_to_json(kind)),
                ];
                match field.get("quantization") {
                    None | Some(Bson::Null) => {}
                    Some(Bson::String(s)) if s == "none" => {}
                    Some(q) => items.push(("quantization".into(), bson_to_json(q.clone()))),
                }
                items
            }
            _ => field
                .iter()
                .map(|(k, v)| (k.clone(), bson_to_json(v.clone())))
                .collect(),
        };
        semantic.sort_by(|a, b| a.0.cmp(&b.0));
        normalized.push(
            semantic
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    normalized.sort();
    Ok(normalized)
}

/// MongoDB-specific search options (upstream's `operation_options`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MongoDbSearchOptions {
    /// Exact (ENN) search instead of approximate; `num_candidates` must
    /// then be unset.
    pub exact: bool,
    /// Approximate-search candidates, `1..=10000` and at least
    /// `skip + top`. Defaults to `min(20 × (skip + top), 10000)`.
    pub num_candidates: Option<u32>,
    /// Keep only hits whose normalized score is at least this, in `[0, 1]`.
    pub score_threshold: Option<f64>,
}

/// A MongoDB collection holding vector records. See the [crate docs](crate).
#[derive(Clone)]
pub struct MongoDbCollection {
    database: Database,
    collection: mongodb::Collection<Document>,
    name: String,
    definition: VectorStoreCollectionDefinition,
    index_names: HashMap<String, String>,
    index_timeout: Duration,
    poll_interval: Duration,
}

impl std::fmt::Debug for MongoDbCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MongoDbCollection")
            .field("name", &self.name)
            .field("database", &self.database.name())
            .finish_non_exhaustive()
    }
}

impl MongoDbCollection {
    fn new(
        database: Database,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        non_empty(name, "collection_name")?;
        if name.contains('$') {
            return Err(Error::Configuration(
                "MongoDB collection names cannot contain '$'.".into(),
            ));
        }
        let key = definition.key_field();
        if !matches!(key.type_.as_deref(), Some("str" | "int" | "ObjectId")) {
            return Err(Error::Configuration(
                "MongoDB keys must be declared str, int or ObjectId.".into(),
            ));
        }
        for field in definition.fields() {
            let stored = field.effective_storage_name();
            if stored.is_empty()
                || stored.contains('\0')
                || stored.contains('.')
                || stored.starts_with('$')
                || stored == SCORE_FIELD
            {
                return Err(Error::Configuration(
                    "MongoDB storage names must be top-level names without NUL, dots, '$' prefixes, or reserved names."
                        .into(),
                ));
            }
            if field.field_type != FieldType::Key && stored == "_id" {
                return Err(Error::Configuration(
                    "MongoDB reserves storage name '_id' for the key field.".into(),
                ));
            }
            match field.field_type {
                FieldType::Vector => {
                    let dims = field.dimensions.unwrap_or_default();
                    if !(1..=8192).contains(&dims) {
                        return Err(Error::Configuration(
                            "MongoDB vector dimensions must be between 1 and 8192.".into(),
                        ));
                    }
                    match field.index_kind.as_ref().map(|k| k.as_str()) {
                        None | Some("default" | "hnsw") => {}
                        Some(other) => {
                            return Err(Error::Configuration(format!(
                                "MongoDB vector index kind '{other}' is not supported."
                            )))
                        }
                    }
                    if !matches!(
                        field.type_.as_deref(),
                        None | Some("float" | "float16" | "float32" | "float64")
                    ) {
                        return Err(Error::Configuration(format!(
                            "MongoDB vector field '{}' must be a float vector.",
                            field.name
                        )));
                    }
                    similarity(field)?;
                }
                FieldType::Data => {
                    if field.is_full_text_indexed == Some(true) {
                        return Err(Error::Configuration(
                            "MongoDB Search text indexes and keyword-hybrid search are not supported."
                                .into(),
                        ));
                    }
                    if field.type_.as_deref() == Some("bytes") {
                        return Err(Error::Configuration(format!(
                            "MongoDB field '{}': bytes cannot be carried in a JSON record.",
                            field.name
                        )));
                    }
                    if field.is_indexed == Some(true)
                        && !VECTOR_FILTER_TYPES
                            .contains(&field.type_.as_deref().unwrap_or_default())
                    {
                        return Err(Error::Configuration(format!(
                            "MongoDB vector filter indexes do not support declared type '{}'.",
                            field.type_.as_deref().unwrap_or_default()
                        )));
                    }
                }
                FieldType::Key => {}
            }
        }
        let collection = database.collection::<Document>(name);
        let this = Self {
            database,
            collection,
            name: name.to_string(),
            definition,
            index_names: HashMap::new(),
            index_timeout: Duration::from_secs(120),
            poll_interval: Duration::from_millis(500),
        };
        this.check_unique_index_names()?;
        Ok(this)
    }

    fn check_unique_index_names(&self) -> Result<()> {
        let mut seen = std::collections::HashSet::new();
        for field in self.definition.vector_fields() {
            let name = self.index_name(field);
            if !seen.insert(name.clone()) {
                return Err(Error::Configuration(format!(
                    "MongoDB vector index names must be unique: {name}."
                )));
            }
        }
        Ok(())
    }

    /// Name the search index of `vector_field` (upstream's
    /// `mongodb.index_name` annotation).
    pub fn with_index_name(
        mut self,
        vector_field: &str,
        index_name: impl Into<String>,
    ) -> Result<Self> {
        let index_name = index_name.into();
        non_empty(&index_name, "mongodb.index_name")?;
        let field = self
            .definition
            .try_get_field(vector_field)
            .filter(|f| f.field_type == FieldType::Vector)
            .ok_or_else(|| {
                Error::Configuration(format!("'{vector_field}' is not a vector field."))
            })?;
        self.index_names.insert(field.name.clone(), index_name);
        self.check_unique_index_names()?;
        Ok(self)
    }

    /// How long [`VectorCollection::ensure_collection_exists`] waits for a
    /// new index to become queryable (default 120 s), and how often it
    /// polls (default 0.5 s).
    pub fn with_index_wait(mut self, timeout: Duration, poll_interval: Duration) -> Result<Self> {
        if timeout.is_zero() || poll_interval.is_zero() {
            return Err(Error::Configuration(
                "index_timeout and poll_interval must be positive.".into(),
            ));
        }
        self.index_timeout = timeout;
        self.poll_interval = poll_interval;
        Ok(self)
    }

    /// The search index name of a vector field.
    pub fn index_name(&self, field: &VectorStoreField) -> String {
        self.index_names
            .get(&field.name)
            .cloned()
            .unwrap_or_else(|| default_index_name(&self.name, field.effective_storage_name()))
    }

    /// The vector search index definition for a vector field (test seam).
    pub fn index_definition(&self, field: &VectorStoreField) -> Result<Document> {
        let mut fields = vec![
            Bson::Document(doc! {
                "type": "vector",
                "path": field.effective_storage_name(),
                "numDimensions": field.dimensions.unwrap_or_default() as i64,
                "similarity": similarity(field)?,
            }),
            Bson::Document(doc! { "type": "filter", "path": "_id" }),
        ];
        for data in self.definition.data_fields() {
            if data.is_indexed == Some(true) {
                fields.push(Bson::Document(
                    doc! { "type": "filter", "path": data.effective_storage_name() },
                ));
            }
        }
        Ok(doc! { "fields": fields })
    }

    /// A record as the BSON document stored for it (test seam).
    pub fn to_document(&self, record: &Value) -> Result<Document> {
        let stored = self.definition.to_storage(record)?;
        let mut document = Document::new();
        for field in self.definition.fields() {
            let name = field.effective_storage_name();
            let value = stored.get(name).cloned();
            match field.field_type {
                FieldType::Key => {
                    let value = value.ok_or_else(|| {
                        Error::Configuration(format!(
                            "Record is missing the key field '{}'.",
                            field.name
                        ))
                    })?;
                    document.insert("_id", key_to_bson(field, &value)?);
                }
                FieldType::Vector => {
                    let value = value.unwrap_or(Value::Null);
                    let bson = if value.is_null() {
                        Bson::Null
                    } else {
                        Bson::Array(
                            codec::dense_vector(field, &value)?
                                .into_iter()
                                .map(Bson::Double)
                                .collect(),
                        )
                    };
                    document.insert(name, bson);
                }
                FieldType::Data => {
                    document.insert(name, data_to_bson(field, &value.unwrap_or(Value::Null))?);
                }
            }
        }
        codec::check_document_size(&document, "MongoDB record")?;
        Ok(document)
    }

    /// A stored document back as a record (test seam).
    pub fn from_document(&self, mut document: Document, include_vectors: bool) -> Result<Value> {
        document.remove(SCORE_FIELD);
        let id = document
            .remove("_id")
            .ok_or_else(|| invalid_response("MongoDB response is missing the '_id' field."))?;
        let mut stored = match codec::document_to_json(document) {
            Value::Object(map) => map,
            _ => unreachable!("a document converts to an object"),
        };
        stored.insert(
            self.definition.key_field_storage_name().to_string(),
            bson_to_json(id),
        );
        self.definition
            .from_storage(&Value::Object(stored), include_vectors)
    }

    fn projection(&self, include_vectors: bool) -> Option<Document> {
        if include_vectors {
            return None;
        }
        let mut projection = Document::new();
        for field in self.definition.vector_fields() {
            projection.insert(field.effective_storage_name(), 0);
        }
        (!projection.is_empty()).then_some(projection)
    }

    /// The `$vectorSearch` aggregation pipeline for a search (test seam).
    pub fn search_pipeline(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        mongo: &MongoDbSearchOptions,
    ) -> Result<Vec<Document>> {
        options.validate()?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                Error::Configuration(
                    "Select a vector field from the MongoDB collection definition.".into(),
                )
            })?;
        let window = options.top.saturating_add(options.skip);
        if window > i64::MAX as usize {
            return Err(Error::Configuration(
                "MongoDB vector search skip + top must fit a signed 64-bit integer.".into(),
            ));
        }
        if mongo.exact && mongo.num_candidates.is_some() {
            return Err(Error::Configuration(
                "num_candidates cannot be supplied with exact search.".into(),
            ));
        }
        if !mongo.exact && window > MAX_NUM_CANDIDATES as usize {
            return Err(Error::Configuration(
                "MongoDB ANN search requires skip + top to be at most 10000; use exact search for a larger window."
                    .into(),
            ));
        }
        let num_candidates = match mongo.num_candidates {
            Some(n) => {
                if !(1..=MAX_NUM_CANDIDATES).contains(&n) {
                    return Err(Error::Configuration(
                        "num_candidates must be between 1 and 10000.".into(),
                    ));
                }
                if (n as usize) < window {
                    return Err(Error::Configuration(
                        "num_candidates must be greater than or equal to skip + top.".into(),
                    ));
                }
                Some(n as i64)
            }
            None if mongo.exact => None,
            None => Some((20 * window).min(MAX_NUM_CANDIDATES as usize) as i64),
        };
        if let Some(t) = mongo.score_threshold {
            if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                return Err(Error::Configuration(
                    "MongoDB score_threshold must be a finite number between 0 and 1.".into(),
                ));
            }
        }
        let query = codec::query_vector(field, vector)?;
        let portable = options
            .filter
            .as_ref()
            .map(|f| vector_filter(&self.definition, f))
            .transpose()?;
        let raw = options
            .provider_filter
            .as_deref()
            .filter(|r| !r.trim().is_empty())
            .map(codec::provider_filter)
            .transpose()?;
        let native = match (portable, raw) {
            (Some(a), Some(b)) => Some(doc! { "$and": [a, b] }),
            (a, b) => a.or(b),
        };
        let mut search = doc! {
            "index": self.index_name(field),
            "path": field.effective_storage_name(),
            "queryVector": query,
            "limit": window as i64,
        };
        match num_candidates {
            None => {
                search.insert("exact", true);
            }
            Some(n) => {
                search.insert("numCandidates", n);
            }
        }
        if let Some(native) = native {
            search.insert("filter", native);
        }
        let mut pipeline = vec![
            doc! { "$vectorSearch": search },
            doc! { "$set": { SCORE_FIELD: { "$meta": "vectorSearchScore" } } },
        ];
        if let Some(t) = mongo.score_threshold {
            pipeline.push(doc! { "$match": { SCORE_FIELD: { "$gte": t } } });
        }
        if options.skip > 0 {
            pipeline.push(doc! { "$skip": options.skip as i64 });
        }
        pipeline.push(doc! { "$limit": options.top as i64 });
        if let Some(projection) = self.projection(options.include_vectors) {
            pipeline.push(doc! { "$project": projection });
        }
        Ok(pipeline)
    }

    /// Search with MongoDB-specific options.
    pub async fn search_with(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
        mongo: &MongoDbSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let pipeline = self.search_pipeline(&vector, options, mongo)?;
        let documents: Vec<Document> = self
            .collection
            .aggregate(pipeline)
            .await
            .map_err(|e| service("search", e))?
            .try_collect()
            .await
            .map_err(|e| service("search", e))?;
        documents
            .into_iter()
            .map(|document| {
                let score = match document.get(SCORE_FIELD) {
                    Some(Bson::Double(f)) if f.is_finite() => *f,
                    Some(Bson::Int32(i)) => f64::from(*i),
                    Some(Bson::Int64(i)) => *i as f64,
                    _ => {
                        return Err(invalid_response(
                            "MongoDB search result is missing a finite vectorSearchScore.",
                        ))
                    }
                };
                Ok(VectorSearchResult {
                    record: self.from_document(document, options.include_vectors)?,
                    score: Some(score),
                    score_kind: Some(VectorSearchResult::SCORE_KIND_RELEVANCE.to_string()),
                })
            })
            .collect()
    }

    async fn list_search_index(&self, name: &str) -> Result<Option<Document>> {
        let indexes: Vec<Document> = self
            .collection
            .list_search_indexes()
            .name(name)
            .await
            .map_err(|e| service("listSearchIndexes", e))?
            .try_collect()
            .await
            .map_err(|e| service("listSearchIndexes", e))?;
        if indexes.len() > 1 {
            return Err(invalid_response(format!(
                "MongoDB returned multiple vector search indexes named '{name}'."
            )));
        }
        Ok(indexes.into_iter().next())
    }

    /// A portable filter as a `$vectorSearch` prefilter (test seam).
    pub fn vector_filter(&self, expression: &FilterExpression) -> Result<Document> {
        vector_filter(&self.definition, expression)
    }

    /// Check an existing search index (as `listSearchIndexes` returns it)
    /// against the model: `Ok(true)` when it matches, `Ok(false)` when its
    /// definition is not yet available, an error when it is incompatible.
    /// Field order and status metadata do not matter.
    pub fn validate_search_index(
        &self,
        field: &VectorStoreField,
        index: &Document,
    ) -> Result<bool> {
        let name = self.index_name(field);
        let type_ok = matches!(index.get("type"), None | Some(Bson::Null))
            || index.get_str("type") == Ok("vectorSearch");
        if index.get_str("name") != Ok(name.as_str()) || !type_ok {
            return Err(Error::Configuration(format!(
                "Existing MongoDB search index '{name}' has an incompatible type."
            )));
        }
        let definition = index
            .get_document("latestDefinition")
            .or_else(|_| index.get_document("definition"));
        let Ok(definition) = definition else {
            return Ok(false);
        };
        if semantic_index_definition(definition)?
            != semantic_index_definition(&self.index_definition(field)?)?
        {
            return Err(Error::Configuration(format!(
                "Existing MongoDB vector search index '{name}' does not match the model definition."
            )));
        }
        Ok(true)
    }

    async fn wait_for_search_index(&self, field: &VectorStoreField) -> Result<()> {
        let deadline = tokio::time::Instant::now() + self.index_timeout;
        let name = self.index_name(field);
        loop {
            if let Some(index) = self.list_search_index(&name).await? {
                let available = self.validate_search_index(field, &index)?;
                if available && index.get_bool("queryable") == Ok(true) {
                    return Ok(());
                }
                let status = index.get_str("status").unwrap_or_default().to_uppercase();
                if INDEX_TERMINAL_FAILURES.contains(&status.as_str()) {
                    return Err(Error::service(format!(
                        "MongoDB vector search index '{name}' entered terminal status '{status}'."
                    )));
                }
            }
            let now = tokio::time::Instant::now();
            if now >= deadline {
                return Err(Error::service(format!(
                    "MongoDB vector search index '{name}' was not queryable within {:?}.",
                    self.index_timeout
                )));
            }
            tokio::time::sleep(self.poll_interval.min(deadline - now)).await;
        }
    }

    /// Create the collection and, when `create_indexes`, its vector search
    /// indexes, waiting for each to become queryable. Existing indexes are
    /// validated, never migrated.
    pub async fn ensure_collection_exists_with(&self, create_indexes: bool) -> Result<()> {
        if !self.collection_exists().await? {
            if let Err(e) = self.database.create_collection(&self.name).await {
                if !self.collection_exists().await? {
                    return Err(service("createCollection", e));
                }
            }
        }
        if !create_indexes {
            return Ok(());
        }
        for field in self.definition.vector_fields() {
            let name = self.index_name(field);
            let mut existing = self.list_search_index(&name).await?;
            if existing.is_none() {
                let model = SearchIndexModel::builder()
                    .definition(self.index_definition(field)?)
                    .name(name.clone())
                    .index_type(SearchIndexType::VectorSearch)
                    .build();
                if let Err(e) = self.collection.create_search_index(model).await {
                    existing = self.list_search_index(&name).await?;
                    if existing.is_none() {
                        return Err(service("createSearchIndexes", e));
                    }
                }
            }
            if let Some(index) = &existing {
                self.validate_search_index(field, index)?;
            }
            self.wait_for_search_index(field).await?;
        }
        Ok(())
    }

    fn keys_to_bson(&self, keys: &[Value]) -> Result<Vec<Bson>> {
        let key = self.definition.key_field();
        keys.iter().map(|k| key_to_bson(key, k)).collect()
    }
}

#[async_trait]
impl VectorCollection for MongoDbCollection {
    fn name(&self) -> &str {
        &self.name
    }

    fn definition(&self) -> &VectorStoreCollectionDefinition {
        &self.definition
    }

    async fn ensure_collection_exists(&self) -> Result<()> {
        self.ensure_collection_exists_with(true).await
    }

    async fn collection_exists(&self) -> Result<bool> {
        collection_exists(&self.database, &self.name).await
    }

    async fn ensure_collection_deleted(&self) -> Result<()> {
        if self.collection_exists().await? {
            self.collection
                .drop()
                .await
                .map_err(|e| service("drop", e))?;
        }
        Ok(())
    }

    async fn upsert(&self, records: Vec<Value>) -> Result<Vec<Value>> {
        let documents = records
            .iter()
            .map(|r| self.to_document(r))
            .collect::<Result<Vec<_>>>()?;
        let mut keys = Vec::with_capacity(documents.len());
        for document in documents {
            let id = document.get("_id").cloned().unwrap_or(Bson::Null);
            self.collection
                .replace_one(doc! { "_id": id.clone() }, &document)
                .upsert(true)
                .await
                .map_err(|e| service("upsert", e))?;
            keys.push(bson_to_json(id));
        }
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        if keys.is_empty() {
            return Ok(Vec::new());
        }
        let prepared = self.keys_to_bson(&keys)?;
        let mut find = self
            .collection
            .find(doc! { "_id": { "$in": prepared.clone() } });
        if let Some(projection) = self.projection(include_vectors) {
            find = find.projection(projection);
        }
        let documents: Vec<Document> = find
            .await
            .map_err(|e| service("find", e))?
            .try_collect()
            .await
            .map_err(|e| service("find", e))?;
        let mut by_key = HashMap::new();
        for document in documents {
            let id = document
                .get("_id")
                .map(codec::key_string)
                .ok_or_else(|| invalid_response("MongoDB response is missing the '_id' field."))?;
            by_key.insert(id, self.from_document(document, include_vectors)?);
        }
        Ok(prepared
            .iter()
            .map(|k| by_key.get(&codec::key_string(k)).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let prepared = self.keys_to_bson(&keys)?;
        if prepared.is_empty() {
            return Ok(());
        }
        self.collection
            .delete_many(doc! { "_id": { "$in": prepared } })
            .await
            .map_err(|e| service("delete", e))?;
        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_with(vector, options, &MongoDbSearchOptions::default())
            .await
    }
}

async fn collection_exists(database: &Database, name: &str) -> Result<bool> {
    let names = database
        .list_collection_names()
        .filter(doc! { "name": name })
        .await
        .map_err(|e| service("listCollections", e))?;
    Ok(!names.is_empty())
}

/// A MongoDB database holding vector collections. See the [crate docs](crate).
#[derive(Clone)]
pub struct MongoDbStore {
    client: Client,
    database: Database,
}

impl std::fmt::Debug for MongoDbStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MongoDbStore")
            .field("database", &self.database.name())
            .finish_non_exhaustive()
    }
}

impl MongoDbStore {
    /// Connect to `uri` and use `database_name`. No I/O until the first
    /// operation (beyond resolving an `mongodb+srv` URI).
    pub async fn connect(uri: &str, database_name: &str) -> Result<Self> {
        Self::connect_with_app_name(uri, database_name, None).await
    }

    /// [`Self::connect`] with a driver application name.
    pub async fn connect_with_app_name(
        uri: &str,
        database_name: &str,
        app_name: Option<&str>,
    ) -> Result<Self> {
        non_empty(uri, "MongoDB uri")?;
        non_empty(database_name, "database_name")?;
        let mut options = mongodb::options::ClientOptions::parse(uri)
            .await
            .map_err(|e| service("connection string parsing", e))?;
        if let Some(app_name) = app_name {
            non_empty(app_name, "app_name")?;
            options.app_name = Some(app_name.to_string());
        }
        let client = Client::with_options(options).map_err(|e| service("client creation", e))?;
        Self::from_client(client, database_name)
    }

    /// From `MONGODB_URI`, `MONGODB_DATABASE_NAME` and (optional)
    /// `MONGODB_APP_NAME`, as upstream.
    pub async fn from_env() -> Result<Self> {
        let uri = std::env::var("MONGODB_URI")
            .map_err(|_| Error::Configuration("MongoDB uri is required (MONGODB_URI).".into()))?;
        let database = std::env::var("MONGODB_DATABASE_NAME").map_err(|_| {
            Error::Configuration("database_name is required (MONGODB_DATABASE_NAME).".into())
        })?;
        let app_name = std::env::var("MONGODB_APP_NAME")
            .ok()
            .filter(|a| !a.is_empty());
        Self::connect_with_app_name(&uri, &database, app_name.as_deref()).await
    }

    /// Use an existing client (it stays the caller's).
    pub fn from_client(client: Client, database_name: &str) -> Result<Self> {
        non_empty(database_name, "database_name")?;
        let database = client.database(database_name);
        Ok(Self { client, database })
    }

    /// The driver client.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// The database's name.
    pub fn database_name(&self) -> &str {
        self.database.name()
    }

    /// A typed collection handle. No I/O.
    pub fn collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<MongoDbCollection> {
        MongoDbCollection::new(self.database.clone(), name, definition)
    }

    /// Drop a collection by name (absent is not an error).
    pub async fn ensure_collection_deleted(&self, name: &str) -> Result<()> {
        self.database
            .collection::<Document>(name)
            .drop()
            .await
            .map_err(|e| service("drop", e))
    }
}

#[async_trait]
impl VectorStore for MongoDbStore {
    fn get_collection(
        &self,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Box<dyn VectorCollection>> {
        Ok(Box::new(self.collection(name, definition)?))
    }

    async fn list_collection_names(&self) -> Result<Vec<String>> {
        self.database
            .list_collection_names()
            .await
            .map_err(|e| service("listCollections", e))
    }

    async fn collection_exists(&self, name: &str) -> Result<bool> {
        collection_exists(&self.database, name).await
    }
}
