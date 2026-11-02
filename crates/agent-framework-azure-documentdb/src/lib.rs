//! An Azure DocumentDB (MongoDB vCore) [`VectorStore`]: BSON documents
//! keyed by `_id`, `cosmosSearch` vector indexes (IVF, HNSW or DiskANN),
//! and `$search` with `cosmosSearch` for retrieval.
//!
//! Port of upstream `agent_framework_azure_documentdb`
//! (`AzureDocumentDBCollection`, `AzureDocumentDBStore`) over the official
//! `mongodb` driver, sharing the BSON codec of `agent-framework-mongodb`.
//!
//! ```no_run
//! use agent_framework_azure_documentdb::{AzureDocumentDbStore, IndexParameters};
//! use agent_framework_core::vectors::{
//!     Filter, IndexKind, VectorCollection, VectorSearchOptions, VectorStoreCollectionDefinition,
//!     VectorStoreField,
//! };
//!
//! # async fn demo() -> agent_framework_core::error::Result<()> {
//! // AZURE_DOCUMENTDB_CONNECTION_STRING and AZURE_DOCUMENTDB_DATABASE_NAME.
//! let store = AzureDocumentDbStore::from_env().await?;
//! let definition = VectorStoreCollectionDefinition::new(vec![
//!     VectorStoreField::key("id").with_type("str"),
//!     // Only indexed fields can be filtered.
//!     VectorStoreField::data("category").with_type("str").indexed(),
//!     VectorStoreField::vector("embedding", 3).with_index_kind(IndexKind::new("hnsw")),
//! ])?;
//! let notes = store
//!     .collection("notes", definition)?
//!     .with_index_parameters("embedding", IndexParameters { m: Some(32), ..Default::default() })?;
//! notes.ensure_collection_exists().await?;
//! let hits = notes
//!     .search(
//!         vec![0.1, 0.2, 0.3],
//!         &VectorSearchOptions::new(5).with_filter(Filter::eq("category", "db")?),
//!     )
//!     .await?;
//! # let _ = hits;
//! # Ok(())
//! # }
//! ```
//!
//! # Model
//!
//! - The key is stored as `_id`: a `str` (at most 2 KiB) or an `int`.
//! - Data types: `str`, `int`, `float`, `bool`, `list` and `dict`.
//! - Each vector field gets a `cosmosSearch` index created with the
//!   `createIndexes` command: `default` / `ivf_flat` → `vector-ivf`,
//!   `hnsw` → `vector-hnsw`, `disk_ann` → `vector-diskann`; similarity
//!   `COS` (`DEFAULT`, `cosine_similarity`), `IP` (`dot_prod`) or `L2`
//!   (`euclidean_distance`). Algorithm parameters come from
//!   [`IndexParameters`]. Indexed data fields get ascending indexes. An
//!   existing index that differs is an error, not a migration.
//! - Search is `$search` / `cosmosSearch` with `k = skip + top` unless
//!   [`DocumentDbSearchOptions::k`] widens it. Scores are DocumentDB's
//!   native `searchScore`: higher is closer for `COS` and `IP`
//!   ([`VectorSearchResult::SCORE_KIND_RELEVANCE`]), lower for `L2`
//!   ([`VectorSearchResult::SCORE_KIND_DISTANCE`]).
//! - Portable filters compile to plain MQL on the key and indexed fields:
//!   equality, membership, ordered comparisons, `exists` / null checks and
//!   list containment, combined with `and` / `or` / `not` (`$nor`). Text
//!   operators are refused: a regular expression is not a literal match.
//!   [`VectorSearchOptions::provider_filter`] takes an MQL filter as a
//!   JSON object, conjoined with the portable one.
//!
//! # Divergences
//!
//! - The `bytes` type is refused: records are JSON, which has no bytes.
//! - Upserts are one `replaceOne` (with `upsert`) per record, in order,
//!   rather than `bulkWrite` batches; a failure reports how many records
//!   were written before it.
//! - There is no filtered (non-vector) retrieval.
//! - Upstream's `provider_annotations` (`azure_documentdb.m`, …) are
//!   [`IndexParameters`] set per field on the collection.

#![cfg(feature = "experimental-vector-stores")]
#![cfg_attr(docsrs, doc(cfg(feature = "experimental-vector-stores")))]

use std::collections::HashMap;
use std::time::Duration;

use agent_framework_core::error::{Error, Result};
use agent_framework_core::vectors::{
    FieldType, Filter, FilterExpression, FilterGroupOperator, FilterOperator, VectorCollection,
    VectorSearchOptions, VectorSearchResult, VectorStore, VectorStoreCollectionDefinition,
    VectorStoreField,
};
use agent_framework_mongodb::codec::{self, bson_to_json, json_to_bson, KeyRules};
use agent_framework_mongodb::mongodb::bson::{doc, Bson, Document};
use agent_framework_mongodb::mongodb::{self, Client, Database};
use async_trait::async_trait;
use futures::TryStreamExt;
use serde_json::Value;

pub use agent_framework_mongodb::mongodb as driver;

/// The field the native `searchScore` is projected into.
pub const SCORE_FIELD: &str = "_af_documentdb_score";
const MAX_ID_BYTES: usize = 2 * 1024;
const MAX_INDEX_PATH_BYTES: usize = 256;
const KEY_BATCH_SIZE: usize = 1_000;
const INDEX_RACE_CODES: &[i32] = &[68, 85, 86];
const INDEX_RACE_NAMES: &[&str] = &[
    "IndexAlreadyExists",
    "IndexKeySpecsConflict",
    "IndexOptionsConflict",
];
const DATA_TYPES: &[&str] = &["str", "int", "float", "bool", "list", "dict"];
const APP_NAME: &str = "agent-framework-azure-documentdb";

fn service(context: &str, e: mongodb::error::Error) -> Error {
    Error::service(format!("Azure DocumentDB {context} failed: {e}"))
}

fn config(message: impl Into<String>) -> Error {
    Error::Configuration(message.into())
}

/// The index algorithm (`kind`) for a field's index kind.
fn index_kind(field: &VectorStoreField) -> Result<&'static str> {
    Ok(
        match field
            .index_kind
            .as_ref()
            .map(|k| k.as_str())
            .unwrap_or("default")
        {
            "default" | "ivf_flat" => "vector-ivf",
            "hnsw" => "vector-hnsw",
            "disk_ann" => "vector-diskann",
            other => {
                return Err(config(format!(
                    "Unsupported Azure DocumentDB index kind '{other}'."
                )))
            }
        },
    )
}

/// The similarity metric for a field's distance function.
fn metric(field: &VectorStoreField) -> Result<&'static str> {
    Ok(
        match field
            .distance_function
            .as_ref()
            .map(|d| d.as_str())
            .unwrap_or("DEFAULT")
        {
            "DEFAULT" | "cosine_similarity" => "COS",
            "dot_prod" => "IP",
            "euclidean_distance" => "L2",
            other => {
                return Err(config(format!(
                    "Unsupported Azure DocumentDB distance function '{other}'."
                )))
            }
        },
    )
}

fn validate_database_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 63 {
        return Err(config("database_name must contain 1-63 UTF-8 bytes."));
    }
    if name.chars().any(|c| "/\\.\"$*<>:|?\0".contains(c)) {
        return Err(config(
            "database_name contains a character unsupported by Azure DocumentDB.",
        ));
    }
    Ok(())
}

fn validate_collection_name(name: &str) -> Result<()> {
    if name.is_empty() || name.len() > 255 {
        return Err(config("collection_name must contain 1-255 UTF-8 bytes."));
    }
    if name.contains('\0') || name.contains('$') || name.starts_with("system.") {
        return Err(config(
            "collection_name contains a reserved Azure DocumentDB name or character.",
        ));
    }
    Ok(())
}

fn validate_storage_name(name: &str, indexed: bool) -> Result<()> {
    if name.is_empty() || name.contains('\0') || name.contains('.') || name.starts_with('$') {
        return Err(config(
            "Azure DocumentDB fields must be non-empty top-level names without '.', '$' prefixes, or NUL.",
        ));
    }
    if indexed && name.len() > MAX_INDEX_PATH_BYTES {
        return Err(config(format!(
            "Azure DocumentDB indexed field paths cannot exceed {MAX_INDEX_PATH_BYTES} UTF-8 bytes."
        )));
    }
    Ok(())
}

fn check_range(value: u32, name: &str, minimum: u32, maximum: Option<u32>) -> Result<u32> {
    if value < minimum || maximum.is_some_and(|m| value > m) {
        let requirement = match maximum {
            Some(m) => format!("between {minimum} and {m}"),
            None => format!("at least {minimum}"),
        };
        return Err(config(format!("{name} must be an integer {requirement}.")));
    }
    Ok(value)
}

/// Upstream's index name: `af_<kind>_<sha256[:32]>`.
pub fn index_name(collection_name: &str, field_storage_name: &str, kind: &str) -> String {
    let digest = codec::sha256_hex(&format!("{collection_name}\0{field_storage_name}\0{kind}"));
    format!("af_{kind}_{}", &digest[..32])
}

/// Vector index algorithm parameters (upstream's `azure_documentdb.*`
/// provider annotations). Only the ones for the field's index kind may be
/// set.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IndexParameters {
    /// IVF: `numLists` (default 1).
    pub num_lists: Option<u32>,
    /// HNSW: `m`, 2–100 (default 16).
    pub m: Option<u32>,
    /// HNSW: `efConstruction`, `2m`–1000 (default 64).
    pub ef_construction: Option<u32>,
    /// DiskANN: `maxDegree`, 20–2048 (default 32).
    pub max_degree: Option<u32>,
    /// DiskANN: `lBuild`, 10–500 (default 50).
    pub l_build: Option<u32>,
}

impl IndexParameters {
    fn validate(&self, kind: &str) -> Result<()> {
        match kind {
            "vector-ivf" => {
                if self.m.is_some()
                    || self.ef_construction.is_some()
                    || self.max_degree.is_some()
                    || self.l_build.is_some()
                {
                    return Err(config("IVF indexes accept only num_lists."));
                }
                check_range(self.num_lists.unwrap_or(1), "num_lists", 1, None)?;
            }
            "vector-hnsw" => {
                if self.num_lists.is_some() || self.max_degree.is_some() || self.l_build.is_some() {
                    return Err(config("HNSW indexes accept only m and ef_construction."));
                }
                let m = check_range(self.m.unwrap_or(16), "m", 2, Some(100))?;
                check_range(
                    self.ef_construction.unwrap_or(64),
                    "ef_construction",
                    2 * m,
                    Some(1000),
                )?;
            }
            _ => {
                if self.num_lists.is_some() || self.m.is_some() || self.ef_construction.is_some() {
                    return Err(config(
                        "DiskANN indexes accept only max_degree and l_build.",
                    ));
                }
                check_range(self.max_degree.unwrap_or(32), "max_degree", 20, Some(2048))?;
                check_range(self.l_build.unwrap_or(50), "l_build", 10, Some(500))?;
            }
        }
        Ok(())
    }
}

/// DocumentDB-specific search options (upstream's `operation_options`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DocumentDbSearchOptions {
    /// Candidate window, at least `skip + top` (the default).
    pub k: Option<u32>,
    /// IVF only: lists probed, 1 to the index's `num_lists`.
    pub n_probes: Option<u32>,
    /// HNSW only: `efSearch`, at least 1.
    pub ef_search: Option<u32>,
    /// DiskANN only: `lSearch`, 10–1000 and at least `k` (default
    /// `max(40, k)`).
    pub l_search: Option<u32>,
    /// Server time limit for the aggregation.
    pub max_time: Option<Duration>,
    /// Keep hits scoring at least this (`COS`, `IP`) or at most this
    /// (`L2`).
    pub score_threshold: Option<f64>,
}

/// A value whose type does not match the field: it can equal nothing.
enum Operand {
    Value(Bson),
    Mismatch,
}

fn filter_operand(field: &VectorStoreField, value: &Value, ordered: bool) -> Result<Operand> {
    if value.is_null() {
        if ordered {
            return Err(config(
                "Ordered Azure DocumentDB filter operands cannot be null.",
            ));
        }
        return Ok(Operand::Value(Bson::Null));
    }
    let kind = field.type_.as_deref();
    if matches!(kind, Some("float" | "int")) {
        if let Value::Number(_) = value {
            return Ok(Operand::Value(json_to_bson(
                value,
                &field.name,
                KeyRules::TopLevelSafe,
            )?));
        }
    }
    if ordered && !matches!(kind, Some("str" | "int" | "float")) {
        return Err(config(
            "Ordered Azure DocumentDB filters require a string, int, or float field.",
        ));
    }
    Ok(match (kind, value) {
        (Some("str"), Value::String(s)) => Operand::Value(Bson::String(s.clone())),
        (Some("bool"), Value::Bool(b)) => Operand::Value(Bson::Boolean(*b)),
        (Some("list"), Value::Array(_)) | (Some("dict"), Value::Object(_)) => {
            Operand::Value(json_to_bson(value, &field.name, KeyRules::TopLevelSafe)?)
        }
        _ => Operand::Mismatch,
    })
}

/// Compiles portable filters into MQL documents.
struct FilterCompiler<'a> {
    definition: &'a VectorStoreCollectionDefinition,
}

impl FilterCompiler<'_> {
    fn false_filter() -> Document {
        doc! { "_id": { "$exists": false } }
    }

    fn present(name: &str) -> Vec<Document> {
        vec![
            doc! { name: { "$exists": true } },
            doc! { name: { "$ne": Bson::Null } },
        ]
    }

    fn compile(&self, expression: &FilterExpression) -> Result<Document> {
        let filter: &Filter = match expression {
            FilterExpression::Group(group) => {
                let children = group
                    .filters
                    .iter()
                    .map(|c| self.compile(c))
                    .collect::<Result<Vec<_>>>()?;
                let op = match group.operator {
                    FilterGroupOperator::And => "$and",
                    FilterGroupOperator::Or => "$or",
                    FilterGroupOperator::Not => "$nor",
                };
                return Ok(doc! { op: children });
            }
            FilterExpression::Condition(f) => f,
        };
        if filter.field_name.contains('.') {
            return Err(config(
                "Azure DocumentDB filters support declared top-level logical fields only.",
            ));
        }
        let field = self
            .definition
            .try_get_field(&filter.field_name)
            .ok_or_else(|| {
                config(format!(
                    "Unknown Azure DocumentDB field '{}'.",
                    filter.field_name
                ))
            })?;
        let name = match field.field_type {
            FieldType::Vector => {
                return Err(config(
                    "Azure DocumentDB portable filters do not operate on vector fields.",
                ))
            }
            FieldType::Key => "_id",
            FieldType::Data => {
                if field.type_.as_deref() == Some("dict") {
                    return Err(config(
                        "Azure DocumentDB filters do not support dictionary fields because BSON document equality is order-sensitive.",
                    ));
                }
                if field.is_indexed != Some(true) {
                    return Err(config(format!(
                        "Azure DocumentDB filter field '{}' must be indexed.",
                        field.name
                    )));
                }
                field.effective_storage_name()
            }
        };
        let value = filter.value.clone().unwrap_or(Value::Null);
        // MongoDB's scalar null comparisons also inspect array elements, so
        // `[null, "x"]` would count as null. A list field is null exactly when
        // it is present and not an array, and non-null when it is an array.
        let is_list = field.type_.as_deref() == Some("list");
        let list_is_null = || doc! { "$and": [ { name: { "$exists": true } }, { name: { "$not": { "$type": "array" } } } ] };
        let list_is_not_null = || doc! { name: { "$type": "array" } };
        Ok(match &filter.operator {
            FilterOperator::Exists => doc! { name: { "$exists": true } },
            FilterOperator::IsNull if is_list => list_is_null(),
            FilterOperator::IsNotNull if is_list => list_is_not_null(),
            FilterOperator::Eq if is_list && value.is_null() => list_is_null(),
            FilterOperator::Ne if is_list && value.is_null() => list_is_not_null(),
            FilterOperator::IsNull => {
                doc! { "$and": [ { name: { "$exists": true } }, { name: { "$eq": Bson::Null } } ] }
            }
            FilterOperator::IsNotNull => doc! { "$and": Self::present(name) },
            op @ (FilterOperator::Eq | FilterOperator::Ne) => {
                // MongoDB's `$eq` / `$ne` on an array field also compare each
                // element, so `[1, 2]` would match `[[1, 2], 3]`: refuse
                // rather than silently diverge from whole-value equality.
                if field.type_.as_deref() == Some("list") && !value.is_null() {
                    return Err(config(
                        "Azure DocumentDB cannot compare list fields as whole values; use contains, contains_any, or contains_all.",
                    ));
                }
                let native = if *op == FilterOperator::Eq { "$eq" } else { "$ne" };
                match filter_operand(field, &value, false)? {
                    Operand::Mismatch if *op == FilterOperator::Eq => Self::false_filter(),
                    Operand::Mismatch => doc! { name: { "$exists": true } },
                    Operand::Value(v) => {
                        let is_null = v == Bson::Null;
                        let condition = doc! { name: { native: v } };
                        if *op == FilterOperator::Ne || is_null {
                            doc! { "$and": [ { name: { "$exists": true } }, condition ] }
                        } else {
                            condition
                        }
                    }
                }
            }
            op @ (FilterOperator::Gt | FilterOperator::Gte | FilterOperator::Lt | FilterOperator::Lte) => {
                let native = match op {
                    FilterOperator::Gt => "$gt",
                    FilterOperator::Gte => "$gte",
                    FilterOperator::Lt => "$lt",
                    _ => "$lte",
                };
                match filter_operand(field, &value, true)? {
                    Operand::Value(v) => doc! { name: { native: v } },
                    Operand::Mismatch => {
                        return Err(config(format!(
                            "Ordered filter operand does not match field '{}'.",
                            field.name
                        )))
                    }
                }
            }
            FilterOperator::Between => {
                let bounds = value.as_array().cloned().unwrap_or_default();
                let lower = bounds.first().cloned().unwrap_or(Value::Null);
                let upper = bounds.get(1).cloned().unwrap_or(Value::Null);
                match (
                    filter_operand(field, &lower, true)?,
                    filter_operand(field, &upper, true)?,
                ) {
                    (Operand::Value(l), Operand::Value(u)) => {
                        doc! { name: { "$gte": l, "$lte": u } }
                    }
                    _ => {
                        return Err(config(format!(
                            "Between operands do not match field '{}'.",
                            field.name
                        )))
                    }
                }
            }
            op @ (FilterOperator::In | FilterOperator::NotIn) => {
                if field.type_.as_deref() == Some("list") {
                    return Err(config(
                        "Use contains, contains_any, or contains_all for Azure DocumentDB list fields.",
                    ));
                }
                // Membership, unlike eq/ne, never matches a null field.
                let mut operands = Vec::new();
                for item in value.as_array().cloned().unwrap_or_default() {
                    if item.is_null() {
                        continue;
                    }
                    if let Operand::Value(v) = filter_operand(field, &item, false)? {
                        operands.push(v);
                    }
                }
                if operands.is_empty() {
                    if *op == FilterOperator::In {
                        Self::false_filter()
                    } else {
                        doc! { "$and": Self::present(name) }
                    }
                } else if *op == FilterOperator::In {
                    doc! { name: { "$in": operands } }
                } else {
                    let mut all = Self::present(name);
                    all.push(doc! { name: { "$nin": operands } });
                    doc! { "$and": all }
                }
            }
            op @ (FilterOperator::Contains | FilterOperator::ContainsAny | FilterOperator::ContainsAll) => {
                if field.type_.as_deref() != Some("list") {
                    return Err(config(
                        "Azure DocumentDB collection membership requires a declared list field.",
                    ));
                }
                let operands = if *op == FilterOperator::Contains {
                    vec![value]
                } else {
                    value.as_array().cloned().unwrap_or_default()
                };
                if operands.iter().any(|o| o.is_array() || o.is_object()) {
                    return Err(config(
                        "Nested collection membership has ambiguous MongoDB array semantics.",
                    ));
                }
                let prepared = operands
                    .iter()
                    .map(|o| json_to_bson(o, &format!("{}[]", field.name), KeyRules::TopLevelSafe))
                    .collect::<Result<Vec<_>>>()?;
                // Guard on the field being an array, not on `$ne: null`:
                // MongoDB compares `$ne` against each element, so `[null, "x"]`
                // would fail a non-null guard although it contains "x".
                let is_array = doc! { name: { "$type": "array" } };
                if prepared.is_empty() {
                    return Ok(if *op == FilterOperator::ContainsAll {
                        is_array
                    } else {
                        Self::false_filter()
                    });
                }
                let native = if *op == FilterOperator::ContainsAll { "$all" } else { "$in" };
                doc! { "$and": [ is_array, { name: { native: prepared } } ] }
            }
            other => {
                return Err(config(format!(
                    "Azure DocumentDB does not support portable filter operator '{other:?}' in this connector. \
                     Literal text operations are not equivalent to regular-expression or analyzed search."
                )))
            }
        })
    }
}

/// The documented defaults made explicit, so an existing index that omits
/// them still matches.
fn canonical_vector_options(value: &Document) -> Result<Document> {
    let kind = value.get_str("kind").ok().map(str::to_string);
    let algorithm: &[&str] = match kind.as_deref() {
        Some("vector-ivf") => &["numLists"],
        Some("vector-hnsw") => &["m", "efConstruction"],
        Some("vector-diskann") => &["maxDegree", "lBuild"],
        _ => &[],
    };
    let mut unknown: Vec<&str> = value
        .keys()
        .map(String::as_str)
        .filter(|k| !["kind", "dimensions", "similarity"].contains(k) && !algorithm.contains(k))
        .collect();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(config(format!(
            "Existing Azure DocumentDB vector index has unsupported option(s): {}.",
            unknown.join(", ")
        )));
    }
    let number = |key: &str, default: Option<i64>| -> Bson {
        match value.get(key) {
            Some(Bson::Int32(i)) => Bson::Int64(i64::from(*i)),
            Some(Bson::Int64(i)) => Bson::Int64(*i),
            Some(Bson::Double(f)) if f.fract() == 0.0 => Bson::Int64(*f as i64),
            Some(other) => other.clone(),
            None => default.map(Bson::Int64).unwrap_or(Bson::Null),
        }
    };
    let similarity = value
        .get_str("similarity")
        .unwrap_or_default()
        .to_uppercase();
    let mut out = doc! {
        "kind": kind.clone().map(Bson::String).unwrap_or(Bson::Null),
        "dimensions": number("dimensions", None),
        "similarity": similarity,
    };
    match kind.as_deref() {
        Some("vector-ivf") => {
            out.insert("numLists", number("numLists", None));
        }
        Some("vector-hnsw") => {
            out.insert("m", number("m", Some(16)));
            out.insert("efConstruction", number("efConstruction", Some(64)));
        }
        Some("vector-diskann") => {
            out.insert("maxDegree", number("maxDegree", Some(32)));
            out.insert("lBuild", number("lBuild", Some(50)));
        }
        _ => {}
    }
    Ok(out)
}

fn same_key(a: &Document, b: &Document) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|((ka, va), (kb, vb))| {
            ka == kb
                && match (va, vb) {
                    (Bson::String(x), Bson::String(y)) => x == y,
                    (x, y) => codec::key_string(x) == codec::key_string(y),
                }
        })
}

/// Whether `expected` is among `indexes`: `Ok(true)` when a compatible
/// index exists, `Ok(false)` when none has its name or key, an error when
/// one with its name or key is incompatible.
pub fn matching_index(indexes: &[Document], expected: &Document) -> Result<bool> {
    let expected_name = expected.get_str("name").unwrap_or_default();
    let expected_key = expected.get_document("key").cloned().unwrap_or_default();
    let path = expected_key.keys().next().cloned().unwrap_or_default();
    let vector = expected.get_document("cosmosSearchOptions").ok();
    for index in indexes {
        let same_name = index.get_str("name") == Ok(expected_name);
        let key_matches = index
            .get_document("key")
            .is_ok_and(|k| same_key(k, &expected_key));
        if !same_name && !key_matches {
            continue;
        }
        if !key_matches {
            return Err(config(format!(
                "Existing Azure DocumentDB index '{expected_name}' has an incompatible key."
            )));
        }
        if let Some(expected_options) = vector {
            let raw = index
                .get_document("cosmosSearch")
                .or_else(|_| index.get_document("cosmosSearchOptions"))
                .map_err(|_| {
                    config(
                        "Existing Azure DocumentDB vector index is missing cosmosSearch options.",
                    )
                })?;
            if canonical_vector_options(raw)? != canonical_vector_options(expected_options)? {
                return Err(config(format!(
                    "Existing Azure DocumentDB vector index on '{path}' is incompatible with the collection definition."
                )));
            }
        } else {
            let flag = |k: &str| matches!(index.get(k), Some(b) if *b != Bson::Boolean(false));
            if flag("unique")
                || flag("sparse")
                || index.contains_key("partialFilterExpression")
                || index.contains_key("expireAfterSeconds")
                || index.contains_key("collation")
            {
                return Err(config(format!(
                    "Existing Azure DocumentDB filter index on '{path}' has incompatible options."
                )));
            }
        }
        return Ok(true);
    }
    Ok(false)
}

/// An Azure DocumentDB collection holding vector records. See the
/// [crate docs](crate).
#[derive(Clone)]
pub struct AzureDocumentDbCollection {
    database: Database,
    collection: mongodb::Collection<Document>,
    name: String,
    definition: VectorStoreCollectionDefinition,
    parameters: HashMap<String, IndexParameters>,
}

impl std::fmt::Debug for AzureDocumentDbCollection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureDocumentDbCollection")
            .field("name", &self.name)
            .field("database", &self.database.name())
            .finish_non_exhaustive()
    }
}

impl AzureDocumentDbCollection {
    fn new(
        database: Database,
        name: &str,
        definition: VectorStoreCollectionDefinition,
    ) -> Result<Self> {
        validate_collection_name(name)?;
        let key = definition.key_field();
        if !matches!(key.type_.as_deref(), Some("str" | "int")) {
            return Err(config(
                "Azure DocumentDB keys must be declared str or int; generated ObjectId keys are unsupported.",
            ));
        }
        for field in definition.fields() {
            let stored = if field.field_type == FieldType::Key {
                "_id"
            } else {
                field.effective_storage_name()
            };
            validate_storage_name(
                stored,
                field.field_type == FieldType::Vector || field.is_indexed == Some(true),
            )?;
            if field.field_type != FieldType::Key && stored == "_id" {
                return Err(config(
                    "Only the key field can map to Azure DocumentDB '_id'.",
                ));
            }
            if field.field_type != FieldType::Key && stored == SCORE_FIELD {
                return Err(config(format!(
                    "Azure DocumentDB field name '{SCORE_FIELD}' is reserved."
                )));
            }
            match field.field_type {
                FieldType::Data => {
                    let kind = field.type_.as_deref().unwrap_or_default();
                    if !DATA_TYPES.contains(&kind) {
                        return Err(config(format!(
                            "Azure DocumentDB data field '{}' needs a supported explicit type; got '{kind}'.",
                            field.name
                        )));
                    }
                    if field.is_full_text_indexed == Some(true) {
                        return Err(config(
                            "Azure DocumentDB full-text and hybrid search are not supported.",
                        ));
                    }
                }
                FieldType::Vector => {
                    let dims = field.dimensions.unwrap_or_default();
                    if !(1..=2000).contains(&dims) {
                        return Err(config(
                            "Azure DocumentDB standard vector dimensions must be an integer from 1 to 2000.",
                        ));
                    }
                    if !matches!(
                        field.type_.as_deref(),
                        None | Some("float" | "float32" | "float64")
                    ) {
                        return Err(config(format!(
                            "Azure DocumentDB vector field '{}' must be a float vector.",
                            field.name
                        )));
                    }
                    IndexParameters::default().validate(index_kind(field)?)?;
                    metric(field)?;
                }
                FieldType::Key => {}
            }
        }
        Ok(Self {
            collection: database.collection::<Document>(name),
            database,
            name: name.to_string(),
            definition,
            parameters: HashMap::new(),
        })
    }

    /// Set a vector field's index algorithm parameters.
    pub fn with_index_parameters(
        mut self,
        vector_field: &str,
        parameters: IndexParameters,
    ) -> Result<Self> {
        let field = self
            .definition
            .try_get_field(vector_field)
            .filter(|f| f.field_type == FieldType::Vector)
            .ok_or_else(|| config(format!("'{vector_field}' is not a vector field.")))?;
        parameters.validate(index_kind(field)?)?;
        self.parameters.insert(field.name.clone(), parameters);
        Ok(self)
    }

    fn parameters(&self, field: &VectorStoreField) -> IndexParameters {
        self.parameters
            .get(&field.name)
            .copied()
            .unwrap_or_default()
    }

    /// The `createIndexes` spec of a vector field's `cosmosSearch` index
    /// (test seam).
    pub fn vector_index_spec(&self, field: &VectorStoreField) -> Result<Document> {
        let kind = index_kind(field)?;
        let p = self.parameters(field);
        let mut options = doc! {
            "kind": kind,
            "dimensions": field.dimensions.unwrap_or_default() as i64,
            "similarity": metric(field)?,
        };
        match kind {
            "vector-ivf" => {
                options.insert("numLists", i64::from(p.num_lists.unwrap_or(1)));
            }
            "vector-hnsw" => {
                options.insert("m", i64::from(p.m.unwrap_or(16)));
                options.insert("efConstruction", i64::from(p.ef_construction.unwrap_or(64)));
            }
            _ => {
                options.insert("maxDegree", i64::from(p.max_degree.unwrap_or(32)));
                options.insert("lBuild", i64::from(p.l_build.unwrap_or(50)));
            }
        }
        let path = field.effective_storage_name();
        Ok(doc! {
            "name": index_name(&self.name, path, "vector"),
            "key": { path: "cosmosSearch" },
            "cosmosSearchOptions": options,
        })
    }

    /// The `createIndexes` spec of an indexed data field (test seam).
    pub fn data_index_spec(&self, field: &VectorStoreField) -> Document {
        let path = field.effective_storage_name();
        doc! { "name": index_name(&self.name, path, "filter"), "key": { path: 1 } }
    }

    /// A portable filter as MQL (test seam).
    pub fn filter_document(&self, expression: &FilterExpression) -> Result<Document> {
        let compiled = FilterCompiler {
            definition: &self.definition,
        }
        .compile(expression)?;
        codec::check_document_size(&compiled, "Azure DocumentDB filter")
            .map_err(|_| config("Azure DocumentDB filters cannot exceed 16 MiB of BSON."))?;
        Ok(compiled)
    }

    /// A record as the BSON document stored for it (test seam).
    pub fn to_document(&self, record: &Value) -> Result<Document> {
        let source = record
            .as_object()
            .ok_or_else(|| config("a vector store record must be a JSON object"))?;
        let mut document = Document::new();
        for field in self.definition.fields() {
            let value = source.get(&field.name).ok_or_else(|| {
                config(format!(
                    "Record is missing Azure DocumentDB field '{}'.",
                    field.name
                ))
            })?;
            let stored = field.effective_storage_name();
            match field.field_type {
                FieldType::Key => {
                    document.insert("_id", self.key_to_bson(value)?);
                }
                FieldType::Vector => {
                    let bson = if value.is_null() {
                        Bson::Null
                    } else {
                        codec::dense_vector(field, value)?;
                        json_to_bson(value, &field.name, KeyRules::TopLevelSafe)?
                    };
                    document.insert(stored, bson);
                }
                FieldType::Data => {
                    document.insert(stored, data_to_bson(field, value)?);
                }
            }
        }
        codec::check_document_size(&document, "Azure DocumentDB record")?;
        Ok(document)
    }

    /// A stored document back as a record (test seam).
    pub fn from_document(&self, mut document: Document, include_vectors: bool) -> Result<Value> {
        document.remove(SCORE_FIELD);
        let id = document
            .remove("_id")
            .ok_or_else(|| Error::service("Azure DocumentDB returned a document without '_id'."))?;
        let key = bson_to_json(id);
        self.key_to_bson(&key)
            .map_err(|_| Error::service("Azure DocumentDB returned a key of the wrong type."))?;
        let mut stored = Value::Object(Default::default());
        let map = stored.as_object_mut().expect("object");
        for (k, v) in document {
            map.insert(k, bson_to_json(v));
        }
        map.insert(self.definition.key_field_storage_name().to_string(), key);
        self.definition.from_storage(&stored, include_vectors)
    }

    fn key_to_bson(&self, value: &Value) -> Result<Bson> {
        match (self.definition.key_field().type_.as_deref(), value) {
            (Some("str"), Value::String(s)) => {
                if s.len() > MAX_ID_BYTES {
                    return Err(config(format!(
                        "Azure DocumentDB string keys cannot exceed {MAX_ID_BYTES} UTF-8 bytes."
                    )));
                }
                Ok(Bson::String(s.clone()))
            }
            (Some("int"), Value::Number(n)) => n.as_i64().map(Bson::Int64).ok_or_else(|| {
                config("Azure DocumentDB integer keys must fit a signed 64-bit BSON integer.")
            }),
            _ => Err(config(
                "Azure DocumentDB key must match its declared string or integer type.",
            )),
        }
    }

    fn projection(&self, include_vectors: bool, score: bool) -> Document {
        let mut projection = doc! { "_id": 1 };
        for field in self.definition.fields() {
            if field.field_type != FieldType::Key
                && (include_vectors || field.field_type != FieldType::Vector)
            {
                projection.insert(field.effective_storage_name(), 1);
            }
        }
        if score {
            projection.insert(SCORE_FIELD, doc! { "$meta": "searchScore" });
        }
        projection
    }

    /// The `$search` / `cosmosSearch` pipeline for a search (test seam).
    pub fn search_pipeline(
        &self,
        vector: &[f32],
        options: &VectorSearchOptions,
        native: &DocumentDbSearchOptions,
    ) -> Result<Vec<Document>> {
        options.validate()?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .ok_or_else(|| {
                config("Select a vector field from the Azure DocumentDB collection definition.")
            })?;
        let query = codec::query_vector(field, vector)?;
        let minimum_k = options.top.saturating_add(options.skip);
        let minimum_k = u32::try_from(minimum_k).map_err(|_| config("skip + top is too large."))?;
        let k = check_range(native.k.unwrap_or(minimum_k), "k", minimum_k, None)?;
        let mut search =
            doc! { "path": field.effective_storage_name(), "vector": query, "k": i64::from(k) };
        match index_kind(field)? {
            "vector-ivf" => {
                if native.ef_search.is_some() || native.l_search.is_some() {
                    return Err(config(
                        "IVF search supports n_probes, not ef_search or l_search.",
                    ));
                }
                if let Some(n) = native.n_probes {
                    let lists = self.parameters(field).num_lists.unwrap_or(1);
                    search.insert(
                        "nProbes",
                        i64::from(check_range(n, "n_probes", 1, Some(lists))?),
                    );
                }
            }
            "vector-hnsw" => {
                if native.n_probes.is_some() || native.l_search.is_some() {
                    return Err(config(
                        "HNSW search supports ef_search, not n_probes or l_search.",
                    ));
                }
                if let Some(ef) = native.ef_search {
                    search.insert(
                        "efSearch",
                        i64::from(check_range(ef, "ef_search", 1, None)?),
                    );
                }
            }
            _ => {
                if native.n_probes.is_some() || native.ef_search.is_some() {
                    return Err(config(
                        "DiskANN search supports l_search, not n_probes or ef_search.",
                    ));
                }
                let l = check_range(
                    native.l_search.unwrap_or(k.max(40)),
                    "l_search",
                    10,
                    Some(1000),
                )?;
                if k > l {
                    return Err(config(
                        "DiskANN search requires k to be less than or equal to l_search.",
                    ));
                }
                search.insert("lSearch", i64::from(l));
            }
        }
        let portable = options
            .filter
            .as_ref()
            .map(|f| self.filter_document(f))
            .transpose()?;
        let raw = options
            .provider_filter
            .as_deref()
            .filter(|r| !r.trim().is_empty())
            .map(codec::provider_filter)
            .transpose()?;
        let filter = match (portable, raw) {
            (Some(a), Some(b)) => Some(doc! { "$and": [a, b] }),
            (a, b) => a.or(b),
        };
        if let Some(filter) = filter.filter(|f| !f.is_empty()) {
            search.insert("filter", filter);
        }
        let mut pipeline = vec![
            doc! { "$search": { "cosmosSearch": search } },
            doc! { "$project": self.projection(options.include_vectors, true) },
        ];
        if let Some(threshold) = native.score_threshold {
            if !threshold.is_finite() {
                return Err(config(
                    "Azure DocumentDB score_threshold must be a finite number.",
                ));
            }
            let op = if metric(field)? == "L2" {
                "$lte"
            } else {
                "$gte"
            };
            pipeline.push(doc! { "$match": { SCORE_FIELD: { op: threshold } } });
        }
        if options.skip > 0 {
            pipeline.push(doc! { "$skip": options.skip as i64 });
        }
        pipeline.push(doc! { "$limit": options.top as i64 });
        Ok(pipeline)
    }

    /// Search with DocumentDB-specific options.
    pub async fn search_with(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
        native: &DocumentDbSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        let pipeline = self.search_pipeline(&vector, options, native)?;
        let field = self
            .definition
            .try_get_vector_field(options.vector_field_name.as_deref())
            .expect("checked by search_pipeline");
        let kind = if metric(field)? == "L2" {
            VectorSearchResult::SCORE_KIND_DISTANCE
        } else {
            VectorSearchResult::SCORE_KIND_RELEVANCE
        };
        let mut aggregate = self.collection.aggregate(pipeline);
        if let Some(max_time) = native.max_time {
            aggregate = aggregate.max_time(max_time);
        }
        let documents: Vec<Document> = aggregate
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
                        return Err(Error::service(
                            "Azure DocumentDB returned an invalid searchScore.",
                        ))
                    }
                };
                Ok(VectorSearchResult {
                    record: self.from_document(document, options.include_vectors)?,
                    score: Some(score),
                    score_kind: Some(kind.to_string()),
                })
            })
            .collect()
    }

    async fn list_indexes(&self) -> Result<Vec<Document>> {
        let reply = self
            .database
            .run_command(doc! { "listIndexes": &self.name })
            .await
            .map_err(|e| service("listIndexes", e))?;
        let batch = reply
            .get_document("cursor")
            .and_then(|c| c.get_array("firstBatch"))
            .map_err(|_| {
                Error::service("Azure DocumentDB returned a malformed listIndexes reply.")
            })?;
        Ok(batch
            .iter()
            .filter_map(|b| b.as_document().cloned())
            .collect())
    }

    async fn ensure_index(&self, expected: Document) -> Result<()> {
        if matching_index(&self.list_indexes().await?, &expected)? {
            return Ok(());
        }
        let name = expected.get_str("name").unwrap_or_default().to_string();
        let command = doc! { "createIndexes": &self.name, "indexes": [expected.clone()] };
        if let Err(e) = self.database.run_command(command).await {
            let race = match e.kind.as_ref() {
                mongodb::error::ErrorKind::Command(c) => {
                    INDEX_RACE_CODES.contains(&c.code)
                        || INDEX_RACE_NAMES.contains(&c.code_name.as_str())
                }
                _ => false,
            };
            if !race {
                return Err(service("createIndexes", e));
            }
        }
        if !matching_index(&self.list_indexes().await?, &expected)? {
            return Err(Error::service(format!(
                "Azure DocumentDB did not expose the requested index '{name}'."
            )));
        }
        Ok(())
    }

    /// Create the collection and, when `create_indexes`, its vector and
    /// filter indexes. Existing compatible indexes are kept; incompatible
    /// ones are an error.
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
        for field in self.definition.fields() {
            match field.field_type {
                FieldType::Vector => self.ensure_index(self.vector_index_spec(field)?).await?,
                FieldType::Data if field.is_indexed == Some(true) => {
                    self.ensure_index(self.data_index_spec(field)).await?
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn keys_to_bson(&self, keys: &[Value]) -> Result<Vec<Bson>> {
        keys.iter().map(|k| self.key_to_bson(k)).collect()
    }
}

/// A data value as BSON, checked against the declared type.
fn data_to_bson(field: &VectorStoreField, value: &Value) -> Result<Bson> {
    let mismatch = || {
        config(format!(
            "Azure DocumentDB field '{}' requires declared type '{}'.",
            field.name,
            field.type_.as_deref().unwrap_or_default()
        ))
    };
    Ok(match (field.type_.as_deref(), value) {
        (_, Value::Null) => Bson::Null,
        (Some("str"), Value::String(s)) => Bson::String(s.clone()),
        (Some("int"), Value::Number(n)) if n.is_i64() || n.is_u64() => {
            json_to_bson(value, &field.name, KeyRules::TopLevelSafe)?
        }
        (Some("float"), Value::Number(n)) => {
            Bson::Double(n.as_f64().filter(|f| f.is_finite()).ok_or_else(|| {
                config(format!("Field '{}' requires a finite number.", field.name))
            })?)
        }
        (Some("bool"), Value::Bool(b)) => Bson::Boolean(*b),
        (Some("list"), Value::Array(_)) | (Some("dict"), Value::Object(_)) => {
            json_to_bson(value, &field.name, KeyRules::TopLevelSafe)?
        }
        _ => return Err(mismatch()),
    })
}

#[async_trait]
impl VectorCollection for AzureDocumentDbCollection {
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
        for (written, document) in documents.iter().enumerate() {
            let id = document.get("_id").cloned().unwrap_or(Bson::Null);
            self.collection
                .replace_one(doc! { "_id": id.clone() }, document)
                .upsert(true)
                .await
                .map_err(|e| {
                    Error::service(format!(
                        "Azure DocumentDB upsert failed after {written} records were applied: {e}"
                    ))
                })?;
            keys.push(bson_to_json(id));
        }
        Ok(keys)
    }

    async fn get(&self, keys: Vec<Value>, include_vectors: bool) -> Result<Vec<Option<Value>>> {
        let prepared = self.keys_to_bson(&keys)?;
        if prepared.is_empty() {
            return Ok(Vec::new());
        }
        let mut by_key = HashMap::new();
        for batch in prepared.chunks(KEY_BATCH_SIZE) {
            let documents: Vec<Document> = self
                .collection
                .find(doc! { "_id": { "$in": batch.to_vec() } })
                .projection(self.projection(include_vectors, false))
                .await
                .map_err(|e| service("find", e))?
                .try_collect()
                .await
                .map_err(|e| service("find", e))?;
            for document in documents {
                let id = document
                    .get("_id")
                    .map(codec::key_string)
                    .unwrap_or_default();
                by_key.insert(id, self.from_document(document, include_vectors)?);
            }
        }
        Ok(prepared
            .iter()
            .map(|k| by_key.get(&codec::key_string(k)).cloned())
            .collect())
    }

    async fn delete(&self, keys: Vec<Value>) -> Result<()> {
        let prepared = self.keys_to_bson(&keys)?;
        let mut attempted = 0;
        for batch in prepared.chunks(KEY_BATCH_SIZE) {
            self.collection
                .delete_many(doc! { "_id": { "$in": batch.to_vec() } })
                .await
                .map_err(|e| {
                    Error::service(format!(
                        "Azure DocumentDB delete failed after {attempted} keys were attempted: {e}"
                    ))
                })?;
            attempted += batch.len();
        }
        Ok(())
    }

    async fn search(
        &self,
        vector: Vec<f32>,
        options: &VectorSearchOptions,
    ) -> Result<Vec<VectorSearchResult>> {
        self.search_with(vector, options, &DocumentDbSearchOptions::default())
            .await
    }
}

async fn collection_exists(database: &Database, name: &str) -> Result<bool> {
    validate_collection_name(name)?;
    let names = database
        .list_collection_names()
        .filter(doc! { "name": name })
        .await
        .map_err(|e| service("listCollections", e))?;
    Ok(!names.is_empty())
}

/// An Azure DocumentDB database holding vector collections. See the
/// [crate docs](crate).
#[derive(Clone)]
pub struct AzureDocumentDbStore {
    client: Client,
    database: Database,
}

impl std::fmt::Debug for AzureDocumentDbStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AzureDocumentDbStore")
            .field("database", &self.database.name())
            .finish_non_exhaustive()
    }
}

impl AzureDocumentDbStore {
    /// Connect with a MongoDB connection string. As upstream, TLS is on,
    /// retryable writes are off (DocumentDB does not support them), and
    /// the application name defaults to `agent-framework-azure-documentdb`.
    pub async fn connect(connection_string: &str, database_name: &str) -> Result<Self> {
        if connection_string.trim().is_empty() {
            return Err(config(
                "A non-empty Azure DocumentDB connection_string is required.",
            ));
        }
        validate_database_name(database_name)?;
        let mut options = mongodb::options::ClientOptions::parse(connection_string)
            .await
            .map_err(|e| service("connection string parsing", e))?;
        if options.tls.is_none() {
            options.tls = Some(mongodb::options::Tls::Enabled(Default::default()));
        }
        if matches!(options.tls, Some(mongodb::options::Tls::Disabled)) {
            return Err(config("Azure DocumentDB requires TLS."));
        }
        if options.retry_writes == Some(true) {
            return Err(config("Azure DocumentDB requires retryWrites=false."));
        }
        options.retry_writes = Some(false);
        if options.app_name.is_none() {
            options.app_name = Some(APP_NAME.to_string());
        }
        let client = Client::with_options(options).map_err(|e| service("client creation", e))?;
        Self::from_client(client, database_name)
    }

    /// From `AZURE_DOCUMENTDB_CONNECTION_STRING` and
    /// `AZURE_DOCUMENTDB_DATABASE_NAME`, as upstream.
    pub async fn from_env() -> Result<Self> {
        let connection_string = std::env::var("AZURE_DOCUMENTDB_CONNECTION_STRING").map_err(|_| {
            config("A non-empty Azure DocumentDB connection_string is required (AZURE_DOCUMENTDB_CONNECTION_STRING).")
        })?;
        let database = std::env::var("AZURE_DOCUMENTDB_DATABASE_NAME")
            .map_err(|_| config("database_name is required (AZURE_DOCUMENTDB_DATABASE_NAME)."))?;
        Self::connect(&connection_string, &database).await
    }

    /// Use an existing client (it stays the caller's).
    pub fn from_client(client: Client, database_name: &str) -> Result<Self> {
        validate_database_name(database_name)?;
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
    ) -> Result<AzureDocumentDbCollection> {
        AzureDocumentDbCollection::new(self.database.clone(), name, definition)
    }

    /// Drop a collection by name (absent is not an error).
    pub async fn ensure_collection_deleted(&self, name: &str) -> Result<()> {
        validate_collection_name(name)?;
        self.database
            .collection::<Document>(name)
            .drop()
            .await
            .map_err(|e| service("drop", e))
    }
}

#[async_trait]
impl VectorStore for AzureDocumentDbStore {
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
